
const ort = require('onnxruntime-node');
const sharp = require('sharp');


class BackgroundRemover {
    /**
     * Initalize the base parameters of the target model
     * @param {ort.InferenceSession} session - onnxruntime inference session
     * @param {Array<number>} mean - Mean values for normalization [R, G, B]
     * @param {Array<number>} std - Standard deviation values [R, G, B]
     */
    constructor(session, mean, std) {
        this.session = session;
        this.mean = mean;
        this.std = std;
    }

    /**
     * Normalize an image for model input
     * @param {sharp.Sharp} image - Image buffer or path
     * @returns {ort.Tensor} - Input tensor for ONNX model
     */
    async normalize(image) {
        const [{ 
            shape: [index, channel, inputHeight, inputWidth] 
        }] = this.session.inputMetadata;

        // Resize and convert image to RGB
        const { data, info: {width, height, channels} } = await image
            .resize(inputWidth, inputHeight, { fit: 'fill' })
            .raw()
            .toBuffer({ resolveWithObject: true });

        // Convert to float array and normalize to [0, 1]
        const floatArray = new Float32Array(data.length);
        let maxVal = 0;
        for (let i = 0; i < data.length; i++) {
            floatArray[i] = data[i];
            if (floatArray[i] > maxVal) { maxVal = floatArray[i]; }
        }
        
        const divisor = Math.max(maxVal, 1e-6);
        for (let i = 0; i < floatArray.length; i++) {
            floatArray[i] /= divisor;
        }

        const normalized = new Float32Array(3 * height * width);

        for (let c = 0; c < 3; c++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    // pixelIdx is from the original HWC buffer (floatArray)
                    const pixelIdx = y * width * channels + x * channels + c;
                    const v = (floatArray[pixelIdx] - this.mean[c]) / this.std[c];
                    const index = c * (height * width) + y * width + x;

                    // write into CHW tensor
                    normalized[index] = v;
                }
            }
        }

        return new ort.Tensor('float32', normalized, [1, 3, height, width]);
    }

    /**
     * Make the output masks for the input image using the inner session.
     * @param {sharp.Sharp} image - The input image 
     * @returns {Promise<sharp.Sharp>} - Masked image
     */
    async mask(image) {

        // Get original image dimensions
        const {
            width: originalWidth, 
            height: originalHeight
        } = await image.metadata();

        const inputName = this.session.inputNames[0];

        // Normalize and prepare input
        const inputTensor = await this.normalize(image.clone());

        // Run inference
        const results = await this.session.run({ [inputName]: inputTensor });
        const outputName = this.session.outputNames[0];

        // Extract prediction data: shape should be [1, 1, 320, 320]
        const {data: predData, dims: [index, channel, height, width]} = results[outputName];
        const sliceData = new Float32Array(height * width);
        
        for (let i = 0; i < height * width; i++) {
            sliceData[i] = predData[i];
        }

        // Find min and max
        let ma = -Infinity;
        let mi = Infinity;
        for (let i = 0; i < sliceData.length; i++) {
            if (sliceData[i] > ma) { ma = sliceData[i]; } 
            if (sliceData[i] < mi) { mi = sliceData[i]; }
        }

        // Normalize to [0, 1]
        const normalizedData = new Float32Array(sliceData.length);
        const range = ma - mi;
        for (let i = 0; i < sliceData.length; i++) {
            normalizedData[i] = (sliceData[i] - mi) / range;
        }

        // Clip to [0, 1] and convert to uint8 [0, 255]
        const uint8Data = new Uint8Array(normalizedData.length);
        for (let i = 0; i < normalizedData.length; i++) {
            const clipped = Math.max(0, Math.min(1, normalizedData[i]));
            uint8Data[i] = Math.round(clipped * 255);
        }

        // Create image from array and resize to original dimensions
        const alpha = await sharp(Buffer.from(uint8Data), { raw: { width: width, height: height, channels: 1 } })
            .resize(originalWidth, originalHeight, { fit: 'fill' })
            .toColourspace('b-w')
            .toBuffer();

        return image
            .clone()
            .ensureAlpha()
            .toColourspace('rgb')
            .joinChannel(alpha, { raw: { width: originalWidth, height: originalHeight, channels: 1 } })
            .png();
    }
}

module.exports = BackgroundRemover;

