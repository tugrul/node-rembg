# Node.js ONNX Runtime Background Removal

Node.js project for removing backgrounds from images using several ONNX model like BiRefNet and u2net.

## Installation

```bash
npm install @tugrul/rembg
```

## Usage

```javascript
const ort = require('onnxruntime-node');
const sharp = require('sharp');

const BackgroundRemover = require('@tugrul/rembg');

async function main(inputPath, outputPath) {
  const session = await ort.InferenceSession.create('./models/u2net_human_seg.onnx');

  const rembg = new BackgroundRemover(session, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
  const result = await rembg.mask(sharp(inputPath));

  await result.toFile(outputPath);
}

main('input.jpg', 'output.png');
```

## API

### Constructor

#### `new BackgroundRemover(session, mean, std)`

Initialize a new `BackgroundRemover` instance with a preloaded ONNX session and normalization parameters.

##### Parameters

* `session` (`ort.InferenceSession`, required)
  An initialized `onnxruntime-node` inference session for the background-removal model.

* `mean` (`number[]`, required)
  Per-channel mean values used for input normalization. Expected format: `[R, G, B]`.

* `std` (`number[]`, required)
  Per-channel standard deviation values used for input normalization. Expected format: `[R, G, B]`.

##### Example

```js
const session = await ort.InferenceSession.create('u2net.onnx');
const mean = [0.485, 0.456, 0.406];
const std  = [0.229, 0.224, 0.225];

const remover = new BackgroundRemover(session, mean, std);
```

### Methods

#### `async normalize(image)`

Normalize an image for model input.

> **Note:** This is an internal helper in most usage scenarios. You usually call `mask()` directly.

##### Signature

```ts
normalize(image: sharp.Sharp): Promise<ort.Tensor>
```

##### Parameters

* `image` (`sharp.Sharp`, required)
  A `sharp` instance representing the input image.
  The method:

  * Resizes the image to the model’s expected spatial size `[inputHeight, inputWidth]` inferred from `session.inputMetadata`.
  * Forces raw RGB data.
  * Scales pixel values and applies channel-wise normalization using the `mean` and `std` passed to the constructor.
  * Reorders data from HWC to CHW.

##### Returns

* `Promise<ort.Tensor>`
  A 4D tensor of shape `[1, 3, height, width]` (`float32`), suitable as input to the ONNX model.

---

#### `async mask(image)`

Run the model on an input image and produce a transparent PNG where the alpha channel is derived from the model’s output mask.

##### Signature

```ts
mask(image: sharp.Sharp): Promise<sharp.Sharp>
```

##### Parameters

* `image` (`sharp.Sharp`, required)
  A `sharp` instance for the original image. The original dimensions are preserved in the final output.

##### Processing Steps

1. Read the original image dimensions via `image.metadata()`.
2. Normalize the image via `normalize()` and create the input tensor.
3. Run inference using the configured `onnxruntime-node` session.
4. Assume the first output has shape `[1, 1, H, W]` and extract it as a single-channel mask.
5. Normalize model output to `[0, 1]`, clip, and convert to `uint8` `[0, 255]`.
6. Construct a single-channel grayscale buffer and resize it back to the original image dimensions.
7. Combine the generated mask as the alpha channel with the original image.
8. Return a `sharp` pipeline configured to output a PNG with transparency.

##### Returns

* `Promise<sharp.Sharp>`
  A `sharp` instance representing the RGBA image with the predicted alpha mask applied.
  You can continue piping or directly write it to disk:

  ```js
  const output = await remover.mask(sharp(inputPath));
  await output.toFile('output.png');
  ```

## Features
- Async/await support
- Automatic image resizing
- Preserves original image dimensions
- PNG output with transparency
- Error handling

## Dependencies
- onnxruntime-node: ONNX model inference
- sharp: Image processing

## Notes & Assumptions

* The model input metadata must describe a 4D tensor `[N, C, H, W]`, where `C = 3` (RGB).
* The model output is assumed to be `[1, 1, H, W]` (single-channel mask).
* The mask is linearly normalized using the min/max values found in the raw output.
* Any errors thrown by `sharp` or `onnxruntime-node` are propagated to the caller.