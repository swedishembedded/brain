# YOLOv5 object detector

An anchor-based, single-stage real-time object detector (Ultralytics YOLOv5,
v6 layout: CSP backbone, SPP-fast block, PAN neck, three-scale anchor head).
brain loads a converted checkpoint and runs it in fp32 on the GPU/CPU
backends, or in int8 on a Rockchip RV1103/RV1106 NPU through brain's own NPU
compiler, and decodes the boxes on the host. Reach for it when you need
bounding-box detection from a YOLOv5 model, on a workstation or on a small
camera board.

## Support

| Capability | Supported |
|---|---|
| Inference (SDK `DetectionPipeline`, library) | [x] |
| Rockchip RV1103/RV1106 NPU (int8, ReLU checkpoints) | [x] |
| Training / fine-tuning | [ ] |
| CLI (`brain <arch> <action>`) | [ ] |
| D-Bus / HTTP | [ ] |
| Batched serving | [ ] |

Only a ReLU checkpoint is validated against a reference today (see the
activation note below). The input may be any size that is a multiple of 32
(320x320 and 640x640 are verified).

## Getting the weights

> **License.** The weights of a stock YOLOv5 model are derived from
> Ultralytics' release, which is licensed **AGPL-3.0** (or under an Ultralytics
> Enterprise license). Using them carries the AGPL's copyleft obligations for
> the software that uses them. brain never downloads them for you: there is no
> auto-fetch and no default model id. See
> [Third-party model licenses](../../compliance/third-party-models.md).

brain reads its own checkpoint format: a safetensors file whose
`brain.config` header carries the network's configuration and whose tensors are
each convolution's `<name>.weight` and `<name>.bias` (batch norm already folded
in). Convert a YOLOv5 ONNX export with the tool in the repository (needs
`numpy`, `onnx` and `safetensors`):

```bash
python tools/yolov5_export/export_yolov5.py --onnx yolov5s_relu.onnx
```

By default the result is written to `local/yolov5s-relu/model.safetensors` in
your model store (`$BRAIN_MODELS_DIR`, else `~/.local/share/brain/models`), where
`local/yolov5s-relu` then resolves as a model id. Pass `--out` to put it
elsewhere. The converter accepts the ReLU export Rockchip publishes for its
NPUs (input `images`, three post-sigmoid head outputs); `--width-multiple` and
`--depth-multiple` default to YOLOv5s.

## Running it from the SDK

The detection pipeline resolves the architecture from the checkpoint, so the
same call serves YOLOv8 and YOLOv5:

```rust
let pipe = brain::DetectionPipeline::from_pretrained("local/yolov5s-relu")?;
let photo = brain::Image::open("street.jpg")?;
for d in pipe.detect(&photo)? {
    println!("class {} at ({:.0},{:.0})-({:.0},{:.0}), conf {:.2}", d.class, d.x1, d.y1, d.x2, d.y2, d.confidence);
}
# Ok::<(), brain::Error>(())
```

`DetectOptions` sets the confidence threshold (default 0.25) and the NMS IoU
threshold (default 0.45). `class` indexes the checkpoint's label set; the
released weights use the 80 COCO classes.

## Running it on a Rockchip NPU

On a board with a Rockchip RV1103/RV1106 NPU (`/dev/rknpu`), the same
`DetectionPipeline` compiles the checkpoint onto the NPU when the device
selection includes it (the default does). Nothing else changes for the caller.
The checkpoint needs its NPU calibration beside it: `model.npu-calibration.json`,
the range of every tensor over representative images at the input size the
board will run. Measure it once on a host:

```sh
make crates/yolov5/examples/npu-calibrate/run ARGS="<model.safetensors> <images-dir> 320"
```

A model without its calibration is refused with a message naming that
command. The NPU path needs a ReLU checkpoint (SiLU has no NPU lowering yet).
`samples/npu/rockchip` is a complete board application: still-image detection
and a live camera stream with the boxes drawn on it.

## Running it without a device

The `yolov5` crate separates what a board needs from the reference forward.
Loading a checkpoint, the layer graph, the anchor decode and NMS need no GPU
device:

```rust
let ckpt = yolov5::weights::open(std::path::Path::new("model.safetensors"))?;
// ckpt.graph.nodes: every convolution, add, concat, pool and upsample in order;
// ckpt.weights.conv(conv): its fp32 weight and bias.
// After running the network anywhere, decode the three head outputs:
// yolov5::decode::detections(&ckpt.config, heads, &letterbox, (w, h), 0.25, 0.45, 300)
# Ok::<(), yolov5::Error>(())
```

The head outputs the decode expects are the network's three post-sigmoid maps
`[3 * (5 + nc), H/stride, W/stride]` for strides 8, 16 and 32. The decode is
the standard YOLOv5 one: `xy = (2*s - 0.5 + grid) * stride`,
`wh = (2*s)^2 * anchor`, score = objectness x class, then class-aware NMS.

## Activation

The activation after each convolution is part of the checkpoint's config
(`relu` or `silu`). Stock YOLOv5 is trained with SiLU; NPU exports are retrained
with ReLU. Both share every tensor name and shape, but only a `relu` checkpoint
has been compared against a reference.
