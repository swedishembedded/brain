<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Face landmarks (OpenSeeFace)

Finds a face in a webcam frame and tracks 66 landmarks on it from frame to
frame: the jaw line, brows, eyes, nose and mouth in the 68-point layout
without the two inner mouth corners. Small enough to run every camera frame
on integrated graphics or a CPU. The networks are OpenSeeFace's MobileNetV3
face detector and landmark models, rebuilt on brain's own kernels from the
released graphs.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| Library (`crates/openseeface`) | [x] |
| Intel NPU / GPU (OpenVINO) | [x] |
| CLI                    | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |

## Getting the weights

Code and weights are BSD-2-Clause. The release lives on GitHub only, so it
is not fetched by `brain pull`; `samples/vision/fishtank/fetch-data.sh`
downloads the three files at a pinned commit, checks their sha256, and puts
them in the models directory as `emilianavt/OpenSeeFace`:

| file | what |
|---|---|
| `mnv3_detection_opt.onnx` | the face detector |
| `lm_model0_opt.onnx` | landmarks, fastest |
| `lm_model3_opt.onnx` | landmarks, most accurate |

## Using it

```rust
let device = gpu_core::Gpu::new(openseeface::model::PIPELINES);
let mut t = openseeface::Tracker::load(&device, dir, openseeface::config::LANDMARKS_FAST, Default::default())?;
if let Some(found) = t.track(&rgb, width, height) {
    let outer_eye_corners = (found.landmarks.points[36], found.landmarks.points[45]);
}
```

The tracker runs the detector only when no face is tracked; otherwise each
frame's crop comes from the previous frame's landmarks. It follows one face.

## On the Intel NPU or GPU

`openseeface::export` writes the network as a static-shape ONNX graph from
brain's imported weights (batch 1, every resize size constant - the released
graphs' run-time shape arithmetic is what the NPU compiler refuses), and
`openseeface::openvino::OvNet` compiles it with OpenVINO for `npu`, `gpu` or
`cpu`. Both executors implement `openseeface::Infer`, which is what the
tracker runs through:

```rust
use openseeface::{config, openvino::OvNet, Tracker};
let det = OvNet::load(dir, config::DETECTOR, openvino_rt::NpuDevice::Npu)?;
let lm = OvNet::load(dir, config::LANDMARKS_FAST, openvino_rt::NpuDevice::Npu)?;
let mut t = Tracker::new(Box::new(det), Box::new(lm), Default::default());
```

The compiled graph is checked against the same reference outputs as brain's
own forward; on the GPU and NPU it computes in half precision, which moves a
decoded landmark by under 0.4 px.
