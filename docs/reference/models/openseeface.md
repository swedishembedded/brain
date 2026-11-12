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
