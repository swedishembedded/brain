# zipdepth - roadmap

Scope: ZipDepth monocular depth (`crates/zipdepth`).

Monocular depth support: train/quantize/infer ZipDepth (pure-conv, 6.1M
params) on CPU/GPU/NPU, load open pretrained weights, and run a realtime
camera/image depth demo. Depth Anything 3 was originally in scope and has
been dropped - ZipDepth is the depth model.

## Not yet done

- [x] Real training data + the SSI/gradient loss: `zipdepth::loss` (MiDaS SSI
      MSE + 4-scale gradient matching, four WGSL kernels, exact gradient
      gradchecked against an f64 oracle in `tests/it/p4_loss.rs`) and
      `zipdepth::dataset` (paired RGB + 16-bit depth directory, hostile-input
      validated, seeded shuffled epochs), wired as `train_loop`'s `BatchSource`
      and `brain zipdepth train --data DIR --grad-weight A`. No run on a real
      dataset has been measured yet.
- [ ] INT8 quantization - activation-range calibration mechanics are fixed,
      but no run against real weights/calibration images has measured how far
      the INT8 scales move or what that does to depth quality
- [ ] Camera capture path is implemented and unit-tested down to the V4L2 ABI
      level, but has not been confirmed against a real webcam
- [ ] GPU/NPU full-model execution through the camera loop is wired but
      likewise unconfirmed against real camera hardware
