# zipdepth - roadmap

Monocular depth support: train/quantize/infer ZipDepth (pure-conv, 6.1M
params) on CPU/GPU/NPU, load open pretrained weights, and run a realtime
camera/image depth demo. Depth Anything 3 was originally in scope and has
been dropped - ZipDepth is the depth model.

## Not yet done

- [ ] Real training data + the SSI/gradient loss (the training loop itself is
      done; data and loss are still placeholder-grade)
- [ ] INT8 quantization - activation-range calibration mechanics are fixed,
      but no run against real weights/calibration images has measured how far
      the INT8 scales move or what that does to depth quality
- [ ] Camera capture path is implemented and unit-tested down to the V4L2 ABI
      level, but has not been confirmed against a real webcam
- [ ] GPU/NPU full-model execution through the camera loop is wired but
      likewise unconfirmed against real camera hardware
