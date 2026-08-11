# splat — roadmap

Gaussian-splatting renderer/trainer (device sort/scan primitives, tiled
rasterizer, interactive viewer, and a differentiable backward pass), used for
brain's world-model image reconstruction pipeline.

## Not yet done

- [ ] Optimization pass on the tiled render pipeline — per-stage cost
      profiling, radix-sort chunk-size tuning, per-tile sort experiments, and
      pipelined present (rendering the next frame while presenting the
      current one).
- [ ] Spherical-harmonics color kernel for degrees 1–3 (only degree 0 is
      implemented).
- [ ] `.splat` / `.spz` file format I/O (only the Inria PLY format is
      supported).
- [ ] Densify/prune during training, needed to train a full scene from
      scratch (as opposed to fitting/refining an existing one).
- [ ] Accumulate-mode attention backward for chunked (multi-view) fits.
- [ ] Backward pass does not model the antialiasing compensation term — it
      assumes `antialiased=false`.
- [ ] Backward pass computes per-view gradients un-chunked; chunking across
      views is not supported.

Finite differences are deliberately not used as the backward-pass oracle:
gaussian rasterization's 1/255 output truncation biases finite-difference
gradients in a way analytic (autograd-checked) gradients do not share.
