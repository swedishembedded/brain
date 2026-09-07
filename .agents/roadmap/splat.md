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

## Serving contract - done

`render` (one-shot) and `fit` (streaming) are now [`capability::Provider`]
actions (`crates/splat/src/caps.rs`), registered in the residency scheduler
(`crates/cli/src/resident_splat.rs`) and the CLI catalog
(`crates/cli/src/catalog.rs`) under `brain/splat` - reachable over `brain do`,
D-Bus, and the event API with no splat-specific plumbing in any transport.
`view` (the interactive SDL fly-through) is deliberately NOT served: it has no
request/response shape. `fit`'s optimization loop
(`crates/splat/src/opt.rs::fit`) grew an `on_step(iter, mse) -> bool` hook,
polled once per completed iteration, that the CLI wires to a no-op-true
closure and the capability action wires to `Progress::step` + the
invocation's cancel token - so a served `fit` run reports live MSE and
aborts cleanly (`Err("cancelled")`) within a few iterations of being asked
to.

### Determinism finding (the step-0 sub-blocker)

`opt::fit`'s run-to-run bit-determinism was not previously established.
Probed by calling it twice on identical inputs and diffing the two returned
scenes' `f32::to_bits()` bitwise (`crates/splat/src/caps.rs`'s
`determinism_probe_fit_twice_on_identical_inputs` test). Measured result (Intel
Arc integrated GPU, Vulkan backend): **`0` differing bits** - `mse
0.000007867729` on both runs, identical to the ULP. The backward kernels
(`splat_grad_reduce`, `splat_bwd_emit`, `splat_bwd_count`, `splat_bwd_keys`,
`splat_project_bwd`) contain zero atomic operations - `splat_grad_reduce` is a
deterministic per-gaussian segmented reduction over id-sorted gradient records
- so bit-determinism was expected, and is now a measured fact rather than an
assumption. Following from this, the `fit` capability action's own
caps-vs-library test is gated at bit-identity for a fit run in isolation; the
tiny (sub-1e-6) deviations it separately measures between a `fit` action's PLY
output and the library call's raw `Splats` are the ALREADY-VERIFIED (item 3)
PLY round-trip's own quantization, not fit nondeterminism - the two are
distinguished by comparing bit-for-bit BEFORE any PLY round trip (this test)
and by tolerance AFTER one (the capability action test, whose wire format IS
PLY bytes).
