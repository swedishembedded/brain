<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 181. One gradient-scale convention; Adam hid the inversion

`adamw_step(.., clip, extra_scale)` is called with `extra_scale = 1/K` to
average K accumulated micro-batches. The optimisers disagreed about what
that means:

- **The device optimiser** (`adamw.wgsl` with `clip_coef`/`clip_coef_wg`)
  multiplied by `extra_scale`, but measured the clip norm on the raw
  summed gradient. With accumulation, that made the clip K times too lax.
- **The host optimisers** (`OffloadAdam`, the pipeline's `FusedAdam`,
  `DataParallel`) divided by `extra_scale`, so they multiplied the gradient
  by K where the device divided it by K. They also clipped on
  `max/norm.max(max)` without torch's `+1e-6`.
- **nemotronasr's** host AdamW ignored `clip` altogether.

The offload-vs-device parity test passed through all of this. Adam divides
the first moment by the root of the second, so a constant gradient scale
cancels except where ε is comparable to √v̂. A scale inverted by K² is
invisible at ε 1e-8. It became visible only with ε at 1e-1.

There is now one convention, torch's `clip_grad_norm_` on the averaged
gradient. Every gradient is multiplied by
`optim::grad_multiplier(raw_sum_sq, clip, extra_scale)`:

```
extra_scale · min(1, clip / (‖g‖ · extra_scale + 1e-6))
```

Both clip kernels compute the same value on the device. The test
`scale_and_clip_follow_torch_on_device_and_host` checks both paths against
an independent torch-semantics reference, at scale 0.25 with the clip
engaged and a large ε.

When testing an optimiser, set ε large enough that the gradient's
magnitude matters. At the default ε an Adam update is nearly
scale-invariant, and a scaling bug passes unnoticed.
