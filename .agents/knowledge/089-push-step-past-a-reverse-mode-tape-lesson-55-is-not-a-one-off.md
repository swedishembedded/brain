<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 89. `push_step` past a reverse-mode tape (lesson #55) is not a one-off - RRDBNet had the identical bug on its LeakyReLU and its scaled residual

`crates/rrdbnet/src/model.rs`'s `lrelu()`/`residual()` helpers called
`Builder::push_step` for the net's LeakyReLU activation and its `x + 0.2 *
f(x)` residual - the exact shape lesson #55 names: a step lands on the
forward list and nothing on the `Op` tape, so `Trace::backward` silently skips
it and every conv weight upstream (which, transitively, is every weight in the
net) gets a zero gradient. RRDBNet is a small, all-conv net with none of
SDXL's transformer surface, and it still had the bug - which is the point:
"push_step in a differentiated chain" is a property of the ESCAPE HATCH, not
of transformer-shaped code, and a second, structurally unrelated model finding
it independently is what makes it a pattern rather than a one-off SDXL defect.

The fix was the same shape as #55: give `vae::blocks::Builder` two real
recorders (`leaky_relu`, backed by the already-existing `leaky_relu_bwd`
kernel; `residual_scale`, reusing the MoE gated-combine kernel `scale_add` as
a scalar multiply, backed by the already-existing `scale_add_dexp`) and route
`rrdbnet::model` through them, rather than inventing a third way to patch
around `push_step`. Both adjoints already existed in the kernel table before
this fix - as with SDXL's transformer half, the missing piece was never a
kernel, it was a tape entry.

One thing #55's UNet gate could not exercise: RRDBNet's `_scale` buffer is a
single one-element host constant read at EVERY residual site (`~4 * num_block`
of them), never a `Grads` target - the gradient must route *through* it to
`fx`, never get assigned *to* it. That is exactly the shared-parameter shape
`gradcheck`'s own T5 `rel_bias` precedent warns a `directional_check` alone
can pass on while a share of the gradient is silently missing, so
`check_rrdbnet` pairs its directional check with `check_rrdbnet_elementwise`
on the residual input FARTHEST from the loss
(`body.0.rdb1.conv5.weight`) - which only exists to test at `num_block >= 2`.
**The thing to carry forward**: `grep push_step` on every model crate that
builds on `vae::blocks::Builder`, not just the ones with a transformer half -
the bug lives in the builder's escape hatch, and any caller can reach for it.
