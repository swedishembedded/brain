// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mask algebra — threshold, morphology, feather, set operations, composite.
//!
//! A mask here is a soft `[0, 1]` NCHW buffer with `C = 1` (the convention
//! `capability::blob::decode_plane` already uses: channel 0 is the mask). Every
//! operation is exact on hard 0/1 masks and is the standard probabilistic
//! extension on soft ones.
//!
//! ## Everything is a composition of existing kernels
//!
//! There is no `threshold.wgsl`, no `dilate.wgsl`, no `blur.wgsl`, and this
//! module does not add any: the operations fall out of kernels brain already
//! has, favoring reuse over a new kernel.
//!
//! | op | composition |
//! |---|---|
//! | [`threshold`] | `film_chan` (shift by `-t`) -> `bsq_quantize` (`sign`) -> `film_chan` (`0.5x + 0.5`) |
//! | [`dilate`] | `maxpool2d`, `K = 2r+1`, stride 1, pad `r` — a max over a square window IS grey dilation |
//! | [`erode`] | `-dilate(-x)`, the standard duality; the negations are `film_chan` |
//! | [`feather`] | depthwise `conv2d_gd` with a `1/K²` box weight |
//! | [`union`] | `add2` then `axpy(-1)` of the product: `a + b - a·b` |
//! | [`intersect`] | `mul` |
//! | [`difference`] | `a · (1 - b)`: `film_chan` then `mul` |
//! | [`invert`] | `film_chan` with scale `-1`, shift `1` |
//! | [`composite`] | `old + mask·(new - old)` |
//! | [`downsample`] | `avgpool2d` (torch's adaptive rule) |
//! | [`broadcast_channels`] | `conv2d_gd`, `K = 1`, `Cin = 1`, all-ones weight |
//!
//! ## Border behaviour, stated because it differs from the host code it replaces
//!
//! * [`dilate`] / [`erode`]: `maxpool2d` seeds its running max from the first
//!   **in-bounds** tap, so out-of-image taps are never selected. Dilation
//!   therefore treats outside as empty (correct), and erosion treats outside as
//!   *foreground* — a border pixel does not erode from the image edge. SciPy's
//!   `binary_erosion(border_value=0)` would erode it. Neither is wrong; they are
//!   different conventions, and this is the one you get.
//! * [`feather`]: `conv2d_gd` zero-pads, so the ramp darkens towards the border.
//!   `zimage::pipeline::feather_mask` clamps at the border (replicate) instead.
//!   **Those are different functions** — migrating that call site here changes
//!   the inpaint boundary ramp. It needs a `pad_mode` word on the conv, or the
//!   caller must accept the change deliberately.
//! * [`downsample`]: `avgpool2d` implements torch's *adaptive* rule, which
//!   reduces to a plain box pool bit-for-bit when the ratio divides exactly.
//!   That strictly generalises `zimage::pipeline::downsample_mask`, whose
//!   integer `w/lw` silently drops the remainder and is only correct for exact
//!   ratios (survey §6.9). Migrating that call site is safe and fixes a latent
//!   bug; the VAE-8x sizes it uses today divide exactly, so the numbers do not
//!   move.

use gpu_core::{f, DeviceBuffer};
use vision::Shape;

use crate::device::Ctx;

/// Hard threshold: `1` where `x > t`, `0` elsewhere (ties go to `0`, matching
/// `torch.where(x > t, 1, 0)`).
///
/// Three dispatches, exact — not a steep sigmoid. `bsq_quantize` is brain's only
/// `sign()` kernel; its `Params` are `[total, inv_sqrt_k]` with `inv_sqrt_k` a
/// bit-cast f32, and it operates **in place** on a single read_write binding, so
/// the shifted copy is what gets quantised.
pub fn threshold(ctx: &Ctx, x: &DeviceBuffer, s: Shape, t: f32) -> DeviceBuffer {
    let ones = vec![1.0f32; s.c as usize];
    let shifted = ctx.affine(x, s, &ones, &vec![-t; s.c as usize]);
    let total = s.numel();
    let step = ctx.gpu.step(
        ctx.ids.need(ctx.ids.bsq_quantize, "bsq_quantize"),
        &[&shifted],
        &[total, f(1.0)],
        total,
    );
    ctx.gpu.submit(&[], &[step]);
    // sign in {-1, +1} -> {0, 1}.
    ctx.affine(&shifted, s, &vec![0.5; s.c as usize], &vec![0.5; s.c as usize])
}

/// Grey dilation by a `(2r+1)²` square structuring element.
///
/// `maxpool2d` Params `[N, C, H, W, K, stride, pad, Ho, Wo]` — note `stride`
/// sits BEFORE `pad`, and `Ho`/`Wo` are passed in, never recomputed by the
/// kernel. At stride 1 with `pad = r` the output extent equals the input, which
/// is what makes a pooling kernel a morphology kernel. Bindings are
/// `[x, y, argmax]`; the argmax buffer is scratch here (it exists for the
/// backward) and is allocated and dropped.
pub fn dilate(ctx: &Ctx, x: &DeviceBuffer, s: Shape, radius: u32) -> DeviceBuffer {
    if radius == 0 {
        return copy(ctx, x, s);
    }
    let k = 2 * radius + 1;
    let out = ctx.buf(s.numel());
    let argmax = ctx.buf(s.numel());
    let total = s.numel();
    let step = ctx.gpu.step(
        ctx.ids.need(ctx.ids.maxpool2d, "maxpool2d"),
        &[x, &out, &argmax],
        &[s.n, s.c, s.h, s.w, k, 1, radius, s.h, s.w],
        total,
    );
    ctx.gpu.submit(&[], &[step]);
    out
}

/// Grey erosion by a `(2r+1)²` square structuring element: `-dilate(-x)`.
pub fn erode(ctx: &Ctx, x: &DeviceBuffer, s: Shape, radius: u32) -> DeviceBuffer {
    if radius == 0 {
        return copy(ctx, x, s);
    }
    let neg = negate(ctx, x, s);
    let d = dilate(ctx, &neg, s, radius);
    negate(ctx, &d, s)
}

/// Box blur by a `(2r+1)²` window — the feather that turns a hard mask edge into
/// a ramp. `radius = 0` is a copy.
///
/// `conv2d_gd` Params
/// `[N, Cin, H, W, Cout, K, stride, pad, dilation, groups, Ho, Wo]`, bindings
/// `[x, w, y]`, weight laid out `[Cout, Cin/G, K, K]`. Depthwise is the
/// `G == Cin == Cout` case, so the weight is `[C, 1, K, K]` filled with `1/K²`.
/// Zero-padded at the border — see this module's header.
pub fn feather(ctx: &Ctx, x: &DeviceBuffer, s: Shape, radius: u32) -> DeviceBuffer {
    if radius == 0 {
        return copy(ctx, x, s);
    }
    let k = 2 * radius + 1;
    let w = ctx.upload(
        "imaging.feather.box",
        &vec![1.0f32 / (k * k) as f32; (s.c * k * k) as usize],
    );
    let out = ctx.buf(s.numel());
    let total = s.numel();
    let step = ctx.gpu.step(
        ctx.ids.need(ctx.ids.conv2d_gd, "conv2d_gd"),
        &[x, &w, &out],
        &[s.n, s.c, s.h, s.w, s.c, k, 1, radius, 1, s.c, s.h, s.w],
        total,
    );
    ctx.gpu.submit(&[], &[step]);
    out
}

/// Area-average a mask (or an image) down to `lh x lw`.
///
/// `avgpool2d` Params `[N, C, H, W, Ho, Wo]`, one thread per output element.
/// The kernel uses torch's adaptive rule
/// (`h0 = floor(ho*H/Ho)`, `h1 = ceil((ho+1)*H/Ho)`), which is a plain box pool
/// when `Ho | H` and stays correct when it does not.
pub fn downsample(ctx: &Ctx, x: &DeviceBuffer, s: Shape, lh: u32, lw: u32) -> (DeviceBuffer, Shape) {
    assert!(lh > 0 && lw > 0 && lh <= s.h && lw <= s.w, "downsample: target must be non-empty and smaller");
    let out_shape = Shape::new(s.n, s.c, lh, lw);
    let out = ctx.buf(out_shape.numel());
    let total = out_shape.numel();
    let step = ctx.gpu.step(
        ctx.ids.need(ctx.ids.avgpool2d, "avgpool2d"),
        &[x, &out],
        &[s.n, s.c, s.h, s.w, lh, lw],
        total,
    );
    ctx.gpu.submit(&[], &[step]);
    (out, out_shape)
}

/// `1 - x`.
pub fn invert(ctx: &Ctx, x: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    ctx.affine(x, s, &vec![-1.0; s.c as usize], &vec![1.0; s.c as usize])
}

/// `a ∩ b` — `mul`, `Params [n]`.
pub fn intersect(ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    let out = ctx.buf(s.numel());
    let n = s.numel();
    let step = ctx.gpu.step(ctx.ids.need(ctx.ids.mul, "mul"), &[a, b, &out], &[n], n);
    ctx.gpu.submit(&[], &[step]);
    out
}

/// `a ∪ b = a + b - a·b`. Exact for hard masks; the probabilistic (noisy-OR)
/// union for soft ones, which is the extension that keeps the result in `[0,1]`
/// — a plain `max` would too, but `max` is not distributive with `intersect`
/// and brain has no elementwise `max` kernel to dispatch anyway.
pub fn union(ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    let n = s.numel();
    let prod = intersect(ctx, a, b, s);
    let out = ctx.buf(n);
    let sum = ctx.gpu.step(ctx.ids.need(ctx.ids.add2, "add2"), &[a, b, &out], &[n], n);
    ctx.gpu.submit(&[], &[sum]);
    // axpy Params `[n, s]` with `s` a bit-cast f32; `out += s * inp`.
    let sub = ctx.gpu.step(
        ctx.ids.need(ctx.ids.axpy, "axpy"),
        &[&out, &prod],
        &[n, f(-1.0)],
        n,
    );
    ctx.gpu.submit(&[], &[sub]);
    out
}

/// `a \ b = a · (1 - b)`.
pub fn difference(ctx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    let nb = invert(ctx, b, s);
    intersect(ctx, a, &nb, s)
}

/// `mask · new + (1 - mask) · old`, written as `old + mask · (new - old)` so it
/// is exact at `mask = 0` and `mask = 1` regardless of rounding.
///
/// All three buffers must have the SAME shape: a 1-channel mask must be
/// broadcast first with [`broadcast_channels`]. Requiring it is deliberate —
/// silently broadcasting is how a 3-channel mask gets multiplied against the
/// wrong channel.
pub fn composite(
    ctx: &Ctx,
    new: &DeviceBuffer,
    old: &DeviceBuffer,
    mask: &DeviceBuffer,
    s: Shape,
) -> DeviceBuffer {
    let n = s.numel();
    let neg_old = negate(ctx, old, s);
    let delta = ctx.buf(n);
    let d = ctx.gpu.step(ctx.ids.need(ctx.ids.add2, "add2"), &[new, &neg_old, &delta], &[n], n);
    ctx.gpu.submit(&[], &[d]);
    let weighted = intersect(ctx, mask, &delta, s);
    let out = ctx.buf(n);
    let sum = ctx.gpu.step(ctx.ids.need(ctx.ids.add2, "add2"), &[old, &weighted, &out], &[n], n);
    ctx.gpu.submit(&[], &[sum]);
    out
}

/// Replicate a 1-channel buffer into `c` identical channels.
///
/// A 1x1 `conv2d_gd` from `Cin = 1` to `Cout = c` with an all-ones weight; no
/// broadcast kernel needed. Params as documented on [`feather`].
pub fn broadcast_channels(ctx: &Ctx, x: &DeviceBuffer, s: Shape, c: u32) -> (DeviceBuffer, Shape) {
    assert_eq!(s.c, 1, "broadcast_channels: input must be single-channel");
    assert!(c >= 1, "broadcast_channels: need at least one output channel");
    let out_shape = Shape::new(s.n, c, s.h, s.w);
    let w = ctx.upload("imaging.broadcast.ones", &vec![1.0f32; c as usize]);
    let out = ctx.buf(out_shape.numel());
    let total = out_shape.numel();
    let step = ctx.gpu.step(
        ctx.ids.need(ctx.ids.conv2d_gd, "conv2d_gd"),
        &[x, &w, &out],
        &[s.n, 1, s.h, s.w, c, 1, 1, 0, 1, 1, s.h, s.w],
        total,
    );
    ctx.gpu.submit(&[], &[step]);
    (out, out_shape)
}

/// Intersection-over-union of two masks, on the **host**.
///
/// This is a reduction to a single scalar, so it is host work by the same
/// argument that keeps `eval::detection`'s metrics on the host: the readback of
/// one number dominates any kernel launch, and there is no reduction kernel with
/// this shape to dispatch. Soft masks use the same fuzzy algebra as [`union`] /
/// [`intersect`], so it agrees with them on hard masks exactly.
///
/// Two empty masks are defined as IoU `1.0` (identical), matching
/// `yolo::boxmath::iou`'s degenerate convention of "identical inputs score 1".
pub fn iou(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "iou: masks must have the same length");
    let mut inter = 0f64;
    let mut union_ = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let i = (x * y) as f64;
        inter += i;
        union_ += (x + y) as f64 - i;
    }
    if union_ <= 0.0 {
        return 1.0;
    }
    (inter / union_) as f32
}

/// `-x`, via `film_chan`.
fn negate(ctx: &Ctx, x: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    ctx.affine(x, s, &vec![-1.0; s.c as usize], &vec![0.0; s.c as usize])
}

/// A fresh buffer holding the same values — the identity affine. Used by the
/// `radius == 0` fast paths so they still return an owned buffer.
fn copy(ctx: &Ctx, x: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    ctx.affine(x, s, &vec![1.0; s.c as usize], &vec![0.0; s.c as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_of_identical_masks_is_one() {
        let m = [0.0f32, 1.0, 1.0, 0.0];
        assert!((iou(&m, &m) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_disjoint_masks_is_zero() {
        assert_eq!(iou(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn iou_half_overlap() {
        // a = {0,1}, b = {1,2}: intersection 1, union 3.
        let a = [1.0f32, 1.0, 0.0];
        let b = [0.0f32, 1.0, 1.0];
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_two_empty_masks_is_one() {
        assert_eq!(iou(&[0.0; 4], &[0.0; 4]), 1.0);
    }
}
