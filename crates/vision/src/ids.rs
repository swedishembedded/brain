// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kernel indices resolved **by name**, never by position.
//!
//! `Gpu::step(kind, ..)` takes a `usize` that indexes the `PIPELINES` array the
//! model handed to `Gpu::new`. Each model owns its own array, so a bare index
//! means nothing outside the model that declared it. Blocks that hard-code such
//! indices (as yolo's did) cannot be shared, and hand-maintained index constants
//! drift: `wm-diamond` declares `K_NLC_NCHW = 8` in one module and `= 18` in
//! another, for the same kernel, because it keeps two pipeline arrays.
//!
//! [`ConvKernelIds::resolve`] maps names → positions once, against whatever
//! `PIPELINES` the owning model declares. Models keep their arrays exactly as
//! they are (yolo's index order is frozen by its checkpoint contract), register
//! only the kernels they dispatch, and the blocks never see a literal index.

/// A kernel this model did not register. Using one panics naming the kernel,
/// rather than dispatching whatever sits at that index.
pub const NONE: usize = usize::MAX;

/// Kernel-pipeline indices for the shared conv blocks, resolved by name.
///
/// Fields are added as kernels land; a model that does not register a given
/// kernel simply resolves it to [`NONE`]. That is the normal case, not an error:
/// yolo registers no depth kernels and depth registers no detection-loss ones.
#[derive(Clone, Copy, Debug)]
pub struct ConvKernelIds {
    // ---- conv: dense (fast-pathed on CPU by NAME) ----
    pub conv2d: usize,
    pub conv2d_dx: usize,
    pub conv2d_dw: usize,
    // ---- conv: grouped + dilated. A DISTINCT name on purpose — `backend-cpu`
    // binds its AVX2/winograd path to the name `conv2d`, and that path is dense:
    // it ignores `groups` and would compute wrong results with no error.
    pub conv2d_gd: usize,
    /// Register-tiled grouped/dilated conv (8x4 group-aligned tile) — same math
    /// as `conv2d_gd`, taken when registered; CPU binds both to one fast path.
    pub conv2d_gd_reg: usize,
    pub conv2d_gd_dx: usize,
    pub conv2d_gd_dw: usize,
    pub conv2d_tiled: usize,
    pub conv_bias: usize,
    /// Register-tiled `conv_bias` (8x4 tile, `conv_act_reg`'s dispatch shape).
    /// Same math; CPU routes both to the same fast path.
    pub conv_bias_reg: usize,
    pub bias_add: usize,
    pub bias_grad: usize,
    // ---- batchnorm ----
    pub bn_stats: usize,
    pub bn_running: usize,
    pub bn_train: usize,
    pub bn_eval: usize,
    pub bn_dstats: usize,
    pub bn_dx: usize,
    pub bn_dgamma: usize,
    pub bn_dbeta: usize,
    // ---- activations. `leaky_relu` at slope 0 IS relu in both directions, so
    // ReLU models need no kernel of their own.
    pub silu: usize,
    pub silu_bwd: usize,
    pub leaky_relu: usize,
    pub leaky_relu_bwd: usize,
    pub sigmoid: usize,
    pub sigmoid_bwd: usize,
    // ---- fused conv -> affine -> act (inference only) ----
    pub conv_act: usize,
    pub conv_act_tiled: usize,
    pub conv_act_reg: usize,
    // ---- spatial ----
    pub avgpool2d: usize,
    pub avgpool2d_dx: usize,
    pub resize_bilinear: usize,
    pub resize_bilinear_dx: usize,
    pub resize_nearest: usize,
    pub resize_nearest_dx: usize,
    pub pixel_shuffle: usize,
    pub pixel_shuffle_dx: usize,
    pub maxpool5: usize,
    pub maxpool5_dx: usize,
    pub upsample2: usize,
    pub upsample2_dx: usize,
    // ---- context / attention (ZipDepth) ----
    pub softmax_k: usize,
    pub softmax_k_dx: usize,
    pub weighted_gap: usize,
    pub weighted_gap_dx: usize,
    pub weighted_gap_dm: usize,
    pub add_chan_bcast: usize,
    pub add_chan_bcast_dv: usize,
    pub broadcast_add_hw: usize,
    pub broadcast_add_hw_da: usize,
    pub convex_upsample: usize,
    pub convex_upsample_dmask: usize,
    pub convex_upsample_dd: usize,
    // ---- elementwise / plumbing ----
    pub mul: usize,
    pub scale_chan: usize,
    pub concat2: usize,
    pub concat_split: usize,
    pub chan_place: usize,
    pub add2: usize,
    /// `out += a` (single read_write binding) — the wgpu-safe accumulate.
    pub add_inplace: usize,
    /// `out += s * in`. A scaled residual (ZipDepth's `x + 0.3*delta`) with no
    /// scale kernel of its own — read-modify-write, so the caller keeps SSA by
    /// copying first, or by clearing `out` via submit's clear-list to get a plain
    /// scaled copy.
    pub axpy: usize,
}

impl ConvKernelIds {
    /// Resolve every field by name against the model's own `PIPELINES`.
    /// Absent kernels become [`NONE`].
    pub fn resolve(pipelines: &[(&str, &str)]) -> ConvKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).unwrap_or(NONE);
        ConvKernelIds {
            conv2d: k("conv2d"),
            conv2d_dx: k("conv2d_dx"),
            conv2d_dw: k("conv2d_dw"),
            conv2d_gd: k("conv2d_gd"),
            conv2d_gd_reg: k("conv2d_gd_reg"),
            conv2d_gd_dx: k("conv2d_gd_dx"),
            conv2d_gd_dw: k("conv2d_gd_dw"),
            conv2d_tiled: k("conv2d_tiled"),
            conv_bias: k("conv_bias"),
            conv_bias_reg: k("conv_bias_reg"),
            bias_add: k("bias_add"),
            bias_grad: k("bias_grad"),
            bn_stats: k("bn_stats"),
            bn_running: k("bn_running"),
            bn_train: k("bn_train"),
            bn_eval: k("bn_eval"),
            bn_dstats: k("bn_dstats"),
            bn_dx: k("bn_dx"),
            bn_dgamma: k("bn_dgamma"),
            bn_dbeta: k("bn_dbeta"),
            silu: k("silu"),
            silu_bwd: k("silu_bwd"),
            leaky_relu: k("leaky_relu"),
            leaky_relu_bwd: k("leaky_relu_bwd"),
            sigmoid: k("sigmoid"),
            sigmoid_bwd: k("sigmoid_bwd"),
            conv_act: k("conv_act"),
            conv_act_tiled: k("conv_act_tiled"),
            conv_act_reg: k("conv_act_reg"),
            avgpool2d: k("avgpool2d"),
            avgpool2d_dx: k("avgpool2d_dx"),
            resize_bilinear: k("resize_bilinear"),
            resize_bilinear_dx: k("resize_bilinear_dx"),
            resize_nearest: k("resize_nearest"),
            resize_nearest_dx: k("resize_nearest_dx"),
            pixel_shuffle: k("pixel_shuffle"),
            pixel_shuffle_dx: k("pixel_shuffle_dx"),
            maxpool5: k("maxpool5"),
            maxpool5_dx: k("maxpool5_dx"),
            upsample2: k("upsample2"),
            upsample2_dx: k("upsample2_dx"),
            softmax_k: k("softmax_k"),
            softmax_k_dx: k("softmax_k_dx"),
            weighted_gap: k("weighted_gap"),
            weighted_gap_dx: k("weighted_gap_dx"),
            weighted_gap_dm: k("weighted_gap_dm"),
            add_chan_bcast: k("add_chan_bcast"),
            add_chan_bcast_dv: k("add_chan_bcast_dv"),
            broadcast_add_hw: k("broadcast_add_hw"),
            broadcast_add_hw_da: k("broadcast_add_hw_da"),
            convex_upsample: k("convex_upsample"),
            convex_upsample_dmask: k("convex_upsample_dmask"),
            convex_upsample_dd: k("convex_upsample_dd"),
            mul: k("mul"),
            scale_chan: k("scale_chan"),
            concat2: k("concat2"),
            concat_split: k("concat_split"),
            chan_place: k("chan_place"),
            add2: k("add2"),
            add_inplace: k("add_inplace"),
            axpy: k("axpy"),
        }
    }

    /// Assert a kernel was registered, panicking with its NAME if not.
    ///
    /// This is the payoff over bare index constants: the failure is
    /// `kernel 'conv2d_gd' not registered` at the dispatch site, instead of a
    /// silently wrong kernel running and producing plausible numbers.
    #[inline]
    pub fn need(&self, id: usize, what: &str) -> usize {
        assert_ne!(id, NONE, "kernel `{what}` is not registered in this model's PIPELINES");
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two models with DIFFERENT pipeline orders must resolve to their own
    /// indices. This is the property the whole crate exists for: with positional
    /// constants, blocks written against model A silently dispatch the wrong
    /// kernel under model B.
    #[test]
    fn resolve_is_order_independent() {
        let a: &[(&str, &str)] = &[("conv2d", ""), ("bn_stats", ""), ("silu", "")];
        let b: &[(&str, &str)] = &[("silu", ""), ("conv2d", ""), ("bn_stats", "")];
        let ia = ConvKernelIds::resolve(a);
        let ib = ConvKernelIds::resolve(b);

        assert_eq!((ia.conv2d, ia.bn_stats, ia.silu), (0, 1, 2));
        assert_eq!((ib.conv2d, ib.bn_stats, ib.silu), (1, 2, 0));

        // Each id indexes its OWN pipeline back to the same kernel name.
        assert_eq!(a[ia.conv2d].0, "conv2d");
        assert_eq!(b[ib.conv2d].0, "conv2d");
    }

    #[test]
    fn unregistered_kernels_resolve_to_none() {
        let ids = ConvKernelIds::resolve(&[("conv2d", "")]);
        assert_eq!(ids.conv2d, 0);
        assert_eq!(ids.maxpool5, NONE, "a kernel the model never registered");
        assert_eq!(ids.need(ids.conv2d, "conv2d"), 0);
    }

    #[test]
    #[should_panic(expected = "kernel `maxpool5` is not registered")]
    fn need_panics_with_the_kernel_name() {
        let ids = ConvKernelIds::resolve(&[("conv2d", "")]);
        ids.need(ids.maxpool5, "maxpool5");
    }

    /// Resolving against the real yolo pipeline order must reproduce yolo's own
    /// hand-written constants exactly — the equivalence that makes the migration
    /// behaviour-preserving. Mirrors `yolo/src/net.rs`'s frozen index order.
    #[test]
    fn resolve_reproduces_yolos_hand_written_indices() {
        let yolo_pipelines: &[(&str, &str)] = &[
            ("conv2d", ""),
            ("conv2d_dx", ""),
            ("conv2d_dw", ""),
            ("bn_stats", ""),
            ("bn_running", ""),
            ("bn_train", ""),
            ("bn_eval", ""),
            ("bn_dstats", ""),
            ("bn_dx", ""),
            ("bn_dgamma", ""),
            ("bn_dbeta", ""),
            ("silu", ""),
            ("silu_bwd", ""),
            ("maxpool5", ""),
            ("maxpool5_dx", ""),
            ("upsample2", ""),
            ("upsample2_dx", ""),
            ("concat2", ""),
            ("concat_split", ""),
            ("add2", ""),
            ("add_inplace", ""),
        ];
        let ids = ConvKernelIds::resolve(yolo_pipelines);
        // The literals here are yolo/src/net.rs's consts.
        assert_eq!(ids.conv2d, 0);
        assert_eq!(ids.conv2d_dx, 1);
        assert_eq!(ids.conv2d_dw, 2);
        assert_eq!(ids.bn_stats, 3);
        assert_eq!(ids.bn_running, 4);
        assert_eq!(ids.bn_train, 5);
        assert_eq!(ids.bn_eval, 6);
        assert_eq!(ids.bn_dstats, 7);
        assert_eq!(ids.bn_dx, 8);
        assert_eq!(ids.bn_dgamma, 9);
        assert_eq!(ids.bn_dbeta, 10);
        assert_eq!(ids.silu, 11);
        assert_eq!(ids.silu_bwd, 12);
        assert_eq!(ids.maxpool5, 13);
        assert_eq!(ids.maxpool5_dx, 14);
        assert_eq!(ids.upsample2, 15);
        assert_eq!(ids.upsample2_dx, 16);
        assert_eq!(ids.concat2, 17);
        assert_eq!(ids.concat_split, 18);
        assert_eq!(ids.add2, 19);
    }
}
