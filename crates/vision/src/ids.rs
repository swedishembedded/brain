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
    // ---- conv ----
    pub conv2d: usize,
    pub conv2d_dx: usize,
    pub conv2d_dw: usize,
    pub conv2d_tiled: usize,
    pub conv_bias: usize,
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
    // ---- activations ----
    pub silu: usize,
    pub silu_bwd: usize,
    // ---- fused conv -> affine -> act (inference only) ----
    pub conv_act: usize,
    pub conv_act_tiled: usize,
    pub conv_act_reg: usize,
    // ---- spatial ----
    pub maxpool5: usize,
    pub maxpool5_dx: usize,
    pub upsample2: usize,
    pub upsample2_dx: usize,
    // ---- elementwise / plumbing ----
    pub concat2: usize,
    pub concat_split: usize,
    pub chan_place: usize,
    pub add2: usize,
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
            conv2d_tiled: k("conv2d_tiled"),
            conv_bias: k("conv_bias"),
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
            conv_act: k("conv_act"),
            conv_act_tiled: k("conv_act_tiled"),
            conv_act_reg: k("conv_act_reg"),
            maxpool5: k("maxpool5"),
            maxpool5_dx: k("maxpool5_dx"),
            upsample2: k("upsample2"),
            upsample2_dx: k("upsample2_dx"),
            concat2: k("concat2"),
            concat_split: k("concat_split"),
            chan_place: k("chan_place"),
            add2: k("add2"),
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
