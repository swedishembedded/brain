// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seam that lets a trainable prologue REPLACE a frozen decoder's skip
//! concatenation, instead of only adding a residual to it.
//!
//! `crates/controlnet` conditions a frozen backbone with a plain additive
//! residual at each injection point - `Unet::run_with_control` adds a
//! ControlNet's zero-conv output onto the skip buffer before the up path's
//! concat runs unchanged. SUPIR's `ZeroSFT` adaptors do not fit that shape at
//! all: they REPLACE the concat itself, reading BOTH sides of the join
//! (`h_ori`, the running up-path hidden state, and `skip`, the popped
//! down-path tensor) and producing the joined tensor by a GroupNorm-affine
//! lerp rather than a channel-axis concat. No amount of residual-adding
//! expresses that, which is why this is a second seam and not a
//! generalisation of the first.
//!
//! [`SkipFuse`] lives here, in `vae::blocks`, and not in `crates/model`
//! (where [`model::attninject::CrossAttnInject`], the OTHER seam
//! `sdxlunet::model::Rec` installs, lives): `crates/vae` depends on
//! `crates/model` and not the reverse, so only `vae::blocks::Builder` can
//! record a trainable prologue - the GroupNorm + zero-init convs a real
//! `SkipFuse` implementor needs are exactly what this module already builds.
//!
//! A `SkipFuse` implementor is Phase 4's job (SUPIR's `adaptors.rs`), not
//! this module's: everything here is the seam plus a no-op default, gated by
//! bit-identity against the un-fused graph.

use gpu_core::DeviceBuffer;

use super::Builder;

/// One NCHW feature map moving through a [`SkipFuse`] call.
#[derive(Clone)]
pub struct Map {
    pub buf: DeviceBuffer,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

/// Replaces (or, by default, reproduces) the up path's skip join, the
/// post-mid-block hook, and each up-block's pre-upsample hook.
///
/// The three methods mirror `LightGLVUNet.forward`'s three injection shapes
/// exactly: [`SkipFuse::fuse_mid`] is the post-middle-block adaptor call with
/// no concat at all (a genuinely distinct call, not a zero-width special
/// case of [`SkipFuse::fuse_skip`]); [`SkipFuse::fuse_skip`] is the
/// concat-replacing join at every one of the 9 output blocks;
/// [`SkipFuse::pre_upsample`] is the cross-attention adaptor some up-blocks
/// apply to their running hidden state right before the nearest-2x upsample.
///
/// A real implementor's [`SkipFuse::fuse_skip`] MUST return
/// `c == h_ori.c + skip.c` - proven by walking SDXL's skip stack in up-path
/// pop order against SUPIR's own channel tables, which reproduces both
/// exactly (`control_c == skip_c` at every join, because the trunk mirrors
/// the encoder that produced the skip). So the frozen SDXL up path's
/// resnets, whose input width is `prev_output_channel + skip_channels`,
/// need no re-import: a real `SkipFuse` slots into exactly the width a plain
/// concat would have produced.
pub trait SkipFuse: Send + Sync {
    /// Extra kernels this implementor's forward dispatches need, beyond
    /// whatever `Builder`'s own kernel set already carries. Checked up front
    /// by [`crate::blocks`]'s callers (mirroring
    /// `model::attninject::CrossAttnInject::kernels`) so a missing kernel is
    /// named at construction, not discovered mid-record.
    fn kernels(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// How many [`SkipFuse::fuse_skip`] calls a full forward makes - the
    /// number a caller asserts the graph actually recorded against, the same
    /// shape `model::attninject::CrossAttnInject::sites` is checked with.
    fn joins(&self) -> usize;

    /// Replace the up path's `k`-th skip join (`k` in pop order, `0` first).
    /// The default concat would be `torch.cat([h_ori, skip], dim=1)`; a real
    /// implementor reads both sides and returns a fused map of the SAME
    /// total width (see the trait doc's shape-preservation note).
    fn fuse_skip(&self, b: &mut Builder<'_>, k: usize, h_ori: &Map, skip: &Map) -> Map;

    /// Called once, right after the mid block, before the up path's first
    /// join. Identity by default.
    fn fuse_mid(&self, _b: &mut Builder<'_>, x: &Map) -> Map {
        x.clone()
    }

    /// Called on up-block `i`'s running hidden state (`i` indexes the
    /// UP-BLOCK, `0..levels-1` - NOT the join index: there are fewer
    /// up-blocks than joins), immediately before that block's nearest-2x
    /// upsample. Identity by default.
    fn pre_upsample(&self, _b: &mut Builder<'_>, _i: usize, x: &Map) -> Map {
        x.clone()
    }
}
