// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DaViT's per-stage overlapping patch-embed downsample (`ConvEmbed` in the
//! reference) - a strided conv plus one LayerNorm, whose placement flips by
//! stage: stage 0 takes raw NCHW pixels (no pre-norm branch even applies -
//! the reference's `pre_norm` guard is gated on the input already being a
//! `[B,N,C]` sequence) and norms AFTER the conv; stages 1-3 take the
//! previous stage's `[B,N,C]` sequence output, norm BEFORE the conv, and
//! skip the post-norm.
//!
//! Conv dispatch reuses `vision::blocks::Conv` (verified precedent for a
//! grouped/dense conv with an explicit weight-name mapping,
//! `vision::blocks::ConvSpec`/`ConvNames`); the LayerNorm and the
//! NCHW<->sequence round trip around it are this module's own, mirroring
//! `fastvlm::encoder::AttentionBlock::forward`'s `nchw_nlc`/`nlc_nchw`
//! pattern (the direct precedent for moving between conv-stage and
//! sequence-stage tensor layouts in this repo).

use gpu_core::{Gpu, Step};
use vision::blocks::{Conv, ConvNames, ConvSpec, Norm};
use vision::ids::ConvKernelIds;
use vision::net::{Ctx, Shape};

use super::config::DavitPatchSpec;

/// Kernel indices this module dispatches directly (LayerNorm and the
/// NCHW<->sequence transpose) - the conv itself resolves its own ids from
/// [`ConvKernelIds`] internally.
#[derive(Clone, Copy)]
pub struct PatchEmbedKernelIds {
    pub nchw_nlc: usize,
    pub nlc_nchw: usize,
    pub layernorm: usize,
}

impl PatchEmbedKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> PatchEmbedKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        PatchEmbedKernelIds { nchw_nlc: k("nchw_nlc"), nlc_nchw: k("nlc_nchw"), layernorm: k("layernorm") }
    }
}

pub struct PatchEmbed {
    conv: Conv,
    conv_ids: ConvKernelIds,
    /// Post-conv sequence-form output.
    seq_out: gpu_core::DeviceBuffer,
    /// LayerNorm output (aliases `seq_out`'s buffer identity when no norm
    /// runs in a given position - always allocated fresh here for clarity).
    normed: gpu_core::DeviceBuffer,
    pre_norm: bool,
    norm_w: String,
    norm_b: String,
    out_shape: Shape,
    eps: f32,
}

impl PatchEmbed {
    /// `in_shape`: NCHW shape of what this stage receives - `c` is the
    /// PREVIOUS stage's output channel width (or 3 for stage 0's raw
    /// pixels); `h`/`w` are the previous stage's output spatial size (or
    /// the input image size for stage 0).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        conv_ids: &ConvKernelIds,
        prefix: &str,
        in_shape: Shape,
        patch: &DavitPatchSpec,
        out_ch: u32,
        eps: f32,
        train: bool,
    ) -> PatchEmbed {
        let ctx = Ctx::new(gpu, conv_ids);
        let spec = ConvSpec::depthwise(out_ch, patch.kernel, patch.stride, patch.padding, vision::blocks::Act::None)
            .with_norm(Norm::None)
            .with_bias();
        // ConvSpec::depthwise sets groups = cout; DaViT's patch-embed conv is
        // DENSE (every input channel contributes to every output channel),
        // so groups must be forced back to 1 after borrowing `depthwise`'s
        // bias/no-norm/no-act defaults.
        let spec = ConvSpec { groups: 1, ..spec };
        let names = ConvNames {
            bias: format!("{prefix}.proj.bias"),
            weight: format!("{prefix}.proj.weight"),
            gamma: String::new(),
            beta: String::new(),
            run_mean: String::new(),
            run_var: String::new(),
        };
        let conv = Conv::with_names(&ctx, &format!("{prefix}.proj"), names, in_shape, spec, train);
        let out_shape = conv.out_shape;
        let tokens = out_shape.h * out_shape.w;
        let numel = (tokens * out_shape.c) as u64;

        PatchEmbed {
            conv_ids: *conv_ids,
            seq_out: gpu.storage(numel),
            normed: gpu.storage(numel),
            pre_norm: patch.pre_norm,
            norm_w: format!("{prefix}.norm.weight"),
            norm_b: format!("{prefix}.norm.bias"),
            out_shape,
            eps,
            conv,
        }
    }

    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }

    pub fn tokens(&self) -> u32 {
        self.out_shape.h * self.out_shape.w
    }

    /// Stage 0 (raw NCHW pixels in, `pre_norm = false`): conv directly on
    /// `x_in`, then `nchw_nlc`, then LayerNorm (post-norm).
    pub fn forward_from_pixels(
        &self,
        gpu: &Gpu,
        k: &PatchEmbedKernelIds,
        ps: &paramstore::ParamStore,
        x_in: &gpu_core::DeviceBuffer,
    ) -> &gpu_core::DeviceBuffer {
        assert!(!self.pre_norm, "forward_from_pixels is stage 0's post-norm path only");
        let ctx = Ctx::new(gpu, &self.conv_ids);
        self.conv.forward(&ctx, ps, x_in);
        let conv_out = self.conv.out();

        let c = self.out_shape.c;
        let rows = self.tokens();
        let total = rows * c;
        let mut s: Vec<Step> = Vec::new();
        s.push(gpu.step(k.nchw_nlc, &[conv_out, &self.seq_out], &[total, c, rows], total));
        let ln = model::block::LayerNormIds::resolve_fwd(gpu, k.layernorm);
        s.push(model::block::layernorm_fwd(gpu, &ln, &self.seq_out, ps.w(&self.norm_w), ps.w(&self.norm_b), &self.normed, c, rows, self.eps));
        gpu.submit(&[], &s);
        &self.normed
    }

    /// Stages 1..3 (sequence in, `pre_norm = true`): LayerNorm the input
    /// sequence first, `nlc_nchw`, conv, `nchw_nlc` back - no post-norm.
    pub fn forward_from_sequence(
        &self,
        gpu: &Gpu,
        k: &PatchEmbedKernelIds,
        ps: &paramstore::ParamStore,
        x_in: &gpu_core::DeviceBuffer,
        in_shape: Shape,
    ) -> &gpu_core::DeviceBuffer {
        assert!(self.pre_norm, "forward_from_sequence is stages 1..3's pre-norm path only");
        let in_rows = in_shape.h * in_shape.w;
        let in_c = in_shape.c;
        let in_total = in_rows * in_c;

        let normed_in = gpu.storage(in_total as u64);
        let nchw_in = gpu.storage(in_total as u64);
        let mut s: Vec<Step> = Vec::new();
        let ln = model::block::LayerNormIds::resolve_fwd(gpu, k.layernorm);
        s.push(model::block::layernorm_fwd(gpu, &ln, x_in, ps.w(&self.norm_w), ps.w(&self.norm_b), &normed_in, in_c, in_rows, self.eps));
        s.push(gpu.step(k.nlc_nchw, &[&normed_in, &nchw_in], &[in_total, in_c, in_rows], in_total));
        gpu.submit(&[], &s);

        let ctx = Ctx::new(gpu, &self.conv_ids);
        self.conv.forward(&ctx, ps, &nchw_in);
        let conv_out = self.conv.out();

        let c = self.out_shape.c;
        let rows = self.tokens();
        let total = rows * c;
        let s2 = vec![gpu.step(k.nchw_nlc, &[conv_out, &self.seq_out], &[total, c, rows], total)];
        gpu.submit(&[], &s2);
        &self.seq_out
    }
}
