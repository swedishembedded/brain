// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FastViTHD vision encoder — built on brain's `crates/vision` Conv framework.
//!
//! Five stages (RepMixer conv token-mixing in 0–2, self-attention in 3–4) built
//! from the shared `Conv`/`ConvSpec` primitives + the attention kernels; no
//! net-new device kernels. This module grows the encoder incrementally; today it
//! establishes the conv pipeline + `Ctx` scaffolding (the same pattern yolo/depth
//! use) and a smoke that runs a `Conv` in this crate.

use std::sync::OnceLock;

use gpu_core::{f, DeviceBuffer};
use model::vit::{chunked_attn_fwd, VitKernelIds, VitShape};
use paramstore::ParamStore;
use vision::{Act, Conv, ConvKernelIds, ConvSpec, Ctx, Norm, Shape};

/// FastViTHD's fixed attention head dimension (num_heads = channels / 32).
const ATTN_HEAD_DIM: u32 = 32;

/// Map the tower PIPELINES into `model::vit`'s kernel-id struct (only the fields
/// the attention core dispatches need to be valid; the rest are unused here).
fn vit_ids() -> VitKernelIds {
    VitKernelIds {
        layernorm: kidx("layernorm"),
        matmul: kidx("matmul"),
        matmul_rows: kidx("matmul_rows"),
        bias_add: kidx("bias_add"),
        gelu_erf: kidx("gelu_erf"),
        scale_chan: kidx("scale_chan"),
        add2: kidx("add2"),
        attn_scores_cross: kidx("attn_scores_cross"),
        attn_softmax_cross: kidx("attn_softmax_cross"),
        attn_apply_cross: kidx("attn_apply_cross"),
        ln_head: 0,
        rope2d: 0,
    }
}

/// Index of a manually-dispatched kernel (not covered by `ConvKernelIds`) in
/// [`PIPELINES`], by name. Panics if absent — a programming error.
fn kidx(name: &str) -> usize {
    PIPELINES.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("kernel `{name}` not in FastViTHD PIPELINES"))
}

/// Kernel registry for the FastViTHD tower: the conv/BN/act family (RepMixer,
/// MobileOne, ConvFFN, PatchEmbed, SE) plus the transpose + attention kernels the
/// stage-4/5 attention blocks need. Resolved by name, so a superset is fine —
/// absent kernels map to `NONE` and only error if actually dispatched.
pub const PIPELINES: &[(&str, &str)] = &[
    // --- conv / bn / activation ---
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv2d_gd", kernels::CONV2D_GD),
    ("conv2d_gd_reg", kernels::CONV2D_GD_REG),
    ("conv2d_gd_dx", kernels::CONV2D_GD_DX),
    ("conv2d_gd_dw", kernels::CONV2D_GD_DW),
    ("conv2d_tiled", kernels::CONV2D_TILED),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("conv_act", kernels::CONV_ACT),
    ("conv_act_tiled", kernels::CONV_ACT_TILED),
    ("conv_act_reg", kernels::CONV_ACT_REG),
    ("conv_epilogue", kernels::CONV_EPILOGUE),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("im2col", kernels::IM2COL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("avgpool2d", kernels::AVGPOOL2D),
    ("avgpool2d_dx", kernels::AVGPOOL2D_DX),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
    ("add_chan_bcast", kernels::ADD_CHAN_BCAST),
    ("add_chan_bcast_dv", kernels::ADD_CHAN_BCAST_DV),
    ("scale_chan", kernels::SCALE_CHAN),
    ("film_chan", kernels::FILM_CHAN), // LayerScale: y = x·(1+s)+b, s=ls-1, b=0
    ("chan_place", kernels::CHAN_PLACE),
    // --- transpose + attention (stage-4/5 blocks) ---
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("layernorm", kernels::LAYERNORM),
    ("gelu", kernels::GELU),
    ("gelu_erf", kernels::GELU_ERF),
    ("matmul", kernels::MATMUL),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
];

/// Cached conv kernel-id resolution against [`PIPELINES`].
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

/// Build a fresh conv `Ctx` bound to `gpu` and the tower kernel ids.
pub fn ctx(gpu: &gpu_core::Gpu) -> Ctx<'_> {
    Ctx::new(gpu, ids())
}

/// The atomic FastViTHD conv primitive: a (grouped/depthwise/dense) conv with an
/// optional BatchNorm, an optional **per-channel** bias (`add_chan_bcast`, which
/// unlike the dense-only fused `conv_bias` handles grouped convs), and an optional
/// **erf**-GELU. Every FastViTHD conv (MobileOne, RepMixer, ConvFFN's fc1/fc2,
/// RepCPE, PatchEmbed, conv_exp) is one of these. Weight keys: `{prefix}.conv.*`
/// (+ `.bn.*` when `bn`) and `{prefix}.bias` (when `bias`).
pub struct ConvUnit {
    conv: Conv,
    bias_name: Option<String>,
    gelu: bool,
    out_shape: Shape,
    biased: DeviceBuffer,
    activated: DeviceBuffer,
    k_add_chan: usize,
    k_gelu: usize,
}

impl ConvUnit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, k: u32, stride: u32, pad: u32, groups: u32, bn: bool, bias: bool, gelu: bool) -> ConvUnit {
        let spec = ConvSpec {
            cout,
            k,
            stride,
            pad,
            groups,
            dilation: 1,
            norm: if bn { Norm::Bn } else { Norm::None },
            act: Act::None,
            bias: false, // bias is applied by us (per-channel, grouped-safe)
        };
        // ConvNames::brain appends `.conv.weight`/`.bn.*` to the prefix.
        let conv = Conv::with_spec(ctx, prefix, in_shape, spec, true);
        let out_shape = conv.out_shape;
        let n = out_shape.numel() as u64;
        ConvUnit {
            bias_name: bias.then(|| format!("{prefix}.bias")),
            gelu,
            out_shape,
            biased: ctx.gpu.storage(n),
            activated: ctx.gpu.storage(n),
            k_add_chan: kidx("add_chan_bcast"),
            k_gelu: kidx("gelu_erf"),
            conv,
        }
    }

    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut p = self.conv.param_list();
        if let Some(b) = &self.bias_name {
            p.push((b.clone(), self.out_shape.c as usize));
        }
        p
    }

    /// Run conv → (+bias) → (GELU); returns the final output buffer.
    pub fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) -> &'a DeviceBuffer {
        self.conv.forward(ctx, ps, x_in);
        let mut cur: &DeviceBuffer = self.conv.out();
        if let Some(b) = &self.bias_name {
            let (nn, c, hw) = (self.out_shape.n, self.out_shape.c, self.out_shape.h * self.out_shape.w);
            ctx.gpu.submit(&[], &[ctx.step(self.k_add_chan, &[cur, ps.w(b), &self.biased], &[nn, c, hw], self.out_shape.numel())]);
            cur = &self.biased;
        }
        if self.gelu {
            let tot = self.out_shape.numel();
            ctx.gpu.submit(&[], &[ctx.step(self.k_gelu, &[cur, &self.activated], &[tot], tot)]);
            cur = &self.activated;
        }
        // SAFETY of lifetimes: `cur` borrows either self.conv.out(), self.biased,
        // or self.activated — all owned by `self`, so `'a` is valid.
        cur
    }
}

/// ConvFFN — the FastViTHD channel-mixer: depthwise 7×7 + BN → 1×1 (`fc1`) +bias
/// → erf-GELU → 1×1 (`fc2`) +bias. Present in every RepMixer and Attention block.
pub struct ConvFFN {
    dw: ConvUnit,
    fc1: ConvUnit,
    fc2: ConvUnit,
    out_shape: Shape,
}

impl ConvFFN {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, ch: u32, mlp_ratio: u32) -> ConvFFN {
        let dw = ConvUnit::new(ctx, &format!("{prefix}.dw"), in_shape, ch, 7, 1, 3, ch, true, false, false);
        let fc1 = ConvUnit::new(ctx, &format!("{prefix}.fc1"), dw.out_shape(), ch * mlp_ratio, 1, 1, 0, 1, false, true, true);
        let fc2 = ConvUnit::new(ctx, &format!("{prefix}.fc2"), fc1.out_shape(), ch, 1, 1, 0, 1, false, true, false);
        let out_shape = fc2.out_shape();
        ConvFFN { dw, fc1, fc2, out_shape }
    }
    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut p = self.dw.param_list();
        p.extend(self.fc1.param_list());
        p.extend(self.fc2.param_list());
        p
    }
    pub fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) -> &'a DeviceBuffer {
        let d = self.dw.forward(ctx, ps, x_in);
        let h = self.fc1.forward(ctx, ps, d);
        self.fc2.forward(ctx, ps, h)
    }
}

/// RepMixerBlock — the FastViTHD stage-0–2 block: a RepMixer token-mixer (a single
/// fused depthwise 3×3 conv + bias, no residual — folded) then a ConvFFN with a
/// LayerScale residual `x = mixer(x) + layer_scale ⊙ ConvFFN(mixer(x))`. The
/// LayerScale is stored as an `sb` buffer `[2C]` (`[scale=ls-1, shift=0]`) applied
/// by `film_chan`, so it needs no per-channel-multiply kernel.
pub struct RepMixerBlock {
    mixer: ConvUnit,
    ffn: ConvFFN,
    ls_name: String,
    scaled: DeviceBuffer,
    out: DeviceBuffer,
    out_shape: Shape,
    k_film: usize,
    k_add2: usize,
}

impl RepMixerBlock {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, ch: u32, mlp_ratio: u32) -> RepMixerBlock {
        let mixer = ConvUnit::new(ctx, &format!("{prefix}.token_mixer"), in_shape, ch, 3, 1, 1, ch, false, true, false);
        let ffn = ConvFFN::new(ctx, &format!("{prefix}.convffn"), mixer.out_shape(), ch, mlp_ratio);
        let out_shape = ffn.out_shape();
        let n = out_shape.numel() as u64;
        RepMixerBlock {
            ls_name: format!("{prefix}.layer_scale_sb"),
            scaled: ctx.gpu.storage(n),
            out: ctx.gpu.storage(n),
            out_shape,
            k_film: kidx("film_chan"),
            k_add2: kidx("add2"),
            mixer,
            ffn,
        }
    }
    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut p = self.mixer.param_list();
        p.extend(self.ffn.param_list());
        p.push((self.ls_name.clone(), 2 * self.out_shape.c as usize)); // [scale(C), shift(C)]
        p
    }
    pub fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) -> &'a DeviceBuffer {
        let t = self.mixer.forward(ctx, ps, x_in);
        let f = self.ffn.forward(ctx, ps, t);
        let (nn, c, h, w) = (self.out_shape.n, self.out_shape.c, self.out_shape.h, self.out_shape.w);
        let tot = self.out_shape.numel();
        // scaled = f · layer_scale  (film_chan with s=ls-1, b=0)
        ctx.gpu.submit(&[], &[ctx.step(self.k_film, &[f, ps.w(&self.ls_name), &self.scaled], &[nn, c, h, w], tot)]);
        // out = mixer(x) + scaled
        ctx.gpu.submit(&[], &[ctx.step(self.k_add2, &[t, &self.scaled, &self.out], &[tot], tot)]);
        &self.out
    }
}

/// PatchEmbed — the FastViTHD inter-stage downsample (stride-2, `in_ch → out_ch`):
/// a grouped large-kernel conv (7×7, stride 2, `groups = in_ch`, fused
/// ReparamLargeKernelConv) then a 1×1 MobileOne. Halves H×W, `out_ch` must be a
/// multiple of `in_ch` (FastViTHD doubles the channels).
pub struct PatchEmbed {
    rlk: ConvUnit,
    proj: ConvUnit,
    out_shape: Shape,
}

impl PatchEmbed {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, out_ch: u32) -> PatchEmbed {
        assert!(out_ch % in_shape.c == 0, "PatchEmbed out_ch must be a multiple of in_ch");
        // Grouped 7×7 stride-2 (groups = in_ch), +bias, +GELU.
        let rlk = ConvUnit::new(ctx, &format!("{prefix}.rlk"), in_shape, out_ch, 7, 2, 3, in_shape.c, false, true, true);
        // 1×1 MobileOne (dense), +bias, +GELU.
        let proj = ConvUnit::new(ctx, &format!("{prefix}.proj"), rlk.out_shape(), out_ch, 1, 1, 0, 1, false, true, true);
        let out_shape = proj.out_shape();
        PatchEmbed { rlk, proj, out_shape }
    }
    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut p = self.rlk.param_list();
        p.extend(self.proj.param_list());
        p
    }
    pub fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) -> &'a DeviceBuffer {
        let r = self.rlk.forward(ctx, ps, x_in);
        self.proj.forward(ctx, ps, r)
    }
}

/// AttentionBlock — the FastViTHD stage-3/4 block: `x = x + ls1 ⊙ MHSA(LN(x))`
/// then `x = x + ls2 ⊙ ConvFFN(x)`. The MHSA runs over the conv feature map by
/// transposing NCHW→NLC (`nchw_nlc`), a per-token LayerNorm over C, a fused-qkv
/// Linear (no bias, no RoPE, no QK-norm, head_dim 32), the shared
/// `model::vit::chunked_attn_fwd` full-attention core, a biased proj Linear, then
/// NLC→NCHW (`nlc_nchw`). LayerScales use `film_chan` (sb = [ls-1, 0]).
pub struct AttentionBlock {
    ch: u32,
    heads: u32,
    ffn: ConvFFN,
    out_shape: Shape,
    // params
    norm_w: String,
    norm_b: String,
    qkv_w: String,
    proj_w: String,
    proj_b: String,
    ls1: String,
    ls2: String,
    // scratch (rows = H*W, N = 1)
    nlc: DeviceBuffer,
    normed: DeviceBuffer,
    qkv: DeviceBuffer,
    ctxb: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    proj: DeviceBuffer,
    back: DeviceBuffer,
    scaled1: DeviceBuffer,
    res1: DeviceBuffer,
    scaled2: DeviceBuffer,
    out: DeviceBuffer,
}

impl AttentionBlock {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, ch: u32, mlp_ratio: u32) -> AttentionBlock {
        assert_eq!(in_shape.c, ch);
        assert_eq!(ch % ATTN_HEAD_DIM, 0, "channels must be a multiple of head_dim 32");
        let rows = in_shape.h * in_shape.w;
        let heads = ch / ATTN_HEAD_DIM;
        let g = ctx.gpu;
        let rc = (rows * ch) as u64;
        let slab = (heads * rows * rows) as u64;
        let ffn = ConvFFN::new(ctx, &format!("{prefix}.convffn"), in_shape, ch, mlp_ratio);
        AttentionBlock {
            ch,
            heads,
            out_shape: in_shape,
            norm_w: format!("{prefix}.norm.weight"),
            norm_b: format!("{prefix}.norm.bias"),
            qkv_w: format!("{prefix}.token_mixer.qkv.weight"),
            proj_w: format!("{prefix}.token_mixer.proj.weight"),
            proj_b: format!("{prefix}.token_mixer.proj.bias"),
            ls1: format!("{prefix}.layer_scale_1_sb"),
            ls2: format!("{prefix}.layer_scale_2_sb"),
            nlc: g.storage(rc),
            normed: g.storage(rc),
            qkv: g.storage(3 * rc),
            ctxb: g.storage(rc),
            scores: g.storage(slab),
            probs: g.storage(slab),
            proj: g.storage(rc),
            back: g.storage(rc),
            scaled1: g.storage(rc),
            res1: g.storage(rc),
            scaled2: g.storage(rc),
            out: g.storage(rc),
            ffn,
        }
    }
    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let (c, c3) = (self.ch as usize, 3 * self.ch as usize);
        let mut p = vec![
            (self.norm_w.clone(), c),
            (self.norm_b.clone(), c),
            (self.qkv_w.clone(), c3 * c),
            (self.proj_w.clone(), c * c),
            (self.proj_b.clone(), c),
            (self.ls1.clone(), 2 * c),
            (self.ls2.clone(), 2 * c),
        ];
        p.extend(self.ffn.param_list());
        p
    }
    pub fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) -> &'a DeviceBuffer {
        let g = ctx.gpu;
        let (nn, ch, h, w) = (self.out_shape.n, self.out_shape.c, self.out_shape.h, self.out_shape.w);
        let rows = h * w;
        let total = self.out_shape.numel();
        let eps = 1e-5f32;

        let mut s = Vec::new();
        // NCHW -> NLC, LayerNorm over C, fused-qkv Linear (no bias).
        s.push(g.step(kidx("nchw_nlc"), &[x_in, &self.nlc], &[total, ch, rows], total));
        s.push(g.step(kidx("layernorm"), &[&self.nlc, ps.w(&self.norm_w), ps.w(&self.norm_b), &self.normed], &[ch, rows, f(eps)], rows));
        let c3 = 3 * ch;
        s.push(g.step(kidx("matmul_rows"), &[&self.normed, ps.w(&self.qkv_w), &self.qkv], &[rows, ch, c3], rows.div_ceil(8) * c3));
        // Full attention over the H*W tokens.
        let sh = VitShape { dim: ch, heads: self.heads, mlp: 0, eps };
        chunked_attn_fwd(g, &vit_ids(), &sh, &self.qkv, &self.ctxb, &self.scores, &self.probs, &[(0, rows)], rows, &mut s);
        // proj + bias, back to NCHW, ls1 residual.
        s.push(g.step(kidx("matmul_rows"), &[&self.ctxb, ps.w(&self.proj_w), &self.proj], &[rows, ch, ch], rows.div_ceil(8) * ch));
        s.push(g.step(kidx("bias_add"), &[&self.proj, ps.w(&self.proj_b)], &[rows, ch], rows * ch));
        s.push(g.step(kidx("nlc_nchw"), &[&self.proj, &self.back], &[total, ch, rows], total));
        s.push(g.step(kidx("film_chan"), &[&self.back, ps.w(&self.ls1), &self.scaled1], &[nn, ch, h, w], total));
        s.push(g.step(kidx("add2"), &[x_in, &self.scaled1, &self.res1], &[total], total));
        g.submit(&[], &s);
        // ConvFFN (submits internally) + ls2 residual.
        let f_out = self.ffn.forward(ctx, ps, &self.res1);
        g.submit(
            &[],
            &[
                g.step(kidx("film_chan"), &[f_out, ps.w(&self.ls2), &self.scaled2], &[nn, ch, h, w], total),
                g.step(kidx("add2"), &[&self.res1, &self.scaled2, &self.out], &[total], total),
            ],
        );
        &self.out
    }
}

/// RepCPE — the FastViTHD conditional positional encoding before the attention
/// stages: a single fused depthwise 7×7 conv + bias (the identity skip is folded
/// into the conv), applied in place. It's exactly a [`ConvUnit`], so
/// `repcpe(ctx, prefix, shape, ch)` is a thin constructor.
pub fn repcpe(ctx: &Ctx, prefix: &str, in_shape: Shape, ch: u32) -> ConvUnit {
    ConvUnit::new(ctx, prefix, in_shape, ch, 7, 1, 3, ch, false, true, false)
}

/// One ordered layer of the FastViTHD forward.
enum Layer {
    Conv(ConvUnit),        // stem convs, RepCPE, conv_exp
    Rep(RepMixerBlock),    // stage 0–2 blocks
    Attn(AttentionBlock),  // stage 3–4 blocks
    Down(PatchEmbed),      // inter-stage downsample
}

impl Layer {
    fn out_shape(&self) -> Shape {
        match self {
            Layer::Conv(c) => c.out_shape(),
            Layer::Rep(b) => b.out_shape(),
            Layer::Attn(b) => b.out_shape(),
            Layer::Down(d) => d.out_shape(),
        }
    }
    fn param_list(&self) -> Vec<(String, usize)> {
        match self {
            Layer::Conv(c) => c.param_list(),
            Layer::Rep(b) => b.param_list(),
            Layer::Attn(b) => b.param_list(),
            Layer::Down(d) => d.param_list(),
        }
    }
    fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x: &'a DeviceBuffer) -> &'a DeviceBuffer {
        match self {
            Layer::Conv(c) => c.forward(ctx, ps, x),
            Layer::Rep(b) => b.forward(ctx, ps, x),
            Layer::Attn(b) => b.forward(ctx, ps, x),
            Layer::Down(d) => d.forward(ctx, ps, x),
        }
    }
}

/// FastViTHD encoder: stem (3 MobileOne convs, /4) → 5 stages (RepMixer blocks in
/// 0–2, RepCPE + AttentionBlocks in 3–4) with PatchEmbed downsamples between → a
/// conv_exp MobileOne (last-dim → last-dim·cls_ratio) → feature_select (NCHW →
/// `[tokens, feature_dim]`). Total downsample 64×, so a 1024² image → 256 tokens ×
/// 3072. (conv_exp's SE gate is a later refinement.)
pub struct Encoder {
    layers: Vec<Layer>,
    feature_dim: u32,
    tokens: u32,
    out: DeviceBuffer,
    k_nchw_nlc: usize,
}

impl Encoder {
    /// Build for `layers`/`embed_dims` (5 stages), `mlp_ratio`, `cls_ratio`, and a
    /// square `input` size. `input` must be a multiple of 64 (the total downsample)
    /// and its /64 grid gives the token count.
    pub fn new(ctx: &Ctx, layers: [u32; 5], dims: [u32; 5], mlp_ratio: u32, cls_ratio: u32, input: u32) -> Encoder {
        assert!(input % 64 == 0, "input must be a multiple of the 64× downsample");
        let mut ls = Vec::new();
        let mut shape = Shape::new(1, 3, input, input);
        // Stem: dense 3→d0 /2, depthwise d0 /2, dense 1×1 d0.
        let stem0 = ConvUnit::new(ctx, "stem.0", shape, dims[0], 3, 2, 1, 1, true, true, true);
        shape = stem0.out_shape();
        ls.push(Layer::Conv(stem0));
        let stem1 = ConvUnit::new(ctx, "stem.1", shape, dims[0], 3, 2, 1, dims[0], true, true, true);
        shape = stem1.out_shape();
        ls.push(Layer::Conv(stem1));
        let stem2 = ConvUnit::new(ctx, "stem.2", shape, dims[0], 1, 1, 0, 1, true, true, true);
        shape = stem2.out_shape();
        ls.push(Layer::Conv(stem2));

        for stage in 0..5 {
            let ch = dims[stage];
            let is_attn = stage >= 3;
            if is_attn {
                let cpe = repcpe(ctx, &format!("stage{stage}.cpe"), shape, ch);
                shape = cpe.out_shape();
                ls.push(Layer::Conv(cpe));
            }
            for b in 0..layers[stage] {
                let p = format!("network.{stage}.{b}");
                if is_attn {
                    let blk = AttentionBlock::new(ctx, &p, shape, ch, mlp_ratio);
                    shape = blk.out_shape();
                    ls.push(Layer::Attn(blk));
                } else {
                    let blk = RepMixerBlock::new(ctx, &p, shape, ch, mlp_ratio);
                    shape = blk.out_shape();
                    ls.push(Layer::Rep(blk));
                }
            }
            if stage < 4 {
                let d = PatchEmbed::new(ctx, &format!("downsample.{stage}"), shape, dims[stage + 1]);
                shape = d.out_shape();
                ls.push(Layer::Down(d));
            }
        }
        // conv_exp: grouped MobileOne last-dim → last-dim·cls_ratio.
        let out_dim = dims[4] * cls_ratio;
        let conv_exp = ConvUnit::new(ctx, "conv_exp", shape, out_dim, 3, 1, 1, dims[4], true, true, true);
        shape = conv_exp.out_shape();
        ls.push(Layer::Conv(conv_exp));

        let tokens = shape.h * shape.w;
        Encoder {
            feature_dim: out_dim,
            tokens,
            out: ctx.gpu.storage((tokens * out_dim) as u64),
            k_nchw_nlc: kidx("nchw_nlc"),
            layers: ls,
        }
    }

    pub fn tokens(&self) -> u32 {
        self.tokens
    }
    pub fn feature_dim(&self) -> u32 {
        self.feature_dim
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        self.layers.iter().flat_map(|l| l.param_list()).collect()
    }

    /// Encode a `[1, 3, input, input]` image into `[tokens, feature_dim]` (NLC),
    /// the sequence the mlp2x_gelu projector consumes.
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, img: &DeviceBuffer) -> Vec<f32> {
        let mut cur: &DeviceBuffer = img;
        for l in &self.layers {
            cur = l.forward(ctx, ps, cur);
        }
        // feature_select: NCHW [1, dim, H, W] -> NLC [tokens, dim].
        let total = self.tokens * self.feature_dim;
        ctx.gpu.submit(&[], &[ctx.step(self.k_nchw_nlc, &[cur, &self.out], &[total, self.feature_dim, self.tokens], total)]);
        ctx.gpu.read(&self.out, total as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;
    use gpu_core::Gpu;
    use paramstore::ParamStore;
    use std::collections::HashMap;
    use vision::Conv;

    /// Random init for a param list, keeping BN running stats sane (var=1, mean=0)
    /// so eval-mode BN never hits sqrt of a negative.
    fn rand_init(plist: &[(String, usize)], rng: &mut Rng) -> HashMap<String, Vec<f32>> {
        plist
            .iter()
            .map(|(n, sz)| {
                let v = if n.ends_with("running_var") {
                    vec![1.0; *sz]
                } else if n.contains("running_mean") {
                    vec![0.0; *sz]
                } else {
                    (0..*sz).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
                };
                (n.clone(), v)
            })
            .collect()
    }

    #[test]
    fn encoder_assembles_and_produces_tokens() {
        // Tiny 5-stage tower (1 block/stage), dims small but attention dims %32==0.
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let enc = Encoder::new(&ctx, [1, 1, 1, 1, 1], [8, 16, 16, 32, 32], 2, 2, 128);
        assert_eq!(enc.tokens(), 4, "128/64 = 2 → 2×2 = 4 tokens");
        assert_eq!(enc.feature_dim(), 64, "32 × cls_ratio 2");
        let plist = enc.param_list();
        assert!(plist.iter().any(|(n, _)| n == "stem.0.conv.weight"));
        assert!(plist.iter().any(|(n, _)| n.starts_with("network.3.")), "attention stage present");
        assert!(plist.iter().any(|(n, _)| n == "conv_exp.conv.weight"));

        let mut rng = Rng::new(7);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let img: Vec<f32> = (0..(3 * 128 * 128) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = enc.forward(&ctx, &ps, &gpu.storage_init("img", &img));
        assert_eq!(out.len(), (4 * 64) as usize, "[tokens, feature_dim]");
        assert!(out.iter().all(|v| v.is_finite()), "encoder output finite");
    }

    #[test]
    fn attention_block_runs_preserves_shape() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 64, 4, 4); // ch 64 → 2 heads × 32; 16 tokens
        let blk = AttentionBlock::new(&ctx, "a", in_shape, 64, 4);
        let plist = blk.param_list();
        for want in ["a.norm.weight", "a.token_mixer.qkv.weight", "a.token_mixer.proj.bias", "a.layer_scale_1_sb", "a.convffn.fc2.conv.weight"] {
            assert!(plist.iter().any(|(n, _)| n == want), "missing {want}");
        }
        let mut rng = Rng::new(6);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let x: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = gpu.read(blk.forward(&ctx, &ps, &gpu.storage_init("x", &x)), in_shape.numel() as usize);
        assert_eq!(out.len(), (64 * 4 * 4) as usize, "AttentionBlock preserves C,H,W");
        assert!(out.iter().all(|v| v.is_finite()), "attention block output finite");
    }

    #[test]
    fn patch_embed_downsamples_and_doubles_channels() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 96, 16, 16);
        let pe = PatchEmbed::new(&ctx, "pe", in_shape, 192); // /2, 96→192
        assert_eq!(pe.out_shape(), Shape::new(1, 192, 8, 8));
        let plist = pe.param_list();
        let mut rng = Rng::new(4);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let x: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = gpu.read(pe.forward(&ctx, &ps, &gpu.storage_init("x", &x)), pe.out_shape().numel() as usize);
        assert_eq!(out.len(), (192 * 8 * 8) as usize);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn repcpe_is_a_depthwise_unit() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 32, 8, 8);
        let pe = repcpe(&ctx, "cpe", in_shape, 32);
        assert_eq!(pe.out_shape(), in_shape); // dw 7×7 pad 3 preserves shape
        let plist = pe.param_list();
        let mut rng = Rng::new(5);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let x: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = gpu.read(pe.forward(&ctx, &ps, &gpu.storage_init("x", &x)), in_shape.numel() as usize);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn repmixer_block_runs_preserves_shape() {
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 16, 8, 8);
        let blk = RepMixerBlock::new(&ctx, "b", in_shape, 16, 4);
        let plist = blk.param_list();
        for want in ["b.token_mixer.conv.weight", "b.convffn.dw.conv.weight", "b.convffn.fc1.conv.weight", "b.convffn.fc2.conv.weight", "b.layer_scale_sb"] {
            assert!(plist.iter().any(|(n, _)| n == want), "missing param {want}");
        }
        let mut rng = Rng::new(3);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let x: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let xb = gpu.storage_init("x", &x);
        let out = blk.forward(&ctx, &ps, &xb);
        let outv = gpu.read(out, blk.out_shape().numel() as usize);
        assert_eq!(outv.len(), (16 * 8 * 8) as usize, "RepMixerBlock preserves C,H,W");
        assert!(outv.iter().all(|v| v.is_finite()), "block output finite");
    }

    #[test]
    fn conv_unit_depthwise_bias_gelu_runs() {
        // RepMixer-style unit: depthwise 3×3, no BN, +bias, +erf-GELU.
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 16, 8, 8);
        let unit = ConvUnit::new(&ctx, "u", in_shape, 16, 3, 1, 1, 16, false, true, true);
        let plist = unit.param_list();
        assert!(plist.iter().any(|(n, _)| n == "u.conv.weight"), "depthwise conv weight present");
        assert!(plist.iter().any(|(n, _)| n == "u.bias"), "per-channel bias present");

        let mut rng = Rng::new(2);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let x: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let xb = gpu.storage_init("x", &x);
        let out = unit.forward(&ctx, &ps, &xb);
        let outv = gpu.read(out, unit.out_shape().numel() as usize);
        assert_eq!(outv.len(), (16 * 8 * 8) as usize, "depthwise 3×3 pad 1 preserves 8×8");
        assert!(outv.iter().all(|v| v.is_finite()), "output finite");
        // erf-GELU has a global minimum ≈ -0.17, so post-activation values can't go
        // much below that — a sanity check that the GELU actually ran.
        assert!(outv.iter().cloned().fold(f32::INFINITY, f32::min) > -0.2, "GELU floor");
    }

    #[test]
    fn conv_runs_in_fastvlm_crate() {
        // Prove the conv framework is wired: one strided conv 3→16 over 8×8 → 4×4.
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 3, 8, 8);
        let conv = Conv::new(&ctx, "c0", in_shape, 16, 3, 2, 1, true);

        let plist = conv.param_list();
        let mut rng = Rng::new(1);
        let init: HashMap<String, Vec<f32>> = plist
            .iter()
            .map(|(n, sz)| {
                let v = if n.ends_with("running_var") {
                    vec![1.0; *sz]
                } else if n.contains("running_mean") {
                    vec![0.0; *sz]
                } else {
                    (0..*sz).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
                };
                (n.clone(), v)
            })
            .collect();
        let ps = ParamStore::new(&gpu, plist, &init);

        let img: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let img_b = gpu.storage_init("img", &img);
        conv.forward(&ctx, &ps, &img_b);
        let out = gpu.read(conv.out(), conv.out_shape.numel() as usize);
        assert_eq!(out.len(), (16 * 4 * 4) as usize, "3→16 stride-2 over 8×8 → 16×4×4");
        assert!(out.iter().all(|v| v.is_finite()), "conv output must be finite");
    }
}
