// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CausalMaskedDiffWithDiT` (CosyVoice 3) forward: speech tokens + a speaker
//! x-vector + a prompt mel -> a target mel spectrogram, via a bare
//! `PreLookaheadLayer` + `repeat_interleave` (NO encoder at all - confirmed
//! absent, not merely unused, from the real `CausalMaskedDiffWithDiT.
//! __init__`/`.inference`) feeding a 22-layer adaLN-zero `DiT` estimator
//! driven by the SAME 10-step classifier-free-guided cosine-scheduled Euler
//! ODE solver `crate::flow` (CosyVoice 2) already implements for its own UNet
//! estimator (`ConditionalCFM`/`CausalConditionalCFM` are literally the same
//! shared reference classes regardless of which estimator module sits inside
//! them).
//!
//! Host CPU throughout, the same convention every other module in this crate
//! uses. Reuses `crate::flow`'s small shared primitives rather than
//! duplicating them: `conv1d`/`conv1d_grouped`/`causal_conv1d`/`mish`/
//! `leaky_relu`/`transpose_{tc_to_cl,cl_to_tc}`/`linear_rows_biased` (conv/
//! activation/layout plumbing), `conds_from_prompt_feat` (the CFM `cond`
//! tensor assembly, byte-for-byte identical to CosyVoice 2's), `cosine_t_span`
//! (the CFM schedule), and `rand_noise` (the fixed seed-0 noise buffer - same
//! `[1, 80, 15000]` shape, same seed, same torch RNG port).
//!
//! ## Condition assembly (`CausalMaskedDiffWithDiT.inference`, no encoder)
//! ```text
//! token   = input_embedding(clamp(concat([prompt_token, token]), min=0))   // [n, 80]
//! h       = pre_lookahead_layer(token)                                     // [n, 80], residual conv, see below
//! mu      = h.repeat_interleave(token_mel_ratio=2, dim=1)                  // [2n, 80] - literally duplicate each frame
//! conds   = conds_from_prompt_feat(prompt_feat, mel_len1, mel_len2)        // identical to CosyVoice 2
//! ```
//! `PreLookaheadLayer(in=80, channels=1024, pre_lookahead_len=3)`:
//! `Conv1d(80,1024,k=4)` over a RIGHT-padded-by-3 input (free via `crate::
//! flow::conv1d`'s `pad=0, lo=n` trick) -> `leaky_relu` -> `Conv1d(1024,80,k=3)`
//! over a LEFT-padded-by-2 (causal) input -> residual add with the ORIGINAL
//! (pre-lookahead) token embedding - the exact same two-conv-plus-residual
//! shape `crate::flow::encoder_forward`'s own `PreLookaheadLayer` step already
//! implements for CosyVoice 2 (verified byte-for-byte identical against the
//! reference's `PreLookaheadLayer.forward`, shared by both generations'
//! encoders), just with input/hidden/output widths `80/1024/80` here instead
//! of `512/512/512`. No `encoder_proj` either: `h` is already `output_size`
//! (80) wide, so it becomes `mu` (the DiT's "text_embed") as-is - confirmed by
//! the real checkpoint carrying no `encoder_proj.*` tensor at all.
//!
//! ## The DiT estimator (`cosyvoice/flow/DiT/{dit,modules}.py`, ported
//! algorithm-for-algorithm, cross-checked against the golden's own internal
//! `InputEmbedding`/`TimestepEmbedding` taps)
//! `TimestepEmbedding`: `SinusPositionEmbedding(256, scale=1000)` - the exact
//! same `[sin ‖ cos]`/`half_dim-1`-denominator formula `crate::flow`'s own
//! `time_embed` already reproduces via `model::hostmath::timestep_embedding`
//! (`flip_sin_to_cos=false, downscale_freq_shift=1.0`) - -> `Linear(256,1024)`
//! -> `SiLU` -> `Linear(1024,1024)`.
//!
//! `InputEmbedding`: `proj = Linear(320, 1024)` over
//! `cat([x, cond, text_embed(=mu), spks_broadcast], dim=-1)` (this exact
//! order, verified against the reference's own `to_cat = [x, cond,
//! text_embed]; to_cat.append(spks)` and independently against the golden's
//! four separately-captured `dit_input_embed_in_{x,cond,text_embed,spks}`
//! tensors) -> `x = conv_pos_embed(x) + x`, where `conv_pos_embed` is a
//! `CausalConvPositionEmbedding(dim=1024, kernel=31, groups=16)`: two
//! `[left-pad 30 -> grouped Conv1d(1024,1024,k=31,groups=16) -> Mish]` stages
//! in sequence (NOT the plain symmetric-pad `ConvPositionEmbedding` - CosyVoice
//! 3's flow decoder is fully causal).
//!
//! `RotaryEmbedding(dim_head=64)`: `x_transformers`'s own convention - 1-D
//! interleaved (adjacent-pair) rotation, NOT the "rotate half the vector"
//! GPT-NeoX/Llama convention: `inv_freq[i] = theta^(-2i/64)` for `i` in
//! `0..32`; pair `(2i, 2i+1)` of each 64-wide head vector rotates by
//! `pos * inv_freq[i]` (verified by reading `x_transformers.x_transformers.
//! RotaryEmbedding.forward`/`apply_rotary_pos_emb`/`rotate_half` directly in
//! the exact venv the golden dumper used, not inferred from the more common
//! half-split convention other RoPE ports in this repo use).
//!
//! 22x `DiTBlock`: `AdaLayerNormZero(1024)` (`SiLU -> Linear(1024,6144)`,
//! chunked into `shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp`;
//! `LayerNorm(1024, elementwise_affine=False, eps=1e-6)` - no learnable
//! affine, confirmed absent from the real checkpoint - modulated
//! `(1+scale)*norm(x)+shift`) -> self-attention (`to_q/to_k/to_v/to_out.0`
//! all carry a bias; RoPE applied to q/k; no attention mask - the
//! non-streaming, full-bidirectional path only, matching CosyVoice 2 flow's
//! own already-recorded streaming gap) -> `x += gate_msa * attn_out` ->
//! the SAME `AdaLayerNormZero` chunking's `_mlp` triple modulates a second
//! bare `LayerNorm` -> `FeedForward(1024, mult=2, approximate="tanh")`
//! (`Linear(1024,2048) -> GELU(tanh) -> Linear(2048,1024)`) -> `x += gate_mlp
//! * ff_out`.
//!
//! `AdaLayerNormZero_Final(1024)`: `SiLU -> Linear(1024,2048)`, chunked into
//! `scale, shift`, modulating one more bare `LayerNorm`. `proj_out =
//! Linear(1024, 80)`, transposed back to channel-major - the estimator's
//! velocity-field output, fed into the SAME Euler-step/CFG driving loop
//! `crate::flow::solve_euler` uses (reimplemented here as [`solve_euler`]
//! rather than made generic over the estimator type, to keep each module a
//! direct, independently-readable transcript of its own reference function -
//! the loop BODY is intentionally byte-for-byte the same shape as `crate::
//! flow::solve_euler`'s).

use crate::cv3_flow_config::{Cv3FlowConfig, DitConfig};
use crate::cv3_flow_import::{Cv3FlowWeights, DitBlockW, DitW};
use crate::flow::{causal_conv1d, conds_from_prompt_feat, conv1d, conv1d_grouped, cosine_t_span, leaky_relu, linear_rows_biased, mish, transpose_cl_to_tc, transpose_tc_to_cl};
use model::hostmath::{l2_normalize, layernorm_rows, matvec, silu, timestep_embedding};

/// `torch.manual_seed(0); torch.randn([1, 80, 15000])` - the SAME fixed CFM
/// noise buffer `crate::flow::rand_noise` already reproduces (same shape,
/// same seed, same shared `ConditionalCFM`/`CausalConditionalCFM` mechanism -
/// see this module's doc). Re-exported under this crate's name rather than
/// requiring every CV3 caller to reach into `crate::flow` for a CV3-owned
/// concept.
pub fn rand_noise() -> Vec<f32> {
    crate::flow::rand_noise()
}

// ---------------------------------------------------------------------------
// PreLookaheadLayer + repeat_interleave (no encoder)
// ---------------------------------------------------------------------------

/// `PreLookaheadLayer(in=80, channels=1024, pre_lookahead_len=3).forward`
/// (`finalize=True`, empty `context`): `token_tc` is `[n, in]`; returns
/// `[n, in]` (residual, same width in/out - see the module doc).
fn pre_lookahead_layer(cfg: &Cv3FlowConfig, w: &crate::cv3_flow_import::Cv3FlowWeights, token_tc: &[f32], n: usize) -> Vec<f32> {
    let d = cfg.input_size as usize;
    let ch = cfg.pre_lookahead_channels as usize;
    let la = cfg.pre_lookahead_len as usize;

    let token_cl = transpose_tc_to_cl(token_tc, n, d);
    let mut h = conv1d(&token_cl, &w.pre_lookahead_conv1, d, ch, n, n, la + 1, 0);
    for v in h.iter_mut() {
        *v = leaky_relu(*v);
    }
    let h2 = causal_conv1d(&h, &w.pre_lookahead_conv2, ch, d, n, 3);
    let mut out = token_tc.to_vec();
    let h2_tc = transpose_cl_to_tc(&h2, d, n);
    for i in 0..n * d {
        out[i] += h2_tc[i];
    }
    out
}

/// `h.repeat_interleave(token_mel_ratio, dim=1)`: `[n, d]` time-major ->
/// `[token_mel_ratio*n, d]`, each input row duplicated `token_mel_ratio`
/// times in place (25 Hz -> 50 Hz, "the simple interpolation operation" the
/// CosyVoice 3 paper describes replacing the conformer encoder with).
fn repeat_interleave_tc(x_tc: &[f32], n: usize, d: usize, ratio: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * ratio * d];
    for i in 0..n {
        for r in 0..ratio {
            out[(i * ratio + r) * d..(i * ratio + r + 1) * d].copy_from_slice(&x_tc[i * d..(i + 1) * d]);
        }
    }
    out
}

/// `spk_embed_affine_layer(F.normalize(xvec, dim=1))`: `[output_size]`.
pub fn speaker_embedding(w: &Cv3FlowWeights, cfg: &Cv3FlowConfig, xvec: &[f32]) -> Vec<f32> {
    let n = l2_normalize(xvec);
    let mut e = matvec(&w.spk_affine.w, &n, cfg.output_size as usize, cfg.spk_embed_dim as usize);
    for (v, b) in e.iter_mut().zip(&w.spk_affine.b) {
        *v += b;
    }
    e
}

/// `token = input_embedding(clamp(concat([prompt_token, token]), min=0))`,
/// `mu = pre_lookahead_layer(token).repeat_interleave(token_mel_ratio)`, and
/// `conds` via [`crate::flow::conds_from_prompt_feat`] - the whole
/// (encoder-free) condition assembly `CausalMaskedDiffWithDiT.inference` does
/// before the CFM Euler loop. Returns `(mu_cl, conds_cl, mel_len1, mel_len2)`,
/// both channel-major `[output_size, mel_len1+mel_len2]`.
pub fn assemble_conditions(
    w: &Cv3FlowWeights,
    cfg: &Cv3FlowConfig,
    prompt_tokens: &[u32],
    gen_tokens: &[u32],
    prompt_feat_tc: &[f32],
    mel_len1: usize,
) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let d = cfg.input_size as usize;
    let mel = cfg.output_size as usize;
    let n = prompt_tokens.len() + gen_tokens.len();

    let mut token_tc = vec![0.0f32; n * d];
    for (i, &id) in prompt_tokens.iter().chain(gen_tokens.iter()).enumerate() {
        let row = id as usize * d;
        token_tc[i * d..(i + 1) * d].copy_from_slice(&w.input_embedding[row..row + d]);
    }

    let h = pre_lookahead_layer(cfg, w, &token_tc, n);
    let ratio = cfg.token_mel_ratio as usize;
    let mu_tc = repeat_interleave_tc(&h, n, d, ratio);
    let n2 = ratio * n;
    let mu_cl = transpose_tc_to_cl(&mu_tc, n2, mel);

    let mel_len2 = n2 - mel_len1;
    let conds_cl = conds_from_prompt_feat(prompt_feat_tc, mel, mel_len1, mel_len2);

    (mu_cl, conds_cl, mel_len1, mel_len2)
}

// ---------------------------------------------------------------------------
// DiT estimator
// ---------------------------------------------------------------------------

/// `SinusPositionEmbedding(256,scale=1000) -> Linear(256,1024) -> SiLU ->
/// Linear(1024,1024)`. Matches `model::hostmath::timestep_embedding`'s
/// `[sin ‖ cos]`/`downscale_freq_shift` convention exactly - see this module's
/// doc.
pub fn time_embed(t: f32, cfg: &DitConfig, w: &crate::cv3_flow_import::TimeEmbedW) -> Vec<f32> {
    let freq_dim = cfg.freq_embed_dim as usize;
    let dim = cfg.dim as usize;
    let emb = timestep_embedding(t * 1000.0, freq_dim, false, 1.0, 10000.0);
    let mut h = matvec(&w.mlp1.w, &emb, dim, freq_dim);
    for (v, b) in h.iter_mut().zip(&w.mlp1.b) {
        *v = silu(*v + b);
    }
    let mut out = matvec(&w.mlp2.w, &h, dim, dim);
    for (v, b) in out.iter_mut().zip(&w.mlp2.b) {
        *v += b;
    }
    out
}

/// `InputEmbedding.forward(x, cond, text_embed, spks)`: all of `x`/`cond`/
/// `text_embed` are `[t, mel_dim]` (mu_dim==mel_dim==80 here); `spks` is
/// `[spk_dim]`. Returns `[t, dim]`.
pub fn input_embed(x_tc: &[f32], cond_tc: &[f32], text_embed_tc: &[f32], spks: &[f32], cfg: &DitConfig, w: &crate::cv3_flow_import::InputEmbedW, t: usize) -> Vec<f32> {
    let mel = cfg.mel_dim as usize;
    let mu = cfg.mu_dim as usize;
    let spk = cfg.spk_dim as usize;
    let dim = cfg.dim as usize;
    let in_w = cfg.input_embed_in() as usize;

    let mut cat = vec![0.0f32; t * in_w];
    for ti in 0..t {
        let base = ti * in_w;
        cat[base..base + mel].copy_from_slice(&x_tc[ti * mel..(ti + 1) * mel]);
        cat[base + mel..base + 2 * mel].copy_from_slice(&cond_tc[ti * mel..(ti + 1) * mel]);
        cat[base + 2 * mel..base + 2 * mel + mu].copy_from_slice(&text_embed_tc[ti * mu..(ti + 1) * mu]);
        cat[base + 2 * mel + mu..base + 2 * mel + mu + spk].copy_from_slice(spks);
    }
    let x1 = linear_rows_biased(&cat, &w.proj, t, in_w, dim);

    let groups = cfg.conv_pos_groups as usize;
    let k = cfg.conv_pos_kernel as usize;
    let x1_cl = transpose_tc_to_cl(&x1, t, dim);
    let mut h = conv1d_grouped(&x1_cl, &w.conv1, dim, dim, t, t, k, k - 1, groups);
    for v in h.iter_mut() {
        *v = mish(*v);
    }
    let mut h2 = conv1d_grouped(&h, &w.conv2, dim, dim, t, t, k, k - 1, groups);
    for v in h2.iter_mut() {
        *v = mish(*v);
    }
    let h2_tc = transpose_cl_to_tc(&h2, dim, t);

    let mut out = x1;
    for i in 0..t * dim {
        out[i] += h2_tc[i];
    }
    out
}

/// 1-D interleaved (adjacent-pair) RoPE tables: `cos_tab`/`sin_tab`, each
/// `[t, dim_head/2]` - `tab[pos][i] = pos * theta^(-2i/dim_head)`. See the
/// module doc's RoPE-convention note.
fn rope_tables(t: usize, dim_head: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim_head / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| theta.powf(-((2 * i) as f32) / dim_head as f32)).collect();
    let mut cos_tab = vec![0.0f32; t * half];
    let mut sin_tab = vec![0.0f32; t * half];
    for pos in 0..t {
        for i in 0..half {
            let angle = pos as f32 * inv_freq[i];
            cos_tab[pos * half + i] = angle.cos();
            sin_tab[pos * half + i] = angle.sin();
        }
    }
    (cos_tab, sin_tab)
}

/// Apply interleaved RoPE to one `[t, heads*dim_head]` buffer in place.
///
/// **Real, empirically-verified quirk of the reference, not a guess**:
/// `x_transformers`'s `AttnProcessor.__call__` calls `apply_rotary_pos_emb`
/// on `query`/`key` while they are STILL `[b, n, heads*dim_head]` - BEFORE
/// the `.view(b, n, heads, dim_head).transpose(1, 2)` split into per-head
/// tensors happens (verified by reading `cosyvoice/flow/DiT/modules.py`'s
/// `AttnProcessor.__call__` directly: the `view`/`transpose` calls are
/// several lines AFTER the `apply_rotary_pos_emb` calls). `apply_rotary_pos_emb`
/// then slices `t[..., :rot_dim]` with `rot_dim = freqs.shape[-1] =
/// dim_head` (64) - so on a 1024-wide (`heads=16 * dim_head=64`) query/key
/// row, ONLY THE FIRST 64 CHANNELS (head 0's slice, once the row is later
/// split into heads) are ever rotated; the remaining 960 channels (heads
/// 1-15) pass through completely untouched
/// (`t_unrotated = t[..., rot_dim:]`, concatenated back unchanged). This
/// looks like an accidental "partial rotary embeddings" (the GPT-J-style
/// feature `x_transformers` supports on purpose for a DIFFERENT use case)
/// rather than an intentional per-head design, but reproducing the
/// reference bit-for-bit means replicating exactly this, not "fixing" it -
/// confirmed empirically: applying RoPE per-head (rotating all 16 heads
/// identically) reproduced the InputEmbedding/TimestepEmbedding sub-stages
/// correctly but made the full Euler loop diverge (cosine ~0.993 against
/// `flow_real_euler_steps.f32`); switching to this partial-rotation
/// behavior is what the "verify against x_transformers's exact convention"
/// instruction anticipated.
fn apply_rope(x: &mut [f32], t: usize, heads: usize, dim_head: usize, cos_tab: &[f32], sin_tab: &[f32]) {
    let half = dim_head / 2;
    let inner = heads * dim_head;
    for ti in 0..t {
        {
            let base = ti * inner; // rotate only the first `dim_head` channels of the row
            for i in 0..half {
                let c = cos_tab[ti * half + i];
                let s = sin_tab[ti * half + i];
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * c - b * s;
                x[base + 2 * i + 1] = b * c + a * s;
            }
        }
    }
}

fn layernorm_no_affine(x: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    let ones = vec![1.0f32; d];
    let zeros = vec![0.0f32; d];
    layernorm_rows(x, &ones, &zeros, rows, d, eps)
}

/// `GELU(approximate="tanh")`: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// One `DiTBlock.forward(x, t, mask=None, rope)`: `x_tc` is `[t, dim]`;
/// `t_hidden` is the (already-computed, shared across all blocks) time
/// embedding `[dim]`.
fn dit_block(x_tc: &[f32], t_hidden: &[f32], w: &DitBlockW, cfg: &DitConfig, t: usize, cos_tab: &[f32], sin_tab: &[f32]) -> Vec<f32> {
    let dim = cfg.dim as usize;
    let heads = cfg.heads as usize;
    let hd = cfg.dim_head as usize;
    let eps = cfg.norm_eps;

    // AdaLayerNormZero: emb = linear(silu(t_hidden)) -> chunk 6.
    let t_silu: Vec<f32> = t_hidden.iter().map(|&v| silu(v)).collect();
    let mut emb = matvec(&w.attn_norm_linear.w, &t_silu, 6 * dim, dim);
    for (v, b) in emb.iter_mut().zip(&w.attn_norm_linear.b) {
        *v += b;
    }
    let (shift_msa, rest) = emb.split_at(dim);
    let (scale_msa, rest) = rest.split_at(dim);
    let (gate_msa, rest) = rest.split_at(dim);
    let (shift_mlp, rest) = rest.split_at(dim);
    let (scale_mlp, gate_mlp) = rest.split_at(dim);

    let norm = layernorm_no_affine(x_tc, t, dim, eps);
    let mut modulated = vec![0.0f32; t * dim];
    for ti in 0..t {
        for j in 0..dim {
            modulated[ti * dim + j] = norm[ti * dim + j] * (1.0 + scale_msa[j]) + shift_msa[j];
        }
    }

    let mut q = linear_rows_biased(&modulated, &w.wq, t, dim, dim);
    let mut k = linear_rows_biased(&modulated, &w.wk, t, dim, dim);
    let v = linear_rows_biased(&modulated, &w.wv, t, dim, dim);
    apply_rope(&mut q, t, heads, hd, cos_tab, sin_tab);
    apply_rope(&mut k, t, heads, hd, cos_tab, sin_tab);

    let mut ctx = vec![0.0f32; t * dim];
    let scale = (hd as f32).sqrt();
    for h in 0..heads {
        for ti in 0..t {
            let mut scores = vec![0.0f32; t];
            for tj in 0..t {
                let mut acc = 0.0f32;
                for j in 0..hd {
                    acc += q[ti * dim + h * hd + j] * k[tj * dim + h * hd + j];
                }
                scores[tj] = acc / scale;
            }
            model::hostmath::softmax(&mut scores);
            for j in 0..hd {
                let mut acc = 0.0f32;
                for tj in 0..t {
                    acc += scores[tj] * v[tj * dim + h * hd + j];
                }
                ctx[ti * dim + h * hd + j] = acc;
            }
        }
    }
    let attn_out = linear_rows_biased(&ctx, &w.wo, t, dim, dim);

    let mut x1 = x_tc.to_vec();
    for ti in 0..t {
        for j in 0..dim {
            x1[ti * dim + j] += gate_msa[j] * attn_out[ti * dim + j];
        }
    }

    let ff_norm = layernorm_no_affine(&x1, t, dim, eps);
    let mut ff_mod = vec![0.0f32; t * dim];
    for ti in 0..t {
        for j in 0..dim {
            ff_mod[ti * dim + j] = ff_norm[ti * dim + j] * (1.0 + scale_mlp[j]) + shift_mlp[j];
        }
    }
    let ff_hidden = cfg.ff_hidden() as usize;
    let mut ff_h = linear_rows_biased(&ff_mod, &w.ff1, t, dim, ff_hidden);
    for v in ff_h.iter_mut() {
        *v = gelu_tanh(*v);
    }
    let ff_out = linear_rows_biased(&ff_h, &w.ff2, t, ff_hidden, dim);

    let mut x2 = x1;
    for ti in 0..t {
        for j in 0..dim {
            x2[ti * dim + j] += gate_mlp[j] * ff_out[ti * dim + j];
        }
    }
    x2
}

/// The full `DiT.forward(x, mask, mu, t, spks, cond, streaming=False)`
/// (non-streaming only): `x`/`mu`/`cond` are `[mel_dim, t]` channel-major;
/// `spks` is `[spk_dim]`. Returns the predicted velocity, `[mel_dim, t]`.
pub fn dit_forward(cfg: &DitConfig, w: &DitW, x_cl: &[f32], mu_cl: &[f32], t_scalar: f32, spks: &[f32], cond_cl: &[f32], t: usize) -> Vec<f32> {
    let mel = cfg.mel_dim as usize;
    let dim = cfg.dim as usize;

    let x_tc = transpose_cl_to_tc(x_cl, mel, t);
    let mu_tc = transpose_cl_to_tc(mu_cl, mel, t);
    let cond_tc = transpose_cl_to_tc(cond_cl, mel, t);

    let t_hidden = time_embed(t_scalar, cfg, &w.time_embed);
    let mut x = input_embed(&x_tc, &cond_tc, &mu_tc, spks, cfg, &w.input_embed, t);

    let (cos_tab, sin_tab) = rope_tables(t, cfg.dim_head as usize, cfg.rope_theta);
    for block in &w.blocks {
        x = dit_block(&x, &t_hidden, block, cfg, t, &cos_tab, &sin_tab);
    }

    // AdaLayerNormZero_Final: emb = linear(silu(t_hidden)) -> chunk 2.
    let t_silu: Vec<f32> = t_hidden.iter().map(|&v| silu(v)).collect();
    let mut emb = matvec(&w.norm_out.linear.w, &t_silu, 2 * dim, dim);
    for (v, b) in emb.iter_mut().zip(&w.norm_out.linear.b) {
        *v += b;
    }
    let (scale, shift) = emb.split_at(dim);
    let norm = layernorm_no_affine(&x, t, dim, cfg.norm_eps);
    let mut xn = vec![0.0f32; t * dim];
    for ti in 0..t {
        for j in 0..dim {
            xn[ti * dim + j] = norm[ti * dim + j] * (1.0 + scale[j]) + shift[j];
        }
    }

    let out_tc = linear_rows_biased(&xn, &w.proj_out, t, dim, mel);
    transpose_tc_to_cl(&out_tc, t, mel)
}

// ---------------------------------------------------------------------------
// CFM Euler loop
// ---------------------------------------------------------------------------

/// `CausalConditionalCFM.solve_euler`, driving the DiT estimator instead of
/// CosyVoice 2's UNet - see this module's doc for why the loop is a
/// deliberate, readable duplicate of `crate::flow::solve_euler`'s rather than
/// a shared generic. Returns every post-step latent (`[mel_dim, t]` each).
pub fn solve_euler(cfg: &Cv3FlowConfig, w: &DitW, x0: &[f32], mu: &[f32], spks: &[f32], cond: &[f32], t: usize, n_timesteps: usize) -> Vec<Vec<f32>> {
    let mel = cfg.dit.mel_dim as usize;
    let t_span = cosine_t_span(n_timesteps);
    let zeros_mu = vec![0.0f32; mel * t];
    let zeros_spks = vec![0.0f32; mel];
    let zeros_cond = vec![0.0f32; mel * t];

    let mut x = x0.to_vec();
    let mut steps = Vec::with_capacity(n_timesteps);
    let mut cur_t = t_span[0];
    let mut dt = t_span[1] - t_span[0];
    let rate = cfg.inference_cfg_rate;
    for step in 1..=n_timesteps {
        let cond_out = dit_forward(&cfg.dit, w, &x, mu, cur_t, spks, cond, t);
        let uncond_out = dit_forward(&cfg.dit, w, &x, &zeros_mu, cur_t, &zeros_spks, &zeros_cond, t);
        for i in 0..mel * t {
            let dphi = (1.0 + rate) * cond_out[i] - rate * uncond_out[i];
            x[i] += dt * dphi;
        }
        cur_t += dt;
        steps.push(x.clone());
        if step < n_timesteps {
            dt = t_span[step + 1] - cur_t;
        }
    }
    steps
}

/// Every intermediate rung of one [`forward`] call.
pub struct ForwardOutput {
    /// `[output_size, mel_len2]` - `feat[:, :, mel_len1:]`.
    pub mel: Vec<f32>,
    pub mu: Vec<f32>,
    pub conds: Vec<f32>,
    pub embedding: Vec<f32>,
    pub euler_steps: Vec<Vec<f32>>,
}

/// The whole `CausalMaskedDiffWithDiT.inference()` forward: condition
/// assembly through the CFM Euler loop, sliced to the generated span.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    w: &Cv3FlowWeights,
    cfg: &Cv3FlowConfig,
    prompt_tokens: &[u32],
    gen_tokens: &[u32],
    xvec: &[f32],
    prompt_feat_tc: &[f32],
    mel_len1: usize,
    noise: &[f32],
    n_timesteps: usize,
) -> ForwardOutput {
    let mel = cfg.output_size as usize;
    let embedding = speaker_embedding(w, cfg, xvec);
    let (mu_cl, conds_cl, mel_len1, mel_len2) = assemble_conditions(w, cfg, prompt_tokens, gen_tokens, prompt_feat_tc, mel_len1);
    let t = mel_len1 + mel_len2;
    assert!(noise.len() >= mel * t, "cv3_flow::forward: noise buffer shorter than {}", mel * t);

    let mut x0 = vec![0.0f32; mel * t];
    let noise_len = noise.len() / mel;
    for c in 0..mel {
        x0[c * t..(c + 1) * t].copy_from_slice(&noise[c * noise_len..c * noise_len + t]);
    }

    let steps = solve_euler(cfg, &w.dit, &x0, &mu_cl, &embedding, &conds_cl, t, n_timesteps);
    let full = steps.last().expect("n_timesteps must be > 0").clone();
    let mut mel_out = vec![0.0f32; mel * mel_len2];
    for c in 0..mel {
        mel_out[c * mel_len2..(c + 1) * mel_len2].copy_from_slice(&full[c * t + mel_len1..c * t + t]);
    }
    ForwardOutput { mel: mel_out, mu: mu_cl, conds: conds_cl, embedding, euler_steps: steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_interleave_duplicates_each_frame() {
        let x = [1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let y = repeat_interleave_tc(&x, 2, 2, 2);
        assert_eq!(y, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_tables_have_the_right_shape_and_zero_position_is_identity() {
        let (cos_tab, sin_tab) = rope_tables(5, 8, 10000.0);
        assert_eq!(cos_tab.len(), 5 * 4);
        assert_eq!(sin_tab.len(), 5 * 4);
        for i in 0..4 {
            assert!((cos_tab[i] - 1.0).abs() < 1e-6, "position 0 must have angle 0 -> cos=1");
            assert!(sin_tab[i].abs() < 1e-6, "position 0 must have angle 0 -> sin=0");
        }
    }

    #[test]
    fn apply_rope_preserves_vector_norm() {
        let t = 4;
        let heads = 2;
        let hd = 8;
        let (cos_tab, sin_tab) = rope_tables(t, hd, 10000.0);
        let mut x: Vec<f32> = (0..t * heads * hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let norm_before: f32 = x.iter().map(|v| v * v).sum();
        apply_rope(&mut x, t, heads, hd, &cos_tab, &sin_tab);
        let norm_after: f32 = x.iter().map(|v| v * v).sum();
        assert!((norm_before - norm_after).abs() < 1e-3, "a rotation must preserve the vector norm: {norm_before} vs {norm_after}");
    }

    #[test]
    fn gelu_tanh_matches_a_reference_value() {
        // GELU(1.0) with the tanh approximation is documented as ~0.8412 in
        // the F5-TTS/BERT literature this formula comes from.
        let got = gelu_tanh(1.0);
        assert!((got - 0.8412).abs() < 1e-3, "gelu_tanh(1.0) = {got}, want ~0.8412");
        assert!(gelu_tanh(0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_t_span_is_reused_and_matches_ten_steps() {
        let span = cosine_t_span(10);
        assert_eq!(span.len(), 11);
    }
}
