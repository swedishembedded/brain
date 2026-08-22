// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CausalMaskedDiffWithXvec` (CosyVoice 2) forward: speech tokens + a
//! speaker x-vector + a prompt mel -> a target mel spectrogram, via
//! `UpsampleConformerEncoder` (relative-position conformer) feeding a
//! `CausalConditionalDecoder` UNet driven by a 10-step classifier-free-guided
//! Euler ODE solver (conditional flow matching).
//!
//! Host CPU throughout (no device dispatch), the same convention
//! `crate::llm` (M5) already established for this crate's autoregressive LM:
//! the architecture here is two genuinely new primitives (ESPnet
//! Transformer-XL-style relative-position attention, and a 56-transformer +
//! 14-resnet-block UNet with no existing brain precedent for either), and
//! `model::hostmath`'s `matvec`/`linear_rows` already dispatch through the
//! AVX2+FMA, rayon-parallel `backend_cpu::fast_ops::matmul_abt` kernel, so a
//! host implementation is not the slow path it would be as plain scalar
//! loops. Getting this correct first and fast later is the deliberate order -
//! a device (WGSL Step-builder) port of this UNet, if the profile ever
//! demands it, is a follow-up that
//! reuses this file as its parity oracle.
//!
//! ## The UNet never actually changes resolution
//! `CausalConditionalDecoder`'s `channels=[256]` is a length-1 tuple, which
//! makes `is_last` true for the (only) down/up stage - so the "downsample"/
//! "upsample" convs the reference names are both stride-1 causal
//! `Conv1d(256,256,3)`, not real resolution changes (verified by reading
//! `resources/cosyvoice/source/cosyvoice/flow/decoder.py` line-for-line, not
//! assumed from the class name). Every tensor in the estimator therefore stays at
//! the SAME time length `T` throughout, which is what makes a flat
//! `Vec<f32>`-per-buffer host implementation tractable at all.
//!
//! ## CFG as two batch-1 forwards, not one batch-2 forward
//! The reference's `solve_euler` runs the estimator on a batch of 2 (index 0
//! = conditional, index 1 = fully unconditional: `mu=0, spks=0, cond=0`, but
//! the SAME `x`/`mask`/`t`). Every op in this UNet (`LayerNorm`, not
//! `BatchNorm`; self-attention within one sequence; causal convs) is strictly
//! per-batch-item independent, so running the estimator twice at batch=1 -
//! once with the real conditions, once with all-zero conditions - and
//! combining `(1+cfg_rate)*cond - cfg_rate*uncond` afterward is exactly
//! equivalent, and avoids ever materializing a batch dimension.
//!
//! ## Deferred: streaming / chunked attention
//! Every real (non-padded) forward in this port is `mask ≡ 1` (batch=1,
//! `mel_len1+mel_len2` frames exactly, never more) - `flow.inference()` never
//! pads. `add_optional_chunk_mask(..., streaming=False, ...)` then reduces to
//! full bidirectional attention with no windowing, which is the only path
//! this file implements (`streaming=True`'s chunked mask is a documented,
//! NOT-implemented gap).
//!
//! ## The fixed CFM noise buffer
//! `CausalConditionalCFM.__init__` does `set_all_random_seed(0);
//! self.rand_noise = torch.randn([1, 80, 15000])` - a plain attribute, not a
//! registered buffer, so it never appears in `flow.pt` (see
//! `crate::flow_import`'s module doc) and must be reconstructed. Because the
//! buffer is a pure function of the seed, [`torch_rng::randn_seed0`] computes
//! it at runtime with a bit-exact Rust port of torch's CPU RNG (`at::mt19937`
//! seeding/tempering plus the AVX2 `normal_fill_16` Box-Muller kernel, see
//! that module's doc for exactly what was ported and how it was verified)
//! rather than shipping a checked-in data asset - a 4.8 MB precomputed blob
//! cannot be committed at all under this repo's `no-large-or-binary-files`
//! gate (`scripts/gates/check-large-files.sh` bans `.bin`/`.f32`/raw tensor
//! dumps at any size, precisely to keep exactly this kind of regenerable
//! buffer out of source control). Verified bit-exact against the golden's own
//! `rand_noise_full_sha256`
//! (`584a26e6eaad944407a96c5999aaa5cd0a6a359309a841eade68bf805c072322`) and
//! against `flow_real_rand_noise_slice.f32`.

use crate::flow_config::FlowConfig;
use crate::flow_import::{CfmBlockW, ConformerLayerW, EstimatorW, FlowWeights, LinearW, ResnetBlockW, SubsampleW};
use audio::conv::{conv1d_ref, Conv1d};
use model::hostmath::{l2_normalize, layernorm_rows, linear_rows, matvec, softmax, timestep_embedding};

#[path = "torch_rng.rs"]
mod torch_rng;

/// `torch.manual_seed(0); torch.randn([1, 80, 15000])`, channel-major - see
/// the module doc's noise-buffer note.
pub fn rand_noise() -> Vec<f32> {
    torch_rng::randn_seed0(80, 15000)
}

// ---------------------------------------------------------------------------
// small host-math helpers not already in `model::hostmath`
// ---------------------------------------------------------------------------

fn add_bias_rows(x: &mut [f32], b: &[f32], rows: usize, c: usize) {
    for r in 0..rows {
        for j in 0..c {
            x[r * c + j] += b[j];
        }
    }
}

/// `pub(crate)` - shared with `crate::cv3_flow`'s DiT estimator (same
/// `Linear(x) + bias` broadcast every per-row projection in this port needs).
pub(crate) fn linear_rows_biased(x: &[f32], w: &LinearW, rows: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = linear_rows(x, &w.w, rows, inn, out);
    add_bias_rows(&mut y, &w.b, rows, out);
    y
}

/// `[t, c]` row-major -> `[c, t]` row-major. `pub(crate)` - shared with
/// `crate::cv3_flow`.
pub(crate) fn transpose_tc_to_cl(x: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * c];
    for ti in 0..t {
        for ci in 0..c {
            y[ci * t + ti] = x[ti * c + ci];
        }
    }
    y
}

/// `[c, t]` row-major -> `[t, c]` row-major. `pub(crate)` - shared with
/// `crate::cv3_flow`.
pub(crate) fn transpose_cl_to_tc(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * c];
    for ci in 0..c {
        for ti in 0..t {
            y[ti * c + ci] = x[ci * t + ti];
        }
    }
    y
}

/// `pub(crate)` - shared with `crate::cv3_flow`'s `CausalConvPositionEmbedding`.
#[inline]
pub(crate) fn mish(x: f32) -> f32 {
    let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    x * softplus.tanh()
}

pub(crate) fn mish_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = mish(*v);
    }
}

/// `pub(crate)` - shared with `crate::cv3_flow`'s `PreLookaheadLayer`.
#[inline]
pub(crate) fn leaky_relu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.01 * x
    }
}

/// `Conv1d(cin, cout, k)` over a `[cin, l]` channel-major buffer, `pad`
/// applied on the LEFT only - `conv1d_ref`'s own "skip an out-of-range
/// column" bound check supplies right-zero-padding for free whenever `lo >
/// l - pad`, so this one call covers both the causal case (`pad = k-1, lo =
/// l`) and `PreLookaheadLayer.conv1`'s right-pad-only case (`pad = 0, lo =
/// l`) - verified against the reference's `F.pad` calls directly, not
/// assumed. `pub(crate)` - shared with `crate::cv3_flow`, which also needs
/// this exact left/right-pad-via-`lo` trick for its `PreLookaheadLayer` (same
/// module, same weight shapes) and its grouped causal `conv_pos_embed`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d(x_cl: &[f32], w: &LinearW, cin: usize, cout: usize, l: usize, lo: usize, k: usize, pad: usize) -> Vec<f32> {
    conv1d_grouped(x_cl, w, cin, cout, l, lo, k, pad, 1)
}

/// [`conv1d`] generalized with a `groups` parameter, for
/// `crate::cv3_flow::CausalConvPositionEmbedding`'s grouped conv (`groups =
/// 16`) - CV2's own convs are all `groups = 1`, so [`conv1d`] stays the
/// ungrouped entry point every existing call site uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_grouped(x_cl: &[f32], w: &LinearW, cin: usize, cout: usize, l: usize, lo: usize, k: usize, pad: usize, groups: usize) -> Vec<f32> {
    let c = Conv1d { n: 1, cin: cin as u32, l: l as u32, cout: cout as u32, k: k as u32, stride: 1, pad: pad as u32, dilation: 1, groups: groups as u32, lo: lo as u32 };
    let mut y = conv1d_ref(&c, x_cl, &w.w);
    for co in 0..cout {
        for t in 0..lo {
            y[co * lo + t] += w.b[co];
        }
    }
    y
}

/// Causal `Conv1d(cin, cout, k)`: left-pad `k-1`, output length == input
/// length. `pub(crate)` - shared with `crate::cv3_flow`.
pub(crate) fn causal_conv1d(x_cl: &[f32], w: &LinearW, cin: usize, cout: usize, t: usize, k: usize) -> Vec<f32> {
    conv1d(x_cl, w, cin, cout, t, t, k, k - 1)
}

// ---------------------------------------------------------------------------
// UpsampleConformerEncoder
// ---------------------------------------------------------------------------

/// `EspnetRelPositionalEncoding.position_encoding(offset=0, size=t)`:
/// `[2t-1, d]`, row `m` at "relative position" `(t-1) - m` (`+ (t-1)` down to
/// `-(t-1)`) - see the module-level derivation in this crate's porting
/// notes/PR description for why the positive/negative/flip dance in the
/// reference collapses to this one closed-form row index.
fn espnet_rel_pos(t: usize, d: usize) -> Vec<f32> {
    let len = 2 * t - 1;
    let half = d / 2;
    let mut pe = vec![0.0f32; len * d];
    for m in 0..len {
        let pos = (t as f32 - 1.0) - m as f32;
        for pi in 0..half {
            let div = (-((2 * pi) as f32) * 10000f32.ln() / d as f32).exp();
            let angle = pos * div;
            pe[m * d + 2 * pi] = angle.sin();
            pe[m * d + 2 * pi + 1] = angle.cos();
        }
    }
    pe
}

/// The Transformer-XL `rel_shift` trick (`RelPositionMultiHeadedAttention.
/// rel_shift`), one head at a time: `raw` is `[t, 2t-1]` (query pos ×
/// relative-position score), returns `[t, t]` (query pos × key pos).
/// Reimplements the reference's pad/view/slice sequence via direct flat-index
/// arithmetic (derived, not guessed - `rel_shift_matches_a_literal_replay`
/// checks it against a literal reimplementation of every reference step for
/// several `t`).
fn rel_shift(raw: &[f32], t: usize) -> Vec<f32> {
    let w = 2 * t - 1;
    let mut out = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let n = i * w + j + t;
            let i2 = n / (2 * t);
            let k2 = n % (2 * t);
            out[i * t + j] = if k2 == 0 { 0.0 } else { raw[i2 * w + (k2 - 1)] };
        }
    }
    out
}

fn linear_no_subsampling(x_tc: &[f32], w: &SubsampleW, t: usize, d: usize) -> Vec<f32> {
    let mut y = linear_rows_biased(x_tc, &w.linear, t, d, d);
    y = layernorm_rows(&y, &w.ln.w, &w.ln.b, t, d, 1e-5);
    let scale = (d as f32).sqrt();
    for v in y.iter_mut() {
        *v *= scale;
    }
    y
}

/// One `ConformerEncoderLayer`: pre-LN `RelPositionMultiHeadedAttention` +
/// pre-LN swish `PositionwiseFeedForward` (no macaron FFN, no conv module -
/// `macaron_style: False, use_cnn_module: False` in the real
/// `cosyvoice2.yaml`). `x_tc` is `[t, d]`; `pos_emb` is `[2t-1, d]`.
fn conformer_layer(x_tc: &[f32], pos_emb: &[f32], t: usize, cfg: &FlowConfig, w: &ConformerLayerW) -> Vec<f32> {
    let d = cfg.encoder.d_model as usize;
    let (heads, hd) = (cfg.encoder.heads as usize, cfg.encoder.head_dim as usize);
    let eps = cfg.encoder.layer_norm_eps;

    let xn = layernorm_rows(x_tc, &w.norm_mha.w, &w.norm_mha.b, t, d, eps);
    let q = linear_rows_biased(&xn, &w.wq, t, d, d);
    let k = linear_rows_biased(&xn, &w.wk, t, d, d);
    let v = linear_rows_biased(&xn, &w.wv, t, d, d);
    let plen = 2 * t - 1;
    let p = linear_rows(pos_emb, &w.w_pos, plen, d, d);

    let mut ctx = vec![0.0f32; t * d];
    let scale = (hd as f32).sqrt();
    {
        let _attn_timer = crate::profile::FlowAttnTimer::start();
        for h in 0..heads {
            let mut q_u = vec![0.0f32; t * hd];
            let mut q_v = vec![0.0f32; t * hd];
            for ti in 0..t {
                for j in 0..hd {
                    let qv = q[ti * d + h * hd + j];
                    q_u[ti * hd + j] = qv + w.pos_bias_u[h * hd + j];
                    q_v[ti * hd + j] = qv + w.pos_bias_v[h * hd + j];
                }
            }
            let mut matrix_bd_raw = vec![0.0f32; t * plen];
            for ti in 0..t {
                for m in 0..plen {
                    let mut acc = 0.0f32;
                    for j in 0..hd {
                        acc += q_v[ti * hd + j] * p[m * d + h * hd + j];
                    }
                    matrix_bd_raw[ti * plen + m] = acc;
                }
            }
            let matrix_bd = rel_shift(&matrix_bd_raw, t);

            for ti in 0..t {
                let mut scores = vec![0.0f32; t];
                for tj in 0..t {
                    let mut ac = 0.0f32;
                    for j in 0..hd {
                        ac += q_u[ti * hd + j] * k[tj * d + h * hd + j];
                    }
                    scores[tj] = (ac + matrix_bd[ti * t + tj]) / scale;
                }
                softmax(&mut scores);
                for j in 0..hd {
                    let mut acc = 0.0f32;
                    for tj in 0..t {
                        acc += scores[tj] * v[tj * d + h * hd + j];
                    }
                    ctx[ti * d + h * hd + j] = acc;
                }
            }
        }
    }
    let attn_out = linear_rows_biased(&ctx, &w.wo, t, d, d);
    let mut x = x_tc.to_vec();
    for i in 0..t * d {
        x[i] += attn_out[i];
    }

    let xn2 = layernorm_rows(&x, &w.norm_ff.w, &w.norm_ff.b, t, d, eps);
    let ff_hidden = cfg.encoder.ff_dim as usize;
    let mut ff = linear_rows_biased(&xn2, &w.ff1, t, d, ff_hidden);
    for v in ff.iter_mut() {
        *v = model::hostmath::silu(*v);
    }
    let ff2 = linear_rows_biased(&ff, &w.ff2, t, ff_hidden, d);
    for i in 0..t * d {
        x[i] += ff2[i];
    }
    x
}

/// `UpsampleConformerEncoder.forward` (`finalize=True` path: no lookahead
/// context, non-streaming). `token_emb` is `[n, input_size]`; returns `[2n,
/// d_model]`.
fn encoder_forward(cfg: &FlowConfig, w: &crate::flow_import::EncoderW, token_emb: &[f32], n: usize) -> Vec<f32> {
    let d = cfg.encoder.d_model as usize;
    let la = cfg.encoder.pre_lookahead_len as usize;

    let mut xs = linear_no_subsampling(token_emb, &w.embed, n, d);
    let pos_emb = espnet_rel_pos(n, d);

    // PreLookaheadLayer, `finalize=True` -> empty context -> right-pad by
    // `pre_lookahead_len`, not the causal left-pad the rest of the estimator
    // uses (see the module doc's "verify empirically" note): conv1 sees a
    // RIGHT-padded window, conv2 is causal, and the whole thing is a residual
    // add over the ORIGINAL (pre-lookahead) embedding.
    let xs_cl = transpose_tc_to_cl(&xs, n, d);
    let mut h = conv1d(&xs_cl, &w.pre_conv1, d, d, n, n, la + 1, 0);
    for v in h.iter_mut() {
        *v = leaky_relu(*v);
    }
    let h2 = causal_conv1d(&h, &w.pre_conv2, d, d, n, 3);
    let h2_tc = transpose_cl_to_tc(&h2, d, n);
    for i in 0..n * d {
        xs[i] += h2_tc[i];
    }

    for layer in &w.layers {
        xs = conformer_layer(&xs, &pos_emb, n, cfg, layer);
    }

    // Upsample1D: nearest x2, then a causal (left-pad `stride*2`) k=5 conv.
    let xs_cl2 = transpose_tc_to_cl(&xs, n, d);
    let n2 = 2 * n;
    let mut up = vec![0.0f32; d * n2];
    for c in 0..d {
        for i in 0..n2 {
            up[c * n2 + i] = xs_cl2[c * n + i / 2];
        }
    }
    let up_out = causal_conv1d(&up, &w.up_layer_conv, d, d, n2, 5);
    let mut xs2 = transpose_cl_to_tc(&up_out, d, n2);

    xs2 = linear_no_subsampling(&xs2, &w.up_embed, n2, d);
    let pos_emb2 = espnet_rel_pos(n2, d);
    for layer in &w.up_layers {
        xs2 = conformer_layer(&xs2, &pos_emb2, n2, cfg, layer);
    }

    layernorm_rows(&xs2, &w.after_norm.w, &w.after_norm.b, n2, d, cfg.encoder.outer_norm_eps)
}

// ---------------------------------------------------------------------------
// CausalConditionalDecoder (the UNet CFM estimator)
// ---------------------------------------------------------------------------

/// One `CausalResnetBlock1D`. `x_cl` is `[dim_in, t]`; returns `[dim_out, t]`.
fn resnet_block(x_cl: &[f32], t_hidden: &[f32], w: &ResnetBlockW, dim_in: usize, dim_out: usize, t: usize) -> Vec<f32> {
    let mut h = causal_conv1d(x_cl, &w.block1_conv, dim_in, dim_out, t, 3);
    {
        let mut h_tc = transpose_cl_to_tc(&h, dim_out, t);
        h_tc = layernorm_rows(&h_tc, &w.block1_ln.w, &w.block1_ln.b, t, dim_out, 1e-5);
        h = transpose_tc_to_cl(&h_tc, t, dim_out);
    }
    mish_slice(&mut h);

    let mut t_mish = t_hidden.to_vec();
    mish_slice(&mut t_mish);
    let mlp_out = matvec(&w.mlp.w, &t_mish, dim_out, t_hidden.len());
    for c in 0..dim_out {
        let add = mlp_out[c] + w.mlp.b[c];
        for ti in 0..t {
            h[c * t + ti] += add;
        }
    }

    let mut h2 = causal_conv1d(&h, &w.block2_conv, dim_out, dim_out, t, 3);
    {
        let mut h2_tc = transpose_cl_to_tc(&h2, dim_out, t);
        h2_tc = layernorm_rows(&h2_tc, &w.block2_ln.w, &w.block2_ln.b, t, dim_out, 1e-5);
        h2 = transpose_tc_to_cl(&h2_tc, t, dim_out);
    }
    mish_slice(&mut h2);

    let res = conv1d(x_cl, &w.res_conv, dim_in, dim_out, t, t, 1, 0);
    for i in 0..dim_out * t {
        h2[i] += res[i];
    }
    h2
}

/// One `BasicTransformerBlock`: pre-LN self-attention (`bias=False` q/k/v,
/// `bias=True` on `to_out`) + pre-LN exact-GELU FFN. `x_tc` is `[t, dim]`.
fn basic_transformer_block(x_tc: &[f32], w: &CfmBlockW, cfg: &FlowConfig, t: usize) -> Vec<f32> {
    let dim = cfg.estimator.channels as usize;
    let (heads, hd) = (cfg.estimator.num_heads as usize, cfg.estimator.attention_head_dim as usize);
    let inner = heads * hd;
    let eps = cfg.estimator.norm_eps;

    let xn = layernorm_rows(x_tc, &w.norm1.w, &w.norm1.b, t, dim, eps);
    let q = linear_rows(&xn, &w.wq, t, dim, inner);
    let k = linear_rows(&xn, &w.wk, t, dim, inner);
    let v = linear_rows(&xn, &w.wv, t, dim, inner);

    let mut ctx = vec![0.0f32; t * inner];
    let scale = (hd as f32).sqrt();
    {
        let _attn_timer = crate::profile::FlowAttnTimer::start();
        for h in 0..heads {
            for ti in 0..t {
                let mut scores = vec![0.0f32; t];
                for tj in 0..t {
                    let mut acc = 0.0f32;
                    for j in 0..hd {
                        acc += q[ti * inner + h * hd + j] * k[tj * inner + h * hd + j];
                    }
                    scores[tj] = acc / scale;
                }
                softmax(&mut scores);
                for j in 0..hd {
                    let mut acc = 0.0f32;
                    for tj in 0..t {
                        acc += scores[tj] * v[tj * inner + h * hd + j];
                    }
                    ctx[ti * inner + h * hd + j] = acc;
                }
            }
        }
    }
    let attn_out = linear_rows_biased(&ctx, &w.wo, t, inner, dim);
    let mut x = x_tc.to_vec();
    for i in 0..t * dim {
        x[i] += attn_out[i];
    }

    let xn3 = layernorm_rows(&x, &w.norm3.w, &w.norm3.b, t, dim, eps);
    let ff_hidden = w.ff1.b.len();
    let mut ff = linear_rows_biased(&xn3, &w.ff1, t, dim, ff_hidden);
    for v in ff.iter_mut() {
        *v = model::hostmath::gelu_exact(*v);
    }
    let ff2 = linear_rows_biased(&ff, &w.ff2, t, ff_hidden, dim);
    for i in 0..t * dim {
        x[i] += ff2[i];
    }
    x
}

fn transformer_stack(x_cl: &[f32], blocks: &[CfmBlockW], cfg: &FlowConfig, dim: usize, t: usize) -> Vec<f32> {
    let mut x_tc = transpose_cl_to_tc(x_cl, dim, t);
    for b in blocks {
        x_tc = basic_transformer_block(&x_tc, b, cfg, t);
    }
    transpose_tc_to_cl(&x_tc, t, dim)
}

/// `SinusoidalPosEmb(320, scale=1000) -> TimestepEmbedding(320, 1024)`,
/// matching `model::hostmath::timestep_embedding`'s `[sin ‖ cos]` /
/// `downscale_freq_shift` convention exactly (`flip_sin_to_cos=false,
/// downscale_freq_shift=1.0` reproduces Matcha's own `half_dim - 1`
/// denominator and `cat([sin, cos])` order bit-for-bit - see the module doc).
fn time_embed(t: f32, cfg: &FlowConfig, w: &EstimatorW) -> Vec<f32> {
    let dim = cfg.estimator.in_channels as usize;
    let emb = timestep_embedding(t * 1000.0, dim, false, 1.0, 10000.0);
    let mut h = matvec(&w.time_mlp1.w, &emb, cfg.estimator.time_embed_dim as usize, dim);
    for (v, b) in h.iter_mut().zip(&w.time_mlp1.b) {
        *v = model::hostmath::silu(*v + b);
    }
    let mut out = matvec(&w.time_mlp2.w, &h, cfg.estimator.time_embed_dim as usize, cfg.estimator.time_embed_dim as usize);
    for (v, b) in out.iter_mut().zip(&w.time_mlp2.b) {
        *v += b;
    }
    out
}

/// `CausalConditionalDecoder.forward` (non-streaming): `x`, `mu`, `cond` are
/// `[80, t]`; `spks` is `[80]`. Returns the predicted velocity `[80, t]`.
#[allow(clippy::too_many_arguments)]
fn estimator_forward(cfg: &FlowConfig, w: &EstimatorW, x: &[f32], mu: &[f32], t_scalar: f32, spks: &[f32], cond: &[f32], t: usize) -> Vec<f32> {
    let mel = cfg.estimator.mel_channels as usize;
    let ch = cfg.estimator.channels as usize;
    let t_hidden = time_embed(t_scalar, cfg, w);

    let in_ch = cfg.estimator.in_channels as usize;
    let mut x_cat = vec![0.0f32; in_ch * t];
    x_cat[..mel * t].copy_from_slice(x);
    x_cat[mel * t..2 * mel * t].copy_from_slice(mu);
    for c in 0..mel {
        x_cat[(2 * mel + c) * t..(2 * mel + c + 1) * t].fill(spks[c]);
    }
    x_cat[3 * mel * t..4 * mel * t].copy_from_slice(cond);

    // down (n_blocks-per-stage transformer blocks; the trailing conv is a
    // stride-1 causal conv, not a real downsample - see the module doc).
    let d0 = resnet_block(&x_cat, &t_hidden, &w.down.resnet, in_ch, ch, t);
    let d0 = transformer_stack(&d0, &w.down.blocks, cfg, ch, t);
    let skip = d0.clone();
    let mut xm = causal_conv1d(&d0, &w.down.conv, ch, ch, t, 3);

    for stage in &w.mid {
        xm = resnet_block(&xm, &t_hidden, &stage.resnet, ch, ch, t);
        xm = transformer_stack(&xm, &stage.blocks, cfg, ch, t);
    }

    let mut up_in = vec![0.0f32; 2 * ch * t];
    up_in[..ch * t].copy_from_slice(&xm);
    up_in[ch * t..].copy_from_slice(&skip);
    let up = resnet_block(&up_in, &t_hidden, &w.up.resnet, 2 * ch, ch, t);
    let up = transformer_stack(&up, &w.up.blocks, cfg, ch, t);
    let up = causal_conv1d(&up, &w.up.conv, ch, ch, t, 3);

    let mut xf = causal_conv1d(&up, &w.final_block_conv, ch, ch, t, 3);
    {
        let mut xf_tc = transpose_cl_to_tc(&xf, ch, t);
        xf_tc = layernorm_rows(&xf_tc, &w.final_block_ln.w, &w.final_block_ln.b, t, ch, 1e-5);
        xf = transpose_tc_to_cl(&xf_tc, t, ch);
    }
    mish_slice(&mut xf);

    conv1d(&xf, &w.final_proj, ch, mel, t, t, 1, 0)
}

// ---------------------------------------------------------------------------
// condition assembly + CFM Euler loop (`CausalMaskedDiffWithXvec.inference`
// + `CausalConditionalCFM.forward`/`solve_euler`)
// ---------------------------------------------------------------------------

/// `spk_embed_affine_layer(F.normalize(xvec, dim=1))`: `[output_size]`.
pub fn speaker_embedding(w: &FlowWeights, cfg: &FlowConfig, xvec: &[f32]) -> Vec<f32> {
    let n = l2_normalize(xvec);
    let mut e = matvec(&w.spk_affine.w, &n, cfg.output_size as usize, cfg.spk_embed_dim as usize);
    for (v, b) in e.iter_mut().zip(&w.spk_affine.b) {
        *v += b;
    }
    e
}

/// `conds`/`mu` assembly: `token = clamp(concat([prompt_token, token]), min=0)`
/// -> `input_embedding` -> encoder -> `encoder_proj` -> `mu` (channel-major
/// `[output_size, 2n]`); `conds[:, :mel_len1] = prompt_feat` (channel-major).
/// Returns `(mu_cl, conds_cl, mel_len1, mel_len2)`.
pub fn assemble_conditions(
    w: &FlowWeights,
    cfg: &FlowConfig,
    prompt_tokens: &[u32],
    gen_tokens: &[u32],
    prompt_feat_tc: &[f32],
    mel_len1: usize,
) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let d = cfg.input_size as usize;
    let mel = cfg.output_size as usize;
    let n = prompt_tokens.len() + gen_tokens.len();

    let mut token_emb = vec![0.0f32; n * d];
    for (i, &id) in prompt_tokens.iter().chain(gen_tokens.iter()).enumerate() {
        let row = id as usize * d;
        token_emb[i * d..(i + 1) * d].copy_from_slice(&w.input_embedding[row..row + d]);
    }

    let h = encoder_forward(cfg, &w.encoder, &token_emb, n);
    let n2 = 2 * n;
    let mu_tc = linear_rows_biased(&h, &w.encoder_proj, n2, cfg.encoder.d_model as usize, mel);
    let mu_cl = transpose_tc_to_cl(&mu_tc, n2, mel);

    let mel_len2 = n2 - mel_len1;
    let conds_cl = conds_from_prompt_feat(prompt_feat_tc, mel, mel_len1, mel_len2);

    (mu_cl, conds_cl, mel_len1, mel_len2)
}

/// `conds[:, :mel_len1] = prompt_feat; conds[:, mel_len1:] = 0`, transposed to
/// channel-major `[mel, mel_len1+mel_len2]` - the CFM `cond` tensor assembly,
/// identical in `CausalMaskedDiffWithXvec.inference` (CosyVoice 2) and
/// `CausalMaskedDiffWithDiT.inference` (CosyVoice 3, `crate::cv3_flow`):
/// both just write the prompt mel into the first `mel_len1` frames of an
/// otherwise-zero `[mel_len1+mel_len2, mel]` buffer and transpose. `pub(crate)`
/// so `crate::cv3_flow` reuses this instead of duplicating it.
pub(crate) fn conds_from_prompt_feat(prompt_feat_tc: &[f32], mel: usize, mel_len1: usize, mel_len2: usize) -> Vec<f32> {
    assert_eq!(prompt_feat_tc.len(), mel_len1 * mel, "conds_from_prompt_feat: prompt_feat_tc length mismatch");
    let n2 = mel_len1 + mel_len2;
    let mut conds_tc = vec![0.0f32; n2 * mel];
    conds_tc[..mel_len1 * mel].copy_from_slice(prompt_feat_tc);
    transpose_tc_to_cl(&conds_tc, n2, mel)
}

/// The cosine `t_scheduler`: `linspace(0,1,n+1)` then `t = 1 - cos(t*pi/2)`.
/// `pub(crate)` - shared with `crate::cv3_flow`'s Euler loop (the SAME
/// `CausalConditionalCFM`/`ConditionalCFM` schedule regardless of which
/// estimator module is inside it).
pub(crate) fn cosine_t_span(n_timesteps: usize) -> Vec<f32> {
    (0..=n_timesteps)
        .map(|i| {
            let lin = i as f32 / n_timesteps as f32;
            1.0 - (lin * 0.5 * std::f32::consts::PI).cos()
        })
        .collect()
}

/// `CausalConditionalCFM.solve_euler`, replayed line-for-line: classifier-free
/// guidance as two batch-1 `estimator_forward` calls (see the module doc),
/// `n_timesteps` Euler steps over the cosine schedule. `x0` is the entry
/// noise (`rand_noise[.., :t]` for a real run - see [`rand_noise`] - or a
/// captured golden entry state for composed-loop replay). Returns every
/// post-step latent (`[80, t]` each), matching `flow_real_euler_steps.f32`'s
/// `[n_timesteps, 1, 80, t]` layout when flattened.
#[allow(clippy::too_many_arguments)]
pub fn solve_euler(cfg: &FlowConfig, w: &EstimatorW, x0: &[f32], mu: &[f32], spks: &[f32], cond: &[f32], t: usize, n_timesteps: usize) -> Vec<Vec<f32>> {
    let mel = cfg.estimator.mel_channels as usize;
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
        let cond_out = estimator_forward(cfg, w, &x, mu, cur_t, spks, cond, t);
        let uncond_out = estimator_forward(cfg, w, &x, &zeros_mu, cur_t, &zeros_spks, &zeros_cond, t);
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

/// Every intermediate rung of one [`forward`] call, for tests/callers that
/// want to inspect the composition rather than just the final mel.
pub struct ForwardOutput {
    /// `[output_size, mel_len2]` - `feat[:, :, mel_len1:]`.
    pub mel: Vec<f32>,
    /// `[output_size, mel_len1 + mel_len2]`, channel-major.
    pub mu: Vec<f32>,
    /// `[output_size, mel_len1 + mel_len2]`, channel-major.
    pub conds: Vec<f32>,
    /// `[output_size]`.
    pub embedding: Vec<f32>,
    /// One `[output_size, mel_len1 + mel_len2]` entry per Euler step.
    pub euler_steps: Vec<Vec<f32>>,
}

/// The whole `CausalMaskedDiffWithXvec.inference()` forward: condition
/// assembly through the CFM Euler loop, sliced to the generated span
/// (`feat[:, :, mel_len1:]`).
#[allow(clippy::too_many_arguments)]
pub fn forward(
    w: &FlowWeights,
    cfg: &FlowConfig,
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
    assert!(noise.len() >= mel * t, "flow::forward: noise buffer shorter than {}", mel * t);

    let mut x0 = vec![0.0f32; mel * t];
    let noise_len = noise.len() / mel;
    for c in 0..mel {
        x0[c * t..(c + 1) * t].copy_from_slice(&noise[c * noise_len..c * noise_len + t]);
    }

    let steps = solve_euler(cfg, &w.estimator, &x0, &mu_cl, &embedding, &conds_cl, t, n_timesteps);
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

    /// A literal, unoptimized reimplementation of `RelPositionMultiHeadedAttention.
    /// rel_shift`'s pad/view/slice sequence (Vec-based, no tensor library),
    /// checked against the closed-form [`rel_shift`] this module actually
    /// uses at several `t` - the regression test for the flat-index algebra
    /// the module doc calls out as "derived, not guessed".
    fn rel_shift_literal(raw: &[f32], t: usize) -> Vec<f32> {
        let w = 2 * t - 1;
        // x_padded = cat([zero_pad, x], dim=-1): [t, w+1]
        let mut padded = vec![0.0f32; t * (w + 1)];
        for i in 0..t {
            padded[i * (w + 1)] = 0.0;
            padded[i * (w + 1) + 1..i * (w + 1) + 1 + w].copy_from_slice(&raw[i * w..(i + 1) * w]);
        }
        // view as [w+1, t] (same flat buffer, t*(w+1) elements = (w+1)*t)
        // then drop the first row -> [w, t], flatten offset by `t`.
        let flat = padded; // length t*(w+1)
        let shifted = &flat[t..]; // drop first `t` elements == first reshaped row
        // view_as raw shape [t, w], i.e. reinterpret `shifted` (length t*(w+1)-t = t*w) as [t, w]
        // then slice last dim to [..t] (since w//2+1 == t for w == 2t-1).
        let mut out = vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                out[i * t + j] = shifted[i * w + j];
            }
        }
        out
    }

    #[test]
    fn rel_shift_matches_a_literal_replay() {
        for &t in &[2usize, 3, 5, 8] {
            let w = 2 * t - 1;
            let raw: Vec<f32> = (0..t * w).map(|i| i as f32 * 0.1 - 3.0).collect();
            let got = rel_shift(&raw, t);
            let want = rel_shift_literal(&raw, t);
            assert_eq!(got, want, "rel_shift mismatch at t={t}");
        }
    }

    #[test]
    fn transpose_round_trips() {
        let t = 5;
        let c = 3;
        let x: Vec<f32> = (0..t * c).map(|i| i as f32).collect();
        let cl = transpose_tc_to_cl(&x, t, c);
        let back = transpose_cl_to_tc(&cl, c, t);
        assert_eq!(x, back);
    }

    #[test]
    fn cosine_t_span_matches_the_reference_formula() {
        let span = cosine_t_span(10);
        assert_eq!(span.len(), 11);
        assert!((span[0]).abs() < 1e-6);
        assert!((span[10] - 1.0).abs() < 1e-6);
        for i in 1..10 {
            assert!(span[i] > span[i - 1], "t_span must be strictly increasing");
        }
    }

    #[test]
    fn rand_noise_has_the_right_shape_and_is_finite() {
        // The sha256-against-the-golden and dumped-slice checks live with the
        // generator itself, `torch_rng::tests`; `tests/flow_parity.rs` also
        // cross-checks this function's output against
        // `testdata/golden/cosyvoice/flow_real_rand_noise_slice.f32` (skips
        // cleanly when the golden is absent) - this is just a cheap in-crate
        // sanity check that nothing upstream broke shape or introduced NaNs.
        let n = rand_noise();
        assert_eq!(n.len(), 80 * 15000);
        assert!(n.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mish_matches_x_tanh_softplus() {
        for &x in &[-5.0f32, -1.0, 0.0, 0.3, 2.0, 25.0] {
            let want = x * (1.0 + x.exp()).ln().tanh();
            assert!((mish(x) - want).abs() < 1e-4, "mish({x}) = {}, want {want}", mish(x));
        }
    }
}
