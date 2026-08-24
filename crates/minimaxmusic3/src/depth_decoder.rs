// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The RVQ depth decoder: a 4-layer causal transformer (RMSNorm, plain
//! multi-head causal self-attention - no RoPE, no QK-norm, no GQA, no
//! bias anywhere - SwiGLU MLP) that autoregressively predicts the 7
//! residual codebooks within one audio frame.
//!
//! Pure host math, not a device (WGSL) forward, for the same reason
//! `condition_encoder` is: a forward call processes at most
//! `num_codebooks` (8) positions, so there is nothing here to
//! parallelize a device dispatch across. `model::hostmath`'s existing
//! `matvec`/`rmsnorm_rows`/`silu` are reused for the pieces that already
//! have a shared host implementation; attention and every backward pass
//! are hand-derived (standard closed-form transformer-block calculus),
//! since this shape - un-rotated, ungrouped, unnormalized-QK causal
//! attention - has no existing device or host counterpart in this
//! workspace to call into.
//!
//! # Two forward paths, one answer
//!
//! [`forward`] is the full-sequence reference: it runs every position and
//! keeps a [`ForwardCache`] of every intermediate so [`backward`] can be
//! taken against it. That is what training and the real-weight parity test
//! use.
//!
//! Inference uses [`KvCache`]/[`step`] instead, one position at a time. The
//! reference checkpoint's own recipe re-runs the whole growing sequence at
//! every depth step (lengths 2..8 per frame, per CFG branch = 35
//! position-forwards where 8 suffice), which every comparable RVQ depth
//! decoder - Sesame CSM, HF Transformers CSM, Moshi's depformer - avoids
//! with exactly this per-frame cache. [`step`] is bit-identical to the
//! corresponding row of [`forward`], gated by a test that `assert_eq!`s the
//! two f32 vectors rather than comparing within an epsilon; see [`step`]'s
//! own doc for the three properties that make that exactness structural.

use checkpoint::safetensors::{self, StTensor};
use model::hostmath::{linear_rows, linear_rows_bwd, matvec, silu, softmax};
use std::collections::HashMap;
use std::path::Path;

use crate::config::DepthDecoderConfig;

#[derive(Clone)]
pub struct AttnW {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
}

#[derive(Clone)]
pub struct MlpW {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone)]
pub struct BlockW {
    pub ln1: Vec<f32>,
    pub attn: AttnW,
    pub ln2: Vec<f32>,
    pub mlp: MlpW,
}

#[derive(Clone)]
pub struct DepthDecoderWeights {
    pub audio_embeddings: Vec<f32>, // [(num_codebooks-1)*audio_vocab_size, hidden]
    pub projection: Vec<f32>,       // [hidden, hidden]
    pub pos_embedding: Vec<f32>,    // [max_position_embeddings, hidden]
    pub layers: Vec<BlockW>,
    pub norm: Vec<f32>, // [hidden]
    pub audio_heads: Vec<Vec<f32>>, // (num_codebooks-1) x [audio_vocab_size, hidden]
}

impl DepthDecoderWeights {
    /// Mutable access to one of the 7 per-layer LINEAR weights by name
    /// (`"layers.{i}.attn.to_q"` etc., no `.weight` suffix - these are
    /// already the only tensors of their kind, unlike the vocoder's convs
    /// which share a name root with a `.bias`) - LoRA's seam, matching
    /// `vocoder::VocoderWeights::conv_weight_mut`'s shape. Only the
    /// attention/MLP projections are ever adapted (`ln1`/`ln2`/`norm`
    /// gains, `pos_embedding`, `audio_embeddings` and `audio_heads` are
    /// not linear-projection weights and are out of scope for this seam).
    pub fn linear_mut(&mut self, name: &str) -> Option<&mut Vec<f32>> {
        let rest = name.strip_prefix("layers.")?;
        let (i, rest) = rest.split_once('.')?;
        let layer = self.layers.get_mut(i.parse::<usize>().ok()?)?;
        match rest {
            "attn.to_q" => Some(&mut layer.attn.wq),
            "attn.to_k" => Some(&mut layer.attn.wk),
            "attn.to_v" => Some(&mut layer.attn.wv),
            "attn.to_out" => Some(&mut layer.attn.wo),
            "gate_proj" => Some(&mut layer.mlp.gate),
            "up_proj" => Some(&mut layer.mlp.up),
            "down_proj" => Some(&mut layer.mlp.down),
            _ => None,
        }
    }

    /// Every LoRA-eligible linear weight's name, `layers.0.attn.to_q` .. -
    /// see [`Self::linear_mut`] for what's excluded and why.
    pub fn linear_names(&self) -> Vec<String> {
        let per_layer = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "gate_proj", "up_proj", "down_proj"];
        (0..self.layers.len()).flat_map(|i| per_layer.iter().map(move |n| format!("layers.{i}.{n}"))).collect()
    }
}

const RMS_EPS: f32 = 1e-6;

pub fn import(dir: &str, cfg: &DepthDecoderConfig) -> Result<DepthDecoderWeights, String> {
    from_tensors(safetensors::read_model_dir(Path::new(dir))?, cfg, dir)
}

pub fn from_tensors(tensors: Vec<StTensor>, cfg: &DepthDecoderConfig, label: &str) -> Result<DepthDecoderWeights, String> {
    let map: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
    let get = |name: &str| -> Result<Vec<f32>, String> { map.get(name).cloned().ok_or_else(|| format!("depth_decoder: missing {name:?} in {label}")) };

    let mut layers = Vec::with_capacity(cfg.num_layers as usize);
    for i in 0..cfg.num_layers {
        let p = format!("layers.{i}");
        layers.push(BlockW {
            ln1: get(&format!("{p}.input_layernorm.weight"))?,
            attn: AttnW {
                wq: get(&format!("{p}.attn.to_q.weight"))?,
                wk: get(&format!("{p}.attn.to_k.weight"))?,
                wv: get(&format!("{p}.attn.to_v.weight"))?,
                wo: get(&format!("{p}.attn.to_out.weight"))?,
            },
            ln2: get(&format!("{p}.post_attention_layernorm.weight"))?,
            mlp: MlpW { gate: get(&format!("{p}.gate_proj.weight"))?, up: get(&format!("{p}.up_proj.weight"))?, down: get(&format!("{p}.down_proj.weight"))? },
        });
    }
    let mut audio_heads = Vec::with_capacity(cfg.num_codebooks as usize - 1);
    for i in 0..cfg.num_codebooks - 1 {
        audio_heads.push(get(&format!("audio_heads.{i}.weight"))?);
    }
    Ok(DepthDecoderWeights {
        audio_embeddings: get("audio_embeddings.weight")?,
        projection: get("projection.weight")?,
        pos_embedding: get("pos_embedding.weight")?,
        layers,
        norm: get("norm.weight")?,
        audio_heads,
    })
}

/// `projection(x)`: a single bias-free `Linear(hidden, hidden)`.
pub fn projection(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, x: &[f32]) -> Vec<f32> {
    matvec(&w.projection, x, cfg.hidden_size as usize, cfg.hidden_size as usize)
}

/// `audio_embeddings(code)`: one row gather from the combined residual-code
/// embedding table (`code` already includes the caller's per-codebook
/// offset, matching the reference's `codes + offsets`).
pub fn audio_embedding_row(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, code: usize) -> Vec<f32> {
    let d = cfg.hidden_size as usize;
    w.audio_embeddings[code * d..(code + 1) * d].to_vec()
}

/// `audio_heads[i](x)`: a bias-free `Linear(hidden, audio_vocab_size)`.
pub fn audio_head(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, i: usize, x: &[f32]) -> Vec<f32> {
    matvec(&w.audio_heads[i], x, cfg.audio_vocab_size as usize, cfg.hidden_size as usize)
}

fn rmsnorm_row(x: &[f32], g: &[f32]) -> (Vec<f32>, f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv_std = (ss / x.len() as f32 + RMS_EPS).powf(-0.5);
    (x.iter().zip(g).map(|(&xi, &gi)| xi * inv_std * gi).collect(), inv_std)
}

fn rmsnorm_row_bwd(x: &[f32], g: &[f32], inv_std: f32, dy: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let d = x.len() as f32;
    let dg: Vec<f32> = dy.iter().zip(x).map(|(&dyi, &xi)| dyi * xi * inv_std).collect();
    let s: f32 = dy.iter().zip(g).zip(x).map(|((&dyi, &gi), &xi)| dyi * gi * xi).sum();
    let dx: Vec<f32> = dy.iter().zip(g).zip(x).map(|((&dyi, &gi), &xi)| inv_std * gi * dyi - xi * inv_std.powi(3) / d * s).collect();
    (dx, dg)
}

struct AttnCache {
    xn: Vec<f32>,           // post-ln1, [s, d]
    probs: Vec<Vec<f32>>,   // per head, [s, s] (row i valid for j<=i)
    ctx: Vec<f32>,          // [s, hq]  (hq = heads*head_dim = d, since no GQA)
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// `xn` is the ln1 output (`rmsnorm` is applied by the caller); it is taken
/// by value because `AttnCache` keeps it verbatim for `attn_bwd`, so moving
/// it in avoids copying `[s, d]` for nothing.
fn attn_fwd(w: &AttnW, xn: Vec<f32>, s: usize, d: usize, heads: usize) -> (Vec<f32>, AttnCache) {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let q = linear_rows(&xn, &w.wq, s, d, d);
    let k = linear_rows(&xn, &w.wk, s, d, d);
    let v = linear_rows(&xn, &w.wv, s, d, d);

    let mut probs = vec![vec![0.0f32; s]; heads * s];
    let mut ctx = vec![0.0f32; s * d];
    for h in 0..heads {
        for i in 0..s {
            let mut scores = vec![f32::NEG_INFINITY; s];
            for j in 0..=i {
                let mut dot = 0.0f32;
                for t in 0..hd {
                    dot += q[i * d + h * hd + t] * k[j * d + h * hd + t];
                }
                scores[j] = dot * scale;
            }
            softmax(&mut scores);
            probs[h * s + i] = scores;
            let scores = &probs[h * s + i];
            for j in 0..=i {
                let p = scores[j];
                for t in 0..hd {
                    ctx[i * d + h * hd + t] += p * v[j * d + h * hd + t];
                }
            }
        }
    }
    let out = linear_rows(&ctx, &w.wo, s, d, d);
    (out, AttnCache { xn, probs, ctx, q, k, v })
}

fn attn_bwd(w: &AttnW, s: usize, d: usize, heads: usize, cache: &AttnCache, d_out: &[f32]) -> (Vec<f32>, AttnW) {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let (d_ctx, dwo) = linear_rows_bwd(&cache.ctx, &w.wo, d_out, s, d, d);

    let mut dq = vec![0.0f32; s * d];
    let mut dk = vec![0.0f32; s * d];
    let mut dv = vec![0.0f32; s * d];
    for h in 0..heads {
        for i in 0..s {
            let probs_i = &cache.probs[h * s + i];
            let mut d_probs = vec![0.0f32; s];
            for j in 0..=i {
                let mut dot = 0.0f32;
                for t in 0..hd {
                    dot += d_ctx[i * d + h * hd + t] * cache.v[j * d + h * hd + t];
                }
                d_probs[j] = dot;
                for t in 0..hd {
                    dv[j * d + h * hd + t] += probs_i[j] * d_ctx[i * d + h * hd + t];
                }
            }
            // softmax backward: d_scores_j = p_j*(d_probs_j - sum_k p_k*d_probs_k)
            let dot_pd: f32 = (0..=i).map(|j| probs_i[j] * d_probs[j]).sum();
            for j in 0..=i {
                let d_score = probs_i[j] * (d_probs[j] - dot_pd);
                for t in 0..hd {
                    dq[i * d + h * hd + t] += d_score * scale * cache.k[j * d + h * hd + t];
                    dk[j * d + h * hd + t] += d_score * scale * cache.q[i * d + h * hd + t];
                }
            }
        }
    }
    let (dxn_q, dwq) = linear_rows_bwd(&cache.xn, &w.wq, &dq, s, d, d);
    let (dxn_k, dwk) = linear_rows_bwd(&cache.xn, &w.wk, &dk, s, d, d);
    let (dxn_v, dwv) = linear_rows_bwd(&cache.xn, &w.wv, &dv, s, d, d);
    let dxn: Vec<f32> = dxn_q.iter().zip(&dxn_k).zip(&dxn_v).map(|((a, b), c)| a + b + c).collect();
    (dxn, AttnW { wq: dwq, wk: dwk, wv: dwv, wo: dwo })
}

struct BlockCache {
    x: Vec<f32>,
    ln1_inv: Vec<f32>,   // per-row inv_std
    attn: AttnCache,
    mid: Vec<f32>, // x + attn_out
    ln2_inv: Vec<f32>,
    norm2: Vec<f32>, // post-ln2, pre-mlp input, [s,d]
    gate: Vec<f32>,
    up: Vec<f32>,
}

/// `x` is taken by value because `BlockCache` keeps it verbatim for
/// `block_bwd`, so moving it in avoids copying `[s, d]` for nothing.
fn block_fwd(w: &BlockW, cfg: &DepthDecoderConfig, x: Vec<f32>, s: usize) -> (Vec<f32>, BlockCache) {
    let d = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let inter = cfg.intermediate_size as usize;

    let mut xn = vec![0.0f32; s * d];
    let mut ln1_inv = vec![0.0f32; s];
    for r in 0..s {
        let (row, inv) = rmsnorm_row(&x[r * d..(r + 1) * d], &w.ln1);
        xn[r * d..(r + 1) * d].copy_from_slice(&row);
        ln1_inv[r] = inv;
    }
    let (attn_out, attn_cache) = attn_fwd(&w.attn, xn, s, d, heads);
    let mid: Vec<f32> = x.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

    let mut norm2 = vec![0.0f32; s * d];
    let mut ln2_inv = vec![0.0f32; s];
    for r in 0..s {
        let (row, inv) = rmsnorm_row(&mid[r * d..(r + 1) * d], &w.ln2);
        norm2[r * d..(r + 1) * d].copy_from_slice(&row);
        ln2_inv[r] = inv;
    }
    let gate = linear_rows(&norm2, &w.mlp.gate, s, d, inter);
    let up = linear_rows(&norm2, &w.mlp.up, s, d, inter);
    let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    let down = linear_rows(&act, &w.mlp.down, s, inter, d);
    let out: Vec<f32> = mid.iter().zip(&down).map(|(a, b)| a + b).collect();

    (out, BlockCache { x, ln1_inv, attn: attn_cache, mid, ln2_inv, norm2, gate, up })
}

fn silu_bwd_scalar(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

fn block_bwd(w: &BlockW, cfg: &DepthDecoderConfig, cache: &BlockCache, s: usize, d_out: &[f32]) -> (Vec<f32>, BlockW) {
    let d = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let inter = cfg.intermediate_size as usize;

    // out = mid + down; d_mid (direct) = d_out, d_down = d_out.
    let d_down = d_out.to_vec();
    let (d_act, d_down_w) = linear_rows_bwd(&cache.gate.iter().zip(&cache.up).map(|(&g, &u)| silu(g) * u).collect::<Vec<f32>>(), &w.mlp.down, &d_down, s, inter, d);
    let mut d_gate = vec![0.0f32; s * inter];
    let mut d_up = vec![0.0f32; s * inter];
    for i in 0..s * inter {
        let (g, u) = (cache.gate[i], cache.up[i]);
        d_gate[i] = d_act[i] * u * silu_bwd_scalar(g);
        d_up[i] = d_act[i] * silu(g);
    }
    let (d_norm2_g, dw_gate) = linear_rows_bwd(&cache.norm2, &w.mlp.gate, &d_gate, s, d, inter);
    let (d_norm2_u, dw_up) = linear_rows_bwd(&cache.norm2, &w.mlp.up, &d_up, s, d, inter);
    let d_norm2: Vec<f32> = d_norm2_g.iter().zip(&d_norm2_u).map(|(a, b)| a + b).collect();

    let mut d_mid_from_ln2 = vec![0.0f32; s * d];
    let mut dw_ln2 = vec![0.0f32; d];
    for r in 0..s {
        let (dx, dg) = rmsnorm_row_bwd(&cache.mid[r * d..(r + 1) * d], &w.ln2, cache.ln2_inv[r], &d_norm2[r * d..(r + 1) * d]);
        d_mid_from_ln2[r * d..(r + 1) * d].copy_from_slice(&dx);
        for i in 0..d {
            dw_ln2[i] += dg[i];
        }
    }
    // d_mid total = d_out (direct residual) + d_mid_from_ln2.
    let d_mid: Vec<f32> = d_out.iter().zip(&d_mid_from_ln2).map(|(a, b)| a + b).collect();

    // mid = x + attn_out: both branches get d_mid unchanged.
    let (d_xn, dw_attn) = attn_bwd(&w.attn, s, d, heads, &cache.attn, &d_mid);
    let mut d_x_from_ln1 = vec![0.0f32; s * d];
    let mut dw_ln1 = vec![0.0f32; d];
    for r in 0..s {
        let (dx, dg) = rmsnorm_row_bwd(&cache.x[r * d..(r + 1) * d], &w.ln1, cache.ln1_inv[r], &d_xn[r * d..(r + 1) * d]);
        d_x_from_ln1[r * d..(r + 1) * d].copy_from_slice(&dx);
        for i in 0..d {
            dw_ln1[i] += dg[i];
        }
    }
    let d_x: Vec<f32> = d_mid.iter().zip(&d_x_from_ln1).map(|(a, b)| a + b).collect();

    (d_x, BlockW { ln1: dw_ln1, attn: dw_attn, ln2: dw_ln2, mlp: MlpW { gate: dw_gate, up: dw_up, down: d_down_w } })
}

pub struct ForwardCache {
    blocks: Vec<BlockCache>,
    final_inv: Vec<f32>,
    s: usize,
}

/// `forward(inputs_embeds)`: `[s, hidden]` -> normalized hidden states
/// `[s, hidden]`, matching `RVQDepthDecoder.forward` exactly, including
/// adding its own `pos_embedding(arange(s))` internally before the first
/// layer (the reference class does this itself, not something its caller
/// pre-adds).
pub fn forward(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, inputs_embeds: &[f32], s: usize) -> (Vec<f32>, ForwardCache) {
    let d = cfg.hidden_size as usize;
    assert_eq!(inputs_embeds.len(), s * d, "depth_decoder::forward: inputs_embeds length mismatch");
    assert!(s <= cfg.max_position_embeddings as usize, "depth_decoder::forward: {s} positions exceeds max_position_embeddings");
    // `positions = arange(s)`, always starting at 0 - the reference adds
    // its own position embedding internally, not something the caller
    // pre-adds.
    let mut h: Vec<f32> = inputs_embeds.iter().zip(&w.pos_embedding[..s * d]).map(|(a, b)| a + b).collect();
    let mut blocks = Vec::with_capacity(w.layers.len());
    for layer in &w.layers {
        let (out, cache) = block_fwd(layer, cfg, h, s);
        h = out;
        blocks.push(cache);
    }
    let mut out = vec![0.0f32; s * d];
    let mut final_inv = vec![0.0f32; s];
    for r in 0..s {
        let (row, inv) = rmsnorm_row(&h[r * d..(r + 1) * d], &w.norm);
        out[r * d..(r + 1) * d].copy_from_slice(&row);
        final_inv[r] = inv;
    }
    (out, ForwardCache { blocks, final_inv, s })
}

/// The depth decoder's per-frame inference KV cache: the causal
/// self-attention keys and values for every position emitted so far, one
/// `[len, hidden]` buffer per layer.
///
/// One frame's depth sequence is at most `num_codebooks` positions and every
/// frame starts over, so this is created fresh per frame (`new`) rather than
/// reset - at real dims the whole thing is `2 * layers * num_codebooks *
/// hidden` floats, a couple of MB.
pub struct KvCache {
    k: Vec<Vec<f32>>, // per layer, [len, hidden] row-major
    v: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    pub fn new(cfg: &DepthDecoderConfig) -> KvCache {
        let cap = cfg.max_position_embeddings as usize * cfg.hidden_size as usize;
        let layers = cfg.num_layers as usize;
        KvCache { k: (0..layers).map(|_| Vec::with_capacity(cap)).collect(), v: (0..layers).map(|_| Vec::with_capacity(cap)).collect(), len: 0 }
    }

    /// Positions appended so far - the position index [`step`] will use next.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One incremental position of [`forward`]: append `input_embed` (`[hidden]`,
/// the caller's own `inputs_embeds` row for position `cache.len()`) and
/// return that position's normalized hidden state, `[hidden]`.
///
/// This is **bit-identical** to `forward(.., &all_rows_so_far, cache.len()+1)`
/// sliced at its last row, and is the inference path: the reference's own
/// recipe re-runs the whole growing sequence at every depth step, which is
/// ~4.4x the arithmetic for the same answer over an 8-codebook frame. Nothing
/// here is cached for a backward pass - training keeps the full-sequence
/// [`forward`]/[`backward`] pair.
///
/// Bit-identity rests on three properties of this architecture, all of which
/// must hold for a change here to stay exact:
/// * every non-attention op is per-row (RMSNorm, and `matvec`, which
///   `linear_rows` calls once per row anyway), so row `j`'s value does not
///   depend on how many rows accompany it;
/// * attention is causal, so row `i` reads only `k`/`v` of rows `<= i`, which
///   were computed identically when those rows were stepped;
/// * `pos_embedding` is indexed by the absolute position, which `cache.len()`
///   tracks - the reference adds it INSIDE the decoder over `arange(s)`, so a
///   caller that pre-added it would double it.
///
/// The masked-softmax row is also exact rather than merely close: [`forward`]
/// softmaxes a length-`s` score row whose `j > i` entries are `-inf`, and
/// `exp(-inf - m) == 0.0` contributes nothing to the sum, so summing only the
/// `j <= i` entries in the same order gives the identical denominator.
pub fn step(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, cache: &mut KvCache, input_embed: &[f32]) -> Vec<f32> {
    let d = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let inter = cfg.intermediate_size as usize;
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    assert_eq!(input_embed.len(), d, "depth_decoder::step: input_embed length mismatch");
    let i = cache.len;
    assert!(i < cfg.max_position_embeddings as usize, "depth_decoder::step: position {i} exceeds max_position_embeddings");

    let mut h: Vec<f32> = input_embed.iter().zip(&w.pos_embedding[i * d..(i + 1) * d]).map(|(a, b)| a + b).collect();
    for (li, layer) in w.layers.iter().enumerate() {
        let (xn, _) = rmsnorm_row(&h, &layer.ln1);
        let q = matvec(&layer.attn.wq, &xn, d, d);
        let k_row = matvec(&layer.attn.wk, &xn, d, d);
        let v_row = matvec(&layer.attn.wv, &xn, d, d);
        cache.k[li].extend_from_slice(&k_row);
        cache.v[li].extend_from_slice(&v_row);
        let (kc, vc) = (&cache.k[li], &cache.v[li]);

        let mut ctx = vec![0.0f32; d];
        for head in 0..heads {
            let mut scores = vec![0.0f32; i + 1];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for t in 0..hd {
                    dot += q[head * hd + t] * kc[j * d + head * hd + t];
                }
                *sc = dot * scale;
            }
            softmax(&mut scores);
            for (j, &p) in scores.iter().enumerate() {
                for t in 0..hd {
                    ctx[head * hd + t] += p * vc[j * d + head * hd + t];
                }
            }
        }
        let attn_out = matvec(&layer.attn.wo, &ctx, d, d);
        let mid: Vec<f32> = h.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

        let (norm2, _) = rmsnorm_row(&mid, &layer.ln2);
        let gate = matvec(&layer.mlp.gate, &norm2, inter, d);
        let up = matvec(&layer.mlp.up, &norm2, inter, d);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        let down = matvec(&layer.mlp.down, &act, d, inter);
        h = mid.iter().zip(&down).map(|(a, b)| a + b).collect();
    }
    cache.len += 1;
    rmsnorm_row(&h, &w.norm).0
}

/// `backward`: `dy [s, hidden]` (gradient w.r.t. `forward`'s output) ->
/// `(d_inputs_embeds, d_layers, d_norm, d_pos_embedding_rows)`, the last
/// being the gradient for `pos_embedding`'s first `s` rows only (the
/// reference always indexes `arange(s)` from row 0).
#[allow(clippy::type_complexity)]
pub fn backward(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, cache: &ForwardCache, dy: &[f32]) -> (Vec<f32>, Vec<BlockW>, Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size as usize;
    let s = cache.s;
    let mut d_h = vec![0.0f32; s * d];
    let mut d_norm = vec![0.0f32; d];
    // The last block's own forward output is the final norm's input, but
    // isn't cached directly - re-derive it the same way block_fwd computed
    // it (`mid + down_proj(silu(gate)*up)`), from the last block's own
    // cached `gate`/`up` (its `norm2` is NOT the MLP's down-proj input,
    // silu(gate)*up is).
    let last_h = {
        let inter = cfg.intermediate_size as usize;
        let last = cache.blocks.last().unwrap();
        let act: Vec<f32> = last.gate.iter().zip(&last.up).map(|(&g, &u)| silu(g) * u).collect();
        let down = linear_rows(&act, &w.layers.last().unwrap().mlp.down, s, inter, d);
        let h: Vec<f32> = last.mid.iter().zip(&down).map(|(a, b)| a + b).collect();
        h
    };
    for r in 0..s {
        let (dx, dg) = rmsnorm_row_bwd(&last_h[r * d..(r + 1) * d], &w.norm, cache.final_inv[r], &dy[r * d..(r + 1) * d]);
        d_h[r * d..(r + 1) * d].copy_from_slice(&dx);
        for i in 0..d {
            d_norm[i] += dg[i];
        }
    }

    let mut d_layers = Vec::with_capacity(w.layers.len());
    for (layer, cache) in w.layers.iter().zip(cache.blocks.iter()).rev() {
        let (d_x, dw) = block_bwd(layer, cfg, cache, s, &d_h);
        d_h = d_x;
        d_layers.push(dw);
    }
    d_layers.reverse();
    // `h = inputs_embeds + pos_embedding[0..s]`: both branches of this
    // elementwise sum receive the same upstream gradient unchanged.
    let d_pos_embedding_rows = d_h.clone();
    (d_h, d_layers, d_norm, d_pos_embedding_rows)
}

/// Random weights at `cfg`'s dims, deterministic from `seed` - shared by
/// this crate's own tests (`depth_decoder`, `depth_lora`, `pipeline`); a
/// gradcheck/wiring fixture always needs a small, random-weight instance,
/// never the real checkpoint, so this lives once here rather than as
/// three near-identical private copies.
#[cfg(test)]
pub(crate) fn random_weights(cfg: &DepthDecoderConfig, seed: u64) -> DepthDecoderWeights {
    use data::rng::Lcg;
    let mut r = Lcg::new(seed);
    let d = cfg.hidden_size as usize;
    let inter = cfg.intermediate_size as usize;
    let lin = |out: usize, inn: usize, r: &mut Lcg| r.vec_scaled(out * inn, 0.2);
    let mut layers = Vec::with_capacity(cfg.num_layers as usize);
    for _ in 0..cfg.num_layers {
        layers.push(BlockW {
            ln1: vec![1.0; d],
            attn: AttnW { wq: lin(d, d, &mut r), wk: lin(d, d, &mut r), wv: lin(d, d, &mut r), wo: lin(d, d, &mut r) },
            ln2: vec![1.0; d],
            mlp: MlpW { gate: lin(inter, d, &mut r), up: lin(inter, d, &mut r), down: lin(d, inter, &mut r) },
        });
    }
    let mut audio_heads = Vec::with_capacity(cfg.num_codebooks as usize - 1);
    for _ in 0..cfg.num_codebooks - 1 {
        audio_heads.push(lin(cfg.audio_vocab_size as usize, d, &mut r));
    }
    DepthDecoderWeights {
        audio_embeddings: lin((cfg.audio_vocab_size * (cfg.num_codebooks - 1)) as usize, d, &mut r),
        projection: lin(d, d, &mut r),
        pos_embedding: lin(cfg.max_position_embeddings as usize, d, &mut r),
        layers,
        norm: vec![1.0; d],
        audio_heads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    #[test]
    fn forward_shape_matches_tiny_config() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_weights(&cfg, 1);
        let s = cfg.num_codebooks as usize;
        let mut r = Lcg::new(2);
        let x = r.vec_scaled(s * cfg.hidden_size as usize, 0.5);
        let (out, _) = forward(&w, &cfg, &x, s);
        assert_eq!(out.len(), s * cfg.hidden_size as usize);
    }

    /// The gate for the inference KV cache: stepping positions one at a time
    /// must reproduce the full-sequence `forward`'s corresponding row
    /// **exactly**, not within a tolerance. `assert_eq!` on the f32 vectors
    /// is deliberate - this is a pure work-reduction, so any drift at all is
    /// a bug in the cached path, not a rounding difference to be absorbed.
    #[test]
    fn kv_cached_step_is_bit_identical_to_the_full_forward() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_weights(&cfg, 21);
        let d = cfg.hidden_size as usize;
        // Every position `pos_embedding` can address, which covers the
        // `num_codebooks`-long depth sequence the AR loop actually walks.
        let steps = cfg.max_position_embeddings as usize;
        let mut r = Lcg::new(22);
        let rows: Vec<Vec<f32>> = (0..steps).map(|_| r.vec_scaled(d, 0.5)).collect();

        let mut cache = KvCache::new(&cfg);
        let mut seq: Vec<f32> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            seq.extend_from_slice(row);
            let s = i + 1;
            let (full, _) = forward(&w, &cfg, &seq, s);
            let cached = step(&w, &cfg, &mut cache, row);
            assert_eq!(cache.len(), s);
            assert_eq!(cached, full[(s - 1) * d..s * d].to_vec(), "position {i}: cached step diverged from the full forward");
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_weights(&cfg, 11);
        let s = cfg.num_codebooks as usize;
        let mut r = Lcg::new(12);
        let x = r.vec_scaled(s * cfg.hidden_size as usize, 0.4);
        let dy = r.vec_scaled(s * cfg.hidden_size as usize, 1.0);

        let (_, cache) = forward(&w, &cfg, &x, s);
        let (dx, d_layers, d_norm, d_pos) = backward(&w, &cfg, &cache, &dy);

        let loss = |x: &[f32]| -> f32 {
            let (out, _) = forward(&w, &cfg, x, s);
            out.iter().zip(&dy).map(|(a, b)| a * b).sum()
        };
        let eps = 5e-3f32;
        let fd = |base: &[f32], i: usize, f: &dyn Fn(&[f32]) -> f32| {
            let mut p = base.to_vec();
            p[i] = base[i] + eps;
            let lp = f(&p);
            p[i] = base[i] - eps;
            let lm = f(&p);
            (lp - lm) / (2.0 * eps)
        };
        for i in (0..x.len()).step_by((x.len() / 11).max(1)) {
            let num = fd(&x, i, &|xx| loss(xx));
            assert!((num - dx[i]).abs() < 2e-2 + 2e-2 * num.abs().max(dx[i].abs()), "dx[{i}]: numeric={num} analytic={}", dx[i]);
        }

        // pos_embedding row 0 (arange(s) always starts at row 0).
        {
            let loss_p = |v: f32| -> f32 {
                let mut ww = w.clone();
                ww.pos_embedding[0] = v;
                let (out, _) = forward(&ww, &cfg, &x, s);
                out.iter().zip(&dy).map(|(a, b)| a * b).sum()
            };
            let orig = w.pos_embedding[0];
            let lp = loss_p(orig + eps);
            let lm = loss_p(orig - eps);
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - d_pos[0]).abs() < 2e-2 + 2e-2 * num.abs().max(d_pos[0].abs()), "pos_embedding[0]: numeric={num} analytic={}", d_pos[0]);
        }

        // norm.weight (final RMSNorm gain) - one representative index.
        {
            let loss_w = |wv: &[f32]| -> f32 {
                let mut ww = w.clone();
                ww.norm = wv.to_vec();
                let (out, _) = forward(&ww, &cfg, &x, s);
                out.iter().zip(&dy).map(|(a, b)| a * b).sum()
            };
            let num = fd(&w.norm, 0, &loss_w);
            assert!((num - d_norm[0]).abs() < 2e-2 + 2e-2 * num.abs().max(d_norm[0].abs()), "norm.weight[0]: numeric={num} analytic={}", d_norm[0]);
        }

        // Every layer's every leaf tensor, one representative index each.
        for (li, dl) in d_layers.iter().enumerate() {
            let check = |name: &str, get: &dyn Fn(&DepthDecoderWeights) -> f32, set: &dyn Fn(&mut DepthDecoderWeights, f32), ana: f32| {
                let mut wp = w.clone();
                let orig = get(&wp);
                set(&mut wp, orig + eps);
                let lp = {
                    let (out, _) = forward(&wp, &cfg, &x, s);
                    out.iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>()
                };
                set(&mut wp, orig - eps);
                let lm = {
                    let (out, _) = forward(&wp, &cfg, &x, s);
                    out.iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>()
                };
                let num = (lp - lm) / (2.0 * eps);
                assert!((num - ana).abs() < 2e-2 + 2e-2 * num.abs().max(ana.abs()), "layers.{li}.{name}: numeric={num} analytic={ana}");
            };
            check("input_layernorm", &|ww| ww.layers[li].ln1[0], &|ww, v| ww.layers[li].ln1[0] = v, dl.ln1[0]);
            check("attn.to_q", &|ww| ww.layers[li].attn.wq[0], &|ww, v| ww.layers[li].attn.wq[0] = v, dl.attn.wq[0]);
            check("attn.to_k", &|ww| ww.layers[li].attn.wk[0], &|ww, v| ww.layers[li].attn.wk[0] = v, dl.attn.wk[0]);
            check("attn.to_v", &|ww| ww.layers[li].attn.wv[0], &|ww, v| ww.layers[li].attn.wv[0] = v, dl.attn.wv[0]);
            check("attn.to_out", &|ww| ww.layers[li].attn.wo[0], &|ww, v| ww.layers[li].attn.wo[0] = v, dl.attn.wo[0]);
            check("post_attention_layernorm", &|ww| ww.layers[li].ln2[0], &|ww, v| ww.layers[li].ln2[0] = v, dl.ln2[0]);
            check("gate_proj", &|ww| ww.layers[li].mlp.gate[0], &|ww, v| ww.layers[li].mlp.gate[0] = v, dl.mlp.gate[0]);
            check("up_proj", &|ww| ww.layers[li].mlp.up[0], &|ww, v| ww.layers[li].mlp.up[0] = v, dl.mlp.up[0]);
            check("down_proj", &|ww| ww.layers[li].mlp.down[0], &|ww, v| ww.layers[li].mlp.down[0] = v, dl.mlp.down[0]);
        }
    }
}
