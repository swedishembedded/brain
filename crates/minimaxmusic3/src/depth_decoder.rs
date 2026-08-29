// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The RVQ depth decoder: a 4-layer causal transformer (RMSNorm, plain
//! multi-head causal self-attention - no RoPE, no QK-norm, no GQA, no
//! bias anywhere - SwiGLU MLP) that autoregressively predicts the 7
//! residual codebooks within one audio frame.
//!
//! Swedish Embedded AB implements autoregressive residual-codebook decoders
//! and their accelerator ports for its clients. If your team needs expertise
//! in moving a memory-bound decode loop off the host and onto a GPU without
//! giving up its parity gates, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # Three paths, one answer
//!
//! [`forward`] is the full-sequence reference: it runs every position and
//! keeps a [`ForwardCache`] of every intermediate so [`backward`] can be
//! taken against it. It is HOST math (`model::hostmath`), and it stays that
//! way deliberately - **do not "finish the job" by porting it to WGSL.** It
//! is the training path, `gradcheck`'s finite-difference reference,
//! `crate::depth_lora`'s substrate, and the oracle every other path here is
//! gated against, including the real-checkpoint diffusers comparison in
//! `tests/depth_decoder_parity.rs` (cosine 0.9999). A device rewrite of it
//! would delete the reference that proves the device path right. Attention
//! and every backward are hand-derived (standard closed-form transformer-block
//! calculus): this shape - un-rotated, ungrouped, unnormalized-QK causal
//! attention - has no existing host counterpart in this workspace.
//!
//! Inference uses a one-position-at-a-time KV-cached step instead. The
//! reference checkpoint's own recipe re-runs the whole growing sequence at
//! every depth step (lengths 2..8 per frame, per CFG branch = 35
//! position-forwards where 8 suffice), which every comparable RVQ depth
//! decoder - Sesame CSM, HF Transformers CSM, Moshi's depformer - avoids
//! with exactly this per-frame cache. That step has two implementations,
//! chosen by [`Decoder`]:
//!
//! * **Host** ([`KvCache`]/[`step`]/[`step_batch`]) - `model::hostmath`,
//!   AVX2+FMA+rayon. Bit-identical to the corresponding row of [`forward`],
//!   gated by a test that `assert_eq!`s the two f32 vectors rather than
//!   comparing within an epsilon; see [`step`]'s own doc for the three
//!   properties that make that exactness structural.
//! * **Device** ([`Resident`]) - the same graph as WGSL dispatches through
//!   `gpu_core`, batched over the two CFG branches. It adds NO kernel: every
//!   op here already had one. This is what `--device gpu` runs; the host path
//!   stays the `--device cpu` path, because at these shapes AVX2+rayon over
//!   48 cores beats the Cranelift JIT's rendering of the same graph.
//!
//! # Why this is memory-bound, and what that implies
//!
//! One position at the released dims is about a GFLOP of arithmetic against
//! gigabytes of weights - an arithmetic intensity around 0.5 FLOP/byte, three orders of
//! magnitude below any modern accelerator's balance point. Every decision
//! below follows from that single number: batch the CFG branches so the
//! weights are read once for both, upload the weights ONCE per generation
//! rather than per frame, and never mind the FLOPs.

use checkpoint::safetensors::{self, StTensor};
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, GemmVariants, KernelIds};
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
    step_batch(w, cfg, &mut [cache], &[input_embed]).pop().expect("step_batch returns one row per cache")
}

/// [`step`] for several independent sequences at once: append one row to each
/// of `caches` (which must all stand at the same position) and return each
/// one's normalized hidden state.
///
/// **Why this exists.** This module is memory-bound, not compute-bound: one
/// `step` at the released dims streams gigabytes of weights to do about a
/// GFLOP of arithmetic.
/// Every GEMV here is `[out, inn] × [inn]`, so the weight matrix is thousands
/// of times larger than the vector it multiplies - running `b` sequences as
/// `b` separate `step` calls re-reads all of it `b` times from DRAM for
/// arithmetic that could share a single pass. `linear_rows` batches the rows
/// into one `matmul_abt`, which walks each weight row once and dots it against
/// every input row while it is still in cache, so the DRAM traffic of `b`
/// positions collapses towards that of one.
///
/// **The caller this is for.** `pipeline::generate_depth_codes` drives exactly
/// two sequences - the CFG conditional and unconditional branches - in
/// lockstep, at the same position, on the same weights, meeting only at the
/// logit blend after both `audio_head`s. Those two rows are the ONLY batching
/// available in this loop, and both of the alternatives are genuinely blocked
/// rather than merely unimplemented: the `num_codebooks` depth steps within a
/// frame are strictly dependent (step `i+1`'s input row is the embedding of
/// the code sampled from step `i`'s logits), and consecutive frames are too
/// (`pipeline::embed_audio_frame` sums all `num_codebooks` codes of a frame
/// before the Global LLM advances).
///
/// **Exactness.** Each `b` is bit-identical to the same sequence stepped
/// alone: `linear_rows` gives every output its own accumulator and reduces the
/// contraction in a row-count-independent order (gated by
/// `hostmath::tests::linear_rows_is_bit_identical_to_a_per_row_matvec_loop`),
/// and every other op here is per-row. So [`step`] is a one-line wrapper on
/// this rather than a second copy, and the `assert_eq!` gate against
/// [`forward`] covers both.
pub fn step_batch(w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, caches: &mut [&mut KvCache], embeds: &[&[f32]]) -> Vec<Vec<f32>> {
    let d = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let inter = cfg.intermediate_size as usize;
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let b = caches.len();
    assert!(b > 0, "depth_decoder::step_batch: needs at least one sequence");
    assert_eq!(b, embeds.len(), "depth_decoder::step_batch: {b} caches but {} input rows", embeds.len());
    let i = caches[0].len;
    for (r, c) in caches.iter().enumerate() {
        assert_eq!(c.len, i, "depth_decoder::step_batch: cache {r} is at position {} but cache 0 is at {i}", c.len);
    }
    for (r, e) in embeds.iter().enumerate() {
        assert_eq!(e.len(), d, "depth_decoder::step_batch: input row {r} length mismatch");
    }
    assert!(i < cfg.max_position_embeddings as usize, "depth_decoder::step_batch: position {i} exceeds max_position_embeddings");

    let pos = &w.pos_embedding[i * d..(i + 1) * d];
    let mut h: Vec<f32> = embeds.iter().flat_map(|e| e.iter().zip(pos).map(|(a, p)| a + p)).collect();
    for (li, layer) in w.layers.iter().enumerate() {
        let mut xn = vec![0.0f32; b * d];
        for r in 0..b {
            xn[r * d..(r + 1) * d].copy_from_slice(&rmsnorm_row(&h[r * d..(r + 1) * d], &layer.ln1).0);
        }
        let q = linear_rows(&xn, &layer.attn.wq, b, d, d);
        let k_rows = linear_rows(&xn, &layer.attn.wk, b, d, d);
        let v_rows = linear_rows(&xn, &layer.attn.wv, b, d, d);

        let mut ctx = vec![0.0f32; b * d];
        for (r, cache) in caches.iter_mut().enumerate() {
            cache.k[li].extend_from_slice(&k_rows[r * d..(r + 1) * d]);
            cache.v[li].extend_from_slice(&v_rows[r * d..(r + 1) * d]);
            let (kc, vc) = (&cache.k[li], &cache.v[li]);
            let (qr, ctxr) = (&q[r * d..(r + 1) * d], &mut ctx[r * d..(r + 1) * d]);
            for head in 0..heads {
                let mut scores = vec![0.0f32; i + 1];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for t in 0..hd {
                        dot += qr[head * hd + t] * kc[j * d + head * hd + t];
                    }
                    *sc = dot * scale;
                }
                softmax(&mut scores);
                for (j, &p) in scores.iter().enumerate() {
                    for t in 0..hd {
                        ctxr[head * hd + t] += p * vc[j * d + head * hd + t];
                    }
                }
            }
        }
        let attn_out = linear_rows(&ctx, &layer.attn.wo, b, d, d);
        let mid: Vec<f32> = h.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

        let mut norm2 = vec![0.0f32; b * d];
        for r in 0..b {
            norm2[r * d..(r + 1) * d].copy_from_slice(&rmsnorm_row(&mid[r * d..(r + 1) * d], &layer.ln2).0);
        }
        let gate = linear_rows(&norm2, &layer.mlp.gate, b, d, inter);
        let up = linear_rows(&norm2, &layer.mlp.up, b, d, inter);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        let down = linear_rows(&act, &layer.mlp.down, b, inter, d);
        h = mid.iter().zip(&down).map(|(a, b)| a + b).collect();
    }
    for cache in caches.iter_mut() {
        cache.len += 1;
    }
    (0..b).map(|r| rmsnorm_row(&h[r * d..(r + 1) * d], &w.norm).0).collect()
}

// ---------------------------------------------------------------------------
// The DEVICE inference path
// ---------------------------------------------------------------------------

/// Every kernel [`Resident`] dispatches.
///
/// **This module adds no WGSL.** Each of the eleven already existed and is
/// already gradient-checked or parity-gated by another model; the depth
/// decoder's block is a plain pre-norm transformer with nothing exotic in it,
/// so the whole port is a composition:
///
/// * RMSNorm -> `rmsnorm_eps` / `rmsnorm_rows` through `block::rms_variant`
///   (the cooperative one is the coalescing fix, and at `rows = 2` the
///   per-element kernel would be TWO threads walking 4096 floats each).
/// * every projection -> `block::gemm_variant`, which resolves to
///   `matmul_gemv` in this decode regime (`m = 2 <= 32`), the naive `matmul`
///   on a backend without workgroup reductions.
/// * KV append + attention -> the batched paged decode trio with `group = 1`
///   (this architecture is plain MHA, so `n_kv_heads == n_heads`). Paging is
///   not a complication here, it is the mechanism that makes attention
///   BATCHED over the two CFG branches without slicing anything: one page per
///   branch, `block_size = max_position_embeddings`, `max_bt = 1`, so branch
///   `b`'s position `j` lives at pool row `b*cap + j`.
/// * `silu(gate)*up` -> `silu_mul` via `block::swiglu_fwd`, which takes the
///   single `total` this kernel's contract demands.
/// * both residual adds -> `add2`.
///
/// `block::gqa_attn_sublayer_decode_step` looks like it would do the whole
/// attention sublayer, and is deliberately NOT used: it dispatches RoPE
/// unconditionally (this architecture has none) and is `b = 1`.
pub const PIPELINES: &[(&str, &str)] = &[
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
];
const K_RMSNORM_EPS: usize = 0;
const K_RMSNORM_ROWS: usize = 1;
const K_MATMUL: usize = 2;
const K_MATMUL_REG3: usize = 3;
const K_MATMUL_GEMV: usize = 4;
const K_KV_APPEND: usize = 5;
const K_SCORES: usize = 6;
const K_SOFTMAX: usize = 7;
const K_APPLY: usize = 8;
const K_SILU_MUL: usize = 9;
const K_ADD2: usize = 10;

/// Only `silu_mul` is ever read out of this - `block::swiglu_fwd` touches
/// nothing else - so every other slot is a dummy index this module never
/// dispatches.
fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: K_RMSNORM_EPS,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        rope: block::UNREGISTERED,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: block::UNREGISTERED,
        gqa_apply: block::UNREGISTERED,
        attn_softmax: block::UNREGISTERED,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: K_SILU_MUL,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows: block::UNREGISTERED,
    }
}

/// One RMSNorm over `rows` rows of width `d`, at this module's [`RMS_EPS`].
///
/// The epsilon is passed explicitly (`rmsnorm_eps`, not the fixed-eps
/// `rmsnorm`) even though 1e-6 is exactly what `rmsnorm.wgsl` hardcodes,
/// because `block::rms_variant`'s two kernels must share one `Params` layout
/// and the cooperative `rmsnorm_rows` reads a third `eps` field. Handing that
/// kernel a two-field param list would read whatever the uniform happened to
/// contain as the epsilon.
fn rms_step(gpu: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, rows: u32, d: u32) -> Step {
    let (kind, threads) = block::rms_variant(gpu, K_RMSNORM_EPS, Some(K_RMSNORM_ROWS), rows, d);
    gpu.step(kind, &[x, w, out], &[d, rows, f(RMS_EPS)], threads)
}

/// `out = x @ Wᵀ` for `x: [m,k]`, `w: [n,k]` - through the shared
/// `block::gemm_variant` selection rule, so this module inherits whatever the
/// selector learns next without a change here. The 256-thread tiled kernel and
/// the GEMV are gated on the device's QUERIED `workgroup_reductions`, never on
/// a backend name, so the Cranelift JIT keeps the reference `matmul`.
fn linear_step(gpu: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
    let variant = if gpu.caps().workgroup_reductions {
        GemmVariants::Fast { gemv: Some(K_MATMUL_GEMV), tiled: K_MATMUL_REG3 }
    } else {
        GemmVariants::Reference(K_MATMUL)
    };
    let (kind, threads) = block::gemm_variant(variant, m, n);
    gpu.step(kind, &[x, w, out], &[m, k, n], threads)
}

/// One layer's weights on the device, plus that layer's own KV pool.
struct DeviceLayer {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    ln2: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    down: DeviceBuffer,
    /// `[batch * max_position_embeddings, hidden]` - one page per batch row.
    pool_k: DeviceBuffer,
    pool_v: DeviceBuffer,
}

/// Every intermediate one [`Resident::step`] needs, allocated once and reused
/// by every layer of every step of every frame.
///
/// Allocating these per dispatch would put a few hundred buffer creations per
/// frame on the driver for tens of kilobytes of actual data - the whole set is
/// well under a megabyte at the released dims, so there is nothing to save by
/// being clever and a lot of allocator churn to avoid.
struct Scratch {
    x: DeviceBuffer,
    xn: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    mid: DeviceBuffer,
    norm2: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    act: DeviceBuffer,
    down: DeviceBuffer,
    out: DeviceBuffer,
    /// `pages[b] = b`, u32. Serves BOTH the append kernel's `blocks` binding
    /// and the score/apply kernels' `block_tables` binding: with `max_bt = 1`
    /// those two indexings (`blocks[b]` and `block_tables[b*1 + 0]`) are the
    /// same read of the same table, so there is no reason to keep two.
    pages: DeviceBuffer,
    /// `offsets[b] = pos`, u32 - rewritten each step.
    offsets: DeviceBuffer,
    /// `seq_lens[b] = pos + 1`, u32 - rewritten each step.
    seq_lens: DeviceBuffer,
}

/// The depth decoder's weights uploaded to one device, with the scratch and
/// the per-frame KV pool for `batch` sequences stepped in lockstep.
///
/// **Lifetime: one generation, not one frame.** At the released dims the
/// layer stack is ~2.3 GB of fp32; a frame is ~18 GB of reads against it, so
/// re-uploading per frame would cost more host->device traffic than the
/// computation reads and would dominate everything this port saves. This is
/// the same lesson `crate::dit::Resident` exists for, at a smaller size and a
/// much higher call rate (8 steps per frame, thousands of frames per track).
///
/// **One card, not two.** The point of `batch = 2` is that the two CFG
/// branches share one pass over the weights, which requires them in one
/// dispatch and therefore on one card. `crate::generate::ar_branch_devices`
/// puts the two Global LLM instances on gpu0/gpu1; this sits beside the first
/// of them.
pub struct Resident {
    layers: Vec<DeviceLayer>,
    norm: DeviceBuffer,
    s: Scratch,
    batch: usize,
    cap: u32,
    /// Positions appended to the pool so far, this frame.
    pos: u32,
}

impl Resident {
    /// Upload `w` and allocate everything `batch` lockstep sequences need.
    pub fn new(gpu: &Gpu, cfg: &DepthDecoderConfig, w: &DepthDecoderWeights, batch: usize) -> Resident {
        assert!(batch > 0, "depth_decoder::Resident: batch must be at least 1");
        let d = cfg.hidden_size as u64;
        let inter = cfg.intermediate_size as u64;
        let cap = cfg.max_position_embeddings;
        let b = batch as u64;
        let heads = cfg.num_attention_heads as u64;
        let pool = b * u64::from(cap) * d;
        let layers = w
            .layers
            .iter()
            .map(|l| DeviceLayer {
                ln1: gpu.storage_init("input_layernorm.weight", &l.ln1),
                wq: gpu.storage_init("attn.to_q.weight", &l.attn.wq),
                wk: gpu.storage_init("attn.to_k.weight", &l.attn.wk),
                wv: gpu.storage_init("attn.to_v.weight", &l.attn.wv),
                wo: gpu.storage_init("attn.to_out.weight", &l.attn.wo),
                ln2: gpu.storage_init("post_attention_layernorm.weight", &l.ln2),
                gate: gpu.storage_init("gate_proj.weight", &l.mlp.gate),
                up: gpu.storage_init("up_proj.weight", &l.mlp.up),
                down: gpu.storage_init("down_proj.weight", &l.mlp.down),
                pool_k: gpu.storage(pool),
                pool_v: gpu.storage(pool),
            })
            .collect();
        let s = Scratch {
            x: gpu.storage(b * d),
            xn: gpu.storage(b * d),
            q: gpu.storage(b * d),
            k: gpu.storage(b * d),
            v: gpu.storage(b * d),
            scores: gpu.storage(b * heads * u64::from(cap)),
            probs: gpu.storage(b * heads * u64::from(cap)),
            ctx: gpu.storage(b * d),
            attn_out: gpu.storage(b * d),
            mid: gpu.storage(b * d),
            norm2: gpu.storage(b * d),
            gate: gpu.storage(b * inter),
            up: gpu.storage(b * inter),
            act: gpu.storage(b * inter),
            down: gpu.storage(b * d),
            out: gpu.storage(b * d),
            pages: gpu.storage(b),
            offsets: gpu.storage(b),
            seq_lens: gpu.storage(b),
        };
        gpu.write(&s.pages, &(0..batch as u32).collect::<Vec<u32>>());
        Resident { layers, norm: gpu.storage_init("norm.weight", &w.norm), s, batch, cap, pos: 0 }
    }

    /// How many sequences this residency steps in lockstep.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Positions appended so far - the position index [`Self::step`] uses next.
    pub fn pos(&self) -> usize {
        self.pos as usize
    }

    /// Start a new frame: the depth sequence restarts at position 0.
    ///
    /// The pool is not cleared, and does not need to be: `seq_lens` bounds
    /// every read to `pos+1` rows, so no stale row is ever addressed.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// The device twin of [`step_batch`]: append one row per sequence and
    /// return each one's normalized hidden state.
    ///
    /// **One `Vec<Step>`, one `submit`, one readback.** The graph is ~17 tiny
    /// dispatches per layer, most of them microseconds of actual work; if each
    /// were submitted (or worse, read back) on its own, the queue round-trips
    /// would cost more than the arithmetic. The single readback at the end is
    /// structural rather than avoidable - the caller samples a code from these
    /// logits before it can build the next position's input row.
    pub fn step(&mut self, gpu: &Gpu, cfg: &DepthDecoderConfig, w: &DepthDecoderWeights, embeds: &[&[f32]]) -> Vec<Vec<f32>> {
        let b = self.batch;
        let d = cfg.hidden_size as usize;
        let heads = cfg.num_attention_heads;
        let hd = cfg.hidden_size / heads;
        let inter = cfg.intermediate_size;
        assert_eq!(b, embeds.len(), "depth_decoder::Resident::step: {b} sequences but {} input rows", embeds.len());
        for (r, e) in embeds.iter().enumerate() {
            assert_eq!(e.len(), d, "depth_decoder::Resident::step: input row {r} length mismatch");
        }
        let pos = self.pos;
        assert!(pos < self.cap, "depth_decoder::Resident::step: position {pos} exceeds max_position_embeddings");

        // `pos_embedding` is a row broadcast added to the input, and the input
        // arrives from the host anyway (it is a projection of a freshly
        // sampled code), so it folds into the upload that has to happen -
        // cheaper than uploading the table and spending a `bias_add` dispatch
        // per step on 2 rows.
        let prow = &w.pos_embedding[pos as usize * d..(pos as usize + 1) * d];
        let x0: Vec<f32> = embeds.iter().flat_map(|e| e.iter().zip(prow).map(|(a, p)| a + p)).collect();
        gpu.write_f32(&self.s.x, &x0);
        gpu.write(&self.s.offsets, &vec![pos; b]);
        gpu.write(&self.s.seq_lens, &vec![pos + 1; b]);

        let (bu, du, cap) = (b as u32, cfg.hidden_size, self.cap);
        let scale = 1.0 / (hd as f32).sqrt();
        // `group = 1` (plain MHA: `n_kv_heads == n_heads`), `max_bt = 1` (one
        // page per sequence, and one page holds every position a frame can
        // reach because `block_size == cap == max_position_embeddings`).
        let attn = [bu, heads, 1, hd, cap, du, cap, 1];
        // `paged_decode_scores_batched`'s Params is `paged_decode_apply_
        // batched`'s plus a trailing `scale`, so build it FROM the shared list
        // rather than writing the eight fields out twice.
        let mut scores_params = [0u32; 9];
        scores_params[..8].copy_from_slice(&attn);
        scores_params[8] = f(scale);
        let kids = kernel_ids();
        let mut steps: Vec<Step> = Vec::with_capacity(self.layers.len() * 17 + 1);
        for l in &self.layers {
            let s = &self.s;
            steps.push(rms_step(gpu, &s.x, &l.ln1, &s.xn, bu, du));
            steps.push(linear_step(gpu, &s.xn, &l.wq, &s.q, bu, du, du));
            steps.push(linear_step(gpu, &s.xn, &l.wk, &s.k, bu, du, du));
            steps.push(linear_step(gpu, &s.xn, &l.wv, &s.v, bu, du, du));
            steps.push(gpu.step(K_KV_APPEND, &[&s.k, &s.pages, &s.offsets, &l.pool_k], &[bu, du, cap], bu * du));
            steps.push(gpu.step(K_KV_APPEND, &[&s.v, &s.pages, &s.offsets, &l.pool_v], &[bu, du, cap], bu * du));
            steps.push(gpu.step(K_SCORES, &[&s.q, &l.pool_k, &s.pages, &s.seq_lens, &s.scores], &scores_params, bu * heads * cap));
            steps.push(gpu.step(K_SOFTMAX, &[&s.scores, &s.seq_lens, &s.probs], &[bu, heads, cap], bu * heads));
            steps.push(gpu.step(K_APPLY, &[&s.probs, &l.pool_v, &s.pages, &s.seq_lens, &s.ctx], &attn, bu * heads * hd));
            steps.push(linear_step(gpu, &s.ctx, &l.wo, &s.attn_out, bu, du, du));
            steps.push(gpu.step(K_ADD2, &[&s.x, &s.attn_out, &s.mid], &[bu * du], bu * du));
            steps.push(rms_step(gpu, &s.mid, &l.ln2, &s.norm2, bu, du));
            steps.push(linear_step(gpu, &s.norm2, &l.gate, &s.gate, bu, du, inter));
            steps.push(linear_step(gpu, &s.norm2, &l.up, &s.up, bu, du, inter));
            // `silu_mul` takes a SINGLE `total`, not `[rows, cols]` - the
            // contract that once produced a silently wrong forward at cosine
            // 0.504 elsewhere in this workspace.
            steps.push(block::swiglu_fwd(gpu, &kids, &s.gate, &s.up, &s.act, bu * inter));
            steps.push(linear_step(gpu, &s.act, &l.down, &s.down, bu, inter, du));
            // Back into `x`: the layer's own input is dead by now, and this
            // makes the next layer read the same buffer the first one did.
            steps.push(gpu.step(K_ADD2, &[&s.mid, &s.down, &s.x], &[bu * du], bu * du));
        }
        steps.push(rms_step(gpu, &self.s.x, &self.norm, &self.s.out, bu, du));
        gpu.submit(&[], &steps);

        let host = gpu.read(&self.s.out, b * d);
        self.pos += 1;
        (0..b).map(|r| host[r * d..(r + 1) * d].to_vec()).collect()
    }
}

/// Which implementation of the KV-cached step one generation runs, and the
/// state that goes with it - the single seam `crate::pipeline` drives.
///
/// [`Decoder::Host`] is not a fallback that exists only for machines without
/// a card: at these shapes it is the RIGHT answer on `--device cpu`, because
/// `hostmath`'s AVX2+FMA+rayon path beats the Cranelift JIT's rendering of
/// the same dispatch graph. So this is a selection, not a staged migration
/// with a dead branch.
pub enum Decoder<'a> {
    /// `model::hostmath`, one [`KvCache`] per sequence.
    Host(Vec<KvCache>),
    /// WGSL through `gpu_core`, one [`Resident`] for every sequence. Boxed
    /// so the two variants are the same size: a `Resident` is one `Vec` and
    /// twenty-odd `DeviceBuffer` handles, and inlining that into every
    /// `Decoder` (including host ones, which want none of it) is the
    /// `large_enum_variant` shape. It is built once per generation, so the
    /// indirection costs nothing measurable.
    Device { gpu: &'a Gpu, res: Box<Resident> },
}

impl Decoder<'static> {
    /// The host path, for `batch` sequences stepped in lockstep.
    pub fn host(cfg: &DepthDecoderConfig, batch: usize) -> Decoder<'static> {
        Decoder::Host((0..batch).map(|_| KvCache::new(cfg)).collect())
    }
}

impl<'a> Decoder<'a> {
    /// The device path: uploads `w` to `gpu` ONCE (see [`Resident`] for why
    /// that matters) and steps `batch` sequences per dispatch.
    pub fn device(gpu: &'a Gpu, cfg: &DepthDecoderConfig, w: &DepthDecoderWeights, batch: usize) -> Decoder<'a> {
        Decoder::Device { gpu, res: Box::new(Resident::new(gpu, cfg, w, batch)) }
    }

    /// How many sequences this decoder steps in lockstep.
    pub fn batch(&self) -> usize {
        match self {
            Decoder::Host(caches) => caches.len(),
            Decoder::Device { res, .. } => res.batch(),
        }
    }

    /// Positions stepped so far in the current frame.
    pub fn pos(&self) -> usize {
        match self {
            Decoder::Host(caches) => caches[0].len(),
            Decoder::Device { res, .. } => res.pos(),
        }
    }

    /// Start a new frame at position 0.
    pub fn reset(&mut self, cfg: &DepthDecoderConfig) {
        match self {
            Decoder::Host(caches) => {
                for c in caches.iter_mut() {
                    *c = KvCache::new(cfg);
                }
            }
            Decoder::Device { res, .. } => res.reset(),
        }
    }

    /// One position for every sequence: [`step_batch`] or [`Resident::step`].
    pub fn step(&mut self, w: &DepthDecoderWeights, cfg: &DepthDecoderConfig, embeds: &[&[f32]]) -> Vec<Vec<f32>> {
        match self {
            Decoder::Host(caches) => {
                let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
                step_batch(w, cfg, &mut refs, embeds)
            }
            Decoder::Device { gpu, res } => res.step(gpu, cfg, w, embeds),
        }
    }
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
/// this crate's own tests (`depth_decoder`, `depth_lora`, `pipeline`) and by
/// `mm3_bench`; a gradcheck/wiring fixture always needs a small,
/// random-weight instance, never the real checkpoint, so this lives once here
/// rather than as three near-identical private copies.
///
/// `pub`, and not `#[cfg(test)]`, for the same reason
/// [`crate::dit_train::random_weights`] and [`crate::train::random_weights`]
/// are: a timing harness needs shape-correct weights without a multi-GB
/// checkpoint, and a dispatch's cost is a function of its shape, not of the
/// values in its buffers.
pub fn random_weights(cfg: &DepthDecoderConfig, seed: u64) -> DepthDecoderWeights {
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

    /// The floors the device path is held to against the host [`forward`].
    ///
    /// **Both, not either.** Cosine is scale-invariant, so on its own it
    /// would sail past a wrong RMSNorm epsilon, a dropped `1/sqrt(head_dim)`
    /// or any other uniform mis-scaling of the whole vector - exactly the
    /// mistakes a hand-mapped kernel port makes. Relative L2 catches those and
    /// is in turn weak where cosine is strong. A port has to clear both.
    const DEV_COS_FLOOR: f32 = 0.999999;
    const DEV_REL_L2_CEIL: f32 = 1e-4;

    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        if den <= 0.0 {
            return num as f32;
        }
        (num / den) as f32
    }

    /// Walk every position a frame can reach on a `batch`-row [`Resident`],
    /// with INDEPENDENT random inputs per row, and compare each row against
    /// that row's own sequence through the host [`forward`].
    ///
    /// The per-row inputs are independent on purpose: with the same row fed to
    /// both branches (which is what the real AR loop does past position 0) a
    /// port that mixed the two batch rows up, or that let one row's KV page
    /// bleed into the other's, would produce exactly the right answer and this
    /// gate would pass. Different rows make that class of bug visible.
    fn check_device_against_forward(gpu: &Gpu, cfg: &DepthDecoderConfig, w: &DepthDecoderWeights, seed: u64, tag: &str) {
        let d = cfg.hidden_size as usize;
        let batch = 2usize;
        let steps = cfg.max_position_embeddings as usize;
        let mut r = Lcg::new(seed);
        // `[position][batch]`, so one step's input rows are one contiguous slice.
        let rows: Vec<Vec<Vec<f32>>> = (0..steps).map(|_| (0..batch).map(|_| r.vec_scaled(d, 0.5)).collect()).collect();

        let mut res = Resident::new(gpu, cfg, w, batch);
        let mut seqs: Vec<Vec<f32>> = vec![Vec::new(); batch];
        let (mut worst_cos, mut worst_rel) = (f32::MAX, 0.0f32);
        for (i, at_pos) in rows.iter().enumerate() {
            let embeds: Vec<&[f32]> = at_pos.iter().map(|row| row.as_slice()).collect();
            assert_eq!(res.pos(), i, "{tag}: position counter");
            let got = res.step(gpu, cfg, w, &embeds);
            for b in 0..batch {
                seqs[b].extend_from_slice(&at_pos[b]);
                let s = i + 1;
                let (full, _) = forward(w, cfg, &seqs[b], s);
                let want = &full[(s - 1) * d..s * d];
                let cos = model::hostmath::cosine(&got[b], want);
                let rel = rel_l2(&got[b], want);
                worst_cos = worst_cos.min(cos);
                worst_rel = worst_rel.max(rel);
                // Both messages carry both numbers: whichever assertion fires
                // first, the other's value is what says whether the fault is a
                // direction error or a scale error.
                assert!(cos >= DEV_COS_FLOOR, "{tag}: row {b} position {i}: cosine {cos} below floor {DEV_COS_FLOOR} (rel_l2 {rel})");
                assert!(rel <= DEV_REL_L2_CEIL, "{tag}: row {b} position {i}: rel_l2 {rel} above ceiling {DEV_REL_L2_CEIL} (cosine {cos})");
            }
        }
        println!("depth_decoder device[{tag}]: worst cosine={worst_cos:.9} worst rel_l2={worst_rel:e}");
    }

    /// The device step at tiny dims on the CPU backend - the branch
    /// `BRAIN_DEVICE=cpu` takes, where `workgroup_reductions` is false and the
    /// selectors fall back to the reference `matmul`/`rmsnorm_eps`. That
    /// branch is invisible on a machine with a card (§F.4), so it gets its own
    /// test rather than riding on the default device's.
    #[test]
    fn the_device_step_matches_the_host_forward_on_the_cpu_backend() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_weights(&cfg, 0x0DEC);
        let gpu = Gpu::new_cpu(PIPELINES);
        check_device_against_forward(&gpu, &cfg, &w, 0x0DED, "tiny/cpu");
    }

    /// The same gate on whatever device this machine really selects - the
    /// cooperative `rmsnorm_rows` and the `matmul_gemv` decode GEMM, which the
    /// CPU backend never reaches.
    #[test]
    fn the_device_step_matches_the_host_forward_on_the_default_device() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_weights(&cfg, 0x0DEC);
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        check_device_against_forward(&gpu, &cfg, &w, 0x0DED, "tiny/device");
    }

    /// The same gate at the RELEASED dims with the REAL weights, which tiny
    /// cannot stand in for: `hidden = 4096` is where the cooperative RMSNorm
    /// and the workgroup-per-column GEMV actually have work to split, and real
    /// weights are the value distribution the tolerance has to hold at. Opt-in
    /// (it uploads ~2.3 GB and runs the host `forward` over every prefix), so
    /// it skips unless `BRAIN_MINIMAXMUSIC3_DEPTH` names the checkpoint.
    #[test]
    fn the_device_step_matches_the_host_forward_at_real_dims() {
        let Ok(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_DEPTH") else {
            println!("skipped: BRAIN_MINIMAXMUSIC3_DEPTH unset");
            return;
        };
        let cfg = DepthDecoderConfig::real();
        let w = import(&dir, &cfg).expect("depth decoder import");
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        check_device_against_forward(&gpu, &cfg, &w, 0x0DEE, "real/device");
    }

    /// A `Decoder` must give the same answer whichever implementation it
    /// holds, and must restart cleanly at position 0 on `reset` - the
    /// per-frame lifecycle `pipeline::generate_depth_codes` drives it through.
    #[test]
    fn a_reset_decoder_reproduces_its_own_first_frame() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_weights(&cfg, 0x0DF0);
        let d = cfg.hidden_size as usize;
        let mut r = Lcg::new(0x0DF1);
        let rows: Vec<Vec<f32>> = (0..cfg.num_codebooks as usize).map(|_| r.vec_scaled(d, 0.5)).collect();
        let gpu = Gpu::new_cpu(PIPELINES);
        for mut dec in [Decoder::host(&cfg, 2), Decoder::device(&gpu, &cfg, &w, 2)] {
            let run = |dec: &mut Decoder| -> Vec<Vec<f32>> {
                dec.reset(&cfg);
                rows.iter().map(|row| dec.step(&w, &cfg, &[row, row]).concat()).collect()
            };
            let first = run(&mut dec);
            assert_eq!(first, run(&mut dec), "a reset decoder replayed a different frame");
            assert_eq!(dec.pos(), rows.len());
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
