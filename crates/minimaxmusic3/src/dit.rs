// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The flow-matching DiT: a 36-layer LayerNorm transformer (partial RoPE -
//! only the first `rotary_dim` of each `attention_head_dim`-wide head
//! rotates - full bidirectional self-attention, no causal mask, since this
//! is a diffusion denoiser over the whole chunk at once) that denoises
//! Flow-VAE latents conditioned on the condition encoder's frame-aligned
//! output.
//!
//! Device (WGSL) forward, unlike `condition_encoder`/`depth_decoder`:
//! `num_layers=36` at `attention_head_dim*num_attention_heads=2048` real
//! dims is genuinely compute-heavy, and unlike the vocoder's DAC decoder
//! this IS a standard transformer block shape with existing reusable
//! device primitives (`model::block`'s `Bidir`/`rope2d_partial`/
//! `LayerNorm`/`swiglu`/`kv_expand`), so nothing here is hand-rolled from
//! raw dispatches the way the vocoder's conv stack had to be.
//!
//! Scope: batch=1 only (every fixture and every real caller in the
//! reference pipeline runs one sequence at a time - the chunked-denoise
//! loop is a Python `for` loop over windows, never a batched tensor).

use audio::conv::{conv1d_fwd, Conv1d, ConvKernels};
use checkpoint::safetensors::{self, StTensor};
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{self, bidir_fwd, layernorm_fwd, rope2d_partial_fwd, swiglu_fwd, Bidir, BidirIds, KernelIds, LayerNormIds};
use std::collections::HashMap;
use std::path::Path;

use crate::config::DitConfig;

pub const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("matmul", kernels::MATMUL),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad_ncl", kernels::BIAS_GRAD_NCL),
    ("kv_expand", kernels::KV_EXPAND),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
];
const CONV1D: usize = 0;
const CONV1D_DX: usize = 1;
const CONV1D_DW: usize = 2;
const MATMUL: usize = 3;
const LAYERNORM: usize = 4;
const LN_STATS: usize = 5;
const LAYERNORM_DX: usize = 6;
const ADD2: usize = 7;
const BIAS_ADD: usize = 8;
#[allow(dead_code)]
const BIAS_GRAD_NCL: usize = 9;
const KV_EXPAND: usize = 10;
const ROPE2D_PARTIAL: usize = 11;
const ATTN_SCORES_BIDIR: usize = 12;
const ATTN_SOFTMAX_BIDIR: usize = 13;
const ATTN_APPLY_BIDIR: usize = 14;
const ATTN_BWD_DSCORES_BIDIR: usize = 15;
const ATTN_BWD_DV_BIDIR: usize = 16;
const ATTN_BWD_DQ_BIDIR: usize = 17;
const ATTN_BWD_DK_BIDIR: usize = 18;
const SILU_MUL: usize = 19;
#[allow(dead_code)]
const SILU_BWD_DA: usize = 20;
#[allow(dead_code)]
const SILU_BWD_DB: usize = 21;

const LN_EPS: f32 = 1e-5;

fn conv_kernels() -> ConvKernels {
    ConvKernels { fwd: CONV1D, dx: CONV1D_DX, dw: CONV1D_DW }
}
fn bidir_ids() -> BidirIds {
    BidirIds { scores: ATTN_SCORES_BIDIR, softmax: ATTN_SOFTMAX_BIDIR, apply: ATTN_APPLY_BIDIR, dscores: ATTN_BWD_DSCORES_BIDIR, dv: ATTN_BWD_DV_BIDIR, dq: ATTN_BWD_DQ_BIDIR, dk: ATTN_BWD_DK_BIDIR }
}
/// Every field but `silu_mul`/`silu_da`/`silu_db` is unread here -
/// `swiglu_fwd` only ever touches `k.silu_mul` - so the rest are dummy
/// indices, never dispatched by this module.
fn kernel_ids() -> KernelIds {
    KernelIds { rmsnorm: 0, rms_inv: 0, rmsnorm_dx: 0, rmsnorm_dw: 0, rope: 0, rope_bwd: 0, gqa_scores: 0, gqa_apply: 0, attn_softmax: 0, gqa_dscores: 0, gqa_dv: 0, gqa_dq: 0, gqa_dk: 0, silu_mul: SILU_MUL, silu_da: SILU_BWD_DA, silu_db: SILU_BWD_DB }
}
fn ln_ids() -> LayerNormIds {
    LayerNormIds { layernorm: LAYERNORM, layernorm_rows: None, ln_stats: LN_STATS, ln_stats_rows: None, layernorm_dx: LAYERNORM_DX, layernorm_dx_rows: None }
}

#[derive(Clone)]
pub struct AttnW {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
}

#[derive(Clone)]
pub struct BlockW {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub attn: AttnW,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub ff_in_w: Vec<f32>,  // [2*ff_inner, inner]
    pub ff_in_b: Vec<f32>,  // [2*ff_inner]
    pub ff_out_w: Vec<f32>, // [inner, ff_inner]
    pub ff_out_b: Vec<f32>, // [inner]
}

#[derive(Clone)]
pub struct DitWeights {
    pub time_proj_weight: Vec<f32>, // [fourier_embedding_dim/2, 1]
    pub time_embed_l1_w: Vec<f32>,
    pub time_embed_l1_b: Vec<f32>,
    pub time_embed_l2_w: Vec<f32>,
    pub time_embed_l2_b: Vec<f32>,
    pub preprocess_conv_w: Vec<f32>, // [concat, concat, 1]
    pub proj_in_w: Vec<f32>,         // [inner, concat]
    pub blocks: Vec<BlockW>,
    pub proj_out_w: Vec<f32>,         // [in_channels, inner]
    pub postprocess_conv_w: Vec<f32>, // [in_channels, in_channels, 1]
}

impl DitWeights {
    /// The mutable slot for one of the 6 LoRA-eligible linear weights per
    /// block (`attn.to_{q,k,v,out}`, `ff_in.weight`, `ff_out.weight` - never
    /// a norm gain/bias, never a bias). Named identically to
    /// `dit_train::BlockD::upload`'s own `Pair` names, so a gradient read
    /// from `dit_train::Trainer::read_grad` and a weight slot from here
    /// always agree on what a name means.
    pub fn linear_mut(&mut self, name: &str) -> Option<&mut Vec<f32>> {
        let rest = name.strip_prefix("blocks.")?;
        let (i, rest) = rest.split_once('.')?;
        let block = self.blocks.get_mut(i.parse::<usize>().ok()?)?;
        match rest {
            "attn.to_q" => Some(&mut block.attn.wq),
            "attn.to_k" => Some(&mut block.attn.wk),
            "attn.to_v" => Some(&mut block.attn.wv),
            "attn.to_out" => Some(&mut block.attn.wo),
            "ff_in.weight" => Some(&mut block.ff_in_w),
            "ff_out.weight" => Some(&mut block.ff_out_w),
            _ => None,
        }
    }

    /// Every name [`Self::linear_mut`] resolves, across every block.
    pub fn linear_names(&self) -> Vec<String> {
        (0..self.blocks.len())
            .flat_map(|i| {
                ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "ff_in.weight", "ff_out.weight"].into_iter().map(move |s| format!("blocks.{i}.{s}"))
            })
            .collect()
    }
}

pub fn import(dir: &str, cfg: &DitConfig) -> Result<DitWeights, String> {
    from_tensors(safetensors::read_model_dir(Path::new(dir))?, cfg, dir)
}

pub fn from_tensors(tensors: Vec<StTensor>, cfg: &DitConfig, label: &str) -> Result<DitWeights, String> {
    let map: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
    let get = |name: &str| -> Result<Vec<f32>, String> { map.get(name).cloned().ok_or_else(|| format!("dit: missing {name:?} in {label}")) };

    let mut blocks = Vec::with_capacity(cfg.num_layers as usize);
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        blocks.push(BlockW {
            norm1_w: get(&format!("{p}.norm1.weight"))?,
            norm1_b: get(&format!("{p}.norm1.bias"))?,
            attn: AttnW {
                wq: get(&format!("{p}.attn.to_q.weight"))?,
                wk: get(&format!("{p}.attn.to_k.weight"))?,
                wv: get(&format!("{p}.attn.to_v.weight"))?,
                wo: get(&format!("{p}.attn.to_out.0.weight"))?,
            },
            norm2_w: get(&format!("{p}.norm2.weight"))?,
            norm2_b: get(&format!("{p}.norm2.bias"))?,
            ff_in_w: get(&format!("{p}.ff_in.weight"))?,
            ff_in_b: get(&format!("{p}.ff_in.bias"))?,
            ff_out_w: get(&format!("{p}.ff_out.weight"))?,
            ff_out_b: get(&format!("{p}.ff_out.bias"))?,
        });
    }
    Ok(DitWeights {
        time_proj_weight: get("time_proj.weight")?,
        time_embed_l1_w: get("time_embed.linear_1.weight")?,
        time_embed_l1_b: get("time_embed.linear_1.bias")?,
        time_embed_l2_w: get("time_embed.linear_2.weight")?,
        time_embed_l2_b: get("time_embed.linear_2.bias")?,
        preprocess_conv_w: get("preprocess_conv.weight")?,
        proj_in_w: get("proj_in.weight")?,
        blocks,
        proj_out_w: get("proj_out.weight")?,
        postprocess_conv_w: get("postprocess_conv.weight")?,
    })
}

impl DitConfig {
    pub fn inner_dim(&self) -> u32 {
        self.num_attention_heads * self.attention_head_dim
    }
    pub fn concat_channels(&self) -> u32 {
        2 * self.in_channels + self.condition_dim
    }
}

/// The random-Fourier timestep embedding (host math - runs once per
/// forward call on a `fourier_embedding_dim`-wide vector, not per token):
/// `angles = 2*pi*t*weight`, `embed = cat(cos(angles), sin(angles))`, then
/// a 2-layer SiLU MLP to `inner_dim`.
///
/// `pub(crate)`, and deliberately takes individual weight slices rather
/// than a whole `&DitWeights`: `dit_shard`'s per-stage forward calls this
/// too, and every stage owns this weight quartet as a REPLICATED parameter
/// (every stage recomputes its own timestep token rather than shipping it
/// over the wire, same reasoning as `ltxv`'s adaLN table) resolved from its
/// own flat per-stage weight map, never a full `DitWeights`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn timestep_embed(time_proj_weight: &[f32], time_embed_l1_w: &[f32], time_embed_l1_b: &[f32], time_embed_l2_w: &[f32], time_embed_l2_b: &[f32], cfg: &DitConfig, timestep: f32) -> Vec<f32> {
    use model::hostmath::{matvec, silu};
    let half = cfg.fourier_embedding_dim as usize / 2;
    let mut embed = vec![0.0f32; 2 * half];
    for i in 0..half {
        let angle = 2.0 * std::f32::consts::PI * timestep * time_proj_weight[i];
        embed[i] = angle.cos();
        embed[half + i] = angle.sin();
    }
    let h1 = matvec(time_embed_l1_w, &embed, cfg.inner_dim() as usize, cfg.fourier_embedding_dim as usize);
    let h1: Vec<f32> = h1.iter().zip(time_embed_l1_b).map(|(&v, &b)| silu(v + b)).collect();
    let h2 = matvec(time_embed_l2_w, &h1, cfg.inner_dim() as usize, cfg.inner_dim() as usize);
    h2.iter().zip(time_embed_l2_b).map(|(&v, &b)| v + b).collect()
}

/// Precompute `cos`/`sin` RoPE tables, `[rows, rotary_dim/2]`, `theta=10000`
/// (the reference `MiniMaxMusic3RotaryEmbedding`'s own default - distinct
/// from the Global LLM's `1e6`).
pub(crate) fn rope_tables(rows: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; rows * half];
    let mut sin = vec![0.0f32; rows * half];
    for t in 0..rows {
        for f in 0..half {
            let inv_freq = 1.0 / theta.powf(2.0 * f as f32 / rotary_dim as f32);
            let angle = t as f32 * inv_freq;
            cos[t * half + f] = angle.cos();
            sin[t * half + f] = angle.sin();
        }
    }
    (cos, sin)
}

pub(crate) struct DeviceBlock {
    norm1_w: DeviceBuffer,
    norm1_b: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    norm2_w: DeviceBuffer,
    norm2_b: DeviceBuffer,
    ff_in_w: DeviceBuffer,
    ff_in_b: DeviceBuffer,
    ff_out_w: DeviceBuffer,
    ff_out_b: DeviceBuffer,
}

pub(crate) fn upload_blocks(gpu: &Gpu, blocks: &[BlockW]) -> Vec<DeviceBlock> {
    blocks
        .iter()
        .map(|b| DeviceBlock {
            norm1_w: gpu.storage_init("norm1.weight", &b.norm1_w),
            norm1_b: gpu.storage_init("norm1.bias", &b.norm1_b),
            wq: gpu.storage_init("to_q.weight", &b.attn.wq),
            wk: gpu.storage_init("to_k.weight", &b.attn.wk),
            wv: gpu.storage_init("to_v.weight", &b.attn.wv),
            wo: gpu.storage_init("to_out.weight", &b.attn.wo),
            norm2_w: gpu.storage_init("norm2.weight", &b.norm2_w),
            norm2_b: gpu.storage_init("norm2.bias", &b.norm2_b),
            ff_in_w: gpu.storage_init("ff_in.weight", &b.ff_in_w),
            ff_in_b: gpu.storage_init("ff_in.bias", &b.ff_in_b),
            ff_out_w: gpu.storage_init("ff_out.weight", &b.ff_out_w),
            ff_out_b: gpu.storage_init("ff_out.bias", &b.ff_out_b),
        })
        .collect()
}

/// One transformer block: `norm1 -> qkv proj -> partial RoPE -> bidir attn
/// -> out proj -> residual -> norm2 -> ff_in -> split -> swiglu -> ff_out
/// -> residual`. `x` is `[rows, inner]`; returns the updated `x`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_fwd(gpu: &Gpu, cfg: &DitConfig, db: &DeviceBlock, x: &DeviceBuffer, cos_b: &DeviceBuffer, sin_b: &DeviceBuffer, rows: usize) -> DeviceBuffer {
    let inner = cfg.inner_dim() as usize;
    let heads = cfg.num_attention_heads;
    let hd = cfg.attention_head_dim;
    let half_rot = cfg.rotary_dim / 2;
    let ff_inner = cfg.ff_inner_dim as usize;
    let stride3 = (3 * inner) as u32;

    let xn1 = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[layernorm_fwd(gpu, &ln_ids(), x, &db.norm1_w, &db.norm1_b, &xn1, inner as u32, rows as u32, LN_EPS)]);

    let q_tmp = gpu.storage((rows * inner) as u64);
    let k_tmp = gpu.storage((rows * inner) as u64);
    let v_tmp = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL, &[&xn1, &db.wq, &q_tmp], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32),
            gpu.step(MATMUL, &[&xn1, &db.wk, &k_tmp], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32),
            gpu.step(MATMUL, &[&xn1, &db.wv, &v_tmp], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32),
        ],
    );

    // Pack q/k/v into one fused [rows, 3*inner] buffer (group=1: a straight
    // copy into each region - same call shape a plain-MHA model's own Q
    // packing already uses elsewhere in this workspace, e.g. `crates/lfm2`).
    let qkv = gpu.storage((rows * 3 * inner) as u64);
    gpu.submit(
        &[],
        &[
            block::kv_expand_fwd(gpu, KV_EXPAND, &q_tmp, &qkv, rows as u32, heads, 1, hd, stride3, 0),
            block::kv_expand_fwd(gpu, KV_EXPAND, &k_tmp, &qkv, rows as u32, heads, 1, hd, stride3, inner as u32),
            block::kv_expand_fwd(gpu, KV_EXPAND, &v_tmp, &qkv, rows as u32, heads, 1, hd, stride3, 2 * inner as u32),
        ],
    );

    gpu.submit(
        &[],
        &[
            rope2d_partial_fwd(gpu, ROPE2D_PARTIAL, &qkv, cos_b, sin_b, rows as u32, heads, half_rot, stride3, 0, hd),
            rope2d_partial_fwd(gpu, ROPE2D_PARTIAL, &qkv, cos_b, sin_b, rows as u32, heads, half_rot, stride3, inner as u32, hd),
        ],
    );

    let bidir_cfg = Bidir { b: 1, t: rows as u32, n_heads: heads, head_dim: hd, stride: stride3, q_off: 0, k_off: inner as u32, v_off: 2 * inner as u32 };
    let scores = gpu.storage((heads * rows as u32 * rows as u32) as u64);
    let probs = gpu.storage((heads * rows as u32 * rows as u32) as u64);
    let ctx = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &bidir_fwd(gpu, &bidir_ids(), &bidir_cfg, &qkv, &scores, &probs, &ctx));

    let proj = gpu.storage((rows * inner) as u64);
    let xmid = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(MATMUL, &[&ctx, &db.wo, &proj], &[rows as u32, inner as u32, inner as u32], (rows * inner) as u32)]);
    gpu.submit(&[], &[gpu.step(ADD2, &[x, &proj, &xmid], &[(rows * inner) as u32], (rows * inner) as u32)]);

    let xn2 = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[layernorm_fwd(gpu, &ln_ids(), &xmid, &db.norm2_w, &db.norm2_b, &xn2, inner as u32, rows as u32, LN_EPS)]);

    let ff_raw = gpu.storage((rows * 2 * ff_inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL, &[&xn2, &db.ff_in_w, &ff_raw], &[rows as u32, inner as u32, (2 * ff_inner) as u32], (rows * 2 * ff_inner) as u32),
            gpu.step(BIAS_ADD, &[&ff_raw, &db.ff_in_b], &[rows as u32, (2 * ff_inner) as u32], (rows * 2 * ff_inner) as u32),
        ],
    );

    // ff_raw = [gate_states | gate] (the reference's own chunk order); the
    // MLP output is gate_states * silu(gate) - swiglu_fwd's (gate, up) args
    // are (silu'd, passthrough), so gate_states plays "up" and gate plays
    // "gate" here. Splitting on the host is cheap relative to the matmuls
    // either side of it and avoids a strided-view kernel just for this cut.
    let ff_host = gpu.read(&ff_raw, rows * 2 * ff_inner);
    let mut up = vec![0.0f32; rows * ff_inner];
    let mut gate = vec![0.0f32; rows * ff_inner];
    for r in 0..rows {
        up[r * ff_inner..(r + 1) * ff_inner].copy_from_slice(&ff_host[r * 2 * ff_inner..r * 2 * ff_inner + ff_inner]);
        gate[r * ff_inner..(r + 1) * ff_inner].copy_from_slice(&ff_host[r * 2 * ff_inner + ff_inner..(r + 1) * 2 * ff_inner]);
    }
    let gate_b = gpu.storage_init("ff_gate", &gate);
    let up_b = gpu.storage_init("ff_up", &up);
    let act = gpu.storage((rows * ff_inner) as u64);
    gpu.submit(&[], &[swiglu_fwd(gpu, &kernel_ids(), &gate_b, &up_b, &act, (rows * ff_inner) as u32)]);

    let ff_out = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            gpu.step(MATMUL, &[&act, &db.ff_out_w, &ff_out], &[rows as u32, ff_inner as u32, inner as u32], (rows * inner) as u32),
            gpu.step(BIAS_ADD, &[&ff_out, &db.ff_out_b], &[rows as u32, inner as u32], (rows * inner) as u32),
        ],
    );

    let x_next = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(ADD2, &[&xmid, &ff_out, &x_next], &[(rows * inner) as u32], (rows * inner) as u32)]);
    x_next
}

/// The embed-stage preamble: `cat([latent, zeros, condition^T], dim=channel)
/// -> preprocess_conv residual -> transpose to feature-last`. `pub(crate)`:
/// `dit_shard`'s embed stage calls this too - the ONLY stage that ever
/// touches `latents`/`condition` (a non-embed stage's input is the previous
/// stage's residual instead, see `dit_shard`'s module doc). Returns
/// `hidden_lc`, `[length, concat_channels]`.
pub(crate) fn preprocess_hidden_lc(gpu: &Gpu, cfg: &DitConfig, preprocess_conv_w: &[f32], latents: &[f32], condition: &[f32], length: usize) -> Vec<f32> {
    let (cin, cdim, concat) = (cfg.in_channels as usize, cfg.condition_dim as usize, cfg.concat_channels() as usize);
    assert_eq!(latents.len(), cin * length, "dit::preprocess_hidden_lc: latents length mismatch");
    assert_eq!(condition.len(), length * cdim, "dit::preprocess_hidden_lc: condition length mismatch");

    // cat([latent, zeros, condition^T], dim=channel) -> [concat, L].
    let mut concat_in = vec![0.0f32; concat * length];
    concat_in[..cin * length].copy_from_slice(latents);
    // zeros region [cin*length .. 2*cin*length) is already 0.
    for t in 0..length {
        for c in 0..cdim {
            concat_in[(2 * cin + c) * length + t] = condition[t * cdim + c];
        }
    }

    let x_in = gpu.storage_init("preprocess_in", &concat_in);
    let pre_w = gpu.storage_init("preprocess_conv.weight", preprocess_conv_w);
    let pre_out = gpu.storage((concat * length) as u64);
    let conv = Conv1d { n: 1, cin: concat as u32, l: length as u32, cout: concat as u32, k: 1, stride: 1, pad: 0, dilation: 1, groups: 1, lo: length as u32 };
    let hidden_ncl = gpu.storage((concat * length) as u64);
    gpu.submit(&[], &[conv1d_fwd(gpu, &conv_kernels(), &conv, &x_in, &pre_w, &pre_out)]);
    gpu.submit(&[], &[gpu.step(ADD2, &[&pre_out, &x_in, &hidden_ncl], &[(concat * length) as u32], (concat * length) as u32)]);

    // transpose [concat, L] -> [L, concat] (host: tiny relative to the
    // transformer stack, and every model-facing buffer downstream is
    // feature-last).
    let hidden_ncl_host = gpu.read(&hidden_ncl, concat * length);
    let mut hidden_lc = vec![0.0f32; length * concat];
    for c in 0..concat {
        for t in 0..length {
            hidden_lc[t * concat + c] = hidden_ncl_host[c * length + t];
        }
    }
    hidden_lc
}

/// `proj_in` over `hidden_lc`'s `length` rows (row 0, the prepended
/// timestep token, is assembled separately by the caller from
/// [`timestep_embed`]). `pub(crate)`: embed-stage-only, like
/// [`preprocess_hidden_lc`]. Returns `[length, inner_dim]`.
pub(crate) fn proj_in_rows(cfg: &DitConfig, proj_in_w: &[f32], hidden_lc: &[f32], length: usize) -> Vec<f32> {
    use model::hostmath::matvec;
    let concat = cfg.concat_channels() as usize;
    let inner = cfg.inner_dim() as usize;
    let mut rows = vec![0.0f32; length * inner];
    for t in 0..length {
        let row = matvec(proj_in_w, &hidden_lc[t * concat..(t + 1) * concat], inner, concat);
        rows[t * inner..(t + 1) * inner].copy_from_slice(&row);
    }
    rows
}

/// The head-stage epilogue: drop row 0 (the timestep token), `proj_out`,
/// transpose to NCL, `postprocess_conv` residual. `pub(crate)`: `dit_shard`'s
/// head stage calls this too - the ONLY stage that ever produces the
/// model's actual output (a non-head stage's output is just its residual,
/// handed to the next stage instead). `x_final_rows` is `[length, inner_dim]`
/// (row 0 of the block stack's own `[rows, inner_dim]` output already
/// dropped by the caller). Returns `[in_channels, length]` NCL.
pub(crate) fn proj_out_postprocess(gpu: &Gpu, cfg: &DitConfig, proj_out_w: &[f32], postprocess_conv_w: &[f32], x_final_rows: &[f32], length: usize) -> Vec<f32> {
    use model::hostmath::matvec;
    let cin = cfg.in_channels as usize;
    let inner = cfg.inner_dim() as usize;
    let mut y_lc = vec![0.0f32; length * cin];
    for t in 0..length {
        let row = matvec(proj_out_w, &x_final_rows[t * inner..(t + 1) * inner], cin, inner);
        y_lc[t * cin..(t + 1) * cin].copy_from_slice(&row);
    }
    let mut y_ncl = vec![0.0f32; cin * length];
    for t in 0..length {
        for c in 0..cin {
            y_ncl[c * length + t] = y_lc[t * cin + c];
        }
    }
    let y_in = gpu.storage_init("postprocess_in", &y_ncl);
    let post_w = gpu.storage_init("postprocess_conv.weight", postprocess_conv_w);
    let post_out = gpu.storage((cin * length) as u64);
    let post_conv = Conv1d { n: 1, cin: cin as u32, l: length as u32, cout: cin as u32, k: 1, stride: 1, pad: 0, dilation: 1, groups: 1, lo: length as u32 };
    let out = gpu.storage((cin * length) as u64);
    gpu.submit(&[], &[conv1d_fwd(gpu, &conv_kernels(), &post_conv, &y_in, &post_w, &post_out)]);
    gpu.submit(&[], &[gpu.step(ADD2, &[&post_out, &y_in, &out], &[(cin * length) as u32], (cin * length) as u32)]);
    gpu.read(&out, cin * length)
}

/// Everything about a denoise chunk that does NOT change from one
/// [`forward_resident`] call to the next: every block's weights uploaded
/// to the device once, and the RoPE tables for this chunk's row count.
///
/// This exists because the denoise loop is the DiT's whole cost centre.
/// `denoise::denoise_chunk` runs `num_inference_steps` Euler steps and
/// evaluates the DiT **twice** per step (the conditional and the
/// zero-condition CFG branch), so a per-call upload re-sends the entire
/// 36-block stack `2 * steps` times per chunk - at `DitConfig::real()`
/// dims that is ~9.7 GB of host->device traffic per evaluation, for
/// weights that are byte-identical every time. The weights depend only on
/// the checkpoint and the RoPE tables only on `rows`, which is fixed
/// within a chunk; neither depends on the timestep, the latents or the
/// condition.
///
/// Lifetime: one chunk. Build it after `length` is known, drop it when the
/// chunk is done - the buffers are the chunk's peak device footprint.
pub struct Resident {
    blocks: Vec<DeviceBlock>,
    cos_b: DeviceBuffer,
    sin_b: DeviceBuffer,
    rows: usize,
}

impl Resident {
    /// Upload `w`'s blocks and build the RoPE tables for a chunk of
    /// `length` latent frames (`rows = length + 1`, the prepended
    /// Fourier-timestep token).
    pub fn new(gpu: &Gpu, cfg: &DitConfig, w: &DitWeights, length: usize) -> Resident {
        let rows = length + 1;
        let (cos_t, sin_t) = rope_tables(rows, cfg.rotary_dim as usize, 10000.0);
        Resident {
            blocks: upload_blocks(gpu, &w.blocks),
            cos_b: gpu.storage_init("rope.cos", &cos_t),
            sin_b: gpu.storage_init("rope.sin", &sin_t),
            rows,
        }
    }

    /// The row count these tables were built for (`length + 1`).
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// The device forward against an already-uploaded [`Resident`] - the
/// production path, called once per CFG branch per Euler step.
///
/// `latents` is `[in_channels, length]` row-major (batch folded away - see
/// the module doc); `condition` is `[length, condition_dim]` row-major
/// (already frame-aligned by the condition encoder). Returns the predicted
/// velocity, `[in_channels, length]`.
///
/// Composed from the same embed-preamble / block-stack / head-epilogue
/// pieces `dit_shard`'s per-stage forward calls, run here over the WHOLE
/// stack (`Shard::whole`, in `dit_shard`'s terms) - one implementation of
/// each piece, never two copies that could drift apart.
pub fn forward_resident(gpu: &Gpu, cfg: &DitConfig, w: &DitWeights, res: &Resident, latents: &[f32], condition: &[f32], timestep: f32, length: usize) -> Vec<f32> {
    let inner = cfg.inner_dim() as usize;
    let rows = length + 1;
    assert_eq!(
        rows, res.rows,
        "dit::forward_resident: Resident was built for {} rows but this call has {rows} - build one Resident per chunk length",
        res.rows
    );
    let hidden_lc = preprocess_hidden_lc(gpu, cfg, &w.preprocess_conv_w, latents, condition, length);

    let temb = timestep_embed(&w.time_proj_weight, &w.time_embed_l1_w, &w.time_embed_l1_b, &w.time_embed_l2_w, &w.time_embed_l2_b, cfg, timestep);
    let proj_rows = proj_in_rows(cfg, &w.proj_in_w, &hidden_lc, length);
    let mut x_host = vec![0.0f32; rows * inner];
    x_host[..inner].copy_from_slice(&temb);
    x_host[inner..].copy_from_slice(&proj_rows);

    let mut x = gpu.storage_init("x0", &x_host);
    for db in &res.blocks {
        x = block_fwd(gpu, cfg, db, &x, &res.cos_b, &res.sin_b, rows);
    }

    let x_final = gpu.read(&x, rows * inner);
    proj_out_postprocess(gpu, cfg, &w.proj_out_w, &w.postprocess_conv_w, &x_final[inner..], length)
}

/// One-shot [`forward_resident`]: uploads the weights, runs one forward,
/// drops them again.
///
/// This is the convenience form for callers that genuinely evaluate the
/// DiT once - the parity tests and `dit_shard`'s single-stage reference.
/// Anything that evaluates the same weights more than once (every real
/// denoise loop) must build a [`Resident`] and call [`forward_resident`],
/// or it pays the full weight upload per evaluation.
pub fn forward(gpu: &Gpu, cfg: &DitConfig, w: &DitWeights, latents: &[f32], condition: &[f32], timestep: f32, length: usize) -> Vec<f32> {
    let res = Resident::new(gpu, cfg, w, length);
    forward_resident(gpu, cfg, w, &res, latents, condition, timestep, length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DitConfig;
    use crate::dit_train;
    use data::rng::Lcg;

    /// A [`Resident`] REUSED across several timesteps and both CFG branches
    /// must produce exactly what a fresh per-call upload produces.
    ///
    /// This is the gate on hoisting the weight upload out of the denoise
    /// loop: the whole point is that nothing observable changes, so an
    /// approximate comparison would not prove it. Bit-for-bit is the right
    /// bar here because the two paths dispatch the identical kernels over
    /// identical bytes in the identical order - only the upload moved.
    #[test]
    fn a_reused_resident_matches_a_fresh_upload_bit_for_bit() {
        let cfg = DitConfig::tiny();
        let gpu = Gpu::new_cpu(PIPELINES);
        let w = dit_train::random_weights(&cfg, 0xD17);
        let length = 3usize;
        let mut r = Lcg::new(0xD17 ^ 0xBEEF);
        let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
        let zero_condition = vec![0.0f32; condition.len()];

        let res = Resident::new(&gpu, &cfg, &w, length);
        assert_eq!(res.rows(), length + 1);

        // Several timesteps, both branches - the real denoise loop's shape.
        for &t in &[0.9f32, 0.5, 0.1] {
            for cond in [&condition, &zero_condition] {
                let fresh = forward(&gpu, &cfg, &w, &latents, cond, t, length);
                let reused = forward_resident(&gpu, &cfg, &w, &res, &latents, cond, t, length);
                assert_eq!(fresh, reused, "resident forward drifted from a fresh upload at t={t}");
            }
        }
    }

    /// A `Resident` built for one chunk length must refuse a call at a
    /// different length rather than silently indexing stale RoPE tables.
    #[test]
    #[should_panic(expected = "build one Resident per chunk length")]
    fn a_resident_refuses_a_different_chunk_length() {
        let cfg = DitConfig::tiny();
        let gpu = Gpu::new_cpu(PIPELINES);
        let w = dit_train::random_weights(&cfg, 0xD18);
        let res = Resident::new(&gpu, &cfg, &w, 3);
        let latents = vec![0.1f32; cfg.in_channels as usize * 4];
        let condition = vec![0.1f32; 4 * cfg.condition_dim as usize];
        let _ = forward_resident(&gpu, &cfg, &w, &res, &latents, &condition, 0.5, 4);
    }
}
