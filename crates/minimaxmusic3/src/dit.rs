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
    // The fast GEMM family. Registering ONLY the naive `matmul` above meant
    // every projection in all 36 blocks ran the one-thread-per-output
    // reference kernel - measured at tens of seconds per denoise step, which
    // is the defect class AGENTS.md calls this repo's most expensive ("a
    // fast kernel a later model never learned about").
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    // The fused bidirectional flash-attention family, all four rungs, so
    // `model::block::flash_bidir_variant` walks the whole ladder from this
    // device's queried caps (§F.7) and this crate inherits any rung the
    // shared selector learns about later without a change here.
    //
    // Same defect class as the GEMM note above, and the same size: with only
    // the materialized `attn_scores/softmax/apply_bidir` trio registered, the
    // three of them measured as three quarters of this DiT's device time at
    // `DitConfig::real()` / 689 latents, each at a low single-digit percent or
    // less of that card's own measured memory roof - all three flagged DEFECT
    // by `mm3_bench dit` against its own floor, while the GEMMs beside them
    // ran an order of magnitude closer to fp32 peak. The kernels that fix it already existed.
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
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
const MATMUL_REG3: usize = 22;
const MATMUL_GEMV: usize = 23;
const FLASH_ATTN_BIDIR: usize = 24;
const FLASH_ATTN_BIDIR_SPLIT: usize = 25;
const FLASH_ATTN_BIDIR_REG: usize = 26;
const FLASH_ATTN_BIDIR_REG2: usize = 27;

const LN_EPS: f32 = 1e-5;

fn conv_kernels() -> ConvKernels {
    ConvKernels { fwd: CONV1D, dx: CONV1D_DX, dw: CONV1D_DW }
}
/// This crate's rung set for [`block::flash_bidir_variant`] - all four
/// registered, so the shared selector picks on queried `DeviceCaps` alone
/// and never on a backend name or a shape.
const FLASH_IDS: block::FlashIds = block::FlashIds {
    bidir: FLASH_ATTN_BIDIR,
    split: Some(FLASH_ATTN_BIDIR_SPLIT),
    reg: Some(FLASH_ATTN_BIDIR_REG),
    reg2: Some(FLASH_ATTN_BIDIR_REG2),
};
fn bidir_ids() -> BidirIds {
    BidirIds { scores: ATTN_SCORES_BIDIR, softmax: ATTN_SOFTMAX_BIDIR, apply: ATTN_APPLY_BIDIR, dscores: ATTN_BWD_DSCORES_BIDIR, dv: ATTN_BWD_DV_BIDIR, dq: ATTN_BWD_DQ_BIDIR, dk: ATTN_BWD_DK_BIDIR }
}
/// `swiglu_fwd` only ever touches `k.silu_mul`, so every RMSNorm/RoPE/GQA slot
/// is [`block::UNREGISTERED`] - the sentinel, never `0`. Index `0` here is
/// `conv1d`, a kernel this module really does dispatch, so a slot holding it
/// would misroute rather than fail (see `block::UNREGISTERED`'s own doc for
/// why that is silent on `backend-cpu`).
fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: block::UNREGISTERED,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
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
        silu_mul: SILU_MUL,
        silu_da: SILU_BWD_DA,
        silu_db: SILU_BWD_DB,
        rmsnorm_rows: block::UNREGISTERED,
    }
}
/// The forward-GEMM pipeline indices one module of this crate registered.
///
/// A struct rather than three arguments because [`linear_step`] is shared with
/// [`crate::dit_train`], whose PIPELINES list is its own and numbers these
/// kernels differently: the SELECTION RULE has one home, the indices stay with
/// whoever registered them.
#[derive(Clone, Copy)]
pub(crate) struct GemmIds {
    /// `matmul.wgsl` - the portable reference (`@opt 2`, one thread per output
    /// element, serial inner reduction). The only tier a device without
    /// workgroup reductions can run.
    pub reference: usize,
    /// `matmul_reg3.wgsl` - 128x128 register-tiled, `@workgroup_size(256)`.
    pub tiled: usize,
    /// `matmul_gemv.wgsl` where registered; `None` where it is not. Requires
    /// `m <= 32`, which [`model::block::gemm_variant`] enforces.
    pub gemv: Option<usize>,
}

/// `out = x @ W^T` for `x: [m,k]`, `w: [n,k]` - through
/// `model::block::gemm_variant`, the SAME shared GEMM selection rule
/// `ltxv::block::linear` and `wan::block::linear` use.
///
/// Both this module and [`crate::dit_train`] used to hardcode the naive
/// `matmul` at every one of their dispatch sites. That is the reference kernel
/// (`@opt 2`); the register-tiled `matmul_reg3` (`@opt 5`) is what every other
/// transformer in this workspace dispatches. On a real P40 the naive kernel put
/// the DiT at a fraction of a percent of the card's fp32 peak.
///
/// The 256-thread tiled/GEMV kernels are gated on the device's QUERIED
/// `workgroup_reductions` capability, so the CPU JIT (which reports it
/// false) keeps the reference kernel and stays correct.
pub(crate) fn linear_step(gpu: &Gpu, ids: GemmIds, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> gpu_core::Step {
    let variant = if gpu.caps().workgroup_reductions {
        block::GemmVariants::Fast { gemv: ids.gemv, tiled: ids.tiled }
    } else {
        block::GemmVariants::Reference(ids.reference)
    };
    let (kind, threads) = block::gemm_variant(variant, m, n);
    gpu.step(kind, &[x, w, out], &[m, k, n], threads)
}

/// This module's own indices for [`linear_step`].
fn gemm_ids() -> GemmIds {
    GemmIds { reference: MATMUL, tiled: MATMUL_REG3, gemv: Some(MATMUL_GEMV) }
}

/// Whether this block's self-attention takes the fused flash path.
///
/// Two conditions, neither of them a backend name:
///
/// * `DeviceCaps::workgroup_reductions` - the QUERIED capability
///   [`linear_step`] already gates the tiled GEMMs on. Every kernel in the
///   flash family is workgroup-cooperative and needs two or three top-level
///   barriers where the Cranelift CPU JIT supports one, so on that backend
///   this is false and the materialized `attn_scores/softmax/apply_bidir`
///   trio stays the reference definition of the math - the branch
///   `BRAIN_DEVICE=cpu` and `make gradcheck` take.
/// * `head_dim <= 128` - the family's hard limit
///   (`block::flash_bidir_step` asserts it), checked here so an out-of-range
///   config is a fallback rather than a panic. `DitConfig::real()` runs at 64
///   and `::tiny()` lower still.
///
/// Everything below the gate is unconditional: the backward pass
/// (`bidir_bwd`, `crate::dit_train`) is untouched and keeps the materialized
/// trio, because the flash kernels are forward-only and the block's backward
/// reads the `probs` slab this path does not write.
fn flash_attn(gpu: &Gpu, head_dim: u32) -> bool {
    gpu.caps().workgroup_reductions && head_dim <= 128
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
    /// `ff_in` is ONE `[2*ff_inner, inner]` weight in the checkpoint and in
    /// [`BlockW`], but the two halves it fuses are contiguous ROW ranges of
    /// it, so on the device it is two independent `[ff_inner, inner]`
    /// linears. That is what lets `block_fwd` project straight into
    /// separate `up`/`gate` buffers instead of producing one fused
    /// `[rows, 2*ff_inner]` activation that then has to be cut apart -
    /// see `block_fwd`'s own comment for what the cut used to cost.
    ///
    /// Only the DEVICE layout changes: `BlockW`, `DitWeights::linear_mut`,
    /// `dit_shard`'s flat map and `dit_lora`'s shape table all still name
    /// one fused `ff_in.weight`, which is what the checkpoint holds.
    ff_in_up_w: DeviceBuffer,
    ff_in_gate_w: DeviceBuffer,
    ff_in_up_b: DeviceBuffer,
    ff_in_gate_b: DeviceBuffer,
    ff_out_w: DeviceBuffer,
    ff_out_b: DeviceBuffer,
}

pub(crate) fn upload_blocks(gpu: &Gpu, blocks: &[BlockW]) -> Vec<DeviceBlock> {
    blocks
        .iter()
        .map(|b| {
            // `[2*ff_inner, inner]` row-major, so rows `[0, ff_inner)` are
            // the `gate_states` ("up") half and `[ff_inner, 2*ff_inner)` the
            // `gate` half - both contiguous slices, never a gather. The bias
            // is `[2*ff_inner]`, which is where `ff_inner` comes from here
            // without threading a `DitConfig` into this function.
            assert_eq!(b.ff_in_b.len() % 2, 0, "ff_in.bias must be [2*ff_inner]");
            let ff_inner = b.ff_in_b.len() / 2;
            let half = b.ff_in_w.len() / 2;
            assert_eq!(b.ff_in_w.len() % b.ff_in_b.len(), 0, "ff_in.weight must be [2*ff_inner, inner]");
            DeviceBlock {
                norm1_w: gpu.storage_init("norm1.weight", &b.norm1_w),
                norm1_b: gpu.storage_init("norm1.bias", &b.norm1_b),
                wq: gpu.storage_init("to_q.weight", &b.attn.wq),
                wk: gpu.storage_init("to_k.weight", &b.attn.wk),
                wv: gpu.storage_init("to_v.weight", &b.attn.wv),
                wo: gpu.storage_init("to_out.weight", &b.attn.wo),
                norm2_w: gpu.storage_init("norm2.weight", &b.norm2_w),
                norm2_b: gpu.storage_init("norm2.bias", &b.norm2_b),
                ff_in_up_w: gpu.storage_init("ff_in.weight.up", &b.ff_in_w[..half]),
                ff_in_gate_w: gpu.storage_init("ff_in.weight.gate", &b.ff_in_w[half..]),
                ff_in_up_b: gpu.storage_init("ff_in.bias.up", &b.ff_in_b[..ff_inner]),
                ff_in_gate_b: gpu.storage_init("ff_in.bias.gate", &b.ff_in_b[ff_inner..]),
                ff_out_w: gpu.storage_init("ff_out.weight", &b.ff_out_w),
                ff_out_b: gpu.storage_init("ff_out.bias", &b.ff_out_b),
            }
        })
        .collect()
}

/// One transformer block: `norm1 -> qkv proj -> partial RoPE -> bidir attn
/// -> out proj -> residual -> norm2 -> ff_in(up) + ff_in(gate) -> swiglu ->
/// ff_out -> residual`. `x` is `[rows, inner]`; returns the updated `x`.
///
/// Stays on the device end to end - no host round trip anywhere in here.
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
            linear_step(gpu, gemm_ids(), &xn1, &db.wq, &q_tmp, rows as u32, inner as u32, inner as u32),
            linear_step(gpu, gemm_ids(), &xn1, &db.wk, &k_tmp, rows as u32, inner as u32, inner as u32),
            linear_step(gpu, gemm_ids(), &xn1, &db.wv, &v_tmp, rows as u32, inner as u32, inner as u32),
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

    let ctx = gpu.storage((rows * inner) as u64);
    if flash_attn(gpu, hd) {
        // One fused dispatch replaces the whole scores/softmax/apply chain.
        // The shape contract `flash_bidir_step` documents is met exactly by
        // the `qkv` slab built above: `[rows, 3*inner]` with
        // `stride = 3*inner`, `q_off/k_off/v_off = 0/inner/2*inner`,
        // `d_model = inner = heads*head_dim`, `bsz = 1`, and `head_dim` 64 at
        // `DitConfig::real()` (asserted <= 128 by the callee). Both arms
        // apply the same `1/sqrt(head_dim)` scale and write the same
        // `[rows, inner]` row-major context, so `to_out` and the residual
        // below cannot tell them apart.
        gpu.submit(&[], &[block::flash_bidir_step(gpu, FLASH_IDS, 1, heads, rows as u32, hd, inner as u32, &qkv, &ctx)]);
    } else {
        let bidir_cfg = Bidir { b: 1, t: rows as u32, n_heads: heads, head_dim: hd, stride: stride3, q_off: 0, k_off: inner as u32, v_off: 2 * inner as u32 };
        let scores = gpu.storage((heads * rows as u32 * rows as u32) as u64);
        let probs = gpu.storage((heads * rows as u32 * rows as u32) as u64);
        gpu.submit(&[], &bidir_fwd(gpu, &bidir_ids(), &bidir_cfg, &qkv, &scores, &probs, &ctx));
    }

    let proj = gpu.storage((rows * inner) as u64);
    let xmid = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[linear_step(gpu, gemm_ids(), &ctx, &db.wo, &proj, rows as u32, inner as u32, inner as u32)]);
    gpu.submit(&[], &[gpu.step(ADD2, &[x, &proj, &xmid], &[(rows * inner) as u32], (rows * inner) as u32)]);

    let xn2 = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[layernorm_fwd(gpu, &ln_ids(), &xmid, &db.norm2_w, &db.norm2_b, &xn2, inner as u32, rows as u32, LN_EPS)]);

    // `ff_in` projects to `[gate_states | gate]` (the reference's own chunk
    // order) and the MLP output is `gate_states * silu(gate)` -
    // `swiglu_fwd`'s (gate, up) args are (silu'd, passthrough), so
    // `gate_states` plays "up" and `gate` plays "gate".
    //
    // The two halves come out of two SEPARATE `[ff_inner, inner]` linears
    // (see [`DeviceBlock::ff_in_up_w`]), so they land already contiguous and
    // already separated and nothing is ever cut apart. This used to be ONE
    // `n = 2*ff_inner` linear whose `[rows, 2*ff_inner]` output was read
    // back to the host, sliced into two `Vec`s and re-uploaded: at
    // `DitConfig::real()` (rows=690, ff_inner=8192) that is 45.2 MB down +
    // 2 x 22.6 MB up PER BLOCK, x36 blocks x 2 forwards per sampler step -
    // and because `Gpu::read` drains the queue it also forced a full
    // pipeline sync per block, serialising the whole forward.
    //
    // Splitting the WEIGHT instead cannot move a bit: `out[m, c] =
    // sum_k xn2[m, k] * w[c, k]` reduces over `k` only, so row `c` of the
    // up half and row `c` of the gate half produce the identical f32 in the
    // identical order that rows `c` and `ff_inner + c` of the fused weight
    // did. The dispatch geometry is unchanged too - `block::gemm_variant`
    // asks for `m.div_ceil(128) * n.div_ceil(128) * 256` threads, and
    // 2 x (n = ff_inner) is exactly 1 x (n = 2*ff_inner).
    let up_b = gpu.storage((rows * ff_inner) as u64);
    let gate_b = gpu.storage((rows * ff_inner) as u64);
    gpu.submit(
        &[],
        &[
            linear_step(gpu, gemm_ids(), &xn2, &db.ff_in_up_w, &up_b, rows as u32, inner as u32, ff_inner as u32),
            linear_step(gpu, gemm_ids(), &xn2, &db.ff_in_gate_w, &gate_b, rows as u32, inner as u32, ff_inner as u32),
            gpu.step(BIAS_ADD, &[&up_b, &db.ff_in_up_b], &[rows as u32, ff_inner as u32], (rows * ff_inner) as u32),
            gpu.step(BIAS_ADD, &[&gate_b, &db.ff_in_gate_b], &[rows as u32, ff_inner as u32], (rows * ff_inner) as u32),
        ],
    );

    let act = gpu.storage((rows * ff_inner) as u64);
    gpu.submit(&[], &[swiglu_fwd(gpu, &kernel_ids(), &gate_b, &up_b, &act, (rows * ff_inner) as u32)]);

    let ff_out = gpu.storage((rows * inner) as u64);
    gpu.submit(
        &[],
        &[
            linear_step(gpu, gemm_ids(), &act, &db.ff_out_w, &ff_out, rows as u32, ff_inner as u32, inner as u32),
            gpu.step(BIAS_ADD, &[&ff_out, &db.ff_out_b], &[rows as u32, inner as u32], (rows * inner) as u32),
        ],
    );

    let x_next = gpu.storage((rows * inner) as u64);
    gpu.submit(&[], &[gpu.step(ADD2, &[&xmid, &ff_out, &x_next], &[(rows * inner) as u32], (rows * inner) as u32)]);

    // Bound the forward's device footprint to ONE block's scratch.
    //
    // The FFN split's readback used to be this block's only queue drain, so
    // deleting it deleted the drain too - and this is not optional
    // bookkeeping: `Gpu::submit` merely appends to the backend's pending
    // list, a pending `Step` holds its buffers alive, and wgpu cannot
    // reclaim a submitted buffer's memory until a poll observes the work
    // complete. Without a drain here all 36 blocks' intermediates stay
    // resident at once - ~257 MB per block at `DitConfig::real()` (rows=690)
    // on top of 9.66 GB of weights, which is a hard `wgpu error: Out of
    // Memory` on a 24 GB P40, measured, not predicted.
    //
    // `Gpu::flush` is NOT enough and was tried: it submits without waiting,
    // so nothing has completed and nothing is reclaimable. It OOMs at
    // exactly the same place. This has to be the blocking form.
    //
    // What the block no longer pays is the TRANSFER: the drain stays, the
    // 45.2 MB down + 2 x 22.6 MB up per block, the host slicing and the two
    // re-uploads are gone.
    gpu.poll_wait();
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
/// Lifetime: one GENERATION, not one chunk. The blocks depend on nothing
/// but the checkpoint, so re-uploading them per chunk re-sent ~9.7 GB of
/// identical bytes for every one of a song's chunks (~59 of them for four
/// minutes of audio, and the re-upload dominated each one). Only the RoPE
/// tables depend on `length`, and they are ~90 kB, so the chunk loop calls
/// [`Resident::rebind`] instead of rebuilding this whole thing - see
/// `denoise::ChunkResidents`, which owns one per card for the whole denoise
/// stage.
///
/// The buffers are that stage's steady-state device footprint, and the
/// stage's own block scope in `generate::generate` is what bounds them: the
/// DiT must be gone before the vocoder loads.
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

    /// Point this resident at a chunk of `length` latent frames, rebuilding
    /// ONLY the RoPE tables - the blocks stay exactly where they are.
    ///
    /// This is why one `Resident` can serve a whole generation. The tables
    /// are `[rows, rotary_dim/2]` twice (~90 kB at `DitConfig::real()` dims
    /// and a full chunk), against ~9.7 GB of blocks; rebuilding them is
    /// free, rebuilding the blocks is the defect this method exists to
    /// remove.
    ///
    /// `rows` moves in the same statement as the buffers it describes, so a
    /// table that disagrees with the row count [`forward_resident`] asserts
    /// on is not merely unlikely, it is unrepresentable. A `length` that
    /// already matches is a no-op - `denoise::chunk_starts` hands out mostly
    /// equal-length chunks, so most calls do nothing at all.
    pub fn rebind(&mut self, gpu: &Gpu, cfg: &DitConfig, length: usize) {
        let rows = length + 1;
        if rows == self.rows {
            return;
        }
        let (cos_t, sin_t) = rope_tables(rows, cfg.rotary_dim as usize, 10000.0);
        self.cos_b = gpu.storage_init("rope.cos", &cos_t);
        self.sin_b = gpu.storage_init("rope.sin", &sin_t);
        self.rows = rows;
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

/// The shared gate on `model::block::UNREGISTERED`: an unused [`KernelIds`]
/// slot must not be able to reach a kernel this crate really dispatches.
///
/// Lives here rather than in either test module because BOTH `dit` (the served
/// forward) and [`crate::dit_train`] (the trainer) fill the same struct from
/// two different pipeline lists, and a gate that exists in only one of them is
/// a gate the other file's next editor will not know about.
///
/// The check is dispatch-linked on purpose. A purely static "every unused slot
/// equals the sentinel" assertion cannot say whether getting it wrong would
/// have mattered; this one names the kernel a wrong slot would actually have
/// run, because it runs the real pass and reads the device's own per-kernel
/// dispatch counters back.
#[cfg(test)]
pub(crate) mod slot_gate {
    use gpu_core::Gpu;
    use model::block::KernelIds;
    use std::collections::BTreeSet;

    /// The kernel NAMES `pass` dispatched on `gpu`.
    pub(crate) fn dispatched(gpu: &Gpu, pass: impl FnOnce()) -> BTreeSet<String> {
        // Arms the tally as well as clearing it - it is off by default.
        gpu.reset_ops_counters();
        pass();
        let r = gpu.ops_counters();
        r.by_kernel.keys().chain(r.uncovered.keys()).cloned().collect()
    }

    /// Assert that no slot outside `used` names a kernel in `dispatched`.
    pub(crate) fn assert_unused_slots_unreachable(ids: &KernelIds, used: &[&str], pipelines: &[(&str, &str)], dispatched: &BTreeSet<String>) {
        assert!(
            pipelines.get(model::block::UNREGISTERED).is_none(),
            "the UNREGISTERED sentinel names a kernel in a {}-entry pipeline list, so a slot \
             holding it would dispatch a real kernel instead of failing",
            pipelines.len()
        );
        assert!(!dispatched.is_empty(), "the pass under test dispatched nothing, so this gate would pass vacuously");
        for (slot, idx) in ids.slots() {
            if used.contains(&slot) {
                continue;
            }
            // Out of range (the sentinel) is the pass condition: such a slot
            // cannot name any kernel at all, so `Gpu::step` panics on it
            // rather than running one.
            let Some((name, _)) = pipelines.get(idx) else { continue };
            assert!(
                !dispatched.contains(*name),
                "unused KernelIds slot `{slot}` holds pipeline index {idx} = `{name}`, which this \
                 pass really dispatches. A builder reading that slot would run `{name}` against \
                 another kernel\'s bindings and uniform - on `backend-cpu` an out-of-bounds read, \
                 not a panic. Unused slots must be `model::block::UNREGISTERED`."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DitConfig;
    use crate::dit_train;
    use data::rng::Lcg;

    /// No `KernelIds` slot this module leaves unused can reach a kernel the
    /// served forward really dispatches (`model::block::UNREGISTERED`).
    ///
    /// This is the gate on defect class "index 0 as an unregistered marker".
    /// `PIPELINES[0]` here is `conv1d`, which `forward` dispatches twice per
    /// call, so filling the thirteen RMSNorm/RoPE/GQA slots with `0` - as this
    /// file did - put a live kernel index in every one of them. Mutation-check
    /// it by putting any of them back to `0`: this test goes red naming
    /// `conv1d`.
    #[test]
    fn no_unused_kernel_slot_can_reach_a_dispatched_kernel() {
        let cfg = DitConfig::tiny();
        let gpu = Gpu::new_cpu(PIPELINES);
        let w = dit_train::random_weights(&cfg, 0xD17);
        let length = 3usize;
        let mut r = Lcg::new(0xD17 ^ 0x5107);
        let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);

        let seen = slot_gate::dispatched(&gpu, || {
            forward(&gpu, &cfg, &w, &latents, &condition, 0.5, length);
        });
        // `swiglu_fwd` is the ONLY `model::block` builder this module hands a
        // `KernelIds` to, and it reads `silu_mul` alone. The two `silu_bwd_*`
        // slots hold their own kernels\' indices (registered, never dispatched
        // here), which the gate covers rather than exempts.
        slot_gate::assert_unused_slots_unreachable(&kernel_ids(), &["silu_mul"], PIPELINES, &seen);
    }

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

    /// Splitting the fused `ff_in` weight at upload must be a bit-for-bit
    /// no-op against the fused linear the block used to run.
    ///
    /// `block_fwd` no longer produces a `[rows, 2*ff_inner]` activation at
    /// all - it runs two `[ff_inner, inner]` linears whose outputs already
    /// ARE `up` and `gate`, so the host round trip that used to cut them
    /// apart is gone. That rests on exactly one claim: `out[m, c] = sum_k
    /// x[m, k] * w[c, k]` reduces over `k` only, so a weight row's output
    /// cannot depend on how many OTHER rows share the dispatch. This is the
    /// gate on it, and equality has to be exact - anything else means the
    /// GEMM's reduction order moves with `n`, and the model's output moved
    /// with it.
    ///
    /// Both shapes matter and neither is redundant. `(640, 256)` is the
    /// production case: `n` a multiple of `matmul_reg3`'s 128-wide tile
    /// either way, so the two half dispatches tile exactly as the fused one
    /// did. `(130, 100)` is deliberately ragged - `m` past one tile, and
    /// neither `2*ff_inner` nor `ff_inner` a multiple of 128 - so the split
    /// changes the partial-tile geometry (`2 * 100.div_ceil(128)` tiles
    /// where the fused form had `200.div_ceil(128)`) and the masking of the
    /// out-of-range lanes has to stay exact too.
    ///
    /// Runs on the pooled test device, not `Gpu::new_cpu`: on the CPU
    /// backend `linear_step` picks the reference `matmul` (the JIT reports
    /// `workgroup_reductions: false`), so pinning this test to the CPU
    /// would never touch `matmul_reg3` - the kernel the real forward runs.
    #[test]
    fn splitting_the_ff_in_weight_matches_the_fused_linear_bit_for_bit() {
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        for (rows, inner, ff_inner) in [(640usize, 256usize, 256usize), (130, 100, 100)] {
            let mut r = Lcg::new(0xF17 ^ rows as u64);
            let x = r.vec_scaled(rows * inner, 0.7);
            let w = r.vec_scaled(2 * ff_inner * inner, 0.5);
            let b = r.vec_scaled(2 * ff_inner, 0.3);
            let xb = gpu.storage_init("x", &x);
            let tag = format!("rows={rows} inner={inner} ff_inner={ff_inner}");

            // The fused form, plus the host slice that used to follow it.
            let wb = gpu.storage_init("ff_in.weight", &w);
            let bb = gpu.storage_init("ff_in.bias", &b);
            let fused = gpu.storage((rows * 2 * ff_inner) as u64);
            gpu.submit(
                &[],
                &[
                    linear_step(&gpu, gemm_ids(), &xb, &wb, &fused, rows as u32, inner as u32, (2 * ff_inner) as u32),
                    gpu.step(BIAS_ADD, &[&fused, &bb], &[rows as u32, (2 * ff_inner) as u32], (rows * 2 * ff_inner) as u32),
                ],
            );
            let fused_host = gpu.read(&fused, rows * 2 * ff_inner);
            let mut want_up = vec![0.0f32; rows * ff_inner];
            let mut want_gate = vec![0.0f32; rows * ff_inner];
            for i in 0..rows {
                let row = &fused_host[i * 2 * ff_inner..(i + 1) * 2 * ff_inner];
                want_up[i * ff_inner..(i + 1) * ff_inner].copy_from_slice(&row[..ff_inner]);
                want_gate[i * ff_inner..(i + 1) * ff_inner].copy_from_slice(&row[ff_inner..]);
            }

            // The split form, exactly as `upload_blocks` + `block_fwd` build it.
            let half = w.len() / 2;
            let up_w = gpu.storage_init("ff_in.weight.up", &w[..half]);
            let gate_w = gpu.storage_init("ff_in.weight.gate", &w[half..]);
            let up_bias = gpu.storage_init("ff_in.bias.up", &b[..ff_inner]);
            let gate_bias = gpu.storage_init("ff_in.bias.gate", &b[ff_inner..]);
            let up = gpu.storage((rows * ff_inner) as u64);
            let gate = gpu.storage((rows * ff_inner) as u64);
            gpu.submit(
                &[],
                &[
                    linear_step(&gpu, gemm_ids(), &xb, &up_w, &up, rows as u32, inner as u32, ff_inner as u32),
                    linear_step(&gpu, gemm_ids(), &xb, &gate_w, &gate, rows as u32, inner as u32, ff_inner as u32),
                    gpu.step(BIAS_ADD, &[&up, &up_bias], &[rows as u32, ff_inner as u32], (rows * ff_inner) as u32),
                    gpu.step(BIAS_ADD, &[&gate, &gate_bias], &[rows as u32, ff_inner as u32], (rows * ff_inner) as u32),
                ],
            );

            assert_eq!(gpu.read(&up, rows * ff_inner), want_up, "ff_in up half moved at {tag}");
            assert_eq!(gpu.read(&gate, rows * ff_inner), want_gate, "ff_in gate half moved at {tag}");
        }
    }

    /// The block stack must not read anything back to the host.
    ///
    /// A forward's device->host readbacks are fixed by its structure, not by
    /// its depth: the embed preamble's transpose, the block stack's final
    /// residual, and the head epilogue's output. THREE, whatever
    /// `num_layers` is. Before the `ff_in` weight split, `block_fwd` also
    /// read its fused FFN activation back per block, so the count was
    /// `3 + num_layers` and each of those reads drained the queue.
    ///
    /// This asserts the structure, which the numbers cannot: a timing that
    /// improved would not distinguish "the round trip is gone" from "the
    /// round trip is still there and the machine was quiet".
    #[test]
    fn a_forward_reads_back_three_times_whatever_the_depth() {
        let cfg = DitConfig::tiny();
        assert!(cfg.num_layers >= 2, "a depth-independent count needs depth to vary");
        let gpu = Gpu::new_cpu(PIPELINES);
        let w = dit_train::random_weights(&cfg, 0xD19);
        let length = 3usize;
        let mut r = Lcg::new(0xD19 ^ 0xBEEF);
        let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
        let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
        let res = Resident::new(&gpu, &cfg, &w, length);

        let before = gpu.stats().expect("the CPU backend counts device ops").readbacks;
        let _ = forward_resident(&gpu, &cfg, &w, &res, &latents, &condition, 0.5, length);
        let after = gpu.stats().expect("the CPU backend counts device ops").readbacks;
        assert_eq!(after - before, 3, "forward_resident readbacks (num_layers={})", cfg.num_layers);
    }

    /// Rebinding a `Resident` to a new chunk length must be
    /// indistinguishable from having built it at that length in the first
    /// place - the whole point of keeping the blocks across chunks is that
    /// only the RoPE tables genuinely depend on `length`.
    ///
    /// Bit-for-bit again, and for the same reason as
    /// [`a_reused_resident_matches_a_fresh_upload_bit_for_bit`]: the two
    /// paths dispatch identical kernels over identical bytes, so anything
    /// less than equality would mean the tables (or the blocks) really did
    /// change.
    #[test]
    fn a_rebound_resident_matches_one_built_at_that_length_bit_for_bit() {
        let cfg = DitConfig::tiny();
        let gpu = Gpu::new_cpu(PIPELINES);
        let w = dit_train::random_weights(&cfg, 0xD19);
        let mut r = Lcg::new(0xD19 ^ 0xBEEF);

        let mut res = Resident::new(&gpu, &cfg, &w, 3);
        assert_eq!(res.rows(), 4);
        for length in [5usize, 5, 2] {
            res.rebind(&gpu, &cfg, length);
            assert_eq!(res.rows(), length + 1, "rebind must move the row count with the tables");
            let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
            let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
            let fresh = Resident::new(&gpu, &cfg, &w, length);
            let a = forward_resident(&gpu, &cfg, &w, &res, &latents, &condition, 0.4, length);
            let b = forward_resident(&gpu, &cfg, &w, &fresh, &latents, &condition, 0.4, length);
            assert_eq!(a, b, "a Resident rebound to {length} drifted from one built at {length}");
        }
    }

    /// Rebinding must move the refusal with it: a `Resident` rebound to a
    /// new length must refuse its OLD one, or the tables and the row count
    /// could disagree and a stale table would go unnoticed.
    #[test]
    #[should_panic(expected = "build one Resident per chunk length")]
    fn a_rebound_resident_refuses_its_previous_chunk_length() {
        let cfg = DitConfig::tiny();
        let gpu = Gpu::new_cpu(PIPELINES);
        let w = dit_train::random_weights(&cfg, 0xD1A);
        let mut res = Resident::new(&gpu, &cfg, &w, 3);
        res.rebind(&gpu, &cfg, 5);
        let latents = vec![0.1f32; cfg.in_channels as usize * 3];
        let condition = vec![0.1f32; 3 * cfg.condition_dim as usize];
        let _ = forward_resident(&gpu, &cfg, &w, &res, &latents, &condition, 0.5, 3);
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
