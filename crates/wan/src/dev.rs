// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device-resident Wan DiT: every block's weights uploaded once, the whole
//! stack recorded as ONE graph, one submit per forward.
//!
//! [`crate::model::WanDit`] rebuilds a block per layer and round-trips the
//! token slab through the host between them - the right shape for a parity
//! reference, and the wrong shape for a 50-step sampling loop that would then
//! re-upload 5.7 GB of weights fifty times. This engine pays that once at build
//! and keeps the residual on the device across the stack.
//!
//! The host still owns the cheap ends (patchify, the timestep MLPs, the text
//! MLP, the head) and calls the SAME helpers the reference does, so the two
//! forwards cannot drift on a convention. The text encoding is embedded once by
//! [`WanDitDev::set_context`] rather than per forward, because a sampler holds
//! it fixed across every step.
//!
//! Residual buffers alternate between two slabs, so the stack costs two
//! `[tokens, dim]` buffers, not `num_layers + 1` - at 32,760 tokens that is the
//! difference between 400 MB and 6.2 GB. A block named in `taps` gets a
//! dedicated output buffer instead, which is what a parity test bisects with;
//! tapping all 30 opts back into the large footprint deliberately.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::block::{
    build_block_steps, build_block_steps_q, BlockDims, BlockWeights, ModBufs, QBlockWeights, QScratch, QTier, Scratch, Sel,
    KERNELS,
};
use crate::config::WanConfig;
use crate::model::{self, Tensors};
use crate::rope::tables;

/// Storage dtype for the DiT's linear weights. Compute stays fp32 throughout
/// (this repo's core-compute-only invariant) - what varies is how each weight
/// is STORED and, for the quantized tiers, which GEMM kernel dequantizes it.
///
/// This is a MEMORY play, not a speed one: quantizing a forward pass's
/// arithmetic does not pay for itself once attention (not the GEMMs) is the
/// dominant cost. What int8 storage buys instead is fit: at fp32 the largest
/// variant's weights do not fit one card; packed int8 with per-row fp32
/// scales does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanDtype {
    /// Every weight resident fp32 - today's only path, unchanged.
    F32,
    /// Source values widened to fp32 on upload (a GGUF's native F16/BF16
    /// tensors dequantize to fp32 through `with_tensor` either way; this
    /// variant exists so a caller can name the storage precision without
    /// implying a compute change - Pascal's fp16 ARITHMETIC rate is ~1/64 of
    /// fp32's, so this engine never computes in fp16). Behaves identically to
    /// `F32` for a [`checkpoint::TensorSource`], since every source here
    /// already decodes to fp32 before this crate sees it.
    F16,
    /// Packed int8 (4 lanes/`u32`) with a per-output-row fp32 scale, DP4A
    /// GEMM. GPU-only: `matmul_i8_dyn.wgsl` is a multi-barrier workgroup
    /// kernel with no CPU-JIT lowering (see its own `@cpu no`).
    Int8,
    /// Packed int4 (8 lanes/`u32`, W4A8: activations stay int8) with a
    /// per-output-row fp32 scale. Correctness-only - no GEMM tuning attempted
    /// here (see `crates/model/src/int4.rs`'s own doc on the naive kernel).
    /// Unlike `Int8`, `matmul_q4_dyn.wgsl` is one-thread-per-output with no
    /// workgroup barrier, so it runs on the CPU JIT too.
    Int4,
}

impl WanDtype {
    /// Parse `--dit-dtype`/`BRAIN_WAN_DIT_DTYPE`'s spelling.
    pub fn from_name(s: &str) -> Result<WanDtype, String> {
        match s {
            "f32" => Ok(WanDtype::F32),
            "f16" => Ok(WanDtype::F16),
            "int8" | "i8" => Ok(WanDtype::Int8),
            "int4" | "i4" => Ok(WanDtype::Int4),
            other => Err(format!("wan: unknown --dit-dtype {other:?} (f32, f16, int8, int4)")),
        }
    }

    /// The short spelling `resident_wan`'s instance key embeds.
    pub fn key(self) -> &'static str {
        match self {
            WanDtype::F32 => "f32",
            WanDtype::F16 => "f16",
            WanDtype::Int8 => "int8",
            WanDtype::Int4 => "int4",
        }
    }

    /// Whether this tier's GEMM has no CPU-JIT lowering (DP4A's multi-barrier
    /// workgroup), so a `device == Some("cpu")` build must be refused early
    /// with a clear error rather than failing deep inside kernel dispatch.
    pub fn gpu_only(self) -> bool {
        matches!(self, WanDtype::Int8)
    }
}

/// The non-block tensors, kept on the host because the ops that read them are
/// a rounding error of the forward and every one of them is shared with
/// [`crate::model`]'s reference path.
fn host_tensor_names() -> Vec<String> {
    let mut v = vec![
        "patch_embedding.weight",
        "patch_embedding.bias",
        "text_embedding.0.weight",
        "text_embedding.0.bias",
        "text_embedding.2.weight",
        "text_embedding.2.bias",
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        "time_projection.1.weight",
        "time_projection.1.bias",
        "head.head.weight",
        "head.head.bias",
        "head.modulation",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    v.sort();
    v
}

/// The resident block-weight storage this engine was built with - the payload
/// [`WanDtype`] names. Kept as an enum (not two optional fields) so a built
/// engine can never end up holding both, or neither.
#[allow(dead_code)] // held only to keep the resident buffers alive for `steps`, same as `_scr`
enum Blocks {
    Dense(Vec<BlockWeights>),
    /// Quantized-storage tier (int8/int4): the packed blocks plus the
    /// dynamic-activation-quant scratch [`build_block_steps_q`] needs, kept
    /// alive for the same reason `_scr` is - referenced by the recorded steps.
    Quant(Vec<QBlockWeights>, QScratch),
}

/// A Wan DiT with the block stack resident on one device.
pub struct WanDitDev {
    gpu: Gpu,
    cfg: WanConfig,
    d: BlockDims,
    grid: (u32, u32, u32),
    tokens: u32,
    steps: Vec<Step>,
    x0: DeviceBuffer,
    /// Held only to keep the RoPE tables alive for the recorded graph: they are
    /// uploaded once in `build` and never touched again, so nothing reads these
    /// after construction (same reason as `_blocks` / `_scr` below).
    _cos: DeviceBuffer,
    _sin: DeviceBuffer,
    ctx: DeviceBuffer,
    /// Index into `pool` holding the last block's output.
    final_idx: usize,
    pool: Vec<DeviceBuffer>,
    tap_idx: HashMap<usize, usize>,
    mods: Vec<ModBufs>,
    host: Tensors,
    _blocks: Blocks,
    _scr: Scratch,
}

impl WanDitDev {
    /// [`Self::build_dtype`] at [`WanDtype::F32`] - the original entry point,
    /// signature UNCHANGED so every existing caller (the pipeline, the
    /// parity tests, `wan_bench`) keeps compiling without touching those
    /// files.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        cfg: &WanConfig,
        src: &dyn checkpoint::TensorSource,
        f: u32,
        h: u32,
        w: u32,
        device: Option<&str>,
        taps: &[usize],
    ) -> WanDitDev {
        Self::build_dtype(cfg, src, f, h, w, device, taps, WanDtype::F32)
    }

    /// Upload every weight in the given storage `dtype` and record the whole
    /// stack. `taps` names block indices whose output must survive the rest
    /// of the stack (each costs one extra `[tokens, dim]` buffer).
    ///
    /// `dtype`'s only effect is which weight upload / block-step builder runs
    /// per block ([`BlockWeights`]/[`build_block_steps`] for
    /// `F32`/`F16`, [`QBlockWeights`]/[`build_block_steps_q`] for
    /// `Int8`/`Int4`) - everything else (RoPE tables, the host tensors, the
    /// residual pool, the taps) is dtype-agnostic.
    #[allow(clippy::too_many_arguments)]
    pub fn build_dtype(
        cfg: &WanConfig,
        src: &dyn checkpoint::TensorSource,
        f: u32,
        h: u32,
        w: u32,
        device: Option<&str>,
        taps: &[usize],
        dtype: WanDtype,
    ) -> WanDitDev {
        if dtype.gpu_only() && device == Some("cpu") {
            panic!("wan: --dit-dtype {} has no CPU-JIT lowering (DP4A) - build on a GPU device", dtype.key());
        }
        let gpu = Gpu::open(device, &KERNELS);
        let d = BlockDims::new(cfg);
        let sel = Sel::new(&gpu);
        let grid = model::patch_grid(cfg, f, h, w);
        let tokens = grid.0 * grid.1 * grid.2;
        let td = (tokens as u64) * (d.dim as u64);

        let mut host: Tensors = HashMap::new();
        for name in host_tensor_names() {
            let data = crate::block::read_named(src, &name);
            host.insert(name, (vec![data.len()], data));
        }

        let mods: Vec<ModBufs> =
            (0..cfg.num_layers).map(|l| ModBufs::new(&gpu, src, &format!("blocks.{l}"), d.dim)).collect();

        let x0 = gpu.storage(td);
        // The RoPE tables are a pure function of the (f, h, w) patch grid, and
        // the grid is fixed for the life of this engine - one latent volume is
        // exactly what `build` records a graph for. So they are built and
        // uploaded ONCE here rather than recomputed and re-uploaded on every
        // forward: at 14k tokens that is ~1.8 M sin/cos pairs and ~14 MB of
        // host-to-device traffic per forward, times 2 forwards a step.
        let r = tables(cfg, grid.0, grid.1, grid.2);
        let cos = gpu.storage((tokens as u64) * (d.head_dim as u64) / 2);
        let sin = gpu.storage((tokens as u64) * (d.head_dim as u64) / 2);
        gpu.write_f32(&cos, &r.cos);
        gpu.write_f32(&sin, &r.sin);
        let ctx = gpu.storage((d.text_len as u64) * (d.dim as u64));
        let scr = Scratch::new(&gpu, d, tokens);

        // pool[0] is the input; 1 and 2 are the alternating residual slabs;
        // anything past that is a dedicated tap buffer.
        let mut pool = vec![x0.clone(), gpu.storage(td), gpu.storage(td)];
        let mut tap_idx = HashMap::new();
        let mut steps = Vec::new();
        let mut cur = 0usize;

        let blocks = match dtype {
            WanDtype::F32 | WanDtype::F16 => {
                let blocks: Vec<BlockWeights> =
                    (0..cfg.num_layers).map(|l| BlockWeights::upload(&gpu, src, &format!("blocks.{l}"))).collect();
                for l in 0..cfg.num_layers {
                    let out = Self::next_out(&mut pool, &mut tap_idx, taps, l, cur, td, &gpu);
                    build_block_steps(&gpu, &mut steps, &sel, &blocks[l], &mods[l], &pool[cur], &pool[out], &scr, &cos, &sin, &ctx, d, tokens);
                    cur = out;
                }
                Blocks::Dense(blocks)
            }
            WanDtype::Int8 | WanDtype::Int4 => {
                let tier = if dtype == WanDtype::Int8 { QTier::Int8 } else { QTier::Int4 };
                let blocks: Vec<QBlockWeights> =
                    (0..cfg.num_layers).map(|l| QBlockWeights::upload(&gpu, src, &format!("blocks.{l}"), d, tier)).collect();
                let qscr = QScratch::new(&gpu, d, tokens);
                for l in 0..cfg.num_layers {
                    let out = Self::next_out(&mut pool, &mut tap_idx, taps, l, cur, td, &gpu);
                    build_block_steps_q(&gpu, &mut steps, &sel, tier, &blocks[l], &mods[l], &pool[cur], &pool[out], &scr, &qscr, &cos, &sin, &ctx, d, tokens);
                    cur = out;
                }
                Blocks::Quant(blocks, qscr)
            }
        };

        WanDitDev {
            gpu,
            cfg: cfg.clone(),
            d,
            grid,
            tokens,
            steps,
            x0,
            _cos: cos,
            _sin: sin,
            ctx,
            final_idx: cur,
            pool,
            tap_idx,
            mods,
            host,
            _blocks: blocks,
            _scr: scr,
        }
    }

    /// Which pool slot block `l`'s output lands in - a tapped block gets a
    /// fresh dedicated buffer, everything else alternates between the two
    /// residual slabs. Shared by both the dense and quantized build loops so
    /// the pool/tap bookkeeping cannot drift between them.
    fn next_out(pool: &mut Vec<DeviceBuffer>, tap_idx: &mut HashMap<usize, usize>, taps: &[usize], l: usize, cur: usize, td: u64, gpu: &Gpu) -> usize {
        if taps.contains(&l) {
            pool.push(gpu.storage(td));
            let i = pool.len() - 1;
            tap_idx.insert(l, i);
            i
        } else if cur == 1 {
            2
        } else {
            1
        }
    }

    pub fn tokens(&self) -> u32 {
        self.tokens
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The recorded block-stack graph - one submit's worth of dispatches.
    /// A profiler groups these by kernel kind; nothing else should need them.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Embed the text encoding and upload it. Call once per prompt: a sampler
    /// holds the context fixed across every step.
    pub fn set_context(&self, context: &[f32], rows: usize) {
        let emb = model::text_embed(&self.cfg, &self.host, context, rows);
        self.set_context_embed(&emb);
    }

    /// Upload an ALREADY-embedded context, `[text_len · dim]` - what
    /// [`crate::model::text_embed`] returns.
    ///
    /// Classifier-free guidance alternates between two contexts on every step,
    /// and `text_embedding` is a `[512, 4096] x [4096, dim]` plus a
    /// `[512, dim] x [dim, dim]` on the HOST - ~9 GFLOP a call at 1.3B widths,
    /// which is real time next to a device forward and is the same answer every
    /// step. Embedding each prompt once and re-uploading is what keeps it out
    /// of the loop.
    pub fn set_context_embed(&self, emb: &[f32]) {
        assert_eq!(emb.len(), self.d.text_len as usize * self.d.dim as usize, "embedded context length");
        self.gpu.write_f32(&self.ctx, emb);
    }

    /// One forward. `latent` is `[C·F·H·W]`; the caller must have supplied the
    /// context first. Returns `[C_out·F·H·W]`.
    pub fn forward(&self, latent: &[f32], t: f32) -> Vec<f32> {
        let cfg = &self.cfg;
        let (f, h, w) = (self.grid.0, self.grid.1 * cfg.patch_size.1 as u32, self.grid.2 * cfg.patch_size.2 as u32);
        let tokens = model::embed_tokens(cfg, &self.host, latent, f, h, w);
        let (e, e0) = model::timestep_cond(cfg, &self.host, t);

        // `cos`/`sin` are NOT written here: the grid cannot change after
        // `build`, so they were uploaded once there.
        self.gpu.write_f32(&self.x0, &tokens);
        for m in &self.mods {
            m.upload(&self.gpu, &e0, self.d.dim as usize);
        }
        self.gpu.submit(&[], &self.steps);
        let x = self.gpu.read(&self.pool[self.final_idx], (self.tokens * self.d.dim) as usize);
        model::postprocess(cfg, &self.host, &x, &e, self.grid)
    }

    /// A tapped block's output `[tokens · dim]`, valid after a [`Self::forward`].
    pub fn read_tap(&self, block: usize) -> Option<Vec<f32>> {
        let i = *self.tap_idx.get(&block)?;
        Some(self.gpu.read(&self.pool[i], (self.tokens * self.d.dim) as usize))
    }

    /// The block-stack output before the head, `[tokens · dim]`.
    pub fn read_last(&self) -> Vec<f32> {
        self.gpu.read(&self.pool[self.final_idx], (self.tokens * self.d.dim) as usize)
    }
}
