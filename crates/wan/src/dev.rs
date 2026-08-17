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

use crate::block::{build_block_steps, open_device, BlockDims, BlockWeights, ModBufs, Scratch, Sel};
use crate::config::WanConfig;
use crate::model::{self, Tensors};
use crate::rope::tables;

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
    _blocks: Vec<BlockWeights>,
    _scr: Scratch,
}

impl WanDitDev {
    /// Upload every weight and record the whole stack. `taps` names block
    /// indices whose output must survive the rest of the stack (each costs one
    /// extra `[tokens, dim]` buffer).
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
        let gpu = open_device(device);
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

        let blocks: Vec<BlockWeights> =
            (0..cfg.num_layers).map(|l| BlockWeights::upload(&gpu, src, &format!("blocks.{l}"))).collect();
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
        for l in 0..cfg.num_layers {
            let out = if taps.contains(&l) {
                pool.push(gpu.storage(td));
                let i = pool.len() - 1;
                tap_idx.insert(l, i);
                i
            } else if cur == 1 {
                2
            } else {
                1
            };
            build_block_steps(&gpu, &mut steps, &sel, &blocks[l], &mods[l], &pool[cur], &pool[out], &scr, &cos, &sin, &ctx, d, tokens);
            cur = out;
        }

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
