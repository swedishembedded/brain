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
    cos: DeviceBuffer,
    sin: DeviceBuffer,
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
        let cos = gpu.storage((tokens as u64) * (d.head_dim as u64) / 2);
        let sin = gpu.storage((tokens as u64) * (d.head_dim as u64) / 2);
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
            cos,
            sin,
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

    /// Embed the text encoding and upload it. Call once per prompt: a sampler
    /// holds the context fixed across every step.
    pub fn set_context(&self, context: &[f32], rows: usize) {
        let emb = model::text_embed(&self.cfg, &self.host, context, rows);
        self.gpu.write_f32(&self.ctx, &emb);
    }

    /// One forward. `latent` is `[C·F·H·W]`; the caller must have supplied the
    /// context first. Returns `[C_out·F·H·W]`.
    pub fn forward(&self, latent: &[f32], t: f32) -> Vec<f32> {
        let cfg = &self.cfg;
        let (f, h, w) = (self.grid.0, self.grid.1 * cfg.patch_size.1 as u32, self.grid.2 * cfg.patch_size.2 as u32);
        let tokens = model::embed_tokens(cfg, &self.host, latent, f, h, w);
        let (e, e0) = model::timestep_cond(cfg, &self.host, t);
        let r = tables(cfg, self.grid.0, self.grid.1, self.grid.2);

        self.gpu.write_f32(&self.x0, &tokens);
        self.gpu.write_f32(&self.cos, &r.cos);
        self.gpu.write_f32(&self.sin, &r.sin);
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
