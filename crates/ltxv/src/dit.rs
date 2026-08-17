// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The full tiny video-only DiT forward, host-orchestrated over
//! [`crate::block::LtxBlock`] - `ltx_core.model.transformer.model.LTXModel.
//! forward`'s video-only path.
//!
//! ## What runs where, and why (mirrors `wan::model`'s own split)
//!
//! Everything OUTSIDE the block stack - `patchify_proj`, the keyframe
//! absolute-position add, the per-token timestep embedding + adaLN-single
//! raw table, RoPE table construction, and the output stage (LayerNorm +
//! modulation + `proj_out`) - runs as plain host math, the same way
//! `wan::model::preprocess`/`postprocess` keep everything outside
//! `WanBlock` on the host. Only the per-block internals (self-attention,
//! text cross-attention, the FFN, and the per-token modulation combine) go
//! through the GPU dispatch graph in [`crate::block`] - that is where this
//! repo's shared attention/RoPE dispatch seam lives and where reuse
//! actually matters.
//!
//! ## The weights loader
//!
//! [`load_tiny_weights`] is a simple name-keyed `safetensors` load of
//! `dit_tiny_weights.safetensors` - the tiny golden's OWN dumped weights,
//! already in the exact canonical name space this crate reads directly (no
//! renaming table, unlike `crate::import`'s VAE loader). A full two-way-
//! coverage importer against the REAL 22B/4349-tensor checkpoint is
//! explicitly OUT of scope for this milestone (recorded as a gap on this
//! port's own roadmap ledger - the op sequence is proven here, real-weight
//! import is a later milestone).

use gpu_core::Gpu;
use vae::blocks::Tensors;

use crate::block::{open_device, BlockTaps, LtxBlock};
use crate::config::LtxDitConfig;
use crate::rope::{ltx_rope_tables, LtxRopeTables};

fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("ltxv dit: missing weight {name}")).1
}

/// Load the tiny golden's own dumped weights
/// (`dit_tiny_weights.safetensors`) - a flat name -> `(shape, data)` map,
/// no renaming (the file is already in the canonical name space this crate
/// reads directly - see this module's doc).
pub fn load_tiny_weights(path: &str) -> Tensors {
    let raw = checkpoint::safetensors::read(path).unwrap_or_else(|e| panic!("ltxv dit: {e}"));
    raw.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

/// `out[r,o] = b[o] + Σ_i x[r,i]·w[o,i]`, `w` row-major `[out_dim, in_dim]` -
/// plain `nn.Linear`, sequential (this milestone's token/dim counts are far
/// below where `wan::model::linear`'s row-parallel split would matter).
fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..r * in_dim + in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b.map(|b| b[o]).unwrap_or(0.0);
            for (xi, wi) in xr.iter().zip(wr) {
                acc += xi * wi;
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_slice(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| silu(v)).collect()
}

/// `torch.nn.LayerNorm(elementwise_affine=False)` - the model's `norm_out`
/// (see `crate::dit`'s doc: LayerNorm, NOT RMSNorm, for the output stage).
fn layernorm_noaffine(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..r * dim + dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = (row[d] - mean) * inv;
        }
    }
    out
}

/// Every tap a parity test bisects with - the golden's own tensor names.
pub struct DitTaps {
    /// `[heads, T, head_dim/2]`.
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    /// `[T, 9*dim]` - the raw `adaln_single.linear` output, BEFORE any
    /// block's own `scale_shift_table` is added.
    pub adaln_table: Vec<f32>,
    /// `[T, dim]` - `adaln_single.emb`'s output (the PixArt MLP, before the
    /// `9*dim` linear).
    pub embedded_timestep: Vec<f32>,
    pub b0_attn1_out: Vec<f32>,
    pub b0_attn2_out: Vec<f32>,
    pub b0_ff_out: Vec<f32>,
    /// One `[T, dim]` entry per layer, in order.
    pub block_out: Vec<Vec<f32>>,
    /// `[T, out_channels]` - the model's final output.
    pub out: Vec<f32>,
}

/// The tiny video-only DiT, weights resident on the host (device residency
/// is per-block, see `LtxBlock::on`).
pub struct LtxDit {
    cfg: LtxDitConfig,
    w: Tensors,
    device: Option<String>,
}

impl LtxDit {
    pub fn new(cfg: LtxDitConfig, weights: Tensors, device: Option<&str>) -> LtxDit {
        cfg.assert_supported();
        assert_eq!(cfg.cross_attention_dim, cfg.inner_dim, "M3 assumes cross_attention_dim == inner_dim (caption_projection=None) - see config.rs's doc");
        LtxDit { cfg, w: weights, device: device.map(str::to_string) }
    }

    pub fn config(&self) -> &LtxDitConfig {
        &self.cfg
    }

    /// Upload one head's `[T, head_dim/2]` `(cos, sin)` slice as device
    /// buffers - shared by every block (RoPE tables do not vary per layer).
    fn upload_rope(gpu: &Gpu, rope: &LtxRopeTables) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
        let mut cos_bufs = Vec::with_capacity(rope.heads);
        let mut sin_bufs = Vec::with_capacity(rope.heads);
        for h in 0..rope.heads {
            let (c, s) = rope.head(h);
            let cb = gpu.storage(c.len() as u64);
            gpu.write_f32(&cb, c);
            let sb = gpu.storage(s.len() as u64);
            gpu.write_f32(&sb, s);
            cos_bufs.push(cb);
            sin_bufs.push(sb);
        }
        (cos_bufs, sin_bufs)
    }

    /// One forward pass, replaying the golden's own inputs.
    ///
    /// `latent`: `[T, in_channels]`. `timesteps`: `[T]` per-token
    /// `denoise_mask*sigma` (BEFORE `timestep_scale_multiplier`, applied
    /// inside). `positions`: `[3, T, 2]` row-major `[start,end)` RoPE bounds
    /// (see `crate::rope`'s doc). `keyframes_mask`: `[T]`, non-zero marks a
    /// keyframe token. `context`: `[context_len, cross_attention_dim]` raw
    /// text context (each block modulates it independently - see
    /// `crate::block`'s doc).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], timesteps: &[f32], positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, t: usize) -> DitTaps {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim as usize;
        assert_eq!(latent.len(), t * cfg.in_channels as usize);
        assert_eq!(timesteps.len(), t);
        assert_eq!(keyframes_mask.len(), t);
        assert_eq!(context.len(), context_len * cfg.cross_attention_dim as usize);

        // ---- patchify_proj + keyframes embedding --------------------------
        let pw = tget(&self.w, "patchify_proj.weight");
        let pb = tget(&self.w, "patchify_proj.bias");
        let mut x = linear(latent, t, cfg.in_channels as usize, pw, Some(pb), dim);
        if cfg.use_keyframes_abs_pos_embedding {
            let kf = tget(&self.w, "keyframes_abs_pos_embedding");
            for ti in 0..t {
                if keyframes_mask[ti] > 0.0 {
                    for d in 0..dim {
                        x[ti * dim + d] += kf[d];
                    }
                }
            }
        }

        // ---- per-token timestep embedding + adaLN-single raw table --------
        let w1 = tget(&self.w, "adaln_single.emb.timestep_embedder.linear_1.weight");
        let b1 = tget(&self.w, "adaln_single.emb.timestep_embedder.linear_1.bias");
        let w2 = tget(&self.w, "adaln_single.emb.timestep_embedder.linear_2.weight");
        let b2 = tget(&self.w, "adaln_single.emb.timestep_embedder.linear_2.bias");
        let mut embedded_timestep = vec![0f32; t * dim];
        for ti in 0..t {
            let ts_scaled = timesteps[ti] * cfg.timestep_scale_multiplier as f32;
            let e = dit::timestep::pixart_timestep_embed(ts_scaled, 256, w1, b1, dim, w2, b2, dim, 10000.0);
            embedded_timestep[ti * dim..ti * dim + dim].copy_from_slice(&e);
        }
        let wl = tget(&self.w, "adaln_single.linear.weight");
        let bl = tget(&self.w, "adaln_single.linear.bias");
        let adaln_rows = cfg.adaln_rows() as usize;
        let adaln_table = linear(&silu_slice(&embedded_timestep), t, dim, wl, Some(bl), adaln_rows * dim);

        // ---- RoPE tables ----------------------------------------------------
        let rope = ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, cfg.positional_embedding_max_pos, positions, t);

        // ---- block stack ------------------------------------------------------
        let gpu = open_device(self.device.as_deref());
        let (cos_bufs, sin_bufs) = Self::upload_rope(&gpu, &rope);

        let mut block_out = Vec::with_capacity(cfg.num_layers as usize);
        let mut b0: Option<BlockTaps> = None;
        for l in 0..cfg.num_layers {
            let blk = LtxBlock::on(gpu.share(), cfg, &self.w, &format!("transformer_blocks.{l}"), t as u32, context_len as u32);
            let (out, taps) = blk.forward(&x, &adaln_table, context, &cos_bufs, &sin_bufs, t as u32);
            if l == 0 {
                b0 = Some(taps);
            }
            x = out;
            block_out.push(x.clone());
        }
        let b0 = b0.expect("num_layers must be >= 1");

        // ---- output stage: LayerNorm(no affine) -> modulate -> proj_out ----
        let sst = tget(&self.w, "scale_shift_table"); // [2, dim]: [shift, scale]
        let mut shift = vec![0f32; t * dim];
        let mut one_plus_scale = vec![0f32; t * dim];
        for ti in 0..t {
            for d in 0..dim {
                shift[ti * dim + d] = sst[d] + embedded_timestep[ti * dim + d];
                one_plus_scale[ti * dim + d] = 1.0 + sst[dim + d] + embedded_timestep[ti * dim + d];
            }
        }
        let normed = layernorm_noaffine(&x, t, dim, cfg.norm_eps);
        let mut xo = vec![0f32; t * dim];
        for i in 0..t * dim {
            xo[i] = normed[i] * one_plus_scale[i] + shift[i];
        }
        let pw2 = tget(&self.w, "proj_out.weight");
        let pb2 = tget(&self.w, "proj_out.bias");
        let out = linear(&xo, t, dim, pw2, Some(pb2), cfg.out_channels as usize);

        DitTaps {
            rope_cos: rope.cos,
            rope_sin: rope.sin,
            adaln_table,
            embedded_timestep,
            b0_attn1_out: b0.attn1_out,
            b0_attn2_out: b0.attn2_out,
            b0_ff_out: b0.ff_out,
            block_out,
            out,
        }
    }
}
