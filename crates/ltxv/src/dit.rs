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

use gpu_core::{DeviceBuffer, Gpu};
use vae::blocks::Tensors;

use crate::block::{open_device, BlockTaps, LtxAvBlock, LtxBlock};
use crate::config::{LtxAudioDitConfig, LtxAvDitConfig, LtxDitConfig};
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

/// Every tensor a [`LtxDit`] forward reads, name + shape, derived from
/// `cfg` - the canonical name space [`load_tiny_weights`] and
/// [`crate::block::LtxBlock`]'s `tget` calls already use. Exists so
/// [`random_tiny_weights`] (this milestone's pipeline) and any future
/// real-checkpoint importer enumerate the SAME list a forward actually
/// reads, the same "manifest drives both sides" discipline
/// `crate::vae3d::LtxVaeConfig::tensor_manifest` uses for the VAE.
pub fn dit_tensor_manifest(cfg: &LtxDitConfig) -> Vec<(String, Vec<usize>)> {
    let dim = cfg.inner_dim as usize;
    let mut m: Vec<(String, Vec<usize>)> = vec![
        ("patchify_proj.weight".into(), vec![dim, cfg.in_channels as usize]),
        ("patchify_proj.bias".into(), vec![dim]),
        ("adaln_single.emb.timestep_embedder.linear_1.weight".into(), vec![dim, 256]),
        ("adaln_single.emb.timestep_embedder.linear_1.bias".into(), vec![dim]),
        ("adaln_single.emb.timestep_embedder.linear_2.weight".into(), vec![dim, dim]),
        ("adaln_single.emb.timestep_embedder.linear_2.bias".into(), vec![dim]),
        ("adaln_single.linear.weight".into(), vec![cfg.adaln_rows() as usize * dim, dim]),
        ("adaln_single.linear.bias".into(), vec![cfg.adaln_rows() as usize * dim]),
        ("scale_shift_table".into(), vec![2, dim]),
        ("proj_out.weight".into(), vec![cfg.out_channels as usize, dim]),
        ("proj_out.bias".into(), vec![cfg.out_channels as usize]),
    ];
    if cfg.use_keyframes_abs_pos_embedding {
        m.push(("keyframes_abs_pos_embedding".into(), vec![dim]));
    }
    for l in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{l}");
        for attn in ["attn1", "attn2"] {
            for proj in ["to_q", "to_k", "to_v"] {
                m.push((format!("{p}.{attn}.{proj}.weight"), vec![dim, dim]));
                m.push((format!("{p}.{attn}.{proj}.bias"), vec![dim]));
            }
            m.push((format!("{p}.{attn}.to_out.0.weight"), vec![dim, dim]));
            m.push((format!("{p}.{attn}.to_out.0.bias"), vec![dim]));
            m.push((format!("{p}.{attn}.q_norm.weight"), vec![dim]));
            m.push((format!("{p}.{attn}.k_norm.weight"), vec![dim]));
        }
        m.push((format!("{p}.ff.net.0.proj.weight"), vec![4 * dim, dim]));
        m.push((format!("{p}.ff.net.2.weight"), vec![dim, 4 * dim]));
        m.push((format!("{p}.scale_shift_table"), vec![cfg.adaln_rows() as usize, dim]));
        m.push((format!("{p}.prompt_scale_shift_table"), vec![2, dim]));
    }
    m
}

/// A random, seeded set of weights at `cfg`'s exact tensor manifest - what a
/// pipeline uses when there is no real 22B checkpoint to import (this whole
/// port's status: the real DiT is 42 GB bf16 and does not exist as a
/// downloadable file this milestone's hardware could hold anyway). Biases
/// are zero; every other tensor is i.i.d. `N(0, 0.02²)`, the same
/// re-initialization std `tools/goldens/ltxv_dit_dump_reference.py` uses for
/// the class's own `torch.empty(...)`-sourced parameters - reused here for
/// every weight, not just those, because this is a WIRING smoke test, not a
/// quality claim: nothing about generation fidelity is being asserted, only
/// that noise in -> a forward -> a decodable video out, at the real op
/// sequence. Deterministic in `seed` (same seed -> bit-identical weights),
/// like every other seeded thing in this pipeline.
pub fn random_tiny_weights(cfg: &LtxDitConfig, seed: u64) -> Tensors {
    let mut rng = data::rng::Rng::new(seed);
    dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = if name.ends_with(".bias") { vec![0.0; n] } else { (0..n).map(|_| (rng.next_gaussian() * 0.02) as f32).collect() };
            (name, (shape, data))
        })
        .collect()
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

/// One `AdaLayerNormSingle`'s per-token raw output: the PixArt timestep MLP
/// (`{prefix}.emb.timestep_embedder.*`, `dit::timestep::pixart_timestep_
/// embed` per row) -> `SiLU` -> `{prefix}.linear` -> `[rows, coeff*dim]`.
/// `rows = timesteps_scaled.len()`; `timesteps_scaled` is ALREADY
/// `timestep_scale_multiplier`-scaled by the caller (the AV gate tables use
/// a DIFFERENT multiplier - `av_ca_timestep_scale_multiplier` - so this
/// function takes pre-scaled values rather than a multiplier itself, see
/// `crate::block`'s and `crate::config`'s docs).
///
/// Every `AdaLayerNormSingle` in this crate's AV model shares this exact
/// shape (`adaln_single`/`audio_adaln_single` at `coeff=9`;
/// `av_ca_{video,audio}_scale_shift_adaln_single` at `coeff=4`;
/// `av_ca_{a2v,v2a}_gate_adaln_single` at `coeff=1`, `rows=1` - a single
/// scalar sigma in) - factored out here instead of four near-identical
/// inline copies. Returns `(raw_linear_table, embedded_timestep)`.
fn ada_layer_norm_single(w: &Tensors, prefix: &str, timesteps_scaled: &[f32], dim: usize, coeff: usize) -> (Vec<f32>, Vec<f32>) {
    let w1 = tget(w, &format!("{prefix}.emb.timestep_embedder.linear_1.weight"));
    let b1 = tget(w, &format!("{prefix}.emb.timestep_embedder.linear_1.bias"));
    let w2 = tget(w, &format!("{prefix}.emb.timestep_embedder.linear_2.weight"));
    let b2 = tget(w, &format!("{prefix}.emb.timestep_embedder.linear_2.bias"));
    let rows = timesteps_scaled.len();
    let mut embedded = vec![0f32; rows * dim];
    for (ti, &t) in timesteps_scaled.iter().enumerate() {
        let e = dit::timestep::pixart_timestep_embed(t, 256, w1, b1, dim, w2, b2, dim, 10000.0);
        embedded[ti * dim..ti * dim + dim].copy_from_slice(&e);
    }
    let wl = tget(w, &format!("{prefix}.linear.weight"));
    let bl = tget(w, &format!("{prefix}.linear.bias"));
    let table = linear(&silu_slice(&embedded), rows, dim, wl, Some(bl), coeff * dim);
    (table, embedded)
}

/// The model's output stage - `LayerNorm(no affine)` -> per-token modulate
/// (this stream's OWN `[2,dim]` output table + `embedded_timestep`, NOT the
/// 9-row per-block modulation vector) -> `proj_out`. Shared between the
/// video-only and AV paths (each stream has its own weight names, passed
/// in) - `LTXModel._process_output`.
#[allow(clippy::too_many_arguments)]
fn output_stage(w: &Tensors, sst_name: &str, proj_name: &str, x: &[f32], embedded_timestep: &[f32], t: usize, dim: usize, out_channels: usize, norm_eps: f32) -> Vec<f32> {
    let sst = tget(w, sst_name); // [2, dim]: [shift, scale]
    let mut shift = vec![0f32; t * dim];
    let mut one_plus_scale = vec![0f32; t * dim];
    for ti in 0..t {
        for d in 0..dim {
            shift[ti * dim + d] = sst[d] + embedded_timestep[ti * dim + d];
            one_plus_scale[ti * dim + d] = 1.0 + sst[dim + d] + embedded_timestep[ti * dim + d];
        }
    }
    let normed = layernorm_noaffine(x, t, dim, norm_eps);
    let mut xo = vec![0f32; t * dim];
    for i in 0..t * dim {
        xo[i] = normed[i] * one_plus_scale[i] + shift[i];
    }
    let pw = tget(w, &format!("{proj_name}.weight"));
    let pb = tget(w, &format!("{proj_name}.bias"));
    linear(&xo, t, dim, pw, Some(pb), out_channels)
}

/// Upload one [`LtxRopeTables`]' per-head `[T, head_dim/2]` `(cos, sin)`
/// slices as device buffers - shared by every block (RoPE tables do not
/// vary per layer). Free function (not tied to [`LtxDit`]) so both
/// [`LtxDit::forward`] and [`LtxAvDit::forward`] - which needs FOUR
/// independent tables, not one - share it.
fn upload_rope_tables(gpu: &Gpu, rope: &LtxRopeTables) -> (Vec<DeviceBuffer>, Vec<DeviceBuffer>) {
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
        let ts_scaled: Vec<f32> = timesteps.iter().map(|&x| x * cfg.timestep_scale_multiplier as f32).collect();
        let (adaln_table, embedded_timestep) = ada_layer_norm_single(&self.w, "adaln_single", &ts_scaled, dim, cfg.adaln_rows() as usize);

        // ---- RoPE tables ----------------------------------------------------
        let rope = ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, &cfg.positional_embedding_max_pos, positions, t);

        // ---- block stack ------------------------------------------------------
        let gpu = open_device(self.device.as_deref());
        let (cos_bufs, sin_bufs) = upload_rope_tables(&gpu, &rope);

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
        let out = output_stage(&self.w, "scale_shift_table", "proj_out", &x, &embedded_timestep, t, dim, cfg.out_channels as usize, cfg.norm_eps);

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

/// Every AV tap a parity test bisects with, beyond one [`DitTaps`] per
/// stream - `tools/goldens/ltxv_av_dit_dump_reference.py`'s own tensor
/// names for the audio<->video-specific state.
pub struct AvDitTaps {
    /// Video's own taps - same fields [`LtxDit::forward`] returns.
    pub video: DitTaps,
    /// Audio's own taps - same shape, audio's dims/weights.
    pub audio: DitTaps,
    /// `[heads, Tv, audio_cross_attention_dim/heads/2]` - video's own
    /// cross-modal RoPE table (built from video's axis-0/time positions).
    pub v_cross_rope_cos: Vec<f32>,
    pub v_cross_rope_sin: Vec<f32>,
    /// Same shape at `Ta` rows - audio's own cross-modal RoPE table.
    pub a_cross_rope_cos: Vec<f32>,
    pub a_cross_rope_sin: Vec<f32>,
    /// `[Tv, 4*video.dim]` - `av_ca_video_scale_shift_adaln_single`'s raw
    /// per-token linear output.
    pub av_video_ss_table: Vec<f32>,
    /// `[Ta, 4*audio.dim]` - the audio counterpart.
    pub av_audio_ss_table: Vec<f32>,
    /// `[video.dim]` - `av_ca_a2v_gate_adaln_single`'s raw SINGLE-row linear
    /// output (driven by audio's scalar sigma).
    pub av_a2v_gate_table: Vec<f32>,
    /// `[audio.dim]` - `av_ca_v2a_gate_adaln_single`'s raw SINGLE-row linear
    /// output (driven by video's scalar sigma).
    pub av_v2a_gate_table: Vec<f32>,
    /// Block-0's raw `audio_to_video_attn`/`video_to_audio_attn` outputs,
    /// BEFORE their `*gate` multiply - same convention as `b0_attn2_out`.
    pub b0_a2v_out: Vec<f32>,
    pub b0_v2a_out: Vec<f32>,
}

/// The tiny audio+video DiT, weights resident on the host (device residency
/// is per-block, see [`LtxAvBlock::on`]) - `LTXModelType::AudioVideo`.
pub struct LtxAvDit {
    cfg: LtxAvDitConfig,
    w: Tensors,
    device: Option<String>,
}

impl LtxAvDit {
    pub fn new(cfg: LtxAvDitConfig, weights: Tensors, device: Option<&str>) -> LtxAvDit {
        cfg.assert_supported();
        LtxAvDit { cfg, w: weights, device: device.map(str::to_string) }
    }

    pub fn config(&self) -> &LtxAvDitConfig {
        &self.cfg
    }

    /// One forward pass, replaying the golden's own inputs - both streams,
    /// full bidirectional cross-attention every block.
    ///
    /// `v_*`/`a_*` mirror [`LtxDit::forward`]'s params, one set per stream
    /// (audio has no `keyframes_mask` - that feature is video-only, see
    /// `crate::config`'s doc). `v_sigma`/`a_sigma`: each stream's SCALAR
    /// sigma (`Modality.sigma`, `[1]`) - the CROSS modality's sigma is what
    /// drives the other stream's AV gate (see `crate::block`'s doc).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        v_latent: &[f32],
        v_timesteps: &[f32],
        v_positions: &[f32],
        v_keyframes_mask: &[f32],
        v_context: &[f32],
        v_context_len: usize,
        tv: usize,
        v_sigma: f32,
        a_latent: &[f32],
        a_timesteps: &[f32],
        a_positions: &[f32],
        a_context: &[f32],
        a_context_len: usize,
        ta: usize,
        a_sigma: f32,
    ) -> AvDitTaps {
        let cfg = &self.cfg;
        let vcfg = &cfg.video;
        let acfg: &LtxAudioDitConfig = &cfg.audio;
        let vdim = vcfg.inner_dim as usize;
        let adim = acfg.inner_dim as usize;
        assert_eq!(v_latent.len(), tv * vcfg.in_channels as usize);
        assert_eq!(a_latent.len(), ta * acfg.in_channels as usize);

        // ---- patchify_proj + video's keyframes embedding -------------------
        let pw = tget(&self.w, "patchify_proj.weight");
        let pb = tget(&self.w, "patchify_proj.bias");
        let mut vx = linear(v_latent, tv, vcfg.in_channels as usize, pw, Some(pb), vdim);
        if vcfg.use_keyframes_abs_pos_embedding {
            let kf = tget(&self.w, "keyframes_abs_pos_embedding");
            for ti in 0..tv {
                if v_keyframes_mask[ti] > 0.0 {
                    for d in 0..vdim {
                        vx[ti * vdim + d] += kf[d];
                    }
                }
            }
        }
        let apw = tget(&self.w, "audio_patchify_proj.weight");
        let apb = tget(&self.w, "audio_patchify_proj.bias");
        let mut ax = linear(a_latent, ta, acfg.in_channels as usize, apw, Some(apb), adim);

        // ---- per-token timestep embeddings + adaLN-single raw tables ------
        // A single model-level timestep_scale_multiplier scales BOTH
        // streams' own per-token timesteps (ltx_core...model.LTXModel has
        // only one such field).
        let v_ts_scaled: Vec<f32> = v_timesteps.iter().map(|&x| x * vcfg.timestep_scale_multiplier as f32).collect();
        let a_ts_scaled: Vec<f32> = a_timesteps.iter().map(|&x| x * vcfg.timestep_scale_multiplier as f32).collect();
        let (v_adaln_table, v_embedded_timestep) = ada_layer_norm_single(&self.w, "adaln_single", &v_ts_scaled, vdim, vcfg.adaln_rows() as usize);
        let (a_adaln_table, a_embedded_timestep) = ada_layer_norm_single(&self.w, "audio_adaln_single", &a_ts_scaled, adim, vcfg.adaln_rows() as usize);

        // ---- AV per-token scale/shift raw tables (each stream's OWN
        // per-token timesteps, same scaling as its main table) -------------
        let (av_video_ss_table, _) = ada_layer_norm_single(&self.w, "av_ca_video_scale_shift_adaln_single", &v_ts_scaled, vdim, 4);
        let (av_audio_ss_table, _) = ada_layer_norm_single(&self.w, "av_ca_audio_scale_shift_adaln_single", &a_ts_scaled, adim, 4);

        // ---- AV gate raw tables (the CROSS modality's scalar sigma,
        // av_ca_timestep_scale_multiplier-scaled - the timestep_scale_
        // multiplier the main tables use cancels out of this factor, see
        // crate::config's av_ca_timestep_scale_multiplier doc) -----------
        let a2v_gate_ts = [a_sigma * cfg.av_ca_timestep_scale_multiplier]; // video's gate <- audio's sigma
        let v2a_gate_ts = [v_sigma * cfg.av_ca_timestep_scale_multiplier]; // audio's gate <- video's sigma
        let (av_a2v_gate_table, _) = ada_layer_norm_single(&self.w, "av_ca_a2v_gate_adaln_single", &a2v_gate_ts, vdim, 1);
        let (av_v2a_gate_table, _) = ada_layer_norm_single(&self.w, "av_ca_v2a_gate_adaln_single", &v2a_gate_ts, adim, 1);

        // ---- RoPE tables: each stream's own self-attention table, plus the
        // SHARED cross-modal (A2V/V2A) time-only table, built at audio's
        // cross_attention_dim from each stream's OWN axis-0 (time)
        // positions - crate::rope's and crate::block's docs.
        let v_rope = ltx_rope_tables(vcfg.inner_dim, vcfg.num_heads, vcfg.positional_embedding_theta, &vcfg.positional_embedding_max_pos, v_positions, tv);
        let a_rope = ltx_rope_tables(acfg.inner_dim, acfg.num_heads, vcfg.positional_embedding_theta, &acfg.positional_embedding_max_pos, a_positions, ta);
        let cross_max_pos = [cfg.cross_pe_max_pos()];
        let v_axis0_positions = &v_positions[0..tv * 2]; // axis 0 = frame, video's own time axis
        let v_cross_rope = ltx_rope_tables(acfg.cross_attention_dim, acfg.num_heads, vcfg.positional_embedding_theta, &cross_max_pos, v_axis0_positions, tv);
        let a_cross_rope = ltx_rope_tables(acfg.cross_attention_dim, acfg.num_heads, vcfg.positional_embedding_theta, &cross_max_pos, a_positions, ta); // audio's own single axis IS its own positions array

        // ---- block stack ------------------------------------------------------
        let gpu = open_device(self.device.as_deref());
        let (v_cos_bufs, v_sin_bufs) = upload_rope_tables(&gpu, &v_rope);
        let (a_cos_bufs, a_sin_bufs) = upload_rope_tables(&gpu, &a_rope);
        let (v_cross_cos_bufs, v_cross_sin_bufs) = upload_rope_tables(&gpu, &v_cross_rope);
        let (a_cross_cos_bufs, a_cross_sin_bufs) = upload_rope_tables(&gpu, &a_cross_rope);

        let mut v_block_out = Vec::with_capacity(vcfg.num_layers as usize);
        let mut a_block_out = Vec::with_capacity(vcfg.num_layers as usize);
        let mut b0v: Option<BlockTaps> = None;
        let mut b0a: Option<BlockTaps> = None;
        let mut b0_a2v: Option<Vec<f32>> = None;
        let mut b0_v2a: Option<Vec<f32>> = None;
        for l in 0..vcfg.num_layers {
            let blk = LtxAvBlock::on(gpu.share(), vcfg, acfg, &self.w, &format!("transformer_blocks.{l}"), v_context_len as u32, a_context_len as u32);
            #[rustfmt::skip]
            let (vout, aout, taps) = blk.forward(
                &vx, &ax, &v_adaln_table, &a_adaln_table, v_context, a_context,
                &v_cos_bufs, &v_sin_bufs, &a_cos_bufs, &a_sin_bufs,
                &v_cross_cos_bufs, &v_cross_sin_bufs, &a_cross_cos_bufs, &a_cross_sin_bufs,
                &av_video_ss_table, &av_audio_ss_table, &av_a2v_gate_table, &av_v2a_gate_table,
                tv as u32, ta as u32,
            );
            if l == 0 {
                b0v = Some(BlockTaps { attn1_out: taps.v_attn1_out, attn2_out: taps.v_attn2_out, ff_out: taps.v_ff_out });
                b0a = Some(BlockTaps { attn1_out: taps.a_attn1_out, attn2_out: taps.a_attn2_out, ff_out: taps.a_ff_out });
                b0_a2v = Some(taps.a2v_out);
                b0_v2a = Some(taps.v2a_out);
            }
            vx = vout;
            ax = aout;
            v_block_out.push(vx.clone());
            a_block_out.push(ax.clone());
        }
        let b0v = b0v.expect("num_layers must be >= 1");
        let b0a = b0a.expect("num_layers must be >= 1");
        let b0_a2v = b0_a2v.expect("num_layers must be >= 1");
        let b0_v2a = b0_v2a.expect("num_layers must be >= 1");

        // ---- output stage: LayerNorm(no affine) -> modulate -> proj_out,
        // per stream ------------------------------------------------------
        let v_out = output_stage(&self.w, "scale_shift_table", "proj_out", &vx, &v_embedded_timestep, tv, vdim, vcfg.out_channels as usize, vcfg.norm_eps);
        let a_out = output_stage(&self.w, "audio_scale_shift_table", "audio_proj_out", &ax, &a_embedded_timestep, ta, adim, acfg.out_channels as usize, vcfg.norm_eps);

        AvDitTaps {
            video: DitTaps {
                rope_cos: v_rope.cos,
                rope_sin: v_rope.sin,
                adaln_table: v_adaln_table,
                embedded_timestep: v_embedded_timestep,
                b0_attn1_out: b0v.attn1_out,
                b0_attn2_out: b0v.attn2_out,
                b0_ff_out: b0v.ff_out,
                block_out: v_block_out,
                out: v_out,
            },
            audio: DitTaps {
                rope_cos: a_rope.cos,
                rope_sin: a_rope.sin,
                adaln_table: a_adaln_table,
                embedded_timestep: a_embedded_timestep,
                b0_attn1_out: b0a.attn1_out,
                b0_attn2_out: b0a.attn2_out,
                b0_ff_out: b0a.ff_out,
                block_out: a_block_out,
                out: a_out,
            },
            v_cross_rope_cos: v_cross_rope.cos,
            v_cross_rope_sin: v_cross_rope.sin,
            a_cross_rope_cos: a_cross_rope.cos,
            a_cross_rope_sin: a_cross_rope.sin,
            av_video_ss_table,
            av_audio_ss_table,
            av_a2v_gate_table,
            av_v2a_gate_table,
            b0_a2v_out: b0_a2v,
            b0_v2a_out: b0_v2a,
        }
    }
}
