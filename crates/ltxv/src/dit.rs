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

use std::cell::RefCell;
use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use model::Shard;
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
        // Real checkpoint shape `[1, dim]` (torch order, GGUF `ne` reversed -
        // confirmed by range-reading the real header), not the `[dim, 1]`
        // an earlier paraphrase of this field transposed.
        m.push(("keyframes_abs_pos_embedding".into(), vec![1, dim]));
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

/// Push one `Attention` module's 12 tensors (`q_norm`/`k_norm`/
/// `to_gate_logits.{weight,bias}`/`to_q`/`to_k`/`to_v`/`to_out.0`, each
/// `{weight,bias}` pair except the norms and the gate bias) - shared by
/// every attention in [`av_dit_tensor_manifest`] (self-, text-cross-, and
/// the two audio<->video cross-attention modules), since all four are the
/// SAME `Attention` class at different `(q_dim, kv_dim, inner_dim)` triples
/// (`crate::block::attention`'s doc has the exact shape algebra this
/// mirrors - confirmed against the real header: `to_out.0`'s output width
/// and `to_gate_logits`' input width both equal `q_dim`, never `inner_dim`,
/// for the audio<->video directions where the two differ).
fn push_attn(m: &mut Vec<(String, Vec<usize>)>, prefix: &str, q_dim: usize, kv_dim: usize, inner_dim: usize) {
    m.push((format!("{prefix}.q_norm.weight"), vec![inner_dim]));
    m.push((format!("{prefix}.k_norm.weight"), vec![inner_dim]));
    m.push((format!("{prefix}.to_gate_logits.weight"), vec![32, q_dim]));
    m.push((format!("{prefix}.to_gate_logits.bias"), vec![32]));
    m.push((format!("{prefix}.to_q.weight"), vec![inner_dim, q_dim]));
    m.push((format!("{prefix}.to_q.bias"), vec![inner_dim]));
    m.push((format!("{prefix}.to_k.weight"), vec![inner_dim, kv_dim]));
    m.push((format!("{prefix}.to_k.bias"), vec![inner_dim]));
    m.push((format!("{prefix}.to_v.weight"), vec![inner_dim, kv_dim]));
    m.push((format!("{prefix}.to_v.bias"), vec![inner_dim]));
    m.push((format!("{prefix}.to_out.0.weight"), vec![q_dim, inner_dim]));
    m.push((format!("{prefix}.to_out.0.bias"), vec![q_dim]));
}

/// Push one FFN's two linears (`net.0.proj`, `net.2`), optionally biased.
/// `has_bias` is a per-instance FACT read off the real header, not derived
/// from `cfg.*.ff_bias`: the real checkpoint's video-stream `ff` truly has
/// no bias (`ff_bias: false` governs it) but its `audio_ff` and BOTH
/// embeddings-connector FFNs carry bias tensors regardless - confirmed by
/// range-reading the real header at every block (0 and 47) and both
/// connectors, not assumed from the single shared `ff_bias` config key.
fn push_ff(m: &mut Vec<(String, Vec<usize>)>, prefix: &str, dim: usize, ff_dim: usize, has_bias: bool) {
    m.push((format!("{prefix}.net.0.proj.weight"), vec![ff_dim, dim]));
    if has_bias {
        m.push((format!("{prefix}.net.0.proj.bias"), vec![ff_dim]));
    }
    m.push((format!("{prefix}.net.2.weight"), vec![dim, ff_dim]));
    if has_bias {
        m.push((format!("{prefix}.net.2.bias"), vec![dim]));
    }
}

/// Push one `AdaLayerNormSingle`'s 6 tensors (`emb.timestep_embedder.
/// linear_{1,2}.{weight,bias}`, `linear.{weight,bias}`) - shared by every
/// adaLN table in [`av_dit_tensor_manifest`] (the model-level tables AND the
/// four AV cross-modal tables), all the same shape family at `(dim, rows)` -
/// see [`ada_layer_norm_single`]'s doc.
fn push_adaln_group(m: &mut Vec<(String, Vec<usize>)>, prefix: &str, dim: usize, rows: usize) {
    m.push((format!("{prefix}.emb.timestep_embedder.linear_1.weight"), vec![dim, 256]));
    m.push((format!("{prefix}.emb.timestep_embedder.linear_1.bias"), vec![dim]));
    m.push((format!("{prefix}.emb.timestep_embedder.linear_2.weight"), vec![dim, dim]));
    m.push((format!("{prefix}.emb.timestep_embedder.linear_2.bias"), vec![dim]));
    m.push((format!("{prefix}.linear.weight"), vec![rows * dim, dim]));
    m.push((format!("{prefix}.linear.bias"), vec![rows * dim]));
}

/// Push one embeddings connector's `1 + num_layers*16` tensors
/// (`learnable_registers` plus `num_layers` pre-LN 1-D transformer blocks,
/// each a self-attention-only [`push_attn`] (12 tensors, no cross-attention -
/// the real header carries no `attn2` under either connector) plus a biased
/// [`push_ff`] (4 tensors, `ff_mult*dim` hidden width) - see
/// [`av_dit_tensor_manifest`]'s doc for the real header's exact counts (129
/// per connector: 1 + 8*16).
fn push_connector(m: &mut Vec<(String, Vec<usize>)>, prefix: &str, num_registers: usize, dim: usize, num_layers: u32, ff_mult: usize) {
    m.push((format!("{prefix}.learnable_registers"), vec![num_registers, dim]));
    for l in 0..num_layers {
        let p = format!("{prefix}.transformer_1d_blocks.{l}");
        push_attn(m, &format!("{p}.attn1"), dim, dim, dim);
        push_ff(m, &format!("{p}.ff"), dim, ff_mult * dim, true);
    }
}

/// Every tensor an [`LtxAvDit`] forward reads OR the real 22B/4349-tensor
/// checkpoint carries (some real tensors - `prompt_adaln_single`/
/// `audio_prompt_adaln_single`, `to_gate_logits`, both embeddings
/// connectors - are not yet consumed by [`LtxAvBlock::forward`]/
/// [`LtxAvDit::forward`]; see `crate::config`'s doc on each field this reads
/// for which). Name + shape, derived from `cfg`, the same "manifest drives
/// both sides" discipline [`dit_tensor_manifest`] uses - this is what a
/// later real-weight-import milestone validates two-way coverage against,
/// so every shape here was cross-checked against the real header (range-read
/// and parsed, 4349 tensors, `general.architecture = "ltxv"`) rather than
/// transcribed from a paraphrase; see this crate's own porting notes for the
/// two shapes that paraphrase got backwards (`keyframes_abs_pos_embedding`
/// and the connectors' `learnable_registers`, both transposed) and the one
/// real asymmetry a naive reading of `ff_bias` would miss (audio's FFN and
/// both connectors' FFNs carry bias; video's main FFN does not).
///
/// Tensor count breaks down exactly as the real header does: 59 top-level +
/// 48 blocks * 84 + 129 * 2 (both connectors) = 4349.
pub fn av_dit_tensor_manifest(cfg: &LtxAvDitConfig) -> Vec<(String, Vec<usize>)> {
    let vcfg = &cfg.video;
    let acfg = &cfg.audio;
    let vdim = vcfg.inner_dim as usize;
    let adim = acfg.inner_dim as usize;
    let rows9 = vcfg.adaln_rows() as usize;
    let mut m: Vec<(String, Vec<usize>)> = Vec::new();

    // ---- top-level embed/head tensors, one pair per stream -------------
    m.push(("patchify_proj.weight".into(), vec![vdim, vcfg.in_channels as usize]));
    m.push(("patchify_proj.bias".into(), vec![vdim]));
    m.push(("audio_patchify_proj.weight".into(), vec![adim, acfg.in_channels as usize]));
    m.push(("audio_patchify_proj.bias".into(), vec![adim]));
    m.push(("proj_out.weight".into(), vec![vcfg.out_channels as usize, vdim]));
    m.push(("proj_out.bias".into(), vec![vcfg.out_channels as usize]));
    m.push(("audio_proj_out.weight".into(), vec![acfg.out_channels as usize, adim]));
    m.push(("audio_proj_out.bias".into(), vec![acfg.out_channels as usize]));
    m.push(("scale_shift_table".into(), vec![2, vdim]));
    m.push(("audio_scale_shift_table".into(), vec![2, adim]));
    if vcfg.use_keyframes_abs_pos_embedding {
        m.push(("keyframes_abs_pos_embedding".into(), vec![1, vdim]));
    }

    // ---- the 8 model-level adaLN-single tables --------------------------
    push_adaln_group(&mut m, "adaln_single", vdim, rows9);
    push_adaln_group(&mut m, "audio_adaln_single", adim, rows9);
    push_adaln_group(&mut m, "prompt_adaln_single", vdim, 2);
    push_adaln_group(&mut m, "audio_prompt_adaln_single", adim, 2);
    push_adaln_group(&mut m, "av_ca_video_scale_shift_adaln_single", vdim, 4);
    push_adaln_group(&mut m, "av_ca_audio_scale_shift_adaln_single", adim, 4);
    push_adaln_group(&mut m, "av_ca_a2v_gate_adaln_single", vdim, 1);
    push_adaln_group(&mut m, "av_ca_v2a_gate_adaln_single", adim, 1);

    // ---- per-block tensors -----------------------------------------------
    for l in 0..vcfg.num_layers {
        let p = format!("transformer_blocks.{l}");
        push_attn(&mut m, &format!("{p}.attn1"), vdim, vdim, vdim);
        push_attn(&mut m, &format!("{p}.attn2"), vdim, vdim, vdim);
        push_attn(&mut m, &format!("{p}.audio_attn1"), adim, adim, adim);
        push_attn(&mut m, &format!("{p}.audio_attn2"), adim, adim, adim);
        // Both AV cross-attention directions run at the AUDIO stream's
        // geometry (`inner_dim = adim`) regardless of which stream is
        // query - `crate::block::LtxAvBlock`'s doc.
        push_attn(&mut m, &format!("{p}.audio_to_video_attn"), vdim, adim, adim);
        push_attn(&mut m, &format!("{p}.video_to_audio_attn"), adim, vdim, adim);
        push_ff(&mut m, &format!("{p}.ff"), vdim, 4 * vdim, false);
        push_ff(&mut m, &format!("{p}.audio_ff"), adim, 4 * adim, true);
        m.push((format!("{p}.scale_shift_table"), vec![rows9, vdim]));
        m.push((format!("{p}.prompt_scale_shift_table"), vec![2, vdim]));
        m.push((format!("{p}.audio_scale_shift_table"), vec![rows9, adim]));
        m.push((format!("{p}.audio_prompt_scale_shift_table"), vec![2, adim]));
        m.push((format!("{p}.scale_shift_table_a2v_ca_video"), vec![5, vdim]));
        m.push((format!("{p}.scale_shift_table_a2v_ca_audio"), vec![5, adim]));
    }

    // ---- both embeddings connectors --------------------------------------
    push_connector(&mut m, "video_embeddings_connector", vcfg.connector_num_learnable_registers as usize, vcfg.connector_inner_dim() as usize, vcfg.connector_num_layers, 4);
    push_connector(&mut m, "audio_embeddings_connector", vcfg.connector_num_learnable_registers as usize, acfg.connector_inner_dim() as usize, vcfg.connector_num_layers, 4);

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

/// Build one pipeline stage's `Tensors` weight subset from a FLAT (name ->
/// data, no shape) checkpoint - `model::Model`/`Shardable`'s own
/// representation - filtered to exactly the names [`LtxDit::run_stage_forward`]
/// reads for `shard` (see [`shard_owns_weight`]). Shapes come from
/// [`dit_tensor_manifest`], the same manifest [`random_tiny_weights`] uses,
/// so a partial shard's tensors are shaped identically to the full model's
/// own. Panics if a needed name is missing from `init`, the same "loud, not
/// silent" convention [`tget`] uses.
pub(crate) fn shard_weights(cfg: &LtxDitConfig, init: &HashMap<String, Vec<f32>>, shard: &Shard) -> Tensors {
    dit_tensor_manifest(cfg)
        .into_iter()
        .filter(|(name, _)| shard_owns_weight(shard, name))
        .map(|(name, shape)| {
            let data = init.get(&name).unwrap_or_else(|| panic!("ltxv dit shard: missing weight {name}")).clone();
            let want: usize = shape.iter().product();
            assert_eq!(data.len(), want, "ltxv dit shard: {name} wrong length ({} vs {want})", data.len());
            (name, (shape, data))
        })
        .collect()
}

/// Whether pipeline stage `shard` needs weight `name` at all - the ONLY
/// weights a stage loads, so a partial shard never materializes the full
/// 48-block stack just to discard most of it (see [`shard_weights`]).
/// `transformer_blocks.{l}.*` follows the block range (`shard.owns(l)`);
/// `adaln_single.*` is REPLICATED - every stage recomputes its own per-token
/// adaLN table from the shared batch (see [`LtxDit::run_stage_forward`]'s
/// doc: the only thing that actually crosses a stage boundary is the
/// residual `x`); `patchify_proj.*`/`keyframes_abs_pos_embedding` are
/// embed-only (they produce the FIRST stage's initial `x`);
/// `scale_shift_table`/`proj_out.*` are head-only (the final projection).
pub(crate) fn shard_owns_weight(shard: &Shard, name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("transformer_blocks.") {
        let l: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        return shard.owns(l);
    }
    if name.starts_with("adaln_single.") {
        return true;
    }
    matches!(name, "patchify_proj.weight" | "patchify_proj.bias" | "keyframes_abs_pos_embedding") && shard.embed
        || matches!(name, "scale_shift_table" | "proj_out.weight" | "proj_out.bias") && shard.head
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

/// One pipeline-stage/training batch for [`LtxDit`] - the DiT's own input
/// shape (a latent + per-token timesteps/RoPE bounds/keyframes + raw text
/// context, optionally a training target), since `model::Batch`'s stock
/// variants (LM tokens, seq2seq, image-splice) carry none of that. Owned
/// data (not `model::Batch<'a>`'s borrowed slices) because it is set via
/// [`LtxDit::load_shard_batch`] and read back across an arbitrary number of
/// later `Model`/`Shardable` calls - the same reason `s3dit::train::
/// ZTrainModel` keeps its own owned `Batch` behind a `load_batch` method
/// instead of trying to fit through `Model::set_batch` (which this type
/// leaves a documented no-op, see `crate::shard`'s module doc).
pub struct DitBatch {
    /// `[t, in_channels]` - only read on the EMBED stage (a non-embed stage's
    /// input `x` comes from [`LtxDit::write_in_res`] instead).
    pub latent: Vec<f32>,
    /// `[t]`, RAW (pre `timestep_scale_multiplier`) per-token sigma.
    pub timesteps: Vec<f32>,
    /// `[3, t, 2]`.
    pub positions: Vec<f32>,
    /// `[t]`, only read on the embed stage.
    pub keyframes_mask: Vec<f32>,
    /// `[context_len, cross_attention_dim]`.
    pub context: Vec<f32>,
    pub context_len: usize,
    pub t: usize,
    /// `[t, out_channels]` training target (flow-matching velocity); `None`
    /// for a forward-only run - [`LtxDit::run_stage_forward`] then returns
    /// `None` even on the head stage (nothing to compute a loss against).
    pub target: Option<Vec<f32>>,
}

/// The tiny video-only DiT, weights resident on the host (device residency
/// is per-block, see `LtxBlock::on`).
pub struct LtxDit {
    cfg: LtxDitConfig,
    w: Tensors,
    device: Option<String>,
    /// Pipeline-parallel placement (`Shard::whole` for the ordinary,
    /// non-sharded path every other module in this crate uses) - see
    /// `crate::shard`'s module doc.
    shard: Shard,
    batch: RefCell<Option<DitBatch>>,
    /// This stage's INPUT-side residual (`res[shard.start]`), written by the
    /// previous stage via [`LtxDit::write_in_res`]; read instead of
    /// patchifying `batch.latent` on a non-embed stage.
    res_in: RefCell<Option<Vec<f32>>>,
    /// This stage's OUTPUT-side residual (`res[shard.end]`, PRE-output-stage)
    /// - set by [`LtxDit::run_stage_forward`], read by [`LtxDit::read_out_res`].
    res_out: RefCell<Option<Vec<f32>>>,
    /// The head stage's last `output_stage` result - the actual generation
    /// output, distinct from `res_out` (which never runs through
    /// `output_stage`). Read by [`LtxDit::take_stage_output`].
    stage_out: RefCell<Option<Vec<f32>>>,
}

impl LtxDit {
    pub fn new(cfg: LtxDitConfig, weights: Tensors, device: Option<&str>) -> LtxDit {
        let shard = Shard::whole(cfg.num_layers as usize);
        LtxDit::build(cfg, weights, device, shard)
    }

    /// Shared constructor: validates `cfg`, wires `weights`/`device`/`shard`,
    /// and zeroes the pipeline-stage scratch (batch + boundary residuals +
    /// last stage output) every shard needs regardless of whether it is ever
    /// actually used as one (see `crate::shard`'s module doc).
    fn build(cfg: LtxDitConfig, weights: Tensors, device: Option<&str>, shard: Shard) -> LtxDit {
        cfg.assert_supported();
        assert_eq!(cfg.cross_attention_dim, cfg.inner_dim, "M3 assumes cross_attention_dim == inner_dim (caption_projection=None) - see config.rs's doc");
        LtxDit {
            cfg,
            w: weights,
            device: device.map(str::to_string),
            shard,
            batch: RefCell::new(None),
            res_in: RefCell::new(None),
            res_out: RefCell::new(None),
            stage_out: RefCell::new(None),
        }
    }

    /// Build one pipeline stage from a FLAT checkpoint (`model::Model`/
    /// `Shardable`'s own representation) - `crate::shard`'s `Shardable::
    /// new_shard`/`Model::new` both delegate here (`Shard::whole` for the
    /// latter). `b`/`t` are accepted only for `Shardable`'s fixed signature
    /// and unused: unlike a GPU-resident model with persistent per-layer
    /// buffers sized at construction (e.g. `qwen3::Qwen`), `LtxBlock::on`
    /// allocates fresh device buffers on every [`Self::forward_blocks`] call,
    /// so a stage's construction needs only its own weight subset
    /// ([`shard_weights`]).
    pub(crate) fn from_flat_weights(cfg: LtxDitConfig, init: &HashMap<String, Vec<f32>>, shard: Shard) -> LtxDit {
        let w = shard_weights(&cfg, init, &shard);
        LtxDit::build(cfg, w, None, shard)
    }

    pub fn config(&self) -> &LtxDitConfig {
        &self.cfg
    }

    /// This instance's pipeline placement.
    pub fn shard(&self) -> &Shard {
        &self.shard
    }

    /// Set this stage's diffusion batch - see [`DitBatch`]'s doc for why this
    /// (owned, richer) seam exists instead of `model::Batch`.
    pub fn load_shard_batch(&self, b: DitBatch) {
        *self.batch.borrow_mut() = Some(b);
    }

    /// The head stage's last `output_stage` result (after
    /// [`Self::run_stage_forward`] has run) - the actual generation output,
    /// distinct from the `Shardable` residual seam (which only ever carries
    /// the PRE-`output_stage` residual across a stage boundary).
    pub fn take_stage_output(&self) -> Vec<f32> {
        self.stage_out.borrow().clone().expect("LtxDit::take_stage_output: run_stage_forward has not produced a head-stage output yet")
    }

    pub(crate) fn weight_names(&self) -> Vec<String> {
        self.w.keys().cloned().collect()
    }
    pub(crate) fn weight(&self, name: &str) -> Vec<f32> {
        tget(&self.w, name).to_vec()
    }

    /// `model::Shardable::read_out_res`'s implementation: this stage's
    /// OUTPUT-side residual (`res[shard.end]`), the pre-`output_stage` `x`
    /// after its own block range - set by [`Self::run_stage_forward`].
    pub(crate) fn read_out_res(&self) -> Vec<f32> {
        self.res_out.borrow().clone().expect("LtxDit::read_out_res: run_stage_forward has not run yet")
    }
    /// `model::Shardable::write_in_res`'s implementation: this stage's
    /// INPUT-side residual (`res[shard.start]`), read by
    /// [`Self::run_stage_forward`] on a non-embed stage instead of
    /// patchifying `batch.latent`.
    pub(crate) fn write_in_res(&self, data: &[f32]) {
        *self.res_in.borrow_mut() = Some(data.to_vec());
    }

    /// Run this stage's forward: patchify (embed stage only, from
    /// `batch.latent`) or the previous stage's residual ([`Self::write_in_res`])
    /// -> this stage's own block range ([`Self::forward_blocks`]) ->
    /// `output_stage` (head stage only). Every stage independently
    /// recomputes the per-token adaLN table and RoPE tables from `batch` and
    /// its OWN (replicated) `adaln_single.*` weights - the only thing that
    /// crosses the stage boundary over the wire is the residual `x` (see
    /// [`shard_owns_weight`]'s doc). Returns the head stage's MSE loss
    /// against `batch.target` (`None` with no target, or on a non-head
    /// stage).
    pub fn run_stage_forward(&self) -> Option<f32> {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim as usize;
        let batch_ref = self.batch.borrow();
        let batch = batch_ref.as_ref().expect("LtxDit::run_stage_forward: no batch (call load_shard_batch first)");
        if self.shard.embed {
            assert_eq!(batch.latent.len(), batch.t * cfg.in_channels as usize, "LtxDit::run_stage_forward: latent length mismatch");
        }

        let ts_scaled: Vec<f32> = batch.timesteps.iter().map(|&x| x * cfg.timestep_scale_multiplier as f32).collect();
        let (adaln_table, embedded_timestep) = ada_layer_norm_single(&self.w, "adaln_single", &ts_scaled, dim, cfg.adaln_rows() as usize);
        let rope = ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, &cfg.positional_embedding_max_pos, &batch.positions, batch.t);
        let gpu = open_device(self.device.as_deref());
        let (cos_bufs, sin_bufs) = upload_rope_tables(&gpu, &rope);

        let x0 = if self.shard.embed {
            let pw = tget(&self.w, "patchify_proj.weight");
            let pb = tget(&self.w, "patchify_proj.bias");
            let mut xx = linear(&batch.latent, batch.t, cfg.in_channels as usize, pw, Some(pb), dim);
            if cfg.use_keyframes_abs_pos_embedding {
                let kf = tget(&self.w, "keyframes_abs_pos_embedding");
                for ti in 0..batch.t {
                    if batch.keyframes_mask[ti] > 0.0 {
                        for d in 0..dim {
                            xx[ti * dim + d] += kf[d];
                        }
                    }
                }
            }
            xx
        } else {
            self.res_in.borrow().clone().expect("LtxDit::run_stage_forward: non-embed stage needs write_in_res first")
        };

        let (x_final, _block_out, _taps) =
            self.forward_blocks(&gpu, &x0, &adaln_table, &batch.context, &cos_bufs, &sin_bufs, batch.t as u32, batch.context_len as u32, self.shard.start as u32, self.shard.end as u32);
        *self.res_out.borrow_mut() = Some(x_final.clone());

        if self.shard.head {
            let out = output_stage(&self.w, "scale_shift_table", "proj_out", &x_final, &embedded_timestep, batch.t, dim, cfg.out_channels as usize, cfg.norm_eps);
            *self.stage_out.borrow_mut() = Some(out.clone());
            batch.target.as_ref().map(|target| {
                assert_eq!(out.len(), target.len(), "LtxDit::run_stage_forward: target length mismatch");
                out.iter().zip(target).map(|(o, g)| (o - g) * (o - g)).sum::<f32>() / out.len().max(1) as f32
            })
        } else {
            None
        }
    }

    /// Run transformer blocks `[lo, hi)` over `x` (the residual-stream host
    /// slice), returning the resulting `x` plus each block's own output copy
    /// and full [`BlockTaps`] (one entry per block in range, index 0 = block
    /// `lo`). The SAME op sequence [`Self::forward`] runs for its own
    /// `[0, num_layers)` range and [`Self::run_stage_forward`] runs for one
    /// pipeline stage's `[shard.start, shard.end)` - a single source of
    /// truth (one function, different bounds) rather than two loops that
    /// could silently drift apart.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_blocks(
        &self,
        gpu: &Gpu,
        x: &[f32],
        adaln_table: &[f32],
        context: &[f32],
        cos_bufs: &[DeviceBuffer],
        sin_bufs: &[DeviceBuffer],
        t: u32,
        context_len: u32,
        lo: u32,
        hi: u32,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<BlockTaps>) {
        let mut xx = x.to_vec();
        let mut block_out = Vec::with_capacity((hi - lo) as usize);
        let mut taps = Vec::with_capacity((hi - lo) as usize);
        for l in lo..hi {
            let blk = LtxBlock::on(gpu.share(), &self.cfg, &self.w, &format!("transformer_blocks.{l}"), t, context_len);
            let (out, tp) = blk.forward(&xx, adaln_table, context, cos_bufs, sin_bufs, t);
            xx = out;
            block_out.push(xx.clone());
            taps.push(tp);
        }
        (xx, block_out, taps)
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

        let (x_final, block_out, mut taps) = self.forward_blocks(&gpu, &x, &adaln_table, context, &cos_bufs, &sin_bufs, t as u32, context_len as u32, 0, cfg.num_layers);
        x = x_final;
        assert!(!taps.is_empty(), "num_layers must be >= 1");
        let b0 = taps.remove(0);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LtxAvDitConfig;
    use std::collections::HashSet;

    /// [`av_dit_tensor_manifest`] at the REAL LTX-2.5 22B config must produce
    /// exactly the real checkpoint's own tensor count and breakdown - 4349
    /// total (range-read and parsed off the real GGUF header), split 59
    /// top-level + 48*84 per-block + 129*2 across both embeddings
    /// connectors. This is the guard a later real-weight-import milestone
    /// leans on: if this count ever drifts from the real header, two-way
    /// coverage there fails loudly by name rather than silently importing a
    /// wrong shape.
    #[test]
    fn av_manifest_matches_the_real_22b_checkpoint_header_count() {
        let cfg = LtxAvDitConfig::ltx25();
        let m = av_dit_tensor_manifest(&cfg);
        assert_eq!(m.len(), 4349, "total tensor count must match the real header");

        let top_level = m.iter().filter(|(n, _)| !n.starts_with("transformer_blocks.") && !n.contains("embeddings_connector")).count();
        assert_eq!(top_level, 59, "top-level (non-block, non-connector) tensor count");

        let per_block = m.iter().filter(|(n, _)| n.starts_with("transformer_blocks.0.")).count();
        assert_eq!(per_block, 84, "block 0's own tensor count");

        let video_connector = m.iter().filter(|(n, _)| n.starts_with("video_embeddings_connector.")).count();
        assert_eq!(video_connector, 129, "video connector tensor count (1 register + 8*16)");
        let audio_connector = m.iter().filter(|(n, _)| n.starts_with("audio_embeddings_connector.")).count();
        assert_eq!(audio_connector, 129, "audio connector tensor count (1 register + 8*16)");

        // No duplicate names, and no accidental shape mismatch on the two
        // shapes an earlier paraphrase transposed.
        let names: HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "manifest must not contain duplicate tensor names");
        assert_eq!(m.iter().find(|(n, _)| n == "keyframes_abs_pos_embedding").unwrap().1, vec![1, 4096]);
        assert_eq!(m.iter().find(|(n, _)| n == "video_embeddings_connector.learnable_registers").unwrap().1, vec![128, 4096]);
        assert_eq!(m.iter().find(|(n, _)| n == "audio_embeddings_connector.learnable_registers").unwrap().1, vec![128, 2048]);

        // The real asymmetry: video's main FFN has no bias, audio's and both
        // connectors' FFNs do.
        assert!(!names.contains("transformer_blocks.0.ff.net.0.proj.bias"));
        assert!(names.contains("transformer_blocks.0.audio_ff.net.0.proj.bias"));
        assert!(names.contains("video_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.bias"));
        assert!(names.contains("audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.bias"));

        // `to_gate_logits` really is on every attention module, including
        // the two audio<->video cross-attention directions - the real
        // header confirms this, not just the two per-stream self-/text-
        // cross-attentions.
        assert!(names.contains("transformer_blocks.0.audio_to_video_attn.to_gate_logits.weight"));
        assert!(names.contains("transformer_blocks.0.video_to_audio_attn.to_gate_logits.weight"));
        assert_eq!(m.iter().find(|(n, _)| n == "transformer_blocks.0.audio_to_video_attn.to_gate_logits.weight").unwrap().1, vec![32, 4096]);
        assert_eq!(m.iter().find(|(n, _)| n == "transformer_blocks.0.video_to_audio_attn.to_gate_logits.weight").unwrap().1, vec![32, 2048]);
    }
}
