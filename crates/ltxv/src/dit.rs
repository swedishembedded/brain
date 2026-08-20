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

use checkpoint::TensorSource;
use gpu_core::{DeviceBuffer, Gpu};
use model::Shard;
use vae::blocks::Tensors;

use crate::block::{load_block_tensors_from_source, open_device, AvBlockTaps, BlockTaps, EmbeddingsConnector, GenerationCache, LtxAvBlock, LtxBlock, LtxBlockQ, QBlockWeights, QTier};
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
///
/// Every attn's `to_gate_logits.{weight,bias}` is ALWAYS listed here
/// (via [`push_attn`], the same helper [`av_dit_tensor_manifest`] uses),
/// regardless of `cfg.apply_gated_attention` - the same "manifest doesn't
/// gate any tensor family, it makes the shape representable" convention
/// [`av_dit_tensor_manifest`]'s doc already states. This is a real-weight
/// requirement, not cosmetic: [`crate::block::LtxBlock::on`] reads
/// `to_gate_logits` off the weight map whenever `cfg.apply_gated_attention`
/// is `true` (e.g. [`LtxDitConfig::ltx25_22b`]), so a caller that builds a
/// gated config's weights FROM this manifest ([`random_tiny_weights`], a
/// real-checkpoint loader) must have it listed or that `tget` panics on a
/// missing name. Both `to_gate_logits.weight`/`.bias` are already on
/// `crate::int8::is_never_quantized`'s substring list, so this addition
/// changes no int8-eligibility classification for any existing caller.
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
            push_attn(&mut m, &format!("{p}.{attn}"), dim, dim, dim);
        }
        m.push((format!("{p}.ff.net.0.proj.weight"), vec![4 * dim, dim]));
        m.push((format!("{p}.ff.net.2.weight"), vec![dim, 4 * dim]));
        m.push((format!("{p}.scale_shift_table"), vec![cfg.adaln_rows() as usize, dim]));
        m.push((format!("{p}.prompt_scale_shift_table"), vec![2, dim]));
    }
    // `video_embeddings_connector`'s 1 + num_layers*16 tensors - gated on
    // `use_embeddings_connector` (unlike `to_gate_logits` above, which is
    // never gated) the same way `keyframes_abs_pos_embedding` just above
    // is: conditional inclusion, so `LtxDitConfig::tiny()` (connector
    // disabled) keeps its exact pre-existing manifest and every test built
    // against it stays byte-identical. Real gap this closes: before this,
    // `random_tiny_weights`/`import_dit` at a connector-enabled config
    // (`LtxDitConfig::ltx25_22b`/`tiny_gated`) built a weight map missing
    // `video_embeddings_connector.*` entirely, so `LtxDit::forward`'s own
    // `route_context_through_connector` call would panic on a missing
    // tensor the first time either config's forward actually ran through
    // this crate's own weight-building path (as opposed to a fixture file
    // loaded directly via `load_tiny_weights`, which bypasses this
    // manifest, the tests/dit_parity.rs `setup_gated` case) - never
    // previously exercised because every existing caller of
    // `random_tiny_weights` passes `LtxDitConfig::tiny()`, not
    // `tiny_gated()`/`ltx25_22b()`.
    if cfg.use_embeddings_connector {
        push_connector(&mut m, "video_embeddings_connector", cfg.connector_num_learnable_registers as usize, cfg.connector_inner_dim() as usize, cfg.connector_num_layers, 4);
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
    // ---- top-level embed/head tensors, one pair per stream -------------
    let mut m: Vec<(String, Vec<usize>)> = vec![
        ("patchify_proj.weight".into(), vec![vdim, vcfg.in_channels as usize]),
        ("patchify_proj.bias".into(), vec![vdim]),
        ("audio_patchify_proj.weight".into(), vec![adim, acfg.in_channels as usize]),
        ("audio_patchify_proj.bias".into(), vec![adim]),
        ("proj_out.weight".into(), vec![vcfg.out_channels as usize, vdim]),
        ("proj_out.bias".into(), vec![vcfg.out_channels as usize]),
        ("audio_proj_out.weight".into(), vec![acfg.out_channels as usize, adim]),
        ("audio_proj_out.bias".into(), vec![acfg.out_channels as usize]),
        ("scale_shift_table".into(), vec![2, vdim]),
        ("audio_scale_shift_table".into(), vec![2, adim]),
    ];
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

/// A random, seeded set of weights at [`av_dit_tensor_manifest`]'s exact
/// shape - the AV counterpart of [`random_tiny_weights`], same convention
/// (biases zero, everything else `N(0, 0.02²)`).
pub fn random_av_tiny_weights(cfg: &LtxAvDitConfig, seed: u64) -> Tensors {
    let mut rng = data::rng::Rng::new(seed);
    av_dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = if name.ends_with(".bias") { vec![0.0; n] } else { (0..n).map(|_| (rng.next_gaussian() * 0.02) as f32).collect() };
            (name, (shape, data))
        })
        .collect()
}

/// [`shard_weights`]'s AV counterpart, filtered through [`av_shard_owns_weight`].
pub(crate) fn av_shard_weights(cfg: &LtxAvDitConfig, init: &HashMap<String, Vec<f32>>, shard: &Shard) -> Tensors {
    av_dit_tensor_manifest(cfg)
        .into_iter()
        .filter(|(name, _)| av_shard_owns_weight(shard, name))
        .map(|(name, shape)| {
            let data = init.get(&name).unwrap_or_else(|| panic!("ltxv av dit shard: missing weight {name}")).clone();
            let want: usize = shape.iter().product();
            assert_eq!(data.len(), want, "ltxv av dit shard: {name} wrong length ({} vs {want})", data.len());
            (name, (shape, data))
        })
        .collect()
}

/// Whether pipeline stage `shard` needs weight `name` at all - [`shard_owns_
/// weight`]'s AV counterpart. `transformer_blocks.{l}.*` follows the block
/// range, same as the video-only path. Every model-level adaLN-single table
/// (video's own, audio's own, BOTH streams' `prompt_adaln_single` twins -
/// unused by the forward today but still real header tensors, see
/// [`av_dit_tensor_manifest`]'s doc - and all four AV cross-modal tables)
/// plus BOTH embeddings connectors are REPLICATED: every stage independently
/// recomputes them from the (replicated) batch, the same "small weight,
/// replicate rather than wire a second boundary for it" call [`shard_owns_
/// weight`]'s doc makes for the video-only path's `adaln_single.*` -
/// extended here to the connectors too because [`LtxAvDit::run_stage_forward`]
/// (unlike the video-only [`LtxDit::run_stage_forward`], which does not yet
/// route `context` through a connector at all) DOES run both connectors on
/// every stage, so their weights must be everywhere the block stack is.
/// `patchify_proj`/`audio_patchify_proj`/`keyframes_abs_pos_embedding` are
/// embed-only (they produce the first stage's initial `vx`/`ax`);
/// `scale_shift_table`/`proj_out.*`/`audio_scale_shift_table`/`audio_
/// proj_out.*` are head-only (the final per-stream projections).
pub(crate) fn av_shard_owns_weight(shard: &Shard, name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("transformer_blocks.") {
        let l: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        return shard.owns(l);
    }
    const REPLICATED_PREFIXES: [&str; 10] = [
        "adaln_single.",
        "audio_adaln_single.",
        "prompt_adaln_single.",
        "audio_prompt_adaln_single.",
        "av_ca_video_scale_shift_adaln_single.",
        "av_ca_audio_scale_shift_adaln_single.",
        "av_ca_a2v_gate_adaln_single.",
        "av_ca_v2a_gate_adaln_single.",
        "video_embeddings_connector.",
        "audio_embeddings_connector.",
    ];
    if REPLICATED_PREFIXES.iter().any(|p| name.starts_with(p)) {
        return true;
    }
    matches!(name, "patchify_proj.weight" | "patchify_proj.bias" | "audio_patchify_proj.weight" | "audio_patchify_proj.bias" | "keyframes_abs_pos_embedding") && shard.embed
        || matches!(name, "scale_shift_table" | "proj_out.weight" | "proj_out.bias" | "audio_scale_shift_table" | "audio_proj_out.weight" | "audio_proj_out.bias") && shard.head
}

/// `out[r,o] = b[o] + Σ_i x[r,i]·w[o,i]`, `w` row-major `[out_dim, in_dim]` -
/// plain `nn.Linear`.
///
/// Row-parallel through `backend_cpu::par::rows_mut` - `wan::model::linear`'s
/// own precedent, reused verbatim rather than reimplemented (kernels.md §F.3):
/// the split is over OUTPUT ROWS and each row's dot products are left exactly
/// as they were (still a straight sequential `+=` walk over `in_dim`, one
/// thread per row), so every output element accumulates in the SAME order
/// regardless of thread count - bit-identical to the old serial form, not
/// merely close, which is what a parity/gradcheck gate can accept
/// unconditionally rather than re-tolerating.
///
/// Measured, not assumed: `ada_layer_norm_single`'s call into this function
/// (the `[t,4096]x[36864,4096]^T` 9-row adaLN table) cost a flat ~21 s per
/// real forward call at the real 22B checkpoint's width - ~11% of one real
/// denoise step - as a naive, single-core scalar loop; this fix replaces that
/// loop without changing one bit of the output.
fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    backend_cpu::par::rows_mut(&mut out, out_dim, |r, orow| {
        let xr = &x[r * in_dim..r * in_dim + in_dim];
        for (o, slot) in orow.iter_mut().enumerate() {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b.map(|b| b[o]).unwrap_or(0.0);
            for (xi, wi) in xr.iter().zip(wr) {
                acc += xi * wi;
            }
            *slot = acc;
        }
    });
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

/// Route `context` through `video_embeddings_connector`/
/// `audio_embeddings_connector` when `enabled` (`cfg.use_embeddings_
/// connector`) - shared by [`LtxDit::forward`] (one call) and
/// [`LtxAvDit::forward`] (two calls, one per stream's own connector
/// prefix/geometry) so the "disabled -> pass `context` through unchanged"
/// and "enabled -> build+run `EmbeddingsConnector`" branches exist in
/// exactly one place. Returns `(context_for_blocks, connector_raw_output)` -
/// the second is EMPTY when disabled (nothing to tap) and otherwise a copy
/// of the first (both are what a parity test's `connector_out` tap and the
/// blocks' own `context` input need, see [`DitTaps::connector_out`]'s doc).
#[allow(clippy::too_many_arguments)]
fn route_context_through_connector(
    gpu: &Gpu,
    w: &Tensors,
    prefix: &str,
    enabled: bool,
    context: &[f32],
    valid: &[f32],
    context_len: u32,
    dim: u32,
    heads: u32,
    head_dim: u32,
    num_layers: u32,
    num_registers: u32,
    gated: bool,
    norm_output: bool,
    theta: f64,
    max_pos: &[u32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    if !enabled {
        return (context.to_vec(), Vec::new());
    }
    let connector = EmbeddingsConnector::on(gpu.share(), w, prefix, dim, heads, head_dim, num_layers, num_registers, gated, norm_output, theta, max_pos, eps);
    let out = connector.forward(context, valid, context_len);
    (out.clone(), out)
}

/// Upload one [`LtxRopeTables`]' per-head `[T, head_dim/2]` `(cos, sin)`
/// slices as device buffers - shared by every block (RoPE tables do not
/// vary per layer). Free function (not tied to [`LtxDit`]) so both
/// [`LtxDit::forward`] and [`LtxAvDit::forward`] - which needs FOUR
/// independent tables, not one - share it.
pub(crate) fn upload_rope_tables(gpu: &Gpu, rope: &LtxRopeTables) -> (Vec<DeviceBuffer>, Vec<DeviceBuffer>) {
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
    /// `[context_len, connector_inner_dim]` - `video_embeddings_connector`'s
    /// own output (`crate::block::EmbeddingsConnector::forward`'s return
    /// value, BEFORE any block's own `prompt_scale_shift_table` modulates
    /// it), the tap a parity test bisects with when `cfg.use_embeddings_
    /// connector` is `true`. EMPTY when the connector is disabled (nothing
    /// to tap - `context` reaches the blocks unchanged).
    pub connector_out: Vec<f32>,
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

    /// [`Self::forward_blocks`]'s quantized-compute (int8/int4) twin -
    /// [`LtxBlockQ::on`] instead of [`LtxBlock::on`], everything else
    /// (RoPE tables, adaLN table, context) identical. Exists so
    /// [`Self::forward_q`] can compare against [`Self::forward`] on the SAME
    /// model-level entry point, changing only the block dispatch tier - see
    /// `crate::block`'s "Quantized-compute" module doc for what this tier
    /// actually buys (capacity, not speed) and its exact scope (video-only,
    /// ten quantizable linears per block).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_blocks_q(
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
        tier: QTier,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<BlockTaps>) {
        let mut xx = x.to_vec();
        let mut block_out = Vec::with_capacity((hi - lo) as usize);
        let mut taps = Vec::with_capacity((hi - lo) as usize);
        for l in lo..hi {
            let blk = LtxBlockQ::on(gpu.share(), &self.cfg, &self.w, &format!("transformer_blocks.{l}"), t, context_len, tier);
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
    /// `crate::block`'s doc) - when `cfg.use_embeddings_connector` is
    /// `true`, this is the PRE-connector embedding (`caption_proj_before_
    /// connector`'s ordering: `caption_projection` already ran upstream,
    /// the connector has not yet) and is routed through
    /// `video_embeddings_connector` ([`crate::block::EmbeddingsConnector`])
    /// before any block reads it; `context_valid`: `[context_len]`, `1.0`
    /// keeps a position's real embedding, `0.0` substitutes that position
    /// with a tiled learnable register row (see [`crate::block::
    /// EmbeddingsConnector::forward`]'s doc) - ignored when the connector is
    /// disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], timesteps: &[f32], positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, t: usize, context_valid: &[f32]) -> DitTaps {
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

        #[rustfmt::skip]
        let (connector_context, connector_out) = route_context_through_connector(
            &gpu, &self.w, "video_embeddings_connector", cfg.use_embeddings_connector, context, context_valid, context_len as u32,
            cfg.connector_inner_dim(), cfg.connector_num_attention_heads, cfg.connector_attention_head_dim,
            cfg.connector_num_layers, cfg.connector_num_learnable_registers, cfg.connector_apply_gated_attention,
            cfg.connector_norm_output, cfg.positional_embedding_theta, &cfg.connector_positional_embedding_max_pos, cfg.norm_eps,
        );

        let (x_final, block_out, mut taps) = self.forward_blocks(&gpu, &x, &adaln_table, &connector_context, &cos_bufs, &sin_bufs, t as u32, context_len as u32, 0, cfg.num_layers);
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
            connector_out,
            b0_attn1_out: b0.attn1_out,
            b0_attn2_out: b0.attn2_out,
            b0_ff_out: b0.ff_out,
            block_out,
            out,
        }
    }

    /// [`Self::forward`]'s quantized-compute (int8/int4) twin: identical op
    /// sequence and identical taps shape - only the block stack dispatches
    /// through [`Self::forward_blocks_q`] (packed int8/int4 weights, dynamic
    /// per-token activation quantization) instead of [`Self::forward_blocks`]'s
    /// plain fp32 path. See `crate::block`'s module doc for the tier itself;
    /// this method exists so a parity test can compare the two model-level
    /// forwards directly, on the SAME weights, changing only `tier`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_q(&self, latent: &[f32], timesteps: &[f32], positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, t: usize, context_valid: &[f32], tier: QTier) -> DitTaps {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim as usize;
        assert_eq!(latent.len(), t * cfg.in_channels as usize);
        assert_eq!(timesteps.len(), t);
        assert_eq!(keyframes_mask.len(), t);
        assert_eq!(context.len(), context_len * cfg.cross_attention_dim as usize);

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

        let ts_scaled: Vec<f32> = timesteps.iter().map(|&x| x * cfg.timestep_scale_multiplier as f32).collect();
        let (adaln_table, embedded_timestep) = ada_layer_norm_single(&self.w, "adaln_single", &ts_scaled, dim, cfg.adaln_rows() as usize);

        let rope = ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, &cfg.positional_embedding_max_pos, positions, t);

        let gpu = open_device(self.device.as_deref());
        let (cos_bufs, sin_bufs) = upload_rope_tables(&gpu, &rope);

        #[rustfmt::skip]
        let (connector_context, connector_out) = route_context_through_connector(
            &gpu, &self.w, "video_embeddings_connector", cfg.use_embeddings_connector, context, context_valid, context_len as u32,
            cfg.connector_inner_dim(), cfg.connector_num_attention_heads, cfg.connector_attention_head_dim,
            cfg.connector_num_layers, cfg.connector_num_learnable_registers, cfg.connector_apply_gated_attention,
            cfg.connector_norm_output, cfg.positional_embedding_theta, &cfg.connector_positional_embedding_max_pos, cfg.norm_eps,
        );

        let (x_final, block_out, mut taps) = self.forward_blocks_q(&gpu, &x, &adaln_table, &connector_context, &cos_bufs, &sin_bufs, t as u32, context_len as u32, 0, cfg.num_layers, tier);
        x = x_final;
        assert!(!taps.is_empty(), "num_layers must be >= 1");
        let b0 = taps.remove(0);

        let out = output_stage(&self.w, "scale_shift_table", "proj_out", &x, &embedded_timestep, t, dim, cfg.out_channels as usize, cfg.norm_eps);

        DitTaps {
            rope_cos: rope.cos,
            rope_sin: rope.sin,
            adaln_table,
            embedded_timestep,
            connector_out,
            b0_attn1_out: b0.attn1_out,
            b0_attn2_out: b0.attn2_out,
            b0_ff_out: b0.ff_out,
            block_out,
            out,
        }
    }
}

// ---------------------------------------------------------------------------
// Real-checkpoint streamed forward (video-only): the int8-compute path
// ([`LtxDit::forward_q`]) still requires the WHOLE model resident as host
// fp32 in `self.w` before it can run a single block, because
// [`LtxDit::forward_blocks_q`] looks up `transformer_blocks.{l}.*` off that
// one map every iteration - fine for a tiny config, not for the real 22B
// checkpoint (88 GB of fp32 for a model this port's own int8 tier exists
// specifically to avoid materializing). [`load_head_tensors_from_source`] +
// [`forward_q_streamed`] are this function's streamed twin: only the small
// non-block tensors are kept resident, and each block is loaded, quantized,
// run, and dropped one at a time straight off a `checkpoint::TensorSource`
// (`crate::gguf_src::LtxvGgufSource` for the real GGUF) - the same bound
// `crate::block::load_block_tensors_from_source`'s own doc documents at
// "one block" granularity, applied here to a full model-level forward.
// ---------------------------------------------------------------------------

/// Every tensor a real checkpoint carries OUTSIDE the 48
/// `transformer_blocks.N.*` families (`patchify_proj`, `adaln_single.*`,
/// `keyframes_abs_pos_embedding`, `video_embeddings_connector.*`, the
/// top-level `scale_shift_table`, `proj_out`) - the "head"
/// [`forward_q_streamed`] keeps resident in host fp32 for an entire
/// generation run, since summed together (a few GB, dominated by the
/// 8-layer connector) they are a small fraction of the 22B model. The other
/// 48/49ths streams in one block at a time inside [`forward_q_streamed`]
/// itself via [`load_block_tensors_from_source`].
pub fn load_head_tensors_from_source(src: &dyn TensorSource, cfg: &LtxDitConfig) -> Tensors {
    let manifest = dit_tensor_manifest(cfg);
    let mut out = Tensors::new();
    for (name, shape) in manifest {
        if name.starts_with("transformer_blocks.") {
            continue;
        }
        let mut data = Vec::new();
        let found = src.with_tensor(&name, &mut |d| data = d.to_vec());
        assert!(found, "load_head_tensors_from_source: missing {name}");
        let want: usize = shape.iter().product();
        assert_eq!(data.len(), want, "load_head_tensors_from_source: {name} wrong length ({} vs {want})", data.len());
        out.insert(name, (shape, data));
    }
    out
}

/// [`LtxDit::forward_q`]'s streamed twin: identical op sequence and output
/// (patchify -> keyframes embed -> adaLN-single table -> RoPE tables ->
/// connector routing -> block stack -> output stage), but never requires
/// the whole 22B model resident as host fp32. `head` is this function's own
/// [`load_head_tensors_from_source`] output. Returns only the final
/// `[t, out_channels]` prediction: a real generation run only needs this,
/// not the full parity-test tap set [`LtxDit::forward_q`] returns.
/// `device`: opens a FRESH [`Gpu`] every call, the same convention
/// [`LtxDit::forward`]/[`LtxDit::forward_q`] already use - measured at the
/// real 22B checkpoint's own scale (a real multi-step generation run, not
/// just the tiny-config case) to reliably release each call's device VRAM
/// before the next call's own block-by-block allocations begin; reusing
/// ONE [`Gpu`] handle across calls instead was tried and measured worse
/// (a second real forward then ran out of device memory that a fresh
/// device open does not).
///
/// `cache`: a per-CHECKPOINT (not per-call, and no longer per-generation)
/// HOST-side [`GenerationCache`] - `crate::pipeline::RealDit` obtains one from
/// [`crate::weightcache`]'s process-wide registry, keyed on the checkpoint's
/// own identity, and passes the SAME handle into every one of a generation's
/// ~20-50 denoise-step forward calls. Because the registry outlives the
/// `RealDit`, a SECOND generation against the same checkpoint - a different
/// prompt, a different size, seconds or hours later - starts warm on its very
/// first forward instead of re-paying the cold-disk cost a real profiling pass
/// measured at 365 s of a 964 s run. It holds two things this function would
/// otherwise recompute identically on every call.
///
/// The first is every block's already-quantized weight bytes, keyed by layer
/// index. On a cache
/// MISS (an empty slot - the common case on a generation's very first
/// forward call), a block is streamed from `src` via
/// [`load_block_tensors_from_source`] and quantized to `tier` exactly as
/// before, but the quantized bytes are ALSO stashed in the cache before
/// upload; on a cache HIT (every later step, same layer), the GGUF read and
/// the CPU-side int8/int4 quantization are both skipped entirely and the
/// cached bytes are re-uploaded to this call's fresh `Gpu` directly
/// ([`LtxBlockQ::on_cached`]) - no device buffer crosses a call boundary
/// (the "fresh `Gpu` every call" constraint above is unaffected), only host
/// bytes do. This is exact, not approximate: `model::int8::quantize_weight`/
/// `model::int4::quantize_weight_q4` are pure functions of the checkpoint's
/// own (immutable) weight bytes, so a cached result and a freshly recomputed
/// one are bit-identical by construction - recomputing on every step was
/// always redundant work, never a source of any different number. Peak EXTRA
/// host memory beyond `head` is now the WHOLE cache once every layer has
/// been visited once, rather than one block's fp32 expansion - measured at
/// the real 22B/Q8_0 width via [`crate::block::CachedQBlockWeights::
/// byte_len`] at ~270 MB/block, ~13 GB for all 48 - a deliberate trade of
/// host RAM for skipping the dominant real cost Phase 8 measured
/// (~86% of one real denoise step was GGUF re-read + re-quantize of the SAME
/// immutable weights, over and over, every single step). That trade is now
/// GOVERNED rather than merely affordable: the cache runs under the byte
/// budget `--limit-ram-total` publishes and evicts by the residency layer's
/// own cost-aware policy when it is tight - see [`crate::weightcache`].
///
/// The second is the embeddings-connector routing below. `context`,
/// `context_valid` and `context_len` are fixed for a whole generation once the
/// prompt has been encoded, and the connector reads nothing else, so its output
/// is the same on every step - while recomputing it means re-uploading the
/// connector's own fp32 weights to this call's fresh device and re-running its
/// transformer stack. Also exact for the same reason as the block half:
/// identical inputs, pure function, so the cached answer IS the recomputed one.
///
/// Traced at `--trace-ltxv 4` (each host stage's duration, the per-generation
/// cache's hit/miss split) and `5` (every layer individually: index, hit or
/// miss, and the GGUF-read / quantize / GPU milliseconds it cost). The
/// per-layer breadcrumb is the one this function's own Phase 8 attribution
/// needed and had to reconstruct by hand from summed `BRAIN_PROFILE` stage
/// totals - the totals stay, since the perf gates read them; the trace is
/// what makes a single anomalous layer visible rather than averaged away.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "info", name = "dit_forward_streamed", skip_all, fields(t = t, context_len = context_len, layers = cfg.num_layers, tier = ?tier))]
pub fn forward_q_streamed(
    cfg: &LtxDitConfig,
    src: &dyn TensorSource,
    head: &Tensors,
    device: Option<&str>,
    tier: QTier,
    latent: &[f32],
    timesteps: &[f32],
    positions: &[f32],
    keyframes_mask: &[f32],
    context: &[f32],
    context_len: usize,
    t: usize,
    context_valid: &[f32],
    cache: &GenerationCache,
) -> Vec<f32> {
    let dim = cfg.inner_dim as usize;
    assert_eq!(latent.len(), t * cfg.in_channels as usize);
    assert_eq!(timesteps.len(), t);
    assert_eq!(keyframes_mask.len(), t);
    assert_eq!(context.len(), context_len * cfg.cross_attention_dim as usize);

    let s_patch = std::time::Instant::now();
    let pw = tget(head, "patchify_proj.weight");
    let pb = tget(head, "patchify_proj.bias");
    let mut x = linear(latent, t, cfg.in_channels as usize, pw, Some(pb), dim);
    if cfg.use_keyframes_abs_pos_embedding {
        let kf = tget(head, "keyframes_abs_pos_embedding");
        for ti in 0..t {
            if keyframes_mask[ti] > 0.0 {
                for d in 0..dim {
                    x[ti * dim + d] += kf[d];
                }
            }
        }
    }
    gpu_core::profile::stage_time("forward_q_streamed: patchify + keyframes (host)", s_patch);
    tracing::debug!(stage = "patchify", ms = s_patch.elapsed().as_secs_f32() * 1e3, keyframes = cfg.use_keyframes_abs_pos_embedding, "host stage done");

    let s_adaln = std::time::Instant::now();
    let ts_scaled: Vec<f32> = timesteps.iter().map(|&x| x * cfg.timestep_scale_multiplier as f32).collect();
    let (adaln_table, embedded_timestep) = ada_layer_norm_single(head, "adaln_single", &ts_scaled, dim, cfg.adaln_rows() as usize);
    gpu_core::profile::stage_time("forward_q_streamed: adaLN-single table (host)", s_adaln);
    tracing::debug!(stage = "adaln_single", ms = s_adaln.elapsed().as_secs_f32() * 1e3, rows = cfg.adaln_rows(), "host stage done");

    let s_rope = std::time::Instant::now();
    let rope = ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, &cfg.positional_embedding_max_pos, positions, t);
    gpu_core::profile::stage_time("forward_q_streamed: RoPE table build (host, f64)", s_rope);
    tracing::debug!(stage = "rope_tables", ms = s_rope.elapsed().as_secs_f32() * 1e3, theta = cfg.positional_embedding_theta, "host stage done");

    let s_open = std::time::Instant::now();
    let gpu = open_device(device);
    gpu_core::profile::stage_time("forward_q_streamed: open_device (fresh Gpu + shader pipeline compile)", s_open);
    // A FRESH `Gpu` per call is this function's deliberate design (see its
    // doc), which also makes every call a fresh adapter enumeration + shader
    // pipeline compile - so this is the one event a device-lifecycle
    // investigation wants correlated against `--trace-gpu`.
    tracing::debug!(stage = "open_device", ms = s_open.elapsed().as_secs_f32() * 1e3, device = device.unwrap_or("(ambient)"), "opened a fresh device for this forward");
    let (cos_bufs, sin_bufs) = upload_rope_tables(&gpu, &rope);

    let s_conn = std::time::Instant::now();
    // The connector's inputs (`context`, `context_valid`, `context_len`) are
    // fixed for a whole generation, so its answer is too - see this function's
    // `cache` doc. Ask the cache first; only a genuinely new context pays.
    let conn_hit = cache.connector_hit(context, context_valid, context_len);
    let conn_was_hit = conn_hit.is_some();
    let connector_context = match conn_hit {
        Some(out) => out,
        None => {
            #[rustfmt::skip]
            let (out, _connector_out) = route_context_through_connector(
                &gpu, head, "video_embeddings_connector", cfg.use_embeddings_connector, context, context_valid, context_len as u32,
                cfg.connector_inner_dim(), cfg.connector_num_attention_heads, cfg.connector_attention_head_dim,
                cfg.connector_num_layers, cfg.connector_num_learnable_registers, cfg.connector_apply_gated_attention,
                cfg.connector_norm_output, cfg.positional_embedding_theta, &cfg.connector_positional_embedding_max_pos, cfg.norm_eps,
            );
            cache.connector_store(context, context_valid, context_len, &out);
            out
        }
    };
    gpu_core::profile::stage_time("forward_q_streamed: embeddings connector routing", s_conn);
    tracing::debug!(
        stage = "connector",
        ms = s_conn.elapsed().as_secs_f32() * 1e3,
        enabled = cfg.use_embeddings_connector,
        layers = cfg.connector_num_layers,
        cache = if conn_was_hit { "hit" } else { "miss" },
        "host stage done"
    );

    // Phase 8 attribution: `forward_q_streamed` was never profiled against a
    // real checkpoint before this pass, so the ~200s/step number this design
    // implies (re-reading and re-quantizing every one of `cfg.num_layers`
    // blocks from `src` on EVERY forward call) had never been split into its
    // three real components. Accumulate each block's own three stage
    // durations across the whole loop and report ONE coarse total per stage
    // under `BRAIN_PROFILE` (`gpu_core::profile::stage_time`'s own
    // convention: a coarse timeline, not a per-iteration log) - this is what
    // tells apart hypothesis (a) GPU dispatch/compute, (b) GGUF
    // read+dequantize-to-fp32 I/O, and (c) host-side int8 quantize+upload.
    let mut t_load = std::time::Duration::ZERO;
    let mut t_quant = std::time::Duration::ZERO;
    let mut t_gpu = std::time::Duration::ZERO;
    let mut misses = 0u32;
    for l in 0..cfg.num_layers {
        let prefix = format!("transformer_blocks.{l}");
        let mut layer_load = std::time::Duration::ZERO;
        let mut layer_quant = std::time::Duration::ZERO;
        // The cache hands back an `Arc` and holds no lock while this call
        // uploads, which is what lets two CFG branches read one checkpoint's
        // cache concurrently (see `crate::weightcache`'s doc). Every step
        // past this block's first cache-populating one skips BOTH stages in
        // the miss arm entirely - `t_load`/`t_quant` and their real GGUF-read
        // and CPU-quantize work are simply not incurred, so this loop does
        // LESS work on a hit rather than merely faster work.
        let hit = cache.block(l as usize, tier);
        let cached = match hit.clone() {
            Some(c) => c,
            None => {
                misses += 1;
                let s0 = std::time::Instant::now();
                let block_tensors = load_block_tensors_from_source(src, cfg, &prefix);
                layer_load = s0.elapsed();
                t_load += layer_load;
                let s1 = std::time::Instant::now();
                let quantized = QBlockWeights::quantize_host(&block_tensors, &prefix, dim, cfg.apply_gated_attention, tier);
                layer_quant = s1.elapsed();
                t_quant += layer_quant;
                cache.store_block(l as usize, tier, quantized)
            }
        };
        let hit = hit.is_some();
        // Device upload (`on_cached`) + GPU forward + wait, timed together as
        // one bucket: `on_cached` never does CPU quantization (that is
        // entirely inside the cache-miss branch above), only device writes of
        // already-quantized bytes, so this bucket is what remains on EVERY
        // step regardless of cache hit/miss - the honest lower bound the
        // "fresh Gpu every call" design (this function's own doc) still pays.
        let s2 = std::time::Instant::now();
        let blk = LtxBlockQ::on_cached(gpu.share(), cfg, &cached, t as u32, context_len as u32, tier);
        let (out, _tp) = blk.forward(&x, &adaln_table, &connector_context, &cos_bufs, &sin_bufs, t as u32);
        let layer_gpu = s2.elapsed();
        t_gpu += layer_gpu;
        // Level 5, per layer: which layer, cache hit or miss, and how long
        // each of the three real costs took. `hit` is the difference between
        // "this step re-read 270 MB off the GGUF and re-quantized it" and
        // "this step only re-uploaded bytes it already had" - the single most
        // load-bearing fact about a streamed step's cost.
        tracing::trace!(
            layer = l,
            cache = if hit { "hit" } else { "miss" },
            load_ms = layer_load.as_secs_f32() * 1e3,
            quant_ms = layer_quant.as_secs_f32() * 1e3,
            gpu_ms = layer_gpu.as_secs_f32() * 1e3,
            "block done"
        );
        x = out;
    }
    gpu_core::profile::stage_time("forward_q_streamed: block GGUF read+dequant (sum over all layers, cache misses only)", std::time::Instant::now() - t_load);
    gpu_core::profile::stage_time("forward_q_streamed: block int8 quantize (sum over all layers, cache misses only)", std::time::Instant::now() - t_quant);
    gpu_core::profile::stage_time("forward_q_streamed: block GPU upload+forward+wait (sum over all layers, every step)", std::time::Instant::now() - t_gpu);
    // The same three sums `BRAIN_PROFILE` already prints, restated as trace
    // fields so a run can be analysed from ONE stream instead of correlating
    // two mechanisms by hand. `BRAIN_PROFILE` is untouched: perf gates
    // elsewhere in the repo parse its output, and consolidating the two is a
    // later decision, not a side effect of adding tracing.
    tracing::debug!(
        cache_misses = misses,
        cache_hits = cfg.num_layers - misses,
        load_ms = t_load.as_secs_f32() * 1e3,
        quant_ms = t_quant.as_secs_f32() * 1e3,
        gpu_ms = t_gpu.as_secs_f32() * 1e3,
        "block stack done"
    );

    output_stage(head, "scale_shift_table", "proj_out", &x, &embedded_timestep, t, dim, cfg.out_channels as usize, cfg.norm_eps)
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

/// One pipeline-stage/training batch for [`LtxAvDit`] - [`DitBatch`]'s AV
/// counterpart: both streams' own inputs, since the A<->V coupling means a
/// stage needs BOTH latents/contexts/sigmas to run its own block range, not
/// just one. See `crate::shard`'s module doc for why the stage boundary
/// carries TWO residuals (video's and audio's), not one.
pub struct AvDitBatch {
    pub v_latent: Vec<f32>,
    pub v_timesteps: Vec<f32>,
    pub v_positions: Vec<f32>,
    pub v_keyframes_mask: Vec<f32>,
    pub v_context: Vec<f32>,
    pub v_context_len: usize,
    pub tv: usize,
    pub v_sigma: f32,
    pub v_context_valid: Vec<f32>,
    pub a_latent: Vec<f32>,
    pub a_timesteps: Vec<f32>,
    pub a_positions: Vec<f32>,
    pub a_context: Vec<f32>,
    pub a_context_len: usize,
    pub ta: usize,
    pub a_sigma: f32,
    pub a_context_valid: Vec<f32>,
    /// `[tv, video.out_channels]` training target; `None` for a forward-only
    /// run.
    pub v_target: Option<Vec<f32>>,
    /// `[ta, audio.out_channels]` training target; `None` for a forward-only
    /// run.
    pub a_target: Option<Vec<f32>>,
}

/// The tiny audio+video DiT, weights resident on the host (device residency
/// is per-block, see [`LtxAvBlock::on`]) - `LTXModelType::AudioVideo`.
pub struct LtxAvDit {
    cfg: LtxAvDitConfig,
    w: Tensors,
    device: Option<String>,
    /// Pipeline-parallel placement (`Shard::whole` for the ordinary,
    /// non-sharded path every other caller in this crate uses) - see
    /// `crate::shard`'s module doc.
    shard: Shard,
    batch: RefCell<Option<AvDitBatch>>,
    /// This stage's INPUT-side residuals (`res[shard.start]`, one per
    /// stream), written by the previous stage via [`LtxAvDit::write_in_res`].
    res_in_v: RefCell<Option<Vec<f32>>>,
    res_in_a: RefCell<Option<Vec<f32>>>,
    /// This stage's OUTPUT-side residuals (`res[shard.end]`, PRE-output-stage,
    /// one per stream) - set by [`LtxAvDit::run_stage_forward`], read by
    /// [`LtxAvDit::read_out_res`].
    res_out_v: RefCell<Option<Vec<f32>>>,
    res_out_a: RefCell<Option<Vec<f32>>>,
    /// The head stage's last `output_stage` results (video, audio) - the
    /// actual generation output, distinct from `res_out_*` (which never runs
    /// through `output_stage`). Read by [`LtxAvDit::take_stage_output`].
    stage_out_v: RefCell<Option<Vec<f32>>>,
    stage_out_a: RefCell<Option<Vec<f32>>>,
}

impl LtxAvDit {
    pub fn new(cfg: LtxAvDitConfig, weights: Tensors, device: Option<&str>) -> LtxAvDit {
        let shard = Shard::whole(cfg.video.num_layers as usize);
        LtxAvDit::build(cfg, weights, device, shard)
    }

    /// Shared constructor - see [`LtxDit::build`]'s doc for the same "zero
    /// the pipeline-stage scratch regardless of whether this is ever
    /// actually run as a shard" reasoning.
    fn build(cfg: LtxAvDitConfig, weights: Tensors, device: Option<&str>, shard: Shard) -> LtxAvDit {
        cfg.assert_supported();
        LtxAvDit {
            cfg,
            w: weights,
            device: device.map(str::to_string),
            shard,
            batch: RefCell::new(None),
            res_in_v: RefCell::new(None),
            res_in_a: RefCell::new(None),
            res_out_v: RefCell::new(None),
            res_out_a: RefCell::new(None),
            stage_out_v: RefCell::new(None),
            stage_out_a: RefCell::new(None),
        }
    }

    /// [`LtxDit::from_flat_weights`]'s AV counterpart - build one pipeline
    /// stage from a FLAT checkpoint, loading only [`av_shard_owns_weight`]'s
    /// subset for `shard`.
    pub(crate) fn from_flat_weights(cfg: LtxAvDitConfig, init: &HashMap<String, Vec<f32>>, shard: Shard) -> LtxAvDit {
        let w = av_shard_weights(&cfg, init, &shard);
        LtxAvDit::build(cfg, w, None, shard)
    }

    pub fn config(&self) -> &LtxAvDitConfig {
        &self.cfg
    }

    /// This instance's pipeline placement.
    pub fn shard(&self) -> &Shard {
        &self.shard
    }

    /// Set this stage's diffusion batch - see [`AvDitBatch`]'s doc.
    pub fn load_shard_batch(&self, b: AvDitBatch) {
        *self.batch.borrow_mut() = Some(b);
    }

    /// The head stage's last `output_stage` results, `(video, audio)` - the
    /// actual generation output, distinct from the `Shardable` residual seam.
    pub fn take_stage_output(&self) -> (Vec<f32>, Vec<f32>) {
        let v = self.stage_out_v.borrow().clone().expect("LtxAvDit::take_stage_output: run_stage_forward has not produced a head-stage video output yet");
        let a = self.stage_out_a.borrow().clone().expect("LtxAvDit::take_stage_output: run_stage_forward has not produced a head-stage audio output yet");
        (v, a)
    }

    pub(crate) fn weight_names(&self) -> Vec<String> {
        self.w.keys().cloned().collect()
    }
    pub(crate) fn weight(&self, name: &str) -> Vec<f32> {
        tget(&self.w, name).to_vec()
    }

    /// `model::Shardable::read_out_res`'s implementation: this stage's
    /// OUTPUT-side residual, video's `[tv*vdim]` then audio's `[ta*adim]`
    /// concatenated into ONE `Vec<f32>` - the `Shardable` trait carries a
    /// single boundary buffer, so the AV coupling's TWO residuals (this
    /// module's own doc) are packed into it at a split point [`Self::
    /// write_in_res`] recovers from the (replicated) batch's own `tv`/`ta`
    /// and `cfg`'s own `vdim`/`adim` - never from the wire itself.
    pub(crate) fn read_out_res(&self) -> Vec<f32> {
        let v = self.res_out_v.borrow().clone().expect("LtxAvDit::read_out_res: run_stage_forward has not run yet (video)");
        let a = self.res_out_a.borrow().clone().expect("LtxAvDit::read_out_res: run_stage_forward has not run yet (audio)");
        [v, a].concat()
    }
    /// `model::Shardable::write_in_res`'s implementation - the inverse split
    /// of [`Self::read_out_res`]'s concatenation.
    pub(crate) fn write_in_res(&self, data: &[f32]) {
        let batch_ref = self.batch.borrow();
        let batch = batch_ref.as_ref().expect("LtxAvDit::write_in_res: no batch (call load_shard_batch first)");
        let vlen = batch.tv * self.cfg.video.inner_dim as usize;
        let alen = batch.ta * self.cfg.audio.inner_dim as usize;
        assert_eq!(data.len(), vlen + alen, "LtxAvDit::write_in_res: boundary residual length mismatch (got {}, want {vlen}+{alen})", data.len());
        *self.res_in_v.borrow_mut() = Some(data[..vlen].to_vec());
        *self.res_in_a.borrow_mut() = Some(data[vlen..].to_vec());
    }

    /// Run this stage's forward - the AV counterpart of [`LtxDit::
    /// run_stage_forward`]: patchify both streams (embed stage only) or
    /// read both streams' previous-stage residuals ([`Self::write_in_res`])
    /// -> this stage's own block range ([`Self::forward_blocks_av`]) ->
    /// `output_stage` for both streams (head stage only). Every stage
    /// independently recomputes both streams' adaLN tables, the four RoPE
    /// tables, and (when enabled) both embeddings connectors from `batch`
    /// and its OWN (replicated) weights - only the two residuals cross the
    /// stage boundary, concatenated per [`Self::read_out_res`]'s doc.
    /// Returns the head stage's combined loss (mean of both streams' MSE
    /// where a target is set for that stream, `None` if neither stream has
    /// one, or on a non-head stage).
    pub fn run_stage_forward(&self) -> Option<f32> {
        let cfg = &self.cfg;
        let vcfg = &cfg.video;
        let acfg = &cfg.audio;
        let vdim = vcfg.inner_dim as usize;
        let adim = acfg.inner_dim as usize;
        let batch_ref = self.batch.borrow();
        let batch = batch_ref.as_ref().expect("LtxAvDit::run_stage_forward: no batch (call load_shard_batch first)");
        if self.shard.embed {
            assert_eq!(batch.v_latent.len(), batch.tv * vcfg.in_channels as usize, "LtxAvDit::run_stage_forward: video latent length mismatch");
            assert_eq!(batch.a_latent.len(), batch.ta * acfg.in_channels as usize, "LtxAvDit::run_stage_forward: audio latent length mismatch");
        }

        let v_ts_scaled: Vec<f32> = batch.v_timesteps.iter().map(|&x| x * vcfg.timestep_scale_multiplier as f32).collect();
        let a_ts_scaled: Vec<f32> = batch.a_timesteps.iter().map(|&x| x * vcfg.timestep_scale_multiplier as f32).collect();
        let (v_adaln_table, v_embedded_timestep) = ada_layer_norm_single(&self.w, "adaln_single", &v_ts_scaled, vdim, vcfg.adaln_rows() as usize);
        let (a_adaln_table, a_embedded_timestep) = ada_layer_norm_single(&self.w, "audio_adaln_single", &a_ts_scaled, adim, vcfg.adaln_rows() as usize);

        let (av_video_ss_table, _) = ada_layer_norm_single(&self.w, "av_ca_video_scale_shift_adaln_single", &v_ts_scaled, vdim, 4);
        let (av_audio_ss_table, _) = ada_layer_norm_single(&self.w, "av_ca_audio_scale_shift_adaln_single", &a_ts_scaled, adim, 4);

        let a2v_gate_ts = [batch.a_sigma * cfg.av_ca_timestep_scale_multiplier];
        let v2a_gate_ts = [batch.v_sigma * cfg.av_ca_timestep_scale_multiplier];
        let (av_a2v_gate_table, _) = ada_layer_norm_single(&self.w, "av_ca_a2v_gate_adaln_single", &a2v_gate_ts, vdim, 1);
        let (av_v2a_gate_table, _) = ada_layer_norm_single(&self.w, "av_ca_v2a_gate_adaln_single", &v2a_gate_ts, adim, 1);

        let v_rope = ltx_rope_tables(vcfg.inner_dim, vcfg.num_heads, vcfg.positional_embedding_theta, &vcfg.positional_embedding_max_pos, &batch.v_positions, batch.tv);
        let a_rope = ltx_rope_tables(acfg.inner_dim, acfg.num_heads, vcfg.positional_embedding_theta, &acfg.positional_embedding_max_pos, &batch.a_positions, batch.ta);
        let cross_max_pos = [cfg.cross_pe_max_pos()];
        let v_axis0_positions = &batch.v_positions[0..batch.tv * 2];
        let v_cross_rope = ltx_rope_tables(acfg.cross_attention_dim, acfg.num_heads, vcfg.positional_embedding_theta, &cross_max_pos, v_axis0_positions, batch.tv);
        let a_cross_rope = ltx_rope_tables(acfg.cross_attention_dim, acfg.num_heads, vcfg.positional_embedding_theta, &cross_max_pos, &batch.a_positions, batch.ta);

        let gpu = open_device(self.device.as_deref());
        let (v_cos_bufs, v_sin_bufs) = upload_rope_tables(&gpu, &v_rope);
        let (a_cos_bufs, a_sin_bufs) = upload_rope_tables(&gpu, &a_rope);
        let (v_cross_cos_bufs, v_cross_sin_bufs) = upload_rope_tables(&gpu, &v_cross_rope);
        let (a_cross_cos_bufs, a_cross_sin_bufs) = upload_rope_tables(&gpu, &a_cross_rope);

        #[rustfmt::skip]
        let (v_connector_context, _v_connector_out) = route_context_through_connector(
            &gpu, &self.w, "video_embeddings_connector", vcfg.use_embeddings_connector, &batch.v_context, &batch.v_context_valid, batch.v_context_len as u32,
            vcfg.connector_inner_dim(), vcfg.connector_num_attention_heads, vcfg.connector_attention_head_dim,
            vcfg.connector_num_layers, vcfg.connector_num_learnable_registers, vcfg.connector_apply_gated_attention,
            vcfg.connector_norm_output, vcfg.positional_embedding_theta, &vcfg.connector_positional_embedding_max_pos, vcfg.norm_eps,
        );
        #[rustfmt::skip]
        let (a_connector_context, _a_connector_out) = route_context_through_connector(
            &gpu, &self.w, "audio_embeddings_connector", vcfg.use_embeddings_connector, &batch.a_context, &batch.a_context_valid, batch.a_context_len as u32,
            acfg.connector_inner_dim(), acfg.connector_num_attention_heads, acfg.connector_attention_head_dim,
            vcfg.connector_num_layers, vcfg.connector_num_learnable_registers, vcfg.connector_apply_gated_attention,
            vcfg.connector_norm_output, vcfg.positional_embedding_theta, &vcfg.connector_positional_embedding_max_pos, vcfg.norm_eps,
        );

        let (vx0, ax0) = if self.shard.embed {
            let pw = tget(&self.w, "patchify_proj.weight");
            let pb = tget(&self.w, "patchify_proj.bias");
            let mut vx = linear(&batch.v_latent, batch.tv, vcfg.in_channels as usize, pw, Some(pb), vdim);
            if vcfg.use_keyframes_abs_pos_embedding {
                let kf = tget(&self.w, "keyframes_abs_pos_embedding");
                for ti in 0..batch.tv {
                    if batch.v_keyframes_mask[ti] > 0.0 {
                        for d in 0..vdim {
                            vx[ti * vdim + d] += kf[d];
                        }
                    }
                }
            }
            let apw = tget(&self.w, "audio_patchify_proj.weight");
            let apb = tget(&self.w, "audio_patchify_proj.bias");
            let ax = linear(&batch.a_latent, batch.ta, acfg.in_channels as usize, apw, Some(apb), adim);
            (vx, ax)
        } else {
            (
                self.res_in_v.borrow().clone().expect("LtxAvDit::run_stage_forward: non-embed stage needs write_in_res first (video)"),
                self.res_in_a.borrow().clone().expect("LtxAvDit::run_stage_forward: non-embed stage needs write_in_res first (audio)"),
            )
        };

        #[rustfmt::skip]
        let (vx_final, ax_final, _v_block_out, _a_block_out, _taps) = self.forward_blocks_av(
            &gpu, &vx0, &ax0, &v_adaln_table, &a_adaln_table, &v_connector_context, &a_connector_context,
            &v_cos_bufs, &v_sin_bufs, &a_cos_bufs, &a_sin_bufs,
            &v_cross_cos_bufs, &v_cross_sin_bufs, &a_cross_cos_bufs, &a_cross_sin_bufs,
            &av_video_ss_table, &av_audio_ss_table, &av_a2v_gate_table, &av_v2a_gate_table,
            batch.tv as u32, batch.ta as u32, self.shard.start as u32, self.shard.end as u32,
        );
        *self.res_out_v.borrow_mut() = Some(vx_final.clone());
        *self.res_out_a.borrow_mut() = Some(ax_final.clone());

        if self.shard.head {
            let v_out = output_stage(&self.w, "scale_shift_table", "proj_out", &vx_final, &v_embedded_timestep, batch.tv, vdim, vcfg.out_channels as usize, vcfg.norm_eps);
            let a_out = output_stage(&self.w, "audio_scale_shift_table", "audio_proj_out", &ax_final, &a_embedded_timestep, batch.ta, adim, acfg.out_channels as usize, vcfg.norm_eps);
            *self.stage_out_v.borrow_mut() = Some(v_out.clone());
            *self.stage_out_a.borrow_mut() = Some(a_out.clone());
            let v_loss = batch.v_target.as_ref().map(|target| {
                assert_eq!(v_out.len(), target.len(), "LtxAvDit::run_stage_forward: video target length mismatch");
                v_out.iter().zip(target).map(|(o, g)| (o - g) * (o - g)).sum::<f32>() / v_out.len().max(1) as f32
            });
            let a_loss = batch.a_target.as_ref().map(|target| {
                assert_eq!(a_out.len(), target.len(), "LtxAvDit::run_stage_forward: audio target length mismatch");
                a_out.iter().zip(target).map(|(o, g)| (o - g) * (o - g)).sum::<f32>() / a_out.len().max(1) as f32
            });
            match (v_loss, a_loss) {
                (Some(v), Some(a)) => Some((v + a) * 0.5),
                (Some(v), None) => Some(v),
                (None, Some(a)) => Some(a),
                (None, None) => None,
            }
        } else {
            None
        }
    }

    /// One forward pass, replaying the golden's own inputs - both streams,
    /// full bidirectional cross-attention every block.
    ///
    /// `v_*`/`a_*` mirror [`LtxDit::forward`]'s params, one set per stream
    /// (audio has no `keyframes_mask` - that feature is video-only, see
    /// `crate::config`'s doc). `v_sigma`/`a_sigma`: each stream's SCALAR
    /// sigma (`Modality.sigma`, `[1]`) - the CROSS modality's sigma is what
    /// drives the other stream's AV gate (see `crate::block`'s doc).
    /// `v_context_valid`/`a_context_valid`: each stream's own [`LtxDit::
    /// forward`]-style connector validity mask - ignored when `cfg.video.
    /// use_embeddings_connector` is `false`.
    /// Run transformer blocks `[lo, hi)` over both streams, returning the
    /// resulting `(vx, ax)` plus each block's own output copy (one stream
    /// each) and full [`AvBlockTaps`] (one entry per block in range, index 0
    /// = block `lo`) - the AV counterpart of [`LtxDit::forward_blocks`], the
    /// SAME "one function, different bounds" single source of truth shared
    /// by [`Self::forward`]'s own `[0, num_layers)` range and
    /// [`Self::run_stage_forward`]'s `[shard.start, shard.end)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn forward_blocks_av(
        &self,
        gpu: &Gpu,
        vx: &[f32],
        ax: &[f32],
        v_adaln_table: &[f32],
        a_adaln_table: &[f32],
        v_context: &[f32],
        a_context: &[f32],
        v_cos_bufs: &[DeviceBuffer],
        v_sin_bufs: &[DeviceBuffer],
        a_cos_bufs: &[DeviceBuffer],
        a_sin_bufs: &[DeviceBuffer],
        v_cross_cos_bufs: &[DeviceBuffer],
        v_cross_sin_bufs: &[DeviceBuffer],
        a_cross_cos_bufs: &[DeviceBuffer],
        a_cross_sin_bufs: &[DeviceBuffer],
        av_video_ss_table: &[f32],
        av_audio_ss_table: &[f32],
        av_a2v_gate_table: &[f32],
        av_v2a_gate_table: &[f32],
        tv: u32,
        ta: u32,
        lo: u32,
        hi: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<AvBlockTaps>) {
        let vcfg = &self.cfg.video;
        let acfg = &self.cfg.audio;
        let mut vxx = vx.to_vec();
        let mut axx = ax.to_vec();
        let mut v_block_out = Vec::with_capacity((hi - lo) as usize);
        let mut a_block_out = Vec::with_capacity((hi - lo) as usize);
        let mut taps = Vec::with_capacity((hi - lo) as usize);
        for l in lo..hi {
            let blk = LtxAvBlock::on(gpu.share(), vcfg, acfg, &self.w, &format!("transformer_blocks.{l}"), (v_context.len() / vcfg.inner_dim as usize) as u32, (a_context.len() / acfg.inner_dim as usize) as u32);
            #[rustfmt::skip]
            let (vout, aout, tp) = blk.forward(
                &vxx, &axx, v_adaln_table, a_adaln_table, v_context, a_context,
                v_cos_bufs, v_sin_bufs, a_cos_bufs, a_sin_bufs,
                v_cross_cos_bufs, v_cross_sin_bufs, a_cross_cos_bufs, a_cross_sin_bufs,
                av_video_ss_table, av_audio_ss_table, av_a2v_gate_table, av_v2a_gate_table,
                tv, ta,
            );
            vxx = vout;
            axx = aout;
            v_block_out.push(vxx.clone());
            a_block_out.push(axx.clone());
            taps.push(tp);
        }
        (vxx, axx, v_block_out, a_block_out, taps)
    }

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
        v_context_valid: &[f32],
        a_latent: &[f32],
        a_timesteps: &[f32],
        a_positions: &[f32],
        a_context: &[f32],
        a_context_len: usize,
        ta: usize,
        a_sigma: f32,
        a_context_valid: &[f32],
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

        // ---- each stream's own embeddings connector (crate::block::
        // EmbeddingsConnector's doc) - vcfg carries the shared layer/
        // register/max-pos/norm-output/gate fields for BOTH connectors
        // (`dit.rs::push_connector`'s doc: the real checkpoint's
        // `audio_embeddings_connector` reuses video's own `connector_num_
        // layers` etc.), only the per-stream geometry (dim/heads/head_dim)
        // and weight prefix differ.
        #[rustfmt::skip]
        let (v_connector_context, v_connector_out) = route_context_through_connector(
            &gpu, &self.w, "video_embeddings_connector", vcfg.use_embeddings_connector, v_context, v_context_valid, v_context_len as u32,
            vcfg.connector_inner_dim(), vcfg.connector_num_attention_heads, vcfg.connector_attention_head_dim,
            vcfg.connector_num_layers, vcfg.connector_num_learnable_registers, vcfg.connector_apply_gated_attention,
            vcfg.connector_norm_output, vcfg.positional_embedding_theta, &vcfg.connector_positional_embedding_max_pos, vcfg.norm_eps,
        );
        #[rustfmt::skip]
        let (a_connector_context, a_connector_out) = route_context_through_connector(
            &gpu, &self.w, "audio_embeddings_connector", vcfg.use_embeddings_connector, a_context, a_context_valid, a_context_len as u32,
            acfg.connector_inner_dim(), acfg.connector_num_attention_heads, acfg.connector_attention_head_dim,
            vcfg.connector_num_layers, vcfg.connector_num_learnable_registers, vcfg.connector_apply_gated_attention,
            vcfg.connector_norm_output, vcfg.positional_embedding_theta, &vcfg.connector_positional_embedding_max_pos, vcfg.norm_eps,
        );

        #[rustfmt::skip]
        let (vx_final, ax_final, v_block_out, a_block_out, mut av_taps) = self.forward_blocks_av(
            &gpu, &vx, &ax, &v_adaln_table, &a_adaln_table, &v_connector_context, &a_connector_context,
            &v_cos_bufs, &v_sin_bufs, &a_cos_bufs, &a_sin_bufs,
            &v_cross_cos_bufs, &v_cross_sin_bufs, &a_cross_cos_bufs, &a_cross_sin_bufs,
            &av_video_ss_table, &av_audio_ss_table, &av_a2v_gate_table, &av_v2a_gate_table,
            tv as u32, ta as u32, 0, vcfg.num_layers,
        );
        vx = vx_final;
        ax = ax_final;
        assert!(!av_taps.is_empty(), "num_layers must be >= 1");
        let b0 = av_taps.remove(0);
        let b0v = BlockTaps { attn1_out: b0.v_attn1_out, attn2_out: b0.v_attn2_out, ff_out: b0.v_ff_out };
        let b0a = BlockTaps { attn1_out: b0.a_attn1_out, attn2_out: b0.a_attn2_out, ff_out: b0.a_ff_out };
        let b0_a2v = b0.a2v_out;
        let b0_v2a = b0.v2a_out;

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
                connector_out: v_connector_out,
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
                connector_out: a_connector_out,
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
