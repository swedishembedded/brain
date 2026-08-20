// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the LTX-2.5 Gemma-4-12B text tower.
//!
//! `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` is a UNIFIED text+vision+
//! audio checkpoint (686 tensors total, range-read and confirmed this
//! session), but LTX-2.5 only ever runs it as a TEXT encoder - the vision
//! tower (`vision_model.*`), the audio-UNDERSTANDING projector
//! (`audio_projector.*`, a *different* thing from
//! `text_embedding_projection.audio_aggregate_embed` below - that one feeds
//! raw audio INTO the model, this one is LTX's own projection OUT of the
//! model's text hidden states) and the unified `multi_modal_projector.*` are
//! never read by LTX's text-conditioning path and are explicitly out of
//! scope here (see [`classify`]'s doc). The checkpoint also embeds five
//! non-tensor ASSET byte blobs (`tokenizer_json` + four `hf_asset__*`
//! files) - these belong to a tokenizer-loading path, not this f32 weight
//! manifest.
//!
//! Real tensor names carry a `model.` prefix
//! (`model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`)
//! that this crate's own canonical name space (`crate::block`'s `tget`
//! calls, `crate::model::load_tiny_weights`) does not - [`import_gemma4`]
//! strips it, mirroring `ltxv::import::import_vae`'s optional `vae.` strip.
//! `text_embedding_projection.*` carries no such prefix in either name
//! space, so it passes through unchanged.
//!
//! Validates in **both directions** via [`validate_manifest`], the same
//! two-way-coverage discipline `ltxv::import`'s own private helper of the
//! same name uses: a missing tensor errors by name, an unused source tensor
//! errors by name, a shape mismatch errors with both shapes, and (unlike
//! `ltxv::import`, which has no unrecognized-name-space case to worry
//! about) any tensor [`classify`] does not recognize at ALL errors by name
//! too - the out-of-scope/asset skip list is a closed, named set, not a
//! catch-all.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

use crate::block::Tensors;
use crate::config::Gemma4Config;

/// LTX's own `text_embedding_projection.video_aggregate_embed` output
/// width. `crate::model::AggregateEmbed`'s video head feeds the DiT video
/// stream's `inner_dim` (`LtxDitConfig::ltx25_22b().inner_dim` in
/// `crates/ltxv`, which this crate does not depend on - see this module's
/// doc for why that dependency is deliberately absent). Confirmed against
/// the real header's `text_embedding_projection.video_aggregate_embed.weight`
/// shape `[4096, 188160]`.
pub const VIDEO_AGGREGATE_OUT_DIM: usize = 4096;

/// LTX's own `text_embedding_projection.audio_aggregate_embed` output
/// width - the audio DiT stream's `inner_dim` twin of
/// [`VIDEO_AGGREGATE_OUT_DIM`]. Confirmed against the real header's
/// `text_embedding_projection.audio_aggregate_embed.weight` shape
/// `[2048, 188160]`.
pub const AUDIO_AGGREGATE_OUT_DIM: usize = 2048;

/// Every tensor [`import_gemma4`] imports, at [`Gemma4Config`]'s own
/// canonical (post-`model.`-strip) names - the same space
/// [`crate::block::Gemma4Layer::on`]'s `tget` calls and
/// [`crate::model::Gemma4Model::forward`] already use.
///
/// One structural fact this manifest encodes that a per-layer-uniform
/// manifest could not: `full_attention` layers have **no** `v_proj` tensor
/// at all (`attention_k_eq_v`'s `v_proj: None` case, `crate::block`'s
/// Mismatch 2) - confirmed by the real header carrying `self_attn.v_proj.
/// weight` for every one of the 40 `sliding_attention` layers and NONE of
/// the 8 `full_attention` ones (layers 5/11/17/23/29/35/41/47), settling a
/// discrepancy against an earlier transcription of this checkpoint's
/// tensor list that had assumed every layer carries a `v_proj`.
pub fn gemma4_tensor_manifest(cfg: &Gemma4Config) -> Vec<(String, Vec<usize>)> {
    let hidden = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let mut m: Vec<(String, Vec<usize>)> = vec![("embed_tokens.weight".into(), vec![cfg.vocab_size as usize, hidden]), ("norm.weight".into(), vec![hidden])];

    for l in 0..cfg.num_hidden_layers {
        let p = format!("layers.{l}");
        let lt = cfg.layer_type(l);
        let head_dim = cfg.head_dim_for(lt) as usize;
        let kv_dim = cfg.kv_heads_for(lt) as usize * head_dim;
        let q_dim = heads * head_dim;

        m.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
        m.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
        m.push((format!("{p}.pre_feedforward_layernorm.weight"), vec![hidden]));
        m.push((format!("{p}.post_feedforward_layernorm.weight"), vec![hidden]));
        m.push((format!("{p}.layer_scalar"), vec![1]));

        m.push((format!("{p}.self_attn.q_proj.weight"), vec![q_dim, hidden]));
        m.push((format!("{p}.self_attn.k_proj.weight"), vec![kv_dim, hidden]));
        if !cfg.k_eq_v_for(lt) {
            m.push((format!("{p}.self_attn.v_proj.weight"), vec![kv_dim, hidden]));
        }
        m.push((format!("{p}.self_attn.o_proj.weight"), vec![hidden, q_dim]));
        m.push((format!("{p}.self_attn.q_norm.weight"), vec![head_dim]));
        m.push((format!("{p}.self_attn.k_norm.weight"), vec![head_dim]));

        m.push((format!("{p}.mlp.gate_proj.weight"), vec![cfg.intermediate_size as usize, hidden]));
        m.push((format!("{p}.mlp.up_proj.weight"), vec![cfg.intermediate_size as usize, hidden]));
        m.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, cfg.intermediate_size as usize]));
    }

    let n_states = cfg.num_hidden_layers as usize + 1;
    let agg_in = hidden * n_states;
    m.push(("text_embedding_projection.video_aggregate_embed.weight".into(), vec![VIDEO_AGGREGATE_OUT_DIM, agg_in]));
    m.push(("text_embedding_projection.video_aggregate_embed.bias".into(), vec![VIDEO_AGGREGATE_OUT_DIM]));
    m.push(("text_embedding_projection.audio_aggregate_embed.weight".into(), vec![AUDIO_AGGREGATE_OUT_DIM, agg_in]));
    m.push(("text_embedding_projection.audio_aggregate_embed.bias".into(), vec![AUDIO_AGGREGATE_OUT_DIM]));
    m
}

/// How one real-checkpoint tensor name routes into (or out of) this crate's
/// import - see this module's doc.
#[derive(Debug, PartialEq, Eq)]
enum SourceKind {
    /// An importable weight tensor, at its CANONICAL (post-`model.`-strip)
    /// name - the name [`gemma4_tensor_manifest`] itself uses.
    Weight(String),
    /// The vision tower, the audio-understanding projector, or the unified
    /// multi-modal projector - never read by LTX's text-only path.
    OutOfScope,
    /// A checkpoint-embedded asset byte blob (`tokenizer_json` or one of
    /// the `hf_asset__*` files) - not an f32 weight tensor at all.
    Asset,
    /// Anything this checkpoint family's real header never contained the
    /// last time this crate checked - a genuinely new/unexpected tensor,
    /// which [`import_gemma4`] refuses rather than silently dropping (the
    /// out-of-scope/asset skip lists are closed, named sets per this
    /// module's doc, not a catch-all).
    Unknown,
}

/// This crate's canonical name for a checkpoint tensor, or `None` when the
/// tensor is not a text-tower weight at all. The public face of
/// [`classify`]'s `Weight` arm, so a second loader (`crate::gguf_src`) can
/// agree with the importer about the name space by construction rather than
/// by a duplicated string rule.
pub fn canonical_weight_name(name: &str) -> Option<String> {
    match classify(name) {
        SourceKind::Weight(c) => Some(c),
        _ => None,
    }
}

/// True for a tensor this crate knowingly does not import: one of the
/// sibling towers LTX's text-only path never reads, or an embedded asset
/// blob. The complement of [`canonical_weight_name`] over everything
/// [`classify`] recognizes - so "recognized" stays one closed set defined in
/// one place, and an unrecognized name is still an error for every loader.
pub fn is_recognized_non_weight(name: &str) -> bool {
    matches!(classify(name), SourceKind::OutOfScope | SourceKind::Asset)
}

fn classify(name: &str) -> SourceKind {
    if let Some(rest) = name.strip_prefix("model.") {
        return SourceKind::Weight(rest.to_string());
    }
    if name.starts_with("text_embedding_projection.") {
        return SourceKind::Weight(name.to_string());
    }
    if name.starts_with("vision_model.") || name.starts_with("audio_projector.") || name.starts_with("multi_modal_projector.") {
        return SourceKind::OutOfScope;
    }
    if name == "tokenizer_json" || name.starts_with("hf_asset__") {
        return SourceKind::Asset;
    }
    SourceKind::Unknown
}

/// Check a name->tensor map against a manifest in both directions - the
/// same two-way-coverage shape `ltxv::import`'s own private helper of the
/// same name takes (missing tensor errors by name, unused source tensor
/// errors by name, shape mismatch errors with both shapes, nothing is ever
/// zero-filled).
fn validate_manifest(map: Tensors, manifest: &[(String, Vec<usize>)], who: &str) -> Result<Tensors, String> {
    for (name, shape) in manifest {
        match map.get(name) {
            None => return Err(format!("gemma4 {who} import: missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("gemma4 {who} import: {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!("gemma4 {who} import: {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("gemma4 {who} import: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// Import the Gemma-4 text tower from the real (or a synthetic, same-shape)
/// checkpoint's full flat tensor list: route every tensor through
/// [`classify`] (strip `model.`, pass `text_embedding_projection.*`
/// through, silently-but-documentedly skip the vision/audio/multi-modal
/// towers and the asset blobs, hard-error on anything unrecognized), then
/// validate two-way coverage of the surviving weight tensors against
/// [`gemma4_tensor_manifest`].
pub fn import_gemma4(tensors: Vec<StTensor>, cfg: &Gemma4Config) -> Result<Tensors, String> {
    let mut map: Tensors = HashMap::with_capacity(tensors.len());
    for t in tensors {
        match classify(&t.name) {
            SourceKind::Weight(canonical) => {
                map.insert(canonical, (t.shape, t.data));
            }
            SourceKind::OutOfScope | SourceKind::Asset => {}
            SourceKind::Unknown => return Err(format!("gemma4 import: unrecognized tensor {}", t.name)),
        }
    }
    validate_manifest(map, &gemma4_tensor_manifest(cfg), "text tower")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One fresh synthetic checkpoint, all-zero data at the manifest's own
    /// shapes, names re-prefixed with `model.` for `layers.*`/
    /// `embed_tokens.weight`/`norm.weight` (matching the real header) and
    /// left bare for `text_embedding_projection.*` (also matching the real
    /// header - see this module's doc). Mirrors `ltxv::import::tests::
    /// build`'s pattern.
    fn build(cfg: &Gemma4Config) -> Vec<StTensor> {
        gemma4_tensor_manifest(cfg)
            .into_iter()
            .map(|(n, s)| {
                let len = s.iter().product();
                let name = if n.starts_with("text_embedding_projection.") { n } else { format!("model.{n}") };
                StTensor { name, shape: s, data: vec![0.0f32; len] }
            })
            .collect()
    }

    #[test]
    fn classify_routes_every_real_header_name_space() {
        assert_eq!(classify("model.embed_tokens.weight"), SourceKind::Weight("embed_tokens.weight".into()));
        assert_eq!(classify("model.layers.5.self_attn.k_proj.weight"), SourceKind::Weight("layers.5.self_attn.k_proj.weight".into()));
        assert_eq!(
            classify("text_embedding_projection.video_aggregate_embed.weight"),
            SourceKind::Weight("text_embedding_projection.video_aggregate_embed.weight".into())
        );
        assert_eq!(classify("vision_model.patch_dense.weight"), SourceKind::OutOfScope);
        assert_eq!(classify("audio_projector.embedding_projection.weight"), SourceKind::OutOfScope);
        assert_eq!(classify("multi_modal_projector.embedding_projection.weight"), SourceKind::OutOfScope);
        assert_eq!(classify("tokenizer_json"), SourceKind::Asset);
        assert_eq!(classify("hf_asset__tokenizer_config.json"), SourceKind::Asset);
        assert_eq!(classify("something_never_seen_before"), SourceKind::Unknown);
    }

    /// Round-trips a synthetic checkpoint through [`import_gemma4`] at BOTH
    /// structurally different layer types (tiny config: 5 sliding + 1
    /// full), and exercises every coverage-failure path: missing, unused
    /// (including the closed OOS/asset skip list working alongside a
    /// genuinely unrecognized extra name), and wrong-shape.
    #[test]
    fn import_gemma4_validates_both_directions_and_skips_out_of_scope_and_assets() {
        let cfg = Gemma4Config::tiny();
        let manifest = gemma4_tensor_manifest(&cfg);
        // sliding (14: no v_proj exclusion) + full (13: v_proj excluded) - 5
        // sliding + 1 full is the tiny config's own real-ratio minimal
        // instance (`config`'s doc).
        assert_eq!(manifest.len(), 2 + 5 * 14 + 13 + 4);

        let mut with_extras = build(&cfg);
        with_extras.push(StTensor { name: "vision_model.pos_norm.bias".into(), shape: vec![2], data: vec![0.0; 2] });
        with_extras.push(StTensor { name: "hf_asset__chat_template.jinja".into(), shape: vec![3], data: vec![0.0; 3] });
        let w = import_gemma4(with_extras, &cfg).expect("out-of-scope and asset tensors must be silently skipped");
        assert_eq!(w.len(), manifest.len());
        drop(w);

        let unknown_err = import_gemma4(vec![StTensor { name: "totally_unrecognized.weight".into(), shape: vec![1], data: vec![0.0] }], &cfg).unwrap_err();
        assert!(unknown_err.contains("totally_unrecognized.weight"), "{unknown_err}");

        let mut missing = build(&cfg);
        missing.retain(|t| t.name != "model.layers.0.self_attn.q_proj.weight");
        let e = import_gemma4(missing, &cfg).unwrap_err();
        assert!(e.contains("layers.0.self_attn.q_proj.weight"), "{e}");

        let mut extra = build(&cfg);
        extra.push(StTensor { name: "model.layers.0.self_attn.q_proj.weight_v2".into(), shape: vec![1], data: vec![0.0] });
        let e = import_gemma4(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");

        let mut wrong = build(&cfg);
        if let Some(t) = wrong.iter_mut().find(|t| t.name == "model.norm.weight") {
            t.shape = vec![1, 1];
            t.data = vec![0.0; 1];
        }
        let e = import_gemma4(wrong, &cfg).unwrap_err();
        assert!(e.contains("norm.weight") && e.contains("expected"), "{e}");
    }

    /// Layer 5 (`full_attention`, tiny config's own instance of the real
    /// checkpoint's `attention_k_eq_v` layers) must NOT appear with a
    /// `v_proj` tensor in the manifest at all - the structural fact this
    /// module's doc records as settling a discrepancy against an earlier
    /// transcription of the real checkpoint's tensor list.
    #[test]
    fn full_attention_layers_have_no_v_proj_in_the_manifest() {
        let cfg = Gemma4Config::tiny();
        let manifest = gemma4_tensor_manifest(&cfg);
        assert!(!manifest.iter().any(|(n, _)| n == "layers.5.self_attn.v_proj.weight"));
        assert!(manifest.iter().any(|(n, _)| n == "layers.0.self_attn.v_proj.weight"));
    }

    /// [`gemma4_tensor_manifest`] at the REAL LTX-2.5 Gemma-4-12B config
    /// must produce exactly the real checkpoint's own importable-tensor
    /// count (670: 2 top-level + 664 per-layer + 4 aggregate-embed), and
    /// that count plus the real header's out-of-scope (11: 1
    /// `audio_projector` + 1 `multi_modal_projector` + 9 `vision_model`)
    /// and asset (5: `tokenizer_json` + 4 `hf_asset__*`) tensors must equal
    /// the real header's own total of 686 - the same self-verification
    /// discipline `ltxv::dit::av_dit_tensor_manifest`'s own real-header test
    /// uses, without materializing the ~54GB a real-shaped all-zero copy of
    /// this manifest's data would take (see `ltxv::import::tests::build`'s
    /// doc for why that concern is real even at the VAE's much smaller
    /// scale).
    #[test]
    fn manifest_matches_the_real_12b_checkpoint_header_count() {
        let cfg = Gemma4Config::gemma4_12b();
        let m = gemma4_tensor_manifest(&cfg);

        let top_level = m.iter().filter(|(n, _)| !n.starts_with("layers.") && !n.starts_with("text_embedding_projection.")).count();
        assert_eq!(top_level, 2, "embed_tokens.weight + norm.weight");
        let per_layer = m.iter().filter(|(n, _)| n.starts_with("layers.")).count();
        assert_eq!(per_layer, 40 * 14 + 8 * 13, "40 sliding * 14 (incl. v_proj) + 8 full * 13 (no v_proj)");
        let aggregate = m.iter().filter(|(n, _)| n.starts_with("text_embedding_projection.")).count();
        assert_eq!(aggregate, 4, "video/audio aggregate_embed, weight+bias each");

        assert_eq!(m.len(), 670, "total importable tensor count must match the real header's in-scope subset");

        const OUT_OF_SCOPE: usize = 11; // audio_projector(1) + multi_modal_projector(1) + vision_model.*(9)
        const ASSETS: usize = 5; // tokenizer_json + 4 hf_asset__*
        assert_eq!(m.len() + OUT_OF_SCOPE + ASSETS, 686, "importable + out-of-scope + assets must equal the real header's own total");
    }
}
