// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the LTX-2.x video VAE, audio VAE, base vocoder and
//! AV DiT.
//!
//! **Two packagings, one name space.** LTX-2.5 ships each component as its
//! own file; LTX-2.3 bundles the transformer (`model.diffusion_model.*`),
//! the video VAE (`vae.*`), the audio VAE (`audio_vae.*`), the vocoder
//! (`vocoder.*`) and the Gemma text projection
//! (`text_embedding_projection.*`, imported by `crates/gemma4`) into ONE
//! file. Every `import_*` below therefore selects its OWN prefix out of
//! whatever list it is handed and ignores the rest, so the same function
//! serves a split file and a bundle - and, because the two releases' VAE,
//! audio-VAE and vocoder tensors are name-for-name and shape-for-shape
//! IDENTICAL (both real headers range-read and diffed: 170 / 102 / 1227
//! tensors, zero differences), no 2.3-specific config exists for any of
//! them. Only the DiT differs, in two flags - see
//! [`crate::config::LtxDitConfig::ltx23_22b`].
//!
//! The real `ltx-2.5-video-vae-conv-bf16.safetensors` file is already in the
//! **canonical** (checkpoint-native) name space: bare `encoder.*` /
//! `decoder.*` / `per_channel_statistics.*` keys, no `vae.` prefix and no
//! further renaming - `ltx_core`'s own `VAE_ENCODER_COMFY_KEYS_FILTER` /
//! `VAE_DECODER_COMFY_KEYS_FILTER` only strip an optional `vae.` prefix and
//! (for the encoder/decoder-only case) a redundant `encoder.`/`decoder.`
//! double-prefix; this file carries neither, confirmed by reading its raw
//! safetensors header directly (170 keys, all already bare). So unlike Wan's
//! diffusers<->native rename, there is exactly one name space here and no
//! remapping table - `import_vae` is [`validate_manifest`] plus an optional
//! `vae.` prefix strip, for robustness against a future monolithic
//! (non-Comfy-split) release of the same checkpoint family.
//!
//! `ltx-2.5-audio-vae-bf16.safetensors` is a DIFFERENT file that carries
//! THREE subsystems in one flat tensor list: `audio_vae.{encoder,decoder,
//! per_channel_statistics}.*` (2.5.audio_vae's own combined encoder+decoder,
//! same "one file, one `import_*`" shape as the video VAE), `vocoder.vocoder.*`
//! (the BASE vocoder this milestone targets), and `vocoder.bwe_generator.*`/
//! `vocoder.mel_stft.*` (the bandwidth-extension stage, explicitly out of
//! scope - see `crate::vocoder`'s module doc). [`import_audio_vae`] and
//! [`import_vocoder`] each first FILTER the raw tensor list down to their own
//! prefix (silently ignoring every tensor belonging to the other two
//! subsystems - by construction, never a partial match), strip that prefix,
//! then run the same two-way [`validate_manifest`] scoped to their own
//! manifest. "Two-way coverage" therefore means "every tensor under this
//! import's own prefix is used, and nothing more" - not "every tensor in the
//! whole file", which would incorrectly demand the BWE stage's tensors too.
//!
//! Validates in **both directions**: a missing tensor errors by name, an
//! unused source tensor errors by name, and a shape mismatch errors with both
//! shapes. Nothing is ever zero-filled.

use checkpoint::gguf::MmapGguf;
use checkpoint::safetensors::StTensor;
use vae::blocks::Tensors;

use crate::audio_vae::AudioVaeConfig;
use crate::config::{LtxAudioDitConfig, LtxAvDitConfig, LtxDitConfig};
use crate::dit::{av_dit_tensor_manifest, dit_tensor_manifest};
use crate::duration_head::DurationHeadConfig;
use crate::upsampler::LatentUpsamplerConfig;
use crate::vae3d::LtxVaeConfig;
use crate::vocoder::VocoderConfig;

/// Check a name→tensor map against a manifest in both directions. Generic
/// over the manifest slice (not `LtxVaeConfig` specifically) so
/// [`import_audio_vae`]/[`import_vocoder`]/`crate::na_decoder::import_na_decoder`
/// reuse the same two-way check rather than restating it.
pub(crate) fn validate_manifest(map: Tensors, manifest: &[(String, Vec<usize>)], who: &str) -> Result<Tensors, String> {
    for (name, shape) in manifest {
        match map.get(name) {
            None => return Err(format!("ltxv {who} import: missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("ltxv {who} import: {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!("ltxv {who} import: {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("ltxv {who} import: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// Import the video VAE (encoder + conv decoder, combined - a single file
/// carries both).
///
/// Handles an optional `vae.` prefix the same way [`dit_name_space`] handles
/// the DiT's own: if ANY name carries it, it is a FILTER and non-matching
/// tensors are dropped; otherwise every tensor is taken bare. LTX-2.5 ships
/// the video VAE in a Comfy-split file whose 170 keys are already bare;
/// LTX-2.3 bundles the SAME 170 tensors (identical names, identical shapes -
/// both real headers were range-read and diffed) under `vae.` alongside the
/// transformer, the audio VAE, the vocoder and the text projection, so the
/// filter is what lets one import serve both packagings.
pub fn import_vae(tensors: Vec<StTensor>, cfg: &LtxVaeConfig) -> Result<Tensors, String> {
    let map: Tensors = if tensors.iter().any(|t| t.name.starts_with("vae.")) {
        tensors.into_iter().filter_map(|t| t.name.strip_prefix("vae.").map(|n| (n.to_string(), (t.shape, t.data)))).collect()
    } else {
        tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
    };
    // A real, correctly-named Lightricks release - `ltx-2.5-video-vae-
    // bf16.safetensors`, not the `-conv-` file this import targets - carries
    // `crate::na_decoder`'s architecture instead of this module's conv
    // decoder, under the SAME `decoder.*` top-level name every conv tensor
    // also uses. Naming that up front, before `validate_manifest` reports
    // the first conv tensor it happens to miss, is the difference between
    // "wrong file, here is why" and "this checkpoint looks damaged".
    if map.keys().any(|k| k.starts_with("decoder.det_stages") || k.starts_with("decoder.diff_blocks")) {
        return Err(
            "ltxv video vae import: this file's decoder is the NA/diffusion architecture \
             (decoder.det_stages.*/decoder.diff_blocks.* tensors found) - crate::na_decoder's \
             ported importer, not this module's conv decoder. Use the Lightricks release with \
             \"-conv-\" in its filename (e.g. ltx-2.5-video-vae-conv-bf16.safetensors) instead."
                .to_string(),
        );
    }
    validate_manifest(map, &cfg.tensor_manifest(), "video vae")
}

/// Import the audio VAE (encoder + decoder + shared per-channel stats) from
/// `ltx-2.5-audio-vae-bf16.safetensors`'s `audio_vae.*` tensors, strip that
/// prefix, and validate two-way coverage against
/// [`AudioVaeConfig::tensor_manifest`]. Tensors under any OTHER prefix
/// (`vocoder.*`) are silently skipped - they belong to [`import_vocoder`],
/// not here (see this module's header).
pub fn import_audio_vae(tensors: Vec<StTensor>, cfg: &AudioVaeConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors
        .into_iter()
        .filter_map(|t| t.name.strip_prefix("audio_vae.").map(|n| (n.to_string(), (t.shape, t.data))))
        .collect();
    validate_manifest(map, &cfg.tensor_manifest(), "audio vae")
}

/// Import the BASE vocoder (no BWE) from `ltx-2.5-audio-vae-bf16.safetensors`'s
/// `vocoder.vocoder.*` tensors, strip the doubled prefix down to the bare
/// names `Vocoder.state_dict()` expects (`conv_pre.*`, `ups.*`,
/// `resblocks.*`, `act_post.*`, `conv_post.*`), and validate two-way coverage
/// against [`VocoderConfig::tensor_manifest`]. `vocoder.bwe_generator.*` and
/// `vocoder.mel_stft.*` never match the `vocoder.vocoder.` prefix, so the BWE
/// weights are silently skipped by construction (mirrors the reference
/// dumper's own local `_VOCODER_BASE_KEYS_FILTER`, deliberately narrower than
/// `ltx_core`'s own `VOCODER_COMFY_KEYS_FILTER` which targets `VocoderWithBWE`).
pub fn import_vocoder(tensors: Vec<StTensor>, cfg: &VocoderConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors
        .into_iter()
        .filter_map(|t| t.name.strip_prefix("vocoder.vocoder.").map(|n| (n.to_string(), (t.shape, t.data))))
        .collect();
    validate_manifest(map, &cfg.tensor_manifest(), "vocoder")
}

/// Import a latent upscaler (spatial or temporal x2, `cfg` selects which)
/// from its own single-subsystem file - the real checkpoints carry bare
/// names (no `latent_upsampler.`-style prefix, confirmed against both real
/// 72-tensor headers), so this is [`validate_manifest`] with no renaming at
/// all, the same shape [`import_vae`] takes for the video VAE's own
/// already-bare checkpoint.
pub fn import_upsampler(tensors: Vec<StTensor>, cfg: &LatentUpsamplerConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    validate_manifest(map, &cfg.tensor_manifest(), "latent upsampler")
}

/// Import the duration head from `ltx-2.5-duration-head-bf16.safetensors`'s
/// `duration_head.*` tensors, strip that prefix, and validate two-way
/// coverage against [`DurationHeadConfig::tensor_manifest`] - the same
/// prefix-then-validate shape [`import_audio_vae`] takes.
pub fn import_duration_head(tensors: Vec<StTensor>, cfg: &DurationHeadConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors
        .into_iter()
        .filter_map(|t| t.name.strip_prefix("duration_head.").map(|n| (n.to_string(), (t.shape, t.data))))
        .collect();
    validate_manifest(map, &cfg.tensor_manifest(), "duration head")
}

/// Import the video-only DiT (`LtxDitConfig`'s canonical names - the same
/// space [`crate::block::LtxBlock`]'s `tget` calls and [`crate::dit::
/// dit_tensor_manifest`] already use) with two-way coverage. There is no
/// RENAMING table: this checkpoint family's own tensor names ARE the
/// reference module tree's, in every release and both file formats. The only
/// spelling difference is the optional [`DIT_BUNDLE_PREFIX`] the safetensors
/// releases carry, which [`dit_name_space`] selects on.
pub fn import_dit(tensors: Vec<StTensor>, cfg: &LtxDitConfig) -> Result<Tensors, String> {
    validate_manifest(dit_name_space(tensors), &dit_tensor_manifest(cfg), "dit")
}

/// Import the audio+video DiT (`LtxAvDitConfig`'s canonical names - see
/// [`crate::dit::av_dit_tensor_manifest`]'s doc) with two-way coverage. Same
/// no-renaming, optional-[`DIT_BUNDLE_PREFIX`] shape [`import_dit`] takes.
/// `cfg` selects the release: [`LtxAvDitConfig::ltx25`] expects
/// `keyframes_abs_pos_embedding` and a bias-free video FFN,
/// [`LtxAvDitConfig::ltx23`] the reverse - a checkpoint imported against the
/// wrong one fails by name in both directions rather than loading a
/// half-covered model.
pub fn import_av_dit(tensors: Vec<StTensor>, cfg: &LtxAvDitConfig) -> Result<Tensors, String> {
    validate_manifest(dit_name_space(tensors), &av_dit_tensor_manifest(cfg), "av dit")
}

// ---------------------------------------------------------------------------
// GGUF: direct loading + ahead-of-time conversion, sharing one manifest
// ---------------------------------------------------------------------------

/// The GGUF `general.architecture` value every LTX-2.x checkpoint carries -
/// `crates/arch`'s own `ltxv` row's id IS this spelling (`gguf: None`, see
/// that row's doc comment), so `brain_arch::by_gguf("ltxv")` resolves
/// without an alias.
pub const GGUF_ARCHITECTURE: &str = "ltxv";

/// The `model.diffusion_model.` prefix an LTX-2.x **safetensors** checkpoint
/// puts on every transformer tensor.
///
/// Both releases use it, but they package differently: LTX-2.5 ships the
/// transformer in a file of its own, so stripping the prefix leaves the file
/// fully covered, while LTX-2.3 bundles the transformer, both VAEs, the
/// vocoder and the text projection into ONE file - there, the prefix is a
/// FILTER, and the tensors that do not match belong to [`import_vae`] /
/// [`import_audio_vae`] / [`import_vocoder`] / `gemma4`'s own text-projection
/// import (all four of which already select their own prefix the same way).
/// The GGUF releases of both carry the bare names with no prefix at all, so
/// [`import_dit`]/[`import_av_dit`] accept either spelling.
pub const DIT_BUNDLE_PREFIX: &str = "model.diffusion_model.";

/// Reduce a raw checkpoint tensor list to the DiT's own bare name space.
///
/// If ANY name carries [`DIT_BUNDLE_PREFIX`], the prefix is treated as a
/// filter and every non-matching tensor is dropped (the LTX-2.3 single-file
/// bundle); otherwise every tensor is taken as-is (the GGUF-derived and
/// LTX-2.5 transformer-only files, already bare). Two-way coverage is then
/// asserted over whatever survives, so a bundle whose DiT subset is
/// incomplete still errors by name.
fn dit_name_space(tensors: Vec<StTensor>) -> Tensors {
    if tensors.iter().any(|t| t.name.starts_with(DIT_BUNDLE_PREFIX)) {
        tensors.into_iter().filter_map(|t| t.name.strip_prefix(DIT_BUNDLE_PREFIX).map(|n| (n.to_string(), (t.shape, t.data)))).collect()
    } else {
        tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
    }
}

/// The [`LtxAvDitConfig`] a checkpoint's own embedded `config` JSON names.
///
/// Unlike `wan::import::dit_config_from_shapes` (derived from tensor SHAPES,
/// because Wan's GGUF carries no config JSON), every real LTX-2.x checkpoint
/// embeds its full diffusers config as one JSON string - a `config` KV entry
/// in the GGUF releases, a `__metadata__["config"]` entry in the safetensors
/// ones, holding `{"transformer": {...}, ...}` - so this reads that JSON
/// directly rather than reverse-engineering shapes. Confirmed by
/// range-reading the real 22B headers of BOTH releases.
///
/// `has_tensor` answers "does this checkpoint carry a tensor with this bare
/// name". It is needed because **the config JSON is not complete**: the two
/// flags that differ between LTX-2.3 and LTX-2.5 are the two the 2.3 config
/// omits entirely.
///
/// * `ff_bias` - absent from 2.3's config, `false` in 2.5's. Resolved from
///   `transformer_blocks.0.ff.net.0.proj.bias`'s presence.
/// * `use_keyframes_abs_pos_embedding` - absent from 2.3's config, `true` in
///   2.5's. Resolved from `keyframes_abs_pos_embedding`'s presence.
///
/// Deriving them from the tensor list rather than defaulting them is this
/// port's "checkpoint reality wins over prose" rule applied where it
/// actually bites: the reference's own absent-key defaults
/// (`model_configurator.py`: `ff_bias=config.get("ff_bias", True)`,
/// `use_keyframes_abs_pos_embedding=config.get(..., False)`) agree with what
/// the real 2.3 header shows, so the two authorities corroborate each other
/// here - but only the tensor list stays right if a future release changes
/// its mind, and only the tensor list is what the manifest must match.
///
/// Every other field this crate models is read by name and errors if absent
/// or the wrong JSON type - never defaulted - with ONE exception that is a
/// known open question rather than a settled reading:
/// `use_prompt_adaln_single` is hardcoded `false` here because that is the
/// only path this crate implements ([`crate::config::LtxDitConfig::
/// use_prompt_adaln_single`], and `assert_supported` panics on `true`). It is
/// NOT read from the config, and it is not safe to assume it is right:
/// neither release's config sets the key, the reference's absent-key default
/// is `True`, and both real headers carry the `prompt_adaln_single.*` /
/// `audio_prompt_adaln_single.*` tensors the reference only builds when it IS
/// `True` - tensors the manifest demands (so import stays two-way complete)
/// and the forward never reads. If the reference reading is right, this
/// affects LTX-2.5 exactly as much as LTX-2.3, and only a real-weight forward
/// can settle it.
pub fn av_dit_config_from_json(config_json: &str, has_tensor: &dyn Fn(&str) -> bool) -> Result<LtxAvDitConfig, String> {
    let full: serde_json::Value = serde_json::from_str(config_json).map_err(|e| format!("ltxv gguf import: config KV is not valid JSON: {e}"))?;
    let t = &full["transformer"];
    if t.is_null() {
        return Err("ltxv gguf import: config KV has no 'transformer' object".into());
    }
    let u = |k: &str| -> Result<u32, String> { t[k].as_u64().map(|v| v as u32).ok_or_else(|| format!("ltxv gguf import: config.transformer.{k} missing or not an integer")) };
    let b = |k: &str| -> Result<bool, String> { t[k].as_bool().ok_or_else(|| format!("ltxv gguf import: config.transformer.{k} missing or not a bool")) };
    let f = |k: &str| -> Result<f64, String> { t[k].as_f64().ok_or_else(|| format!("ltxv gguf import: config.transformer.{k} missing or not a number")) };
    let arr_u32 = |k: &str, n: usize| -> Result<Vec<u32>, String> {
        let a = t[k].as_array().ok_or_else(|| format!("ltxv gguf import: config.transformer.{k} missing or not an array"))?;
        if a.len() != n {
            return Err(format!("ltxv gguf import: config.transformer.{k} has {} elements, expected {n}", a.len()));
        }
        a.iter().map(|v| v.as_u64().map(|x| x as u32).ok_or_else(|| format!("ltxv gguf import: config.transformer.{k} has a non-integer element"))).collect()
    };

    let video_heads = u("num_attention_heads")?;
    let video_head_dim = u("attention_head_dim")?;
    let audio_heads = u("audio_num_attention_heads")?;
    let audio_head_dim = u("audio_attention_head_dim")?;
    let in_channels = u("in_channels")?;
    let pos_max = arr_u32("positional_embedding_max_pos", 3)?;
    let audio_pos_max = arr_u32("audio_positional_embedding_max_pos", 1)?;
    let connector_pos_max = arr_u32("connector_positional_embedding_max_pos", 1)?;

    // Absent from LTX-2.3's config, present (`false`) in LTX-2.5's - see
    // this function's doc for why the tensor list, not a default, settles it.
    let ff_bias = t["ff_bias"].as_bool().unwrap_or_else(|| has_tensor("transformer_blocks.0.ff.net.0.proj.bias"));

    let video = LtxDitConfig {
        inner_dim: video_heads * video_head_dim,
        num_heads: video_heads,
        num_layers: u("num_layers")?,
        in_channels,
        out_channels: u("out_channels")?,
        cross_attention_dim: u("cross_attention_dim")?,
        ff_bias,
        cross_attention_adaln: b("cross_attention_adaln")?,
        use_prompt_adaln_single: false,
        use_keyframes_abs_pos_embedding: t["use_keyframes_abs_pos_embedding"].as_bool().unwrap_or_else(|| has_tensor("keyframes_abs_pos_embedding")),
        norm_eps: f("norm_eps")? as f32,
        positional_embedding_theta: f("positional_embedding_theta")?,
        positional_embedding_max_pos: [pos_max[0], pos_max[1], pos_max[2]],
        timestep_scale_multiplier: u("timestep_scale_multiplier")?,
        use_middle_indices_grid: b("use_middle_indices_grid")?,
        apply_gated_attention: b("apply_gated_attention")?,
        connector_num_layers: u("connector_num_layers")?,
        connector_num_attention_heads: u("connector_num_attention_heads")?,
        connector_attention_head_dim: u("connector_attention_head_dim")?,
        connector_num_learnable_registers: u("connector_num_learnable_registers")?,
        connector_positional_embedding_max_pos: [connector_pos_max[0]],
        connector_norm_output: b("connector_norm_output")?,
        caption_proj_before_connector: b("caption_proj_before_connector")?,
        connector_apply_gated_attention: b("connector_apply_gated_attention")?,
        // Not a real Lightricks `config.transformer` key (see
        // `LtxDitConfig::use_embeddings_connector`'s doc) - defaults `true`
        // when absent (every real checkpoint this importer parses carries
        // both embeddings connectors' tensors, checked two-way below
        // regardless), but honors an explicit value when present so THIS
        // crate's own re-serialized checkpoints (`import_gguf`'s output, or
        // a synthetic test fixture) round-trip exactly rather than silently
        // reverting to the default.
        use_embeddings_connector: t["use_embeddings_connector"].as_bool().unwrap_or(true),
    };
    let audio = LtxAudioDitConfig {
        inner_dim: audio_heads * audio_head_dim,
        num_heads: audio_heads,
        in_channels,
        out_channels: u("audio_out_channels")?,
        cross_attention_dim: u("audio_cross_attention_dim")?,
        // Mirrors the video stream's resolved flag. Nothing reads this
        // field (`crate::dit::push_ff`'s doc: audio's FFN bias is a
        // per-instance fact, and both releases carry it), and the
        // reference's own separate `audio_ff_bias` key is set by neither
        // release's config, so there is no independent value to transcribe.
        ff_bias,
        positional_embedding_max_pos: [audio_pos_max[0]],
        connector_num_attention_heads: u("audio_connector_num_attention_heads")?,
        connector_attention_head_dim: u("audio_connector_attention_head_dim")?,
    };
    let av_ca_timestep_scale_multiplier = f("av_ca_timestep_scale_multiplier")? as f32;
    Ok(LtxAvDitConfig { video, audio, av_ca_timestep_scale_multiplier })
}

/// [`av_dit_config_from_json`] over a GGUF's own `config` KV, after checking
/// the file really is an [`GGUF_ARCHITECTURE`] checkpoint. Tensor presence
/// is answered by the mmap itself, so the two absent-in-2.3 flags resolve
/// from the file being loaded rather than from a default.
pub fn av_dit_config_from_kv(mg: &MmapGguf) -> Result<LtxAvDitConfig, String> {
    let arch = mg.kv().get("general.architecture").and_then(|v| v.as_str());
    if arch != Some(GGUF_ARCHITECTURE) {
        return Err(format!("ltxv gguf import: general.architecture is {arch:?}, expected {GGUF_ARCHITECTURE:?}"));
    }
    let raw = mg.kv().get("config").and_then(|v| v.as_str()).ok_or("ltxv gguf import: no 'config' KV string (not an LTX-2.x checkpoint)")?;
    av_dit_config_from_json(raw, &|name| mg.shape(name).is_some())
}

/// Check a GGUF's own tensor SHAPES against [`av_dit_tensor_manifest`] in
/// both directions - no dequant, so a mismatched file is refused before a
/// single tensor is decoded (mirrors `wan::import::validate_dit_shapes`).
/// Shared by [`import_gguf`] and `crate::gguf_src::LtxvGgufSource::from_mmap`
/// so the two consumers validate against the exact same manifest call and
/// cannot silently drift apart - this crate's whole "one name table" answer
/// to the no-renaming case (see `crate::gguf_src`'s module doc).
pub(crate) fn validate_av_dit_gguf_shapes(mg: &MmapGguf, cfg: &LtxAvDitConfig) -> Result<(), String> {
    let manifest = av_dit_tensor_manifest(cfg);
    for (name, shape) in &manifest {
        let got = mg.shape(name).ok_or_else(|| format!("ltxv gguf import: missing tensor {name}"))?;
        let want_n: usize = shape.iter().product();
        let got_n: usize = got.iter().product();
        if got_n != want_n {
            return Err(format!("ltxv gguf import: {name} has {got_n} elements (shape {got:?}), expected {want_n} (shape {shape:?})"));
        }
    }
    if mg.names().len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = mg.names().iter().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("ltxv gguf import: unused source tensors: {extra:?}"));
    }
    Ok(())
}

/// Convert an LTX-2.x AV DiT GGUF into a brain-native fp32 safetensors
/// checkpoint at `out_path`, carrying a `ModelCard` with family `"ltxv"`.
///
/// **DiT only** - like `wan::import::import_gguf` and for the same reason:
/// this checkpoint carries the transformer alone (confirmed: the real
/// header's 4349 tensors are exactly [`av_dit_tensor_manifest`]'s own count
/// at the real config, no VAE/text-encoder tensors mixed in). The VAEs, the
/// Gemma-4 text encoder and the tokenizer come from their own source
/// (`crates/arch`'s `ltxv` row's `weights_env`).
///
/// **Streaming, one tensor at a time**, same discipline as `wan`'s: the
/// two-way manifest check ([`validate_av_dit_gguf_shapes`]) runs on shapes
/// alone before any tensor is dequantized, and only then is each tensor
/// decoded from the mmap, written through [`checkpoint::weightio::
/// StWriter`] and dropped - peak host memory is one tensor's fp32
/// expansion, never the whole model.
pub fn import_gguf(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let cfg = av_dit_config_from_kv(mg)?;
    validate_av_dit_gguf_shapes(mg, &cfg)?;
    let manifest = av_dit_tensor_manifest(&cfg);

    let id = id_override.unwrap_or("brain/ltxv-gguf");
    let mut card = checkpoint::st::ModelCard::new(id, "ltxv");
    card.param_count = Some(manifest.iter().map(|(_, s)| s.iter().product::<usize>() as u64).sum());
    let config = serde_json::json!({
        "video": {
            "inner_dim": cfg.video.inner_dim,
            "num_heads": cfg.video.num_heads,
            "num_layers": cfg.video.num_layers,
            "in_channels": cfg.video.in_channels,
            "out_channels": cfg.video.out_channels,
            "cross_attention_dim": cfg.video.cross_attention_dim,
            "apply_gated_attention": cfg.video.apply_gated_attention,
            "connector_num_layers": cfg.video.connector_num_layers,
        },
        "audio": {
            "inner_dim": cfg.audio.inner_dim,
            "num_heads": cfg.audio.num_heads,
            "cross_attention_dim": cfg.audio.cross_attention_dim,
        },
        "av_ca_timestep_scale_multiplier": cfg.av_ca_timestep_scale_multiplier,
    });
    let plan: Vec<(String, Vec<u64>)> = manifest.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
    let mut w = checkpoint::weightio::StWriter::create(out_path, &plan, &config, Some(&card)).map_err(|e| e.to_string())?;
    for (name, _) in &manifest {
        let data = mg.tensor(name).ok_or_else(|| format!("{name}: missing tensor data"))?.map_err(|e| format!("{name}: dequant failed: {e}"))?;
        w.write(name, &data).map_err(|e| e.to_string())?;
    }
    w.finish().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One fresh synthetic checkpoint, all-zero data at the manifest's real
    /// shapes, names optionally prefixed (`""` for bare, `"vae."` for the
    /// monolith-prefixed case).
    fn build(prefix: &str, manifest: &[(String, Vec<usize>)]) -> Vec<StTensor> {
        manifest
            .iter()
            .map(|(n, s)| {
                let len = s.iter().product();
                StTensor { name: format!("{prefix}{n}"), shape: s.clone(), data: vec![0.0f32; len] }
            })
            .collect()
    }

    /// Both directions of [`validate`], and both accepted name spaces, all in
    /// ONE test rather than several.
    ///
    /// This checkpoint's real channel widths (up to 4096, `decoder.up_blocks.1.
    /// conv.weight` alone is `[4096,1024,3,3,3]`) put the full manifest at
    /// ~726M elements (~2.9GB as f32) - the `full.clone()` pattern that works
    /// fine for Wan's much narrower VAE multiplies that into double-digit GB
    /// once several such tests build/clone their own copy and run
    /// concurrently (the default), which SIGKILLed this suite before this
    /// fix. One test, one owned synthetic checkpoint at a time (each `build`
    /// call's result is moved into `import_vae` and dropped before the next
    /// is constructed - never cloned, never two resident at once), keeps peak
    /// RSS here to that single ~2.9GB regardless of test-thread count.
    #[test]
    fn import_validates_both_directions_and_both_name_spaces() {
        let cfg = LtxVaeConfig::conv25();
        let manifest = cfg.tensor_manifest();

        let w = import_vae(build("", &manifest), &cfg).expect("bare names");
        assert_eq!(w.len(), manifest.len());
        drop(w);

        let w2 = import_vae(build("vae.", &manifest), &cfg).expect("vae.-prefixed names");
        assert_eq!(w2.len(), manifest.len());
        drop(w2);

        let mut missing = build("", &manifest);
        missing.retain(|t| t.name != "decoder.up_blocks.7.conv.conv.weight");
        let e = import_vae(missing, &cfg).unwrap_err();
        assert!(e.contains("decoder.up_blocks.7.conv.conv.weight"), "{e}");

        let mut extra = build("", &manifest);
        extra.push(StTensor { name: "decoder.up_blocks.99.conv.conv.weight".into(), shape: vec![1], data: vec![0.0] });
        let e = import_vae(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");

        let mut wrong = build("", &manifest);
        if let Some(t) = wrong.iter_mut().find(|t| t.name == "encoder.conv_in.conv.weight") {
            t.shape = vec![1, 1];
            t.data = vec![0.0; 1];
        }
        let e = import_vae(wrong, &cfg).unwrap_err();
        assert!(e.contains("encoder.conv_in.conv.weight") && e.contains("expected"), "{e}");
    }

    /// A real, correctly-named Lightricks release
    /// (`ltx-2.5-video-vae-bf16.safetensors`, NOT the `-conv-` file this
    /// import targets) carries a structurally different decoder -
    /// `crate::na_decoder`'s architecture, `decoder.det_stages.*`/
    /// `decoder.diff_blocks.*` - that this importer cannot read. Handing it
    /// the wrong file must name what it actually got, not report the first
    /// conv-decoder tensor it happens to miss as if the file were merely
    /// incomplete.
    #[test]
    fn import_names_the_na_decoder_architecture_instead_of_a_generic_missing_tensor() {
        let cfg = LtxVaeConfig::conv25();
        let mut na_shaped = build("", &cfg.tensor_manifest());
        na_shaped.retain(|t| !t.name.starts_with("decoder."));
        na_shaped.push(StTensor { name: "decoder.det_stages.0.0.attn.qkv.weight".into(), shape: vec![1], data: vec![0.0] });
        na_shaped.push(StTensor { name: "decoder.diff_blocks.0.attn.qkv.weight".into(), shape: vec![1], data: vec![0.0] });

        let e = import_vae(na_shaped, &cfg).unwrap_err();
        assert!(e.contains("na_decoder") || e.contains("NA decoder") || e.contains("det_stages"), "{e}");
        assert!(e.contains("-conv-"), "{e}");
    }

    /// [`import_audio_vae`] and [`import_vocoder`] each filter their own
    /// prefix out of a SHARED synthetic file (mirroring the real checkpoint
    /// carrying both subsystems), silently ignore the other's tensors AND a
    /// stray `vocoder.bwe_generator.*`/`vocoder.mel_stft.*` tensor (BWE, out
    /// of scope), and still catch a missing/extra/wrong-shape tensor within
    /// their own subset.
    #[test]
    fn audio_vae_and_vocoder_imports_share_one_file_and_validate_independently() {
        let vae_cfg = AudioVaeConfig::ltx25();
        let voc_cfg = VocoderConfig::ltx25();
        let vae_manifest = vae_cfg.tensor_manifest();
        let voc_manifest = voc_cfg.tensor_manifest();

        // `StTensor` is not `Clone`, so build the shared-file list fresh for
        // each import rather than cloning one - each import only reads its
        // own prefix's subset anyway.
        let shared = |vae_manifest: &[(String, Vec<usize>)], voc_manifest: &[(String, Vec<usize>)]| -> Vec<StTensor> {
            let mut all = build("audio_vae.", vae_manifest);
            all.extend(build("vocoder.vocoder.", voc_manifest));
            // BWE tensors this checkpoint also carries - must be ignored by both.
            all.push(StTensor { name: "vocoder.bwe_generator.conv_pre.weight".into(), shape: vec![1], data: vec![0.0] });
            all.push(StTensor { name: "vocoder.mel_stft.mel_basis".into(), shape: vec![1], data: vec![0.0] });
            all
        };

        let w = import_audio_vae(shared(&vae_manifest, &voc_manifest), &vae_cfg).expect("audio vae from the shared file");
        assert_eq!(w.len(), vae_manifest.len());
        drop(w);

        let v = import_vocoder(shared(&vae_manifest, &voc_manifest), &voc_cfg).expect("vocoder from the shared file");
        assert_eq!(v.len(), voc_manifest.len());
        drop(v);

        let mut missing = build("audio_vae.", &vae_manifest);
        missing.retain(|t| t.name != "audio_vae.decoder.conv_out.conv.weight");
        let e = import_audio_vae(missing, &vae_cfg).unwrap_err();
        assert!(e.contains("decoder.conv_out.conv.weight"), "{e}");

        let mut extra = build("vocoder.vocoder.", &voc_manifest);
        extra.push(StTensor { name: "vocoder.vocoder.conv_pre.extra".into(), shape: vec![1], data: vec![0.0] });
        let e = import_vocoder(extra, &voc_cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");

        let mut wrong = build("audio_vae.", &vae_manifest);
        if let Some(t) = wrong.iter_mut().find(|t| t.name == "audio_vae.encoder.conv_in.conv.weight") {
            t.shape = vec![1, 1];
            t.data = vec![0.0; 1];
        }
        let e = import_audio_vae(wrong, &vae_cfg).unwrap_err();
        assert!(e.contains("encoder.conv_in.conv.weight") && e.contains("expected"), "{e}");
    }

    /// [`import_upsampler`] against both real configs' bare-name manifests,
    /// both directions.
    #[test]
    fn upsampler_import_validates_both_directions_and_both_modes() {
        for cfg in [LatentUpsamplerConfig::spatial_x2(), LatentUpsamplerConfig::temporal_x2()] {
            let manifest = cfg.tensor_manifest();
            let w = import_upsampler(build("", &manifest), &cfg).expect("bare names");
            assert_eq!(w.len(), manifest.len());
            drop(w);

            let mut missing = build("", &manifest);
            missing.retain(|t| t.name != "final_conv.weight");
            let e = import_upsampler(missing, &cfg).unwrap_err();
            assert!(e.contains("final_conv.weight"), "{e}");

            let mut extra = build("", &manifest);
            extra.push(StTensor { name: "res_blocks.99.conv1.weight".into(), shape: vec![1], data: vec![0.0] });
            let e = import_upsampler(extra, &cfg).unwrap_err();
            assert!(e.contains("unused source tensors"), "{e}");
        }
    }

    /// [`import_duration_head`] strips the `duration_head.` prefix and
    /// validates both directions.
    #[test]
    fn duration_head_import_validates_both_directions() {
        let cfg = DurationHeadConfig::ltx25();
        let manifest = cfg.tensor_manifest();

        let w = import_duration_head(build("duration_head.", &manifest), &cfg).expect("prefixed names");
        assert_eq!(w.len(), manifest.len());
        drop(w);

        let mut missing = build("duration_head.", &manifest);
        missing.retain(|t| t.name != "duration_head.mlp_out.weight");
        let e = import_duration_head(missing, &cfg).unwrap_err();
        assert!(e.contains("mlp_out.weight"), "{e}");

        let mut extra = build("duration_head.", &manifest);
        extra.push(StTensor { name: "duration_head.mlp_out.extra".into(), shape: vec![1], data: vec![0.0] });
        let e = import_duration_head(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");
    }

    /// [`import_dit`] (video-only) validates both directions on the tiny
    /// config's own manifest, bare names (no renaming - this checkpoint
    /// family's own tensor names ARE the canonical ones).
    #[test]
    fn dit_import_validates_both_directions() {
        let cfg = LtxDitConfig::tiny();
        let manifest = dit_tensor_manifest(&cfg);

        let w = import_dit(build("", &manifest), &cfg).expect("bare names");
        assert_eq!(w.len(), manifest.len());
        drop(w);

        let mut missing = build("", &manifest);
        missing.retain(|t| t.name != "patchify_proj.weight");
        let e = import_dit(missing, &cfg).unwrap_err();
        assert!(e.contains("patchify_proj.weight"), "{e}");

        let mut extra = build("", &manifest);
        extra.push(StTensor { name: "transformer_blocks.99.attn1.to_q.weight".into(), shape: vec![1], data: vec![0.0] });
        let e = import_dit(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");
    }

    /// [`import_av_dit`] validates both directions on the tiny AV config's
    /// own manifest - the same two-way check, over the much larger manifest
    /// [`av_dit_tensor_manifest`] produces (both streams, every AV
    /// cross-modal table, both connectors).
    #[test]
    fn av_dit_import_validates_both_directions() {
        let cfg = LtxAvDitConfig::tiny();
        let manifest = av_dit_tensor_manifest(&cfg);

        let w = import_av_dit(build("", &manifest), &cfg).expect("bare names");
        assert_eq!(w.len(), manifest.len());
        drop(w);

        let mut missing = build("", &manifest);
        missing.retain(|t| t.name != "transformer_blocks.0.audio_to_video_attn.to_gate_logits.weight");
        let e = import_av_dit(missing, &cfg).unwrap_err();
        assert!(e.contains("audio_to_video_attn.to_gate_logits.weight"), "{e}");

        let mut extra = build("", &manifest);
        extra.push(StTensor { name: "video_embeddings_connector.transformer_1d_blocks.99.ff.net.2.weight".into(), shape: vec![1], data: vec![0.0] });
        let e = import_av_dit(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");

        let mut wrong = build("", &manifest);
        if let Some(t) = wrong.iter_mut().find(|t| t.name == "keyframes_abs_pos_embedding") {
            t.shape = vec![1, 1];
            t.data = vec![0.0];
        }
        let e = import_av_dit(wrong, &cfg).unwrap_err();
        assert!(e.contains("keyframes_abs_pos_embedding") && e.contains("expected"), "{e}");
    }

    /// The GGUF path end to end on a synthetic fixture: [`av_dit_config_from_kv`]
    /// reads the config back off the embedded `config` KV, [`import_gguf`]
    /// converts every manifest tensor, and the written checkpoint round-trips
    /// through [`checkpoint::weightio::WeightReader`] with the exact values
    /// the synthetic file carried (proves `mg.tensor()` -> `StWriter::write`
    /// did not transpose or truncate anything). Mirrors `wan::import`'s own
    /// `gguf_tests`/`dit_tests` split - this is the "does the whole path
    /// work" test; `av_dit_config_from_kv`'s error cases below are the
    /// negative coverage.
    #[test]
    fn gguf_import_round_trips_a_synthetic_checkpoint() {
        use checkpoint::gguf::GgufValue;
        use checkpoint::gguf_write::{write, TensorOut};

        let cfg = LtxAvDitConfig::tiny();
        let manifest = av_dit_tensor_manifest(&cfg);
        let mut seed = 0u64;
        let tensors: Vec<TensorOut> = manifest
            .iter()
            .map(|(name, shape)| {
                seed += 1;
                let n: usize = shape.iter().product();
                let data: Vec<u8> = (0..n).flat_map(|i| (((i as u64 + seed) % 997) as f32 * 0.01 - 5.0).to_le_bytes()).collect();
                TensorOut { name: name.clone(), shape: shape.clone(), ty: 0u32, data }
            })
            .collect();
        let config_kv = serde_json::json!({
            "transformer": {
                "num_attention_heads": cfg.video.num_heads,
                "attention_head_dim": cfg.video.head_dim(),
                "num_layers": cfg.video.num_layers,
                "in_channels": cfg.video.in_channels,
                "out_channels": cfg.video.out_channels,
                "cross_attention_dim": cfg.video.cross_attention_dim,
                "ff_bias": cfg.video.ff_bias,
                "cross_attention_adaln": cfg.video.cross_attention_adaln,
                "use_keyframes_abs_pos_embedding": cfg.video.use_keyframes_abs_pos_embedding,
                "norm_eps": cfg.video.norm_eps,
                "positional_embedding_theta": cfg.video.positional_embedding_theta,
                "positional_embedding_max_pos": cfg.video.positional_embedding_max_pos,
                "timestep_scale_multiplier": cfg.video.timestep_scale_multiplier,
                "use_middle_indices_grid": cfg.video.use_middle_indices_grid,
                "apply_gated_attention": cfg.video.apply_gated_attention,
                "connector_apply_gated_attention": cfg.video.connector_apply_gated_attention,
                "connector_num_layers": cfg.video.connector_num_layers,
                "connector_num_attention_heads": cfg.video.connector_num_attention_heads,
                "connector_attention_head_dim": cfg.video.connector_attention_head_dim,
                "connector_num_learnable_registers": cfg.video.connector_num_learnable_registers,
                "connector_positional_embedding_max_pos": cfg.video.connector_positional_embedding_max_pos,
                "connector_norm_output": cfg.video.connector_norm_output,
                "caption_proj_before_connector": cfg.video.caption_proj_before_connector,
                "use_embeddings_connector": cfg.video.use_embeddings_connector,
                "audio_num_attention_heads": cfg.audio.num_heads,
                "audio_attention_head_dim": cfg.audio.head_dim(),
                "audio_out_channels": cfg.audio.out_channels,
                "audio_cross_attention_dim": cfg.audio.cross_attention_dim,
                "audio_positional_embedding_max_pos": cfg.audio.positional_embedding_max_pos,
                "audio_connector_num_attention_heads": cfg.audio.connector_num_attention_heads,
                "audio_connector_attention_head_dim": cfg.audio.connector_attention_head_dim,
                "av_ca_timestep_scale_multiplier": cfg.av_ca_timestep_scale_multiplier,
            },
        })
        .to_string();
        let kvs = vec![("general.architecture".to_string(), GgufValue::String("ltxv".to_string())), ("config".to_string(), GgufValue::String(config_kv))];

        let dir = std::env::temp_dir().join(format!("ltxv-gguf-import-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src_path = dir.join("toy.gguf").to_string_lossy().into_owned();
        write(&src_path, &kvs, &tensors, 32).unwrap();

        let mg = MmapGguf::open(&src_path).unwrap();
        let read_back_cfg = av_dit_config_from_kv(&mg).expect("config KV must parse");
        assert_eq!(read_back_cfg, cfg, "the config read off the KV must match what was written");

        let out_path = dir.join("toy.brain.safetensors").to_string_lossy().into_owned();
        import_gguf(&mg, &out_path, Some("test/ltxv-tiny")).expect("import must succeed on a fully-covered synthetic file");

        let reader = checkpoint::weightio::WeightReader::open(&out_path).unwrap();
        assert_eq!(reader.card().expect("a model card must be written").id, "test/ltxv-tiny");
        for (name, shape) in &manifest {
            let n: usize = shape.iter().product();
            let got = reader.tensor(name).unwrap_or_else(|| panic!("missing {name} in the converted checkpoint"));
            assert_eq!(got.len(), n, "{name}");
        }
        // Spot-check one tensor's VALUES round-tripped exactly (not just its
        // length) - the same "gguf -> mg.tensor() -> StWriter::write did not
        // silently reorder/transpose" check `wan`'s own gguf import test makes.
        let (first_name, first_shape) = &manifest[0];
        let want: Vec<f32> = {
            let n: usize = first_shape.iter().product();
            (0..n).map(|i| ((i as u64 + 1) % 997) as f32 * 0.01 - 5.0).collect()
        };
        let got = checkpoint::weightio::WeightReader::open(&out_path).unwrap().tensor(first_name).unwrap();
        assert_eq!(got, want, "{first_name} values must round-trip exactly");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint whose `general.architecture` isn't `"ltxv"`, or whose
    /// `config` KV is absent/malformed, must be refused by name - never a
    /// panic, never a config built from partial/default data.
    #[test]
    fn av_dit_config_from_kv_rejects_a_non_ltxv_or_malformed_checkpoint() {
        use checkpoint::gguf::GgufValue;
        use checkpoint::gguf_write::{write, TensorOut};

        let dir = std::env::temp_dir().join(format!("ltxv-gguf-kv-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let no_arch_path = dir.join("no_arch.gguf").to_string_lossy().into_owned();
        write(&no_arch_path, &[], &[TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: vec![0u8; 4] }], 32).unwrap();
        let mg = MmapGguf::open(&no_arch_path).unwrap();
        let e = av_dit_config_from_kv(&mg).unwrap_err();
        assert!(e.contains("general.architecture"), "{e}");

        let wrong_arch_path = dir.join("wrong_arch.gguf").to_string_lossy().into_owned();
        let kvs = vec![("general.architecture".to_string(), GgufValue::String("wan".to_string()))];
        write(&wrong_arch_path, &kvs, &[TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: vec![0u8; 4] }], 32).unwrap();
        let mg2 = MmapGguf::open(&wrong_arch_path).unwrap();
        let e2 = av_dit_config_from_kv(&mg2).unwrap_err();
        assert!(e2.contains("general.architecture"), "{e2}");

        let no_config_path = dir.join("no_config.gguf").to_string_lossy().into_owned();
        let kvs3 = vec![("general.architecture".to_string(), GgufValue::String("ltxv".to_string()))];
        write(&no_config_path, &kvs3, &[TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: vec![0u8; 4] }], 32).unwrap();
        let mg3 = MmapGguf::open(&no_config_path).unwrap();
        let e3 = av_dit_config_from_kv(&mg3).unwrap_err();
        assert!(e3.contains("'config' KV"), "{e3}");

        let bad_json_path = dir.join("bad_json.gguf").to_string_lossy().into_owned();
        let kvs4 = vec![
            ("general.architecture".to_string(), GgufValue::String("ltxv".to_string())),
            ("config".to_string(), GgufValue::String("not json".to_string())),
        ];
        write(&bad_json_path, &kvs4, &[TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: vec![0u8; 4] }], 32).unwrap();
        let mg4 = MmapGguf::open(&bad_json_path).unwrap();
        let e4 = av_dit_config_from_kv(&mg4).unwrap_err();
        assert!(e4.contains("not valid JSON"), "{e4}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------
    // LTX-2.3
    // -----------------------------------------------------------------

    /// One checkpoint's `config` JSON. `omit_release_flags` reproduces the
    /// real LTX-2.3 config, which carries neither `ff_bias` nor
    /// `use_keyframes_abs_pos_embedding`; LTX-2.5's carries both.
    fn transformer_config_json(cfg: &LtxAvDitConfig, omit_release_flags: bool) -> String {
        let mut t = serde_json::json!({
            "num_attention_heads": cfg.video.num_heads,
            "attention_head_dim": cfg.video.head_dim(),
            "num_layers": cfg.video.num_layers,
            "in_channels": cfg.video.in_channels,
            "out_channels": cfg.video.out_channels,
            "cross_attention_dim": cfg.video.cross_attention_dim,
            "cross_attention_adaln": cfg.video.cross_attention_adaln,
            "norm_eps": cfg.video.norm_eps,
            "positional_embedding_theta": cfg.video.positional_embedding_theta,
            "positional_embedding_max_pos": cfg.video.positional_embedding_max_pos,
            "timestep_scale_multiplier": cfg.video.timestep_scale_multiplier,
            "use_middle_indices_grid": cfg.video.use_middle_indices_grid,
            "apply_gated_attention": cfg.video.apply_gated_attention,
            "connector_apply_gated_attention": cfg.video.connector_apply_gated_attention,
            "connector_num_layers": cfg.video.connector_num_layers,
            "connector_num_attention_heads": cfg.video.connector_num_attention_heads,
            "connector_attention_head_dim": cfg.video.connector_attention_head_dim,
            "connector_num_learnable_registers": cfg.video.connector_num_learnable_registers,
            "connector_positional_embedding_max_pos": cfg.video.connector_positional_embedding_max_pos,
            "connector_norm_output": cfg.video.connector_norm_output,
            "caption_proj_before_connector": cfg.video.caption_proj_before_connector,
            "use_embeddings_connector": cfg.video.use_embeddings_connector,
            "audio_num_attention_heads": cfg.audio.num_heads,
            "audio_attention_head_dim": cfg.audio.head_dim(),
            "audio_out_channels": cfg.audio.out_channels,
            "audio_cross_attention_dim": cfg.audio.cross_attention_dim,
            "audio_positional_embedding_max_pos": cfg.audio.positional_embedding_max_pos,
            "audio_connector_num_attention_heads": cfg.audio.connector_num_attention_heads,
            "audio_connector_attention_head_dim": cfg.audio.connector_attention_head_dim,
            "av_ca_timestep_scale_multiplier": cfg.av_ca_timestep_scale_multiplier,
        });
        if !omit_release_flags {
            t["ff_bias"] = serde_json::json!(cfg.video.ff_bias);
            t["use_keyframes_abs_pos_embedding"] = serde_json::json!(cfg.video.use_keyframes_abs_pos_embedding);
        }
        serde_json::json!({ "transformer": t }).to_string()
    }

    /// [`av_dit_tensor_manifest`] at [`LtxAvDitConfig::ltx23`] reproduces the
    /// real LTX-2.3 22B checkpoint header, and differs from the LTX-2.5 one
    /// in exactly the documented way.
    ///
    /// Both real 22B headers were range-read (the safetensors 8-byte
    /// little-endian JSON length, then the JSON - metadata, never weights)
    /// and diffed name by name and shape by shape; the LTX-2.3 GGUF header
    /// (`general.architecture = "ltxv"`, same as 2.5) was parsed the same way
    /// and its 4444 bare names are the SAME set as the safetensors file's
    /// 4444 `model.diffusion_model.*` names. The numbers below are that diff,
    /// transcribed:
    ///
    /// * 4349 (2.5) and 4444 (2.3) tensors.
    /// * 2.5 alone has `keyframes_abs_pos_embedding`.
    /// * 2.3 alone has 96 video-FFN bias tensors (48 blocks x 2).
    /// * Zero shape mismatches across the 4348 shared names.
    ///
    /// Manifests are names and shapes only - no tensor data is built here, so
    /// this runs at the real 22B widths for the cost of a few string
    /// allocations rather than the ~88 GB an fp32 materialization would need.
    #[test]
    fn ltx23_manifest_reproduces_the_real_22b_header() {
        use std::collections::BTreeMap;

        let m23: BTreeMap<String, Vec<usize>> = av_dit_tensor_manifest(&LtxAvDitConfig::ltx23()).into_iter().collect();
        let m25: BTreeMap<String, Vec<usize>> = av_dit_tensor_manifest(&LtxAvDitConfig::ltx25()).into_iter().collect();
        assert_eq!(m23.len(), 4444, "real LTX-2.3 22B header tensor count");
        assert_eq!(m25.len(), 4349, "real LTX-2.5 22B header tensor count");

        let only25: Vec<&str> = m25.keys().filter(|k| !m23.contains_key(*k)).map(String::as_str).collect();
        assert_eq!(only25, ["keyframes_abs_pos_embedding"], "2.5 must add exactly the keyframe marker");

        let only23: Vec<&str> = m23.keys().filter(|k| !m25.contains_key(*k)).map(String::as_str).collect();
        assert_eq!(only23.len(), 96, "48 blocks x (ff.net.0.proj.bias + ff.net.2.bias)");
        assert!(
            only23.iter().all(|n| n.starts_with("transformer_blocks.") && (n.ends_with(".ff.net.0.proj.bias") || n.ends_with(".ff.net.2.bias"))),
            "2.3's extra tensors must all be the video FFN's own bias: {only23:?}"
        );
        // The audio FFN's and both connectors' biases are NOT part of that
        // difference - both releases carry them (`crate::dit::push_ff`).
        for n in ["transformer_blocks.0.audio_ff.net.0.proj.bias", "video_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.bias"] {
            assert!(m23.contains_key(n) && m25.contains_key(n), "{n} must be in both");
        }

        for (name, s23) in &m23 {
            if let Some(s25) = m25.get(name) {
                assert_eq!(s23, s25, "{name} must have the same shape in both releases");
            }
        }
        assert_eq!(m23["transformer_blocks.47.ff.net.0.proj.bias"], vec![4 * 4096]);
        assert_eq!(m23["transformer_blocks.47.ff.net.2.bias"], vec![4096]);
        assert_eq!(m25["keyframes_abs_pos_embedding"], vec![1, 4096]);
    }

    /// The two flags LTX-2.3's `config` omits are resolved from the
    /// checkpoint's own tensor list, not from a default.
    ///
    /// The mutation that matters is the last block: the SAME 2.3-shaped JSON
    /// (both keys absent) against a tensor list that says the opposite must
    /// produce the opposite config. A constant fallback - even one that
    /// happened to be right for today's 2.3 - would pass the first two
    /// assertions and fail this one.
    #[test]
    fn absent_release_flags_are_resolved_from_the_tensor_list() {
        let tiny25 = LtxAvDitConfig::tiny(); // ff_bias false, keyframes true
        let tiny23 = LtxAvDitConfig {
            video: LtxDitConfig { ff_bias: true, use_keyframes_abs_pos_embedding: false, ..tiny25.video },
            audio: LtxAudioDitConfig { ff_bias: true, ..tiny25.audio },
            ..tiny25
        };

        // A `has_tensor` built from a manifest is exactly what a real
        // checkpoint answers, so the two sides cannot drift.
        let has = |cfg: &LtxAvDitConfig| {
            let names: std::collections::HashSet<String> = av_dit_tensor_manifest(cfg).into_iter().map(|(n, _)| n).collect();
            move |n: &str| names.contains(n)
        };

        // 2.3: keys absent, tensor list carries the FFN bias and no keyframe
        // marker.
        let h23 = has(&tiny23);
        let got = av_dit_config_from_json(&transformer_config_json(&tiny23, true), &h23).expect("2.3-shaped config must parse");
        assert_eq!(got, tiny23);

        // 2.5: keys present and honored.
        let h25 = has(&tiny25);
        let got = av_dit_config_from_json(&transformer_config_json(&tiny25, false), &h25).expect("2.5-shaped config must parse");
        assert_eq!(got, tiny25);

        // Same 2.3-shaped JSON, opposite tensor list -> opposite answer.
        let flipped = av_dit_config_from_json(&transformer_config_json(&tiny23, true), &h25).expect("must parse");
        assert!(!flipped.video.ff_bias, "ff_bias must follow the tensor list");
        assert!(flipped.video.use_keyframes_abs_pos_embedding, "the keyframe marker must follow the tensor list");
    }

    /// [`import_av_dit`] accepts both packagings and refuses the wrong
    /// release.
    ///
    /// LTX-2.5 ships the transformer alone; LTX-2.3 bundles it under
    /// [`DIT_BUNDLE_PREFIX`] next to the VAEs, the vocoder and the text
    /// projection. Run at tiny widths (a 22B synthetic file would be tens of
    /// GB) but at the real STRUCTURE: the two configs differ by the same two
    /// flags the real releases do.
    #[test]
    fn av_dit_import_accepts_the_bundle_and_refuses_the_wrong_release() {
        let cfg25 = LtxAvDitConfig::tiny();
        let cfg23 = LtxAvDitConfig {
            video: LtxDitConfig { ff_bias: true, use_keyframes_abs_pos_embedding: false, ..cfg25.video },
            ..cfg25
        };
        let m23 = av_dit_tensor_manifest(&cfg23);

        // Bare names (the GGUF-derived spelling).
        let w = import_av_dit(build("", &m23), &cfg23).expect("bare 2.3 names");
        assert_eq!(w.len(), m23.len());
        drop(w);

        // The bundle: the DiT under its prefix, plus the other subsystems'
        // tensors, which must be filtered out rather than reported as unused.
        let bundle = |m: &[(String, Vec<usize>)]| -> Vec<StTensor> {
            let mut all = build(DIT_BUNDLE_PREFIX, m);
            for other in ["vae.encoder.conv_in.conv.weight", "audio_vae.decoder.conv_out.conv.bias", "vocoder.vocoder.conv_pre.weight", "text_embedding_projection.video_aggregate_embed.bias"] {
                all.push(StTensor { name: other.into(), shape: vec![1], data: vec![0.0] });
            }
            all
        };
        let w = import_av_dit(bundle(&m23), &cfg23).expect("bundled 2.3 names");
        assert_eq!(w.len(), m23.len());
        drop(w);

        // The same file at the 2.5 config: the keyframe marker is missing and
        // the 2 x num_layers FFN bias tensors are unused. Either direction
        // must error by name - never a silently half-covered model.
        let e = import_av_dit(bundle(&m23), &cfg25).unwrap_err();
        assert!(e.contains("keyframes_abs_pos_embedding"), "{e}");

        // ...and the mirror: a 2.5 file imported as 2.3 misses the biases.
        let e = import_av_dit(build("", &av_dit_tensor_manifest(&cfg25)), &cfg23).unwrap_err();
        assert!(e.contains("ff.net.0.proj.bias"), "{e}");
    }
}
