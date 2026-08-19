// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the LTX-2.5 video VAE, audio VAE and base vocoder.
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
/// carries both). Strips an optional `vae.` monolith prefix so a future
/// non-Comfy-split release of the same checkpoint family imports unchanged;
/// the real Comfy-split file this milestone targets already has none.
pub fn import_vae(tensors: Vec<StTensor>, cfg: &LtxVaeConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors
        .into_iter()
        .map(|t| {
            let name = t.name.strip_prefix("vae.").map(str::to_string).unwrap_or(t.name);
            (name, (t.shape, t.data))
        })
        .collect();
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
/// dit_tensor_manifest`] already use) with two-way coverage. Names already
/// bare (no monolith prefix, unlike [`import_vae`]'s `vae.` case) - this
/// checkpoint family's own tensor names ARE the reference module tree's, so
/// this is [`validate_manifest`] with no renaming step at all.
pub fn import_dit(tensors: Vec<StTensor>, cfg: &LtxDitConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    validate_manifest(map, &dit_tensor_manifest(cfg), "dit")
}

/// Import the audio+video DiT (`LtxAvDitConfig`'s canonical names - see
/// [`crate::dit::av_dit_tensor_manifest`]'s doc) with two-way coverage. Same
/// no-renaming shape [`import_dit`] takes.
pub fn import_av_dit(tensors: Vec<StTensor>, cfg: &LtxAvDitConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    validate_manifest(map, &av_dit_tensor_manifest(cfg), "av dit")
}

// ---------------------------------------------------------------------------
// GGUF: direct loading + ahead-of-time conversion, sharing one manifest
// ---------------------------------------------------------------------------

/// The GGUF `general.architecture` value every LTX-2.x checkpoint carries -
/// `crates/arch`'s own `ltxv` row's id IS this spelling (`gguf: None`, see
/// that row's doc comment), so `brain_arch::by_gguf("ltxv")` resolves
/// without an alias.
pub const GGUF_ARCHITECTURE: &str = "ltxv";

/// The [`LtxAvDitConfig`] a checkpoint's own embedded `config` KV names.
///
/// Unlike `wan::import::dit_config_from_shapes` (derived from tensor SHAPES,
/// because Wan's GGUF carries no config JSON), every real LTX-2.x GGUF
/// embeds its full diffusers config as ONE JSON-string-valued KV entry
/// (`config`, holding `{"transformer": {...}, "scheduler": {...}}` -
/// confirmed by range-reading the real 22B header), so this reads that JSON
/// directly rather than reverse-engineering shapes. Every field this crate
/// models is read by name and errors if absent or the wrong JSON type -
/// never defaulted - except `use_prompt_adaln_single`, which the reference
/// config never exposes as a flag at all (see [`LtxDitConfig::
/// use_prompt_adaln_single`]'s doc: this port only implements `false`, the
/// checkpoint-independent value every real config resolves to for the ONE
/// op sequence this crate runs).
pub fn av_dit_config_from_kv(mg: &MmapGguf) -> Result<LtxAvDitConfig, String> {
    let arch = mg.kv().get("general.architecture").and_then(|v| v.as_str());
    if arch != Some(GGUF_ARCHITECTURE) {
        return Err(format!("ltxv gguf import: general.architecture is {arch:?}, expected {GGUF_ARCHITECTURE:?}"));
    }
    let raw = mg.kv().get("config").and_then(|v| v.as_str()).ok_or("ltxv gguf import: no 'config' KV string (not an LTX-2.x checkpoint)")?;
    let full: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("ltxv gguf import: config KV is not valid JSON: {e}"))?;
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

    let video = LtxDitConfig {
        inner_dim: video_heads * video_head_dim,
        num_heads: video_heads,
        num_layers: u("num_layers")?,
        in_channels,
        out_channels: u("out_channels")?,
        cross_attention_dim: u("cross_attention_dim")?,
        ff_bias: b("ff_bias")?,
        cross_attention_adaln: b("cross_attention_adaln")?,
        use_prompt_adaln_single: false,
        use_keyframes_abs_pos_embedding: b("use_keyframes_abs_pos_embedding")?,
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
    };
    let audio = LtxAudioDitConfig {
        inner_dim: audio_heads * audio_head_dim,
        num_heads: audio_heads,
        in_channels,
        out_channels: u("audio_out_channels")?,
        cross_attention_dim: u("audio_cross_attention_dim")?,
        ff_bias: b("ff_bias")?,
        positional_embedding_max_pos: [audio_pos_max[0]],
        connector_num_attention_heads: u("audio_connector_num_attention_heads")?,
        connector_attention_head_dim: u("audio_connector_attention_head_dim")?,
    };
    let av_ca_timestep_scale_multiplier = f("av_ca_timestep_scale_multiplier")? as f32;
    Ok(LtxAvDitConfig { video, audio, av_ca_timestep_scale_multiplier })
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
                "connector_num_layers": cfg.video.connector_num_layers,
                "connector_num_attention_heads": cfg.video.connector_num_attention_heads,
                "connector_attention_head_dim": cfg.video.connector_attention_head_dim,
                "connector_num_learnable_registers": cfg.video.connector_num_learnable_registers,
                "connector_positional_embedding_max_pos": cfg.video.connector_positional_embedding_max_pos,
                "connector_norm_output": cfg.video.connector_norm_output,
                "caption_proj_before_connector": cfg.video.caption_proj_before_connector,
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
}
