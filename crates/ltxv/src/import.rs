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

use checkpoint::safetensors::StTensor;
use vae::blocks::Tensors;

use crate::audio_vae::AudioVaeConfig;
use crate::vae3d::LtxVaeConfig;
use crate::vocoder::VocoderConfig;

/// Check a name→tensor map against a manifest in both directions. Generic
/// over the manifest slice (not `LtxVaeConfig` specifically) so
/// [`import_audio_vae`]/[`import_vocoder`] reuse the same two-way check
/// rather than restating it.
fn validate_manifest(map: Tensors, manifest: &[(String, Vec<usize>)], who: &str) -> Result<Tensors, String> {
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
}
