// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the LTX-2.5 video VAE.
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
//! remapping table - `import_vae` is [`validate`] plus an optional `vae.`
//! prefix strip, for robustness against a future monolithic (non-Comfy-split)
//! release of the same checkpoint family.
//!
//! Validates in **both directions**: a missing tensor errors by name, an
//! unused source tensor errors by name, and a shape mismatch errors with both
//! shapes. Nothing is ever zero-filled.

use checkpoint::safetensors::StTensor;
use vae::blocks::Tensors;

use crate::vae3d::LtxVaeConfig;

/// Check a name→tensor map against the config's manifest in both directions.
fn validate(map: Tensors, cfg: &LtxVaeConfig) -> Result<Tensors, String> {
    let manifest = cfg.tensor_manifest();
    for (name, shape) in &manifest {
        match map.get(name) {
            None => return Err(format!("ltxv vae import: missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("ltxv vae import: {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!("ltxv vae import: {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("ltxv vae import: unused source tensors: {extra:?}"));
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
    validate(map, cfg)
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
}
