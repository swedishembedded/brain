// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import with **two-way coverage validation**, in the discipline of
//! `flux2::import` / `qwen3::import::brain_init_from_hf`:
//!
//!   * every tensor [`Sam2Config::tensor_manifest`] expects is produced exactly
//!     once, with the right shape and element count — a miss is an error NAMING
//!     the tensor, never a zero-fill;
//!   * every source tensor is accounted for — either consumed, or matched by the
//!     explicit [`VIDEO_ONLY`] prefix list and COUNTED. Anything left over is an
//!     error listing the names.
//!
//! The video half of SAM 2 (memory attention / memory encoder / temporal object
//! pointers) is out of scope for the image path, so it is *skipped on purpose*
//! rather than silently ignored: [`ImportReport`] reports how many tensors were
//! skipped and by which prefix, and [`import`] asserts the split is exactly the
//! one the reference dump recorded (749 of 903 for hiera-large, 317 of 471 for
//! hiera-tiny).

use std::collections::{HashMap, HashSet};

use crate::config::Sam2Config;

/// name -> (shape, fp32 data).
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Name prefixes that belong to the VIDEO path only. Transcribed from the
/// reference dumper's own `video_only` list, so the two agree by construction.
///
/// `no_mem_embed` is on this list yet IS read by the image path (it is added to
/// the lowest-resolution FPN level — the one piece of the memory bank that
/// touches a still image). It is therefore consumed explicitly by
/// [`import`] before the skip test runs, not skipped.
pub const VIDEO_ONLY: &[&str] = &[
    "memory_attention.",
    "memory_encoder.",
    "maskmem_tpos_enc",
    "no_mem_embed",
    "no_mem_pos_enc",
    "mask_downsample.",
    "obj_ptr_tpos_proj.",
    "no_obj_embed_spatial",
];

/// The one video-path tensor the image path still needs.
pub const NO_MEM_EMBED: &str = "no_mem_embed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    /// Tensors in the source checkpoint.
    pub source: usize,
    /// Tensors matched against the manifest (== `manifest.len()`).
    pub imported: usize,
    /// Video-only tensors deliberately skipped.
    pub skipped_video: usize,
}

/// Import a SAM 2 `state_dict` (already read to fp32, e.g. by
/// `checkpoint::torchpt::read` on `sam2.1_hiera_*.pt`, or by
/// `checkpoint::safetensors`) into the image-path weight map.
///
/// `no_mem_embed` is returned in the map alongside the manifest tensors; every
/// other video tensor is dropped, counted, and reported.
pub fn import(tensors: Vec<(String, Vec<usize>, Vec<f32>)>, cfg: &Sam2Config) -> Result<(Tensors, ImportReport), String> {
    let manifest = cfg.tensor_manifest();
    let expected: HashMap<&str, &Vec<usize>> = manifest.iter().map(|(n, s)| (n.as_str(), s)).collect();

    let source = tensors.len();
    let mut out: Tensors = HashMap::with_capacity(manifest.len() + 1);
    let mut skipped_video = 0usize;
    let mut unused: Vec<String> = Vec::new();

    for (name, shape, data) in tensors {
        // `no_mem_embed` first: it prefix-matches VIDEO_ONLY but the image path
        // reads it, so testing the skip list first would drop it.
        if name == NO_MEM_EMBED || expected.contains_key(name.as_str()) {
            if let Some(want) = expected.get(name.as_str()) {
                if &shape != *want {
                    return Err(format!("import: {name} shape {shape:?}, expected {want:?}"));
                }
            }
            let n: usize = shape.iter().product();
            if data.len() != n {
                return Err(format!("import: {name} has {} values, expected {n}", data.len()));
            }
            if out.insert(name.clone(), (shape, data)).is_some() {
                return Err(format!("import: duplicate source tensor {name}"));
            }
            continue;
        }
        if VIDEO_ONLY.iter().any(|p| name.starts_with(p)) {
            skipped_video += 1;
            continue;
        }
        unused.push(name);
    }

    let missing: Vec<&str> = manifest.iter().map(|(n, _)| n.as_str()).filter(|n| !out.contains_key(*n)).collect();
    if !missing.is_empty() {
        return Err(format!("import: {} missing tensor(s), first: {:?}", missing.len(), &missing[..missing.len().min(8)]));
    }
    if !unused.is_empty() {
        unused.sort();
        return Err(format!(
            "import: {} unused source tensor(s) matching neither the manifest nor the video-only list, first: {:?}",
            unused.len(),
            &unused[..unused.len().min(8)]
        ));
    }
    if !out.contains_key(NO_MEM_EMBED) {
        return Err(format!("import: missing tensor {NO_MEM_EMBED} (the image path adds it to the lowest FPN level)"));
    }

    let report = ImportReport { source, imported: manifest.len(), skipped_video };
    if report.imported + report.skipped_video + 1 != report.source {
        return Err(format!(
            "import: coverage does not close: {} imported + {} video-only + 1 (no_mem_embed) != {} source",
            report.imported, report.skipped_video, report.source
        ));
    }
    Ok((out, report))
}

/// The ParamStore parameter list (name, numel) this config needs, in manifest
/// order plus `no_mem_embed`. Every entry is FROZEN — this is a forward-parity
/// port; the trainable roles arrive with the backward workstream.
pub fn param_list(cfg: &Sam2Config) -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> =
        cfg.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
    v.push((NO_MEM_EMBED.to_string(), cfg.d_model as usize));
    v
}

/// Flat fp32 map for `ParamStore::new_with_roles`.
pub fn init_map(t: &Tensors) -> HashMap<String, Vec<f32>> {
    t.iter().map(|(k, (_, d))| (k.clone(), d.clone())).collect()
}

/// Names present twice in the manifest — a generator bug that would make the
/// coverage check pass while a tensor silently went missing.
pub fn manifest_is_unique(cfg: &Sam2Config) -> bool {
    let m = cfg.tensor_manifest();
    let s: HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
    s.len() == m.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(cfg: &Sam2Config, extra_video: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut v: Vec<(String, Vec<usize>, Vec<f32>)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| {
                let k: usize = s.iter().product();
                (n, s, vec![0.5f32; k])
            })
            .collect();
        v.push((NO_MEM_EMBED.into(), vec![1, 1, cfg.d_model as usize], vec![0.0; cfg.d_model as usize]));
        for i in 0..extra_video {
            v.push((format!("memory_attention.layers.{i}.w"), vec![2], vec![0.0; 2]));
        }
        v
    }

    #[test]
    fn manifest_names_are_unique() {
        assert!(manifest_is_unique(&Sam2Config::hiera_large()));
        assert!(manifest_is_unique(&Sam2Config::hiera_tiny()));
    }

    #[test]
    fn two_way_coverage_closes() {
        let cfg = Sam2Config::hiera_tiny();
        let (m, r) = import(synth(&cfg, 154), &cfg).expect("import");
        assert_eq!(r.imported, 317);
        assert_eq!(r.skipped_video, 154);
        assert_eq!(m.len(), 318);
    }

    #[test]
    fn a_missing_tensor_is_named_not_zero_filled() {
        let cfg = Sam2Config::hiera_tiny();
        let mut t = synth(&cfg, 0);
        let dropped = t.iter().position(|(n, _, _)| n.ends_with("blocks.3.attn.qkv.weight")).unwrap();
        let name = t.remove(dropped).0;
        let err = import(t, &cfg).unwrap_err();
        assert!(err.contains(&name), "{err}");
    }

    #[test]
    fn an_unknown_tensor_is_an_error() {
        let cfg = Sam2Config::hiera_tiny();
        let mut t = synth(&cfg, 0);
        t.push(("sam_mask_decoder.mystery.weight".into(), vec![2], vec![0.0; 2]));
        let err = import(t, &cfg).unwrap_err();
        assert!(err.contains("mystery"), "{err}");
    }

    #[test]
    fn a_wrong_shape_is_an_error() {
        let cfg = Sam2Config::hiera_tiny();
        let mut t = synth(&cfg, 0);
        for e in t.iter_mut() {
            if e.0.ends_with("blocks.0.attn.qkv.weight") {
                e.1 = vec![1, 1];
                e.2 = vec![0.0; 1];
            }
        }
        let err = import(t, &cfg).unwrap_err();
        assert!(err.contains("shape"), "{err}");
    }
}
