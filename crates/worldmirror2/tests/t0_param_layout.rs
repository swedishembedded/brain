// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T0 — device-free parameter-layout gate: `MirrorConfig::param_list()` must
//! agree tensor-for-tensor (names AND shapes) with the committed safetensors
//! header fixture dumped from the reference HY-WorldMirror-2.0 checkpoint.
//! With MIRROR_CKPT set, also validates a strict live import of the real file.

use std::collections::HashMap;

use worldmirror2::config::MirrorConfig;

fn fixture() -> HashMap<String, Vec<usize>> {
    let raw = include_str!("golden/header.json");
    let v: serde_json::Value = serde_json::from_str(raw).unwrap();
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, shape)| {
            let s = shape.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            (k.clone(), s)
        })
        .collect()
}

#[test]
fn param_list_matches_reference_header() {
    let cfg = MirrorConfig::default();
    let ours: HashMap<String, Vec<usize>> = cfg.param_list().into_iter().collect();
    let theirs = fixture();

    let mut missing: Vec<&String> = theirs.keys().filter(|k| !ours.contains_key(*k)).collect();
    missing.sort();
    assert!(missing.is_empty(), "param_list is missing {} tensors, e.g. {:?}", missing.len(), &missing[..missing.len().min(8)]);

    let mut extra: Vec<&String> = ours.keys().filter(|k| !theirs.contains_key(*k)).collect();
    extra.sort();
    assert!(extra.is_empty(), "param_list declares {} unknown tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(8)]);

    for (k, shape) in &ours {
        assert_eq!(shape, &theirs[k], "shape mismatch for `{k}`");
    }
    assert_eq!(ours.len(), 1545);
}

#[test]
fn total_params_are_1_26b() {
    let n: usize = MirrorConfig::default()
        .param_list()
        .iter()
        .map(|(_, s)| s.iter().product::<usize>())
        .sum();
    // 5.05 GB of f32 → ~1.263 B parameters.
    assert!((1_200_000_000..1_300_000_000).contains(&n), "total params {n}");
}

/// The camera head's refine-net blocks are dispatched from `cam::cam_shape`
/// and SIZED by `param_list`; when those two disagree the block reads past the
/// end of its own weights, which is undefined state rather than a wrong number
/// and so shows up only as whatever the allocator happened to leave there.
///
/// A non-default `heads`/`mlp_ratio` is the case that catches it: at the
/// reference config every candidate constant coincides.
#[test]
fn cam_block_shape_matches_its_declared_weights() {
    let cfg = MirrorConfig { heads: 8, mlp_ratio: 3, ..MirrorConfig::default() };
    let p: HashMap<String, Vec<usize>> = cfg.param_list().into_iter().collect();
    let sh = worldmirror2::cam::cam_shape(cfg.dim as u32, cfg.heads as u32, cfg.mlp_ratio as u32);
    let d2 = sh.dim as usize;
    assert_eq!(p["cam_head.refine_net.0.attn.qkv.weight"], vec![3 * d2, d2]);
    assert_eq!(p["cam_head.refine_net.0.mlp.fc1.weight"], vec![sh.mlp as usize, d2]);
    assert_eq!(p["cam_head.refine_net.0.mlp.fc2.weight"], vec![d2, sh.mlp as usize]);
    assert_eq!(d2 % sh.heads as usize, 0, "head split must divide the block width");
}

/// Live gate against the real checkpoint (slow, needs the 5 GB file).
#[test]
fn live_import_strict() {
    let Ok(path) = std::env::var("MIRROR_CKPT") else { return };
    let cfg = MirrorConfig::default();
    let map = worldmirror2::import::load(&path, &cfg).expect("strict import");
    assert_eq!(map.len(), 1545);
}
