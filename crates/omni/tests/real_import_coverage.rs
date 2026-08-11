// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `omni::import::hf_to_brain` against EVERY one of the real checkpoint's
//! 28 010 tensor names (not the hand-picked shape samples in
//! `import.rs`'s own unit tests) — the strongest coverage check available
//! without the weight bytes themselves, since `model.safetensors.index.json`
//! lists every name up front.
//!
//! Real-weight-adjacent, so it follows the engine's opt-in-env-var pattern:
//! skips cleanly when the checkpoint dir is not present.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test real_import_coverage -- --ignored`

use std::collections::HashSet;
use std::path::PathBuf;

fn hf_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("BRAIN_OMNI_HF_DIR").ok()?);
    d.join("model.safetensors.index.json").exists().then_some(d)
}

#[test]
#[ignore]
fn every_real_tensor_name_maps_or_is_a_known_qkv_leaf() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset or model.safetensors.index.json missing");
        return;
    };
    let idx_json = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("read index");
    let idx: serde_json::Value = serde_json::from_str(&idx_json).expect("parse index");
    let weight_map = idx["weight_map"].as_object().expect("weight_map object");
    assert_eq!(weight_map.len(), 28010, "index tensor count drifted from the recorded checkpoint shape");

    let mut unmapped = Vec::new();
    let mut seen_brain_names: HashSet<String> = HashSet::new();
    let mut collisions = Vec::new();
    for name in weight_map.keys() {
        if omni::import::is_qkv_fuse_leaf(name) {
            continue;
        }
        match omni::import::hf_to_brain(name) {
            Some(bn) => {
                if !seen_brain_names.insert(bn.clone()) {
                    collisions.push(bn);
                }
            }
            None => unmapped.push(name.clone()),
        }
    }

    assert!(
        unmapped.is_empty(),
        "{} real tensor names have no mapping (first 20: {:?})",
        unmapped.len(),
        &unmapped[..unmapped.len().min(20)]
    );
    assert!(collisions.is_empty(), "two HF tensors mapped to the same brain name: {collisions:?}");

    println!("omni::import::hf_to_brain covers all {} non-qkv-leaf real tensor names, no collisions.", weight_map.len());
}

/// The real 70 GB source -> ~35 GB int8-native import, run for the first
/// time against all 15 shards (`.agents/roadmap/omni.md` recorded this as
/// "not yet done -- mechanism proven on synthetic and partial data only, 4 of
/// 15 shards"). Two checks beyond the metadata-level name mapping above,
/// which never touches actual written bytes:
///
/// 1. Two-way name coverage against the REAL WRITTEN output (not a dry-run
///    prediction): every expected brain name (derived the same way
///    `import_as` derives them -- `hf_to_brain` plus the audio QKV fuse
///    pattern) is present in the output file, and every name IN the output
///    file was expected -- no extra, unexplained tensors either.
/// 2. A bounded VALUE spot-check (exhaustive byte-for-byte over 35 GB would
///    dominate this test's runtime for marginal extra confidence): the
///    largest quantized tensor (`thinker.embed_tokens.weight`, int8-packed +
///    its per-channel `.scale`) dequantizes close to the real HF source, and
///    a 1-D (never-quantized) tensor matches the source exactly.
///
/// Real-weight-adjacent AND real-disk-adjacent: skips cleanly when the
/// checkpoint dir is absent, and needs ~35 GB free at `BRAIN_OMNI_IMPORT_OUT`
/// (defaults to a scratch path under the system temp dir, removed on success).
///
/// Slow (streams the full 70 GB source): expect real wall-clock minutes, not
/// seconds -- this is the actual mechanism running for real, not a mock.
#[test]
#[ignore]
fn full_real_import_produces_a_two_way_covered_checkpoint() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset or model.safetensors.index.json missing");
        return;
    };
    let shards = std::fs::read_dir(&dir).expect("read hf dir").filter(|e| e.as_ref().is_ok_and(|e| e.path().extension().is_some_and(|x| x == "safetensors"))).count();
    assert_eq!(shards, 15, "expected all 15 real shards on disk, found {shards} -- this test needs the full checkpoint, not a partial one");

    let out_path = std::env::var("BRAIN_OMNI_IMPORT_OUT").unwrap_or_else(|_| {
        std::env::temp_dir().join(format!("omni_full_import_{}.safetensors", std::process::id())).to_string_lossy().into_owned()
    });

    println!("running the real 70 GB -> ~35 GB import against all 15 shards -- this streams the full source, expect real minutes...");
    let t0 = std::time::Instant::now();
    omni::import::import_as(dir.to_str().unwrap(), &out_path, Some("omni-full-import-test")).expect("import_as must succeed against the real, complete checkpoint");
    println!("import_as finished in {:.1}s", t0.elapsed().as_secs_f64());

    let out_bytes = std::fs::metadata(&out_path).expect("stat output").len();
    println!("output: {:.1} GB", out_bytes as f64 / 1e9);
    assert!(out_bytes > 20_000_000_000, "output is only {out_bytes} bytes -- far short of the expected ~35 GB, something truncated");
    assert!(out_bytes < 50_000_000_000, "output is {out_bytes} bytes -- far over the expected ~35 GB, quantization did not engage as expected");

    // ---- 1. two-way name coverage against the real written file ----
    let idx_json = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("read index");
    let idx: serde_json::Value = serde_json::from_str(&idx_json).expect("parse index");
    let weight_map = idx["weight_map"].as_object().expect("weight_map object");

    let mut expected: HashSet<String> = HashSet::new();
    let mut qkv_layers: HashSet<u32> = HashSet::new();
    for name in weight_map.keys() {
        if omni::import::is_qkv_fuse_leaf(name) {
            let b: u32 = name.strip_prefix("thinker.audio_tower.layers.").unwrap().split_once('.').unwrap().0.parse().unwrap();
            qkv_layers.insert(b);
            continue;
        }
        if let Some(bn) = omni::import::hf_to_brain(name) {
            expected.insert(bn);
        }
    }
    for b in qkv_layers {
        expected.insert(format!("audio.blocks.{b}.qkv.weight"));
        expected.insert(format!("audio.blocks.{b}.qkv.bias"));
    }

    let out_reader = checkpoint::weightio::WeightReader::open(&out_path).expect("open real import output");
    let all_names: HashSet<String> = out_reader.names().map(str::to_string).collect();
    // A `.scale`-suffixed name is a SYNTHETIC quantization sibling only if
    // stripping the suffix yields a name that is ITSELF a real (quantized)
    // tensor in the output -- NOT every `.scale`-suffixed name qualifies:
    // Code2Wav's `pre_transformer.layers.N.{mlp,self_attn}_layer_scale.scale`
    // is a real HF tensor whose OWN name happens to end in `.scale` (a
    // layer-scale gain, unrelated to quantization), so a blanket suffix
    // filter here would (and on the real checkpoint, initially did) wrongly
    // exclude 16 real tensors from the coverage comparison.
    let is_synthetic_scale_sibling = |n: &str| n.strip_suffix(".scale").is_some_and(|base| all_names.contains(base));
    let actual: HashSet<String> = all_names.iter().filter(|n| !is_synthetic_scale_sibling(n)).cloned().collect();

    let missing: Vec<&String> = expected.difference(&actual).collect();
    let extra: Vec<&String> = actual.difference(&expected).collect();
    assert!(missing.is_empty(), "{} expected tensors missing from the real import output (first 20: {:?})", missing.len(), &missing[..missing.len().min(20)]);
    assert!(extra.is_empty(), "{} unexplained tensors in the real import output (first 20: {:?})", extra.len(), &extra[..extra.len().min(20)]);
    println!("two-way name coverage: {} tensors, exact match against the real written output.", actual.len());

    // Every quantized (U32) tensor has exactly one .scale sibling, and vice versa.
    // `dtype()`, not `tensor_u32().is_some()`: the latter PANICS on a
    // non-U32 tensor by design (Phase 4's "errors loudly, never silently
    // empty" fix) rather than returning None, so it is the wrong accessor
    // for a "is this U32?" probe over tensors of mixed, unknown dtype.
    let scale_names: HashSet<String> = out_reader.names().filter(|n| n.ends_with(".scale")).map(str::to_string).collect();
    for base in &actual {
        let is_u32 = out_reader.dtype(base) == Some("U32");
        let has_scale = scale_names.contains(&format!("{base}.scale"));
        assert_eq!(is_u32, has_scale, "'{base}': quantized (U32) XOR has a .scale sibling -- these must always agree");
    }

    // ---- 2. bounded value spot-check against the real HF source ----
    let src = checkpoint::weightio::WeightReader::open_hf_dir(&dir).expect("open real source for comparison");

    // A never-quantized 1-D tensor: byte-identical (both are F32, no lossy step).
    let src_norm = src.tensor("thinker.model.norm.weight").expect("source thinker.model.norm.weight");
    let out_norm = out_reader.tensor("thinker.norm.weight").expect("output thinker.norm.weight");
    assert_eq!(src_norm, out_norm, "a never-quantized 1-D tensor must round-trip byte-identical");

    // The largest quantized tensor: dequantize the packed U32 + per-channel
    // scale and compare against the real source within int8's own known
    // quantization error (matches model::moe's own 0.0084 rel-L2 precedent
    // for this exact packing -- generous here since this is per-element, not
    // an aggregated GEMM error).
    let src_embed = src.tensor("thinker.model.embed_tokens.weight").expect("source embed_tokens");
    // The WRITTEN shape is already [n, k/4] (import.rs's own plan:
    // `vec![n, k / 4]` -- StWriter::create_mixed plans one u32 per 4 packed
    // int8 lanes), not [n, k] -- `kg` names that k/4 explicitly so this
    // doesn't get divided by 4 a second time (the bug this comment replaces:
    // that second division under-read the packed buffer by 4x, caught by the
    // `packed.len() == n * kg` assertion below on the FIRST run against the
    // real checkpoint).
    let shape = out_reader.shape("thinker.embed_tokens.weight").expect("output embed_tokens shape").to_vec();
    let (n, kg) = (shape[0] as usize, shape[1] as usize);
    let k = kg * 4;
    let packed = out_reader.tensor_u32("thinker.embed_tokens.weight").expect("output embed_tokens packed");
    let scale = out_reader.tensor("thinker.embed_tokens.weight.scale").expect("output embed_tokens scale");
    assert_eq!(packed.len(), n * kg);
    assert_eq!(scale.len(), n);

    let mut max_abs_err = 0f32;
    let mut max_val = 0f32;
    for r in 0..n {
        let s = scale[r];
        for g in 0..kg {
            let word = packed[r * kg + g];
            for b in 0..4 {
                let q = ((word >> (8 * b)) & 0xFF) as u8 as i8;
                let got = q as f32 * s;
                let want = src_embed[r * k + g * 4 + b];
                max_abs_err = max_abs_err.max((got - want).abs());
                max_val = max_val.max(want.abs());
            }
        }
    }
    let rel_err = max_abs_err / max_val.max(1e-8);
    println!("embed_tokens dequant vs real source: max_abs_err={max_abs_err:.6} max_val={max_val:.4} rel_err={rel_err:.4}");
    // Per-channel int8 (127 levels) on real weight distributions: a rel error
    // comfortably under 5% is the expected accuracy floor, not a coincidence.
    assert!(rel_err < 0.05, "embed_tokens dequant strayed too far from the real source: rel_err={rel_err}");

    // Clean up the ~35 GB output unless the caller pointed at a path they
    // want to keep (BRAIN_OMNI_IMPORT_OUT set explicitly).
    if std::env::var("BRAIN_OMNI_IMPORT_OUT").is_err() {
        std::fs::remove_file(&out_path).ok();
    }
}
