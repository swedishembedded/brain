// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint import coverage. Env-gated; skips without weights.
//!
//! BRAIN_FLUX2_TRANSFORMER    diffusers transformer/ dir or .safetensors (klein 4B)
//! BRAIN_FLUX2_GGUF           BFL-named BF16 GGUF (klein 9B)

use flux2::{import_bfl, import_diffusers, Flux2Config};

#[test]
fn diffusers_4b_imports_with_full_coverage() {
    let Ok(path) = std::env::var("BRAIN_FLUX2_TRANSFORMER") else {
        brain_testutil::skip("BRAIN_FLUX2_TRANSFORMER unset");
        return;
    };
    let p = std::path::Path::new(&path);
    let mut tensors = Vec::new();
    if p.is_dir() {
        // diffusers layout: one or more *.safetensors alongside config.json
        let mut files: Vec<_> = std::fs::read_dir(p)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no .safetensors under {path}");
        for f in files {
            tensors.extend(checkpoint::safetensors::read(f.to_str().unwrap()).unwrap());
        }
    } else {
        tensors = checkpoint::safetensors::read(&path).unwrap();
    }
    assert_eq!(tensors.len(), 169, "diffusers 4B layout has 169 tensors");
    let cfg = Flux2Config::klein_4b();
    let map = import_diffusers(tensors, &cfg).unwrap();
    assert_eq!(map.len(), 149);
    // spot-check a fused qkv is finite and non-degenerate
    let (shape, w) = &map["double_blocks.0.img_attn.qkv.weight"];
    assert_eq!(shape, &vec![3 * cfg.hidden, cfg.hidden]);
    assert!(w.iter().all(|v| v.is_finite()));
    let mean_abs: f32 = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32;
    assert!(mean_abs > 1e-4, "suspicious near-zero weights: {mean_abs}");
}

#[test]
fn gguf_9b_imports_with_full_coverage() {
    let Ok(path) = std::env::var("BRAIN_FLUX2_GGUF") else {
        brain_testutil::skip("BRAIN_FLUX2_GGUF unset");
        return;
    };
    let tensors = checkpoint::gguf::read(&path).unwrap();
    let map = import_bfl(tensors, &Flux2Config::klein_9b()).unwrap();
    assert_eq!(map.len(), 201);
}
