// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single ZImageTransformerBlock forward parity vs diffusers.
//!
//! Golden (`tests/golden/zimage_block.safetensors`, committed, baked by
//! `tools/zimage_block_dump_reference.py`): a small block
//! (dim 48, 2 heads, T 8) with random weights + inputs and its reference output.
//! Exercises adaLN folding, the double-RMSNorm sandwich, QK-norm attention with
//! multi-axis interleaved RoPE, and SwiGLU. No external weights needed — the
//! golden is self-contained. Runs on the CPU backend.

use std::collections::HashMap;

use zimage::{BlockDims, ZImageBlock};

/// Resolve a fixture under the fetched `testdata/` tree (`make fetch/testdata`;
/// override the root with `BRAIN_TESTDATA`).
use brain_testutil::testdata;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn zimage_block_matches_diffusers() {
    let fixture = testdata("golden/zimage/zimage_block.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent — run `make fetch/testdata`");
        return;
    }
    let st = checkpoint::safetensors::read(&fixture).expect("read block golden");
    let mut weights: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut input: HashMap<String, Vec<f32>> = HashMap::new();
    for t in st {
        if let Some(name) = t.name.strip_prefix('_') {
            input.insert(name.to_string(), t.data);
        } else {
            weights.insert(format!("blk.{}", t.name), (t.shape, t.data));
        }
    }

    let d = BlockDims::new(48, 2);
    assert_eq!((d.head_dim, d.cdim, d.hidden), (24, 48, 128));
    let t = 8u32;
    let blk = ZImageBlock::new(&weights, "blk", d, t, true, Some("cpu"));

    let got = blk.forward(&input["x"], &input["c"], &input["cos"], &input["sin"]);
    let want = &input["out"];
    assert_eq!(got.len(), want.len(), "output len {} != golden {}", got.len(), want.len());

    let cos = cosine(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    let want_max = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    eprintln!("Z-Image block parity: cosine={cos:.6}  max_abs={max_abs:.5}  (|want|max={want_max:.3})");
    assert!(cos >= 0.9999, "cosine {cos:.6} < 0.9999");
    assert!(max_abs <= 1e-2 * want_max.max(1.0), "max_abs {max_abs:.5} too large");
}
