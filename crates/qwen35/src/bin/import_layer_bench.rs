// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Partial-layer validation for the `model::fp8`/`model::int8` speedups: runs
//! `qwen35::import::import_layer` (the same call `qwen35::stream`'s
//! per-layer window fill makes) over a handful of REAL layer shards under
//! `BRAIN_QWEN35_DIR`, timed end to end, at the real 27B checkpoint's own
//! dimensions.
//!
//! This is the "partial re-run" validation `import_profile`'s isolated,
//! single-tensor numbers cannot stand in for on their own: it exercises the
//! WHOLE per-layer import path (every FP8 tensor a layer has, not just one),
//! through the real `MmapSafetensors`/`dequantize_fp8_pairs`/`quantize_weight`
//! call chain, at the scale a real streaming pass actually uses. It does NOT
//! replace the full 64-layer `streaming_forward.rs::
//! full_chain_streams_all_64_real_layers_within_budget` ignored test (which
//! also exercises the GPU mixer math and the RSS budget) - only the import
//! stage this task's fix touches.
//!
//! Usage: `import_layer_bench <BRAIN_QWEN35_DIR> [first_layer] [count]`
//!   first_layer default 0, count default 6.

use std::path::PathBuf;
use std::time::Instant;

use qwen35::config::Qwen35Config;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(a.get(1).unwrap_or_else(|| panic!("usage: import_layer_bench <BRAIN_QWEN35_DIR> [first_layer] [count]")));
    let first: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let count: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);

    let cfg = Qwen35Config::qwen38_27b();
    println!("import_layer_bench: dir={} layers {first}..{} (of {})", dir.display(), first + count, cfg.n_layers);

    let mut total = 0f64;
    for l in first..first + count {
        let shard = dir.join(format!("layers-{l}.safetensors"));
        if !shard.exists() {
            eprintln!("layer {l}: {} missing, stopping early", shard.display());
            break;
        }
        let t0 = Instant::now();
        let reader = checkpoint::mmap::MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
        let out = qwen35::import::import_layer(&reader, &cfg, l, 128).unwrap_or_else(|e| panic!("import_layer({l}): {e}"));
        let secs = t0.elapsed().as_secs_f64();
        total += secs;
        let numel: usize = out.values().map(|v| v.len()).sum();
        println!("layer {l:>2}: {:>7.3} s  ({} tensors, {numel} elements)", secs, out.len());
    }
    println!("\ntotal for {count} layers: {total:.3} s ({:.3} s/layer avg)", total / count as f64);
    println!("extrapolated full 64-layer import cost: {:.1} s ({:.2} min)", total / count as f64 * 64.0, total / count as f64 * 64.0 / 60.0);
}
