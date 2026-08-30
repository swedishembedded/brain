// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Diagnostic: track per-position cosine divergence between int8 and fp32
//! across varying layer depths on the REAL Qwen3.8-27B weights.
//!
//! This extends `gguf_i8_vs_fp32_real`'s single-depth comparison into a
//! depth-sweep that shows exactly how the GDN recurrent-state quantization
//! error compounds along BOTH sequence position and layer depth.
//!
//! The key finding from the roadmap: at 32 real layers, cosine drops from
//! 0.9888 (pos 0) to 0.7988 (pos 2) - the error compounds along the
//! SEQUENCE, not just with depth. This test makes that visible at depths
//! that fit in one card's VRAM.
//!
//! ```text
//! BRAIN_QWEN35_GGUF=/path/to/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gguf_i8_divergence_diagnostic -- --nocapture
//! ```
//!
//! Control the depth sweep with `BRAIN_QWEN35_I8_DIAG_DEPTHS=4,8,12` (default:
//! 4,8). Each depth builds BOTH an fp32 and int8 stage from the same GGUF
//! bytes, so the VRAM cost is roughly `depth * 1.5 GB` (fp32 side, dominant).

use checkpoint::gguf::MmapGguf;
use model::shard::Shard;
use qwen35::int8_gguf_resident::{resident_config, shard_source};
use qwen35::model::Qwen35;

/// Tokens to push through each depth. Keeps VRAM bounded while showing the
/// position-compounding trend (the roadmap's key finding).
const STEPS: u32 = 8;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)) as f32 }
}

fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt()
}

/// Parse `BRAIN_QWEN35_I8_DIAG_DEPTHS` as a comma-separated list of layer
/// counts to sweep. Default: 4 and 8 (safe on every box with a discrete GPU).
fn depths() -> Vec<u32> {
    std::env::var("BRAIN_QWEN35_I8_DIAG_DEPTHS")
        .ok()
        .map(|s| s.split(',').filter_map(|d| d.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8])
}

#[test]
fn per_position_cosine_divergence_across_depths() {
    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
        return;
    };
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("no discrete GPU - fp32 side needs VRAM");
        return;
    }

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let base_cfg = resident_config(&mg, STEPS).expect("resident_config");
    let d = base_cfg.d_model as usize;
    let depths = depths();

    // Shared prompt - real tokens, not synthetic.
    let gtok = mg.tokenizer().expect("embedded tokenizer");
    let tok = data::qwen_tokenizer::QwenBpe::from_gguf(&gtok).expect("QwenBpe::from_gguf");
    let ids = {
        use data::tokenizer::Tokenizer;
        tok.encode("The capital city of France is Paris, and the capital city of Germany is")
    };
    assert!(ids.len() as u32 >= STEPS, "prompt must be at least {STEPS} tokens");

    // Header for the divergence table.
    println!();
    println!("=== Per-position cosine(int8, fp32) divergence across depth ===");
    println!("prompt: \"The capital city of France is Paris, and the capital city of Germany is\"");
    println!("steps:  {STEPS}");
    println!();

    // Column header: one column per position.
    print!("{:>6}", "depth");
    for pos in 0..STEPS {
        print!("  pos {pos:>2}");
    }
    print!("  {:>8}", "worst");
    println!();

    let mut all_worst = 1.0f32;

    for &layers in &depths {
        let mut cfg = base_cfg.clone();
        cfg.n_layers = layers;

        let shard = Shard { start: 0, end: layers as usize, embed: false, head: false, gpu_index: Shard::ANY_GPU };
        let src = shard_source(&mg, &cfg, &shard).unwrap_or_else(|e| panic!("fetch plan for {layers} layers: {e}"));

        eprintln!("  building {layers} layers (fp32 + int8)...");
        let t0 = std::time::Instant::now();
        let i8 = Qwen35::new_i8_shard(cfg.clone(), 1, STEPS, &src, shard.clone());
        let fp32 = Qwen35::new_fp32_shard_src(cfg.clone(), 1, STEPS, &src, shard.clone());
        eprintln!("  built in {:.1} s", t0.elapsed().as_secs_f64());

        i8.reset_decode_cache();
        fp32.reset_decode_cache();

        let mut worst = 1.0f32;
        print!("{layers:>6}");
        for (pos, &id) in ids.iter().take(STEPS as usize).enumerate() {
            let row = mg.tensor_range("token_embd.weight", id as usize * d, d).expect("embedding row").expect("dequantize");
            let a = i8.step_with_input(id, Some(&row));
            let b = fp32.step_with_input(id, Some(&row));
            let c = cosine(&a, &b);
            worst = worst.min(c);
            all_worst = all_worst.min(c);
            print!("  {c:>8.4}");
            // Detailed per-position line for debugging.
            let ri = rms(&a);
            let rf = rms(&b);
            eprintln!("    depth={layers:>2} pos={pos} tok={id}: cosine={c:.6} rms_i8={ri:.4} rms_fp32={rf:.4}");
        }
        print!("  {worst:>8.4}");
        println!();
    }

    println!();
    println!("worst cosine overall: {all_worst:.6}");
    println!();

    // The roadmap's key finding: at 32 layers, cosine drops to ~0.799 at
    // position 2. At 8 layers, worst cosine is ~0.986. This test's floor is
    // set to the 8-layer measurement minus margin - enough to catch regressions
    // without failing on the known per-layer quantization cost.
    //
    // If you raise the depth beyond what fits in VRAM, this floor will need
    // to be lowered accordingly. The floor is NOT a quality claim; it is a
    // regression guard calibrated against measured hardware.
    assert!(all_worst > 0.95, "int8 has decorrelated from fp32: worst cosine {all_worst} across all depths/positions");

    // Print the trend: does cosine degrade with position? With depth?
    println!("=== Trend analysis ===");
    if depths.len() >= 2 {
        let first = depths[0];
        let last = depths[depths.len() - 1];
        println!("depth {first}..{last}: compare worst-per-position cosines to see depth compounding");
        println!("within each depth: compare pos 0 vs pos {STEPS} to see sequence compounding");
    }
    println!("The roadmap reports: 32 layers gives cosine 0.9888→0.7988 over positions 0→2.");
    println!("This test's depths are limited by fp32 VRAM; the trend should be visible even at 8 layers.");
}
