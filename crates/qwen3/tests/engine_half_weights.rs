// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving engine keeps its linears in a checkpoint's own half precision
//! (bf16 or fp16) when asked: half the memory of fp32 with no quantization
//! error beyond the checkpoint's own rounding. The reference is the fp32
//! engine given the same weights rounded to that precision: both then compute
//! the same function in fp32 arithmetic, prefill (tiled GEMM) and decode
//! (GEMV) alike.

use std::collections::HashMap;

use model::paged::BlockTable;
use qwen3::serve::Engine;
use qwen3::{Dtype, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// The linears and the head at `dt`; the embedding table and norms, which
/// the engine keeps in fp32, untouched.
fn round(w: &HashMap<String, Vec<f32>>, dt: Dtype) -> HashMap<String, Vec<f32>> {
    let r = |x: f32| match dt {
        Dtype::BF16 => half::bf16::from_f32(x).to_f32(),
        Dtype::F16 => half::f16::from_f32(x).to_f32(),
        _ => x,
    };
    let linear = |k: &str| k == "lm_head.weight" || ((k.contains(".attn.w") || k.contains(".mlp.")) && k.ends_with(".weight"));
    w.iter().map(|(k, v)| (k.clone(), if linear(k) { v.iter().map(|&x| r(x)).collect() } else { v.clone() })).collect()
}

#[test]
fn half_precision_linears_compute_what_fp32_does_on_the_rounded_weights() {
    if gpu_disabled() {
        return;
    }
    // Untied, so the head is a linear of its own; a 40-token prompt so
    // prefill runs the tiled GEMM rather than the GEMV.
    let cfg = QwenConfig { tie_embeddings: false, block_size: 64, ..QwenConfig::tiny() };
    let w = qwen3::init_weights(&cfg, 3);
    let prompt: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    for dt in [Dtype::BF16, Dtype::F16] {
        let rounded = round(&w, dt);
        let run = |eng: &mut Engine| -> (Vec<f32>, Vec<Vec<f32>>) {
            let mut table = BlockTable::new();
            let hidden = eng.prefill_for_perf(&mut table, &prompt);
            let mut steps = Vec::new();
            for t in [5u32, 9, 2] {
                let row = vec![0.01 * t as f32; cfg.d_model as usize];
                steps.push(eng.forward_batched_embed(&mut [&mut table], &row));
            }
            (hidden, steps)
        };
        let mut reference = Engine::from_map(cfg.clone(), &rounded, 4, 32, 1, 16, 64, false, false);
        let mut half = Engine::from_map_tier(cfg.clone(), &w, 4, 32, 1, 16, 64, false, dt);
        assert_eq!(half.weights_tier(), dt, "the engine keeps the requested tier");
        let (want, want_steps) = run(&mut reference);
        let (got, got_steps) = run(&mut half);
        let err = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(err(&got, &want) < 1e-5, "{dt:?} prefill differs by {:e}", err(&got, &want));
        for (g, r) in got_steps.iter().zip(&want_steps) {
            assert!(err(g, r) < 1e-5, "{dt:?} decode differs by {:e}", err(g, r));
        }
        // The head: greedy tokens through the on-device logits.
        let prompts = vec![prompt[..9].to_vec()];
        assert_eq!(half.generate_greedy(&prompts, 12, None), reference.generate_greedy(&prompts, 12, None), "{dt:?} greedy tokens");
    }
}
