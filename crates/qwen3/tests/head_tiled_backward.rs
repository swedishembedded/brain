// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of large-vocabulary language
// models on consumer GPUs for its clients. If your team needs expertise in
// fitting 7B-class fine-tuning into a fixed memory budget then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A head too large for one storage binding (a 152k-token vocabulary at
//! d = 3584 is a 2.18 GB fp32 table against a 2 GiB binding limit) is
//! differentiated a vocab tile at a time, exactly as the forward applies it:
//! the gradients equal the untiled build's, and the recorded forward and
//! backward hold one pass per tile - on the same GEMM kernels an untiled head
//! uses - instead of one binding of the whole table.

use std::collections::HashMap;

use qwen3::{LoraCfg, Qwen, QwenConfig};

/// The tile budget is a process-global environment variable: the tests that
/// set it run one at a time.
static TILE_BUDGET: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const TOKENS: [u32; 12] = [1, 5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2];
const TARGETS: [u32; 12] = [5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2, 1];

fn config(tied: bool, lora: bool) -> QwenConfig {
    QwenConfig { vocab: 48, block_size: 12, tie_embeddings: tied, lora: lora.then(|| LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
}

fn grads(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>) -> (f32, Vec<(String, Vec<f32>)>, Qwen) {
    let m = Qwen::new(cfg.clone(), 1, 12, init);
    m.set_batch(&TOKENS, &TARGETS);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let g = model::Model::optimized_params(&m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, g, m)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |s, x| s.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / scale))
}

#[test]
fn a_head_beyond_the_binding_budget_is_differentiated_in_tiles() {
    if gpu_disabled() {
        return;
    }
    let _serial = TILE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    for (tied, lora) in [(true, false), (false, false), (false, true)] {
        let cfg = config(tied, lora);
        let mut init = qwen3::init_weights(&cfg, 3);
        if lora {
            // Adapters that matter: a zero B would make every A gradient vanish.
            for (n, v) in init.iter_mut().filter(|(n, _)| n.ends_with(".lora_b")) {
                v.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 7 + n.len()) % 11) as f32 * 0.02 - 0.1);
            }
        }
        std::env::remove_var("BRAIN_TILE_BUDGET_WORDS");
        let (want_loss, want, _) = grads(&cfg, &init);
        // A third of the head per binding: three vocab tiles.
        std::env::set_var("BRAIN_TILE_BUDGET_WORDS", (cfg.vocab as u64 * cfg.d_model as u64 / 3).to_string());
        assert!(model::block::vocab_tiles(cfg.vocab as u64, cfg.d_model as u64).len() > 1, "the budget forces tiling");
        let (loss, got, tiled) = grads(&cfg, &init);
        std::env::remove_var("BRAIN_TILE_BUDGET_WORDS");

        let label = format!("tied={tied} lora={lora}");
        assert!((loss - want_loss).abs() < 1e-5 * want_loss.abs().max(1.0), "{label}: loss {loss} vs {want_loss}");
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            assert!(max_rel(g, w) < 1e-4, "{label}: {name} gradient differs ({})", max_rel(g, w));
        }
        let (fwd, bwd) = (tiled.cost_fwd(), tiled.cost_bwd());
        let dispatched = |r: &gpu_core::cost::CostReport, k: &str| r.by_kernel.contains_key(k) || r.uncovered.contains_key(k);
        assert!(dispatched(&fwd, "copy_cols") && dispatched(&bwd, "copy_cols"), "{label}: each vocab tile moves through a dense scratch, in both directions");
        assert!(!dispatched(&fwd, "matmul_tile"), "{label}: the tiled head runs on the GEMM kernels, not the one-thread-per-output column tile");
    }
}

/// The same for tables packed in bf16 (a LoRA build over a bf16 base): a tile
/// is a run of `d_model / 2`-word rows, and the tiled build equals the untiled one.
#[test]
fn a_bf16_head_beyond_the_binding_budget_is_differentiated_in_tiles() {
    if gpu_disabled() {
        return;
    }
    let _serial = TILE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    for tied in [true, false] {
        let cfg = config(tied, true);
        let mut init = qwen3::init_weights(&cfg, 3);
        for (n, v) in init.iter_mut().filter(|(n, _)| n.ends_with(".lora_b")) {
            v.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 7 + n.len()) % 11) as f32 * 0.02 - 0.1);
        }
        let build = || Qwen::new_lora_dt(cfg.clone(), 1, 12, &init, qwen3::Dtype::BF16);
        let run = |m: &Qwen| {
            m.set_batch(&TOKENS, &TARGETS);
            m.zero_grads();
            let loss = m.forward();
            m.backward();
            let g: Vec<(String, Vec<f32>)> = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
            (loss, g)
        };
        std::env::remove_var("BRAIN_TILE_BUDGET_WORDS");
        let whole = build();
        if whole.head_dtype() != qwen3::Dtype::BF16 {
            return; // no bf16 storage path on this device
        }
        let (want_loss, want) = run(&whole);
        // A third of the packed head per binding: several vocab tiles.
        std::env::set_var("BRAIN_TILE_BUDGET_WORDS", (cfg.vocab as u64 * cfg.d_model as u64 / 2 / 3).to_string());
        let tiled = build();
        std::env::remove_var("BRAIN_TILE_BUDGET_WORDS");
        let (loss, got) = run(&tiled);
        let label = format!("tied={tied}");
        assert!((loss - want_loss).abs() < 1e-5 * want_loss.abs().max(1.0), "{label}: loss {loss} vs {want_loss}");
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            assert!(max_rel(g, w) < 1e-4, "{label}: {name} gradient differs ({})", max_rel(g, w));
        }
        let dispatched = |r: &gpu_core::cost::CostReport, k: &str| r.by_kernel.contains_key(k) || r.uncovered.contains_key(k);
        assert!(dispatched(&tiled.cost_fwd(), "copy_cols") && dispatched(&tiled.cost_bwd(), "copy_cols"), "{label}: the packed head is applied in passes too");
    }
}
