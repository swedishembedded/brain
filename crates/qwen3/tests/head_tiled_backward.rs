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
//! the gradients equal the untiled build's, and the recorded backward holds
//! one dispatch per tile instead of one binding of the whole table.

use std::collections::HashMap;

use qwen3::{LoraCfg, Qwen, QwenConfig};

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
        let bwd = tiled.cost_bwd();
        let dispatched = |k: &str| bwd.by_kernel.contains_key(k) || bwd.uncovered.contains_key(k);
        assert!(dispatched("matmul_dx_tile"), "{label}: the head's input gradient is applied tile by tile");
        assert_eq!(dispatched("matmul_dw_tile"), !lora, "{label}: a trainable head's weight gradient is written tile by tile");
    }
}
