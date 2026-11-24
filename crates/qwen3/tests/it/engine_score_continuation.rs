// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for returning the exact policy a served model sampled from, for its
// clients. If your team needs expertise in on-policy reinforcement learning against a serving engine then you can
// procure our services by sending an email to info@swedishembedded.com.

//! `PagedDecoder::score_continuation` on the serving engine: the log-probability of a continuation under the
//! model's full distribution at a temperature, scored through the serving forward, equals the trainer's
//! teacher-forced log-probability of the same tokens, and a call leaves the engine's KV pool as it found it.

use model::logprobs::token_logprobs_scaled;
use model::serve::PagedDecoder;
use qwen3::serve::Engine;
use qwen3::{Dtype, Qwen, QwenConfig};

fn engine_and_trainer() -> (Engine, Qwen) {
    let cfg = QwenConfig { block_size: 16, ..QwenConfig::tiny() };
    let weights = qwen3::init_weights(&cfg, 7);
    let engine = Engine::from_map_tier(cfg.clone(), &weights, 4, 64, 2, 4, 8, false, Dtype::F32);
    (engine, Qwen::new(cfg, 1, 16, &weights))
}

#[test]
fn a_scored_continuation_is_the_trainers_teacher_forced_log_probability() {
    const TOLERANCE: f32 = 1e-3;
    let (mut engine, trainer) = engine_and_trainer();
    let tokens = [3u32, 9, 1, 14, 2, 8, 5, 11];
    let from = 4;
    for temperature in [1.0f32, 0.7] {
        let free_before = engine.free_blocks();
        let served = engine.score_continuation(0, &tokens, from, temperature).expect("the engine scores");
        assert_eq!(engine.free_blocks(), free_before, "scoring returns every block it took");
        let inputs = &tokens[..tokens.len() - 1];
        let targets: Vec<u32> = tokens[1..].to_vec();
        let all = token_logprobs_scaled(&trainer, inputs, &targets, 1.0 / temperature).expect("the trainer scores");
        let want = &all[from - 1..];
        assert_eq!(served.len(), tokens.len() - from);
        for (i, (s, w)) in served.iter().zip(want).enumerate() {
            assert!((s - w).abs() < TOLERANCE, "temperature {temperature}, token {i}: served {s}, trainer {w}");
        }
    }
}

#[test]
fn scoring_refuses_what_it_cannot_score_by_name() {
    let (mut engine, _) = engine_and_trainer();
    let too_long = vec![1u32; engine.max_seq_len() + 1];
    assert!(engine.score_continuation(0, &too_long, 1, 1.0).unwrap_err().contains("exceed"));
    assert!(engine.score_continuation(0, &[1, 2, 99_999], 1, 1.0).unwrap_err().contains("outside the model vocabulary"));
}

/// SPEC: the top list of a scored position is the most probable tokens of the row the serving forward computed there,
/// under `softmax(logits / T)` over the whole row: it equals an independent f64 softmax of the trainer's logits for
/// the same prefix within the engine's tolerance, is ordered most probable first, and the figure of the scored token
/// is its own top entry bit for bit when it is among them (one pass, one arithmetic).
#[test]
fn a_scored_continuation_reports_the_top_of_the_row_it_scored() {
    const TOLERANCE: f64 = 1e-3;
    const TOP: usize = 5;
    let (mut engine, trainer) = engine_and_trainer();
    let tokens = [3u32, 9, 1, 14, 2, 8, 5, 11];
    let from = 4;
    let logits = trainer.logits_all(&tokens[..tokens.len() - 1]);
    let vocab = logits.len() / (tokens.len() - 1);
    for temperature in [1.0f32, 0.7] {
        let scored = engine.score_continuation_top(0, &tokens, from, temperature, TOP).expect("the engine scores the top");
        let figures = engine.score_continuation(0, &tokens, from, temperature).expect("the engine scores");
        assert_eq!(scored.len(), tokens.len() - from);
        for (i, position) in scored.iter().enumerate() {
            assert_eq!(position.log_prob.to_bits(), figures[i].to_bits(), "position {i}: the top pass scores the token as the plain pass does");
            let p = from + i;
            let scaled: Vec<f64> = logits[(p - 1) * vocab..p * vocab].iter().map(|&z| f64::from(z) / f64::from(temperature)).collect();
            let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = max + scaled.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
            let mut ranked: Vec<(u32, f64)> = scaled.iter().enumerate().map(|(t, z)| (t as u32, z - lse)).collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            assert_eq!(position.top.len(), TOP);
            for (got, want) in position.top.iter().zip(&ranked) {
                assert!((f64::from(got.1) - want.1).abs() < TOLERANCE, "temperature {temperature}, position {i}: served {got:?}, trainer {want:?}");
            }
            if let Some(entry) = position.top.iter().find(|e| e.0 == tokens[p]) {
                assert_eq!(entry.1.to_bits(), position.log_prob.to_bits(), "position {i}: the token's figure is its top entry");
            }
        }
    }
}
