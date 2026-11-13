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
