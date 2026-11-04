// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving engine prefills prompts that mix token ids with ready-made
//! embedding rows (vision-language models splice image features into the
//! text stream). The model's own KV-cache prefill is the reference; chunk
//! boundaries may fall anywhere, including inside and across image regions,
//! and only the leading token run may be served from the prefix cache.

use model::paged::BlockTable;
use qwen3::model::PrefillInput;
use qwen3::serve::Engine;
use qwen3::{Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn cfg() -> QwenConfig {
    QwenConfig { block_size: 32, ..QwenConfig::tiny() }
}

/// Deterministic embedding rows unlike any token's.
fn image_rows(n: usize, d: usize, seed: f32) -> Vec<f32> {
    (0..n * d).map(|i| 0.5 * ((i as f32) * 0.37 + seed).sin()).collect()
}

/// Two image regions between token runs: `t t t t t [img x3] t t [img x2] t`.
fn mixed<'a>(a: &'a [f32], b: &'a [f32], d: usize) -> Vec<PrefillInput<'a>> {
    let mut v: Vec<PrefillInput> = [2u32, 9, 4, 11, 7].iter().map(|&t| PrefillInput::Token(t)).collect();
    v.extend(a.chunks(d).map(PrefillInput::Embed));
    v.extend([5u32, 1].iter().map(|&t| PrefillInput::Token(t)));
    v.extend(b.chunks(d).map(PrefillInput::Embed));
    v.push(PrefillInput::Token(3));
    v
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

#[test]
fn mixed_prefill_matches_the_model_at_every_chunk_size() {
    if gpu_disabled() {
        return;
    }
    let cfg = cfg();
    let d = cfg.d_model as usize;
    let w = qwen3::init_weights(&cfg, 7);
    let (a, b) = (image_rows(3, d, 0.3), image_rows(2, d, 1.9));
    let inputs = mixed(&a, &b, d);
    let want = Qwen::from_tensors_decode(cfg.clone(), &w, cfg.block_size).prefill(&inputs);
    for max_prefill in [1, 2, 3, 4, 16] {
        let mut eng = Engine::from_map(cfg.clone(), &w, 4, 32, 1, 12, max_prefill, false, false);
        let got = eng.prefill_mixed(&mut BlockTable::new(), &inputs);
        let err = max_rel(&got, &want);
        assert!(err < 1e-4, "max_prefill {max_prefill}: engine differs from the model by {err:e} relative");
    }
}

#[test]
fn embedding_rows_equal_to_the_token_rows_prefill_identically() {
    if gpu_disabled() {
        return;
    }
    let cfg = cfg();
    let d = cfg.d_model as usize;
    let w = qwen3::init_weights(&cfg, 7);
    let prompt = [2u32, 9, 4, 11, 7, 3, 8];
    let emb = &w["tok.weight"];
    let rows: Vec<f32> = prompt[2..5].iter().flat_map(|&t| emb[t as usize * d..(t as usize + 1) * d].iter().copied()).collect();
    let mut inputs: Vec<PrefillInput> = prompt[..2].iter().map(|&t| PrefillInput::Token(t)).collect();
    inputs.extend(rows.chunks(d).map(PrefillInput::Embed));
    inputs.extend(prompt[5..].iter().map(|&t| PrefillInput::Token(t)));

    let mut eng = Engine::from_map(cfg.clone(), &w, 4, 32, 1, 12, 3, false, false);
    let tokens = eng.prefill_for_perf(&mut BlockTable::new(), &prompt);
    let mut eng = Engine::from_map(cfg.clone(), &w, 4, 32, 1, 12, 3, false, false);
    let spliced = eng.prefill_mixed(&mut BlockTable::new(), &inputs);
    assert_eq!(spliced, tokens, "an embedding row equal to the token's own row must not change the result");
}

#[test]
fn only_the_leading_token_run_is_served_from_the_prefix_cache() {
    if gpu_disabled() {
        return;
    }
    let cfg = cfg();
    let d = cfg.d_model as usize;
    let w = qwen3::init_weights(&cfg, 7);
    let (a, b) = (image_rows(3, d, 0.3), image_rows(2, d, 1.9));
    let (a2, b2) = (image_rows(3, d, 4.1), image_rows(2, d, 2.7));
    let first = mixed(&a, &b, d);
    let second = mixed(&a2, &b2, d);
    let want = Qwen::from_tensors_decode(cfg.clone(), &w, cfg.block_size).prefill(&second);

    let mut eng = Engine::from_map(cfg.clone(), &w, 4, 32, 1, 12, 4, false, false);
    let mut t1 = BlockTable::new();
    eng.prefill_mixed(&mut t1, &first);
    // The leading run is 5 tokens: exactly one full block of 4 is cacheable;
    // the block holding the image rows must not be indexed.
    assert_eq!(eng.prefix_stats().2, 1, "only the leading token run's full blocks are indexed");
    let got = eng.prefill_mixed(&mut BlockTable::new(), &second);
    assert_eq!(eng.prefix_stats().0, 4, "the second prompt reuses the leading block");
    let err = max_rel(&got, &want);
    assert!(err < 1e-4, "prefix reuse changed the result by {err:e} relative");
}
