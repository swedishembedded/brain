// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving-equivalence gates for recurrent-depth
// decoders for its clients. If your team needs expertise in proving a paged,
// continuous-batching engine reproduces a prelude/core/coda model's own forward
// past, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The paged engine must produce exactly the tokens the model's own
//! full-sequence forward produces, for every sequence in a batch, with each
//! stage holding its own block table and paged history.

use std::collections::HashMap;

use huginn::config::HuginnConfig;
use huginn::init::init_weights;
use huginn::model::Huginn;
use huginn::serve::Engine;

use ::model::paged::BlockTable;
use ::model::serve::PagedDecoder;

fn state_vector(cfg: &HuginnConfig) -> Vec<f32> {
    // The reference's `like-init` state is a draw; any fixed draw serves, as
    // long as the engine and the materialized forward use the same one.
    let w = init_weights(cfg, 41);
    w["tok.weight"][..cfg.d_model as usize].to_vec()
}

fn argmax(s: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, v) in s.iter().enumerate() {
        if *v > best.1 {
            best = (i, *v);
        }
    }
    best.0 as u32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// The materialized forward over `seq`, starting every row from `state`; the
/// last row is the next-token distribution after `seq`.
fn model_last_row(cfg: &HuginnConfig, w: &HashMap<String, Vec<f32>>, steps: u32, state: &[f32], seq: &[u32]) -> Vec<f32> {
    let model = Huginn::new(cfg.clone(), 1, seq.len() as u32, w);
    model.set_tokens(seq);
    model.set_state(&state.repeat(seq.len()));
    let rows = model.forward_rows(steps);
    let v = cfg.vocab as usize;
    rows[(seq.len() - 1) * v..].to_vec()
}

#[test]
fn a_paged_greedy_stream_matches_the_materialized_forward() {
    let cfg = HuginnConfig::tiny();
    let w = init_weights(&cfg, 29);
    let state = state_vector(&cfg);
    let steps = cfg.mean_recurrence;
    let (max_tokens, block_size) = (64u32, 4u32);
    let mut eng = Engine::from_source(cfg.clone(), steps, &w, max_tokens, block_size);
    eng.set_state(&state);

    let prompt = vec![3u32, 5, 7, 11];
    let mut table = BlockTable::new();
    let hidden = eng.prefill(&mut table, &prompt);

    // The prefill path (chunk attention, m = n) must land on the same
    // distribution as the materialized last row. The two reduce attention in
    // a different order, so the gate is absolute.
    let want = model_last_row(&cfg, &w, steps, &state, &prompt);
    let got = eng.logits(&hidden);
    let abs = max_abs(&got, &want);
    assert!(abs < 1e-3, "prefill logits max_abs {abs}");
    let first = argmax(&got);
    assert_eq!(first, argmax(&want), "first token after prefill");

    // Two further greedy steps through the recorded decode graph.
    let mut seq = prompt.clone();
    let mut tok = first;
    seq.push(tok);
    for step in 0..2 {
        let mut refs: [&mut BlockTable; 1] = [&mut table];
        tok = eng.forward_batched_greedy(&mut refs, &[tok])[0];
        let want = model_last_row(&cfg, &w, steps, &state, &seq);
        assert_eq!(tok, argmax(&want), "decode step {step} at length {}", seq.len());
        seq.push(tok);
    }
}

#[test]
fn two_sequences_interleaved_serve_like_run_alone() {
    let cfg = HuginnConfig::tiny();
    let w = init_weights(&cfg, 37);
    let state = state_vector(&cfg);
    let steps = cfg.mean_recurrence;
    let prompts: [Vec<u32>; 2] = [vec![5, 7, 11], vec![3, 9, 13, 17, 19]];

    // Two sequences served alternately through one engine.
    let mut eng = Engine::from_source(cfg.clone(), steps, &w, 64, 4);
    eng.set_state(&state);
    let mut tables: Vec<BlockTable> = prompts.iter().map(|_| BlockTable::new()).collect();
    let mut streams: Vec<Vec<u32>> = Vec::new();
    for (table, prompt) in tables.iter_mut().zip(&prompts) {
        let hidden = eng.prefill(table, prompt);
        let mut s = prompt.clone();
        s.push(eng.admit_greedy(&hidden));
        streams.push(s);
    }
    for _ in 0..3 {
        let inputs: Vec<u32> = streams.iter().map(|s| *s.last().unwrap()).collect();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        for (i, tok) in eng.forward_batched_greedy(&mut refs, &inputs).into_iter().enumerate() {
            streams[i].push(tok);
        }
    }

    // The same prompt served alone on a fresh engine must stream identically:
    // the shared pool and the interleaving may not change a sequence's tokens.
    for (i, prompt) in prompts.iter().enumerate() {
        let mut alone = Engine::from_source(cfg.clone(), steps, &w, 64, 4);
        alone.set_state(&state);
        let mut table = BlockTable::new();
        let hidden = alone.prefill(&mut table, prompt);
        let mut s = prompt.clone();
        s.push(alone.admit_greedy(&hidden));
        for _ in 0..3 {
            let mut refs: [&mut BlockTable; 1] = [&mut table];
            let tok = alone.forward_batched_greedy(&mut refs, &[*s.last().unwrap()])[0];
            s.push(tok);
        }
        assert_eq!(streams[i], s, "sequence {i} differs from its solo run");
    }
}

#[test]
fn a_sequence_charges_one_page_per_stage() {
    let cfg = HuginnConfig::tiny();
    let w = init_weights(&cfg, 43);
    let (max_tokens, block_size) = (64u32, 4u32);
    let mut eng = Engine::from_source(cfg.clone(), cfg.mean_recurrence, &w, max_tokens, block_size);

    let prompt = vec![2u32, 4, 6, 8, 10];
    let mut table = BlockTable::new();
    let _ = eng.prefill(&mut table, &prompt);

    let stages = (cfg.n_prelude + cfg.mean_recurrence * cfg.n_recurrent + cfg.n_coda) as usize;
    let used = (prompt.len() as u32).div_ceil(block_size);
    let free = eng.stage_free_blocks();
    assert_eq!(free.len(), stages, "one allocator per stage");
    let want = max_tokens / block_size - used;
    for (stage, f) in free.iter().enumerate() {
        assert_eq!(*f, want, "stage {stage} free blocks");
    }
    assert_eq!(eng.free_blocks(), want);
    assert_eq!(eng.blocks_for(prompt.len() as u32), used);
}

/// The backend the test binary actually got, not the one the run intended.
#[test]
fn the_model_runs_on_the_backend_the_run_requested() {
    let want = std::env::var("BRAIN_BACKEND").unwrap_or_default().trim().to_ascii_lowercase();
    if want.is_empty() {
        return;
    }
    let cfg = HuginnConfig::tiny();
    let w = init_weights(&cfg, 17);
    let _eng = Engine::from_source(cfg, 1, &w, 8, 2);
    let got = gpu_core::backend_name();
    eprintln!("requested backend {want:?}, running on {got:?}");
    assert_eq!(got, want, "asked for {want}, ran on {got}");
}
