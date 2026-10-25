// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving-equivalence gates for recurrent-depth
// language models for its clients. If your team needs expertise in proving a
// paged, continuous-batching engine reproduces the model's own forward past
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving engine must produce exactly the tokens the model's own
//! full-sequence forward produces, for every sequence in a batch, with the
//! paged pool shared and each recurrent pass holding its own block table.

use ouro::config::OuroConfig;
use ouro::init::init_weights;
use ouro::model::Ouro;
use ouro::serve::Engine;

use ::model::paged::BlockTable;
use ::model::serve::PagedDecoder;

fn reference_next(cfg: &OuroConfig, w: &std::collections::HashMap<String, Vec<f32>>, seq: &[u32]) -> u32 {
    let m = Ouro::new_inference(cfg.clone(), 1, seq.len() as u32, w);
    m.set_tokens(seq);
    m.trunk_forward();
    let last = cfg.total_ut_steps as usize - 1;
    m.head_forward(last);
    let logits = m.read_logits();
    let v = cfg.vocab as usize;
    let row = &logits[(seq.len() - 1) * v..seq.len() * v];
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, x) in row.iter().enumerate() {
        if *x > best.1 {
            best = (i, *x);
        }
    }
    best.0 as u32
}

#[test]
fn paged_engine_reproduces_the_models_own_greedy_streams() {
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 33);
    let (passes, block_size, max_tokens) = (cfg.total_ut_steps, 4u32, 64u32);
    let mut eng = Engine::from_source(cfg.clone(), &w, max_tokens, block_size);

    // Two sequences at different prompt lengths, interleaved step by step.
    let prompts: [Vec<u32>; 2] = [vec![5, 7, 11], vec![3, 9, 13, 17, 19]];
    let mut tables: Vec<BlockTable> = prompts.iter().map(|_| BlockTable::new()).collect();
    let mut streams: Vec<Vec<u32>> = Vec::new();
    for (table, prompt) in tables.iter_mut().zip(&prompts) {
        let hidden = eng.prefill(table, prompt);
        let first = eng.admit_greedy(&hidden);
        let want_first = reference_next(&cfg, &w, prompt);
        assert_eq!(first, want_first, "first token after prefill");
        let mut s = prompt.clone();
        s.push(first);
        streams.push(s);
    }
    // The pool charges one block per pass per page of context.
    let used: u32 = tables.iter().map(|t| t.len()).sum();
    let expected_blocks: u32 = prompts.iter().map(|p| eng.blocks_for(p.len() as u32)).sum();
    assert_eq!(eng.free_blocks(), max_tokens / block_size - expected_blocks, "used {used}");

    for _ in 0..4 {
        let inputs: Vec<u32> = streams.iter().map(|s| *s.last().unwrap()).collect();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let next = eng.forward_batched_greedy(&mut refs, &inputs);
        for (i, tok) in next.into_iter().enumerate() {
            let want = reference_next(&cfg, &w, &streams[i]);
            assert_eq!(tok, want, "sequence {i} at length {}: engine {tok} vs model {want}", streams[i].len());
            streams[i].push(tok);
        }
    }
    // Every pass's table tracks its own sequence's length exactly.
    for (table, s) in tables.iter().zip(&streams) {
        assert_eq!(table.len(), s.len() as u32 - 1);
    }
    let _ = passes;
}

/// A prompt long enough to select the register-tiled GEMM in prefill (128+
/// rows): the same engine, the same greedy token as the model's own forward.
#[test]
fn a_long_prompt_prefill_selects_the_tiled_linear_and_still_matches() {
    let mut cfg = OuroConfig::tiny();
    cfg.block_size = 256;
    let w = init_weights(&cfg, 41);
    let mut eng = Engine::from_source(cfg.clone(), &w, 256, 4);
    let prompt: Vec<u32> = (0..160u32).map(|i| (i * 7 + 3) % 29).collect();
    let mut table = BlockTable::new();
    let hidden = eng.prefill(&mut table, &prompt);
    let first = eng.admit_greedy(&hidden);
    let mut seq = prompt.clone();
    seq.push(first);
    let mut refs = [&mut table];
    let next = eng.forward_batched_greedy(&mut refs, &[first])[0];
    assert_eq!(next, reference_next(&cfg, &w, &seq));
}
