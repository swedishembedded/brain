// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Speculation with the DFlash2 draft model on the tiny Ripple: the retained hidden-state taps, and the proof that
//! drafting never changes what greedy decoding emits.
//!
//! Swedish Embedded AB implements speculative decoding - drafters that read a target's internals, verifiers and the
//! state rollback a hybrid stack needs - for clients serving large models on modest hardware. If your team needs
//! expertise in LLM serving throughput then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The target is the tiny hybrid model of the other tests (sliding window 4, one global layer, 4-row blocks) and the draft
//! a tiny random DFlash2 (`tools/goldens/dflash2_reference_forward.py tiny --out testdata/golden/dflash2/tiny`): hidden 64, three layers reading the
//! target's three layers, a window of 6, blocks of five rows. Both windows are shorter than the sequences, so the
//! target's rings and the draft's wrap many times.

use std::path::Path;

use model::paged::BlockTable;
use model::serve::{PagedDecoder, Request, Scheduler, SpecPolicy};
use ricursive::{DraftSizing, RicursiveEngine, RicursiveModel, ServeConfig};

use crate::serve::{engine_with, serve_config, stream, tiny_config, tiny_dir, weights};
use crate::taps::scaled_error;

const DRAFT: &str = "golden/dflash2/tiny/draft";

/// A tap row is the residual rounded to bf16; against the reference's f32 it can differ by half a bf16 step.
const TAP_TOLERANCE: f32 = 5e-3;

fn draft_dir() -> Option<String> {
    let dir = brain_testutil::testdata(DRAFT);
    if Path::new(&dir).join("model.safetensors").exists() {
        Some(dir)
    } else {
        brain_testutil::skip(&format!("{dir} absent: run tools/goldens/dflash2_reference_forward.py tiny --out testdata/golden/dflash2/tiny"));
        None
    }
}

fn sizing(draft: &str) -> DraftSizing {
    let cfg = dflash2::Dflash2Config::from_json(&std::fs::read_to_string(format!("{draft}/config.json")).unwrap()).unwrap();
    let weight_bytes = dflash2::HfDraft::open(Path::new(draft)).unwrap().device_weight_bytes(&cfg);
    DraftSizing { cfg, weight_bytes }
}

fn drafted_engine(dir: &str, draft: &str, serve: ServeConfig) -> RicursiveEngine {
    let mut e = engine_with(dir, ServeConfig { drafter: Some(sizing(draft)), ..serve });
    e.set_drafter(Path::new(draft)).expect("set_drafter");
    e
}

/// The step-1 forward's residual stream after each layer for `ids`: `[layers][t, hidden]`.
fn reference_hidden(dir: &str, ids: &[u32]) -> Vec<Vec<f32>> {
    let cfg = tiny_config(dir);
    let w = weights(dir, &cfg);
    RicursiveModel::new(cfg, &w, None).forward(ids).expect("forward").layer_outputs
}

/// The retained rows of every position of `table` equal the reference forward's hidden state at the tapped layers.
fn assert_taps(e: &RicursiveEngine, dir: &str, table: &BlockTable, ids: &[u32], label: &str) {
    let n = table.len() as usize;
    let hidden = tiny_config(dir).hidden_size;
    let layers = e.tap_layers().to_vec();
    let reference = reference_hidden(dir, &ids[..n]);
    let got = e.read_taps(table, 0, n as u32).expect("taps");
    assert_eq!(got.len(), n * layers.len() * hidden);
    for (j, &l) in layers.iter().enumerate() {
        let mine: Vec<f32> = got.chunks(layers.len() * hidden).flat_map(|row| row[j * hidden..(j + 1) * hidden].iter().copied()).collect();
        let err = scaled_error(&format!("{label}: tap of layer {l}"), &mine, &reference[l]);
        assert!(err <= TAP_TOLERANCE, "{label}: the retained rows of layer {l} differ from the forward's by {err:.3e}");
    }
}

#[test]
fn retained_taps_equal_the_forward_hidden_states_through_prefill_decode_verify_and_prefix_reuse() {
    let (Some(dir), Some(draft)) = (tiny_dir(), draft_dir()) else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let ids = stream(3, 60, vocab);
    let mut e = drafted_engine(&dir, &draft, serve_config(5, 2));
    assert_eq!(e.tap_layers(), &[0, 1, 2], "the draft reads the three layers of the tiny target");

    // Chunked prefill (chunks of 5, slices of 2), then decode steps.
    let mut table = BlockTable::new();
    crate::serve::prefill_in_slices(&mut e, &mut table, &ids[..13], 2);
    assert_taps(&e, &dir, &table, &ids, "after prefill");
    for &t in &ids[13..22] {
        e.decode_logits(&mut [&mut table], &[t]);
    }
    assert_taps(&e, &dir, &table, &ids, "after decode");

    // A verify of four rows of which the scheduler keeps two: the rows it truncates away leave no trace, and the next
    // tokens' own rows replace them.
    let before = table.len();
    e.forward_batched_verify(&mut [&mut table], &[vec![ids[22], ids[23], 99, 98]], 1);
    e.commit_verified(&mut [&mut table], &[2]);
    let mut alloc_free = 0;
    table.truncate(before + 2, e.alloc_mut());
    alloc_free += e.free_blocks();
    assert!(alloc_free > 0);
    assert_taps(&e, &dir, &table, &ids, "after a rejected verify");
    for &t in &ids[24..30] {
        e.decode_logits(&mut [&mut table], &[t]);
    }
    assert_taps(&e, &dir, &table, &ids, "after the rejected rows were replaced");
    e.release_table(&mut table);

    // Prefix reuse: a prompt filed at 22 tokens (five blocks and a partly filled tail), resumed by a longer one. The
    // adopted blocks carry their rows, and the copy of the tail block carries its two.
    let mut first = BlockTable::new();
    crate::serve::prefill_in_slices(&mut e, &mut first, &ids[..22], usize::MAX);
    let (hits_before, ..) = e.prefix_stats();
    let mut second = BlockTable::new();
    crate::serve::prefill_in_slices(&mut e, &mut second, &ids[..31], usize::MAX);
    assert!(e.prefix_stats().0 > hits_before, "the second prompt resumed from the first");
    assert_taps(&e, &dir, &second, &ids, "resumed from a cached prefix");
    assert_taps(&e, &dir, &first, &ids, "the sequence the prefix came from");
}

#[test]
fn taps_are_refused_for_layers_the_model_does_not_have() {
    let Some(dir) = tiny_dir() else { return };
    let mut e = engine_with(&dir, serve_config(5, 0));
    assert!(e.tap_layers().is_empty() && e.read_taps(&BlockTable::new(), 0, 0).is_err(), "no taps until enabled");
    assert!(e.enable_taps(&[3]).unwrap_err().contains("distinct layers"));
    assert!(e.enable_taps(&[1, 1]).is_err() && e.enable_taps(&[]).is_err());
    e.enable_taps(&[2, 0]).unwrap();
    e.enable_taps(&[2, 0]).unwrap();
    assert!(e.enable_taps(&[0, 1]).unwrap_err().contains("already retained"));
}

fn greedy(prompt: &[u32], max_new: usize) -> Request {
    Request { prompt: prompt.to_vec(), max_new, stop: Vec::new() }
}

/// Plain greedy output of each prompt, one by one on fresh engines with speculation off.
fn plain_outputs(dir: &str, serve: &ServeConfig, prompts: &[Vec<u32>], max_new: usize) -> Vec<Vec<u32>> {
    prompts
        .iter()
        .map(|p| {
            let mut sched = Scheduler::new(engine_with(dir, serve.clone()), 4);
            sched.set_speculation(SpecPolicy { enabled: false, ..SpecPolicy::default() });
            let id = sched.submit(greedy(p, max_new));
            sched.run().remove(&id).expect("completed")
        })
        .collect()
}

/// Speculation with the draft model equals plain greedy exactly, whatever it proposes: several sequences of different
/// lengths decode together over many steps past both windows, so proposals are made, verified, partly accepted and
/// rolled back while the target's rings and the draft's wrap.
#[test]
fn speculation_with_the_draft_model_equals_plain_greedy() {
    let (Some(dir), Some(draft)) = (tiny_dir(), draft_dir()) else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let prompts: Vec<Vec<u32>> = [(1u32, 9usize), (2, 15), (3, 30), (4, 6)].iter().map(|&(seed, n)| stream(seed, n, vocab)).collect();
    let max_new = 40;
    let serve = serve_config(8, 2);
    let want = plain_outputs(&dir, &serve, &prompts, max_new);

    let mut sched = Scheduler::new(drafted_engine(&dir, &draft, serve), 4);
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(greedy(p, max_new))).collect();
    let done = sched.run();
    for (i, (id, plain)) in ids.iter().zip(&want).enumerate() {
        assert_eq!(&done[id], plain, "prompt {i}: speculative output differs from plain greedy");
    }
    let stats = sched.spec_stats();
    let drafted = sched.decoder_mut().draft_stats().expect("a draft model is attached");
    eprintln!("speculation with the draft model: {stats:?}; draft {drafted:?}");
    assert!(stats.verify_steps > 0 && stats.proposed > 0, "the draft model never proposed anything: {stats:?}");
    assert!(drafted.rounds > 0 && drafted.fed_rows > 0, "the draft did not run: {drafted:?}");
}

/// A prompt that resumes from a cached prefix, under speculation, answers as it does cold and plain: the drafter's own
/// state (rebuilt from the taps of the adopted blocks) agrees with the target's.
#[test]
fn speculation_after_a_prefix_hit_equals_plain_greedy() {
    let (Some(dir), Some(draft)) = (tiny_dir(), draft_dir()) else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let first = stream(8, 22, vocab);
    let mut second = first.clone();
    second.extend(stream(9, 9, vocab));
    let max_new = 36;
    let serve = serve_config(8, 2);
    let want = plain_outputs(&dir, &serve, &[first.clone(), second.clone()], max_new);

    let mut sched = Scheduler::new(drafted_engine(&dir, &draft, serve), 4);
    let a = sched.submit(greedy(&first, max_new));
    let out_a = sched.run().remove(&a).unwrap();
    let hits = sched.prefix_stats().0;
    let b = sched.submit(greedy(&second, max_new));
    let out_b = sched.run().remove(&b).unwrap();
    assert!(sched.prefix_stats().0 > hits, "the second prompt resumed from the first");
    assert_eq!((out_a, out_b), (want[0].clone(), want[1].clone()));
    assert!(sched.spec_stats().verify_steps > 0);
}

/// The shared conformance suite, with the engine carrying the draft model (taps retained for every row it runs): batching,
/// prefix reuse and, under a drafter that is right, partly right and wrong, speculation with accepts and rollbacks.
#[test]
fn the_conformance_suite_passes_with_the_draft_model_attached() {
    use model::serve::conformance::{self, Conformance};
    if tiny_dir().is_none() || draft_dir().is_none() {
        return;
    }
    fn make() -> RicursiveEngine {
        let dir = brain_testutil::testdata("golden/ricursive/tiny");
        drafted_engine(&dir, &brain_testutil::testdata(DRAFT), serve_config(8, 4))
    }
    conformance::run_all(&Conformance { make, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None });
}

#[test]
fn a_draft_model_for_another_target_is_refused_by_name() {
    let (Some(dir), Some(draft)) = (tiny_dir(), draft_dir()) else { return };
    let e = engine_with(&dir, serve_config(5, 0));
    let right = sizing(&draft).cfg;
    e.check_drafter(&right).expect("the tiny draft fits the tiny target");
    let past = dflash2::Dflash2Config { target_layers: vec![0, 1, 7], ..right.clone() };
    assert!(e.check_drafter(&past).unwrap_err().contains("reads layer 7"));
    let wide = dflash2::Dflash2Config { d_model: 128, ..right.clone() };
    assert!(e.check_drafter(&wide).unwrap_err().contains("width 128"));
    let big_vocab = dflash2::Dflash2Config { vocab: 500, ..right };
    assert!(e.check_drafter(&big_vocab).unwrap_err().contains("vocabulary 500"));
}

/// Beside latent sparse attention the scheduler keeps every verify inside one cluster block of the index, so each row reads
/// the index a plain decode step at its position reads: the output equals plain greedy decoding with the same indexer.
/// Needs a device with workgroup barriers (the software adapter, or a card); skips by name on the CPU backend.
#[test]
fn speculation_beside_sparse_attention_equals_plain_greedy_with_it() {
    let (Some(dir), Some(draft)) = (tiny_dir(), draft_dir()) else { return };
    if !crate::lsa::lsa_runs_here() {
        return;
    }
    let cfg = tiny_config(&dir);
    let vocab = cfg.vocab_size as u32;
    // A small budget (the index drops keys), cluster blocks of 8 keys: sequences of 9 to 70 tokens close several.
    let ix = crate::lsa::indexer(&cfg, crate::lsa::lsa_config(3, 6), 29);
    let serve = ServeConfig { sparse_attention: true, ..serve_config(8, 2) };
    let sparse = |serve: ServeConfig| {
        let mut e = engine_with(&dir, serve);
        e.set_lsa(&ix).expect("set_lsa");
        e.set_lsa_dense_until(0);
        e
    };
    let prompts: Vec<Vec<u32>> = [(21u32, 9usize), (22, 15), (23, 30)].iter().map(|&(seed, n)| stream(seed, n, vocab)).collect();
    let max_new = 32;
    let want: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let mut sched = Scheduler::new(sparse(serve.clone()), 4);
            sched.set_speculation(SpecPolicy { enabled: false, ..SpecPolicy::default() });
            let id = sched.submit(greedy(p, max_new));
            sched.run().remove(&id).expect("completed")
        })
        .collect();

    let mut e = sparse(ServeConfig { drafter: Some(sizing(&draft)), ..serve });
    e.set_drafter(Path::new(&draft)).expect("set_drafter");
    let mut sched = Scheduler::new(e, 4);
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(greedy(p, max_new))).collect();
    let done = sched.run();
    for (i, (id, plain)) in ids.iter().zip(&want).enumerate() {
        assert_eq!(&done[id], plain, "prompt {i}: speculation beside sparse attention differs from plain greedy with it");
    }
    let stats = sched.spec_stats();
    assert!(stats.verify_steps > 0 && stats.proposed > 0, "nothing was verified beside the index: {stats:?}");
    let reads = sched.decoder_mut().lsa_stats().expect("sparse attention is on");
    assert!(reads.reads.selected_keys < reads.reads.causal_keys, "the global layer read sparsely: {reads:?}");
    eprintln!("speculation beside sparse attention: {stats:?}");
}

/// The same property under the conformance suite's scripted drafters (right, partly right, wrong): accepted proposals,
/// rejections and rollbacks beside the index, with the draft model's taps retained for every row.
#[test]
fn scripted_speculation_beside_sparse_attention_equals_plain_greedy() {
    use model::serve::conformance::{self, Conformance};
    if tiny_dir().is_none() || draft_dir().is_none() || !crate::lsa::lsa_runs_here() {
        return;
    }
    fn make() -> RicursiveEngine {
        let dir = brain_testutil::testdata("golden/ricursive/tiny");
        let ix = crate::lsa::indexer(&tiny_config(&dir), crate::lsa::lsa_config(3, 6), 31);
        let mut e = drafted_engine(&dir, &brain_testutil::testdata(DRAFT), ServeConfig { sparse_attention: true, ..serve_config(8, 4) });
        e.set_lsa(&ix).expect("set_lsa");
        e.set_lsa_dense_until(0);
        e
    }
    let c = Conformance { make, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None };
    conformance::speculation_equals_plain_greedy(&c);
    conformance::batched_output_equals_single_request_output(&c);
    conformance::a_prefix_hit_prefills_only_the_suffix(&c);
}
