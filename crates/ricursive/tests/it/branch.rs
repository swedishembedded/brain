// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prefix reuse at branch points: prompts that share a document or a system prompt and then diverge, and a conversation
//! re-sent turn after turn.
//!
//! Swedish Embedded AB implements solutions for agent workloads that resend long conversation histories to large
//! language models for its clients. If your team needs expertise in prefix reuse for hybrid sliding/global models then
//! you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The tiny model's window is 4 and its KV blocks hold 4 rows, so a 24-token shared prefix is six times the window and
//! six whole blocks. A prompt that leaves a cached one files the ring and the shared blocks at the divergence (rounded
//! down to a block) as a prefix of its own; the next prompt that shares it resumes there.

use model::paged::BlockTable;
use model::serve::PagedDecoder;
use ricursive::{RicursiveEngine, ServeConfig};

use crate::dflash::{assert_taps, draft_dir, drafted_engine};
use crate::serve::{check_row, engine, engine_with, prefill_in_slices, recompute_last, serve_config, stream, tiny_config, tiny_dir};

const SHARED: usize = 24;

/// `shared` tokens common to three prompts, each continued by its own `tail`.
fn prompts(vocab: u32, tail: usize) -> Vec<Vec<u32>> {
    let shared = stream(41, SHARED, vocab);
    (0..3u32)
        .map(|i| {
            let mut p = shared.clone();
            p.extend(stream(50 + i, tail, vocab));
            p
        })
        .collect()
}

/// Prefill the three prompts one after another on `e`: the first files itself, the second leaves it at the branch point,
/// the third resumes there. Returns each prefill's hidden row and the tokens the cache counted as reused by it.
fn serve_three(e: &mut RicursiveEngine, ps: &[Vec<u32>]) -> (Vec<Vec<f32>>, Vec<u64>, Vec<BlockTable>) {
    let (mut hidden, mut reused, mut tables) = (Vec::new(), Vec::new(), Vec::new());
    for p in ps {
        let before = e.prefix_stats().0;
        let mut t = BlockTable::new();
        hidden.push(prefill_in_slices(e, &mut t, p, usize::MAX));
        reused.push(e.prefix_stats().0 - before);
        tables.push(t);
    }
    (hidden, reused, tables)
}

#[test]
fn prompts_that_share_a_prefix_longer_than_the_window_reuse_it_up_to_the_branch_point() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let ps = prompts(vocab, 6);
    let mut worst = 0.0f32;
    for chunk in [8u32, 5] {
        let mut e = engine(&dir, chunk, 4);
        let (hidden, reused, mut tables) = serve_three(&mut e, &ps);
        assert_eq!(reused, [0, 0, SHARED as u64], "chunk {chunk}: only the third prompt resumes, from the branch point the second filed");
        assert_eq!(e.prefix_stats().2, 4, "the three prompts and the branch point are cached");
        for (i, (h, p)) in hidden.iter().zip(&ps).enumerate() {
            check_row(&format!("chunk {chunk}: prompt {i}"), &e.logits(h), &recompute_last(&dir, p), &mut worst);
        }
        // Decoding on from the resumed sequence is the cold sequence's.
        let mut ids = ps[2].clone();
        for step in 0..8 {
            let next = stream(70 + step, 1, vocab)[0];
            let l = e.decode_logits(&mut [&mut tables[2]], &[next]);
            ids.push(next);
            check_row(&format!("chunk {chunk}: decode {step}"), &l[0], &recompute_last(&dir, &ids), &mut worst);
        }
        for mut t in tables {
            e.release_table(&mut t);
        }
    }
    eprintln!("branch points: worst scaled error {worst:.3e}");
}

#[test]
fn a_hint_files_a_shared_prefix_before_any_prompt_has_diverged() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let ps = prompts(vocab, 6);
    let mut e = engine(&dir, 8, 4);
    let mut first = BlockTable::new();
    assert!(e.hint_shared_prefix(&mut first, &ps[0], SHARED + 2), "the hint is taken: rounded down to a block it is {SHARED} tokens");
    let h = prefill_in_slices(&mut e, &mut first, &ps[0], usize::MAX);
    assert_eq!(e.prefix_stats().2, 2, "the branch point and the prompt");
    let mut second = BlockTable::new();
    let hidden = prefill_in_slices(&mut e, &mut second, &ps[1], usize::MAX);
    assert_eq!(e.prefix_stats().0, SHARED as u64, "the second prompt resumed from the hinted prefix");
    let mut worst = 0.0f32;
    check_row("hinted first", &e.logits(&h), &recompute_last(&dir, &ps[0]), &mut worst);
    check_row("resumed second", &e.logits(&hidden), &recompute_last(&dir, &ps[1]), &mut worst);
    let mut late = BlockTable::new();
    assert!(!e.hint_shared_prefix(&mut late, &ps[0][..8], 8), "a prefix shorter than a cacheable prompt is declined");
}

/// A conversation re-sent each turn: prompt N+1 is prompt N, its reply and a new turn. The cache holds prompt N as it was
/// filed, so the next turn resumes from it and prefills the reply and the new turn; the reply itself is not cached (the
/// scheduler hands a decoder only the table when a sequence ends, not the tokens it generated).
#[test]
fn a_resent_history_resumes_from_the_previous_turn_and_equals_a_cold_prefill() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let mut e = engine(&dir, 8, 3);
    let mut history = stream(61, 28, vocab);
    let mut worst = 0.0f32;
    for turn in 0..3u32 {
        let before = e.prefix_stats().0;
        let mut t = BlockTable::new();
        let h = prefill_in_slices(&mut e, &mut t, &history, usize::MAX);
        let reused = e.prefix_stats().0 - before;
        check_row(&format!("turn {turn}"), &e.logits(&h), &recompute_last(&dir, &history), &mut worst);
        if turn > 0 {
            assert_eq!(reused as usize, history.len() - 6 - 5, "turn {turn} resumes from the previous prompt and prefills its reply and the new turn");
        }
        // The reply: six decoded tokens (greedy), then the next turn's five new tokens.
        let mut reply = Vec::new();
        let mut next = history[history.len() - 1];
        for _ in 0..6 {
            next = e.forward_batched_greedy(&mut [&mut t], &[next])[0];
            reply.push(next);
        }
        e.release_table(&mut t);
        history.extend(&reply);
        history.extend(stream(80 + turn, 5, vocab));
    }
    eprintln!("re-sent history: worst scaled error {worst:.3e}");
}

/// The cache is bounded by its snapshot slots whatever arrives, and gives every block back under pressure.
#[test]
fn branch_snapshots_stay_within_the_slots_and_are_evicted_under_pressure() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let serve = serve_config(8, 2);
    let total = serve.num_blocks;
    let mut e = engine_with(&dir, serve);
    // Six prompts, pairs sharing a document: every divergence wants a branch snapshot, only two slots exist.
    let mut tables = Vec::new();
    for i in 0..6u32 {
        let mut p = stream(100 + i / 2, SHARED, vocab);
        p.extend(stream(110 + i, 6, vocab));
        let mut t = BlockTable::new();
        prefill_in_slices(&mut e, &mut t, &p, usize::MAX);
        e.release_table(&mut t);
        tables.push(t);
        assert!(e.prefix_stats().2 <= 2, "{} entries after prompt {i}, the engine has two snapshot slots", e.prefix_stats().2);
    }
    assert!(e.free_blocks() < total, "the cached prefixes hold blocks");
    let freed = e.reclaim_prefix(u32::MAX);
    assert!(freed > 0 && e.free_blocks() == total, "reclaiming under pressure returns every block ({} of {total})", e.free_blocks());
    assert_eq!(e.prefix_stats().2, 0);
}

/// The same reuse with the draft model attached: the taps of the shared blocks, and of the copied ones, are the forward's.
#[test]
fn branch_points_carry_the_retained_taps() {
    let (Some(dir), Some(draft)) = (tiny_dir(), draft_dir()) else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    // A tail that is not a whole block, so the third prompt's resume copies nothing and the first's filed tail is copied.
    let ps = prompts(vocab, 6);
    let mut e = drafted_engine(&dir, &draft, serve_config(8, 4));
    let (_, reused, tables) = serve_three(&mut e, &ps);
    assert_eq!(reused, [0, 0, SHARED as u64]);
    for (t, p) in tables.iter().zip(&ps) {
        assert_taps(&e, &dir, t, p, "branch point");
    }
}

/// With latent sparse attention (software adapter): a prompt resumed from a branch point reads the index as the same
/// prompt cold, whose pages the branch entry shares with the blocks it holds.
#[test]
fn branch_points_with_sparse_attention_equal_a_cold_run() {
    let Some(dir) = tiny_dir() else { return };
    if !crate::lsa::lsa_runs_here() {
        return;
    }
    let cfg = tiny_config(&dir);
    let ps = prompts(cfg.vocab_size as u32, 6);
    let ix = crate::lsa::indexer(&cfg, crate::lsa::lsa_config(3, 6), 29);
    let sparse = |slots: u32| {
        let mut e = engine_with(&dir, ServeConfig { sparse_attention: true, ..serve_config(8, slots) });
        e.set_lsa(&ix).expect("set_lsa");
        e.set_lsa_dense_until(0);
        e
    };
    let mut warm = sparse(4);
    let (hidden, reused, tables) = serve_three(&mut warm, &ps);
    assert_eq!(reused, [0, 0, SHARED as u64]);
    let mut worst = 0.0f32;
    for (i, p) in ps.iter().enumerate() {
        let mut cold = sparse(0);
        let mut t = BlockTable::new();
        let c = prefill_in_slices(&mut cold, &mut t, p, usize::MAX);
        check_row(&format!("sparse prompt {i}"), &warm.logits(&hidden[i]), &cold.logits(&c), &mut worst);
    }
    drop(tables);
}

#[test]
fn the_conformance_suite_passes_with_branch_points() {
    use model::serve::conformance::{self, Conformance};
    if tiny_dir().is_none() {
        return;
    }
    fn make() -> RicursiveEngine {
        engine(&brain_testutil::testdata("golden/ricursive/tiny"), 8, 4)
    }
    conformance::run_all(&Conformance { make, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None });
}
