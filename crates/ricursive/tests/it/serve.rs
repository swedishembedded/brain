// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The paged hybrid serving engine against the full-recompute forward and the
//! shared serving conformance suite.
//!
//! Swedish Embedded AB implements solutions for serving long-context dense
//! decoders whose layers mix windowed and global attention at a fixed memory
//! budget for its clients. If your team needs expertise in hybrid KV layouts
//! and prefix reuse then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The tiny model has a sliding window of 4, so prompts and decodes of a dozen
//! tokens wrap the rings several times; the engine's blocks hold 4 rows, so
//! sequences cross block boundaries too. The reference is step 1's
//! `RicursiveModel::forward` over the whole prefix (causal attention makes a
//! row of the full forward the same as the forward of its prefix).
//!
//! Run `python -I tools/goldens/ricursive_dump_reference.py tiny ...` for the
//! fixture; without it the tests skip by name.

use std::path::Path;

use model::paged::BlockTable;
use model::serve::conformance::{self, Conformance};
use model::serve::{PagedDecoder, PrefillProgress};
use ricursive::{open_safetensors, RicursiveConfig, RicursiveEngine, RicursiveModel, ServeConfig};

use crate::taps::scaled_error;

const TINY: &str = "golden/ricursive/tiny";

/// Engine against reference, per logits row: scaled error bound. The engine
/// computes rope angles on the device where the reference reads a host table,
/// and attends through different kernels; the measured worst case is in the
/// roadmap ledger.
pub(crate) const TOLERANCE: f32 = 1e-5;

pub(crate) fn tiny_dir() -> Option<String> {
    let dir = brain_testutil::testdata(TINY);
    if Path::new(&dir).join("ricursive_tiny_weights.safetensors").exists() {
        Some(dir)
    } else {
        brain_testutil::skip(&format!("{dir} absent: run tools/goldens/ricursive_dump_reference.py tiny"));
        None
    }
}

pub(crate) fn tiny_config(dir: &str) -> RicursiveConfig {
    RicursiveConfig::from_file(&format!("{dir}/config.json")).expect("config")
}

pub(crate) fn weights(dir: &str, cfg: &RicursiveConfig) -> checkpoint::remap::RemapSource<'static> {
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    open_safetensors(&format!("{dir}/ricursive_tiny_weights.safetensors"), cfg, &layers).expect("open weights")
}

pub(crate) fn serve_config(prefill_chunk: u32, prefix_slots: u32) -> ServeConfig {
    ServeConfig { block_size: 4, num_blocks: 160, max_seq_len: 96, max_batch: 6, prefix_slots, prefill_chunk, sparse_attention: false }
}

pub(crate) fn engine(dir: &str, prefill_chunk: u32, prefix_slots: u32) -> RicursiveEngine {
    engine_with(dir, serve_config(prefill_chunk, prefix_slots))
}

pub(crate) fn engine_with(dir: &str, serve: ServeConfig) -> RicursiveEngine {
    let cfg = tiny_config(dir);
    RicursiveEngine::from_source(cfg.clone(), Box::new(weights(dir, &cfg)), serve, None).expect("engine")
}

/// A reproducible token stream inside the tiny vocabulary.
pub(crate) fn stream(seed: u32, n: usize, vocab: u32) -> Vec<u32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x % vocab
        })
        .collect()
}

/// The reference's logits at the last position of `ids`, by recomputing the whole prefix.
pub(crate) fn recompute_last(dir: &str, ids: &[u32]) -> Vec<f32> {
    let cfg = tiny_config(dir);
    let w = weights(dir, &cfg);
    let vocab = cfg.vocab_size;
    let out = RicursiveModel::new(cfg, &w, None).forward(ids).expect("forward");
    out.logits[(ids.len() - 1) * vocab..].to_vec()
}

pub(crate) fn check_row(label: &str, got: &[f32], want: &[f32], worst: &mut f32) {
    let err = scaled_error(label, got, want);
    assert!(err <= TOLERANCE, "{label}: scaled error {err:.3e} exceeds {TOLERANCE:.0e}");
    *worst = worst.max(err);
}

/// Prefill `prompt` through `prefill_chunk` slices of `budget` tokens; the last hidden row.
pub(crate) fn prefill_in_slices(e: &mut RicursiveEngine, table: &mut BlockTable, prompt: &[u32], budget: usize) -> Vec<f32> {
    loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk(table, prompt, budget) {
            return h;
        }
    }
}

#[test]
fn decode_across_window_wraps_equals_full_recompute_at_every_step() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    assert_eq!(cfg.sliding_window, 4);
    let ids = stream(7, 10 + 14, cfg.vocab_size as u32);
    let mut worst = 0.0f32;
    // (engine prefill chunk, prompt slice budget): chunks that straddle the window (4) and the block size (4).
    for (chunk, budget) in [(1u32, usize::MAX), (3, usize::MAX), (5, 2), (8, 7), (8, usize::MAX)] {
        let mut e = engine(&dir, chunk, 0);
        let mut table = BlockTable::new();
        let hidden = prefill_in_slices(&mut e, &mut table, &ids[..10], budget);
        check_row(&format!("chunk {chunk} budget {budget}: prefill"), &e.logits(&hidden), &recompute_last(&dir, &ids[..10]), &mut worst);
        for pos in 10..ids.len() {
            let logits = e.decode_logits(&mut [&mut table], &[ids[pos]]);
            check_row(&format!("chunk {chunk} budget {budget}: decode at {pos}"), &logits[0], &recompute_last(&dir, &ids[..=pos]), &mut worst);
        }
        e.release_table(&mut table);
        assert_eq!(e.free_blocks(), 160, "release returns every block");
    }
    eprintln!("decode equivalence: worst scaled error {worst:.3e} (tolerance {TOLERANCE:.0e})");
}

#[test]
fn concurrent_sequences_at_different_positions_decode_as_each_alone() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    // Three sequences whose lengths differ across a window wrap (4) and a block boundary (4).
    let seqs: Vec<Vec<u32>> = [(1u32, 6usize), (2, 9), (3, 13)].iter().map(|&(seed, n)| stream(seed, n + 8, vocab)).collect();
    let prompt_lens = [6usize, 9, 13];
    let mut worst = 0.0f32;

    let mut together = engine(&dir, 4, 0);
    let mut tables: Vec<BlockTable> = prompt_lens.iter().map(|_| BlockTable::new()).collect();
    for ((t, s), &n) in tables.iter_mut().zip(&seqs).zip(&prompt_lens) {
        prefill_in_slices(&mut together, t, &s[..n], usize::MAX);
    }
    let mut alone: Vec<(RicursiveEngine, BlockTable)> = Vec::new();
    for (s, &n) in seqs.iter().zip(&prompt_lens) {
        let mut e = engine(&dir, 4, 0);
        let mut t = BlockTable::new();
        prefill_in_slices(&mut e, &mut t, &s[..n], usize::MAX);
        alone.push((e, t));
    }
    for step in 0..8 {
        let inputs: Vec<u32> = seqs.iter().zip(&prompt_lens).map(|(s, &n)| s[n + step]).collect();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let batched = together.decode_logits(&mut refs, &inputs);
        for (i, ((e, t), logits)) in alone.iter_mut().zip(&batched).enumerate() {
            let solo = e.decode_logits(&mut [t], &[inputs[i]]);
            let mut w = 0.0;
            check_row(&format!("step {step} seq {i}: batched vs alone"), logits, &solo[0], &mut w);
            let n = prompt_lens[i] + step + 1;
            check_row(&format!("step {step} seq {i}: batched vs recompute"), logits, &recompute_last(&dir, &seqs[i][..n]), &mut worst);
            worst = worst.max(w);
        }
    }
    eprintln!("batching: worst scaled error {worst:.3e} (tolerance {TOLERANCE:.0e})");
}

#[test]
fn a_prefix_longer_than_the_window_resumes_to_the_cold_logits() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    let mut worst = 0.0f32;
    // 20 tokens end on a block boundary; 18 end inside a block (the shared tail is copied before it is written).
    for filed_len in [20usize, 18] {
        let ids = stream(11, filed_len + 6 + 4, vocab);
        let mut e = engine(&dir, 4, 2);
        let mut first = BlockTable::new();
        prefill_in_slices(&mut e, &mut first, &ids[..filed_len], usize::MAX);
        // The filing sequence keeps decoding (writing past the shared tail) while a second request adopts it.
        let mut stream_a = Vec::new();
        for &token in &ids[filed_len..filed_len + 2] {
            stream_a.push(e.decode_logits(&mut [&mut first], &[token]).remove(0));
        }

        let extended = &ids[..filed_len + 6];
        let mut second = BlockTable::new();
        let hidden = prefill_in_slices(&mut e, &mut second, extended, usize::MAX);
        assert_eq!(second.cached_rows() as usize, filed_len, "the second request resumes from the filed prompt");
        assert!(e.prefix_stats().0 >= filed_len as u64, "the hit is counted");
        check_row(&format!("filed {filed_len}: warm prefill"), &e.logits(&hidden), &recompute_last(&dir, extended), &mut worst);
        let warm = e.decode_logits(&mut [&mut second], &[ids[filed_len + 6]]).remove(0);
        check_row(&format!("filed {filed_len}: warm decode"), &warm, &recompute_last(&dir, &ids[..filed_len + 7]), &mut worst);
        // The first sequence's rows were not disturbed by the adoption.
        let more = e.decode_logits(&mut [&mut first], &[ids[filed_len + 2]]).remove(0);
        check_row(&format!("filed {filed_len}: filing sequence after adoption"), &more, &recompute_last(&dir, &ids[..filed_len + 3]), &mut worst);
        for (i, l) in stream_a.iter().enumerate() {
            check_row(&format!("filed {filed_len}: filing sequence decode {i}"), l, &recompute_last(&dir, &ids[..filed_len + 1 + i]), &mut worst);
        }

        // An exact repeat needs no compute and answers with the filed hidden row.
        let mut third = BlockTable::new();
        // (The second request's longer prompt superseded the first's entry, as a conversation's later turn does.)
        let again = prefill_in_slices(&mut e, &mut third, extended, usize::MAX);
        assert_eq!(third.cached_rows() as usize, extended.len());
        check_row(&format!("filed {filed_len}: exact repeat"), &e.logits(&again), &recompute_last(&dir, extended), &mut worst);
        let repeat = e.decode_logits(&mut [&mut third], &[ids[filed_len + 6]]).remove(0);
        check_row(&format!("filed {filed_len}: decode after an exact repeat"), &repeat, &recompute_last(&dir, &ids[..filed_len + 7]), &mut worst);
        let free = e.free_blocks();
        for t in [&mut first, &mut second, &mut third] {
            e.release_table(t);
        }
        assert!(e.free_blocks() > free);
        let cached = e.prefix_stats().2;
        assert!(cached >= 1, "the prompts stay filed after their sequences end");
        e.reclaim_prefix(u32::MAX);
        assert_eq!(e.free_blocks(), 160, "reclaiming the cache returns every block");
    }
    eprintln!("prefix reuse: worst scaled error {worst:.3e} (tolerance {TOLERANCE:.0e})");
}

fn conformance_engine() -> RicursiveEngine {
    let dir = brain_testutil::testdata(TINY);
    engine(&dir, 8, 4)
}

#[test]
fn the_shared_serving_conformance_suite_passes_on_the_tiny_model() {
    if tiny_dir().is_none() {
        return;
    }
    conformance::run_all(&Conformance { make: conformance_engine, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None });
}

#[test]
fn the_memory_plan_charges_global_kv_per_token_and_rings_per_slot() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let serve = serve_config(8, 2);
    let e = engine(&dir, 8, 2);
    let (global, rings) = e.kv_bytes();
    // 1 global layer of 3, K and V, kv_dim 32 f32 per row.
    assert_eq!(ricursive::global_kv_bytes_per_token(&cfg), 2 * 32 * 4);
    assert_eq!(global, ricursive::global_kv_bytes_per_token(&cfg) * u64::from(serve.num_blocks * serve.block_size));
    // 2 sliding layers, window 4 plus the chunk's 8 slack rows, live and snapshot slots.
    assert_eq!(ricursive::ring_bytes_per_slot(&cfg, 8), 2 * 2 * (4 + 8) * 32 * 4);
    assert_eq!(rings, ricursive::ring_bytes_per_slot(&cfg, 8) * u64::from(serve.max_batch + serve.prefix_slots));
    assert_eq!(e.kv_pool_bytes(), global + rings);
}
