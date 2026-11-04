// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latent sparse attention in the Qwen3 serving engine on the real Qwen3-0.6B
//! checkpoint (skipped when the model store does not hold it).
//!
//! Swedish Embedded AB implements long-context serving whose attention reads
//! a bounded set of keys per token for its clients. If your team needs
//! expertise in sparse attention for production LLM serving, you can procure
//! our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use model::lsa::indexer::{IndexGeometry, Indexer};
use model::lsa::LsaConfig;
use qwen3::serve::Engine;

const BLOCK: u32 = 16;
const MAX_PREFILL: u32 = 512;

fn checkpoint() -> Option<String> {
    let dir = brain_testutil::model_dir("Qwen/Qwen3-0.6B").unwrap_or_default();
    // The store keeps either the converted brain file or the upstream directory.
    let brain_file = std::path::Path::new(&dir).join("model.brain.safetensors");
    if brain_file.exists() {
        return Some(brain_file.to_string_lossy().into_owned());
    }
    if !std::path::Path::new(&dir).join("config.json").exists() {
        brain_testutil::skip(&format!("Qwen/Qwen3-0.6B is not in the model store ({dir})"));
        return None;
    }
    Some(dir)
}

fn engine(path: &str, max_tokens: u32) -> Engine {
    let (cfg, src) = qwen3::open_checkpoint(path).expect("open Qwen3-0.6B");
    let mbt = max_tokens.div_ceil(BLOCK);
    Engine::from_source_tier(cfg, &*src, BLOCK, 2 * mbt + 8, 2, mbt, MAX_PREFILL, false, qwen3::Dtype::F32)
}

fn geometry(path: &str) -> IndexGeometry {
    let (c, _) = qwen3::open_checkpoint(path).expect("open Qwen3-0.6B");
    IndexGeometry { n_layers: c.n_layers as usize, n_heads: c.n_heads as usize, n_kv_heads: c.n_kv_heads as usize, head_dim: c.head_dim as usize, rope_base: c.rope_theta }
}

/// A prompt of ordinary tokens (ids below 20k are mostly word pieces).
fn prompt(rng: &mut Lcg, n: usize) -> Vec<u32> {
    (0..n).map(|_| 200 + rng.next_u32() % 20_000).collect()
}

/// With a budget past the context every bucket is picked and every key
/// expanded, so the sparse engine reads exactly the dense engine's keys: its
/// greedy continuation must be the dense one, through prefill (three closed
/// cluster blocks, chunks cut at block boundaries) and decode (index pages
/// built as blocks close mid-generation).
#[test]
fn a_budget_past_the_context_generates_what_dense_attention_does() {
    let Some(path) = checkpoint() else { return };
    let mut rng = Lcg::random();
    let p = prompt(&mut rng, 3 * 512 + 500);
    let new = 24;
    let dense = engine(&path, 4096).generate_greedy(std::slice::from_ref(&p), new, None);

    let cfg = LsaConfig { top_p: 1024, top_k: 4096, ..LsaConfig::default() };
    let indexer = Indexer::random(cfg, geometry(&path), rng.next_u64()).unwrap();
    let mut sparse_engine = engine(&path, 4096);
    sparse_engine.set_lsa(&indexer).unwrap();
    let sparse = sparse_engine.generate_greedy(std::slice::from_ref(&p), new, None);
    let (_, (pages, _), built) = sparse_engine.lsa_stats().unwrap();
    println!("dense  {:?}\nsparse {:?}\npages in use {pages}, built {built}", dense[0], sparse[0]);
    assert!(built >= 4, "a 2036-token prompt plus {new} tokens closes four cluster blocks; {built} pages were built");
    assert_eq!(sparse, dense, "every-key sparse attention diverged from dense attention");
}

/// An int8 KV cache is refused by name rather than read wrongly.
#[test]
fn an_int8_kv_cache_is_refused() {
    let Some(path) = checkpoint() else { return };
    let (cfg, src) = qwen3::open_checkpoint(&path).expect("open Qwen3-0.6B");
    let mut e = Engine::from_source_tier(cfg, &*src, BLOCK, 64, 2, 32, MAX_PREFILL, true, qwen3::Dtype::F32);
    let indexer = Indexer::random(LsaConfig::default(), geometry(&path), 1).unwrap();
    let err = e.set_lsa(&indexer).unwrap_err();
    assert!(err.contains("f32"), "{err}");
}

/// The sparse path is the one running: over the same prompt the last hidden
/// state agrees with dense attention at a budget past the context, and moves
/// away from it when the budget reads only a few long-range keys.
#[test]
fn the_budget_decides_what_prefill_reads() {
    let Some(path) = checkpoint() else { return };
    let mut rng = Lcg::random();
    let p = prompt(&mut rng, 4 * 512 + 300);
    let hidden = |e: &mut Engine| {
        let mut table = model::paged::BlockTable::new();
        e.prefill_for_perf(&mut table, &p)
    };
    let dense = hidden(&mut engine(&path, 4096));
    let geo = geometry(&path);
    let rel = |a: &[f32]| (a.iter().zip(&dense).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / dense.iter().map(|y| y * y).sum::<f32>()).sqrt();
    let mut full = engine(&path, 4096);
    full.set_lsa(&Indexer::random(LsaConfig { top_p: 1024, top_k: 4096, ..LsaConfig::default() }, geo, rng.next_u64()).unwrap()).unwrap();
    let mut tiny = engine(&path, 4096);
    tiny.set_lsa(&Indexer::random(LsaConfig { top_p: 2, top_k: 8, ..LsaConfig::default() }, geo, rng.next_u64()).unwrap()).unwrap();
    let (full_err, tiny_err) = (rel(&hidden(&mut full)), rel(&hidden(&mut tiny)));
    println!("relative error of the last hidden state: full budget {full_err:.2e}, 8-key budget {tiny_err:.2e}");
    assert!(full_err < 1e-3, "a budget past the context should reproduce dense attention: {full_err}");
    assert!(tiny_err > 10.0 * full_err.max(1e-6), "an 8-key budget changed nothing ({tiny_err}): the sparse path is not running");
}
