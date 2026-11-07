// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latent sparse attention on Ripple's global layers, through the shared binding.
//!
//! Swedish Embedded AB implements long-context serving whose attention reads a
//! bounded set of keys per token for its clients. If your team needs
//! expertise in sparse attention for production LLM serving, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! The refusals run on every backend. The sparse reads run the
//! workgroup-cooperative LSA kernels, which only a device with workgroup
//! barriers executes (the CPU JIT does not), so those tests skip by name on it
//! and are the ones to run on a GPU:
//! `make test/one P=brain-ricursive T=it ARGS=lsa` with `--device gpu`.
//!
//! The tiny model's global layer is layer 2 (layers 0 and 1 slide), its cluster
//! blocks are 8 keys over 4-row KV blocks, so a 26-token prompt and 14 decode
//! steps close five blocks across prefill chunks and decode.

use model::lsa::indexer::{IndexGeometry, Indexer};
use model::lsa::tree::TreeConfig;
use model::lsa::LsaConfig;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use ricursive::{RicursiveConfig, RicursiveEngine, ServeConfig};

use crate::serve::{check_row, engine, engine_with, prefill_in_slices, recompute_last, serve_config, stream, tiny_config, tiny_dir};

/// Whether this device runs the workgroup-cooperative LSA kernels.
fn lsa_runs_here() -> bool {
    let runs = gpu_core::testgpu::dev(&[]).caps().workgroup_reductions;
    if !runs {
        brain_testutil::skip_unavailable("LSA: this backend does not run workgroup-barrier kernels (GPU-only)");
    }
    runs
}

fn geometry(cfg: &RicursiveConfig) -> IndexGeometry {
    // Global layers apply no RoPE, so the index reads unrotated heads.
    IndexGeometry { n_layers: cfg.num_hidden_layers, n_heads: cfg.num_attention_heads, n_kv_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, rope_base: 0.0 }
}

fn lsa_config(top_p: usize, top_k: usize) -> LsaConfig {
    LsaConfig { index_dim: 4, pool_size: 1, bucket_size: 2, bucket_cap: 4, cluster_block: 8, top_p, top_k, sink: 1, window: 8, early: None }
}

/// A random indexer over the global layers only.
fn indexer(cfg: &RicursiveConfig, lsa: LsaConfig, seed: u64) -> Indexer {
    let mut ix = Indexer::random(lsa, geometry(cfg), seed).expect("random indexer");
    for (l, layer) in ix.layers.iter_mut().enumerate() {
        if cfg.layer_type(l) != ricursive::LayerType::Full {
            *layer = None;
        }
    }
    ix
}

fn sparse_engine(dir: &str, prefill_chunk: u32, prefix_slots: u32, ix: &Indexer) -> RicursiveEngine {
    let mut e = engine_with(dir, ServeConfig { sparse_attention: true, ..serve_config(prefill_chunk, prefix_slots) });
    e.set_lsa(ix).expect("set_lsa");
    e.set_lsa_dense_until(0);
    e
}

/// A budget that covers every key (every bucket picked, `top_k` past the context) reads what dense attention
/// reads, so the engine must answer as the dense forward at every prefill chunk and decode step, across the closing
/// of cluster blocks.
#[test]
fn a_budget_past_the_context_equals_dense_through_prefill_chunks_and_decode() {
    let Some(dir) = tiny_dir() else { return };
    if !lsa_runs_here() {
        return;
    }
    let cfg = tiny_config(&dir);
    let ids = stream(5, 26 + 14, cfg.vocab_size as u32);
    let ix = indexer(&cfg, lsa_config(64, 256), 17);
    let mut worst = 0.0f32;
    for (chunk, budget) in [(8u32, usize::MAX), (5, 3), (3, usize::MAX)] {
        let mut e = sparse_engine(&dir, chunk, 0, &ix);
        let mut table = BlockTable::new();
        let hidden = prefill_in_slices(&mut e, &mut table, &ids[..26], budget);
        check_row(&format!("chunk {chunk}: prefill"), &e.logits(&hidden), &recompute_last(&dir, &ids[..26]), &mut worst);
        for pos in 26..ids.len() {
            let logits = e.decode_logits(&mut [&mut table], &[ids[pos]]);
            check_row(&format!("chunk {chunk}: decode at {pos}"), &logits[0], &recompute_last(&dir, &ids[..=pos]), &mut worst);
        }
        let stats = e.lsa_stats().expect("sparse attention is on");
        assert!(stats.pages_built >= 4, "40 tokens close at least four cluster blocks of 8; {} pages were built", stats.pages_built);
        assert!(stats.reads.rows >= ids.len() as u64 - 1, "every row of the global layer read sparsely: {} rows", stats.reads.rows);
        assert_eq!(stats.reads.selected_keys, stats.reads.causal_keys, "a budget past the context selects every causal key");
        e.release_table(&mut table);
        assert_eq!(e.lsa_stats().unwrap().pages_in_use, 0, "releasing the sequence frees its index pages");
    }
    eprintln!("LSA full budget: worst scaled error {worst:.3e} (tolerance {:.0e})", crate::serve::TOLERANCE);
}

/// A small budget reads fewer keys than dense attention would (the counters prove it), and so changes the answer.
#[test]
fn a_small_budget_reads_fewer_keys_than_dense_attention() {
    let Some(dir) = tiny_dir() else { return };
    if !lsa_runs_here() {
        return;
    }
    let cfg = tiny_config(&dir);
    let ids = stream(9, 40, cfg.vocab_size as u32);
    let ix = indexer(&cfg, lsa_config(2, 3), 23);
    let mut e = sparse_engine(&dir, 8, 0, &ix);
    let mut table = BlockTable::new();
    prefill_in_slices(&mut e, &mut table, &ids[..30], usize::MAX);
    let mut last = Vec::new();
    for &t in &ids[30..] {
        last = e.decode_logits(&mut [&mut table], &[t]).remove(0);
    }
    let stats = e.lsa_stats().unwrap();
    assert!(stats.reads.selected_keys < stats.reads.causal_keys, "{} selected of {} causal keys", stats.reads.selected_keys, stats.reads.causal_keys);
    let dense = recompute_last(&dir, &ids);
    assert!(last.iter().zip(&dense).any(|(a, b)| (a - b).abs() > 1e-4), "a sparse read that changes nothing is not reading sparsely");
}

/// With sparse attention on, a prompt that resumes from a cached prefix answers as the same prompt cold, and
/// concurrent sequences at different positions decode as each alone.
#[test]
fn prefix_reuse_and_batching_match_cold_and_alone_runs_with_sparse_attention() {
    let Some(dir) = tiny_dir() else { return };
    if !lsa_runs_here() {
        return;
    }
    let cfg = tiny_config(&dir);
    let vocab = cfg.vocab_size as u32;
    let ix = indexer(&cfg, lsa_config(3, 6), 29);
    let mut worst = 0.0f32;

    // Prefix: filed at 25 tokens (three closed blocks, a partly filled tail), resumed by 6 more.
    let ids = stream(13, 25 + 6 + 3, vocab);
    let mut warm = sparse_engine(&dir, 4, 2, &ix);
    let mut first = BlockTable::new();
    prefill_in_slices(&mut warm, &mut first, &ids[..25], usize::MAX);
    let mut second = BlockTable::new();
    let hidden = prefill_in_slices(&mut warm, &mut second, &ids[..31], usize::MAX);
    assert_eq!(second.cached_rows(), 25);
    let mut cold = sparse_engine(&dir, 4, 0, &ix);
    let mut cold_table = BlockTable::new();
    let cold_hidden = prefill_in_slices(&mut cold, &mut cold_table, &ids[..31], usize::MAX);
    check_row("warm vs cold prefill", &warm.logits(&hidden), &cold.logits(&cold_hidden), &mut worst);
    for &t in &ids[31..] {
        let w = warm.decode_logits(&mut [&mut second], &[t]).remove(0);
        let c = cold.decode_logits(&mut [&mut cold_table], &[t]).remove(0);
        check_row("warm vs cold decode", &w, &c, &mut worst);
    }
    let kept = warm.lsa_stats().unwrap().pages_in_use;
    assert!(kept > 0, "the snapshot keeps the index pages of its blocks");
    for t in [&mut first, &mut second] {
        warm.release_table(t);
    }
    assert!(warm.lsa_stats().unwrap().pages_in_use > 0, "the filed prompt's pages outlive its sequences");
    warm.reclaim_prefix(u32::MAX);
    assert_eq!(warm.lsa_stats().unwrap().pages_in_use, 0, "reclaiming the cache frees every index page");

    // Batching: three sequences at different positions, together and alone.
    let seqs: Vec<Vec<u32>> = [(1u32, 12usize), (2, 19), (3, 27)].iter().map(|&(seed, n)| stream(seed, n + 6, vocab)).collect();
    let lens = [12usize, 19, 27];
    let mut together = sparse_engine(&dir, 4, 0, &ix);
    let mut tables: Vec<BlockTable> = lens.iter().map(|_| BlockTable::new()).collect();
    let mut alone: Vec<(RicursiveEngine, BlockTable)> = Vec::new();
    for ((t, s), &n) in tables.iter_mut().zip(&seqs).zip(&lens) {
        prefill_in_slices(&mut together, t, &s[..n], usize::MAX);
        let mut e = sparse_engine(&dir, 4, 0, &ix);
        let mut at = BlockTable::new();
        prefill_in_slices(&mut e, &mut at, &s[..n], usize::MAX);
        alone.push((e, at));
    }
    for step in 0..6 {
        let inputs: Vec<u32> = seqs.iter().zip(&lens).map(|(s, &n)| s[n + step]).collect();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let batched = together.decode_logits(&mut refs, &inputs);
        for (i, ((e, t), logits)) in alone.iter_mut().zip(&batched).enumerate() {
            let solo = e.decode_logits(&mut [t], &[inputs[i]]);
            check_row(&format!("step {step} seq {i}: batched vs alone"), logits, &solo[0], &mut worst);
        }
    }
    eprintln!("LSA prefix and batching: worst scaled error {worst:.3e} (tolerance {:.0e})", crate::serve::TOLERANCE);
}

/// What the engine refuses, on any backend: an indexer that covers a sliding layer, was built for other
/// geometry or for rotated heads, or reads through a tree; an engine opened without the kernels.
#[test]
fn an_indexer_the_engine_cannot_serve_is_refused_by_name() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let good = indexer(&cfg, lsa_config(3, 6), 31);
    let mut e = engine_with(&dir, ServeConfig { sparse_attention: true, ..serve_config(4, 0) });

    let mut covers_sliding = good.clone();
    covers_sliding.layers[0] = Indexer::random(lsa_config(3, 6), geometry(&cfg), 1).unwrap().layers[0].clone();
    let err = e.set_lsa(&covers_sliding).unwrap_err();
    assert!(err.contains("layer 0") && err.contains("densely"), "{err}");

    let wrong_heads = Indexer { geometry: IndexGeometry { n_heads: 8, ..geometry(&cfg) }, ..good.clone() };
    let err = e.set_lsa(&wrong_heads).unwrap_err();
    assert!(err.contains("cannot index a model"), "{err}");

    let rotated = Indexer { geometry: IndexGeometry { rope_base: 10_000.0, ..geometry(&cfg) }, ..good.clone() };
    let err = e.set_lsa(&rotated).unwrap_err();
    assert!(err.contains("rope_base"), "{err}");

    let tree = Indexer { tree: Some(TreeConfig { node_bucket_size: 2, node_cluster_block: vec![4, 4], beam_width: vec![2, 2], start_nodes: None, page_size: 16, growth_buckets: 4 }), ..good.clone() };
    assert!(e.set_lsa(&tree).unwrap_err().contains("tree"));

    let mut dense_only = engine(&dir, 4, 0);
    assert!(dense_only.set_lsa(&good).unwrap_err().contains("sparse_attention"));

    // A valid indexer attaches where the device runs the kernels and is refused by name where it does not.
    match e.set_lsa(&good) {
        Ok(()) => assert!(e.lsa_stats().is_some()),
        Err(err) => assert!(err.contains("workgroup barriers"), "{err}"),
    }
}

/// The memory charge is the formula the attachment allocates by, and grows with the pool, not the sequence.
#[test]
fn the_memory_charge_covers_projections_pages_and_scratch() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let ix = indexer(&cfg, lsa_config(3, 6), 37);
    let serve = ServeConfig { sparse_attention: true, ..serve_config(4, 0) };
    let small = ricursive::lsa_device_bytes(&ix, &serve);
    let big = ricursive::lsa_device_bytes(&ix, &ServeConfig { num_blocks: serve.num_blocks * 2, ..serve.clone() });
    assert!(small > 0 && big > small, "pages scale with the block pool: {small} then {big}");
    if lsa_runs_here() {
        let mut e = engine_with(&dir, serve);
        e.set_lsa(&ix).unwrap();
        assert_eq!(e.lsa_stats().unwrap().device_bytes, small, "the charge equals what the attachment holds");
    }
}
