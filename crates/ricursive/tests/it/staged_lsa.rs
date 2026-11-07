// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latent sparse attention when the global layers are spread over pipeline stages: one binding per stage device
//! over the global layers that stage owns, with the index pages of every stage riding the shared block tables.
//!
//! Swedish Embedded AB implements solutions for serving million-token contexts over several accelerators, each
//! holding the layers, the KV cache and the sparse-attention index its memory allows, for its clients. If your
//! team needs expertise in sharding long-context inference then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! The tiny golden model has one global layer, which cannot be split over two stages, so these tests run a
//! synthetic four-layer model (sliding, global, sliding, global; random weights, no reference implementation
//! needed: every check compares two runs of this engine). The sparse kernels are workgroup-cooperative, so the
//! tests need a device that runs them (the software Vulkan adapter, `BRAIN_DEVICE=software`; two device handles
//! of it coexist) and skip by name on the CPU JIT.

use model::lsa::indexer::{IndexGeometry, Indexer};
use model::lsa::LsaConfig;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use ricursive::{RicursiveConfig, RicursiveEngine, ServeConfig};

use crate::serve::{prefill_in_slices, stream};
use crate::synthetic::{synthetic, Shape};

const LAYERS: [&str; 4] = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"];

fn model() -> crate::synthetic::Model {
    synthetic(&LAYERS, Shape { heads: 4, kv_heads: 2, head_dim: 16 })
}

fn runs_here() -> bool {
    let runs = gpu_core::testgpu::dev(&[]).caps().workgroup_reductions;
    if !runs {
        brain_testutil::skip_unavailable("LSA: this backend does not run workgroup-barrier kernels (GPU-only)");
    }
    runs
}

fn serve(prefix_slots: u32, sparse: bool) -> ServeConfig {
    ServeConfig { sparse_attention: sparse, ..ServeConfig::new(4, 160, 96, 4, prefix_slots, 8) }
}

/// An indexer over both global layers (1 and 3). `top_p`/`top_k` choose the budget.
fn indexer(cfg: &RicursiveConfig, top_p: usize, top_k: usize) -> Indexer {
    let geometry = IndexGeometry { n_layers: cfg.num_hidden_layers, n_heads: cfg.num_attention_heads, n_kv_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, rope_base: 0.0 };
    let lsa = LsaConfig { index_dim: 4, pool_size: 1, bucket_size: 2, bucket_cap: 4, cluster_block: 8, top_p, top_k, sink: 1, window: 8, early: None };
    Indexer::random(lsa, geometry, 17).expect("indexer").restricted_to(|l| cfg.layer_type(l) == ricursive::LayerType::Full)
}

/// Prefill `prompt` in slices, then decode `decode` more tokens: every logits row.
fn trace(e: &mut RicursiveEngine, ids: &[u32], prompt: usize) -> Vec<Vec<f32>> {
    let mut t = BlockTable::new();
    let h = prefill_in_slices(e, &mut t, &ids[..prompt], 5);
    let mut rows = vec![e.logits(&h)];
    for &token in &ids[prompt..] {
        rows.push(e.decode_logits(&mut [&mut t], &[token]).remove(0));
    }
    e.release_table(&mut t);
    rows
}

fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().zip(b).map(|(x, y)| crate::taps::scaled_error("row", x, y)).fold(0.0, f32::max)
}

/// Spec: splitting the layers over two devices changes nothing about what a sparse read selects or answers:
/// the two-stage engine (one binding per stage) gives the single-stage engine's logits, and reads the same keys,
/// at a budget small enough that the selection matters.
#[test]
fn a_two_stage_sparse_engine_equals_the_single_stage_sparse_engine() {
    if !runs_here() {
        return;
    }
    let m = model();
    let ids = stream(5, 26 + 14, m.cfg.vocab_size as u32);
    let ix = indexer(&m.cfg, 2, 3);
    let mut one = m.engine(serve(0, true));
    one.set_lsa(&ix).expect("single stage");
    let mut two = m.staged(serve(0, true));
    two.set_lsa(&ix).expect("two stages");
    let (a, b) = (trace(&mut one, &ids, 26), trace(&mut two, &ids, 26));
    let err = worst(&b, &a);
    let (sa, sb) = (one.lsa_stats().unwrap(), two.lsa_stats().unwrap());
    eprintln!("two-stage sparse vs single-stage sparse: worst scaled error {err:.3e}; selected {} of {} causal keys (single {} of {})", sb.reads.selected_keys, sb.reads.causal_keys, sa.reads.selected_keys, sa.reads.causal_keys);
    assert!(err <= 1e-6, "{err:.3e}");
    assert!(sb.reads.selected_keys < sb.reads.causal_keys, "a small budget reads fewer keys than dense attention");
    assert_eq!((sb.reads.selected_keys, sb.reads.causal_keys), (sa.reads.selected_keys, sa.reads.causal_keys), "the same keys are selected, split over two devices");
    assert_eq!(sb.pages_built, 2 * sa.pages_built, "each stage builds the pages of its own global layer: one per cluster block per stage");
    assert_eq!(sb.pages_in_use, 0, "releasing the sequence frees every stage's pages");
}

/// Spec: at a budget past the context every causal key is selected, so the staged sparse engine answers as the
/// dense engine (a single stage with no index).
#[test]
fn a_full_budget_across_stages_equals_the_dense_engine() {
    if !runs_here() {
        return;
    }
    let m = model();
    let ids = stream(5, 26 + 14, m.cfg.vocab_size as u32);
    let mut dense = m.engine(serve(0, false));
    let mut two = m.staged(serve(0, true));
    two.set_lsa(&indexer(&m.cfg, 64, 256)).expect("two stages");
    let (a, b) = (trace(&mut dense, &ids, 26), trace(&mut two, &ids, 26));
    let err = worst(&b, &a);
    let stats = two.lsa_stats().unwrap();
    eprintln!("full-budget two-stage sparse vs dense: worst scaled error {err:.3e}");
    assert!(err <= 1e-5, "{err:.3e}");
    assert_eq!(stats.reads.selected_keys, stats.reads.causal_keys);
    assert!(stats.pages_built >= 8, "two global layers' worth of pages: {}", stats.pages_built);
}

/// Spec: a prompt resumed from a cached prefix reads the pages the snapshot retained on every stage: the warm
/// logits equal the cold ones and the single-stage engine's, with pages shared (built once) and freed with the
/// last holder.
#[test]
fn prefix_reuse_across_stages_with_sparse_attention_on() {
    if !runs_here() {
        return;
    }
    let m = model();
    let ids = stream(11, 34, m.cfg.vocab_size as u32);
    let ix = indexer(&m.cfg, 2, 3);
    let run = |e: &mut RicursiveEngine| {
        let mut first = BlockTable::new();
        prefill_in_slices(e, &mut first, &ids[..24], usize::MAX);
        let built_before = e.lsa_stats().unwrap().pages_built;
        let mut second = BlockTable::new();
        let h = prefill_in_slices(e, &mut second, &ids[..30], usize::MAX);
        assert_eq!(second.cached_rows(), 24, "the second request resumes from the filed prompt");
        let warm = vec![e.logits(&h), e.decode_logits(&mut [&mut second], &[ids[30]]).remove(0)];
        let rebuilt = e.lsa_stats().unwrap().pages_built - built_before;
        for t in [&mut first, &mut second] {
            e.release_table(t);
        }
        let held = e.lsa_stats().unwrap().pages_in_use;
        e.reclaim_prefix(u32::MAX);
        assert_eq!(e.lsa_stats().unwrap().pages_in_use, 0, "reclaiming the cache frees every stage's pages");
        (warm, rebuilt, held)
    };
    let mut one = m.engine(serve(2, true));
    one.set_lsa(&ix).unwrap();
    let mut two = m.staged(serve(2, true));
    two.set_lsa(&ix).unwrap();
    let ((wa, ra, ha), (wb, rb, hb)) = (run(&mut one), run(&mut two));
    let err = worst(&wb, &wa);
    eprintln!("staged prefix reuse: worst scaled error {err:.3e}; pages rebuilt for the resumed prompt {rb} (single {ra}); pages held by the cache {hb} (single {ha})");
    assert!(err <= 1e-6, "{err:.3e}");
    assert_eq!((rb, hb), (2 * ra, 2 * ha), "every stage reuses and retains its own pages");
    assert!(ha > 0, "the cached prefix keeps its index pages");
}
