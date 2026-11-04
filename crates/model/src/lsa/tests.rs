// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::attend::{dense_selection, sparse_attention, AttentionShape};
use super::space::{HeadLayout, RotaryLayout};
use super::*;
use crate::hostmath;
use data::rng::Lcg;

/// A small configuration that still has several blocks, pools and buckets.
fn small() -> LsaConfig {
    LsaConfig { index_dim: 8, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 3, top_k: 9, sink: 2, window: 32, early: None }
}

fn unit_rows(rng: &mut Lcg, rows: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        let row: Vec<f32> = (0..dim).map(|_| rng.signed()).collect();
        out.extend(hostmath::l2_normalize(&row));
    }
    out
}

#[test]
fn config_refuses_sizes_that_lose_keys_or_cannot_merge() {
    assert!(LsaConfig::default().validate().is_ok());
    let narrow_window = LsaConfig { window: 256, ..LsaConfig::default() };
    assert!(narrow_window.validate().unwrap_err().contains("window"));
    let tight_cap = LsaConfig { bucket_cap: 15, ..LsaConfig::default() };
    assert!(tight_cap.validate().unwrap_err().contains("bucket_cap"));
    let unnested = LsaConfig { pool_size: 3, ..LsaConfig::default() };
    assert!(unnested.validate().is_err());
    let early = LsaConfig { early: Some(EarlyBudget { position: 10, top_k: 8192, top_p: 1 }), ..LsaConfig::default() };
    assert!(early.validate().is_err());
}

#[test]
fn ward_partitions_a_block_into_exactly_its_buckets_under_the_cap() {
    let cfg = LsaConfig::default();
    let mut rng = Lcg::random();
    let keys = unit_rows(&mut rng, cfg.cluster_block, cfg.index_dim);
    let pools = ward::pool_keys(&keys, cfg.index_dim, cfg.pool_size);
    let sizes = vec![cfg.pool_size as u32; cfg.pools_per_block()];
    let labels = ward::ward_labels(&pools, &sizes, cfg.index_dim, cfg.buckets_per_block(), cfg.bucket_cap as u32).unwrap();
    let mut counts = vec![0usize; cfg.buckets_per_block()];
    for &l in &labels {
        counts[l as usize] += cfg.pool_size;
    }
    assert!(counts.iter().all(|&c| c > 0 && c <= cfg.bucket_cap), "bucket sizes {counts:?}");
    // Numbered in the order of each bucket's first pool.
    let mut seen = 0u32;
    for &l in &labels {
        assert!(l <= seen, "bucket {l} appears before bucket {seen}");
        seen = seen.max(l + 1);
    }
}

#[test]
fn ward_merges_parallel_pools_first() {
    // Pools 0, 2 point one way and 1, 3 another: merging parallel vectors
    // loses no alignment, so the two clusters are found exactly.
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    let pools: Vec<f32> = [a, b, a, b].concat();
    let labels = ward::ward_labels(&pools, &[1, 1, 1, 1], 2, 2, 2).unwrap();
    assert_eq!(labels, vec![0, 1, 0, 1]);
}

#[test]
fn incremental_appends_build_the_same_index_as_one_batch() {
    let cfg = small();
    let mut rng = Lcg::random();
    let keys = unit_rows(&mut rng, 3 * cfg.cluster_block + 5, cfg.index_dim);
    let whole = BucketIndex::build(cfg, &keys).unwrap();
    let mut pieces = BucketIndex::new(cfg).unwrap();
    for chunk in keys.chunks(7 * cfg.index_dim) {
        pieces.push_keys(chunk).unwrap();
    }
    assert_eq!(whole.len(), keys.len() / cfg.index_dim);
    assert_eq!(whole.closed_blocks(), 3);
    assert_eq!(whole.centroids(), pieces.centroids());
    assert_eq!(whole.members(), pieces.members());
    // Members are positions of the bucket's own block, ascending.
    for (row, members) in whole.members().chunks(cfg.bucket_cap).enumerate() {
        let block = row / cfg.buckets_per_block();
        let held: Vec<i32> = members.iter().copied().filter(|&m| m >= 0).collect();
        assert_eq!(held.len(), whole.sizes()[row] as usize);
        assert!(held.windows(2).all(|w| w[0] < w[1]));
        assert!(held.iter().all(|&m| m as usize / cfg.cluster_block == block));
    }
}

#[test]
fn truncation_rolls_back_the_open_block_only() {
    let cfg = small();
    let mut rng = Lcg::random();
    let keys = unit_rows(&mut rng, cfg.cluster_block + 10, cfg.index_dim);
    let mut index = BucketIndex::build(cfg, &keys).unwrap();
    index.truncate(cfg.cluster_block + 4).unwrap();
    assert_eq!(index.len(), cfg.cluster_block + 4);
    let err = index.truncate(cfg.cluster_block - 1).unwrap_err();
    assert!(err.contains("rebuild"), "{err}");
    assert_eq!(index.len(), cfg.cluster_block + 4, "a refused truncation changes nothing");
    let rejected = vec![2.0f32; cfg.index_dim];
    assert!(index.push_keys(&rejected).is_err(), "a key that is not unit length is refused");
    assert_eq!(index.len(), cfg.cluster_block + 4, "a refused append changes nothing");
}

#[test]
fn a_selection_is_unique_causal_and_always_holds_window_and_sinks() {
    let cfg = small();
    let mut rng = Lcg::random();
    let t = 5 * cfg.cluster_block + 3;
    let index = BucketIndex::build(cfg, &unit_rows(&mut rng, t, cfg.index_dim)).unwrap();
    let positions: Vec<u32> = (0..t as u32).collect();
    let selection = select(&index, &unit_rows(&mut rng, t, cfg.index_dim), &positions).unwrap();
    for (r, &p) in positions.iter().enumerate() {
        let p = p as usize;
        let row = selection.row(r);
        let mut sorted: Vec<i32> = row.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), row.len(), "row {r} repeats a key");
        assert!(row.iter().all(|&j| j >= 0 && j as usize <= p), "row {r} reads past its position");
        let window_start = (p + 1).saturating_sub(cfg.window);
        assert!((window_start..=p).all(|j| row.contains(&(j as i32))), "row {r} lacks its window");
        assert!((0..cfg.sink.min(p + 1)).all(|j| row.contains(&(j as i32))), "row {r} lacks its sinks");
        let long_range: Vec<i32> = row.iter().copied().filter(|&j| j as usize >= cfg.sink && (j as usize) < window_start).collect();
        assert!(long_range.len() <= cfg.top_k);
        // Long-range keys come only from blocks closed before the query's own.
        assert!(long_range.iter().all(|&j| (j as usize) < p / cfg.cluster_block * cfg.cluster_block));
    }
}

#[test]
fn a_budget_past_the_context_reads_every_causal_key_and_is_dense_attention() {
    let cfg = LsaConfig { top_p: 1 << 20, top_k: 1 << 20, ..small() };
    let mut rng = Lcg::random();
    let t = 4 * cfg.cluster_block + 9;
    let index = BucketIndex::build(cfg, &unit_rows(&mut rng, t, cfg.index_dim)).unwrap();
    let positions: Vec<u32> = (0..t as u32).collect();
    let selection = select(&index, &unit_rows(&mut rng, t, cfg.index_dim), &positions).unwrap();
    for (r, &p) in positions.iter().enumerate() {
        let mut row = selection.row(r).to_vec();
        row.sort_unstable();
        assert_eq!(row, (0..=p as i32).collect::<Vec<_>>(), "row {r}");
    }

    let shape = AttentionShape { n_heads: 4, n_kv_heads: 2, head_dim: 8 };
    let q: Vec<f32> = (0..t * 32).map(|_| rng.signed()).collect();
    let k: Vec<f32> = (0..t * 16).map(|_| rng.signed()).collect();
    let v: Vec<f32> = (0..t * 16).map(|_| rng.signed()).collect();
    let scale = 1.0 / (8f32).sqrt();
    let sparse = sparse_attention(&q, &k, &v, shape, scale, &selection).unwrap();
    let dense = sparse_attention(&q, &k, &v, shape, scale, &dense_selection(&positions)).unwrap();
    let worst = sparse.iter().zip(&dense).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-6, "sparse attention over every causal key differs from dense by {worst}");
}

#[test]
fn a_query_must_already_be_in_the_index() {
    let cfg = small();
    let mut rng = Lcg::random();
    let index = BucketIndex::build(cfg, &unit_rows(&mut rng, 40, cfg.index_dim)).unwrap();
    let err = select(&index, &unit_rows(&mut rng, 1, cfg.index_dim), &[40]).unwrap_err();
    assert!(err.contains("past"), "{err}");
}

#[test]
fn the_index_space_sums_query_groups_and_drops_rotated_channels() {
    let layout = HeadLayout { n_heads: 4, n_kv_heads: 2, head_dim: 4, rotary: RotaryLayout::Prefix(1) };
    assert_eq!(layout.index_dim().unwrap(), 6);
    let q: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let k: Vec<f32> = (0..8).map(|i| (i % 3) as f32 + 1.0).collect();
    let iq = layout.index_queries(&q).unwrap();
    let ik = layout.index_keys(&k).unwrap();
    // Unnormalised: group 0 sums heads 0,1 and group 1 heads 2,3, channels 1..4.
    let raw_q = [1.0 + 5.0, 2.0 + 6.0, 3.0 + 7.0, 9.0 + 13.0, 10.0 + 14.0, 11.0 + 15.0];
    let raw_k = [2.0, 3.0, 1.0, 3.0, 1.0, 2.0];
    let unit_q = hostmath::l2_normalize(&raw_q);
    let unit_k = hostmath::l2_normalize(&raw_k);
    assert!(iq.iter().zip(&unit_q).all(|(a, b)| (a - b).abs() < 1e-6));
    assert!(ik.iter().zip(&unit_k).all(|(a, b)| (a - b).abs() < 1e-6));
    // The summed position-free logit is what the index scores.
    let logit: f32 = (0..4).map(|h| (1..4).map(|c| q[h * 4 + c] * k[(h / 2) * 4 + c]).sum::<f32>()).sum();
    let scored = dot(&iq, &ik) as f32 * raw_q.iter().map(|x| x * x).sum::<f32>().sqrt() * raw_k.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((logit - scored).abs() < 1e-3, "{logit} vs {scored}");
    let whole = HeadLayout { rotary: RotaryLayout::Suffix(4), ..layout };
    assert!(whole.index_dim().is_err());
}

/// The official reference implementation's own output on fixed inputs:
/// `tools/goldens/lsa_dump_reference.py` writes it.
#[test]
fn matches_the_reference_implementation() {
    let path = brain_testutil::testdata("lsa/flat_reference.safetensors");
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip(&format!("LSA reference golden absent: {path} (tools/goldens/lsa_dump_reference.py)"));
        return;
    }
    let tensors = checkpoint::safetensors::read(&path).unwrap();
    let get = |name: &str| tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden lacks {name}"));
    let cfg_t = &get("config").data;
    let cfg = LsaConfig {
        index_dim: get("keys").shape[1],
        pool_size: cfg_t[0] as usize,
        bucket_size: cfg_t[1] as usize,
        bucket_cap: cfg_t[2] as usize,
        cluster_block: cfg_t[3] as usize,
        top_p: cfg_t[4] as usize,
        top_k: cfg_t[5] as usize,
        sink: cfg_t[6] as usize,
        window: cfg_t[7] as usize,
        early: None,
    };
    let index = BucketIndex::build(cfg, &get("keys").data).unwrap();
    let members: Vec<i32> = get("bucket_members").data.iter().map(|&x| x as i32).collect();
    assert_eq!(index.members(), &members[..], "bucket membership differs from the reference");
    let worst = index.centroids().iter().zip(&get("bucket_centroids").data).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-5, "centroids differ by {worst}");

    let positions: Vec<u32> = get("positions").data.iter().map(|&x| x as u32).collect();
    let selection = select(&index, &get("queries").data, &positions).unwrap();
    let want_lens: Vec<u32> = get("lengths").data.iter().map(|&x| x as u32).collect();
    assert_eq!(selection.lens, want_lens, "selection lengths differ");
    let want_ids: Vec<i32> = get("selected").data.iter().map(|&x| x as i32).collect();
    assert_eq!(selection.ids, want_ids, "selected keys differ from the reference");
}

/// The reference's node bucket levels and beam read on fixed inputs, beaming
/// from the top level and from the lowest level of at most `start_nodes`
/// node buckets: `tools/goldens/lsa_dump_reference.py` writes them.
#[test]
fn the_tree_matches_the_reference_implementation() {
    for case in ["beam", "start"] {
        let path = brain_testutil::testdata(&format!("lsa/tree_reference_{case}.safetensors"));
        if !std::path::Path::new(&path).exists() {
            brain_testutil::skip(&format!("LSA tree golden absent: {path} (tools/goldens/lsa_dump_reference.py)"));
            return;
        }
        let tensors = checkpoint::safetensors::read(&path).unwrap();
        let get = |name: &str| tensors.iter().find(|t| t.name == name);
        let need = |name: &str| get(name).unwrap_or_else(|| panic!("golden lacks {name}"));
        let c = &need("config").data;
        let cfg = LsaConfig {
            index_dim: need("keys").shape[1],
            pool_size: c[0] as usize,
            bucket_size: c[1] as usize,
            bucket_cap: c[2] as usize,
            cluster_block: c[3] as usize,
            top_p: c[4] as usize,
            top_k: c[5] as usize,
            sink: c[6] as usize,
            window: c[7] as usize,
            early: None,
        };
        let t = &need("tree").data;
        let (levels, _) = need("levels").data.as_chunks::<2>();
        let tree = tree::TreeConfig {
            node_bucket_size: t[0] as usize,
            node_cluster_block: levels.iter().map(|l| l[0] as usize).collect(),
            beam_width: levels.iter().map(|l| l[1] as usize).collect(),
            start_nodes: (t[4] > 0.0).then_some(t[4] as usize),
            page_size: t[2] as usize,
            growth_buckets: t[3] as usize,
        };
        let keys = &need("keys").data;
        let index = BucketIndex::build(cfg, keys).unwrap();
        let h = tree::build_hierarchy(&index, keys, &tree).unwrap();
        for (l, (centroids, children)) in h.node_centroids.iter().zip(&h.children).enumerate() {
            let want: Vec<u32> = need(&format!("children_{}", l + 1)).data.iter().map(|&x| x as u32).collect();
            assert_eq!(children, &want, "{case}: level {} node buckets differ from the reference", l + 1);
            let worst = centroids.iter().zip(&need(&format!("node_centroids_{}", l + 1)).data).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst < 1e-5, "{case}: level {} centroids differ by {worst}", l + 1);
        }
        assert!(get(&format!("children_{}", h.children.len() + 1)).is_none(), "{case}: the reference built more levels");
        let positions: Vec<u32> = need("positions").data.iter().map(|&x| x as u32).collect();
        let selection = tree::select_tree(&index, &h, &tree, &need("queries").data, &positions).unwrap();
        let want_lens: Vec<u32> = need("lengths").data.iter().map(|&x| x as u32).collect();
        assert_eq!(selection.lens, want_lens, "{case}: selection lengths differ");
        let want_ids: Vec<i32> = need("selected").data.iter().map(|&x| x as i32).collect();
        assert_eq!(selection.ids, want_ids, "{case}: selected keys differ from the reference");
        // The beam prunes: some queries read differently from the flat read.
        let flat = select(&index, &need("queries").data, &positions).unwrap();
        let pruned = (0..positions.len()).filter(|&r| flat.row(r) != selection.row(r)).count();
        assert!(pruned > 0, "{case}: the beam read exactly what the flat read reads");
        println!("{case}: {} levels of {:?} node buckets match the reference; {pruned} of {} queries read otherwise than flat", h.children.len(), h.nodes(), positions.len());
    }
}
