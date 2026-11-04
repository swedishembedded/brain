// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::attend::{sparse_attention, AttentionShape};
use super::engine::*;
use super::indexer::{IndexGeometry, Indexer};
use super::*;
use crate::paged::{BlockAllocator, BlockTable};
use crate::paged_sparse::{gqa_kernel, KERNEL};
use data::rng::Lcg;

const NH: u32 = 4;
const NKV: u32 = 2;
const HD: u32 = 16;
const BS: u32 = 8;
const MBT: u32 = 32;

fn cfg() -> LsaConfig {
    LsaConfig { index_dim: 8, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 3, top_k: 10, sink: 2, window: 32, early: None }
}

fn geometry() -> IndexGeometry {
    IndexGeometry { n_layers: 2, n_heads: NH as usize, n_kv_heads: NKV as usize, head_dim: HD as usize, rope_base: 10000.0 }
}

struct Fixture {
    g: gpu_core::Gpu,
    pool_k: Vec<gpu_core::DeviceBuffer>,
    pool_v: Vec<gpu_core::DeviceBuffer>,
    /// Per layer, the host copies of the pools.
    host_k: Vec<Vec<f32>>,
    host_v: Vec<Vec<f32>>,
    alloc: BlockAllocator,
}

fn fixture(rng: &mut Lcg) -> Option<Fixture> {
    let pipes: &'static [(&str, &str)] = Box::leak(vec![device::KERNELS[0], device::KERNELS[1], gqa_kernel(NH / NKV).unwrap(), KERNEL, device::KERNELS[2]].into_boxed_slice());
    let g = gpu_core::testgpu::dev(pipes);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("LSA engine: workgroup-barrier kernels need a GPU");
        return None;
    }
    let num_blocks = 64u32;
    let n = (num_blocks * BS * NKV * HD) as usize;
    let host_k: Vec<Vec<f32>> = (0..2).map(|_| (0..n).map(|_| rng.signed()).collect()).collect();
    let host_v: Vec<Vec<f32>> = (0..2).map(|_| (0..n).map(|_| rng.signed()).collect()).collect();
    let pool_k = host_k.iter().map(|k| g.storage_init("pk", k)).collect();
    let pool_v = host_v.iter().map(|v| g.storage_init("pv", v)).collect();
    Some(Fixture { g, pool_k, pool_v, host_k, host_v, alloc: BlockAllocator::new(num_blocks, BS) })
}

fn kernels() -> LsaKernels {
    LsaKernels { index_vectors: 0, read: 1, ward: 4, attend: 2, layout: crate::paged_sparse::Layout::PerKvHead, attend_few: 2, layout_few: crate::paged_sparse::Layout::PerKvHead, index_keys_packed: None }
}

/// A sequence's contiguous K or V rows gathered from a pool.
fn gather(pool: &[f32], table: &BlockTable, len: u32) -> Vec<f32> {
    let kv_row = (NKV * HD) as usize;
    (0..len).flat_map(|p| {
        let (b, o) = table.locate(p, BS);
        pool[((b * BS + o) as usize) * kv_row..][..kv_row].to_vec()
    }).collect()
}

/// Decode rows of two sequences: the device context equals the host oracle
/// over the selection the device made, every selection is causal and holds
/// its window and sinks, and each closed block got exactly one page.
#[test]
fn a_decode_step_reads_its_selection_and_matches_the_oracle() {
    let mut rng = Lcg::random();
    let Some(mut f) = fixture(&mut rng) else { return };
    let indexer = Indexer::random(cfg(), geometry(), rng.next_u64()).unwrap();
    let kv = KvGeometry { block_size: BS, max_blocks_per_seq: MBT };
    let mut lsa = LsaAttention::new(&f.g, &indexer, kernels(), kv, 8, 16, None).unwrap();
    let lens = [150u32, 70];
    let mut tables: Vec<BlockTable> = lens.iter().map(|&n| {
        let mut t = BlockTable::new();
        t.reserve(n, &mut f.alloc).unwrap();
        t
    }).collect();
    {
        let mut rows: Vec<(&mut BlockTable, u32, u32)> = tables.iter_mut().zip(lens).map(|(t, n)| (t, n, 1)).collect();
        lsa.prepare(&f.g, &mut rows, KeyPools { keys: &f.pool_k, scales: None }).unwrap();
    }
    for (t, &n) in tables.iter().zip(&lens) {
        assert_eq!(t.index_pages().len(), (n / 32) as usize, "one page per closed cluster block");
    }
    assert_eq!(lsa.built_pages(), (150 / 32 + 70 / 32) as u64);

    let rows = lens.len() as u32;
    let q_h: Vec<f32> = (0..rows * NH * HD).map(|_| rng.signed()).collect();
    let positions: Vec<u32> = lens.iter().map(|n| n - 1).collect();
    let g = &f.g;
    let q = g.storage_init("q", &q_h);
    let pos = g.storage(rows as u64);
    g.write(&pos, &positions);
    let seqlen = g.storage(rows as u64);
    g.write(&seqlen, &lens);
    let mut bt = vec![0u32; (rows * MBT) as usize];
    for (r, t) in tables.iter().enumerate() {
        bt[r * MBT as usize..][..t.blocks().len()].copy_from_slice(t.blocks());
    }
    let block_tables = g.storage(bt.len() as u64);
    g.write(&block_tables, &bt);
    let ctx = g.storage((rows * NH * HD) as u64);
    for layer in 0..2 {
        let mut s = Vec::new();
        let b = LayerBuffers { q: &q, pos: &pos, seqlen: &seqlen, block_tables: &block_tables, pool_k: &f.pool_k[layer], pool_v: &f.pool_v[layer], scales: None, ctx: &ctx };
        lsa.record_layer(g, &mut s, layer, rows, &b).unwrap();
        g.submit(&[], &s);
        let got = g.read(&ctx, (rows * NH * HD) as usize);
        let (sel, meta) = lsa.read_selection(g, rows);
        for (r, t) in tables.iter().enumerate() {
            let n = meta[r * 3] as usize;
            assert_eq!(meta[r * 3 + 2], lens[r]);
            let ids: Vec<i32> = sel[r].iter().map(|&j| j as i32).collect();
            assert_eq!(ids.len(), n);
            let p = positions[r] as i32;
            assert!(ids.iter().all(|&j| j <= p), "row {r} reads past itself");
            assert!((p - 31..=p).all(|j| ids.contains(&j)), "row {r} lacks its window");
            assert!(ids.contains(&0) && ids.contains(&1), "row {r} lacks its sinks");
            assert!(n < lens[r] as usize, "row {r} read every key: nothing sparse was tested");
            let selection = Selection { width: n, ids, lens: vec![n as u32] };
            let k = gather(&f.host_k[layer], t, lens[r]);
            let v = gather(&f.host_v[layer], t, lens[r]);
            let shape = AttentionShape { n_heads: NH as usize, n_kv_heads: NKV as usize, head_dim: HD as usize };
            let want = sparse_attention(&q_h[r * (NH * HD) as usize..][..(NH * HD) as usize], &k, &v, shape, 1.0 / (HD as f32).sqrt(), &selection).unwrap();
            let worst = want.iter().zip(&got[r * (NH * HD) as usize..]).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst < 1e-5, "layer {layer} row {r}: context differs from the oracle by {worst}");
        }
    }

    // A sequence over the same KV blocks shares the pages instead of
    // rebuilding them; truncation and release give them back.
    let mut twin = tables[0].fork(&mut f.alloc);
    lsa.prepare(&f.g, &mut [(&mut twin, 150, 1)], KeyPools { keys: &f.pool_k, scales: None }).unwrap();
    assert_eq!(twin.index_pages(), tables[0].index_pages());
    assert_eq!(lsa.built_pages(), (150 / 32 + 70 / 32) as u64, "the twin's pages were rebuilt");
    let in_use = lsa.page_usage().0;
    twin.truncate(64, &mut f.alloc);
    lsa.prepare(&f.g, &mut [(&mut twin, 64, 1)], KeyPools { keys: &f.pool_k, scales: None }).unwrap();
    assert_eq!(twin.index_pages().len(), 2);
    assert_eq!(lsa.page_usage().0, in_use, "shared pages stay while the original holds them");
    for t in tables.iter_mut().chain(std::iter::once(&mut twin)) {
        lsa.release(t);
        t.release(&mut f.alloc);
    }
    assert_eq!(lsa.page_usage().0, 0, "every page is free once every sequence released");
}

#[test]
fn an_indexer_round_trips_and_refuses_another_model() {
    let mut rng = Lcg::random();
    let indexer = Indexer::random(cfg(), geometry(), rng.next_u64()).unwrap();
    let dir = std::env::temp_dir().join(format!("brain-lsa-indexer-{}", rng.next_u64()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("indexer.safetensors");
    indexer.save(path.to_str().unwrap()).unwrap();
    let back = Indexer::load(path.to_str().unwrap()).unwrap();
    assert_eq!(back, indexer);
    let other = IndexGeometry { n_kv_heads: 4, ..geometry() };
    assert!(back.check_model(&other).is_err());
    assert!(back.check_model(&geometry()).is_ok());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pca_projections_are_orthonormal_leading_directions() {
    let mut rng = Lcg::random();
    let geo = IndexGeometry { n_layers: 1, n_heads: 2, n_kv_heads: 1, head_dim: 16, rope_base: 0.0 };
    let c = LsaConfig { index_dim: 4, ..cfg() };
    // Keys concentrated on the first four channels.
    let keys: Vec<f32> = (0..400).flat_map(|_| (0..16).map(|i| if i < 4 { 3.0 * rng.signed() } else { 0.01 * rng.signed() }).collect::<Vec<_>>()).collect();
    let indexer = Indexer::pca(c, geo, &[keys]).unwrap();
    let w = &indexer.layers[0].as_ref().unwrap().wk;
    for a in 0..4 {
        for b in 0..4 {
            let dot: f32 = (0..16).map(|i| w[i * 4 + a] * w[i * 4 + b]).sum();
            assert!((dot - f32::from(u8::from(a == b))).abs() < 1e-4, "columns {a},{b} dot {dot}");
        }
        let outside: f32 = (4..16).map(|i| w[i * 4 + a] * w[i * 4 + a]).sum();
        assert!(outside < 1e-3, "direction {a} leaks {outside} outside the key subspace");
    }
}
