// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prefill attention of one layer at Qwen3-0.6B's shape (16 query heads, 8 KV
//! heads of 128): dense `paged_flash_prefill` against the sparse read of a
//! latent-sparse-attention selection, across context lengths. Run by hand:
//! `make test/one P=brain-model T=lsa_prefill_bench ARGS="--ignored --nocapture"`.
//!
//! Swedish Embedded AB implements long-context prefill whose attention cost
//! grows linearly instead of quadratically for its clients. If your team needs
//! expertise in measuring and closing the gap to the roofline on long
//! prompts, you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::Lcg;
use model::paged_sparse::{gqa_kernel, gqa_kernel_kv, rows_kernel_kv, Layout, PackedKv, SparseAttention, SparseBuffers, KERNEL};

const NH: u32 = 16;
const NKV: u32 = 8;
const HD: u32 = 128;
const BS: u32 = 16;

fn rand(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.signed()).collect()
}

/// A selection shaped like LSA's: `top_k` long-range keys below the window
/// in runs of 4 consecutive positions (LSA never splits a pool), the window,
/// the sinks.
fn selection(rng: &mut Lcg, t: u32, top_k: u32, window: u32, sink: u32) -> (Vec<u32>, u32) {
    let stride = top_k + window + sink + 3;
    let mut sel = vec![0u32; (t * stride) as usize];
    for i in 0..t {
        let full = &mut sel[(i * stride) as usize..][..stride as usize];
        let (header, row) = full.split_at_mut(3);
        let window_start = (i + 1).saturating_sub(window);
        let mut n = 0usize;
        let far = window_start.saturating_sub(sink);
        let pools = far / 4;
        for _ in 0..(top_k / 4).min(pools) {
            let start = sink + 4 * (rng.next_u32() % pools);
            for j in start..start + 4 {
                row[n] = j;
                n += 1;
            }
        }
        for j in window_start..=i {
            row[n] = j;
            n += 1;
        }
        for j in 0..sink.min(window_start) {
            row[n] = j;
            n += 1;
        }
        header.copy_from_slice(&[n as u32, 0, t]);
    }
    (sel, stride)
}

#[test]
#[ignore = "benchmark: run by hand on a GPU"]
fn dense_vs_sparse_prefill_attention() {
    let pipes: &'static [(&str, &str)] = Box::leak(vec![("paged_flash_prefill", kernels::PAGED_FLASH_PREFILL), KERNEL, gqa_kernel(NH / NKV).unwrap()].into_boxed_slice());
    let g = gpu_core::testgpu::dev(pipes);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("prefill attention kernels need workgroup barriers (a GPU)");
        return;
    }
    eprintln!("device {}", g.kind());
    let mut rng = Lcg::random();
    for &t in &[4096u32, 8192, 16384, 32768, 65536] {
        let max_bt = t.div_ceil(BS);
        let kv = (max_bt * BS * NKV * HD) as usize;
        let q = g.storage_init("q", &rand(&mut rng, (t * NH * HD) as usize));
        let pk = g.storage_init("pk", &rand(&mut rng, kv));
        let pv = g.storage_init("pv", &rand(&mut rng, kv));
        let ctx = g.storage((t * NH * HD) as u64);
        let table: Vec<u32> = (0..max_bt).collect();
        let bt_rows: Vec<u32> = (0..t).flat_map(|_| table.iter().copied()).collect();
        let bt_dense = g.storage(bt_rows.len() as u64);
        g.write(&bt_dense, &bt_rows);
        let lens: Vec<u32> = (1..=t).collect();
        let sl = g.storage(t as u64);
        g.write(&sl, &lens);
        let dense = vec![g.dispatch(
            0,
            &[&q, &pk, &pv, &bt_dense, &sl, &ctx],
            &[t, NH, NKV, HD, NH / NKV, BS, max_bt],
            gpu_core::Dispatch::Workgroups(NH * t.div_ceil(64)),
        )];
        let t_dense = gpu_core::profile::best_of(&g, &dense, 2);
        drop(bt_dense);

        let bt = g.storage(max_bt as u64);
        g.write(&bt, &table);
        let shape = SparseAttention { n_heads: NH, n_kv_heads: NKV, head_dim: HD, block_size: BS, max_bt };
        for &(top_k, window) in &[(2048u32, 512u32), (4096, 512)] {
            let (sel_h, stride) = selection(&mut rng, t, top_k, window, 4);
            let sel = g.storage(sel_h.len() as u64);
            g.write(&sel, &sel_h);
            let bufs = SparseBuffers { q: &q, pool_k: &pk, pool_v: &pv, block_tables: &bt, sel: &sel, scales: None, ctx: &ctx };
            let steps = vec![shape.step(&g, 2, Layout::PerKvHead, t, stride, &bufs).unwrap()];
            let t_sparse = gpu_core::profile::best_of(&g, &steps, 2);
            eprintln!(
                "T={t:>6}  dense {:>9.2} ms  sparse(k={top_k},w={window}) {:>9.2} ms  dense/sparse {:>5.2}x",
                t_dense * 1e3,
                t_sparse * 1e3,
                t_dense / t_sparse
            );
        }
    }
}

/// The served default, an int8 KV cache: dense `paged_flash_prefill_i8`
/// against the packed sparse read (a quarter of the f32 bytes per key), at a
/// small model's budget.
#[test]
#[ignore = "benchmark: run by hand on a GPU"]
fn dense_vs_sparse_prefill_attention_int8() {
    let pipes: &'static [(&str, &str)] = Box::leak(vec![("paged_flash_prefill_i8", kernels::PAGED_FLASH_PREFILL_I8), gqa_kernel_kv(NH / NKV, PackedKv::Int8).unwrap(), rows_kernel_kv(NH, NKV, HD, PackedKv::Int8).unwrap()].into_boxed_slice());
    let g = gpu_core::testgpu::dev(pipes);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("prefill attention kernels need workgroup barriers (a GPU)");
        return;
    }
    eprintln!("device {}", g.kind());
    let mut rng = Lcg::random();
    for &t in &[4096u32, 8192, 16384, 32768, 65536] {
        let max_bt = t.div_ceil(BS);
        let slots = (max_bt * BS) as usize;
        let words: Vec<u32> = (0..slots * (NKV * HD / 4) as usize).map(|_| rng.next_u32()).collect();
        let scales: Vec<f32> = (0..slots * NKV as usize).map(|_| 0.01 + 0.01 * rng.unit()).collect();
        let q = g.storage_init("q", &rand(&mut rng, (t * NH * HD) as usize));
        let (pk, pv) = (g.storage(words.len() as u64), g.storage(words.len() as u64));
        g.write(&pk, &words);
        g.write(&pv, &words);
        let (sk, sv) = (g.storage_init("sk", &scales), g.storage_init("sv", &scales));
        let ctx = g.storage((t * NH * HD) as u64);
        let table: Vec<u32> = (0..max_bt).collect();
        let bt_rows: Vec<u32> = (0..t).flat_map(|_| table.iter().copied()).collect();
        let bt_dense = g.storage(bt_rows.len() as u64);
        g.write(&bt_dense, &bt_rows);
        let lens: Vec<u32> = (1..=t).collect();
        let sl = g.storage(t as u64);
        g.write(&sl, &lens);
        let dense = vec![g.dispatch(
            0,
            &[&q, &pk, &pv, &sk, &sv, &bt_dense, &sl, &ctx],
            &[t, NH, NKV, HD, NH / NKV, BS, max_bt],
            gpu_core::Dispatch::Workgroups(NH * t.div_ceil(64)),
        )];
        let t_dense = gpu_core::profile::best_of(&g, &dense, 1);
        drop(bt_dense);
        let bt = g.storage(max_bt as u64);
        g.write(&bt, &table);
        let shape = SparseAttention { n_heads: NH, n_kv_heads: NKV, head_dim: HD, block_size: BS, max_bt };
        for &(top_k, window) in &[(1024u32, 512u32), (2048, 512)] {
            let (sel_h, stride) = selection(&mut rng, t, top_k, window, 4);
            let sel = g.storage(sel_h.len() as u64);
            g.write(&sel, &sel_h);
            let bufs = SparseBuffers { q: &q, pool_k: &pk, pool_v: &pv, block_tables: &bt, sel: &sel, scales: Some((&sk, &sv)), ctx: &ctx };
            let per_kv_head = gpu_core::profile::best_of(&g, &[shape.step(&g, 1, Layout::PerKvHead, t, stride, &bufs).unwrap()], 1);
            let whole_row = gpu_core::profile::best_of(&g, &[shape.step(&g, 2, Layout::PerRow, t, stride, &bufs).unwrap()], 1);
            eprintln!(
                "int8 KV  T={t:>6}  dense {:>9.2} ms  sparse(k={top_k},w={window}) per-KV-head {:>9.2} ms  whole-row {:>9.2} ms  dense/whole-row {:>5.2}x",
                t_dense * 1e3,
                per_kv_head * 1e3,
                whole_row * 1e3,
                t_dense / whole_row
            );
        }
    }
}
