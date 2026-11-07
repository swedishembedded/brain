// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::*;
use crate::lsa::attend::{sparse_attention, AttentionShape};
use crate::lsa::space::{HeadLayout, RotaryLayout};
use crate::lsa::{select, BucketIndex, LsaConfig};
use data::rng::Rng;
use gpu_core::select::Dtype;

/// The shared test device for `pipes`, or `None` - reported as a skip - on a
/// backend that does not run workgroup-barrier kernels (the CPU JIT).
fn barrier_gpu(pipes: &'static [(&'static str, &'static str)]) -> Option<Gpu> {
    let g = gpu_core::testgpu::dev(pipes);
    if g.caps().workgroup_reductions {
        Some(g)
    } else {
        brain_testutil::skip_unavailable("paged sparse attention: this backend does not run workgroup-barrier kernels (GPU-only)");
        None
    }
}

/// Sequences of `lens` keys scattered over one shared pool through
/// interleaved block tables, so the paged indirection is really exercised.
struct Cache {
    shape: SparseAttention,
    /// Per sequence, contiguous `[len, n_kv * hd]` keys and values.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    tables: Vec<u32>,
    pool_k: Vec<f32>,
    pool_v: Vec<f32>,
}

impl Cache {
    fn new(rng: &mut Rng, shape: SparseAttention, lens: &[u32]) -> Cache {
        let batch = lens.len() as u32;
        let kv_row = (shape.n_kv_heads * shape.head_dim) as usize;
        let num_blocks = batch * shape.max_bt;
        let tables: Vec<u32> = (0..batch).flat_map(|b| (0..shape.max_bt).map(move |i| b + i * batch)).collect();
        let mut gauss = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_gaussian() as f32).collect() };
        let k: Vec<Vec<f32>> = lens.iter().map(|&t| gauss(t as usize * kv_row)).collect();
        let v: Vec<Vec<f32>> = lens.iter().map(|&t| gauss(t as usize * kv_row)).collect();
        let mut pool_k = vec![0f32; (num_blocks * shape.block_size) as usize * kv_row];
        let mut pool_v = pool_k.clone();
        for (b, &len) in lens.iter().enumerate() {
            for tok in 0..len {
                let phys = tables[b * shape.max_bt as usize + (tok / shape.block_size) as usize];
                let dst = (phys * shape.block_size + tok % shape.block_size) as usize * kv_row;
                let src = tok as usize * kv_row;
                pool_k[dst..dst + kv_row].copy_from_slice(&k[b][src..src + kv_row]);
                pool_v[dst..dst + kv_row].copy_from_slice(&v[b][src..src + kv_row]);
            }
        }
        Cache { shape, k, v, tables, pool_k, pool_v }
    }

    /// Run `kernel` over `q` and `rows` on the device; `pools` are the pool
    /// contents as uploaded (f32, or packed bf16 for the bf16 variant).
    fn run(&self, g: &Gpu, kernel: usize, per_kv_head: bool, q: &[f32], rows: &SparseRows, pools: (&[u32], &[u32])) -> Vec<f32> {
        let qb = g.storage_init("q", q);
        let pk = g.storage(pools.0.len() as u64);
        g.write(&pk, pools.0);
        let pv = g.storage(pools.1.len() as u64);
        g.write(&pv, pools.1);
        let bt = g.storage(self.tables.len() as u64);
        g.write(&bt, &self.tables);
        let sel = g.storage(rows.sel.len().max(1) as u64);
        g.write(&sel, &rows.sel);
        let ctx = g.storage(q.len() as u64);
        let bufs = SparseBuffers { q: &qb, pool_k: &pk, pool_v: &pv, block_tables: &bt, sel: &sel, scales: None, ctx: &ctx };
        let step = self.shape.step(g, kernel, if per_kv_head { Layout::PerKvHead } else { Layout::PerHead }, rows.rows, rows.stride, &bufs).unwrap();
        g.submit(&[], &[step]);
        g.read(&ctx, q.len())
    }

    fn f32_pools(&self) -> (Vec<u32>, Vec<u32>) {
        (self.pool_k.iter().map(|x| x.to_bits()).collect(), self.pool_v.iter().map(|x| x.to_bits()).collect())
    }

    fn attention_shape(&self) -> AttentionShape {
        AttentionShape { n_heads: self.shape.n_heads as usize, n_kv_heads: self.shape.n_kv_heads as usize, head_dim: self.shape.head_dim as usize }
    }

    /// The host oracle: `selection`'s rows over one sequence's contiguous K/V.
    fn oracle(&self, q: &[f32], selection: &Selection, k: &[f32], v: &[f32]) -> Vec<f32> {
        let scale = 1.0 / (self.shape.head_dim as f32).sqrt();
        sparse_attention(q, k, v, self.attention_shape(), scale, selection).unwrap()
    }
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Selecting `0..len` in ascending order is the dense walk, so the sparse
/// kernel must reproduce `paged_flash_decode` bit for bit: the indirection
/// may change which keys are read, never how they are reduced.
#[test]
fn selecting_every_key_in_order_reproduces_paged_flash_decode_bit_for_bit() {
    static PIPES: &[(&str, &str)] = &[("paged_flash_decode", kernels::PAGED_FLASH_DECODE), KERNEL];
    let Some(g) = barrier_gpu(PIPES) else { return };
    let shape = SparseAttention { n_heads: 4, n_kv_heads: 2, head_dim: 8, block_size: 4, max_bt: 26 };
    let lens = [1u32, 7, 8, 9, 41, 100];
    let mut rng = Rng::new(data::rng::random_seed());
    let cache = Cache::new(&mut rng, shape, &lens);
    let batch = lens.len() as u32;
    let q: Vec<f32> = (0..batch * shape.n_heads * shape.head_dim).map(|_| rng.next_gaussian() as f32).collect();

    let positions: Vec<u32> = lens.iter().map(|&t| t - 1).collect();
    let dense = crate::lsa::attend::dense_selection(&positions);
    let rows = SparseRows::pack(&dense, &(0..batch).collect::<Vec<_>>(), &lens).unwrap();
    let (pk, pv) = cache.f32_pools();
    let sparse = cache.run(&g, 1, false, &q, &rows, (&pk, &pv));

    let qb = g.storage_init("q", &q);
    let pkb = g.storage_init("pk", &cache.pool_k);
    let pvb = g.storage_init("pv", &cache.pool_v);
    let bt = g.storage(cache.tables.len() as u64);
    g.write(&bt, &cache.tables);
    let sl = g.storage(batch as u64);
    g.write(&sl, &lens);
    let ctx = g.storage(q.len() as u64);
    let step = g.dispatch(
        0,
        &[&qb, &pkb, &pvb, &bt, &sl, &ctx],
        &[batch, shape.n_heads, shape.n_kv_heads, shape.head_dim, shape.n_heads / shape.n_kv_heads, shape.block_size, shape.max_bt],
        gpu_core::Dispatch::Workgroups(batch * shape.n_heads),
    );
    g.submit(&[], &[step]);
    let flash = g.read(&ctx, q.len());

    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&sparse), bits(&flash), "sparse attention over every key differs from paged_flash_decode");
}

/// An LSA selection over real keys, many query rows per sequence (the
/// prefill shape), checked against the host oracle - and genuinely sparse.
#[test]
fn an_lsa_selection_matches_the_host_oracle() {
    static PIPES: &[(&str, &str)] = &[KERNEL];
    let Some(g) = barrier_gpu(PIPES) else { return };
    let shape = SparseAttention { n_heads: 4, n_kv_heads: 2, head_dim: 16, block_size: 8, max_bt: 32 };
    let layout = HeadLayout { n_heads: 4, n_kv_heads: 2, head_dim: 16, rotary: RotaryLayout::None };
    let cfg = LsaConfig { index_dim: layout.index_dim().unwrap(), pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 4, top_k: 12, sink: 2, window: 32, early: None };
    let lens = [3 * 32 + 20u32, 2 * 32 + 3];
    let mut rng = Rng::new(data::rng::random_seed());
    let cache = Cache::new(&mut rng, shape, &lens);
    let q_row = (shape.n_heads * shape.head_dim) as usize;

    let (pk, pv) = cache.f32_pools();
    let mut sparse_rows = 0usize;
    for (seq, &len) in lens.iter().enumerate() {
        let index = BucketIndex::build(cfg, &layout.index_keys(&cache.k[seq]).unwrap()).unwrap();
        let positions: Vec<u32> = (0..len).step_by(5).chain([len - 1]).collect();
        let q: Vec<f32> = (0..positions.len() * q_row).map(|_| rng.next_gaussian() as f32).collect();
        let selection = select(&index, &layout.index_queries(&q).unwrap(), &positions).unwrap();
        sparse_rows += positions.iter().enumerate().filter(|(r, &p)| (selection.lens[*r] as usize) < p as usize + 1).count();

        let table_rows = vec![seq as u32; positions.len()];
        let live = vec![len; positions.len()];
        let rows = SparseRows::pack(&selection, &table_rows, &live).unwrap();
        let device = cache.run(&g, 0, false, &q, &rows, (&pk, &pv));
        let host = cache.oracle(&q, &selection, &cache.k[seq], &cache.v[seq]);
        let w = worst(&device, &host);
        assert!(w < 1e-4, "sequence {seq}: device vs host oracle maxabs {w}");
    }
    assert!(sparse_rows > 0, "the fixture must leave some keys unread, or nothing sparse was tested");
}

/// A position past the sequence's live keys is refused when packing, and
/// masked by the kernel if it gets there anyway (a selection gone stale).
#[test]
fn a_position_the_sequence_does_not_hold_is_never_read() {
    static PIPES: &[(&str, &str)] = &[KERNEL];
    let shape = SparseAttention { n_heads: 2, n_kv_heads: 1, head_dim: 8, block_size: 4, max_bt: 8 };
    let lens = [20u32, 20];
    let selection = Selection { width: 3, ids: vec![2, 9, 15], lens: vec![3] };
    assert!(SparseRows::pack(&selection, &[0], &[12]).is_err(), "position 15 of a 12-key sequence must be refused");

    let Some(g) = barrier_gpu(PIPES) else { return };
    let mut rng = Rng::new(data::rng::random_seed());
    let cache = Cache::new(&mut rng, shape, &lens);
    let q: Vec<f32> = (0..(shape.n_heads * shape.head_dim) as usize).map(|_| rng.next_gaussian() as f32).collect();
    // Packed against 20 live keys, then the sequence is cut back to 12.
    let mut rows = SparseRows::pack(&selection, &[0], &[20]).unwrap();
    rows.sel[2] = 12; // the header's live keys
    let (pk, pv) = cache.f32_pools();
    let device = cache.run(&g, 0, false, &q, &rows, (&pk, &pv));
    let kept = Selection { width: 2, ids: vec![2, 9], lens: vec![2] };
    let host = cache.oracle(&q, &kept, &cache.k[0], &cache.v[0]);
    assert!(worst(&device, &host) < 1e-5, "the masked position was read");
}

/// The bf16-pool variant reads packed bf16 K/V; against the oracle over the
/// same bf16-rounded values it adds only reduction-order error.
#[test]
fn the_bf16_pool_variant_matches_the_oracle_on_rounded_kv() {
    let (n1, s1) = kernels::template::dtype_variant(KERNEL.0, KERNEL.1, "pool_k", Dtype::BF16).unwrap();
    let (name, src) = kernels::template::dtype_variant(n1, s1, "pool_v", Dtype::BF16).unwrap();
    let pipes: &'static [(&str, &str)] = Box::leak(vec![(name, src)].into_boxed_slice());
    let Some(g) = barrier_gpu(pipes) else { return };
    let shape = SparseAttention { n_heads: 4, n_kv_heads: 2, head_dim: 128, block_size: 16, max_bt: 8 };
    let lens = [100u32];
    let mut rng = Rng::new(data::rng::random_seed());
    let cache = Cache::new(&mut rng, shape, &lens);
    let q: Vec<f32> = (0..(shape.n_heads * shape.head_dim) as usize).map(|_| rng.next_gaussian() as f32).collect();
    let selection = Selection { width: 6, ids: vec![0, 1, 33, 50, 98, 99], lens: vec![6] };
    let rows = SparseRows::pack(&selection, &[0], &lens).unwrap();
    let device = cache.run(&g, 0, false, &q, &rows, (&crate::half::pack_bf16(&cache.pool_k), &crate::half::pack_bf16(&cache.pool_v)));
    let round = |x: &[f32]| x.iter().map(|&f| f32::from_bits((crate::half::f32_to_bf16(f) as u32) << 16)).collect::<Vec<_>>();
    let host = cache.oracle(&q, &selection, &round(&cache.k[0]), &round(&cache.v[0]));
    let w = worst(&device, &host);
    assert!(w < 1e-4, "bf16 pools vs oracle on rounded K/V maxabs {w}");
}

/// The per-KV-head kernel agrees with the per-head kernel at the group sizes
/// models use (1, 2, 3, 4, 5, 8), on ragged rows over two sequences, to float
/// reassociation: its tiles are 16 keys, the per-head kernel's 8.
#[test]
fn the_per_kv_head_kernel_matches_the_per_head_kernel() {
    for (nh, nkv, hd) in [(16u32, 8u32, 64u32), (8, 2, 64), (4, 4, 64), (16, 2, 64), (40, 8, 64), (12, 4, 64), (80, 8, 128), (40, 8, 128)] {
        let gqa = gqa_kernel(nh / nkv).unwrap();
        let pipes: &'static [(&str, &str)] = Box::leak(vec![KERNEL, gqa].into_boxed_slice());
        let Some(g) = barrier_gpu(pipes) else { return };
        let shape = SparseAttention { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: 8, max_bt: 32 };
        let lens = [200u32, 77];
        let mut rng = Rng::new(data::rng::random_seed());
        let cache = Cache::new(&mut rng, shape, &lens);
        let rows_per_seq = 9usize;
        let width = 40usize;
        let mut ids = vec![-1i32; 2 * rows_per_seq * width];
        let mut row_lens = Vec::new();
        let (mut tables, mut live) = (Vec::new(), Vec::new());
        for (seq, &len) in lens.iter().enumerate() {
            for r in 0..rows_per_seq {
                let row = seq * rows_per_seq + r;
                let n = 1 + (r * 7) % width;
                for (i, slot) in ids[row * width..][..n].iter_mut().enumerate() {
                    *slot = ((i * 13 + r * 5) % len as usize) as i32;
                }
                row_lens.push(n as u32);
                tables.push(seq as u32);
                live.push(len);
            }
        }
        let selection = Selection { width, ids, lens: row_lens };
        let rows = SparseRows::pack(&selection, &tables, &live).unwrap();
        let q: Vec<f32> = (0..rows.rows * nh * hd).map(|_| rng.next_gaussian() as f32).collect();
        let (pk, pv) = cache.f32_pools();
        let per_head = cache.run(&g, 0, false, &q, &rows, (&pk, &pv));
        let per_kv_head = cache.run(&g, 1, true, &q, &rows, (&pk, &pv));
        let w = worst(&per_kv_head, &per_head);
        assert!(w < 1e-5, "group {}: per-KV-head vs per-head maxabs {w}", nh / nkv);
    }
}

/// Random packed pool bytes (NaN codes avoided for fp8) and per-(slot, kv head)
/// scales, with the f32 pool they dequantise to.
fn packed_pool(rng: &mut Rng, slots: usize, n_kv: usize, hd: usize, kv: PackedKv) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let bytes: Vec<u8> = (0..slots * n_kv * hd)
        .map(|_| {
            let b = (rng.next_gaussian() * 40.0) as i32 as u8;
            if kv == PackedKv::Fp8 && b & 0x7f == 0x7f { 0x7e } else { b }
        })
        .collect();
    let scales: Vec<f32> = (0..slots * n_kv).map(|_| 0.01 + rng.next_gaussian().abs() as f32 * 0.02).collect();
    let words = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let deq = bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let v = match kv {
                PackedKv::Int8 => b as i8 as f32,
                PackedKv::Fp8 => crate::kv_tier::decode_e4m3(b),
            };
            v * scales[i / hd]
        })
        .collect();
    (words, scales, deq)
}

/// The packed-pool kernels stage the very values a host dequantisation gives,
/// then run the f32 kernel's arithmetic: bit-identical to it over the
/// dequantised pool, int8 and fp8, on ragged rows of two sequences.
#[test]
fn packed_pools_read_what_the_f32_kernel_reads_dequantised() {
    for (kv, (nh, nkv, hd)) in [PackedKv::Int8, PackedKv::Fp8].into_iter().flat_map(|kv| [(8u32, 4u32, 64u32), (80, 8, 128)].map(move |g| (kv, g))) {
        let pipes: &'static [(&str, &str)] = Box::leak(vec![gqa_kernel(nh / nkv).unwrap(), gqa_kernel_kv(nh / nkv, kv).unwrap()].into_boxed_slice());
        let Some(g) = barrier_gpu(pipes) else { return };
        let shape = SparseAttention { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: 8, max_bt: 32 };
        let mut rng = Rng::new(data::rng::random_seed());
        let slots = (2 * 32 * 8) as usize;
        let (words_k, scales_k, deq_k) = packed_pool(&mut rng, slots, nkv as usize, hd as usize, kv);
        let (words_v, scales_v, deq_v) = packed_pool(&mut rng, slots, nkv as usize, hd as usize, kv);
        let tables: Vec<u32> = (0..2u32).flat_map(|b| (0..32u32).map(move |i| b + i * 2)).collect();
        let width = 50usize;
        let (lens, rows_per) = ([180u32, 90], 6usize);
        let mut ids = vec![-1i32; 2 * rows_per * width];
        let (mut row_lens, mut table_rows, mut live) = (Vec::new(), Vec::new(), Vec::new());
        for (seq, &len) in lens.iter().enumerate() {
            for r in 0..rows_per {
                let row = seq * rows_per + r;
                let n = 1 + (r * 9) % width;
                for (i, slot) in ids[row * width..][..n].iter_mut().enumerate() {
                    *slot = ((i * 11 + r * 3) % len as usize) as i32;
                }
                row_lens.push(n as u32);
                table_rows.push(seq as u32);
                live.push(len);
            }
        }
        let rows = SparseRows::pack(&Selection { width, ids, lens: row_lens }, &table_rows, &live).unwrap();
        let q: Vec<f32> = (0..rows.rows * nh * hd).map(|_| rng.next_gaussian() as f32).collect();
        let run = |kernel: usize, pk: &gpu_core::DeviceBuffer, pv: &gpu_core::DeviceBuffer, scales: Option<(&gpu_core::DeviceBuffer, &gpu_core::DeviceBuffer)>| {
            let qb = g.storage_init("q", &q);
            let bt = g.storage(tables.len() as u64);
            g.write(&bt, &tables);
            let sel = g.storage(rows.sel.len() as u64);
            g.write(&sel, &rows.sel);
            let ctx = g.storage(q.len() as u64);
            let b = SparseBuffers { q: &qb, pool_k: pk, pool_v: pv, block_tables: &bt, sel: &sel, scales, ctx: &ctx };
            g.submit(&[], &[shape.step(&g, kernel, Layout::PerKvHead, rows.rows, rows.stride, &b).unwrap()]);
            g.read(&ctx, q.len())
        };
        let (dk, dv) = (g.storage_init("dk", &deq_k), g.storage_init("dv", &deq_v));
        let f32_out = run(0, &dk, &dv, None);
        let (pk, pv) = (g.storage(words_k.len() as u64), g.storage(words_v.len() as u64));
        g.write(&pk, &words_k);
        g.write(&pv, &words_v);
        let (sk, sv) = (g.storage_init("sk", &scales_k), g.storage_init("sv", &scales_v));
        let packed_out = run(1, &pk, &pv, Some((&sk, &sv)));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert!(f32_out.iter().any(|x| *x != 0.0));
        assert_eq!(bits(&packed_out), bits(&f32_out), "{kv:?} pool: the packed kernel differs from the f32 kernel over the dequantised pool");
    }
}

/// The whole-row kernel (one workgroup per query row, every head) agrees with
/// the f32 per-KV-head kernel over the dequantised pool - to float rounding, as
/// it applies each key's scale after the dot product rather than per element -
/// at the model geometries it is built for, int8 and fp8.
#[test]
fn the_whole_row_kernel_matches_over_the_dequantised_pool() {
    for (nh, nkv, hd) in [(16u32, 8u32, 128u32), (32, 8, 128), (8, 2, 64), (40, 8, 128), (80, 8, 128), (64, 8, 128)] {
        for kv in [PackedKv::Int8, PackedKv::Fp8] {
            let pipes: &'static [(&str, &str)] = Box::leak(vec![gqa_kernel(nh / nkv).unwrap(), rows_kernel_kv(nh, nkv, hd, kv).unwrap()].into_boxed_slice());
            let Some(g) = barrier_gpu(pipes) else { return };
            let shape = SparseAttention { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: 8, max_bt: 32 };
            let mut rng = Rng::new(data::rng::random_seed());
            let slots = (2 * 32 * 8) as usize;
            let (words_k, scales_k, deq_k) = packed_pool(&mut rng, slots, nkv as usize, hd as usize, kv);
            let (words_v, scales_v, deq_v) = packed_pool(&mut rng, slots, nkv as usize, hd as usize, kv);
            let tables: Vec<u32> = (0..2u32).flat_map(|b| (0..32u32).map(move |i| b + i * 2)).collect();
            let width = 45usize;
            let (lens, rows_per) = ([200u32, 77], 7usize);
            let mut ids = vec![-1i32; 2 * rows_per * width];
            let (mut row_lens, mut table_rows, mut live) = (Vec::new(), Vec::new(), Vec::new());
            for (seq, &len) in lens.iter().enumerate() {
                for r in 0..rows_per {
                    let row = seq * rows_per + r;
                    let n = 1 + (r * 13) % width;
                    for (i, slot) in ids[row * width..][..n].iter_mut().enumerate() {
                        *slot = ((i * 7 + r * 5) % len as usize) as i32;
                    }
                    row_lens.push(n as u32);
                    table_rows.push(seq as u32);
                    live.push(len);
                }
            }
            let rows = SparseRows::pack(&Selection { width, ids, lens: row_lens }, &table_rows, &live).unwrap();
            let q: Vec<f32> = (0..rows.rows * nh * hd).map(|_| rng.next_gaussian() as f32).collect();
            let run = |kernel: usize, layout: Layout, pk: &gpu_core::DeviceBuffer, pv: &gpu_core::DeviceBuffer, scales: Option<(&gpu_core::DeviceBuffer, &gpu_core::DeviceBuffer)>| {
                let qb = g.storage_init("q", &q);
                let bt = g.storage(tables.len() as u64);
                g.write(&bt, &tables);
                let sel = g.storage(rows.sel.len() as u64);
                g.write(&sel, &rows.sel);
                let ctx = g.storage(q.len() as u64);
                let b = SparseBuffers { q: &qb, pool_k: pk, pool_v: pv, block_tables: &bt, sel: &sel, scales, ctx: &ctx };
                g.submit(&[], &[shape.step(&g, kernel, layout, rows.rows, rows.stride, &b).unwrap()]);
                g.read(&ctx, q.len())
            };
            let (dk, dv) = (g.storage_init("dk", &deq_k), g.storage_init("dv", &deq_v));
            let want = run(0, Layout::PerKvHead, &dk, &dv, None);
            let (pk, pv) = (g.storage(words_k.len() as u64), g.storage(words_v.len() as u64));
            g.write(&pk, &words_k);
            g.write(&pv, &words_v);
            let (sk, sv) = (g.storage_init("sk", &scales_k), g.storage_init("sv", &scales_v));
            let got = run(1, Layout::PerRow, &pk, &pv, Some((&sk, &sv)));
            let w = worst(&got, &want);
            assert!(w < 1e-4, "{nh}/{nkv}/{hd} {kv:?}: whole-row kernel vs per-KV-head maxabs {w}");
        }
    }
}

/// The per-KV-head f32 kernel at group 10 (Ripple's 80 heads over 8 KV heads of 128) and group 5 (40 over 8)
/// against the scalar oracle `lsa::attend`, on LSA selections that leave keys unread.
#[test]
fn the_wide_group_kernels_match_the_scalar_oracle() {
    for (nh, nkv, hd) in [(80u32, 8u32, 128u32), (40, 8, 128)] {
        let gqa = gqa_kernel(nh / nkv).unwrap();
        let pipes: &'static [(&str, &str)] = Box::leak(vec![gqa].into_boxed_slice());
        let Some(g) = barrier_gpu(pipes) else { return };
        let shape = SparseAttention { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: 8, max_bt: 32 };
        let layout = HeadLayout { n_heads: nh as usize, n_kv_heads: nkv as usize, head_dim: hd as usize, rotary: RotaryLayout::None };
        let cfg = LsaConfig { index_dim: layout.index_dim().unwrap(), pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 4, top_k: 12, sink: 2, window: 32, early: None };
        let lens = [3 * 32 + 20u32, 2 * 32 + 3];
        let mut rng = Rng::new(data::rng::random_seed());
        let cache = Cache::new(&mut rng, shape, &lens);
        let q_row = (nh * hd) as usize;
        let (pk, pv) = cache.f32_pools();
        let mut sparse_rows = 0usize;
        for (seq, &len) in lens.iter().enumerate() {
            let index = BucketIndex::build(cfg, &layout.index_keys(&cache.k[seq]).unwrap()).unwrap();
            let positions: Vec<u32> = (0..len).step_by(7).chain([len - 1]).collect();
            let q: Vec<f32> = (0..positions.len() * q_row).map(|_| rng.next_gaussian() as f32).collect();
            let selection = select(&index, &layout.index_queries(&q).unwrap(), &positions).unwrap();
            sparse_rows += positions.iter().enumerate().filter(|(r, &p)| (selection.lens[*r] as usize) < p as usize + 1).count();
            let rows = SparseRows::pack(&selection, &vec![seq as u32; positions.len()], &vec![len; positions.len()]).unwrap();
            let device = cache.run(&g, 0, true, &q, &rows, (&pk, &pv));
            let host = cache.oracle(&q, &selection, &cache.k[seq], &cache.v[seq]);
            let w = worst(&device, &host);
            assert!(w < 1e-4, "{nh}/{nkv}/{hd} sequence {seq}: device vs host oracle maxabs {w}");
        }
        assert!(sparse_rows > 0, "the fixture must leave some keys unread");
    }
}

/// Every kernel variant an engine can register is valid WGSL, and its workgroup memory fits the core-compute
/// profile's 16 KiB. Runs on every backend: the cooperative kernels cannot execute on the CPU JIT, but they can be
/// parsed and validated, which catches a bad template before a device ever compiles it.
#[test]
fn every_registered_sparse_variant_is_valid_wgsl_within_workgroup_memory() {
    let mut variants: Vec<(&'static str, &'static str)> = Vec::new();
    for group in [1u32, 2, 4, 5, 8, 10] {
        variants.push(gqa_kernel(group).unwrap());
        for kv in [PackedKv::Int8, PackedKv::Fp8] {
            variants.push(gqa_kernel_kv(group, kv).unwrap());
        }
    }
    for (nh, nkv, hd) in [(16u32, 8u32, 128u32), (32, 8, 128), (8, 2, 64), (40, 8, 128), (64, 8, 128), (80, 8, 128), (48, 8, 128), (24, 4, 96)] {
        for kv in [PackedKv::Int8, PackedKv::Fp8] {
            variants.push(rows_kernel_kv(nh, nkv, hd, kv).unwrap());
        }
    }
    for (name, src) in variants {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let shared: u32 = module.global_variables.iter().filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup).map(|(_, v)| module.types[v.ty].inner.size(module.to_ctx())).sum();
        eprintln!("{name}: {shared} bytes of workgroup memory");
        assert!(shared <= 16 * 1024, "{name}: {shared} bytes of workgroup memory");
    }
}

/// The whole-row workgroup keeps the shapes already validated and widens it for head counts that do not divide 64.
#[test]
fn the_whole_row_workgroup_is_geometry_derived() {
    assert_eq!(rows_geometry(16, 8, 128).unwrap(), (64, 8));
    assert_eq!(rows_geometry(32, 8, 128).unwrap(), (64, 8));
    assert_eq!(rows_geometry(8, 2, 64).unwrap(), (64, 8));
    assert_eq!(rows_geometry(40, 8, 128).unwrap(), (80, 8));
    assert_eq!(rows_geometry(80, 8, 128).unwrap(), (160, 4));
    assert!(rows_geometry(300, 10, 128).is_err(), "no workgroup of at most 256 threads holds 300 heads");
    assert!(rows_geometry(16, 8, 130).is_err() && rows_geometry(15, 8, 128).is_err());
}
