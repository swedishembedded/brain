// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::device::*;
use super::*;
use data::rng::Lcg;

fn barrier_gpu(pipes: &'static [(&'static str, &'static str)]) -> Option<gpu_core::Gpu> {
    let g = gpu_core::testgpu::dev(pipes);
    if g.caps().workgroup_reductions {
        Some(g)
    } else {
        brain_testutil::skip_unavailable("LSA device kernels: this backend does not run workgroup-barrier kernels (GPU-only)");
        None
    }
}

fn unit_rows(rng: &mut Lcg, rows: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        let row: Vec<f32> = (0..dim).map(|_| rng.signed()).collect();
        out.extend(crate::hostmath::l2_normalize(&row));
    }
    out
}

/// Half-split RoPE of every head of `row` at `pos`, the forward kernels' angle.
fn rotate(row: &mut [f32], head_dim: usize, pos: u32, base: f32) {
    let half = head_dim / 2;
    for head in row.chunks_exact_mut(head_dim) {
        for m in 0..half {
            let angle = pos as f32 * base.powf(-((2 * m) as f32) / head_dim as f32);
            let (sn, cs) = angle.sin_cos();
            let (x0, x1) = (head[m], head[m + half]);
            head[m] = x0 * cs - x1 * sn;
            head[m + half] = x1 * cs + x0 * sn;
        }
    }
}

/// Index vectors from rotated heads scattered over a buffer equal the host
/// index space's from the same heads before rotation, then projected - to a
/// narrow index, to one wider than a workgroup, or not at all (the native
/// index space).
#[test]
fn index_vectors_unrotate_sum_groups_and_project() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let (nh, nkv, hd) = (8usize, 2usize, 16usize);
    for (dim, identity) in [(12usize, false), (100, false), (nkv * hd, true)] {
        index_vectors_case(&g, nh, nkv, hd, dim, identity);
    }
}

fn index_vectors_case(g: &gpu_core::Gpu, nh: usize, nkv: usize, hd: usize, dim: usize, identity: bool) {
    let base = 10000.0f32;
    let heads = IndexHeads { n_kv_heads: nkv as u32, head_dim: hd as u32, rope: Rope::Base(base), index_dim: dim as u32, identity };
    let unused = g.storage(1);
    let mut rng = Lcg::random();
    let w: Vec<f32> = (0..nkv * hd * dim).map(|_| rng.signed()).collect();
    let project = |v: &[f32]| -> Vec<f32> {
        let rows = v.len() / (nkv * hd);
        let mut out = vec![0f32; rows * dim];
        for r in 0..rows {
            for c in 0..dim {
                out[r * dim + c] = if identity { v[r * nkv * hd + c] } else { (0..nkv * hd).map(|i| v[r * nkv * hd + i] * w[i * dim + c]).sum() };
            }
        }
        out.chunks_exact(dim).flat_map(crate::hostmath::l2_normalize).collect()
    };
    for (n_heads, rows) in [(nh, 5usize), (nkv, 7)] {
        let width = n_heads * hd;
        let pre: Vec<f32> = (0..rows * width).map(|_| rng.signed()).collect();
        let positions: Vec<u32> = (0..rows as u32).map(|r| r * 37 + 3).collect();
        // Rows land at scattered, non-contiguous offsets of a larger buffer.
        let offsets: Vec<u32> = (0..rows as u32).map(|r| (rows as u32 - 1 - r) * width as u32 * 2 + 5).collect();
        let mut x = vec![0f32; rows * width * 2 + 5];
        for r in 0..rows {
            let mut row = pre[r * width..][..width].to_vec();
            rotate(&mut row, hd, positions[r], base);
            x[offsets[r] as usize..][..width].copy_from_slice(&row);
        }
        let want = if n_heads == nh {
            let summed: Vec<f32> = pre.chunks_exact(width).flat_map(|row| {
                let mut s = vec![0f32; nkv * hd];
                for h in 0..nh {
                    for c in 0..hd {
                        s[(h / (nh / nkv)) * hd + c] += row[h * hd + c];
                    }
                }
                s
            }).collect();
            project(&summed)
        } else {
            project(&pre)
        };
        let xb = g.storage_init("x", &x);
        let off = g.storage(rows as u64);
        g.write(&off, &offsets);
        let pos = g.storage(rows as u64);
        g.write(&pos, &positions);
        let wb = g.storage_init("w", &w);
        let out = g.storage((rows * dim) as u64);
        let step = heads.step(g, 0, n_heads as u32, rows as u32, &IndexVectorBuffers { x: &xb, row_off: &off, pos: &pos, w: &wb, out: &out, inv_freq: &unused, out0: 0 }).unwrap();
        g.submit(&[], &[step]);
        let got = g.read(&out, rows * dim);
        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4, "{n_heads}-head rows into {dim} channels (identity {identity}): device index vectors differ from the host by {worst}");
    }
}

/// The device read equals `read_scores` given the device's own scores and
/// built-page count, over pages uploaded in scrambled order, with rows whose
/// index lags one block behind.
#[test]
fn the_device_read_matches_the_host_read() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let small = LsaConfig { index_dim: 16, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 5, top_k: 13, sink: 2, window: 32, early: Some(EarlyBudget { position: 70, top_k: 6, top_p: 2 }) };
    // A narrow index, one wider than a workgroup, and rows that see more
    // buckets than a workgroup has threads several times over.
    read_case(&g, small, 6);
    read_case(&g, LsaConfig { index_dim: 160, ..small }, 6);
    read_case(&g, LsaConfig { bucket_size: 2, bucket_cap: 4, top_p: 24, top_k: 60, early: None, ..small }, 14);
}

fn read_case(g: &gpu_core::Gpu, cfg: LsaConfig, blocks: usize) {
    let mut rng = Lcg::random();
    let t = blocks * cfg.cluster_block + 11;
    let index = BucketIndex::build(cfg, &unit_rows(&mut rng, t, cfg.index_dim)).unwrap();
    let blocks = index.closed_blocks();
    let max_pages = 16u32;
    let pool = IndexPool::new(g, cfg, max_pages, 1).unwrap();
    // Block b lives in page (b * 3 + 1) % max_pages.
    let page_of = |b: usize| ((b * 3 + 1) % max_pages as usize) as u32;
    let bpb = cfg.buckets_per_block();
    for b in 0..blocks {
        let block = ClosedBlock {
            centroids: index.centroids()[b * bpb * cfg.index_dim..][..bpb * cfg.index_dim].to_vec(),
            members: index.members()[b * bpb * cfg.bucket_cap..][..bpb * cfg.bucket_cap].to_vec(),
            sizes: index.sizes()[b * bpb..][..bpb].to_vec(),
        };
        pool.write(g, page_of(b), 0, &block).unwrap();
    }
    let mut positions: Vec<u32> = vec![0, 1, 31, 32, 40, 63, 64, 69, 70, 100, 150, 191, 192, t as u32 - 1, t as u32 - 2];
    positions.extend((0..12).map(|_| rng.next_u32() % t as u32));
    let rows = positions.len();
    // Every third row's index lags one block behind its closed blocks.
    let built: Vec<usize> = positions.iter().enumerate().map(|(r, &p)| {
        let closed = (p as usize + 1) / cfg.cluster_block;
        if r % 3 == 2 { closed.saturating_sub(1) } else { closed }
    }).collect();
    let geom = ReadGeometry { cfg, max_pages };
    let mut tab = vec![0u32; rows * (max_pages as usize + 1)];
    for r in 0..rows {
        let row = &mut tab[r * (max_pages as usize + 1)..][..max_pages as usize + 1];
        row[0] = built[r] as u32;
        for b in 0..built[r] {
            row[1 + b] = page_of(b);
        }
    }
    let iq_h = unit_rows(&mut rng, rows, cfg.index_dim);
    let seqlen: Vec<u32> = positions.iter().map(|p| p + 1).collect();
    let iq = g.storage_init("iq", &iq_h);
    let page_tab = g.storage(tab.len() as u64);
    g.write(&page_tab, &tab);
    let sl = g.storage(rows as u64);
    g.write(&sl, &seqlen);
    let scores = g.storage((rows as u64) * geom.score_stride() as u64);
    let width = geom.row_stride() as usize;
    let sel = g.storage((rows * width) as u64);
    let stats = g.storage(ReadStats::WORDS);
    let bufs = ReadBuffers { iq: &iq, centroids: &pool.centroids, members: &pool.members, page_tab: &page_tab, seqlen: &sl, scores: &scores, sel: &sel, row0: 0, stats: &stats };
    let step = geom.step(g, 1, rows as u32, &bufs).unwrap();
    // Twice: the counters accumulate across reads.
    g.submit(&[], &[step.clone(), step]);
    let dev_scores = g.read(&scores, rows * geom.score_stride() as usize);
    let dev_sel: Vec<u32> = g.read(&sel, rows * width).iter().map(|x| x.to_bits()).collect();

    let counted = ReadStats::read(g, &stats);
    let selected: u64 = (0..rows).map(|r| u64::from(dev_sel[r * width])).sum();
    let live: u64 = seqlen.iter().map(|&n| u64::from(n)).sum();
    assert_eq!(counted, ReadStats { rows: 2 * rows as u64, selected_keys: 2 * selected, causal_keys: 2 * live }, "the read's counters");

    let stride = geom.score_stride() as usize;
    let mut lagged = 0;
    for r in 0..rows {
        let host = read_scores(&index, &dev_scores[r * stride..][..stride], stride, &positions[r..r + 1], built[r]).unwrap();
        let want: Vec<u32> = host.row(0).iter().map(|&j| j as u32).collect();
        let row = &dev_sel[r * width..][..width];
        let n = row[0] as usize;
        assert_eq!(row[1..3], [r as u32, seqlen[r]], "row {r} header");
        assert_eq!(&row[3..3 + n], &want[..], "row {r} (position {}, {} pages built) selects differently", positions[r], built[r]);
        lagged += usize::from(built[r] < (positions[r] as usize + 1) / cfg.cluster_block);
    }
    assert!(lagged > 0, "the fixture must exercise a lagging index");
}

/// Index keys read from a packed pool by token slot equal the f32 kernel's
/// over the dequantised pool, bit for bit, int8 and fp8.
#[test]
fn packed_index_keys_match_the_dequantised_pool() {
    use crate::paged_sparse::PackedKv;
    for kv in [PackedKv::Int8, PackedKv::Fp8] {
        let pipes: &'static [(&str, &str)] = Box::leak(vec![KERNELS[0], index_keys_kernel(kv).unwrap()].into_boxed_slice());
        let Some(g) = barrier_gpu(pipes) else { return };
        let (nkv, hd, dim, slots) = (2usize, 16usize, 8usize, 40usize);
        let heads = IndexHeads { n_kv_heads: nkv as u32, head_dim: hd as u32, rope: Rope::Base(10000.0), index_dim: dim as u32, identity: false };
        let unused = g.storage(1);
        let mut rng = Lcg::random();
        let bytes: Vec<u8> = (0..slots * nkv * hd).map(|_| {
            let b = rng.next_u32() as u8;
            if kv == PackedKv::Fp8 && b & 0x7f == 0x7f { 0x7e } else { b }
        }).collect();
        let scales: Vec<f32> = (0..slots * nkv).map(|_| 0.01 + rng.unit() * 0.05).collect();
        let deq: Vec<f32> = bytes.iter().enumerate().map(|(i, &b)| {
            let v = match kv { PackedKv::Int8 => b as i8 as f32, PackedKv::Fp8 => crate::kv_tier::decode_e4m3(b) };
            v * scales[i / hd]
        }).collect();
        let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        let key_slots: Vec<u32> = vec![7, 0, 33, 12, 39, 5];
        let pos: Vec<u32> = vec![100, 3, 250, 17, 4000, 64];
        let rows = key_slots.len();
        let w: Vec<f32> = (0..nkv * hd * dim).map(|_| rng.signed()).collect();
        let (wb, posb) = (g.storage_init("w", &w), g.storage(rows as u64));
        g.write(&posb, &pos);
        let f32_pool = g.storage_init("pool", &deq);
        let off = g.storage(rows as u64);
        g.write(&off, &key_slots.iter().map(|s| s * (nkv * hd) as u32).collect::<Vec<_>>());
        let out_f32 = g.storage((rows * dim) as u64);
        let s1 = heads.step(&g, 0, nkv as u32, rows as u32, &IndexVectorBuffers { x: &f32_pool, row_off: &off, pos: &posb, w: &wb, out: &out_f32, inv_freq: &unused, out0: 0 }).unwrap();
        let packed = g.storage(words.len() as u64);
        g.write(&packed, &words);
        let (sb, slotb, out_p) = (g.storage_init("s", &scales), g.storage(rows as u64), g.storage((rows * dim) as u64));
        g.write(&slotb, &key_slots);
        let s2 = heads.step_packed_keys(&g, 1, rows as u32, &PackedKeyBuffers { pool: &packed, scales: &sb, slots: &slotb, pos: &posb, w: &wb, out: &out_p, inv_freq: &unused, out0: 0 }).unwrap();
        g.submit(&[], &[s1, s2]);
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(g.read(&out_p, rows * dim)), bits(g.read(&out_f32, rows * dim)), "{kv:?} index keys differ from the dequantised f32 pool's");
    }
}

/// Heads rotated by a per-channel inverse-frequency table (YaRN), with its
/// uniform attention factor, un-rotate to the index vectors of the unrotated
/// heads.
#[test]
fn index_vectors_unrotate_a_frequency_table() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let (nkv, hd, dim, rows) = (2usize, 16usize, 8usize, 6usize);
    let heads = IndexHeads { n_kv_heads: nkv as u32, head_dim: hd as u32, rope: Rope::Table, index_dim: dim as u32, identity: false };
    let mut rng = Lcg::random();
    let table: Vec<f32> = (0..hd / 2).map(|m| 0.7f32.powi(m as i32) * (0.5 + rng.unit())).collect();
    let factor = 1.3f32;
    let w: Vec<f32> = (0..nkv * hd * dim).map(|_| rng.signed()).collect();
    let pre: Vec<f32> = (0..rows * nkv * hd).map(|_| rng.signed()).collect();
    let positions: Vec<u32> = (0..rows as u32).map(|r| 1000 + 977 * r).collect();
    let mut rotated = pre.clone();
    for (r, row) in rotated.chunks_exact_mut(nkv * hd).enumerate() {
        for head in row.chunks_exact_mut(hd) {
            for m in 0..hd / 2 {
                let (sn, cs) = (positions[r] as f32 * table[m]).sin_cos();
                let (x0, x1) = (head[m], head[m + hd / 2]);
                head[m] = factor * (x0 * cs - x1 * sn);
                head[m + hd / 2] = factor * (x1 * cs + x0 * sn);
            }
        }
    }
    let want: Vec<f32> = pre
        .chunks_exact(nkv * hd)
        .flat_map(|row| crate::hostmath::l2_normalize(&(0..dim).map(|c| (0..nkv * hd).map(|i| row[i] * w[i * dim + c]).sum::<f32>()).collect::<Vec<_>>()))
        .collect();
    let (xb, wb, tb) = (g.storage_init("x", &rotated), g.storage_init("w", &w), g.storage_init("t", &table));
    let off = g.storage(rows as u64);
    g.write(&off, &(0..rows as u32).map(|r| r * (nkv * hd) as u32).collect::<Vec<_>>());
    let pos = g.storage(rows as u64);
    g.write(&pos, &positions);
    let out = g.storage((rows * dim) as u64);
    let step = heads.step(&g, 0, nkv as u32, rows as u32, &IndexVectorBuffers { x: &xb, row_off: &off, pos: &pos, w: &wb, out: &out, inv_freq: &tb, out0: 0 }).unwrap();
    g.submit(&[], &[step]);
    let got = g.read(&out, rows * dim);
    let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    // Angles reach ~9000 rad, where f32 sin/cos of the host and of the device
    // differ in the fourth decimal; the un-rotation itself adds nothing.
    assert!(worst < 1e-3, "table un-rotation differs from the unrotated heads by {worst}");
}

/// The device Ward builds the page the host reference builds - identical
/// buckets, members and centroids - on keys gathered around well-separated
/// directions, and a valid partition (every key once, sizes within the cap,
/// every bucket used) on random keys.
#[test]
fn the_device_ward_builds_the_host_page() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let cfg = LsaConfig::default();
    let d = cfg.index_dim;
    let mut rng = Lcg::random();
    let mut pool = IndexPool::new(&g, cfg, 4, 1).unwrap();
    pool.keep_sums(&g);
    let scratch = g.storage(IndexPool::ward_scratch_words(&cfg));
    let bpb = cfg.buckets_per_block();
    let run = |keys: &[f32], page: u32, base: u32| -> (Vec<f32>, Vec<u32>) {
        let kb = g.storage_init("keys", keys);
        let step = pool.ward_step(&g, 2, &kb, &scratch, page, base, 0).unwrap();
        g.submit(&[], &[step]);
        let c = g.read(&pool.centroids, 4 * bpb * d);
        let m: Vec<u32> = g.read(&pool.members, 4 * bpb * cfg.bucket_cap).iter().map(|x| x.to_bits()).collect();
        (c[page as usize * bpb * d..][..bpb * d].to_vec(), m[page as usize * bpb * cfg.bucket_cap..][..bpb * cfg.bucket_cap].to_vec())
    };
    // Each direction is shared by exactly two pools, so the merges are
    // unambiguous: every pool's cheapest partner is its twin, by far.
    let directions = cfg.pools_per_block() - bpb;
    let centers = unit_rows(&mut rng, directions, d);
    let mut keys = Vec::with_capacity(cfg.cluster_block * d);
    for k in 0..cfg.cluster_block {
        let center = &centers[((k / cfg.pool_size) % directions) * d..][..d];
        let row: Vec<f32> = center.iter().map(|c| c + 0.01 * rng.signed()).collect();
        keys.extend(crate::hostmath::l2_normalize(&row));
    }
    let base = 3 * cfg.cluster_block as u32;
    let host = cluster_block(&cfg, &keys, base as usize).unwrap();
    let (centroids, members) = run(&keys, 2, base);
    let host_members: Vec<u32> = host.members.iter().map(|&m| if m < 0 { NO_KEY } else { m as u32 }).collect();
    assert_eq!(members, host_members, "device buckets differ from the host's on unambiguous keys");
    let worst = centroids.iter().zip(&host.centroids).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-4, "centroids differ by {worst}");
    // Each bucket's key sum, kept for a tree's node buckets.
    let sums = g.read(pool.sums.as_ref().unwrap(), 4 * bpb * d);
    for (b, row) in host_members.chunks_exact(cfg.bucket_cap).enumerate() {
        let mut want = vec![0f32; d];
        for &m in row.iter().take_while(|&&m| m != NO_KEY) {
            for (w, x) in want.iter_mut().zip(&keys[(m - base) as usize * d..][..d]) {
                *w += x;
            }
        }
        let got = &sums[(2 * bpb + b) * d..][..d];
        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4, "bucket {b}'s key sum differs by {worst}");
    }

    // Spread keys, and keys all near one direction as a model's keys often
    // are (costs nearly tied): every key lands in exactly one bucket.
    let shared = unit_rows(&mut rng, 1, d);
    let correlated: Vec<f32> = (0..cfg.cluster_block)
        .flat_map(|_| {
            let row: Vec<f32> = shared.iter().map(|c| c + 0.08 * rng.signed()).collect();
            crate::hostmath::l2_normalize(&row)
        })
        .collect();
    // Repeated tokens give repeated keys: exact ties everywhere.
    let few = unit_rows(&mut rng, 3, d);
    let repeated: Vec<f32> = (0..cfg.cluster_block).flat_map(|_| few[(rng.next_u32() % 3) as usize * d..][..d].to_vec()).collect();
    for (name, keys) in [("spread", unit_rows(&mut rng, cfg.cluster_block, d)), ("correlated", correlated), ("repeated", repeated)] {
        let (_, members) = run(&keys, 0, 0);
        let mut seen = vec![0u32; cfg.cluster_block];
        for row in members.chunks_exact(cfg.bucket_cap) {
            let held: Vec<u32> = row.iter().copied().take_while(|&m| m != NO_KEY).collect();
            assert!(!held.is_empty(), "{name} keys: an empty bucket");
            assert!(held.windows(2).all(|w| w[0] < w[1]), "{name} keys: members out of order");
            for &m in &held {
                seen[m as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "{name} keys: {} keys are in no bucket or in two", seen.iter().filter(|&&n| n != 1).count());
    }
}

/// Training's clustering: every block of two sequences in one dispatch, keys
/// wider than a 64-channel chunk, each block's members numbered from its own
/// position in its sequence - the host's buckets for each block.
#[test]
fn the_device_ward_clusters_every_block_of_wide_keys() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let cfg = LsaConfig { index_dim: 160, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 4, top_k: 16, sink: 2, window: 32, early: None };
    let (d, cb, seq_blocks, seqs) = (cfg.index_dim, cfg.cluster_block, 3u32, 2u32);
    let slots = seq_blocks * seqs;
    let mut rng = Lcg::random();
    let pool = IndexPool::new(&g, cfg, 1, slots).unwrap();
    let scratch = g.storage(slots as u64 * IndexPool::ward_scratch_words(&cfg));
    // Twin pools per direction, so the merges are unambiguous.
    let directions = cfg.pools_per_block() - cfg.buckets_per_block();
    let mut keys = Vec::new();
    for _ in 0..slots {
        let centers = unit_rows(&mut rng, directions, d);
        for k in 0..cb {
            let row: Vec<f32> = centers[((k / cfg.pool_size) % directions) * d..][..d].iter().map(|c| c + 0.01 * rng.signed()).collect();
            keys.extend(crate::hostmath::l2_normalize(&row));
        }
    }
    let kb = g.storage_init("keys", &keys);
    g.submit(&[], &[pool.ward_step(&g, 2, &kb, &scratch, 0, 0, seq_blocks).unwrap()]);
    let bpb = cfg.buckets_per_block();
    let centroids = g.read(&pool.centroids, slots as usize * bpb * d);
    let members: Vec<u32> = g.read(&pool.members, slots as usize * bpb * cfg.bucket_cap).iter().map(|x| x.to_bits()).collect();
    for w in 0..slots as usize {
        let base = (w % seq_blocks as usize) * cb;
        let host = cluster_block(&cfg, &keys[w * cb * d..][..cb * d], base).unwrap();
        let host_members: Vec<u32> = host.members.iter().map(|&m| if m < 0 { NO_KEY } else { m as u32 }).collect();
        assert_eq!(&members[w * bpb * cfg.bucket_cap..][..bpb * cfg.bucket_cap], &host_members[..], "slot {w}: device buckets differ from the host's");
        let worst = centroids[w * bpb * d..][..bpb * d].iter().zip(&host.centroids).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4, "slot {w}: centroids differ by {worst}");
    }
}

/// The device tree read selects what the host tree selects: three levels of
/// node buckets in a scrambled node pool, beaming from the top and from the
/// lowest level of at most `start_nodes` node buckets, over queries at every
/// growth stage.
#[test]
fn the_device_tree_read_matches_the_host_tree() {
    use super::tree::{build_hierarchy, growth_at, select_tree, TreeConfig};
    use super::tree_device::{NodePool, TreeReadBuffers, TreeReadGeometry, KERNEL};
    static PIPES: [(&str, &str); 1] = [KERNEL];
    let Some(g) = barrier_gpu(&PIPES) else { return };
    let cfg = LsaConfig { top_p: 24, top_k: 96, ..LsaConfig::default() };
    let d = cfg.index_dim;
    let mut rng = Lcg::random();
    let t = 12 * cfg.cluster_block + 100;
    let centers = unit_rows(&mut rng, 12, d);
    let keys: Vec<f32> = (0..t)
        .flat_map(|_| {
            let c = &centers[(rng.next_u32() % 12) as usize * d..][..d];
            let row: Vec<f32> = c.iter().map(|x| x + 0.35 * rng.signed()).collect();
            crate::hostmath::l2_normalize(&row)
        })
        .collect();
    let index = BucketIndex::build(cfg, &keys).unwrap();
    let blocks = index.closed_blocks();
    let bpb = cfg.buckets_per_block();
    let positions: Vec<u32> = [0u32, 511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 5000, 6143, 6144, t as u32 - 1].into_iter().chain((0..19).map(|_| rng.next_u32() % t as u32)).collect();
    let rows = positions.len();
    let queries = unit_rows(&mut rng, rows, d);
    // Beaming from the top, from `start_nodes`, and with levels that never
    // grow (every visible bucket a candidate of the last stage).
    for (start_nodes, page_size) in [(None, 1024), (Some(40), 1024), (None, 16 * 512)] {
        let tree = TreeConfig { node_bucket_size: 2, node_cluster_block: vec![32, 8, 4], beam_width: vec![6, 4, 3], start_nodes, page_size, growth_buckets: 64 };
        let h = build_hierarchy(&index, &keys, &tree).unwrap();
        let want = select_tree(&index, &h, &tree, &queries, &positions).unwrap();

        let geom = TreeReadGeometry { cfg, tree: tree.clone(), max_pages: blocks as u32 };
        // Index pages and node pages in scrambled order.
        let max_pages = blocks as u32;
        let pool = IndexPool::new(&g, cfg, max_pages, 1).unwrap();
        let page_of = |b: usize| ((b * 5 + 3) % blocks) as u32;
        for b in 0..blocks {
            let block = ClosedBlock {
                centroids: index.centroids()[b * bpb * d..][..bpb * d].to_vec(),
                members: index.members()[b * bpb * cfg.bucket_cap..][..bpb * cfg.bucket_cap].to_vec(),
                sizes: index.sizes()[b * bpb..][..bpb].to_vec(),
            };
            pool.write(&g, page_of(b), 0, &block).unwrap();
        }
        let node_pages: Vec<u32> = (1..=tree.top_level()).map(|l| geom.max_node_pages(l)).collect();
        let nodes = NodePool::new(&g, geom.clone(), node_pages.clone(), 1).unwrap();
        // A permutation of each level's pages (5 is coprime to their counts).
        assert!(node_pages.iter().all(|&n| n % 5 != 0));
        let node_page_of = |l: usize, p: usize| ((p * 5 + 3) % node_pages[l - 1] as usize) as u32;
        let size = tree.node_bucket_size;
        for l in 1..=h.children.len() {
            let npp = geom.nodes_per_page(l) as usize;
            for p in 0..h.children[l - 1].len() / size / npp {
                let c = &h.node_centroids[l - 1][p * npp * d..][..npp * d];
                let k = &h.children[l - 1][p * npp * size..][..npp * size];
                nodes.write(&g, 0, l, node_page_of(l, p), c, k).unwrap();
            }
        }
        let page_buckets = tree.page_size / cfg.cluster_block * bpb;
        let stride = geom.table_stride() as usize;
        let mut tables = vec![0u32; rows * stride];
        for (r, &pos) in positions.iter().enumerate() {
            let growth = growth_at(&tree, pos as usize / tree.page_size * page_buckets, page_buckets);
            let pages: Vec<u32> = (0..blocks).map(page_of).collect();
            let held: Vec<Vec<u32>> = (1..=tree.top_level()).map(|l| (0..growth.made[l - 1].div_ceil(geom.nodes_per_page(l) as usize)).map(|p| node_page_of(l, p)).collect()).collect();
            geom.write_row(&mut tables[r * stride..][..stride], &pages, &growth, &held).unwrap();
        }
        let tb = g.storage(tables.len() as u64);
        g.write(&tb, &tables);
        // The index queries head the scratch.
        let cand = g.storage(super::tree_device::cand_words(&geom, rows as u32));
        g.write_f32(&cand, &queries);
        let seqlen = g.storage(rows as u64);
        g.write(&seqlen, &positions.iter().map(|p| p + 1).collect::<Vec<u32>>());
        let width = geom.row_stride() as usize;
        let sel = g.storage((rows * width) as u64);
        let stats = g.storage(ReadStats::WORDS);
        let level_row0 = nodes.level_row0(0);
        let bufs = TreeReadBuffers { seqlen: &seqlen, centroids: &pool.centroids, members: &pool.members, nodes: &nodes.nodes, tables: &tb, cand: &cand, sel: &sel, stats: &stats, row0: 0, level_row0: &level_row0 };
        g.submit(&[], &[geom.step(&g, 0, rows as u32, &bufs).unwrap()]);
        let got: Vec<u32> = g.read(&sel, rows * width).iter().map(|x| x.to_bits()).collect();
        for r in 0..rows {
            let row = &got[r * width..][..width];
            let n = row[0] as usize;
            let host: Vec<u32> = want.row(r).iter().map(|&k| k as u32).collect();
            if row[3..3 + n] != host[..] {
                let only_dev: Vec<u32> = row[3..3 + n].iter().copied().filter(|k| !host.contains(k)).collect();
                let only_host: Vec<u32> = host.iter().copied().filter(|k| !row[3..3 + n].contains(k)).collect();
                panic!("start_nodes {start_nodes:?}, page {page_size}: row {r} (position {}) selects differently: device only {only_dev:?}, host only {only_host:?}", positions[r]);
            }
        }
    }
}
