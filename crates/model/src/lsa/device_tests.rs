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
/// index space's from the same heads before rotation, then projected.
#[test]
fn index_vectors_unrotate_sum_groups_and_project() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let (nh, nkv, hd, dim) = (8usize, 2usize, 16usize, 12usize);
    let base = 10000.0f32;
    let heads = IndexHeads { n_kv_heads: nkv as u32, head_dim: hd as u32, rope_base: base, index_dim: dim as u32 };
    let mut rng = Lcg::random();
    let w: Vec<f32> = (0..nkv * hd * dim).map(|_| rng.signed()).collect();
    let project = |v: &[f32]| -> Vec<f32> {
        let rows = v.len() / (nkv * hd);
        let mut out = vec![0f32; rows * dim];
        for r in 0..rows {
            for c in 0..dim {
                out[r * dim + c] = (0..nkv * hd).map(|i| v[r * nkv * hd + i] * w[i * dim + c]).sum();
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
        let step = heads.step(&g, 0, n_heads as u32, rows as u32, &IndexVectorBuffers { x: &xb, row_off: &off, pos: &pos, w: &wb, out: &out }).unwrap();
        g.submit(&[], &[step]);
        let got = g.read(&out, rows * dim);
        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4, "{n_heads}-head rows: device index vectors differ from the host by {worst}");
    }
}

/// The device read equals `read_scores` given the device's own scores and
/// built-page count, over pages uploaded in scrambled order, with rows whose
/// index lags one block behind.
#[test]
fn the_device_read_matches_the_host_read() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let Some(g) = barrier_gpu(PIPES) else { return };
    let cfg = LsaConfig { index_dim: 16, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 32, top_p: 5, top_k: 13, sink: 2, window: 32, early: Some(EarlyBudget { position: 70, top_k: 6, top_p: 2 }) };
    let mut rng = Lcg::random();
    let t = 6 * cfg.cluster_block + 11;
    let index = BucketIndex::build(cfg, &unit_rows(&mut rng, t, cfg.index_dim)).unwrap();
    let blocks = index.closed_blocks();
    let max_pages = 8u32;
    let pool = IndexPool::new(&g, cfg, max_pages).unwrap();
    // Block b lives in page (b * 3 + 1) % max_pages.
    let page_of = |b: usize| ((b * 3 + 1) % max_pages as usize) as u32;
    let bpb = cfg.buckets_per_block();
    for b in 0..blocks {
        let block = ClosedBlock {
            centroids: index.centroids()[b * bpb * cfg.index_dim..][..bpb * cfg.index_dim].to_vec(),
            members: index.members()[b * bpb * cfg.bucket_cap..][..bpb * cfg.bucket_cap].to_vec(),
            sizes: index.sizes()[b * bpb..][..bpb].to_vec(),
        };
        pool.write(&g, page_of(b), &block).unwrap();
    }
    let positions: Vec<u32> = vec![0, 1, 31, 32, 40, 63, 64, 69, 70, 100, 150, 191, 192, t as u32 - 1, t as u32 - 2];
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
    let width = geom.selection_width() as usize;
    let sel = g.storage((rows * width) as u64);
    let meta = g.storage((rows * 3) as u64);
    let bufs = ReadBuffers { iq: &iq, centroids: &pool.centroids, members: &pool.members, page_tab: &page_tab, seqlen: &sl, scores: &scores, sel: &sel, row_meta: &meta };
    let step = geom.step(&g, 1, rows as u32, &bufs).unwrap();
    g.submit(&[], &[step]);
    let dev_scores = g.read(&scores, rows * geom.score_stride() as usize);
    let dev_sel: Vec<u32> = g.read(&sel, rows * width).iter().map(|x| x.to_bits()).collect();
    let dev_meta: Vec<u32> = g.read(&meta, rows * 3).iter().map(|x| x.to_bits()).collect();

    let stride = geom.score_stride() as usize;
    let mut lagged = 0;
    for r in 0..rows {
        let host = read_scores(&index, &dev_scores[r * stride..][..stride], stride, &positions[r..r + 1], built[r]).unwrap();
        let want: Vec<u32> = host.row(0).iter().map(|&j| j as u32).collect();
        let n = dev_meta[r * 3] as usize;
        assert_eq!(dev_meta[r * 3 + 1..r * 3 + 3], [r as u32, seqlen[r]], "row {r} metadata");
        assert_eq!(&dev_sel[r * width..][..n], &want[..], "row {r} (position {}, {} pages built) selects differently", positions[r], built[r]);
        lagged += usize::from(built[r] < (positions[r] as usize + 1) / cfg.cluster_block);
    }
    assert!(lagged > 0, "the fixture must exercise a lagging index");
}
