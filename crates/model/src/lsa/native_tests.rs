// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::native::*;
use super::Selection;
use data::rng::Lcg;

/// A random causal selection per query: itself, then each earlier key with
/// probability one half, at most `width` keys.
fn random_selection(rng: &mut Lcg, t: usize, width: usize) -> Selection {
    let mut ids = vec![-1i32; t * width];
    let mut lens = vec![0u32; t];
    for i in 0..t {
        let mut row = vec![i as i32];
        for j in 0..i {
            if row.len() < width && rng.next_u32().is_multiple_of(2) {
                row.push(j as i32);
            }
        }
        ids[i * width..][..row.len()].copy_from_slice(&row);
        lens[i] = row.len() as u32;
    }
    Selection { width, ids, lens }
}

/// Host oracle in f64: attention over each query's selection, its outputs and
/// the gradients of `sum(ctx * d_ctx)`.
struct Oracle {
    ctx: Vec<f64>,
    d_q: Vec<f64>,
    d_k: Vec<f64>,
    d_v: Vec<f64>,
}

fn oracle(s: &NativeShape, sel: &[Selection], q: &[f32], k: &[f32], v: &[f32], d_ctx: &[f32]) -> Oracle {
    let (t, nh, nkv, hd) = (s.t as usize, s.n_heads as usize, s.n_kv_heads as usize, s.head_dim as usize);
    let g = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    let rows = s.b as usize * t;
    let mut o = Oracle { ctx: vec![0.0; rows * nh * hd], d_q: vec![0.0; rows * nh * hd], d_k: vec![0.0; rows * nkv * hd], d_v: vec![0.0; rows * nkv * hd] };
    let qa = |r: usize, h: usize, c: usize| f64::from(q[(r * nh + h) * hd + c]);
    let ka = |r: usize, h: usize, c: usize| f64::from(k[(r * nkv + h) * hd + c]);
    let va = |r: usize, h: usize, c: usize| f64::from(v[(r * nkv + h) * hd + c]);
    let da = |r: usize, h: usize, c: usize| f64::from(d_ctx[(r * nh + h) * hd + c]);
    for (b, selection) in sel.iter().enumerate() {
        for i in 0..t {
            let r = b * t + i;
            let keys: Vec<usize> = selection.row(i).iter().map(|&j| b * t + j as usize).collect();
            for h in 0..nh {
                let kh = h / g;
                let sc: Vec<f64> = keys.iter().map(|&kr| scale * (0..hd).map(|c| qa(r, h, c) * ka(kr, kh, c)).sum::<f64>()).collect();
                let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                let z: f64 = sc.iter().map(|x| (x - m).exp()).sum();
                let p: Vec<f64> = sc.iter().map(|x| (x - m).exp() / z).collect();
                for c in 0..hd {
                    o.ctx[(r * nh + h) * hd + c] = keys.iter().zip(&p).map(|(&kr, &pj)| pj * va(kr, kh, c)).sum();
                }
                let dp: Vec<f64> = keys.iter().map(|&kr| (0..hd).map(|c| da(r, h, c) * va(kr, kh, c)).sum()).collect();
                let dsum: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                for (e, &kr) in keys.iter().enumerate() {
                    let ds = p[e] * (dp[e] - dsum);
                    for c in 0..hd {
                        o.d_q[(r * nh + h) * hd + c] += scale * ds * ka(kr, kh, c);
                        o.d_k[(kr * nkv + kh) * hd + c] += scale * ds * qa(r, h, c);
                        o.d_v[(kr * nkv + kh) * hd + c] += p[e] * da(r, h, c);
                    }
                }
            }
        }
    }
    o
}

fn worst(dev: &[f32], host: &[f64]) -> f64 {
    let scale = host.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-12);
    dev.iter().zip(host).fold(0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs())) / scale
}

/// The kernels compute attention over each query's selection and its exact
/// gradients - the host oracle's - across grouped heads, several sequences,
/// more keys than one tile and a head wider than one workgroup.
#[test]
fn native_sparse_attention_matches_the_host_oracle() {
    static PIPES: &[(&str, &str)] = &KERNELS;
    let g = gpu_core::testgpu::dev(PIPES);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("native sparse attention kernels need workgroup barriers (GPU-only)");
        return;
    }
    let ids = NativeKernels { fwd: 0, bwd_dq: 1, bwd_dkv: 2 };
    let mut rng = Lcg::random();
    for (t, nh, nkv, hd, width) in [(150u32, 4u32, 2u32, 16u32, 96u32), (70, 6, 2, 96, 70), (40, 2, 2, 8, 12), (50, 10, 2, 32, 40), (30, 8, 1, 256, 20)] {
        let s = NativeShape { b: 2, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, width };
        let sel: Vec<Selection> = (0..2).map(|_| random_selection(&mut rng, t as usize, width as usize)).collect();
        let rows = (2 * t) as usize;
        let rand = |rng: &mut Lcg, n: usize| -> Vec<f32> { (0..n).map(|_| rng.signed()).collect() };
        let (q, k, v, d_ctx) = (rand(&mut rng, rows * (nh * hd) as usize), rand(&mut rng, rows * (nkv * hd) as usize), rand(&mut rng, rows * (nkv * hd) as usize), rand(&mut rng, rows * (nh * hd) as usize));
        let want = oracle(&s, &sel, &q, &k, &v, &d_ctx);

        let index = g.storage(s.index_words());
        g.write(&index, &s.pack(&sel).unwrap());
        let (qb, kb, vb, db) = (g.storage_init("q", &q), g.storage_init("k", &k), g.storage_init("v", &v), g.storage_init("d_ctx", &d_ctx));
        let (ctx, stats) = (g.storage(q.len() as u64), g.storage(s.stats_words()));
        let (d_q, d_k, d_v) = (g.storage(q.len() as u64), g.storage(k.len() as u64), g.storage(v.len() as u64));
        let mut steps = vec![s.forward(&g, &ids, &qb, &kb, &vb, &index, &ctx, &stats).unwrap()];
        steps.extend(s.backward(&g, &ids, &NativeBackward { q: &qb, k: &kb, v: &vb, ctx: &ctx, d_ctx: &db, index: &index, stats: &stats, d_q: &d_q, d_k: &d_k, d_v: &d_v }).unwrap());
        g.submit(&[], &steps);
        let errs = [
            ("ctx", worst(&g.read(&ctx, q.len()), &want.ctx)),
            ("d_q", worst(&g.read(&d_q, q.len()), &want.d_q)),
            ("d_k", worst(&g.read(&d_k, k.len()), &want.d_k)),
            ("d_v", worst(&g.read(&d_v, v.len()), &want.d_v)),
        ];
        println!("t {t} heads {nh}/{nkv} head_dim {hd} width {width}: {errs:?}");
        for (name, err) in errs {
            assert!(err < 1e-4, "{name} differs from the oracle by {err} of its largest value (t {t}, heads {nh}/{nkv}, head_dim {hd})");
        }
    }
}

/// The packed index holds each row's keys and, per key, exactly the queries
/// that selected it; a selection the kernels cannot run is refused.
#[test]
fn packing_inverts_the_selection_and_refuses_bad_rows() {
    let s = NativeShape { b: 1, t: 3, n_heads: 1, n_kv_heads: 1, head_dim: 4, width: 2 };
    let sel = Selection { width: 2, ids: vec![0, -1, 1, 0, 2, 0], lens: vec![1, 2, 2] };
    let words = s.pack(&[sel]).unwrap();
    // Rows [count, row, live keys, keys...] of stride 5, then offsets, then
    // the queries per key.
    assert_eq!(words, vec![1, 0, 1, 0, 0, 2, 1, 2, 1, 0, 2, 2, 3, 2, 0, /* off */ 0, 3, 4, 5, /* inv */ 0, 1, 2, 1, 2, 0]);
    let future = Selection { width: 2, ids: vec![1, -1, 1, -1, 2, -1], lens: vec![1, 1, 1] };
    assert!(s.pack(&[future]).is_err(), "a key after its query must be refused");
    let empty = Selection { width: 2, ids: vec![-1; 6], lens: vec![0, 1, 1] };
    assert!(s.pack(&[empty]).is_err(), "an empty row must be refused");
}

/// The device inverts a selection exactly as the host packs it: every key's
/// queries, in ascending order, over sequences whose keys are selected by
/// none, few or all of the queries.
#[test]
fn the_device_inversion_is_the_host_packing() {
    static PIPES: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    let pipes = PIPES.get_or_init(|| KERNELS.iter().chain(&gpu_core::sort::KERNELS).copied().collect());
    let g = gpu_core::testgpu::dev(pipes);
    let sort = gpu_core::sort::SortKernels { scan: gpu_core::sort::ScanKernels { scan_block: 5, scan_add: 6 }, sort_hist: 7, sort_scatter: 8 };
    let ids = SelectKernels { index_vectors: usize::MAX, read: usize::MAX, ward: usize::MAX, pairs: 3, offsets: 4, sort };
    let mut rng = Lcg::random();
    for (b, t, width) in [(3u32, 300u32, 40u32), (1, 1000, 100), (2, 5, 5)] {
        let shape = NativeShape { b, t, n_heads: 2, n_kv_heads: 1, head_dim: 8, width };
        let sel: Vec<Selection> = (0..b).map(|_| random_selection(&mut rng, t as usize, width as usize)).collect();
        let want = shape.pack(&sel).unwrap();
        // The device sees only the rows, as the read writes them.
        let rows_words = (b * t * (width + crate::paged_sparse::ROW_HEADER)) as usize;
        let mut words = want[..rows_words].to_vec();
        words.resize(want.len(), 0);
        let index = g.storage(shape.index_words());
        g.write(&index, &words);
        let inversion = Inversion::new(&g, shape, width);
        let mut steps = Vec::new();
        inversion.record(&g, &ids, &index, &mut steps);
        g.submit(&[], &steps);
        let got: Vec<u32> = g.read(&index, want.len()).iter().map(|x| x.to_bits()).collect();
        let used = rows_words + (b * t) as usize + 1 + want[rows_words + (b * t) as usize] as usize;
        assert_eq!(&got[..used], &want[..used], "b {b} t {t} width {width}: the device inverse differs from the host's");
    }
}
