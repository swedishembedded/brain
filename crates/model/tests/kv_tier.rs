// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The compact KV tiers (`bf16`, per-row `int8`, per-row `fp8` E4M3) against the `f32` tier, on the
//! CPU kernels and on the real GPU.
//!
//! Swedish Embedded AB implements long-context inference serving for clients
//! whose GPU memory decides how many users one card carries. If your team needs
//! expertise in shrinking a KV cache without losing accuracy, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! A tier has two halves, and each is gated against a bound that follows from
//! the format rather than from what the kernels happened to produce:
//!
//! 1. **Write** - `KvKernels::append` into a multi-block pool whose blocks are
//!    not contiguous. What the device stored, read back and decoded on the
//!    host, is within the format's rounding bound of the source row: `bf16`
//!    keeps 8 significant bits (relative error `2^-8`), `int8` rounds to the
//!    nearest multiple of the row's scale `absmax / 127` (absolute error
//!    `scale / 2`).
//! 2. **Read** - decode attention (scores and apply) and the fused prefill over
//!    the compact plane equal the `f32` kernels run over that same decoded
//!    data. Only the load path differs, so the bound is float round-off, not a
//!    quantisation bound.
//!
//! Together they bound a tier's whole error by its storage rounding alone; the
//! model-level logits agreement is gated in `crates/qwen35`.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::paged_attention_fused;
use model::kv_tier::{kernel_list, FlashDecodeShape, KvAppend, KvElement, KvKernels, KvPlane, KvTier, PrefillShape};
use model::ops::PagedDecodeShape;

/// Real-device tests share one adapter; see `kv_bf16_roundtrip.rs`.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn kernels() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| kernel_list(&[]))
}

struct Shape {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    block_size: u32,
    num_blocks: u32,
}

impl Shape {
    fn kv_stride(&self) -> u32 {
        self.n_kv * self.head_dim
    }
    fn rows(&self) -> u64 {
        self.num_blocks as u64 * self.block_size as u64
    }
}

fn u32s(g: &Gpu, v: &[u32]) -> DeviceBuffer {
    let b = g.storage(v.len().max(1) as u64);
    g.write(&b, v);
    b
}

/// One sequence of `t` tokens whose logical blocks sit at `phys` (deliberately
/// out of order and not contiguous): `(blocks, offsets)` per token.
fn place(t: u32, block_size: u32, phys: &[u32]) -> (Vec<u32>, Vec<u32>) {
    (0..t).map(|j| (phys[(j / block_size) as usize], j % block_size)).unzip()
}

/// Append `rows` (`[t, kv_stride]`) to a fresh plane of `tier` at `(blocks, offsets)`.
fn filled(g: &Gpu, tier: KvTier, s: &Shape, rows: &[f32], blocks: &[u32], offsets: &[u32]) -> KvPlane {
    let t = blocks.len() as u32;
    let plane = KvPlane::new(g, tier, s.rows(), s.kv_stride() as u64, s.head_dim as u64);
    let k = KvKernels::resolve(g, tier).expect("kernels registered");
    let src = g.storage_init("kv.src", rows);
    let step = k.append(g, &plane, &src, &u32s(g, blocks), &u32s(g, offsets), KvAppend { batch: t, kv_stride: s.kv_stride(), block_size: s.block_size, head_dim: s.head_dim });
    g.submit(&[], &[step]);
    plane
}

/// An `f32` plane holding exactly `values` (`[rows, kv_stride]`).
fn f32_plane(g: &Gpu, s: &Shape, values: &[f32]) -> KvPlane {
    let plane = KvPlane::new(g, KvTier::F32, s.rows(), s.kv_stride() as u64, s.head_dim as u64);
    g.write_f32(plane.data(), values);
    plane
}

fn random(seed: u64, n: usize) -> Vec<f32> {
    // Mixed magnitudes: a few large entries per row make int8's absmax scale
    // matter, which is the case a flat unit-variance row would hide.
    let mut rng = Lcg::new(seed);
    let mut v = rng.vec_scaled(n, 1.0);
    for (i, x) in v.iter_mut().enumerate() {
        if i % 37 == 0 {
            *x *= 6.0;
        }
    }
    v
}

/// Gate 1: what the device stored is within the format's rounding bound.
fn write_is_within_the_formats_rounding_bound(g: &Gpu, tier: KvTier, label: &str) {
    let s = Shape { n_heads: 4, n_kv: 2, head_dim: 64, block_size: 16, num_blocks: 8 };
    let t = 70u32; // five blocks, the last partial
    let (blocks, offsets) = place(t, s.block_size, &[6, 1, 4, 7, 2]);
    let rows = random(0x7157, (t * s.kv_stride()) as usize);
    let plane = filled(g, tier, &s, &rows, &blocks, &offsets);
    let stored = plane.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);

    let mut worst = 0f32;
    for j in 0..t as usize {
        let slot = (blocks[j] * s.block_size + offsets[j]) as usize;
        for head in 0..s.n_kv as usize {
            let (src, got) = (&rows[j * s.kv_stride() as usize + head * s.head_dim as usize..][..s.head_dim as usize], &stored[slot * s.kv_stride() as usize + head * s.head_dim as usize..][..s.head_dim as usize]);
            let absmax = src.iter().fold(0f32, |m, x| m.max(x.abs()));
            for (d, (&x, &y)) in src.iter().zip(got).enumerate() {
                let bound = match tier.element() {
                    KvElement::F32 => 0.0,
                    KvElement::Bf16 => x.abs() * 2f32.powi(-8),
                    // round-to-nearest to a multiple of absmax/127, plus f32 slack on the division.
                    KvElement::Int8 => absmax / 127.0 * 0.5 * (1.0 + 1e-5),
                    // E4M3 on x / (absmax / 448): half an ulp is 2^-4 of a normal value (3
                    // mantissa bits), and 2^-10 absolute in the subnormal range below 2^-6.
                    KvElement::Fp8E4m3 => (x.abs() * 2f32.powi(-4)).max(absmax / 448.0 * 2f32.powi(-10)) * (1.0 + 1e-5),
                };
                let err = (x - y).abs();
                worst = worst.max(if bound > 0.0 { err / bound } else { err });
                assert!(err <= bound, "{label} {tier} token {j} head {head} d {d}: stored {y} for {x} (err {err}, bound {bound})");
            }
        }
    }
    // The slots no token was written to are never read; not asserted on.
    eprintln!("{label} {tier} write: worst err/bound {worst:.3}");
}

/// Gate 2a: decode attention over the compact plane == `f32` decode attention
/// over the plane's own decoded values.
fn decode_over_a_compact_plane_equals_f32_over_its_decoded_values(g: &Gpu, tier: KvTier, label: &str) {
    let s = Shape { n_heads: 4, n_kv: 2, head_dim: 64, block_size: 16, num_blocks: 12 };
    let group = s.n_heads / s.n_kv;
    // Two sequences of different lengths sharing the pool, blocks interleaved.
    let lens = [70u32, 41];
    let tables: [&[u32]; 2] = [&[6, 1, 4, 7, 2], &[9, 0, 11]];
    let max_bt = 5u32;
    let cap = *lens.iter().max().unwrap();

    let (mut blocks, mut offsets, mut krows, mut vrows) = (vec![], vec![], vec![], vec![]);
    for (b, &len) in lens.iter().enumerate() {
        let (bl, of) = place(len, s.block_size, tables[b]);
        blocks.extend(bl);
        offsets.extend(of);
        krows.extend(random(0xA000 + b as u64, (len * s.kv_stride()) as usize));
        vrows.extend(random(0xB000 + b as u64, (len * s.kv_stride()) as usize));
    }
    let kc = filled(g, tier, &s, &krows, &blocks, &offsets);
    let vc = filled(g, tier, &s, &vrows, &blocks, &offsets);
    let kdec = kc.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);
    let vdec = vc.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);
    let (kf, vf) = (f32_plane(g, &s, &kdec), f32_plane(g, &s, &vdec));

    let batch = lens.len() as u32;
    let mut table_h = vec![0u32; (batch * max_bt) as usize];
    for (b, t) in tables.iter().enumerate() {
        table_h[b * max_bt as usize..b * max_bt as usize + t.len()].copy_from_slice(t);
    }
    let (table, seq_lens) = (u32s(g, &table_h), u32s(g, &lens));
    let q = g.storage_init("q", &random(0xC0DE, (batch * s.n_heads * s.head_dim) as usize));
    let shape = PagedDecodeShape { batch, n_heads: s.n_heads, group, head_dim: s.head_dim, block_size: s.block_size, kv_stride: s.kv_stride(), cap, max_bt, scale: 1.0 / (s.head_dim as f32).sqrt() };

    let (compact, reference) = (KvKernels::resolve(g, tier).unwrap(), KvKernels::resolve(g, KvTier::F32).unwrap());
    let n_scores = (batch * s.n_heads * cap) as u64;
    let (sc_c, sc_r) = (g.storage(n_scores), g.storage(n_scores));
    g.submit(&[], &[compact.scores(g, &q, &kc, &table, &seq_lens, &sc_c, shape), reference.scores(g, &q, &kf, &table, &seq_lens, &sc_r, shape)]);
    let (scores_c, scores_r) = (g.read(&sc_c, n_scores as usize), g.read(&sc_r, n_scores as usize));
    for (b, &len) in lens.iter().enumerate() {
        for h in 0..s.n_heads as usize {
            for j in 0..len as usize {
                let i = (b * s.n_heads as usize + h) * cap as usize + j;
                let tol = 1e-4 * scores_r[i].abs().max(1.0);
                assert!((scores_c[i] - scores_r[i]).abs() <= tol, "{label} {tier} scores b={b} h={h} j={j}: {} vs {}", scores_c[i], scores_r[i]);
            }
        }
    }

    // Shared probabilities (host softmax of the reference scores), so the apply
    // comparison isolates the V load.
    let mut probs = vec![0f32; n_scores as usize];
    for (b, &len) in lens.iter().enumerate() {
        for h in 0..s.n_heads as usize {
            let base = (b * s.n_heads as usize + h) * cap as usize;
            let row = &scores_r[base..base + len as usize];
            let m = row.iter().fold(f32::MIN, |a, &x| a.max(x));
            let z: f32 = row.iter().map(|x| (x - m).exp()).sum();
            for (j, x) in row.iter().enumerate() {
                probs[base + j] = (x - m).exp() / z;
            }
        }
    }
    let probs = g.storage_init("probs", &probs);
    let n_ctx = (batch * s.n_heads * s.head_dim) as u64;
    let (cx_c, cx_r) = (g.storage(n_ctx), g.storage(n_ctx));
    g.submit(&[], &[compact.apply(g, &probs, &vc, &table, &seq_lens, &cx_c, shape), reference.apply(g, &probs, &vf, &table, &seq_lens, &cx_r, shape)]);
    let (ctx_c, ctx_r) = (g.read(&cx_c, n_ctx as usize), g.read(&cx_r, n_ctx as usize));
    assert!(ctx_r.iter().any(|x| x.abs() > 1e-2), "{label} {tier}: the reference context is all zeros, so the comparison proves nothing");
    for (i, (c, r)) in ctx_c.iter().zip(&ctx_r).enumerate() {
        assert!((c - r).abs() <= 1e-4 * r.abs().max(1.0), "{label} {tier} apply [{i}]: {c} vs {r}");
    }
    eprintln!("{label} {tier} decode read: scores and apply match f32-over-decoded");
}

/// Gate 2b: the fused head_dim-256 prefill over compact planes == the same
/// kernel over `f32` planes holding their decoded values. Chunk rows `start..`
/// of one sequence in a single-block window at a non-zero physical block.
fn fused_prefill_over_compact_planes_equals_f32_over_their_decoded_values(g: &Gpu, tier: KvTier, label: &str) {
    let s = Shape { n_heads: 4, n_kv: 2, head_dim: 256, block_size: 160, num_blocks: 3 };
    let (start, n) = (37u32, 90u32); // a chunk after a 37-token prefix: rows span two 64-row query tiles
    let phys = 2u32;
    let total = start + n;
    let (blocks, offsets) = (vec![phys; total as usize], (0..total).collect::<Vec<u32>>());
    let krows = random(0xD00D, (total * s.kv_stride()) as usize);
    let vrows = random(0xD00E, (total * s.kv_stride()) as usize);
    let kc = filled(g, tier, &s, &krows, &blocks, &offsets);
    let vc = filled(g, tier, &s, &vrows, &blocks, &offsets);
    let kdec = kc.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);
    let vdec = vc.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);
    let (kf, vf) = (f32_plane(g, &s, &kdec), f32_plane(g, &s, &vdec));

    let q = g.storage_init("q", &random(0xFEED, (n * s.n_heads * s.head_dim) as usize));
    let block_ids = u32s(g, &vec![phys; n as usize]);
    let seq_lens = u32s(g, &(0..n).map(|i| start + i + 1).collect::<Vec<_>>());
    let shape = PrefillShape { n, n_heads: s.n_heads, n_kv_heads: s.n_kv, head_dim: s.head_dim, block_size: s.block_size, max_bt: 1 };
    let words = (n * s.n_heads * s.head_dim) as u64;
    let (out_c, out_r) = (g.storage(words), g.storage(words));
    let (compact, reference) = (KvKernels::resolve(g, tier).unwrap(), KvKernels::resolve(g, KvTier::F32).unwrap().portable_prefill());
    g.submit(&[], &[compact.flash_prefill_hd256(g, &q, &kc, &vc, &block_ids, &seq_lens, &out_c, shape), reference.flash_prefill_hd256(g, &q, &kf, &vf, &block_ids, &seq_lens, &out_r, shape)]);
    let (c, r) = (g.read(&out_c, words as usize), g.read(&out_r, words as usize));
    assert!(r.iter().any(|x| x.abs() > 1e-2), "{label} {tier}: the reference output is all zeros, so the comparison proves nothing");
    let worst = c.iter().zip(&r).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0f32, f32::max);
    eprintln!("{label} {tier} fused prefill: worst relative difference to f32-over-decoded {worst:.2e}");
    assert!(worst <= 1e-3, "{label} {tier}: fused prefill over the compact plane differs from f32-over-decoded by {worst}");
}

/// Gate 2c: the fused split-key decode over compact planes equals exact
/// attention (f64, on the host) over the planes' decoded values - several
/// sequences of different lengths, the longest crossing several key splits and
/// not a multiple of the tile, blocks scattered through the pool, `n_heads / n_kv`
/// query heads sharing each kv head.
fn fused_decode_equals_exact_attention_over_the_decoded_planes(g: &Gpu, tier: KvTier, label: &str, n_heads: u32, n_kv: u32) {
    let s = Shape { n_heads, n_kv, head_dim: 256, block_size: 64, num_blocks: 80 };
    let group = (s.n_heads / s.n_kv) as usize;
    let lens = [3000u32, 70, 1500, 1];
    let max_bt = (*lens.iter().max().unwrap()).div_ceil(s.block_size);
    // Hand each sequence its blocks from one shuffled list, so no two logical
    // neighbours are physical neighbours.
    let mut ids: Vec<u32> = (0..s.num_blocks).collect();
    ids.sort_by_key(|i| (i * 37 + 11) % 101); // 37 is coprime to 101, so the keys are distinct
    let (mut tables, mut cursor) = (Vec::new(), 0usize);
    for &len in &lens {
        let n = len.div_ceil(s.block_size) as usize;
        tables.push(ids[cursor..cursor + n].to_vec());
        cursor += n;
    }
    let (mut blocks, mut offsets, mut krows, mut vrows) = (vec![], vec![], vec![], vec![]);
    for (b, &len) in lens.iter().enumerate() {
        let (bl, of) = place(len, s.block_size, &tables[b]);
        blocks.extend(bl);
        offsets.extend(of);
        krows.extend(random(0xE000 + b as u64, (len * s.kv_stride()) as usize));
        vrows.extend(random(0xF000 + b as u64, (len * s.kv_stride()) as usize));
    }
    let (kc, vc) = (filled(g, tier, &s, &krows, &blocks, &offsets), filled(g, tier, &s, &vrows, &blocks, &offsets));
    let kdec = kc.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);
    let vdec = vc.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);

    let batch = lens.len() as u32;
    let mut table_h = vec![0u32; (batch * max_bt) as usize];
    for (b, t) in tables.iter().enumerate() {
        table_h[b * max_bt as usize..b * max_bt as usize + t.len()].copy_from_slice(t);
    }
    // Queries large enough that the softmax is peaky: a flat one would hide a
    // wrong key window behind an average.
    let q_h: Vec<f32> = random(0xC0FFEE, (batch * s.n_heads * s.head_dim) as usize).iter().map(|x| x * 0.5).collect();
    let (q, table, seq_lens) = (g.storage_init("q", &q_h), u32s(g, &table_h), u32s(g, &lens));
    let ctx = g.storage((batch * s.n_heads * s.head_dim) as u64);
    let kernels = KvKernels::resolve(g, tier).unwrap();
    let shape = FlashDecodeShape { batch, n_heads: s.n_heads, n_kv_heads: s.n_kv, head_dim: s.head_dim, block_size: s.block_size, max_bt, cap: *lens.iter().max().unwrap(), pin_splits: false };
    assert!(kernels.flash_decode_available(g, s.head_dim, s.n_heads / s.n_kv), "{label} {tier}: the fused decode must be available for this shape");
    g.submit(&[], &kernels.flash_decode(g, &q, &kc, &vc, &table, &seq_lens, &ctx, shape));
    let got = g.read(&ctx, (batch * s.n_heads * s.head_dim) as usize);

    let (hd, scale) = (s.head_dim as usize, 1.0 / (s.head_dim as f64).sqrt());
    let mut worst = 0f64;
    for (b, &len) in lens.iter().enumerate() {
        for h in 0..s.n_heads as usize {
            let kvh = h / group;
            let qh = &q_h[(b * s.n_heads as usize + h) * hd..][..hd];
            let row = |j: usize| -> usize {
                let slot = (tables[b][j / s.block_size as usize] * s.block_size) as usize + j % s.block_size as usize;
                slot * s.kv_stride() as usize + kvh * hd
            };
            let scores: Vec<f64> = (0..len as usize).map(|j| scale * (0..hd).map(|d| qh[d] as f64 * kdec[row(j) + d] as f64).sum::<f64>()).collect();
            let m = scores.iter().fold(f64::MIN, |a, &x| a.max(x));
            let z: f64 = scores.iter().map(|x| (x - m).exp()).sum();
            for d in 0..hd {
                let want: f64 = (0..len as usize).map(|j| (scores[j] - m).exp() / z * vdec[row(j) + d] as f64).sum();
                let have = got[(b * s.n_heads as usize + h) * hd + d] as f64;
                let err = (want - have).abs();
                worst = worst.max(err / want.abs().max(0.05));
                assert!(err <= 2e-5 + 2e-4 * want.abs(), "{label} {tier} b={b} h={h} d={d}: fused decode {have} vs exact {want}");
            }
        }
    }
    eprintln!("{label} {tier} fused decode, group {group}: worst relative error to exact {worst:.2e}");
}

/// Gate 3: the whole attention (scores, softmax, apply) over a compact plane against the
/// ORIGINAL `f32` rows - not their decoded values - so the format's quantisation error is
/// measured at the attention output, which is what a model sees. Tolerances are
/// format-specific relative L2 bounds on the context (round-off of the format's mantissa
/// averaged over the keys); the printed numbers are the measurements.
fn attention_against_the_original_f32_rows_stays_within_the_formats_tolerance(g: &Gpu, tier: KvTier, label: &str) {
    let s = Shape { n_heads: 4, n_kv: 2, head_dim: 64, block_size: 16, num_blocks: 12 };
    let (len, max_bt) = (120u32, 8u32);
    let phys = [6u32, 1, 4, 7, 2, 9, 0, 11];
    let (blocks, offsets) = place(len, s.block_size, &phys);
    let (krows, vrows) = (random(0xD1, (len * s.kv_stride()) as usize), random(0xD2, (len * s.kv_stride()) as usize));
    let (kc, vc) = (filled(g, tier, &s, &krows, &blocks, &offsets), filled(g, tier, &s, &vrows, &blocks, &offsets));
    let (kf, vf) = (filled(g, KvTier::F32, &s, &krows, &blocks, &offsets), filled(g, KvTier::F32, &s, &vrows, &blocks, &offsets));
    let (table, seq_lens) = (u32s(g, &phys), u32s(g, &[len]));
    let q = g.storage_init("q", &random(0xD3, (s.n_heads * s.head_dim) as usize));
    let shape = PagedDecodeShape { batch: 1, n_heads: s.n_heads, group: s.n_heads / s.n_kv, head_dim: s.head_dim, block_size: s.block_size, kv_stride: s.kv_stride(), cap: len, max_bt, scale: 1.0 / (s.head_dim as f32).sqrt() };
    let run = |kernels: &KvKernels, k: &KvPlane, v: &KvPlane| -> Vec<f32> {
        let n_scores = (s.n_heads * len) as usize;
        let sc = g.storage(n_scores as u64);
        g.submit(&[], &[kernels.scores(g, &q, k, &table, &seq_lens, &sc, shape)]);
        let mut probs = g.read(&sc, n_scores);
        for row in probs.chunks_mut(len as usize) {
            let m = row.iter().fold(f32::MIN, |a, &x| a.max(x));
            let z: f32 = row.iter().map(|x| (x - m).exp()).sum();
            row.iter_mut().for_each(|x| *x = (*x - m).exp() / z);
        }
        let (probs, ctx) = (g.storage_init("probs", &probs), g.storage((s.n_heads * s.head_dim) as u64));
        g.submit(&[], &[kernels.apply(g, &probs, v, &table, &seq_lens, &ctx, shape)]);
        g.read(&ctx, (s.n_heads * s.head_dim) as usize)
    };
    let want = run(&KvKernels::resolve(g, KvTier::F32).unwrap(), &kf, &vf);
    let have = run(&KvKernels::resolve(g, tier).unwrap(), &kc, &vc);
    let rel_l2 = (have.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / want.iter().map(|b| b * b).sum::<f32>()).sqrt();
    let bound = match tier.element() {
        KvElement::F32 => 1e-6,
        KvElement::Bf16 => 4e-3,
        KvElement::Int8 => 2e-2,
        KvElement::Fp8E4m3 => 6e-2,
    };
    eprintln!("{label} {tier} attention vs original f32: rel-L2 {rel_l2:.3e} (bound {bound:.0e})");
    assert!(rel_l2 <= bound, "{label} {tier}: attention output differs from the f32 original by rel-L2 {rel_l2} (bound {bound})");
}

/// A calibrated ceiling caps the stored magnitude of a scaled plane: no element is
/// stored above `clip`, the typical elements keep their resolution, and an
/// unscaled plane refuses a ceiling.
fn a_calibrated_ceiling_caps_the_stored_magnitude(g: &Gpu, tier: KvTier, label: &str) {
    let s = Shape { n_heads: 2, n_kv: 2, head_dim: 8, block_size: 4, num_blocks: 2 };
    let (blocks, offsets) = place(1, s.block_size, &[1]);
    // head 0 carries an outlier of 100, head 1 stays within 1.
    let mut row = vec![0.5f32; s.kv_stride() as usize];
    row[3] = 100.0;
    row[8 + 2] = -0.75;
    let mut plane = KvPlane::new(g, tier, s.rows(), s.kv_stride() as u64, s.head_dim as u64);
    assert!(plane.set_clip(g, &[2.0, f32::MAX]), "{label} {tier}: a scaled plane takes a ceiling");
    let k = KvKernels::resolve(g, tier).unwrap();
    let src = g.storage_init("kv.src", &row);
    g.submit(&[], &[k.append(g, &plane, &src, &u32s(g, &blocks), &u32s(g, &offsets), KvAppend { batch: 1, kv_stride: s.kv_stride(), block_size: s.block_size, head_dim: s.head_dim })]);
    let stored = plane.read_dequantized(g, s.rows() as usize, s.kv_stride() as usize, s.head_dim as usize);
    let slot = (blocks[0] * s.block_size + offsets[0]) as usize * s.kv_stride() as usize;
    assert!((stored[slot + 3] - 2.0).abs() < 1e-5, "{label} {tier}: the outlier is clamped to the ceiling, got {}", stored[slot + 3]);
    assert!((stored[slot] - 0.5).abs() < 0.05, "{label} {tier}: a typical element keeps its value under the ceiling, got {}", stored[slot]);
    assert!((stored[slot + 10] + 0.75).abs() < 0.02, "{label} {tier}: the uncalibrated head is untouched, got {}", stored[slot + 10]);
}

#[test]
fn a_ceiling_is_refused_by_unscaled_planes() {
    let g = Gpu::new_cpu(kernels());
    for tier in [KvTier::F32, KvTier::BF16] {
        let mut plane = KvPlane::new(&g, tier, 8, 16, 8);
        assert!(!plane.set_clip(&g, &[1.0, 1.0]), "{tier} has no row scale to cap");
    }
}

/// Every finite E4M3 value survives a round trip through the device: appended at scale 1 (each
/// row carries 448, so the row scale is exactly 1) it is stored as its own byte, and the read
/// kernels decode that byte back to the same value. Exhaustive over the 254 finite codes, so the
/// encode rounding, the subnormal range and both decode paths are pinned, not sampled.
fn every_finite_e4m3_value_round_trips_through_append_and_the_read_kernel(g: &Gpu, label: &str) {
    let tier = KvTier::FP8;
    let s = Shape { n_heads: 1, n_kv: 1, head_dim: 64, block_size: 8, num_blocks: 1 };
    let values: Vec<f32> = (0..=255u8).filter(|c| c & 0x7F != 0x7F).map(model::kv_tier::decode_e4m3).collect();
    let mut rows = Vec::new();
    for chunk in values.chunks(63) {
        rows.push(448.0);
        rows.extend_from_slice(chunk);
        rows.resize(rows.len().next_multiple_of(64), 0.0);
    }
    let t = (rows.len() / 64) as u32;
    let (blocks, offsets) = place(t, s.block_size, &[0]);
    let plane = filled(g, tier, &s, &rows, &blocks, &offsets);
    let stored = plane.read_dequantized(g, s.rows() as usize, 64, 64);
    assert_eq!(&stored[..rows.len()], &rows[..], "{label}: append stored a different value than it was given");

    // One-hot queries read each element back through the scores kernel: score[j] = K[j][d].
    let (table, seq_lens) = (u32s(g, &[0]), u32s(g, &[t]));
    let kernels = KvKernels::resolve(g, tier).unwrap();
    let shape = PagedDecodeShape { batch: 1, n_heads: 1, group: 1, head_dim: 64, block_size: s.block_size, kv_stride: 64, cap: t, max_bt: 1, scale: 1.0 };
    for d in 0..64usize {
        let mut q = vec![0f32; 64];
        q[d] = 1.0;
        let (q, sc) = (g.storage_init("q", &q), g.storage(t as u64));
        g.submit(&[], &[kernels.scores(g, &q, &plane, &table, &seq_lens, &sc, shape)]);
        let got = g.read(&sc, t as usize);
        for j in 0..t as usize {
            assert_eq!(got[j], rows[j * 64 + d], "{label}: scores kernel read {} for stored {} (row {j}, dim {d})", got[j], rows[j * 64 + d]);
        }
    }
}

#[test]
fn every_finite_e4m3_value_round_trips_on_cpu_and_gpu() {
    every_finite_e4m3_value_round_trips_through_append_and_the_read_kernel(&Gpu::new_cpu(kernels()), "cpu");
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    every_finite_e4m3_value_round_trips_through_append_and_the_read_kernel(&Gpu::new_gpu(kernels()), "gpu");
}

fn on_cpu(f: impl Fn(&Gpu, KvTier, &str)) {
    let g = Gpu::new_cpu(kernels());
    for tier in [KvTier::BF16, KvTier::INT8, KvTier::FP8] {
        f(&g, tier, "cpu");
    }
}

fn on_gpu(f: impl Fn(&Gpu, KvTier, &str)) {
    on_gpu_in(&[KvTier::BF16, KvTier::INT8, KvTier::FP8], f);
}

/// [`on_gpu`] with the `f32` tier too, for gates whose reference is not another
/// tier's kernel.
fn on_gpu_with_f32(f: impl Fn(&Gpu, KvTier, &str)) {
    on_gpu_in(&KvTier::ALL, f);
}

fn on_gpu_in(tiers: &[KvTier], f: impl Fn(&Gpu, KvTier, &str)) {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let g = Gpu::new_gpu(kernels());
    for &tier in tiers {
        f(&g, tier, "gpu");
    }
}

#[test]
fn a_compact_plane_stores_each_row_within_its_formats_rounding_bound_on_cpu() {
    on_cpu(write_is_within_the_formats_rounding_bound);
}

#[test]
fn attention_over_a_compact_plane_stays_within_the_formats_tolerance_of_the_f32_original_on_cpu() {
    let g = Gpu::new_cpu(kernels());
    for tier in KvTier::ALL {
        attention_against_the_original_f32_rows_stays_within_the_formats_tolerance(&g, tier, "cpu");
    }
}

#[test]
fn a_calibrated_ceiling_caps_a_scaled_planes_stored_magnitude_on_cpu() {
    let g = Gpu::new_cpu(kernels());
    for tier in [KvTier::INT8, KvTier::FP8] {
        a_calibrated_ceiling_caps_the_stored_magnitude(&g, tier, "cpu");
    }
}

#[test]
fn a_compact_plane_stores_each_row_within_its_formats_rounding_bound_on_gpu() {
    on_gpu(write_is_within_the_formats_rounding_bound);
}

#[test]
fn decode_attention_reads_a_compact_plane_exactly_as_f32_reads_its_decoded_values_on_cpu() {
    on_cpu(decode_over_a_compact_plane_equals_f32_over_its_decoded_values);
}

#[test]
fn decode_attention_reads_a_compact_plane_exactly_as_f32_reads_its_decoded_values_on_gpu() {
    on_gpu(decode_over_a_compact_plane_equals_f32_over_its_decoded_values);
}

#[test]
fn fused_prefill_reads_compact_planes_exactly_as_f32_reads_their_decoded_values_on_gpu() {
    on_gpu(|g, tier, label| {
        if !paged_attention_fused(g, true, false, 256, 0) {
            brain_testutil::skip_unavailable("this device does not select the fused head_dim-256 prefill kernel");
            return;
        }
        fused_prefill_over_compact_planes_equals_f32_over_their_decoded_values(g, tier, label);
    });
}

#[test]
fn fused_decode_reads_compact_planes_as_exact_attention_reads_their_decoded_values_on_gpu() {
    on_gpu_with_f32(|g, tier, label| {
        if !g.caps().workgroup_reductions {
            brain_testutil::skip_unavailable("this device does not run workgroup-barrier kernels");
            return;
        }
        // Qwen3.8-27B's group of six, and the 35B-A3B sibling's eight.
        for (n_heads, n_kv) in [(12, 2), (16, 2)] {
            fused_decode_equals_exact_attention_over_the_decoded_planes(g, tier, label, n_heads, n_kv);
        }
    });
}

/// The served 27B's int8 decode attention (24 query heads over 4 kv heads at head_dim 256) on a CUDA device, where
/// it runs the native kernel, at the context lengths agents run at: each sequence's blocks are scattered across the
/// pool, and every sequence but the first resumes from one cached prefix (its first `shared_blocks` blocks are the
/// SAME physical blocks, the way the prefix cache shares them). The output equals exact attention over the decoded
/// planes at the bound the WGSL tier is held to.
fn native_decode_is_exact_at_agent_contexts(g: &Gpu, tier: KvTier, lens: &[u32], shared_blocks: u32) {
    let (n_heads, n_kv, block_size) = (24u32, 4u32, 64u32);
    let label = format!("{tier} lens {lens:?}, {shared_blocks} shared blocks");
    let own_blocks: u32 = lens.iter().enumerate().map(|(b, &len)| len.div_ceil(block_size) - if b == 0 { 0 } else { shared_blocks }).sum();
    let num_blocks = (shared_blocks + own_blocks) * 2 + 1;
    let mut next = 0u32;
    // Every other physical block, walked backwards, so no two logical neighbours are physical neighbours.
    let mut take = |n: u32| -> Vec<u32> {
        let ids = (0..n).map(|i| num_blocks - 1 - 2 * (next + i)).collect();
        next += n;
        ids
    };
    let prefix = take(shared_blocks);
    let tables: Vec<Vec<u32>> = lens
        .iter()
        .enumerate()
        .map(|(b, &len)| {
            let n = len.div_ceil(block_size);
            if b == 0 {
                return take(n);
            }
            let mut t = prefix.clone();
            t.extend(take(n - shared_blocks));
            t
        })
        .collect();
    let s = Shape { n_heads, n_kv, head_dim: 256, block_size, num_blocks };
    let stride = s.kv_stride() as usize;
    let prefix_tokens = (shared_blocks * block_size) as usize;
    let (prefix_k, prefix_v) = (random(0xA1, prefix_tokens * stride), random(0xA2, prefix_tokens * stride));
    let (mut blocks, mut offsets, mut krows, mut vrows) = (vec![], vec![], vec![], vec![]);
    for (b, &len) in lens.iter().enumerate() {
        let skip = if b == 0 { 0 } else { prefix_tokens };
        // The shared prefix is written once, by the second sequence; the others only point at it.
        let from = if b == 1 { 0 } else { skip };
        let (bl, of) = place(len, block_size, &tables[b]);
        blocks.extend_from_slice(&bl[from..]);
        offsets.extend_from_slice(&of[from..]);
        if b == 1 {
            krows.extend_from_slice(&prefix_k);
            vrows.extend_from_slice(&prefix_v);
        }
        krows.extend(random(0xB000 + b as u64, (len as usize - skip) * stride));
        vrows.extend(random(0xC000 + b as u64, (len as usize - skip) * stride));
    }
    let (kc, vc) = (filled(g, tier, &s, &krows, &blocks, &offsets), filled(g, tier, &s, &vrows, &blocks, &offsets));
    let kdec = kc.read_dequantized(g, s.rows() as usize, stride, s.head_dim as usize);
    let vdec = vc.read_dequantized(g, s.rows() as usize, stride, s.head_dim as usize);

    let batch = lens.len() as u32;
    let max_bt = tables.iter().map(|t| t.len()).max().unwrap() as u32;
    let mut table_h = vec![0u32; (batch * max_bt) as usize];
    for (b, t) in tables.iter().enumerate() {
        table_h[b * max_bt as usize..b * max_bt as usize + t.len()].copy_from_slice(t);
    }
    let q_h: Vec<f32> = random(0xC0FFEE, (batch * n_heads * 256) as usize).iter().map(|x| x * 0.5).collect();
    let (q, table, seq_lens) = (g.storage_init("q", &q_h), u32s(g, &table_h), u32s(g, lens));
    let ctx = g.storage((batch * n_heads * 256) as u64);
    let kernels = KvKernels::resolve(g, tier).unwrap();
    let shape = FlashDecodeShape { batch, n_heads, n_kv_heads: n_kv, head_dim: 256, block_size, max_bt, cap: *lens.iter().max().unwrap(), pin_splits: false };
    assert!(kernels.flash_decode_available(g, 256, n_heads / n_kv), "{label}: the fused decode must be available");
    let (slow, native) = match () {
        _ if tier == KvTier::INT8 => ("paged_flash_decode_gqa_hd256#GRP=6#kv=int8", "paged_flash_decode_gqa_hd256_kv8"),
        _ if tier == KvTier::FP8 => ("paged_flash_decode_gqa_hd256#GRP=6#kv=fp8", "paged_flash_decode_gqa_hd256_kvfp8"),
        _ if tier == KvTier::BF16 => ("paged_flash_decode_gqa_hd256#GRP=6#kv=bf16", "paged_flash_decode_gqa_hd256_bf16"),
        _ => ("paged_flash_decode_gqa_hd256#GRP=6", "paged_flash_decode_gqa_hd256_f32"),
    };
    let kernel = g.kernel_index(slow).expect("the decode kernel is registered");
    let params = [batch, n_heads, n_kv, 256, n_heads / n_kv, block_size, max_bt, 1, 1];
    assert_eq!(g.native_kernel_for(kernel, &params), Some(native), "{label}: the decode attention must run the native kernel");
    let combine = g.kernel_index("paged_flash_decode_gqa_combine").expect("the combine is registered");
    assert_eq!(g.native_kernel_for(combine, &[batch, n_heads, 256, 64]), Some("paged_flash_decode_gqa_combine"), "{label}: the merge of the splits must run the native kernel");
    g.submit(&[], &kernels.flash_decode(g, &q, &kc, &vc, &table, &seq_lens, &ctx, shape));
    let got = g.read(&ctx, (batch * n_heads * 256) as usize);

    let (hd, group, scale) = (256usize, (n_heads / n_kv) as usize, 1.0 / 16.0f64);
    let mut worst = 0f64;
    for (b, &len) in lens.iter().enumerate() {
        // Every kv head's planes, the first and last head of a group among them.
        for h in [0usize, group - 1, group + 2, 2 * group + 3, 4 * group - 1] {
            let kvh = h / group;
            let qh = &q_h[(b * n_heads as usize + h) * hd..][..hd];
            let row = |j: usize| -> usize {
                let slot = (tables[b][j / block_size as usize] * block_size) as usize + j % block_size as usize;
                slot * stride + kvh * hd
            };
            let scores: Vec<f64> = (0..len as usize).map(|j| scale * (0..hd).map(|d| qh[d] as f64 * kdec[row(j) + d] as f64).sum::<f64>()).collect();
            let m = scores.iter().fold(f64::MIN, |a, &x| a.max(x));
            let z: f64 = scores.iter().map(|x| (x - m).exp()).sum();
            for d in 0..hd {
                let want: f64 = (0..len as usize).map(|j| (scores[j] - m).exp() / z * vdec[row(j) + d] as f64).sum();
                let have = got[(b * n_heads as usize + h) * hd + d] as f64;
                let err = (want - have).abs();
                worst = worst.max(err / want.abs().max(0.05));
                assert!(err <= 2e-5 + 2e-4 * want.abs(), "{label} b={b} len={len} h={h} d={d}: native decode {have} vs exact {want}");
            }
        }
    }
    eprintln!("native decode at {label}: worst relative error to exact {worst:.2e}");
}

#[test]
fn native_decode_of_every_tier_is_exact_at_agent_contexts_over_a_shared_prefix_on_cuda() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    // Opened as CUDA explicitly: the ambient GPU may be another backend, which keeps the WGSL tier.
    let g = match Gpu::try_new_cuda(kernels()) {
        Ok(g) if g.caps().arch.compute_capability.is_some() => g,
        _ => {
            brain_testutil::skip_unavailable("native int8 decode attention: no CUDA device");
            return;
        }
    };
    // One sequence at 48k; 1k + 13k + 48k with the two long ones on one 10,240-token prefix; eight agents at 1-4k
    // on one 1,024-token prefix; sixteen short ones (partly empty splits, a one-key sequence).
    for tier in KvTier::ALL {
        native_decode_is_exact_at_agent_contexts(&g, tier, &[48_000], 0);
        native_decode_is_exact_at_agent_contexts(&g, tier, &[1024, 13_000, 48_000], 160);
        native_decode_is_exact_at_agent_contexts(&g, tier, &[4096, 1100, 2047, 3000, 1025, 4000, 1500, 2500], 16);
        native_decode_is_exact_at_agent_contexts(&g, tier, &[1, 129, 64, 65, 300, 127, 128, 700, 5, 200, 333, 1000, 63, 2, 900, 450], 0);
    }
}

/// Device time of the 27B's int8 decode attention (split pass plus merge) per layer at agent context lengths, with the
/// achieved rate on the KV bytes it must read, for the native kernel and for the generated WGSL tier.
fn time_decode(g: &Gpu, tier: KvTier, batch: u32, len: u32) -> f64 {
    let (n_heads, n_kv, block_size) = (24u32, 4u32, 64u32);
    let per = len.div_ceil(block_size);
    let num_blocks = batch * per;
    let s = Shape { n_heads, n_kv, head_dim: 256, block_size, num_blocks };
    // Whatever the fresh planes hold: the rate does not depend on the values.
    let plane = || KvPlane::new(g, tier, s.rows(), s.kv_stride() as u64, s.head_dim as u64);
    let (kc, vc) = (plane(), plane());
    // Sequence b owns blocks b, b + batch, b + 2 batch, ...: interleaved, as a busy pool leaves them.
    let table_h: Vec<u32> = (0..batch).flat_map(|b| (0..per).map(move |i| i * batch + b)).collect();
    let q_h = random(0xC0FFEE, (batch * n_heads * 256) as usize);
    let (q, table, seq_lens) = (g.storage_init("q", &q_h), u32s(g, &table_h), u32s(g, &vec![len; batch as usize]));
    let ctx = g.storage((batch * n_heads * 256) as u64);
    let kernels = KvKernels::resolve(g, tier).unwrap();
    let shape = FlashDecodeShape { batch, n_heads, n_kv_heads: n_kv, head_dim: 256, block_size, max_bt: per, cap: len, pin_splits: false };
    let steps = kernels.flash_decode(g, &q, &kc, &vc, &table, &seq_lens, &ctx, shape);
    g.submit(&[], &steps);
    g.poll_wait();
    g.set_kernel_timing(true);
    let mut best = f64::MAX;
    for _ in 0..9 {
        g.reset_kernel_times();
        let steps = kernels.flash_decode(g, &q, &kc, &vc, &table, &seq_lens, &ctx, shape);
        g.submit(&[], &steps);
        g.poll_wait();
        best = best.min(g.kernel_times().unwrap_or_default().iter().map(|(_, ms, _)| ms).sum::<f64>());
    }
    g.set_kernel_timing(false);
    best
}

#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn decode_attention_timing_report() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(native) = Gpu::try_new_cuda(kernels()) else {
        return brain_testutil::skip_unavailable("int8 decode timing: no CUDA device");
    };
    gpu_core::set_native_kernels(false);
    let generated = Gpu::try_new_cuda(kernels()).expect("a second CUDA handle");
    gpu_core::set_native_kernels(true);
    for tier in KvTier::ALL {
    for (batch, len) in [(1u32, 1024u32), (1, 12_800), (1, 48_000), (8, 1024), (8, 12_800), (16, 12_800), (8, 48_000)] {
        // K and V 8-bit rows plus one f32 scale per (token, kv head) row each.
        let bytes = 2.0 * batch as f64 * len as f64 * 4.0 * (256.0 + 4.0);
        let (tn, tg) = (time_decode(&native, tier, batch, len), time_decode(&generated, tier, batch, len));
        eprintln!(
            "{tier} decode attention batch {batch:2} x {len:6} keys: native {:8.1} us ({:6.0} GB/s), generated {:8.1} us ({:6.0} GB/s), {:.1}x",
            tn * 1e3,
            bytes / (tn * 1e-3) / 1e9,
            tg * 1e3,
            bytes / (tg * 1e-3) / 1e9,
            tg / tn
        );
    }
    }
}
