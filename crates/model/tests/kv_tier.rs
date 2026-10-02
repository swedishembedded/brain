// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The compact KV tiers (`bf16`, per-row `int8`) against the `f32` tier, on the
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
use model::kv_tier::{kernel_list, FlashDecodeShape, KvAppend, KvKernels, KvPlane, KvTier, PrefillShape};
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
                let bound = match tier {
                    KvTier::F32 => 0.0,
                    KvTier::Bf16 => x.abs() * 2f32.powi(-8),
                    // round-to-nearest to a multiple of absmax/127, plus f32 slack on the division.
                    KvTier::Int8 => absmax / 127.0 * 0.5 * (1.0 + 1e-5),
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
    let shape = PrefillShape { n, n_heads: s.n_heads, n_kv_heads: s.n_kv, head_dim: s.head_dim, block_size: s.block_size };
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
    let shape = FlashDecodeShape { batch, n_heads: s.n_heads, n_kv_heads: s.n_kv, head_dim: s.head_dim, block_size: s.block_size, max_bt, cap: *lens.iter().max().unwrap() };
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

fn on_cpu(f: impl Fn(&Gpu, KvTier, &str)) {
    let g = Gpu::new_cpu(kernels());
    for tier in [KvTier::Bf16, KvTier::Int8] {
        f(&g, tier, "cpu");
    }
}

fn on_gpu(f: impl Fn(&Gpu, KvTier, &str)) {
    on_gpu_in(&[KvTier::Bf16, KvTier::Int8], f);
}

/// [`on_gpu`] with the `f32` tier too, for gates whose reference is not another
/// tier's kernel.
fn on_gpu_with_f32(f: impl Fn(&Gpu, KvTier, &str)) {
    on_gpu_in(&KvTier::ALL, f);
}

fn on_gpu_in(tiers: &[KvTier], f: impl Fn(&Gpu, KvTier, &str)) {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
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
