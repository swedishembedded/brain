// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One batch, several LoRA adapters: the bank kernels against the single-adapter
//! runtime path.
//!
//! Swedish Embedded AB implements multi-adapter LoRA serving on a single
//! resident base model for its clients. If your team needs expertise in
//! serving many fine-tunes from one set of weights then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! ## The contract under test
//!
//! [`model::dispatch::lora_bank_rows`] applies `y_i += s·B_slot(i)·(A_slot(i)·x_i)`
//! to every row `i` of a batch, where the adapter is chosen PER ROW by a slot
//! vector indexing a bank `A_bank [S, r_max, k]`, `Bt_bank [S, r_max, n]`. The
//! oracle is [`model::dispatch::lora_rows_off`], the single-adapter path the
//! runtime-LoRA model already ships: each row's correction must equal that path
//! run on that row alone with that row's adapter.
//!
//! The spec-level requirements, each its own test:
//!
//! * mixed, unsorted, interleaved slots including NONE and an out-of-range slot,
//!   ranks below `r_max` (down to 0), a row window inside a larger batch, and the
//!   delta landing in a column window of a wider output (the fused qkv / gate-up
//!   case) - on prefill and on decode shapes;
//! * rows that take no correction are BIT-identical to their input, and so is
//!   every output column outside the window;
//! * on an int8 base the correction still arrives whole, because it is added to
//!   the dequantized output from the fp32 activation.
//!
//! ## Tolerance
//!
//! The bank's `A·x` folds `k` in a different order than the oracle's GEMM, so the
//! two agree to fp32 accumulation error, not bit for bit. The deltas here are
//! O(1); the gate is `max|Δ| <= 1e-4` and cosine `>= 1 - 1e-6` per row.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::GemmVariants;
use gpu_core::select::KernelVariant;
use model::dispatch::{lora_bank_group_rows_as, lora_bank_rows, lora_bank_rows_as, lora_rows_off, mm8_rows_off, rank_word, I8Scratch, LoraBank, LoraBankKernels, LoraBankRows, LoraGroupBank, LoraGroupOut, LoraW, LORA_SLOT_NONE};
use model::int8::quantize_weight;

/// Every kernel here JIT-compiles on the CPU backend, so the whole file runs on it.
const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("lora_delta", kernels::LORA_DELTA),
    ("lora_xa_rows", kernels::LORA_XA_ROWS),
    ("lora_xa_rows_serial", kernels::LORA_XA_ROWS_SERIAL),
    ("lora_delta_rows", kernels::LORA_DELTA_ROWS),
    ("lora_delta_rows_multi", kernels::LORA_DELTA_ROWS_MULTI),
    ("lora_xa_rows_tiled", kernels::LORA_XA_ROWS_TILED),
    ("lora_delta_rows_tiled", kernels::LORA_DELTA_ROWS_TILED),
    ("lora_delta_rows_multi_tiled", kernels::LORA_DELTA_ROWS_MULTI_TILED),
];

/// The int8 base path adds the DP4A GEMM and the activation quantizer, which
/// only a GPU backend can register.
const PIPES_I8: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("lora_delta", kernels::LORA_DELTA),
    ("lora_xa_rows", kernels::LORA_XA_ROWS),
    ("lora_xa_rows_serial", kernels::LORA_XA_ROWS_SERIAL),
    ("lora_delta_rows", kernels::LORA_DELTA_ROWS),
];

const MAX_ABS: f32 = 1e-4;
const MIN_COSINE: f64 = 1.0 - 1e-6;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn upload(g: &Gpu, data: &[f32]) -> DeviceBuffer {
    g.storage_init("lora_bank_rows", data)
}

fn upload_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.storage(data.len() as u64);
    g.write(&b, data);
    b
}

fn fill(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}

/// One adapter in host form: `A [rank, k]`, `Bt [rank, n]` (scale folded in).
struct Adapter {
    rank: usize,
    a: Vec<f32>,
    bt: Vec<f32>,
}

fn adapter(seed: u64, rank: usize, k: usize, n: usize) -> Adapter {
    // Scaled so `B·(A·x)` is O(1) for unit-amplitude x: A rows have norm ~sqrt(k)/sqrt(3)·a_amp.
    let a_amp = 1.5 / (k as f32).sqrt();
    let b_amp = 1.5 / (rank.max(1) as f32).sqrt();
    Adapter { rank, a: fill(seed, rank * k, a_amp), bt: fill(seed ^ 0x9E37, rank * n, b_amp) }
}

/// The bank as the device stores it. Rows `rank..r_max` of every slot hold
/// NONZERO garbage, so a kernel that reads past a slot's rank fails loudly
/// instead of being rescued by zero padding.
fn bank(g: &Gpu, ads: &[Adapter], r_max: usize, k: usize, n: usize) -> LoraBank {
    let s = ads.len();
    let mut a = fill(0xBAD0, s * r_max * k, 3.0);
    let mut bt = fill(0xBAD1, s * r_max * n, 3.0);
    for (i, ad) in ads.iter().enumerate() {
        a[i * r_max * k..i * r_max * k + ad.rank * k].copy_from_slice(&ad.a);
        bt[i * r_max * n..i * r_max * n + ad.rank * n].copy_from_slice(&ad.bt);
    }
    // The rank word as a bank stores it: the width in the low half, the rank in the high half.
    let rank: Vec<u32> = ads.iter().map(|ad| rank_word(ad.rank as u32, ad.rank as u32)).collect();
    LoraBank { a: upload(g, &a), bt: upload(g, &bt), rank: upload_u32(g, &rank), slots: s as u32, r_max: r_max as u32 }
}

/// The oracle's GEMM tier: the per-element reference `matmul`, on every
/// backend - the simplest kernel that computes `A·x`, and one that shares no
/// reduction structure with the bank's cooperative projection.
fn oracle_tier(g: &Gpu) -> GemmVariants {
    GemmVariants::Reference(idx(g, "matmul"))
}

/// Row `row` of `x` corrected ALONE by `ad` through `lora_rows_off`: the `n`
/// floats of `s·B·(A·x_row)`.
fn solo_delta(g: &Gpu, ad: &Adapter, x: &DeviceBuffer, row: usize, k: usize, n: usize) -> Vec<f32> {
    let lo = LoraW { a: upload(g, &ad.a), bt: upload(g, &ad.bt), r: ad.rank as u32 };
    let tscr = g.storage(ad.rank.max(1) as u64);
    let out = upload(g, &vec![0f32; n]);
    let steps = lora_rows_off(g, oracle_tier(g), idx(g, "lora_delta"), &lo, &tscr, x, &out, row as u32, 0, 1, k as u32, n as u32);
    g.submit(&[], &steps);
    g.read(&out, n)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A batch problem: `total` rows of which `[row0, row0 + rows)` are corrected,
/// into the column window `[col0, col0 + n)` of a `stride`-wide output.
struct Problem {
    total: usize,
    row0: usize,
    rows: usize,
    k: usize,
    n: usize,
    r_max: usize,
    stride: usize,
    col0: usize,
    /// One slot per row of `total` (NONE or any value, valid or not).
    slot: Vec<u32>,
}

/// Run the bank kernels over `p` on `base` (`[total, stride]`) and gate every
/// row and column against the solo oracle.
/// Returns how many rows took a correction.
fn check(g: &Gpu, p: &Problem, ads: &[Adapter], x: &[f32], base: &[f32]) -> usize {
    check_as(g, p, ads, x, base, None)
}

/// [`check`] with the row-tiled EPILOGUE alone: the barrier-free projection computes `t`, the
/// `lora_delta_rows_tiled` kernel (which has no barrier and runs on every backend) applies it.
fn check_tiled_epilogue(g: &Gpu, p: &Problem, ads: &[Adapter], x: &[f32], base: &[f32]) -> usize {
    check_with(g, p, ads, x, base, None, true)
}

/// [`check`] with the bank kernels' variant forced (`None`: what the selector picks).
fn check_as(g: &Gpu, p: &Problem, ads: &[Adapter], x: &[f32], base: &[f32], variant: Option<KernelVariant>) -> usize {
    check_with(g, p, ads, x, base, variant, false)
}

fn check_with(g: &Gpu, p: &Problem, ads: &[Adapter], x: &[f32], base: &[f32], variant: Option<KernelVariant>, tiled_epilogue: bool) -> usize {
    let kernels = LoraBankKernels::resolve(g).expect("the three bank kernels are registered");
    let (xb, outb) = (upload(g, x), upload(g, base));
    let bk = bank(g, ads, p.r_max, p.k, p.n);
    let slot = upload_u32(g, &p.slot);
    let tscr = g.storage((p.rows * p.r_max) as u64);
    let rq = LoraBankRows {
        kernels,
        bank: &bk,
        slot: &slot,
        tscr: &tscr,
        x: &xb,
        out: &outb,
        row0: p.row0 as u32,
        rows: p.rows as u32,
        k: p.k as u32,
        n: p.n as u32,
        out_stride: p.stride as u32,
        out_col0: p.col0 as u32,
    };
    let steps = match variant {
        Some(v) => lora_bank_rows_as(g, &rq, v),
        None => lora_bank_rows(g, &rq),
    };
    let mut steps = steps.to_vec();
    if tiled_epilogue {
        let tiled = kernels.delta_tiled.expect("lora_delta_rows_tiled is registered");
        let params = [p.rows as u32, p.n as u32, p.r_max as u32, bk.slots, p.stride as u32, p.col0 as u32, p.row0 as u32];
        steps[1] = g.dispatch(tiled, &[&outb, &tscr, &bk.bt, &slot, &bk.rank], &params, gpu_core::Dispatch::Threads((p.rows as u32).div_ceil(4) * p.n as u32));
    }
    g.submit(&[], &steps);
    let got = g.read(&outb, p.total * p.stride);

    let mut corrected = 0;
    for i in 0..p.total {
        let row = &got[i * p.stride..(i + 1) * p.stride];
        let was = &base[i * p.stride..(i + 1) * p.stride];
        let live = i >= p.row0 && i < p.row0 + p.rows && (p.slot[i] as usize) < ads.len() && ads[p.slot[i] as usize].rank > 0;
        if !live {
            assert!(bits_equal(row, was), "row {i} (slot {:#x}) takes no correction and must be bit-identical", p.slot[i]);
            continue;
        }
        corrected += 1;
        // Everything outside the window is untouched.
        assert!(bits_equal(&row[..p.col0], &was[..p.col0]), "row {i}: columns before the window moved");
        assert!(bits_equal(&row[p.col0 + p.n..], &was[p.col0 + p.n..]), "row {i}: columns after the window moved");
        let want_delta = solo_delta(g, &ads[p.slot[i] as usize], &xb, i, p.k, p.n);
        let got_delta: Vec<f32> = row[p.col0..p.col0 + p.n].iter().zip(&was[p.col0..p.col0 + p.n]).map(|(a, b)| a - b).collect();
        let (cos, err) = (cosine(&got_delta, &want_delta), max_abs(&got_delta, &want_delta));
        assert!(err <= MAX_ABS, "row {i} slot {}: max|delta - solo| {err:.2e} > {MAX_ABS:.0e}", p.slot[i]);
        assert!(cos >= MIN_COSINE, "row {i} slot {}: cosine {cos:.8} < {MIN_COSINE}", p.slot[i]);
    }
    corrected
}

fn problem_inputs(p: &Problem, seed: u64) -> (Vec<f32>, Vec<f32>) {
    (fill(seed, p.total * p.k, 1.0), fill(seed ^ 0x77, p.total * p.stride, 2.0))
}

/// Ranks below `r_max` from 16 down to 0 (a slot that is loaded but empty).
fn mixed_adapters(k: usize, n: usize) -> Vec<Adapter> {
    [8usize, 3, 16, 1, 0].iter().enumerate().map(|(i, &r)| adapter(0x100 + i as u64, r, k, n)).collect()
}

const N: u32 = LORA_SLOT_NONE;

#[test]
fn a_prefill_batch_of_mixed_unsorted_slots_matches_each_row_run_alone() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (k, n) = (256, 192);
    let ads = mixed_adapters(k, n);
    // 40 rows, a 31-row window starting at row 3. Slots are unsorted and
    // interleaved; NONE, the empty-rank slot 4 and an out-of-range slot (7) all
    // appear inside the window, and the rows outside it carry live slots that
    // must NOT be touched.
    let cycle = [2u32, N, 0, 2, 3, N, 1, 4, 0, 7, 2, 1, 3, N, 0, 1];
    let slot: Vec<u32> = (0..40).map(|i| cycle[(i * 5 + 3) % cycle.len()]).collect();
    let p = Problem { total: 40, row0: 3, rows: 31, k, n, r_max: 16, stride: n + 96, col0: 40, slot };
    let (x, base) = problem_inputs(&p, 0xA1);
    assert!(check(&g, &p, &ads, &x, &base) > 10, "the window must hold many corrected rows");
}

#[test]
fn a_decode_row_matches_the_solo_path_for_every_slot_including_none() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (k, n) = (512, 384);
    let ads = mixed_adapters(k, n);
    for s in [N, 0, 1, 2, 3, 4] {
        let p = Problem { total: 1, row0: 0, rows: 1, k, n, r_max: 16, stride: n, col0: 0, slot: vec![s] };
        let (x, base) = problem_inputs(&p, 0xB2 + s as u64);
        let corrected = check(&g, &p, &ads, &x, &base);
        // NONE and the empty-rank slot 4 take no correction; every other slot does.
        assert_eq!(corrected, usize::from(s < 4), "slot {s:#x}");
    }
}

#[test]
fn a_decode_batch_with_a_different_adapter_per_row_matches_the_solo_path() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (k, n) = (512, 384);
    let ads = mixed_adapters(k, n);
    // The fused-qkv case: the delta lands in the middle third of a 3n-wide row.
    let slot = vec![3u32, N, 2, 0, 0, N, 1, 2];
    let p = Problem { total: 8, row0: 0, rows: 8, k, n, r_max: 16, stride: 3 * n, col0: n, slot };
    let (x, base) = problem_inputs(&p, 0xC3);
    assert_eq!(check(&g, &p, &ads, &x, &base), 6);
}

#[test]
fn every_row_none_leaves_the_output_untouched() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (k, n) = (128, 64);
    let ads = mixed_adapters(k, n);
    let p = Problem { total: 5, row0: 0, rows: 5, k, n, r_max: 16, stride: n, col0: 0, slot: vec![N; 5] };
    let (x, base) = problem_inputs(&p, 0xD4);
    let kernels = LoraBankKernels::resolve(&g).unwrap();
    let (xb, outb) = (upload(&g, &x), upload(&g, &base));
    let bk = bank(&g, &ads, 16, k, n);
    let slot = upload_u32(&g, &p.slot);
    let tscr = g.storage(5 * 16);
    let steps = lora_bank_rows(
        &g,
        &LoraBankRows { kernels, bank: &bk, slot: &slot, tscr: &tscr, x: &xb, out: &outb, row0: 0, rows: 5, k: k as u32, n: n as u32, out_stride: n as u32, out_col0: 0 },
    );
    g.submit(&[], &steps);
    assert!(bits_equal(&g.read(&outb, 5 * n), &base));
}

#[test]
fn on_an_int8_base_the_correction_arrives_whole_and_none_rows_are_the_plain_int8_output() {
    if !gpu_core::testgpu::dev(PIPES).caps().workgroup_reductions {
        brain_testutil::skip_unavailable("int8 DP4A needs a GPU backend, current is the CPU backend");
        return;
    }
    let g = gpu_core::testgpu::dev(PIPES_I8);
    let (m, k, n, r_max) = (64usize, 256usize, 192usize, 16usize);
    let ads = mixed_adapters(k, n);
    let x = fill(0xE5, m * k, 1.0);
    let w = fill(0xF6, n * k, 0.05);
    let (wq, sw) = quantize_weight(&w, n, k);

    let i8tier = GemmVariants::Fast { gemv: None, tiled: idx(&g, "matmul_i8_dyn") };
    let scr = I8Scratch::new(&g, m as u64, m as u64, &[k as u32]);
    let xb = upload(&g, &x);
    let mut steps = Vec::new();
    scr.quant_rows(&g, [idx(&g, "max_abs_row"), idx(&g, "quant_pack")], &mut steps, &xb, 0, m as u32, k as u32);
    let (pb, sb) = (upload_u32(&g, &wq), upload(&g, &sw));
    let run = |correct: bool| -> Vec<f32> {
        let o = g.storage((m * n) as u64);
        let mut s = steps.clone();
        s.push(mm8_rows_off(&g, i8tier, &scr, &pb, &sb, &o, 0, 0, m as u32, k as u32, n as u32));
        if correct {
            let slot: Vec<u32> = (0..m).map(|i| [2u32, N, 0, 3, 1][i % 5]).collect();
            let bk = bank(&g, &ads, r_max, k, n);
            let tscr = g.storage((m * r_max) as u64);
            s.extend(lora_bank_rows(
                &g,
                &LoraBankRows {
                    kernels: LoraBankKernels::resolve(&g).unwrap(),
                    bank: &bk,
                    slot: &upload_u32(&g, &slot),
                    tscr: &tscr,
                    x: &xb,
                    out: &o,
                    row0: 0,
                    rows: m as u32,
                    k: k as u32,
                    n: n as u32,
                    out_stride: n as u32,
                    out_col0: 0,
                },
            ));
        }
        g.submit(&[], &s);
        g.read(&o, m * n)
    };
    let (plain, corrected) = (run(false), run(true));
    for i in 0..m {
        let s = [2usize, usize::MAX, 0, 3, 1][i % 5];
        let (a, b) = (&plain[i * n..(i + 1) * n], &corrected[i * n..(i + 1) * n]);
        if s == usize::MAX {
            assert!(bits_equal(a, b), "row {i} is NONE and must equal the plain int8 output");
            continue;
        }
        // The oracle is the host: s·B·(A·x_i) in f64 from the numbers the device was given.
        let ad = &ads[s];
        let mut want = vec![0f32; n];
        let t: Vec<f64> = (0..ad.rank).map(|j| (0..k).map(|c| ad.a[j * k + c] as f64 * x[i * k + c] as f64).sum()).collect();
        for (o, w) in want.iter_mut().enumerate() {
            *w = (0..ad.rank).map(|j| t[j] * ad.bt[j * n + o] as f64).sum::<f64>() as f32;
        }
        let got: Vec<f32> = a.iter().zip(b).map(|(p, c)| c - p).collect();
        // The int8 output is O(10); the subtraction costs a few ulps of it.
        let err = max_abs(&got, &want);
        assert!(err <= 1e-3, "row {i} slot {s}: correction on the int8 output differs from the host oracle by {err:.2e}");
        assert!(cosine(&got, &want) >= MIN_COSINE, "row {i}: cosine {}", cosine(&got, &want));
    }
}

/// The variants a device can run: the row-wise cooperative and barrier-free kernels, and the
/// row-tiled pair where workgroup reductions execute (it folds its projection behind barriers,
/// which the CPU JIT mis-executes).
fn variants(g: &Gpu) -> Vec<KernelVariant> {
    let mut v = vec![KernelVariant::Reference];
    if g.caps().workgroup_reductions {
        v.push(KernelVariant::WorkgroupPerOutput);
        v.push(KernelVariant::RowTiled);
    }
    v
}

/// The row-tiled pair against the solo oracle on the shapes that decide it: a tile whose rows
/// share a slot (a rollout group), tiles of mixed slots, a ragged last tile, NONE and
/// out-of-range slots inside a tile, ranks below `r_max`, and a window inside a larger batch.
#[test]
fn every_variant_matches_each_row_run_alone_on_uniform_mixed_and_ragged_tiles() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (k, n) = (512, 192);
    let ads = mixed_adapters(k, n);
    let cases: Vec<(&str, usize, usize, usize, Vec<u32>)> = vec![
        ("one slot, whole tiles", 32, 0, 32, vec![2; 32]),
        ("one slot, ragged tail", 31, 0, 31, vec![0; 31]),
        ("one slot with NONE rows inside tiles", 24, 0, 24, (0..24).map(|i| if i % 5 == 1 { N } else { 3 }).collect()),
        ("alternating slots", 20, 0, 20, (0..20).map(|i| (i % 4) as u32).collect()),
        ("runs of slots with an out-of-range and an empty-rank slot", 40, 3, 33, (0..40).map(|i| [0u32, 0, 0, 0, 1, 1, 1, 1, 7, 7, 4, 4, 2, 2, 2, 2][i % 16]).collect()),
        ("fewer rows than a tile", 3, 0, 3, vec![1, 1, 1]),
    ];
    for (what, total, row0, rows, slot) in cases {
        let p = Problem { total, row0, rows, k, n, r_max: 16, stride: n + 48, col0: 24, slot };
        let (x, base) = problem_inputs(&p, 0x51 + total as u64);
        for v in variants(&g) {
            assert!(check_as(&g, &p, &ads, &x, &base, Some(v)) > 0, "{what} ({v:?}): the window holds corrected rows");
        }
        assert!(check_tiled_epilogue(&g, &p, &ads, &x, &base) > 0, "{what} (tiled epilogue): the window holds corrected rows");
    }
}

/// A group bank of three linears over one activation, the layout `AdapterBank` stores: each
/// slot's `A` blocks stacked along the rank axis, its `Bt` blocks end to end, the rank word
/// carrying the width and the adapter's rank. Every variant must give each row of each site
/// the solo path's correction and leave the bits of rows without one.
#[test]
fn every_variant_corrects_a_group_of_linears_as_each_site_run_alone() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (k, r_max) = (384usize, 16usize);
    let ns = [96usize, 64, 128];
    let ranks = [8usize, 16, 3];
    // adapters[slot][site]
    let adapters: Vec<Vec<Adapter>> = ranks.iter().enumerate().map(|(s, &r)| ns.iter().enumerate().map(|(i, &n)| adapter(0x300 + (s * 7 + i) as u64, r, k, n)).collect()).collect();
    let (slots, g_sites) = (ranks.len(), ns.len());
    let total_n: usize = ns.iter().sum();
    // A [slots, g*r_max, k], Bt [slots, r_max * sum n], padding rows hold garbage a rank bound must not read.
    let mut a = fill(0xBAD2, slots * g_sites * r_max * k, 3.0);
    let mut bt = fill(0xBAD3, slots * r_max * total_n, 3.0);
    for (s, sites) in adapters.iter().enumerate() {
        let mut off = 0;
        for (i, ad) in sites.iter().enumerate() {
            let a0 = (s * g_sites * r_max + i * r_max) * k;
            a[a0..a0 + ad.rank * k].copy_from_slice(&ad.a);
            let b0 = s * r_max * total_n + r_max * off;
            bt[b0..b0 + ad.rank * ns[i]].copy_from_slice(&ad.bt);
            off += ns[i];
        }
    }
    let word: Vec<u32> = ranks.iter().map(|&r| rank_word(r as u32, (g_sites * r_max) as u32)).collect();
    let bank = LoraGroupBank { a: upload(&g, &a), bt: upload(&g, &bt), rank: upload_u32(&g, &word), slots: slots as u32, r_max: r_max as u32, spare: [g.storage(1), g.storage(1)] };
    let rows = 37usize;
    let slot: Vec<u32> = (0..rows).map(|i| if i < 16 { 1 } else { [0u32, 2, N, 2, 2, 0, 1][i % 7] }).collect();
    let x = fill(0x77, rows * k, 1.0);
    let bases: Vec<Vec<f32>> = ns.iter().enumerate().map(|(i, &n)| fill(0x99 + i as u64, rows * n, 2.0)).collect();
    let xb = upload(&g, &x);
    let kernels = LoraBankKernels::resolve(&g).unwrap();
    // `None`: the barrier-free projection with the row-tiled group epilogue, which runs on every backend.
    let mut runs: Vec<Option<KernelVariant>> = variants(&g).into_iter().map(Some).collect();
    runs.push(None);
    for v in runs {
        let outs: Vec<DeviceBuffer> = bases.iter().map(|b| upload(&g, b)).collect();
        let tscr = g.storage((rows * g_sites * r_max) as u64);
        let group: Vec<LoraGroupOut> = outs.iter().zip(&ns).map(|(o, &n)| LoraGroupOut { out: o, n: n as u32, stride: n as u32, col0: 0 }).collect();
        let slot_buf = upload_u32(&g, &slot);
        let mut steps = lora_bank_group_rows_as(&g, kernels, &bank, &slot_buf, &tscr, &xb, &group, rows as u32, k as u32, v.unwrap_or(KernelVariant::Reference)).to_vec();
        if v.is_none() {
            let mut params = vec![rows as u32, g_sites as u32, r_max as u32, slots as u32, 0];
            let mut off = 0u32;
            for &n in &ns {
                params.extend([n as u32, n as u32, 0, off]);
                off += r_max as u32 * n as u32;
            }
            params.resize(5 + 4 * 4, 0);
            let tiled = kernels.delta_multi_tiled.expect("lora_delta_rows_multi_tiled is registered");
            steps[1] = g.dispatch(tiled, &[&tscr, &bank.bt, &slot_buf, &bank.rank, &outs[0], &outs[1], &outs[2], &bank.spare[0]], &params, gpu_core::Dispatch::Threads((rows as u32).div_ceil(4) * total_n as u32));
        }
        let v = format!("{v:?}");
        g.submit(&[], &steps);
        for (site, (&n, out)) in ns.iter().zip(&outs).enumerate() {
            let got = g.read(out, rows * n);
            for i in 0..rows {
                let (row, was) = (&got[i * n..(i + 1) * n], &bases[site][i * n..(i + 1) * n]);
                if slot[i] == N {
                    assert!(bits_equal(row, was), "{v:?}: site {site} row {i} is NONE and must keep its bits");
                    continue;
                }
                let want = solo_delta(&g, &adapters[slot[i] as usize][site], &xb, i, k, n);
                let delta: Vec<f32> = row.iter().zip(was).map(|(a, b)| a - b).collect();
                let err = max_abs(&delta, &want);
                assert!(err <= MAX_ABS, "{v:?}: site {site} row {i} slot {}: max|delta - solo| {err:.2e}", slot[i]);
            }
        }
    }
}

/// The bank kernels' device time at the 27B's shapes (k = 5120 groups of 4 and 2 linears, a single
/// 6144 -> 5120 and a 17408 -> 5120 linear, rank 32): each variant's device time per call over rows
/// 8/32/64 with every row on one slot and with rows over four slots. Prints a table; it asserts
/// nothing, so it only runs when asked (`cargo test ... -- --ignored lora_bank_kernels_cost`).
#[test]
#[ignore = "a measurement, run on a GPU"]
fn lora_bank_kernels_cost_at_27b_shapes() {
    let g = gpu_core::testgpu::dev(PIPES);
    if !g.caps().workgroup_reductions {
        eprintln!("no workgroup reductions here; nothing to compare");
        return;
    }
    let kernels = LoraBankKernels::resolve(&g).unwrap();
    let r_max = 32usize;
    // Device time of the pair's kernels (timestamp queries inside one pass), not the host's wall time.
    let time = |steps: &[gpu_core::Step]| -> f64 {
        let prof = gpu_core::profile::profile(&g, "lora bank", steps, 20);
        assert!(prof.device_timed, "this backend has no device timestamps");
        prof.rows.iter().map(|r| r.secs).sum::<f64>() * 1e6
    };
    // (label, k, outputs)
    let shapes: [(&str, usize, Vec<usize>); 4] = [("group 4 (qkv z b a)", 5120, vec![10240, 6144, 48, 48]), ("group 2 (gate up)", 5120, vec![17408, 17408]), ("single out_proj", 6144, vec![5120]), ("single down", 17408, vec![5120])];
    println!("{:<22} {:>5} {:>6} {:>12} {:>12} {:>12}", "site", "rows", "slots", "reference us", "row-wise us", "row-tiled us");
    for (label, k, ns) in &shapes {
        let total_n: usize = ns.iter().sum();
        let sites = ns.len();
        for rows in [8usize, 16, 24, 32, 64] {
            for slots_used in [1usize, 4, 5] {
                let slots = 4usize;
                let a = upload(&g, &fill(1, slots * sites * r_max * k, 0.02));
                let bt = upload(&g, &fill(2, slots * r_max * total_n, 0.02));
                let rank = upload_u32(&g, &vec![rank_word(32, (sites * r_max) as u32); slots]);
                // 1 or 4: contiguous runs of rows per slot; 5 stands for four slots interleaved row by row.
                let slot = upload_u32(&g, &(0..rows).map(|i| if slots_used == 5 { (i % 4) as u32 } else { (i * slots_used / rows) as u32 }).collect::<Vec<_>>());
                let x = upload(&g, &fill(3, rows * k, 1.0));
                let tscr = g.storage((rows * sites * r_max) as u64);
                let outs: Vec<DeviceBuffer> = ns.iter().map(|&n| g.storage((rows * n) as u64)).collect();
                let cost = |v: KernelVariant| -> f64 {
                    if sites == 1 {
                        let bank = LoraBank { a: a.clone(), bt: bt.clone(), rank: rank.clone(), slots: slots as u32, r_max: r_max as u32 };
                        let rq = LoraBankRows { kernels, bank: &bank, slot: &slot, tscr: &tscr, x: &x, out: &outs[0], row0: 0, rows: rows as u32, k: *k as u32, n: ns[0] as u32, out_stride: ns[0] as u32, out_col0: 0 };
                        time(&lora_bank_rows_as(&g, &rq, v))
                    } else {
                        let bank = LoraGroupBank { a: a.clone(), bt: bt.clone(), rank: rank.clone(), slots: slots as u32, r_max: r_max as u32, spare: [g.storage(1), g.storage(1)] };
                        let group: Vec<LoraGroupOut> = outs.iter().zip(ns).map(|(o, &n)| LoraGroupOut { out: o, n: n as u32, stride: n as u32, col0: 0 }).collect();
                        time(&lora_bank_group_rows_as(&g, kernels, &bank, &slot, &tscr, &x, &group, rows as u32, *k as u32, v))
                    }
                };
                println!("{label:<22} {rows:>5} {slots_used:>6} {:>12.1} {:>12.1} {:>12.1}", cost(KernelVariant::Reference), cost(KernelVariant::WorkgroupPerOutput), cost(KernelVariant::RowTiled));
            }
        }
    }
}

/// One row-tiled projection over interleaved slots at the 27B's `down` shape, for a profiler to
/// attach to (`ncu --kernel-name regex:lora_xa_rows_tiled`); asserts nothing.
#[test]
#[ignore = "a profiler target, run on a GPU"]
fn lora_bank_row_tiled_projection_over_interleaved_slots_for_a_profiler() {
    let g = gpu_core::testgpu::dev(PIPES);
    if !g.caps().workgroup_reductions {
        return;
    }
    let (rows, k, r_max, slots) = (32usize, 17408usize, 32usize, 4usize);
    let a = upload(&g, &fill(1, slots * r_max * k, 0.02));
    let bt = upload(&g, &fill(2, slots * r_max * 5120, 0.02));
    let rank = upload_u32(&g, &vec![rank_word(32, r_max as u32); slots]);
    let slot = upload_u32(&g, &(0..rows).map(|i| (i % 4) as u32).collect::<Vec<_>>());
    let x = upload(&g, &fill(3, rows * k, 1.0));
    let tscr = g.storage((rows * r_max) as u64);
    let out = g.storage((rows * 5120) as u64);
    let bank = LoraBank { a, bt, rank, slots: slots as u32, r_max: r_max as u32 };
    let rq = LoraBankRows { kernels: LoraBankKernels::resolve(&g).unwrap(), bank: &bank, slot: &slot, tscr: &tscr, x: &x, out: &out, row0: 0, rows: rows as u32, k: k as u32, n: 5120, out_stride: 5120, out_col0: 0 };
    for _ in 0..3 {
        g.submit(&[], &lora_bank_rows_as(&g, &rq, KernelVariant::RowTiled));
    }
    g.read(&out, 1);
}
