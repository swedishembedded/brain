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
use model::dispatch::{lora_bank_rows, lora_rows_off, mm8_rows_off, I8Scratch, LoraBank, LoraBankKernels, LoraBankRows, LoraW, LORA_SLOT_NONE};
use model::int8::quantize_weight;

/// Every kernel here JIT-compiles on the CPU backend, so the whole file runs on it.
const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("lora_delta", kernels::LORA_DELTA),
    ("lora_xa_rows", kernels::LORA_XA_ROWS),
    ("lora_xa_rows_serial", kernels::LORA_XA_ROWS_SERIAL),
    ("lora_delta_rows", kernels::LORA_DELTA_ROWS),
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
    let rank: Vec<u32> = ads.iter().map(|ad| ad.rank as u32).collect();
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
    let kernels = LoraBankKernels::resolve(g).expect("the three bank kernels are registered");
    let (xb, outb) = (upload(g, x), upload(g, base));
    let bk = bank(g, ads, p.r_max, p.k, p.n);
    let slot = upload_u32(g, &p.slot);
    let tscr = g.storage((p.rows * p.r_max) as u64);
    let steps = lora_bank_rows(
        g,
        &LoraBankRows {
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
        },
    );
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
