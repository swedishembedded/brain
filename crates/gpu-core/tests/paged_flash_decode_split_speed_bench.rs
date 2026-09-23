// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M2.7: interleaved min-of-N A/B, three decode-attention arms at
//! Qwen3-0.6B's real decode-head shape (`n_heads=16, n_kv_heads=8,
//! head_dim=128`) - the number this milestone's roadmap ledger entry
//! reports (checklist §F.6/E.0), matching `kq_gemv_reg_speed_bench.rs`'s own
//! precedent for the same kind of claim:
//!
//!   - `triad`: `paged_decode_scores_wg` -> `decode_softmax_batched` ->
//!     `paged_decode_apply_batched`, the three kernels `qwen3::serve`
//!     ACTUALLY dispatches for decode today (M2.1's own baseline).
//!   - `flash` (unsplit): `paged_flash_decode`, M2.1's fused-but-disabled
//!     kernel - one workgroup per (sequence, head), serialising the WHOLE
//!     key range's tiles.
//!   - `split` (M2.7): `paged_flash_decode_split` -> `paged_flash_decode_
//!     combine`, the same fused algorithm with the key range chopped into
//!     `n_splits` independent workgroups per (sequence, head) - the
//!     occupancy fix M2.1's own header named as a follow-up and this
//!     repo's decision-4 convention required a FRESH profile of on this
//!     hardware (an Intel Arc Xe-LPG iGPU), not a skip-by-precedent from
//!     the P40 numbers M2.1 recorded.
//!
//! `#[ignore]`d - a measurement, not a correctness gate (that is
//! `crates/model/src/paged.rs::flash_tests::
//! paged_flash_decode_split_matches_batched_triad`). Run manually:
//! `cargo test --release --offline -p brain-gpu-core --test
//! paged_flash_decode_split_speed_bench -- --ignored --nocapture`.

use std::time::{Duration, Instant};

use gpu_core::Gpu;

/// `model::block::PAGED_SCORES_PER_WORKGROUP` - duplicated here rather than
/// pulled in as a dependency (`brain-gpu-core` cannot depend on
/// `brain-model`, which itself depends on `brain-gpu-core`); the same
/// constant `qwen_bench::flash_decode_bench`'s own reproducer already
/// hardcodes for the identical reason.
const PAGED_SCORES_PER_WORKGROUP: u32 = 16;

/// Continuous dispatches of `steps` for `dur`, so a device at its DVFS idle
/// floor (checklist §E.0b - measured on this exact integrated Arc: 350 MHz
/// idle vs ~2150 MHz ramped) reaches the clock a real workload runs at
/// BEFORE anything is timed. Identical to `kq_gemv_reg_speed_bench.rs`'s own
/// `ramp` - this milestone's own instruction names that file as the
/// template.
fn ramp(g: &Gpu, steps: &[gpu_core::Step], dur: Duration) {
    let t0 = Instant::now();
    while t0.elapsed() < dur {
        g.submit(&[], steps);
        g.poll_wait();
    }
}

/// One `poll_wait`-bracketed submit's wall time.
fn once(g: &Gpu, steps: &[gpu_core::Step]) -> f64 {
    let t0 = Instant::now();
    g.submit(&[], steps);
    g.poll_wait();
    t0.elapsed().as_secs_f64()
}

struct Shape {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    bs: u32,
    seq: u32,
}

/// All the device buffers/params one shape needs, built once and reused
/// across every arm and every round-robin rep - only the ARMS' own steps
/// differ per iteration, never the data.
struct Fixture {
    // triad
    triad_steps_params: (Vec<u32>, Vec<u32>, Vec<u32>),
    // shared decode inputs
    scores_threads: u32,
    // flash (unsplit)
    flash_params: Vec<u32>,
    // split
    split_params: Vec<u32>,
    combine_params: Vec<u32>,
    n_splits: u32,
}

fn build(g: &Gpu, sh: &Shape, tiles_per_split: u32) -> (Fixture, Vec<gpu_core::DeviceBuffer>) {
    let (batch, n_heads, n_kv_heads, head_dim, bs) = (sh.batch, sh.n_heads, sh.n_kv_heads, sh.head_dim, sh.bs);
    let group = n_heads / n_kv_heads;
    let kv_stride = n_kv_heads * head_dim;
    let mbs = sh.seq.div_ceil(bs);
    let cap = mbs * bs;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut rng = data::rng::Rng::new(17);
    let q: Vec<f32> = (0..batch * n_heads * head_dim).map(|_| rng.next_gaussian() as f32).collect();
    let pool_len = (mbs * bs * kv_stride) as usize;
    let pk: Vec<f32> = (0..pool_len).map(|_| rng.next_gaussian() as f32).collect();
    let pv: Vec<f32> = (0..pool_len).map(|_| rng.next_gaussian() as f32).collect();
    // One block table per sequence, contiguous blocks per sequence (a real
    // steady-state serving session's own layout, matching `flash_decode_
    // bench`'s fixture) - `batch` independent sequences share nothing.
    let bt: Vec<u32> = (0..batch * mbs).collect();
    let seq_lens = vec![cap; batch as usize];

    let qb = g.storage_init("q", &q);
    let poolk = g.storage_init("pk", &pk);
    let poolv = g.storage_init("pv", &pv);
    let btb = g.storage((batch * mbs) as u64);
    g.write(&btb, &bt);
    let sl = g.storage(batch as u64);
    g.write(&sl, &seq_lens);

    let sc = g.storage((batch * n_heads * cap) as u64);
    let pr = g.storage((batch * n_heads * cap) as u64);
    let ctx_triad = g.storage((batch * n_heads * head_dim) as u64);
    let ctx_flash = g.storage((batch * n_heads * head_dim) as u64);
    let ctx_split = g.storage((batch * n_heads * head_dim) as u64);

    let scores_total = batch * n_heads * cap;
    let scores_threads = scores_total.div_ceil(PAGED_SCORES_PER_WORKGROUP) * 64;

    let triad_p0 = vec![batch, n_heads, group, head_dim, bs, kv_stride, cap, mbs, scale.to_bits()];
    let triad_p1 = vec![batch, n_heads, cap];
    let triad_p2 = vec![batch, n_heads, group, head_dim, bs, kv_stride, cap, mbs];

    // `paged_flash_decode`'s own `Params`: batch, n_heads, n_kv_heads,
    // head_dim, group, block_size (the pool's physical block size, `bs`),
    // max_bt (blocks per sequence, `mbs`).
    let flash_params = vec![batch, n_heads, n_kv_heads, head_dim, group, bs, mbs];

    let bc = 8u32; // paged_flash_decode_split's own const BC
    let ntiles = cap / bc; // cap is a multiple of bc*bs by construction here for every seq tried
    let n_splits = ntiles.div_ceil(tiles_per_split).max(1);

    let split_params = vec![batch, n_heads, n_kv_heads, head_dim, group, bs, mbs, n_splits, tiles_per_split];
    let combine_params = vec![batch, n_heads, head_dim, n_splits];

    let part_m = g.storage((batch * n_heads * n_splits) as u64);
    let part_l = g.storage((batch * n_heads * n_splits) as u64);
    let part_o = g.storage((batch * n_heads * n_splits * head_dim) as u64);

    let fx = Fixture {
        triad_steps_params: (triad_p0, triad_p1, triad_p2),
        scores_threads,
        flash_params,
        split_params,
        combine_params,
        n_splits,
    };
    // Keep every buffer alive for the caller's whole measurement window.
    let bufs = vec![qb, poolk, poolv, btb, sl, sc, pr, ctx_triad, ctx_flash, ctx_split, part_m, part_l, part_o];
    (fx, bufs)
}

/// `kernel index -> Vec<gpu_core::Step>` closures, one per arm, so the
/// round-robin loop below just calls each in turn without duplicating the
/// dispatch code per rep.
fn triad_steps(g: &Gpu, bufs: &[gpu_core::DeviceBuffer], fx: &Fixture) -> Vec<gpu_core::Step> {
    let (qb, poolk, _poolv, btb, sl, sc, pr, ctx_triad, ..) =
        (&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &bufs[5], &bufs[6], &bufs[7]);
    let (p0, p1, p2) = &fx.triad_steps_params;
    vec![
        g.step(0, &[qb, poolk, btb, sl, sc], p0, fx.scores_threads),
        g.step(1, &[sc, sl, pr], p1, p1[0] * p1[1]),
        g.step(2, &[pr, &bufs[2], btb, sl, ctx_triad], p2, p2[0] * p2[1] * p2[3]),
    ]
}
fn flash_steps(g: &Gpu, bufs: &[gpu_core::DeviceBuffer], fx: &Fixture) -> Vec<gpu_core::Step> {
    let (qb, poolk, poolv, btb, sl, ctx_flash) = (&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &bufs[8]);
    let p = &fx.flash_params;
    // One workgroup per (batch, head).
    vec![g.dispatch(3, &[qb, poolk, poolv, btb, sl, ctx_flash], p, gpu_core::Dispatch::Workgroups(p[0] * p[1]))]
}
fn split_steps(g: &Gpu, bufs: &[gpu_core::DeviceBuffer], fx: &Fixture) -> Vec<gpu_core::Step> {
    let (qb, poolk, poolv, btb, sl) = (&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4]);
    let (part_m, part_l, part_o, ctx_split) = (&bufs[10], &bufs[11], &bufs[12], &bufs[9]);
    let sp = &fx.split_params;
    let split_wgs = sp[0] * sp[1] * fx.n_splits; // one workgroup per (batch, head, split)
    let cp = &fx.combine_params;
    let combine_threads = cp[0] * cp[1] * 128; // batch * n_heads * 128
    vec![
        g.dispatch(5, &[qb, poolk, poolv, btb, sl, part_m, part_l, part_o], sp, gpu_core::Dispatch::Workgroups(split_wgs)),
        g.step(6, &[part_m, part_l, part_o, ctx_split], cp, combine_threads),
    ]
}

const REPS: usize = 20;

/// Round-robin (triad, flash, split), min-of-`REPS` each - `bench_matmul.rs`'s
/// own `time_arms` precedent for why round-robin rather than back-to-back
/// per-arm batches: run-to-run drift on a shared box is otherwise attributed
/// entirely to whichever arm runs second/third.
fn measure(g: &Gpu, bufs: &[gpu_core::DeviceBuffer], fx: &Fixture) -> (f64, f64, f64) {
    // Warm every arm once (pipeline creation / JIT, not timed).
    once(g, &triad_steps(g, bufs, fx));
    once(g, &flash_steps(g, bufs, fx));
    once(g, &split_steps(g, bufs, fx));

    let (mut t_triad, mut t_flash, mut t_split) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for _ in 0..REPS {
        t_triad = t_triad.min(once(g, &triad_steps(g, bufs, fx)));
        t_flash = t_flash.min(once(g, &flash_steps(g, bufs, fx)));
        t_split = t_split.min(once(g, &split_steps(g, bufs, fx)));
    }
    (t_triad, t_flash, t_split)
}

/// Main sweep: batch x seq, `tiles_per_split` chosen so `n_splits ~= 8`
/// regardless of `cap` (a fixed SPLIT COUNT, not a fixed tile-per-split
/// constant, so the comparison at seq=512 and seq=4096 both exercise "the
/// same relative occupancy boost", not a boost that mechanically shrinks as
/// `cap` grows for a fixed token-per-split size).
#[test]
#[ignore]
fn split_vs_flash_vs_triad_across_batch_and_seq() {
    let g = Gpu::new(&[
        ("paged_decode_scores_wg", kernels::PAGED_DECODE_SCORES_WG),
        ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
        ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
        ("paged_flash_decode", kernels::PAGED_FLASH_DECODE),
        ("_unused4", kernels::PAGED_FLASH_DECODE_I8),
        ("paged_flash_decode_split", kernels::PAGED_FLASH_DECODE_SPLIT),
        ("paged_flash_decode_combine", kernels::PAGED_FLASH_DECODE_COMBINE),
    ]);

    let (n_heads, n_kv_heads, head_dim, bs) = (16u32, 8u32, 128u32, 16u32);
    let desired_splits = 8u32;

    // Ramp on the split kernel's own steps (the arm under test) before ANY
    // shape is timed - checklist §E.0b, `kq_gemv_reg_speed_bench.rs`'s own
    // discipline.
    {
        let sh = Shape { batch: 32, n_heads, n_kv_heads, head_dim, bs, seq: 512 };
        let ntiles = (sh.seq.div_ceil(bs) * bs) / 8;
        let tps = ntiles.div_ceil(desired_splits).max(1);
        let (fx, bufs) = build(&g, &sh, tps);
        ramp(&g, &split_steps(&g, &bufs, &fx), Duration::from_secs(3));
    }

    println!("\nM2.7 split-key FlashDecode: triad vs flash (unsplit, disabled) vs split+combine");
    println!("shape: n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim}, desired_splits~={desired_splits}\n");
    for &seq in &[512u32, 4096] {
        for &batch in &[1u32, 2, 4, 8, 32, 128] {
            let sh = Shape { batch, n_heads, n_kv_heads, head_dim, bs, seq };
            let ntiles = (sh.seq.div_ceil(bs) * bs) / 8;
            let tps = ntiles.div_ceil(desired_splits).max(1);
            let (fx, bufs) = build(&g, &sh, tps);
            let n_splits = fx.n_splits;
            let (t_triad, t_flash, t_split) = measure(&g, &bufs, &fx);
            println!(
                "seq={seq:>4} batch={batch:>3}  triad {:>8.4} ms  flash {:>8.4} ms  split(n={n_splits:>3}) {:>8.4} ms   split/triad {:.2}x  split/flash {:.2}x",
                t_triad * 1e3,
                t_flash * 1e3,
                t_split * 1e3,
                t_split / t_triad,
                t_split / t_flash,
            );
        }
    }
}

/// Focused sweep at the worst-case shape (`batch=1`, `seq=4096`: the lowest
/// baseline occupancy AND the longest serialised tile walk) across several
/// `tiles_per_split` choices, so "no win at desired_splits=8" cannot hide a
/// win at a different split granularity.
#[test]
#[ignore]
fn split_granularity_sweep_at_worst_case_shape() {
    let g = Gpu::new(&[
        ("paged_decode_scores_wg", kernels::PAGED_DECODE_SCORES_WG),
        ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
        ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
        ("paged_flash_decode", kernels::PAGED_FLASH_DECODE),
        ("_unused4", kernels::PAGED_FLASH_DECODE_I8),
        ("paged_flash_decode_split", kernels::PAGED_FLASH_DECODE_SPLIT),
        ("paged_flash_decode_combine", kernels::PAGED_FLASH_DECODE_COMBINE),
    ]);
    let (n_heads, n_kv_heads, head_dim, bs) = (16u32, 8u32, 128u32, 16u32);
    let sh = Shape { batch: 1, n_heads, n_kv_heads, head_dim, bs, seq: 4096 };
    let ntiles = (sh.seq.div_ceil(bs) * bs) / 8;

    {
        let (fx, bufs) = build(&g, &sh, ntiles.div_ceil(8).max(1));
        ramp(&g, &split_steps(&g, &bufs, &fx), Duration::from_secs(3));
    }

    println!("\nM2.7 split-granularity sweep @ batch=1 seq=4096 (worst-case occupancy shape)\n");
    for &desired_splits in &[2u32, 4, 8, 16, 32, 64] {
        let tps = ntiles.div_ceil(desired_splits).max(1);
        let sh = Shape { batch: 1, n_heads, n_kv_heads, head_dim, bs, seq: 4096 };
        let (fx, bufs) = build(&g, &sh, tps);
        let n_splits = fx.n_splits;
        let (t_triad, t_flash, t_split) = measure(&g, &bufs, &fx);
        println!(
            "desired_splits={desired_splits:>3} (actual n_splits={n_splits:>3}, tiles_per_split={tps:>3})  triad {:>8.4} ms  flash {:>8.4} ms  split {:>8.4} ms   split/triad {:.2}x  split/flash {:.2}x",
            t_triad * 1e3,
            t_flash * 1e3,
            t_split * 1e3,
            t_split / t_triad,
            t_split / t_flash,
        );
    }
}
