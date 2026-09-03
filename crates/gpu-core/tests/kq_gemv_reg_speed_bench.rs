// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M13: A/B speed measurement, `matmul_kq_gemv` vs `matmul_kq_gemv_reg`,
//! across the decode regime - the number the commit message / roadmap ledger
//! entry for this milestone reports, matching `matmul_q4_speed_bench.rs`'s
//! own precedent for the same kind of claim (measured, not guessed, per
//! `AGENTS.md`'s checklist §F.6/E.0).
//!
//! `#[ignore]`d - a measurement, not a correctness gate (that is
//! `kq_gemv_reg_upgrade.rs`, in the same directory). Run manually:
//! `cargo test --release --offline -p brain-gpu-core --test
//! kq_gemv_reg_speed_bench -- --ignored --nocapture`. Bracketed with
//! `poll_wait()` via `gpu_core::profile::best_of`, so this times the device,
//! not the host (checklist §E.0).

use std::time::{Duration, Instant};

use data::rng::Lcg;
use gpu_core::Gpu;

const REPS: usize = 30;

/// Continuous dispatches of `steps` for `dur`, so a device at its DVFS idle
/// floor (checklist §E.0b - measured on an integrated Arc, the exact card
/// this bench actually runs on: 350 MHz idle vs 2150 MHz ramped, roughly 1:1
/// with achieved throughput) reaches the clock a real workload runs at
/// BEFORE anything is timed. A single warm-up dispatch (what `best_of`
/// itself does) pays pipeline creation and JIT compilation but is nowhere
/// near enough to pay DVFS.
fn ramp(g: &Gpu, steps: &[gpu_core::Step], dur: Duration) {
    let t0 = Instant::now();
    while t0.elapsed() < dur {
        g.submit(&[], steps);
        g.poll_wait();
    }
}

/// The exact `MREG` bucket ladder `gpu_core::upgrade::GEMV_MREG_BUCKETS`
/// uses (private to that crate module, duplicated here for the same reason
/// `matmul_q4_speed_bench.rs`'s own copy is: measuring production's real
/// per-`m` specialisation, never a single worst-case build - a variant
/// compiled for 32 rows is a documented regression at 1 row).
const MREG_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 32];

fn smallest_bucket_covering(m: u32) -> u32 {
    *MREG_BUCKETS.iter().find(|&&b| m <= b).expect("m must be <= the largest bucket (32)")
}

fn rand_i8_codes(seed: u64, n: usize) -> Vec<i8> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| (r.next_u32() % 201) as i32 as i8 - 100).collect()
}

fn rand_unsigned_codes(seed: u64, n: usize, bits: u32) -> Vec<i32> {
    let span = (1u32 << bits).min(32);
    let mut r = Lcg::new(seed);
    (0..n).map(|_| (r.next_u32() % span) as i32).collect()
}

fn rand_pos_f32(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| lo + (r.next_u32() % 100_000) as f32 / 100_000.0 * (hi - lo)).collect()
}

fn pack_i8_words(codes: &[i8]) -> Vec<u32> {
    codes.chunks_exact(4).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u8 as u32) << (8 * b)))).collect()
}

fn pack_affine_words(codes: &[i32], bits: u32) -> Vec<u32> {
    let per_word = 32 / bits as usize;
    let mask = (1u32 << bits) - 1;
    codes
        .chunks_exact(per_word)
        .map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u32 & mask) << (bits * b as u32))))
        .collect()
}

fn build_wsz(ds: &[f32], dm: &[f32], n: usize, ng: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * 2 * ng];
    for r in 0..n {
        for g in 0..ng {
            out[r * 2 * ng + 2 * g] = ds[r * ng + g];
            out[r * 2 * ng + 2 * g + 1] = dm[r * ng + g];
        }
    }
    out
}

fn host_group_sums(codes_x: &[i8], m: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let s: i32 = codes_x[r * k + g * group..r * k + (g + 1) * group].iter().map(|&v| v as i32).sum();
            out[r * ng + g] = s as f32;
        }
    }
    out
}

const GROUP: usize = 32;

/// A model's `d_model`-shaped decode GEMV: `k = n = 2048`, matching
/// `matmul_q4_speed_bench.rs`'s own shape choice.
#[test]
#[ignore]
fn gemv_vs_gemv_reg_across_decode_rows() {
    for &bits in &[4u32, 8u32] {
        let (kq_name, kq_src) = kernels::template::interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", bits)]).unwrap();
        let mut kernels_list = vec![(kq_name, kq_src)];
        let variants: Vec<(&'static str, &'static str)> = MREG_BUCKETS
            .iter()
            .map(|&b| {
                kernels::template::interned("matmul_kq_gemv_reg", kernels::MATMUL_KQ_GEMV_REG, &[("CODE_BITS", bits), ("MREG", b)])
                    .unwrap()
            })
            .collect();
        kernels_list.extend(variants);
        let g = Gpu::new(&kernels_list);
        if !g.caps().workgroup_reductions || !g.caps().numeric.int8_dot {
            eprintln!("skipping CODE_BITS={bits}: no packed int8 dot / workgroup reductions on this device");
            continue;
        }
        let k_gemv = g.kernel_index(kq_name).unwrap();
        let (k, n) = (2048u32, 2048u32);
        let ng = (k / GROUP as u32) as usize;

        // Ramp the device BEFORE timing anything (checklist §E.0b) - an idle
        // integrated GPU sits at its DVFS floor and a sub-second probe
        // measures that floor, not the clock a real workload runs at.
        {
            let m = 32u32;
            let codes_x = rand_i8_codes(9999, (m * k) as usize);
            let codes_w = rand_unsigned_codes(9998, (n * k) as usize, bits);
            let sx = rand_pos_f32(9997, m as usize, 0.3, 1.7);
            let ds = rand_pos_f32(9996, n as usize * ng, 0.01, 0.5);
            let dm = rand_pos_f32(9995, n as usize * ng, 0.05, 1.5);
            let xq_words = pack_i8_words(&codes_x);
            let xq = g.storage(xq_words.len() as u64);
            g.write(&xq, &xq_words);
            let wq_words = pack_affine_words(&codes_w, bits);
            let wq = g.storage(wq_words.len() as u64);
            g.write(&wq, &wq_words);
            let sxb = g.storage_init("sx", &sx);
            let wsz = g.storage_init("wsz", &build_wsz(&ds, &dm, n as usize, ng));
            let xgs = g.storage_init("xgs", &host_group_sums(&codes_x, m as usize, k as usize, GROUP));
            let out = g.storage((m * n) as u64);
            let ramp_steps = vec![g.step(k_gemv, &[&xq, &wq, &sxb, &wsz, &xgs, &out], &[m, k, n], n * 64)];
            ramp(&g, &ramp_steps, Duration::from_secs(3));
        }

        println!("\nmatmul_kq_gemv vs matmul_kq_gemv_reg (per-m MREG bucket), CODE_BITS={bits}, k={k} n={n}, m swept:\n");
        for &m in &[1u32, 2, 4, 8, 16, 32] {
            let bucket = smallest_bucket_covering(m);
            let reg_name = kernels::template::interned(
                "matmul_kq_gemv_reg",
                kernels::MATMUL_KQ_GEMV_REG,
                &[("CODE_BITS", bits), ("MREG", bucket)],
            )
            .unwrap()
            .0;
            let k_reg = g.kernel_index(reg_name).unwrap();

            let codes_x = rand_i8_codes(9200 + u64::from(m), (m * k) as usize);
            let codes_w = rand_unsigned_codes(9300 + u64::from(m), (n * k) as usize, bits);
            let sx = rand_pos_f32(9400 + u64::from(m), m as usize, 0.3, 1.7);
            let ds = rand_pos_f32(9500 + u64::from(m), n as usize * ng, 0.01, 0.5);
            let dm = rand_pos_f32(9600 + u64::from(m), n as usize * ng, 0.05, 1.5);

            let xq_words = pack_i8_words(&codes_x);
            let xq = g.storage(xq_words.len() as u64);
            g.write(&xq, &xq_words);
            let wq_words = pack_affine_words(&codes_w, bits);
            let wq = g.storage(wq_words.len() as u64);
            g.write(&wq, &wq_words);
            let sxb = g.storage_init("sx", &sx);
            let wsz = g.storage_init("wsz", &build_wsz(&ds, &dm, n as usize, ng));
            let xgs = g.storage_init("xgs", &host_group_sums(&codes_x, m as usize, k as usize, GROUP));
            let out = g.storage((m * n) as u64);

            let st_gemv = vec![g.step(k_gemv, &[&xq, &wq, &sxb, &wsz, &xgs, &out], &[m, k, n], n * 64)];
            let st_reg = vec![g.step(k_reg, &[&xq, &wq, &sxb, &wsz, &xgs, &out], &[m, k, n], n * 64)];
            let t_gemv = gpu_core::profile::best_of(&g, &st_gemv, REPS);
            let t_reg = gpu_core::profile::best_of(&g, &st_reg, REPS);
            // Dominant traffic: packed weight codes + interleaved (ds,dm).
            let bytes = u64::from(n) * u64::from(k) * u64::from(bits) / 8 + u64::from(n) * (ng as u64) * 2 * 4;
            let gbs_gemv = bytes as f64 / t_gemv / 1e9;
            let gbs_reg = bytes as f64 / t_reg / 1e9;
            println!(
                "m={m:>3}  matmul_kq_gemv {:>8.4} ms ({:>7.2} GB/s)   matmul_kq_gemv_reg {:>8.4} ms ({:>7.2} GB/s)   speedup {:.2}x",
                t_gemv * 1e3,
                gbs_gemv,
                t_reg * 1e3,
                gbs_reg,
                t_gemv / t_reg
            );
        }
    }
}
