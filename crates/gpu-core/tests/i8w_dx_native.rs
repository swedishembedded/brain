// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `matmul_i8w_dx` (`gpu_core::Fused::I8wDx`): the input gradient
//! `dX = dY @ deq(W)` of a linear whose frozen weight is resident as packed
//! int8 with group-32 scales - the backward a device trainer runs over an int8
//! base.
//!
//! Swedish Embedded AB implements quantised-base fine-tuning on GPUs for its
//! clients. If your team needs expertise in training adapters against the
//! int8 model you actually serve, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! There is no WGSL twin to match bits against, so the kernel is held to an
//! f64 oracle over the SAME dequantised weight, `q * sw` rounded to fp32 as
//! the kernel stages it: every element within the fp32 summation bound
//! `n * eps * sum_l |dy * w|` (plus one rounding of the prior value in the
//! accumulating mode). A wrong scale group, a dropped column or a misread int8
//! sign is orders of magnitude outside that.

use gpu_core::{Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const SENTINEL: u32 = 0x7fc0_dead;

fn device() -> Option<Gpu> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.has_fused(Fused::I8wDx) {
        brain_testutil::skip_unavailable("the native int8-weight input gradient is not offered on this device");
        return None;
    }
    Some(gpu)
}

/// Packed signed bytes over the full range, and group scales of both
/// magnitudes, as `model::int8` lays them out: `[n, k/4]` words, `[n, k/32]`.
fn weight(n: usize, k: usize, seed: u64) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut r = data::rng::Lcg::new(seed);
    let wq: Vec<u32> = (0..n * k / 4).map(|_| r.next_u32()).collect();
    let sw: Vec<f32> = (0..n * k / 32).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 2e-5).collect();
    let mut deq = vec![0f32; n * k];
    for row in 0..n {
        for c in 0..k {
            let q = ((wq[row * k / 4 + c / 4] >> (8 * (c % 4))) & 0xff) as u8 as i8;
            deq[row * k + c] = q as f32 * sw[row * k / 32 + c / 32];
        }
    }
    (wq, sw, deq)
}

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..len).map(|_| (r.next_u32() % 20_001) as f32 / 10_000.0 - 1.0).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A buffer with `lead` sentinel words before the window and a few after.
fn windowed(gpu: &Gpu, words: &[u32], lead: u64) -> (backend_api::DeviceBuffer, (u64, u64), usize) {
    let total = words.len() + lead as usize + 7;
    let b = gpu.storage(total as u64);
    gpu.write(&b, &vec![SENTINEL; total]);
    gpu.write_at(&b, lead, words);
    (b, (lead, words.len() as u64), total)
}

fn check(gpu: &Gpu, m: usize, k: usize, n: usize, accumulate: bool, lead: u64) {
    let seed = (m * 31 + k * 7 + n) as u64;
    let dy = values(m * n, seed + 1);
    let (wq, sw, deq) = weight(n, k, seed + 2);
    let prior = values(m * k, seed + 3);
    let (bdy, rdy, _) = windowed(gpu, &bits(&dy), lead);
    let (bwq, rwq, _) = windowed(gpu, &wq, lead);
    let (bsw, rsw, _) = windowed(gpu, &bits(&sw), lead);
    let (bout, rout, total) = windowed(gpu, &bits(&prior), lead);
    let params = [m as u32, k as u32, n as u32, u32::from(accumulate)];
    let step = gpu
        .fused_step_sliced(Fused::I8wDx, &[&bdy, &bwq, &bsw, &bout], &[rdy, rwq, rsw, rout], &params)
        .expect("the kernel serves this shape");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let raw: Vec<u32> = gpu.read(&bout, total).iter().map(|f| f.to_bits()).collect();
    let (lo, hi) = (lead as usize, lead as usize + m * k);
    assert!(raw[..lo].iter().chain(&raw[hi..]).all(|&v| v == SENTINEL), "wrote outside its window at {m}x{k}x{n}");
    let eps = f32::EPSILON as f64;
    for row in 0..m {
        for c in 0..k {
            let (mut want, mut mag) = (0f64, 0f64);
            for l in 0..n {
                let t = dy[row * n + l] as f64 * deq[l * k + c] as f64;
                want += t;
                mag += t.abs();
            }
            let p = prior[row * k + c] as f64;
            if accumulate {
                want += p;
            }
            let got = f32::from_bits(raw[lo + row * k + c]) as f64;
            let bound = (n as f64 + 2.0) * eps * mag + if accumulate { eps * (p.abs() + want.abs()) } else { 0.0 } + 1e-30;
            assert!(
                (got - want).abs() <= bound,
                "m={m} k={k} n={n} acc={accumulate}: [{row},{c}] got {got} want {want} (|delta| {:.3e} > bound {bound:.3e})",
                (got - want).abs()
            );
        }
    }
}

#[test]
fn the_int8_weight_input_gradient_matches_an_f64_oracle() {
    let Some(gpu) = device() else { return };
    for accumulate in [false, true] {
        for (m, k, n) in [(1usize, 32usize, 1usize), (5, 64, 7), (127, 160, 129), (128, 128, 128), (130, 288, 33), (200, 96, 300)] {
            for lead in [0u64, 4, 5] {
                check(&gpu, m, k, n, accumulate, lead);
            }
        }
    }
}

#[test]
fn shapes_it_does_not_serve_are_declined() {
    let Some(gpu) = device() else { return };
    let b = gpu.storage(64);
    assert!(gpu.fused_step(Fused::I8wDx, &[&b, &b, &b, &b], &[4, 48, 4, 0]).is_none(), "K not a whole number of scale groups");
    assert!(gpu.fused_step(Fused::I8wDx, &[&b, &b, &b, &b], &[4, 32, 4]).is_none(), "short params");
    assert!(gpu.fused_step(Fused::I8wDx, &[&b, &b, &b, &b], &[0, 32, 4, 0]).is_none(), "no rows");
}
