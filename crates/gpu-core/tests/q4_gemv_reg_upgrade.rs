// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the `matmul_q4_gemv` -> `matmul_q4_gemv_reg` transparent
//! upgrade (`gpu_core::upgrade`). The W4A8 twin of `i8_gemv_reg_upgrade.rs`:
//! `crates/model/tests/matmul_q4_speed_bench.rs`'s
//! `gemv_vs_gemv_reg_at_qwen35_decode_shapes` measured `_reg` winning
//! (1.55-1.88x) at every real qwen35 decode shape - unlike the earlier,
//! un-templated measurement `gpu_core::upgrade`'s own module doc records as a
//! killed hypothesis, the per-bucket templated builds this table actually
//! dispatches do not regress. This file is the correctness half that measurement
//! doesn't cover: same bar as every other row in `upgrade::UPGRADES` -
//! byte-identical results, not "close".
//!
//! Activations are packed int8 (4 lanes/`u32`, same layout `matmul_i8_gemv`
//! reads); weights are packed int4 (8 nibbles/`u32`, sign-extended via
//! `shl`+arithmetic `shr` per `matmul_q4_gemv.wgsl`'s header). The oracle sums
//! each 32-element weight-scale GROUP (4 packed words = 8 nibbles x 4 words)
//! exactly in `i64`, then folds groups in `f64` - the kernel's own fold is f32
//! over a stride-64 partition, so the tolerance is scaled by the terms'
//! MAGNITUDE, mirroring `i8_gemv_reg_upgrade.rs`'s reasoning exactly.

use gpu_core::Gpu;

/// `matmul_q4_gemv` is the upgraded slot; `matmul_q4_gemv_ref` is the SAME
/// source under a name `upgrade::UPGRADES` does not know, so one handle
/// dispatches both the upgraded and the un-upgraded form in the same submit.
const KERNELS: &[(&str, &str)] =
    &[("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV), ("matmul_q4_gemv_ref", kernels::MATMUL_Q4_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// `matmul_q4_gemv`'s own contract bound (`REQUIRES m <= 32`).
const MAX_ROWS: u32 = 32;

/// Packed signed int8 activations, 4 lanes/`u32`, little-endian - identical
/// layout to `i8_gemv_reg_upgrade.rs`'s `packed()`. Generated directly rather
/// than via `model::int8`: this crate cannot depend on `brain-model` (the
/// dependency runs the other way).
fn packed_x(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                let b = ((r.next_u32() % 255) as i32 - 127) as i8;
                w |= (b as u8 as u32) << (8 * lane);
            }
            w
        })
        .collect()
}

/// Packed signed int4 weights, 8 nibbles/`u32`, nibble `b` in bits
/// `[4b, 4b+4)`. Swept over the FULL signed nibble range `-8..=7` (not the
/// real quantizer's `-7..=7`) - this is a kernel bit-identity test, not a
/// quantizer range test, and the wider sweep is strictly more coverage of
/// the sign-extension path.
fn packed_w(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..8 {
                let n = ((r.next_u32() % 16) as i32 - 8) as i8;
                w |= (n as u8 as u32 & 0xF) << (4 * lane);
            }
            w
        })
        .collect()
}

fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect()
}

/// Packed `u32` weight words per weight-scale group (`model::int8::GROUP`(32) / 8
/// nibbles-per-word), matching `matmul_q4_gemv.wgsl`'s own `WPG4`.
const WPG4: usize = 4;

/// `out[m,n] = sx[m] * sum_g (sum_{k in g} xq . wq_nibbles) * sw[n,g]`, group
/// sums exact in `i64`, folded in `f64`. Returns `(value, magnitude)` per
/// element, `magnitude` being `sum_g |term|`.
fn oracle(xq: &[u32], wq: &[u32], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let kgx = k / 4; // x words per row
    let kgw = k / 8; // w words per row
    let ng = kgw / WPG4;
    let x_lane = |w: u32, lane: usize| (((w >> (8 * lane)) & 0xff) as u8 as i8) as i64;
    let w_nibble = |w: u32, b: usize| {
        let shifted = (w as i32) << (28 - 4 * b);
        i64::from(shifted >> 28)
    };
    let mut out = vec![0f32; m * n];
    let mut mag = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            let mut amag = 0f64;
            for gr in 0..ng {
                let mut ia = 0i64;
                for wi in 0..WPG4 {
                    let g = gr * WPG4 + wi;
                    let wv = wq[ni * kgw + g];
                    let xbase = mi * kgx + 2 * g;
                    let (xw0, xw1) = (xq[xbase], xq[xbase + 1]);
                    for b in 0..8usize {
                        let xb = if b < 4 { x_lane(xw0, b) } else { x_lane(xw1, b - 4) };
                        ia += w_nibble(wv, b) * xb;
                    }
                }
                let term = ia as f64 * sw[ni * ng + gr] as f64;
                acc += term;
                amag += term.abs();
            }
            out[mi * n + ni] = (acc * sx[mi] as f64) as f32;
            mag[mi * n + ni] = (amag * sx[mi] as f64) as f32;
        }
    }
    (out, mag)
}

/// Run one shape through both slots, returning
/// `(upgraded, reference, oracle, oracle_magnitude)`.
fn run(gpu: &Gpu, m: u32, k: u32, n: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(k as usize % 32, 0, "k must be a whole number of 32-element weight-scale groups");
    let (kgx, kgw) = (k / 4, k / 8);
    let ng = kgw / WPG4 as u32;
    let xq = packed_x((m * kgx) as usize, u64::from(m) * 7 + 1);
    let wq = packed_w((n * kgw) as usize, u64::from(n) * 13 + 3);
    let sx = scales(m as usize, u64::from(m) + 101);
    let sw = scales((n * ng) as usize, u64::from(n) + 202);

    let xb = gpu.storage(u64::from(m * kgx));
    let wb = gpu.storage(u64::from(n * kgw));
    let sxb = gpu.storage(u64::from(m));
    let swb = gpu.storage(u64::from(n * ng));
    gpu.write(&xb, &xq);
    gpu.write(&wb, &wq);
    gpu.write_f32(&sxb, &sx);
    gpu.write_f32(&swb, &sw);
    let a = gpu.storage(u64::from(m * n));
    let b = gpu.storage(u64::from(m * n));
    gpu.submit(
        &[],
        &[
            gpu.step(K_GEMV, &[&xb, &wb, &sxb, &swb, &a], &[m, k, n], n * 64),
            gpu.step(K_REF, &[&xb, &wb, &sxb, &swb, &b], &[m, k, n], n * 64),
        ],
    );
    gpu.poll_wait();
    let (want, mag) = oracle(&xq, &wq, &sx, &sw, m as usize, k as usize, n as usize);
    (gpu.read(&a, (m * n) as usize), gpu.read(&b, (m * n) as usize), want, mag)
}

/// The row is ACTIVE on this device, with one appended pipeline per bucket -
/// without this, the bit-identity test below would pass trivially on a
/// handle where the upgrade never fired.
#[test]
fn the_upgrade_is_active_and_carries_the_whole_bucket_ladder() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("matmul_q4_gemv needs a packed int8 dot");
        return;
    }
    assert_eq!(
        gpu.physical_kernel_names(K_GEMV),
        vec![
            "matmul_q4_gemv_reg#MREG=1",
            "matmul_q4_gemv_reg#MREG=2",
            "matmul_q4_gemv_reg#MREG=4",
            "matmul_q4_gemv_reg#MREG=8",
            "matmul_q4_gemv_reg#MREG=16",
            "matmul_q4_gemv_reg#MREG=32",
        ],
        "the shape-specialised q4 GEMV upgrade must be active on a device that selects it"
    );
    // The reference alias is deliberately NOT upgraded - it is what makes the
    // comparison below a comparison.
    assert_eq!(gpu.physical_kernel_names(K_REF), vec!["matmul_q4_gemv_ref"]);
}

/// BYTE-identical at every row count the kernel supports, and EXACT against
/// the integer oracle - so a bit-identical pair cannot be identically wrong.
#[test]
fn the_register_kernel_is_byte_identical_and_exact() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("no packed int8 dot on this device");
        return;
    }
    // `k` spans many 64-word strides and one whole-group-but-not-64-word k,
    // plus one `n` that is not a multiple of anything. Every `k` is a
    // multiple of 32 - K must be a whole number of weight-scale groups, and
    // qwen35's own real shapes (5120, 6144, 10240, 12288, 17408) are all
    // multiples of 32.
    for (k, n) in [(1024u32, 384u32), (768, 512), (1088, 129), (128, 71)] {
        for m in 1..=MAX_ROWS {
            let (up, refr, want, mag) = run(&gpu, m, k, n);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&up),
                bits(&refr),
                "matmul_q4_gemv_reg must be BYTE-identical to matmul_q4_gemv at m={m}, k={k}, n={n} - \
                 both form and fold the same terms in the same order, so a difference here is a real defect, \
                 not rounding"
            );
            for (i, ((g, w), mg)) in up.iter().zip(&want).zip(&mag).enumerate() {
                let tol = mg * 1e-5 + 1e-6;
                assert!(
                    (g - w).abs() <= tol,
                    "m={m} k={k} n={n} element {i}: got {g:e}, oracle {w:e} (magnitude {mg:e}) - each group's \
                     sum is exact, so a miss this large is a nibble-order, sign-extension or scale-indexing \
                     error, not noise"
                );
            }
        }
    }
}
