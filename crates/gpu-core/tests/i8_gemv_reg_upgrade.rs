// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the `matmul_i8_gemv` -> `matmul_i8_gemv_reg` transparent
//! upgrade (`gpu_core::upgrade`), on real hardware. The int8 twin of
//! `gemv_reg_upgrade.rs`, and it holds the same two claims for the same
//! reason: every model in this tree inherits the substitution without knowing
//! it happened, so there is no call site where a reviewer would see a
//! tolerance being introduced.
//!
//! Bit-identity between the pair is still asserted on the raw bits, but it is
//! now a property of the two kernels doing the identical operations in the
//! identical order rather than of integer associativity: the weight scale is
//! per 32-element GROUP of K, so each word is converted and scaled as it is
//! consumed (see `matmul_i8_gemv.wgsl`'s header) and the accumulator is f32.
//!
//! The oracle is still an oracle, not a second opinion: the per-group sums are
//! computed on the host in `i64` (exact) and folded in `f64`. What that cannot
//! be is EXACT to the last bit, because the kernel's own fold is f32 over a
//! stride-64 partition. So the tolerance is scaled by the sum of the terms'
//! MAGNITUDES - the backward-stable bound - rather than by the (possibly
//! heavily cancelled) result, which is the difference between a tolerance that
//! means something and one fitted to what passed.

use gpu_core::Gpu;

/// `matmul_i8_gemv` is the upgraded slot; `matmul_i8_gemv_ref` is the SAME
/// source under a name the upgrade table does not know (`upgrade::UPGRADES`
/// keys on the registered NAME), so one handle can dispatch both the upgraded
/// and the un-upgraded form in the same submit and compare them.
const KERNELS: &[(&str, &str)] =
    &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV), ("matmul_i8_gemv_ref", kernels::MATMUL_I8_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// `matmul_i8_gemv`'s own contract bound (`REQUIRES m <= 32`), and therefore
/// the last `MREG` bucket.
const MAX_ROWS: u32 = 32;

/// Deterministic signed bytes packed 4-per-`u32`, little-endian - the layout
/// `dot4I8Packed` reads and `model::int8::quantize_weight` writes. Generated
/// directly rather than by quantising floats: these kernels consume ALREADY
/// packed int8, so packing real weights here would test the quantiser, and
/// `brain-model` is not reachable from this crate anyway (it depends on it).
fn packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                // Full signed range including the extremes, so a lane-order or
                // sign-extension mistake cannot hide in small magnitudes.
                let b = ((r.next_u32() % 255) as i32 - 127) as i8;
                w |= (b as u8 as u32) << (8 * lane);
            }
            w
        })
        .collect()
}

fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect()
}

/// Packed `u32` words per weight-scale group (`model::int8::GROUP` / 4).
const WPG: usize = 8;

/// `out[m,n] = sx[m] * sum_g (sum_{k in g} xq . wq) * sw[n,g]`, the inner sums
/// in `i64` and the fold in `f64`. Returns `(value, magnitude)` per element,
/// the magnitude being `sum_g |term|` - what a floating-point fold of these
/// terms can be off by, per unit of rounding.
fn oracle(xq: &[u32], wq: &[u32], sx: &[f32], sw: &[f32], m: usize, kg: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let lanes = |w: u32| (0..4).map(move |l| ((w >> (8 * l)) & 0xff) as u8 as i8 as i64);
    let ng = kg / WPG;
    let mut out = vec![0f32; m * n];
    let mut mag = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            let mut amag = 0f64;
            for gr in 0..ng {
                let mut ia = 0i64;
                for g in gr * WPG..gr * WPG + WPG {
                    ia += lanes(xq[mi * kg + g]).zip(lanes(wq[ni * kg + g])).map(|(a, b)| a * b).sum::<i64>();
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
fn run(gpu: &Gpu, m: u32, kg: u32, n: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(kg as usize % WPG, 0, "kg must be a whole number of weight-scale groups");
    let ng = kg / WPG as u32;
    let xq = packed((m * kg) as usize, u64::from(m) * 7 + 1);
    let wq = packed((n * kg) as usize, u64::from(n) * 13 + 3);
    let sx = scales(m as usize, u64::from(m) + 101);
    let sw = scales((n * ng) as usize, u64::from(n) + 202);

    let xb = gpu.storage(u64::from(m * kg));
    let wb = gpu.storage(u64::from(n * kg));
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
            gpu.step(K_GEMV, &[&xb, &wb, &sxb, &swb, &a], &[m, kg, n], n * 64),
            gpu.step(K_REF, &[&xb, &wb, &sxb, &swb, &b], &[m, kg, n], n * 64),
        ],
    );
    gpu.poll_wait();
    let (want, mag) = oracle(&xq, &wq, &sx, &sw, m as usize, kg as usize, n as usize);
    (gpu.read(&a, (m * n) as usize), gpu.read(&b, (m * n) as usize), want, mag)
}

/// The row is ACTIVE on this device, with one appended pipeline per bucket.
///
/// Without this, the bit-identity test below would pass trivially on a handle
/// where the upgrade never fired - both slots would simply be
/// `matmul_i8_gemv`.
#[test]
fn the_upgrade_is_active_and_carries_the_whole_bucket_ladder() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("matmul_i8_gemv needs workgroup reductions and a packed int8 dot");
        return;
    }
    assert_eq!(
        gpu.physical_kernel_names(K_GEMV),
        vec![
            "matmul_i8_gemv_reg#MREG=1",
            "matmul_i8_gemv_reg#MREG=2",
            "matmul_i8_gemv_reg#MREG=4",
            "matmul_i8_gemv_reg#MREG=8",
            "matmul_i8_gemv_reg#MREG=16",
            "matmul_i8_gemv_reg#MREG=32",
        ],
        "the shape-specialised int8 GEMV upgrade must be active on a device that selects it"
    );
    // The reference alias is deliberately NOT upgraded - it is what makes the
    // comparison below a comparison.
    assert_eq!(gpu.physical_kernel_names(K_REF), vec!["matmul_i8_gemv_ref"]);
}

/// BYTE-identical at every row count the kernel supports, and EXACT against
/// the integer oracle - so a bit-identical pair cannot be identically wrong.
#[test]
fn the_register_kernel_is_byte_identical_and_exact() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("no packed int8 dot on this device");
        return;
    }
    // `kg` spans many 64-wide strides (so the fold really folds 64 non-trivial
    // partials), plus one `kg` that is a whole number of scale groups but not
    // of the 64-wide stride, and one `n` that is not a multiple of anything.
    // Every `kg` is a multiple of 8 - K must be a whole number of 32-element
    // weight-scale groups.
    for (kg, n) in [(128u32, 384u32), (96, 512), (136, 129), (16, 71)] {
        for m in 1..=MAX_ROWS {
            let (up, refr, want, mag) = run(&gpu, m, kg, n);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&up),
                bits(&refr),
                "matmul_i8_gemv_reg must be BYTE-identical to matmul_i8_gemv at m={m}, kg={kg}, n={n} - \
                 both form and fold the same terms in the same order, so a difference here is a real defect, \
                 not rounding"
            );
            // Each group's sum is exact integer; the cross-group fold is f32,
            // so the bound is a few roundings of the terms' MAGNITUDE - not of
            // the cancelled result, and not a number fitted to what passed.
            for (i, ((g, w), mg)) in up.iter().zip(&want).zip(&mag).enumerate() {
                let tol = mg * 1e-5 + 1e-6;
                assert!(
                    (g - w).abs() <= tol,
                    "m={m} kg={kg} n={n} element {i}: got {g:e}, oracle {w:e} (magnitude {mg:e}) - each group's \
                     sum is exact, so a miss this large is a lane-order, sign-extension or scale-indexing error, \
                     not noise"
                );
            }
        }
    }
}
