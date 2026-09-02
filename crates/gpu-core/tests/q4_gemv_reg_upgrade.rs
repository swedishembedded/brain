// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the `matmul_q4_gemv` -> `matmul_q4_gemv_reg` transparent
//! upgrade (`gpu_core::upgrade`), on real hardware. The W4A8 (q4) twin of
//! `i8_gemv_reg_upgrade.rs`, and it holds the same two claims for the same
//! reason: every model in this tree inherits the substitution without
//! knowing it happened, so there is no call site where a reviewer would see
//! a tolerance being introduced.
//!
//! Bit-identity between the pair is asserted on the raw bits, and it is
//! EXACT rather than merely close for a stronger reason than the int8 case:
//! the eight nibble x int8-byte products a weight word contributes are
//! integers of bounded magnitude (`|nibble| <= 8`, `|byte| <= 127`, so each
//! product is `<= 1016` and the eight together are `<= 8128`, far inside
//! `i32`), so their sum does not depend on whether it is formed via eight
//! scalar MACs (`matmul_q4_gemv`) or two `dot4I8Packed` calls
//! (`matmul_q4_gemv_reg`) - integer addition without overflow is exactly
//! associative. The per-group integer sum is therefore bit-for-bit the same
//! between the two kernels, and the f32 fold across words/groups that
//! follows is unchanged in order, so the two kernels are bit-identical BY
//! CONSTRUCTION, not by care.
//!
//! The oracle is still an oracle, not a second opinion: the per-group sums
//! are computed on the host in `i64` (exact) and folded in `f64`. What that
//! cannot be is EXACT to the last bit, because the kernel's own fold is f32
//! over a stride-64 partition - so the tolerance is scaled by the sum of the
//! terms' MAGNITUDES (the backward-stable bound), not by the (possibly
//! heavily cancelled) result.

use gpu_core::Gpu;

/// `matmul_q4_gemv` is the upgraded slot; `matmul_q4_gemv_ref` is the SAME
/// source under a name the upgrade table does not know (`upgrade::UPGRADES`
/// keys on the registered NAME), so one handle can dispatch both the
/// upgraded and the un-upgraded form in the same submit and compare them.
const KERNELS: &[(&str, &str)] =
    &[("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV), ("matmul_q4_gemv_ref", kernels::MATMUL_Q4_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// `matmul_q4_gemv`'s own contract bound (`REQUIRES m <= 32`), and therefore
/// the last `MREG` bucket.
const MAX_ROWS: u32 = 32;

/// Deterministic signed int8 activations packed 4-per-`u32`, little-endian -
/// the layout `dot4I8Packed` reads and `model::int8::quant_pack` writes.
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

/// Deterministic signed int4 (nibble) weights packed 8-per-`u32`,
/// little-endian along K - `model::int4::quantize_weight_q4`'s layout. Full
/// signed nibble range `-8..=7` (all 16 bit patterns reachable), so a
/// sign-extension mistake at either extreme cannot hide in a small magnitude.
fn packed_w(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for nib in 0..8 {
                let v = ((r.next_u32() % 16) as i32 - 8) as i8;
                w |= (v as u8 as u32 & 0xF) << (4 * nib);
            }
            w
        })
        .collect()
}

fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect()
}

/// One group is 32 logical K values: `model::int8::GROUP`, shared by the q4
/// tier (`model::int4`'s own module doc - Q4_0's block is Q8_0's).
const GROUP: usize = 32;

fn nibble_at(wq: &[u32], row: usize, kgw: usize, kidx: usize) -> i64 {
    let word = wq[row * kgw + kidx / 8];
    let raw = ((word >> (4 * (kidx % 8))) & 0xF) as u8;
    (((raw << 4) as i8) >> 4) as i64
}

fn byte_at(xq: &[u32], row: usize, kgx: usize, kidx: usize) -> i64 {
    let word = xq[row * kgx + kidx / 4];
    let raw = ((word >> (8 * (kidx % 4))) & 0xFF) as u8;
    (raw as i8) as i64
}

/// `out[m,n] = sx[m] * sum_g (sum_{k in g} nibble(w) * byte(x)) * sw[n,g]`,
/// the inner sums in `i64` and the fold in `f64`. Returns `(value,
/// magnitude)` per element, the magnitude being `sum_g |term|` - what a
/// floating-point fold of these terms can be off by, per unit of rounding.
fn oracle(xq: &[u32], wq: &[u32], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let kgx = k / 4;
    let kgw = k / 8;
    let ng = k / GROUP;
    let mut out = vec![0f32; m * n];
    let mut mag = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            let mut amag = 0f64;
            for gr in 0..ng {
                let mut ia = 0i64;
                for kidx in gr * GROUP..gr * GROUP + GROUP {
                    ia += nibble_at(wq, ni, kgw, kidx) * byte_at(xq, mi, kgx, kidx);
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
    assert_eq!(k as usize % GROUP, 0, "k must be a whole number of weight-scale groups");
    let (kgx, kgw, ng) = (k / 4, k / 8, k / GROUP as u32);
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

/// The row is ACTIVE on this device, with one appended pipeline per bucket.
///
/// Without this, the bit-identity test below would pass trivially on a
/// handle where the upgrade never fired - both slots would simply be
/// `matmul_q4_gemv`.
#[test]
fn the_upgrade_is_active_and_carries_the_whole_bucket_ladder() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("matmul_q4_gemv needs workgroup reductions and a packed int8 dot");
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
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("no packed int8 dot on this device");
        return;
    }
    // `k` spans many 64-wide strides (so the fold really folds 64
    // non-trivial partials), plus one `k` that is a whole number of scale
    // groups but not of the 64-wide stride, and one `n` that is not a
    // multiple of anything. Every `k` is a multiple of 32.
    for (k, n) in [(1024u32, 384u32), (768, 512), (1088, 129), (128, 71)] {
        for m in 1..=MAX_ROWS {
            let (up, refr, want, mag) = run(&gpu, m, k, n);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&up),
                bits(&refr),
                "matmul_q4_gemv_reg must be BYTE-identical to matmul_q4_gemv at m={m}, k={k}, n={n} - \
                 both form and fold the same terms in the same order, so a difference here is a real \
                 defect, not rounding"
            );
            // Each group's sum is exact integer; the cross-group fold is f32,
            // so the bound is a few roundings of the terms' MAGNITUDE - not
            // of the cancelled result, and not a number fitted to what
            // passed.
            for (i, ((g, w), mg)) in up.iter().zip(&want).zip(&mag).enumerate() {
                let tol = mg * 1e-5 + 1e-6;
                assert!(
                    (g - w).abs() <= tol,
                    "m={m} k={k} n={n} element {i}: got {g:e}, oracle {w:e} (magnitude {mg:e}) - each \
                     group's sum is exact, so a miss this large is a lane-order, sign-extension or \
                     scale-indexing error, not noise"
                );
            }
        }
    }
}
