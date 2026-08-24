// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the `matmul_gemv` -> `matmul_gemv_reg` transparent upgrade
//! (`gpu_core::upgrade`), on real hardware.
//!
//! The upgrade's whole justification is that the register-accumulator kernel is
//! a **drop-in**: same `Params`, same bindings, same `n * 64` thread count, and
//! BYTE-IDENTICAL results. That last claim is the one a test has to hold,
//! because every model in this tree inherits the substitution without knowing
//! it happened - there is no call site where a reviewer would see a tolerance
//! being introduced.
//!
//! It is bit-identical by construction, not by luck: the register kernel keeps
//! the same k-stride (`k = t; k += 64`), gives every output its own
//! accumulator, and folds the same 64 partials in the same ascending order.
//! Nothing is reassociated, so this gate is `assert_eq!` on the raw f32 BITS.
//! A cosine or rel_l2 comparison here would be strictly weaker and would hide
//! exactly the class of change (a reordered reduction) that this kernel pair
//! must never make - and cosine alone is weaker still, because it is scale
//! invariant: an RMSNorm-epsilon mutation elsewhere in this tree (1e-6 to
//! 1e-2, a uniform mis-scaling) scored cosine 1.000000 and was caught only by
//! a relative-L2 check.
//!
//! The reference is `kernels::MATMUL_GEMV` registered under a SECOND name. It
//! is the same source const - not a copy - and the second name is what keeps it
//! out of the upgrade table's reach (`upgrade::UPGRADES` keys on the registered
//! NAME), so one handle can dispatch both the upgraded and the un-upgraded
//! form in the same submit and compare them.

use gpu_core::Gpu;

/// The kernel set. `matmul_gemv` is the upgraded slot; `matmul_gemv_ref` is the
/// SAME source under a name the upgrade table does not know, i.e. the
/// workgroup-accumulator kernel every caller gets today.
const KERNELS: &[(&str, &str)] =
    &[("matmul_gemv", kernels::MATMUL_GEMV), ("matmul_gemv_ref", kernels::MATMUL_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// `matmul_gemv`'s own contract bound, and therefore the last `MREG` bucket.
const MAX_ROWS: u32 = 32;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    r.vec_scaled(n, 1.0)
}

/// `out = x @ Wᵀ` in f64 - an independent oracle, so a bit-identical PAIR that
/// is identically wrong still fails.
fn oracle(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += f64::from(x[mi * k + ki]) * f64::from(w[ni * k + ki]);
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Run one shape through both slots, returning `(upgraded, reference)`.
fn run(gpu: &Gpu, m: u32, k: u32, n: u32) -> (Vec<f32>, Vec<f32>) {
    let x = fill((m * k) as usize, u64::from(m) * 7 + 1);
    let w = fill((n * k) as usize, u64::from(n) * 13 + 3);
    let xb = gpu.storage(u64::from(m * k));
    let wb = gpu.storage(u64::from(n * k));
    gpu.write_f32(&xb, &x);
    gpu.write_f32(&wb, &w);
    let a = gpu.storage(u64::from(m * n));
    let b = gpu.storage(u64::from(m * n));
    gpu.submit(
        &[],
        &[
            gpu.step(K_GEMV, &[&xb, &wb, &a], &[m, k, n], n * 64),
            gpu.step(K_REF, &[&xb, &wb, &b], &[m, k, n], n * 64),
        ],
    );
    gpu.poll_wait();
    (gpu.read(&a, (m * n) as usize), gpu.read(&b, (m * n) as usize))
}

/// The row is ACTIVE on this device, with one appended pipeline per bucket.
///
/// Without this, the bit-identity test below would pass trivially on a handle
/// where the upgrade never fired - both slots would simply be `matmul_gemv`.
/// Which bucket a given `m` selects is pinned by `upgrade`'s own unit tests;
/// this asserts the other half, that the ladder reached the real device.
#[test]
fn the_upgrade_is_active_and_carries_the_whole_bucket_ladder() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable(
            "matmul_gemv is not selectable on a backend without workgroup reductions",
        );
        return;
    }
    let physical = gpu.physical_kernel_names(K_GEMV);
    assert_eq!(
        physical,
        vec![
            "matmul_gemv_reg#MREG=1",
            "matmul_gemv_reg#MREG=2",
            "matmul_gemv_reg#MREG=4",
            "matmul_gemv_reg#MREG=8",
            "matmul_gemv_reg#MREG=16",
            "matmul_gemv_reg#MREG=32",
        ],
        "the shape-specialised GEMV upgrade must be active on a device that selects it"
    );
    // The reference alias is deliberately NOT upgraded - it is what makes the
    // comparison below a comparison.
    assert_eq!(gpu.physical_kernel_names(K_REF), vec!["matmul_gemv_ref"]);
}

/// BYTE-identical at every row count the kernel supports, across the three
/// shapes a real decoder block dispatches (square, wide, tall) and one shape
/// whose `k` is not a multiple of the 64-wide stride.
#[test]
fn the_register_kernel_is_byte_identical_to_the_workgroup_one() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("no workgroup reductions on this device");
        return;
    }
    // Small enough to stay quick, large enough that `k` spans many 64-wide
    // strides (so the fold really folds 64 non-trivial partials), plus one
    // ragged `k` and one `n` that is not a multiple of anything.
    let shapes = [(512u32, 384u32), (384, 512), (517, 129), (64, 71)];
    for (k, n) in shapes {
        for m in 1..=MAX_ROWS {
            let (up, refr) = run(&gpu, m, k, n);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&up),
                bits(&refr),
                "matmul_gemv_reg must be BYTE-identical to matmul_gemv at m={m}, k={k}, n={n} - \
                 the two keep the same k-stride order and the same 64-partial fold, so a \
                 difference here means one of them reassociated"
            );
        }
    }
}

/// ...and both agree with an independent f64 oracle, so a bit-identical pair
/// cannot be identically wrong (checklist §F.5: kernel-vs-kernel agreement
/// cannot tell you which one is right).
#[test]
fn both_kernels_agree_with_a_host_f64_oracle() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("no workgroup reductions on this device");
        return;
    }
    for m in [1u32, 2, 3, 7, 16, 32] {
        let (k, n) = (137u32, 61u32);
        let x = fill((m * k) as usize, u64::from(m) * 7 + 1);
        let w = fill((n * k) as usize, u64::from(n) * 13 + 3);
        let want = oracle(&x, &w, m as usize, k as usize, n as usize);
        let (up, refr) = run(&gpu, m, k, n);
        for (label, got) in [("upgraded", &up), ("reference", &refr)] {
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0.0f64, f64::max);
            assert!(err < 1e-3, "{label} diverges from the f64 oracle at m={m}: {err:.3e}");
        }
    }
}

/// `BRAIN_NO_KERNEL_UPGRADE=1` must pin the handle back onto the kernel the
/// model registered - the A/B switch every measurement of this row was taken
/// with, and the fallback if a driver ever mishandles the register variant.
///
/// A subprocess because the switch is read ONCE per process (the policy must
/// stay fixed for a handle's lifetime), so it cannot be toggled in-test.
#[test]
fn the_env_switch_pins_the_handle_back_onto_the_registered_kernel() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["the_upgrade_is_active_and_carries_the_whole_bucket_ladder", "--exact", "--nocapture"])
        .env("BRAIN_NO_KERNEL_UPGRADE", "1")
        .output()
        .expect("re-run this test binary with the upgrade disabled");
    let text =
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(
        !out.status.success(),
        "with BRAIN_NO_KERNEL_UPGRADE=1 the ladder must be absent, so the activity assertion \
         must FAIL - if it passed, the switch no longer disables the table:\n{text}"
    );
    assert!(
        text.contains("must be active on a device that selects it") || text.contains("skip"),
        "the failure must be the ladder assertion, not some unrelated breakage:\n{text}"
    );
}
