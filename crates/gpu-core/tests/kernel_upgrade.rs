// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The transparent kernel upgrade, end to end on a real device.
//!
//! This is the test that would have caught the `crates/vae` defect: a model
//! registers the kernel it has always registered, dispatches the thread
//! count it has always dispatched, and must come out running the faster
//! kernel with the same answer. Nothing in this file names a call site —
//! that is the point.
//!
//! Its own binary, because `BRAIN_NO_KERNEL_UPGRADE` is process-wide and
//! `bench_max_abs_row.rs` needs it set.

/// A "model" that only knows about the slow kernel — exactly what
/// `crates/flux2`, `crates/qwen` and `crates/s3dit` register today.
const AS_A_MODEL_REGISTERS_IT: &[(&str, &str)] = &[("max_abs_row", kernels::MAX_ABS_ROW)];

fn fill(m: u32, k: u32) -> Vec<f32> {
    (0..(m * k) as usize)
        .map(|i| {
            let v = (((i * 37) % 197) as f32 / 197.0) - 0.5;
            if i % 811 == 0 {
                v * 23.0
            } else {
                v
            }
        })
        .collect()
}

/// Host reference for `sx[r] = max(max|x[r,:]|, 1e-8) / 127`.
///
/// Compared with a small relative tolerance, NOT bit-exactly: the shader's
/// `/ 127.0` is not required to round like the host's, so a host reference can
/// differ in the last couple of ulp for either kernel. What must be *exactly*
/// equal is the two KERNELS against each other, since `max` is associative and
/// exact — that is asserted in `bench_max_abs_row.rs`, which can register both.
fn host_scales(x: &[f32], m: u32, k: u32) -> Vec<f32> {
    (0..m as usize)
        .map(|r| {
            let a = x[r * k as usize..(r + 1) * k as usize].iter().fold(0f32, |a, &v| a.max(v.abs()));
            a.max(1e-8) / 127.0
        })
        .collect()
}

/// Within 1e-6 relative — see `host_scales`.
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6 * w.abs().max(1e-6), "{what}: row {i}: {g} vs {w}");
    }
}

/// A model that registered only `max_abs_row` gets `max_abs_rows` appended to
/// its pipeline set — **after** its own kernels, so every index it hard-codes
/// still resolves.
#[test]
fn fast_variant_is_appended_without_moving_indices() {
    let gpu = gpu_core::testgpu::dev(AS_A_MODEL_REGISTERS_IT);
    assert_eq!(gpu.kernel_name(0), Some("max_abs_row"), "the model's own index is unmoved");
    assert_eq!(
        gpu.kernel_index("max_abs_rows"),
        Some(1),
        "the drop-in fast variant must be compiled into every handle that registers the slow one"
    );
}

/// The dispatch a model already writes — `step(K_MAX_ABS_ROW, .., threads = m)`
/// — produces the right scales whichever kernel the redirect picks, on this
/// device. On a GPU that means the cooperative kernel ran with `m * 64`
/// invocations from an unchanged call site; on the CPU JIT (no workgroup
/// barrier) it means the redirect correctly stood down.
#[test]
fn unchanged_dispatch_site_gets_correct_scales() {
    let gpu = gpu_core::testgpu::dev(AS_A_MODEL_REGISTERS_IT);
    for &(m, k) in &[(1u32, 3072u32), (7, 65), (64, 300), (512, 1024), (129, 3072)] {
        let x = fill(m, k);
        let xb = gpu.storage_init("x", &x);
        let sx = gpu.storage((m * 4) as u64);
        // Verbatim the shape of `qwen3::q8` / `s3dit::block` / flux2's call.
        let s = gpu.step(0, &[&xb, &sx], &[m, k], m);
        gpu.submit(&[], &[s]);
        assert_close(&gpu.read(&sx, m as usize), &host_scales(&x, m, k), &format!("{m}x{k}"));
    }
}

/// The redirect must NOT leak into the caller's index space. Profilers and cost
/// harnesses map `meta.kernel` through **their own** kernel list — the FLUX.2
/// DiT profiler indexes a 14-entry `const KERNELS` array with it — so a step
/// carrying the appended slot panics them. `meta` therefore records what the
/// caller asked for; the backend's `BRAIN_PROFILE` records what ran.
#[test]
fn meta_stays_in_the_callers_index_space() {
    let gpu = gpu_core::testgpu::dev(AS_A_MODEL_REGISTERS_IT);
    let x = gpu.storage_init("x", &fill(8, 64));
    let sx = gpu.storage(8 * 4);
    let s = gpu.step(0, &[&x, &sx], &[8, 64], 8);
    let m = s.meta().expect("step built through the facade");
    assert_eq!(m.kernel, 0, "meta must name the slot the CALLER dispatched");
    assert_eq!(m.threads, 8, "and the thread count the caller asked for");
    assert!(
        m.kernel < AS_A_MODEL_REGISTERS_IT.len(),
        "meta.kernel must stay indexable in the model's own kernel list"
    );
    gpu.submit(&[], &[s]);
}

/// A sliced dispatch (`step_sliced`, which is how the FLUX.2 int8 path quantizes
/// a row range) is redirected the same way: the offsets rebase both buffers, so
/// the cooperative kernel sees rows `0..m` and needs no offset arithmetic of its
/// own.
#[test]
fn sliced_dispatch_site_gets_correct_scales() {
    let gpu = gpu_core::testgpu::dev(AS_A_MODEL_REGISTERS_IT);
    let (rows, k) = (256u32, 1024u32);
    let (r0, m) = (64u32, 96u32);
    let x = fill(rows, k);
    let xb = gpu.storage_init("x", &x);
    let sx = gpu.storage((rows * 4) as u64);
    let s = gpu.step_sliced(
        0,
        &[&xb, &sx],
        &[(r0 as u64 * k as u64, m as u64 * k as u64), (r0 as u64, m as u64)],
        &[m, k],
        m,
    );
    gpu.submit(&[], &[s]);
    let got = gpu.read(&sx, rows as usize);
    let want = host_scales(&x[(r0 * k) as usize..((r0 + m) * k) as usize], m, k);
    assert_close(&got[r0 as usize..(r0 + m) as usize], &want, "sliced rows");
}
