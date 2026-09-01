// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Closing the `step_buf` blind spot in `gpu_core::upgrade`.
//!
//! The `matmul_gemv` -> `matmul_gemv_reg` row is shape-specialised: it needs
//! the caller's own row count `m` to pick the smallest `MREG` bucket that
//! covers it. `Gpu::step`/`step_sliced` carry `params` for exactly this, but
//! `Gpu::step_buf`'s uniform lives in a caller-owned buffer the seam cannot
//! read - so before this test existed, a `step_buf` caller of a
//! shape-specialised row always fell back to the kernel it registered,
//! silently un-upgraded. `Gpu::step_buf_shaped` closes that: the caller
//! already computed the values it wrote into its buffer, so handing them back
//! to the seam costs nothing extra.

const KERNELS: &[(&str, &str)] = &[("matmul_gemv", kernels::MATMUL_GEMV)];
const K_GEMV: usize = 0;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    r.vec_scaled(n, 1.0)
}

/// `step_buf` (no shape hint) leaves a shape-specialised row exactly where it
/// was before this milestone: un-upgraded. This is the control - without it,
/// the bucket-ladder test below would not prove the NEW method did anything.
#[test]
fn step_buf_without_a_shape_hint_stays_on_the_registered_kernel() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("matmul_gemv is not selectable without workgroup reductions");
        return;
    }
    if !gpu.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("this backend cannot time individual kernels");
        return;
    }
    let (m, k, n) = (4u32, 128u32, 64u32);
    let x = gpu.storage_init("x", &fill((m * k) as usize, 1));
    let w = gpu.storage_init("w", &fill((n * k) as usize, 2));
    let out = gpu.storage(u64::from(m * n));
    let ubuf = gpu.uniform_dynamic(3);
    gpu.write(&ubuf, &[m, k, n]);
    gpu.reset_kernel_times();
    let s = gpu.step_buf(K_GEMV, &ubuf, &[&x, &w, &out], n * 64);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let times = gpu.kernel_times().expect("kernel timing armed above");
    let ran: Vec<&str> = times.iter().filter(|(_, _, c)| *c > 0).map(|(name, _, _)| name.as_str()).collect();
    assert_eq!(ran, vec!["matmul_gemv"], "no shape to probe with: must dispatch exactly the registered kernel");
}

/// `step_buf_shaped` gives the seam the same values `step` would have read
/// from `params`, even though the real uniform lives in a caller-owned
/// buffer - the row resolves the smallest `MREG` bucket covering `m`, exactly
/// as `step` already does (`gemv_reg_upgrade.rs`'s own ladder).
#[test]
fn step_buf_shaped_reaches_the_bucket_ladder() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("matmul_gemv is not selectable without workgroup reductions");
        return;
    }
    if !gpu.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("this backend cannot time individual kernels");
        return;
    }
    let (k, n) = (128u32, 64u32);
    for (m, want) in [(1u32, "matmul_gemv_reg#MREG=1"), (3, "matmul_gemv_reg#MREG=4"), (32, "matmul_gemv_reg#MREG=32")] {
        let x = gpu.storage_init("x", &fill((m * k) as usize, 1));
        let w = gpu.storage_init("w", &fill((n * k) as usize, 2));
        let out = gpu.storage(u64::from(m * n));
        let ubuf = gpu.uniform_dynamic(3);
        gpu.write(&ubuf, &[m, k, n]);
        gpu.reset_kernel_times();
        let s = gpu.step_buf_shaped(K_GEMV, &ubuf, &[&x, &w, &out], &[m, k, n], n * 64);
        gpu.submit(&[], &[s]);
        gpu.poll_wait();
        let times = gpu.kernel_times().expect("kernel timing armed above");
        let ran: Vec<&str> = times.iter().filter(|(_, _, c)| *c > 0).map(|(name, _, _)| name.as_str()).collect();
        assert_eq!(ran, vec![want], "m={m}: wrong bucket dispatched through step_buf_shaped");
    }
}

/// And the result itself is correct, not just the pipeline name: a
/// `step_buf_shaped` call and a `step` call at the same shape resolve to the
/// same bucket, so this is `assert_eq!` on raw bits, same discipline as
/// `gemv_reg_upgrade.rs`.
#[test]
fn step_buf_shaped_result_is_byte_identical_to_step() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("matmul_gemv is not selectable without workgroup reductions");
        return;
    }
    for m in [1u32, 5, 17, 32] {
        let (k, n) = (137u32, 61u32);
        let x = fill((m * k) as usize, u64::from(m) * 7 + 1);
        let w = fill((n * k) as usize, u64::from(n) * 13 + 3);
        let xb = gpu.storage(u64::from(m * k));
        let wb = gpu.storage(u64::from(n * k));
        gpu.write_f32(&xb, &x);
        gpu.write_f32(&wb, &w);
        let via_step = gpu.storage(u64::from(m * n));
        let via_buf = gpu.storage(u64::from(m * n));
        let ubuf = gpu.uniform_dynamic(3);
        gpu.write(&ubuf, &[m, k, n]);
        gpu.submit(
            &[],
            &[
                gpu.step(K_GEMV, &[&xb, &wb, &via_step], &[m, k, n], n * 64),
                gpu.step_buf_shaped(K_GEMV, &ubuf, &[&xb, &wb, &via_buf], &[m, k, n], n * 64),
            ],
        );
        gpu.poll_wait();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&gpu.read(&via_step, (m * n) as usize)),
            bits(&gpu.read(&via_buf, (m * n) as usize)),
            "m={m}: step and step_buf_shaped must resolve to the same bucket"
        );
    }
}

/// `BRAIN_NO_KERNEL_UPGRADE=1` must pin `step_buf_shaped` back onto the
/// registered kernel too - the shape hint is only ever a hint for the seam,
/// never a second dispatch path around it.
#[test]
fn the_env_switch_disables_step_buf_shaped_too() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["step_buf_shaped_reaches_the_bucket_ladder", "--exact", "--nocapture"])
        .env("BRAIN_NO_KERNEL_UPGRADE", "1")
        .output()
        .expect("re-run this test binary with the upgrade disabled");
    let text =
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(
        !out.status.success(),
        "with BRAIN_NO_KERNEL_UPGRADE=1 the bucket ladder must be absent, so the assertion above \
         must FAIL - if it passed, the switch no longer disables step_buf_shaped:\n{text}"
    );
    assert!(
        text.contains("wrong bucket dispatched") || text.contains("skip"),
        "the failure must be the ladder assertion, not some unrelated breakage:\n{text}"
    );
}
