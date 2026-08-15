// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dual-backend bf16/f16 storage-tier roundtrip for the 3 forward conv
//! kernels B8 templatized (`conv2d.wgsl`, `conv1d.wgsl`, `conv_bias.wgsl`) -
//! the same "prove the templated KERNEL round-trips" standard
//! `bf16_roundtrip.rs`/`f16_roundtrip.rs` set for the matmul family, but
//! dispatched directly against `kernels::template::dtype_variant`'s output
//! rather than through `model::ops::Ops` (no `Ops::conv2d` facade exists -
//! these three kernels' very different stride/pad/dilation/groups/bias
//! parameter shapes, spread across many per-model call sites, make one
//! shared `Ops` method a poor fit; see this crate's B8 ledger entry).
//!
//! **Tolerance** follows the exact same derivation as
//! `bf16_roundtrip.rs`/`f16_roundtrip.rs`: only the WEIGHT narrows, so each
//! output element's absolute error is bounded by `2^-(bits+1) * sum(|x*w|)`
//! over exactly the taps the kernel's own zero-pad boundary logic includes -
//! computed by a host reference that mirrors that boundary logic tap-for-tap
//! (not a closed-form output-size formula), so the tolerance can never
//! silently omit a tap the kernel did include or vice versa.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;

/// The three `_on_gpu` tests below each build real `Gpu::new_wgpu` devices
/// directly (no `gpu_core::testgpu::dev` sharing - the whole point here is a
/// specific, fixed backend per test, not a shared one), so under `cargo
/// test`'s default multi-threaded run they can run concurrently and race
/// their own independent device builds against each other - the same driver
/// hazard `crates/gpu-core/tests/device_sharing.rs`'s `DEVICE_SERIAL` (and
/// its copies in `device_churn.rs`, `crates/lfm2/tests/chunked_equiv.rs`)
/// exist to prevent. Same fix here.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn bits_of(dt: Dtype) -> i32 {
    match dt {
        Dtype::BF16 => 7,
        Dtype::F16 => 10,
        other => panic!("bits_of: unexpected tier {other:?}"),
    }
}

// ------------------------------------------------------------- conv2d ----

/// `(y, tol)` for `conv2d.wgsl`'s exact bias-free NCHW math ("same" padding:
/// `stride=1, pad=(K-1)/2, Ho=H, Wo=W`, mirrored tap-for-tap from the kernel
/// source, not a closed-form formula).
#[allow(clippy::too_many_arguments)]
fn host_conv2d(x: &[f32], w: &[f32], n: u32, cin: u32, h: u32, wd: u32, cout: u32, k: u32, stride: u32, pad: u32, ho: u32, wo: u32, bits: i32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0f32; (n * cout * ho * wo) as usize];
    let mut tol = vec![0f32; y.len()];
    for nn in 0..n {
        for co in 0..cout {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = 0f64;
                    let mut abs_sum = 0f64;
                    for ci in 0..cin {
                        for kh in 0..k {
                            let hi_b = oh * stride + kh;
                            if hi_b < pad {
                                continue;
                            }
                            let hi = hi_b - pad;
                            if hi >= h {
                                continue;
                            }
                            for kw in 0..k {
                                let wi_b = ow * stride + kw;
                                if wi_b < pad {
                                    continue;
                                }
                                let wi = wi_b - pad;
                                if wi >= wd {
                                    continue;
                                }
                                let xi = (((nn * cin + ci) * h + hi) * wd + wi) as usize;
                                let wi_idx = (((co * cin + ci) * k + kh) * k + kw) as usize;
                                let term = x[xi] as f64 * w[wi_idx] as f64;
                                acc += term;
                                abs_sum += term.abs();
                            }
                        }
                    }
                    let idx = (((nn * cout + co) * ho + oh) * wo + ow) as usize;
                    y[idx] = acc as f32;
                    tol[idx] = (abs_sum * 2f64.powi(-(bits + 1))) as f32 + 1e-5;
                }
            }
        }
    }
    (y, tol)
}

/// Which real backend a test wants - `Gpu::new_cpu`/`Gpu::new_wgpu` each need
/// the FULL kernel list up front (unlike a pre-built `Gpu`, which cannot
/// register a new kernel after construction), so every `run_*` helper below
/// takes the constructor itself, not an already-built `Gpu`.
type Backend = fn(&[(&str, &str)]) -> Gpu;

#[allow(clippy::too_many_arguments)]
fn run_conv2d(backend: Backend, dt: Dtype, seed: u64, label: &str) {
    let (n, cin, h, wd, cout, k, stride, pad) = (1u32, 3u32, 5u32, 5u32, 4u32, 3u32, 1u32, 1u32);
    let (ho, wo) = (h, wd); // "same" padding at stride 1, pad=(k-1)/2
    let bits = bits_of(dt);

    let (vname, vsrc) = kernels::template::dtype_variant("conv2d", kernels::CONV2D, "w", dt).unwrap();
    let gpu = backend(&[(vname, vsrc)]);
    let idx = gpu.kernel_index(vname).unwrap_or_else(|| panic!("{vname} not registered"));

    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled((n * cin * h * wd) as usize, 1.0);
    let w_h = rng.vec_scaled((cout * cin * k * k) as usize, 1.0);
    let packed = pack(&w_h, dt);

    let x = gpu.storage((n * cin * h * wd) as u64);
    gpu.write_f32(&x, &x_h);
    let w = gpu.storage(packed.len() as u64);
    gpu.write(&w, &packed);
    let y = gpu.storage((n * cout * ho * wo) as u64);

    let steps = [gpu.step(idx, &[&x, &w, &y], &[n, cin, h, wd, cout, k, stride, pad, ho, wo], n * cout * ho * wo)];
    gpu.submit(&[], &steps);
    let got = gpu.read(&y, (n * cout * ho * wo) as usize);

    let (want, tol) = host_conv2d(&x_h, &w_h, n, cin, h, wd, cout, k, stride, pad, ho, wo, bits);
    assert_conv_close(&got, &want, &tol, label);
}

// ------------------------------------------------------------- conv1d ----

#[allow(clippy::too_many_arguments)]
fn host_conv1d(x: &[f32], w: &[f32], n: u32, cin: u32, l: u32, cout: u32, k: u32, stride: u32, pad: u32, dilation: u32, groups: u32, lo: u32, bits: i32) -> (Vec<f32>, Vec<f32>) {
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut y = vec![0f32; (n * cout * lo) as usize];
    let mut tol = vec![0f32; y.len()];
    for nn in 0..n {
        for co in 0..cout {
            let g = co / cout_g;
            let ci0 = g * cin_g;
            for ol in 0..lo {
                let mut acc = 0f64;
                let mut abs_sum = 0f64;
                for cl in 0..cin_g {
                    let ci = ci0 + cl;
                    for kw in 0..k {
                        let li_b = ol * stride + kw * dilation;
                        if li_b < pad {
                            continue;
                        }
                        let li = li_b - pad;
                        if li >= l {
                            continue;
                        }
                        let xi = ((nn * cin + ci) * l + li) as usize;
                        let wi_idx = ((co * cin_g + cl) * k + kw) as usize;
                        let term = x[xi] as f64 * w[wi_idx] as f64;
                        acc += term;
                        abs_sum += term.abs();
                    }
                }
                let idx = ((nn * cout + co) * lo + ol) as usize;
                y[idx] = acc as f32;
                tol[idx] = (abs_sum * 2f64.powi(-(bits + 1))) as f32 + 1e-5;
            }
        }
    }
    (y, tol)
}

fn run_conv1d(backend: Backend, dt: Dtype, seed: u64, label: &str) {
    let (n, cin, l, cout, k, stride, pad, dilation, groups) = (1u32, 2u32, 6u32, 3u32, 3u32, 1u32, 1u32, 1u32, 1u32);
    let lo = l; // "same" padding at stride 1, dilation 1, pad=(k-1)/2
    let bits = bits_of(dt);

    let (vname, vsrc) = kernels::template::dtype_variant("conv1d", kernels::CONV1D, "w", dt).unwrap();
    let gpu = backend(&[(vname, vsrc)]);
    let idx = gpu.kernel_index(vname).unwrap_or_else(|| panic!("{vname} not registered"));

    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled((n * cin * l) as usize, 1.0);
    let w_h = rng.vec_scaled((cout * (cin / groups) * k) as usize, 1.0);
    let packed = pack(&w_h, dt);

    let x = gpu.storage((n * cin * l) as u64);
    gpu.write_f32(&x, &x_h);
    let w = gpu.storage(packed.len() as u64);
    gpu.write(&w, &packed);
    let y = gpu.storage((n * cout * lo) as u64);

    let steps = [gpu.step(idx, &[&x, &w, &y], &[n, cin, l, cout, k, stride, pad, dilation, groups, lo], n * cout * lo)];
    gpu.submit(&[], &steps);
    let got = gpu.read(&y, (n * cout * lo) as usize);

    let (want, tol) = host_conv1d(&x_h, &w_h, n, cin, l, cout, k, stride, pad, dilation, groups, lo, bits);
    assert_conv_close(&got, &want, &tol, label);
}

// ----------------------------------------------------------- conv_bias ---

#[allow(clippy::too_many_arguments)]
fn run_conv_bias(backend: Backend, dt: Dtype, seed: u64, label: &str) {
    let (n, cin, h, wd, cout, k, stride, pad) = (1u32, 3u32, 5u32, 5u32, 4u32, 3u32, 1u32, 1u32);
    let (ho, wo) = (h, wd);
    let bits = bits_of(dt);

    let (vname, vsrc) = kernels::template::dtype_variant("conv_bias", kernels::CONV_BIAS, "w", dt).unwrap();
    let gpu = backend(&[(vname, vsrc)]);
    let idx = gpu.kernel_index(vname).unwrap_or_else(|| panic!("{vname} not registered"));

    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled((n * cin * h * wd) as usize, 1.0);
    let w_h = rng.vec_scaled((cout * cin * k * k) as usize, 1.0);
    let bias_h = rng.vec_scaled(cout as usize, 1.0);
    let packed = pack(&w_h, dt);

    let x = gpu.storage((n * cin * h * wd) as u64);
    gpu.write_f32(&x, &x_h);
    let w = gpu.storage(packed.len() as u64);
    gpu.write(&w, &packed);
    let bias = gpu.storage(cout as u64);
    gpu.write_f32(&bias, &bias_h);
    let y = gpu.storage((n * cout * ho * wo) as u64);

    let steps = [gpu.step(idx, &[&x, &w, &bias, &y], &[n, cin, h, wd, cout, k, stride, pad, ho, wo], n * cout * ho * wo)];
    gpu.submit(&[], &steps);
    let got = gpu.read(&y, (n * cout * ho * wo) as usize);

    let (want_no_bias, tol) = host_conv2d(&x_h, &w_h, n, cin, h, wd, cout, k, stride, pad, ho, wo, bits);
    let mut want = want_no_bias;
    for nn in 0..n {
        for co in 0..cout {
            for oh in 0..ho {
                for ow in 0..wo {
                    let idx = (((nn * cout + co) * ho + oh) * wo + ow) as usize;
                    want[idx] += bias_h[co as usize];
                }
            }
        }
    }
    assert_conv_close(&got, &want, &tol, label);
}

// --------------------------------------------------------------- shared ---

fn pack(w: &[f32], dt: Dtype) -> Vec<u32> {
    match dt {
        Dtype::BF16 => model::half::pack_bf16(w),
        Dtype::F16 => model::half::pack_f16(w),
        other => panic!("pack: unexpected tier {other:?}"),
    }
}

fn assert_conv_close(got: &[f32], want: &[f32], tol: &[f32], label: &str) {
    assert_eq!(got.len(), want.len());
    let mut worst: f32 = 0.0;
    for i in 0..got.len() {
        let err = (got[i] - want[i]).abs();
        worst = worst.max(err / tol[i].max(1e-12));
        assert!(err <= tol[i], "{label}: elem {i} got {} want {} (err {err}, tol {})", got[i], want[i], tol[i]);
    }
    eprintln!("{label}: worst err/tol ratio {worst:.4}");
}

#[test]
fn conv2d_bf16_and_f16_match_f32_reference_on_cpu() {
    for dt in [Dtype::BF16, Dtype::F16] {
        run_conv2d(Gpu::new_cpu, dt, 0xC02D ^ dt as u64, &format!("cpu/conv2d/{dt:?}"));
    }
}

#[test]
fn conv2d_bf16_and_f16_match_f32_reference_on_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if skip_gpu() {
        eprintln!("conv2d_bf16_and_f16_match_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("conv2d_bf16_and_f16_match_f32_reference_on_gpu: running on a real wgpu device");
    for dt in [Dtype::BF16, Dtype::F16] {
        run_conv2d(Gpu::new_wgpu, dt, 0xC02D ^ dt as u64, &format!("gpu/conv2d/{dt:?}"));
    }
}

#[test]
fn conv1d_bf16_and_f16_match_f32_reference_on_cpu() {
    for dt in [Dtype::BF16, Dtype::F16] {
        run_conv1d(Gpu::new_cpu, dt, 0xC01D ^ dt as u64, &format!("cpu/conv1d/{dt:?}"));
    }
}

#[test]
fn conv1d_bf16_and_f16_match_f32_reference_on_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if skip_gpu() {
        eprintln!("conv1d_bf16_and_f16_match_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("conv1d_bf16_and_f16_match_f32_reference_on_gpu: running on a real wgpu device");
    for dt in [Dtype::BF16, Dtype::F16] {
        run_conv1d(Gpu::new_wgpu, dt, 0xC01D ^ dt as u64, &format!("gpu/conv1d/{dt:?}"));
    }
}

#[test]
fn conv_bias_bf16_and_f16_match_f32_reference_on_cpu() {
    for dt in [Dtype::BF16, Dtype::F16] {
        run_conv_bias(Gpu::new_cpu, dt, 0xB1A5 ^ dt as u64, &format!("cpu/conv_bias/{dt:?}"));
    }
}

#[test]
fn conv_bias_bf16_and_f16_match_f32_reference_on_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if skip_gpu() {
        eprintln!("conv_bias_bf16_and_f16_match_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("conv_bias_bf16_and_f16_match_f32_reference_on_gpu: running on a real wgpu device");
    for dt in [Dtype::BF16, Dtype::F16] {
        run_conv_bias(Gpu::new_wgpu, dt, 0xB1A5 ^ dt as u64, &format!("gpu/conv_bias/{dt:?}"));
    }
}
