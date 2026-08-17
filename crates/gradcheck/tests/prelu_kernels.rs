// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the `prelu` kernel family, driven directly through
//! `gpu_core` like `depth_kernels.rs` / `glue.rs` — no model is built.
//!
//! PReLU is the ArcFace / IResNet-100 activation:
//!     y = x            where x > 0
//!     y = a[c] * x     where x <= 0
//! with `a` a **trainable** parameter — one slope per channel, or a single
//! shared slope. That trainability is the whole reason the family exists:
//! `leaky_relu` has the same forward shape but a *constant* slope bit-cast into
//! its uniform, so it has no `da` and would pin every learned slope at its init.
//!
//! Three techniques, each chosen for what it can catch:
//!
//! 1. **CPU-reference parity** (`prelu_ref` / `prelu_bwd_ref` below, re-derived
//!    from the definition rather than shared with the kernel — an oracle that
//!    shares code with the thing it checks proves nothing). Forward is a single
//!    fp32 multiply per element, so parity is asserted at **1e-6 absolute** on
//!    inputs bounded by 1. `da` is a reduction whose *order* differs (the kernel
//!    folds 64 workgroup partials; the oracle walks ascending n,h,w) so it is
//!    asserted at **1e-4 relative**.
//!
//! 2. **Finite differences** on `L = <y, dy>` for both `dx` and `da`. PReLU is
//!    piecewise-linear, which makes FD unusually sharp here *and* unusually
//!    trappy: within `eps` of the kink at x == 0 a central difference straddles
//!    two different linear pieces and is simply the wrong quantity, so `dx`
//!    probes skip any coordinate with `|x| <= 10*eps`. `L` is exactly affine in
//!    each `a[c]` (perturbing a slope cannot change a sign), so the `da` central
//!    difference is exact to fp32 round-off and needs no such filter.
//!
//! 3. **Channel isolation.** Perturbing `a[j]` may change output channel `j` and
//!    NOTHING else. A wrong channel decode (`idx % C` instead of
//!    `(idx / (H*W)) % C`) still produces plausible numbers of the right shape
//!    and passes a careless forward test; it cannot survive this one.
//!
//! Shapes are deliberately hostile to the backward's stride-64 plane walk:
//! `H*W = 42` leaves 22 of the 64 lanes idle every plane, and `H*W = 135` is
//! two full strides plus a 7-element tail. `C` is never a multiple of anything.
//!
//! Two contracts this file pins down because getting them wrong is silent:
//!   * `prelu_bwd` **accumulates** into `da` (`da[c] = da[c] + s`), so callers
//!     must zero it — `g.submit(&[&dab], &[step])`. `da_accumulates` asserts a
//!     second unzeroed dispatch doubles the result.
//!   * `da` is **always [C]**, one partial per channel, even when `nslope == 1`.
//!     With a shared slope the true gradient is the SUM of those C entries, and
//!     the kernel deliberately does not compute it (two workgroups adding into
//!     `da[0]` would race, and this engine has no atomics). `shared_slope_da_is_
//!     per_channel_partials` asserts exactly that sum against FD.
//!
//! **Two backward variants, and why every backward test runs both.**
//! `prelu_bwd` is barrier-free (one invocation per channel) and `prelu_bwd_wg`
//! is the cooperative workgroup-per-channel twin — same bindings, same
//! `Params`, different thread count. The pair exists because
//! `DeviceCaps::workgroup_reductions` is **false on the CPU backend**: its
//! split-at-barrier JIT mis-executes `var<workgroup>` + `workgroupBarrier()`
//! kernels, and `prelu_bwd_wg` returns `da == 0` there while `dx` stays
//! correct — a PReLU whose slopes never move, training to a plausible loss.
//! So `prelu_bwd_wg` is skipped, not run, when the queried cap is false; every
//! device runs at least the reference. `backward_variants_agree` pins the two
//! against each other wherever both are legal.
//!
//! Runs on any device. Under `BRAIN_DEVICE=cpu` the cooperative variant is
//! correctly skipped rather than silently asserted.

use data::rng::Lcg;
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("prelu", kernels::PRELU),                 // 0
    ("prelu_bwd", kernels::PRELU_BWD),         // 1  barrier-free reference
    ("leaky_relu", kernels::LEAKY_RELU),       // 2 (pre-existing, already gated)
    ("prelu_bwd_wg", kernels::PRELU_BWD_WG),   // 3  cooperative, caps-gated
];
const K_PRELU: usize = 0;
const K_PRELU_BWD: usize = 1;
const K_LEAKY: usize = 2;
const K_PRELU_BWD_WG: usize = 3;

/// The backward variants this device may legally run: the reference always,
/// the cooperative one only where the QUERIED cap says barriers work.
/// Each entry is `(kernel name, kernel index, threads-per-channel)`.
fn bwd_variants(gpu: &Gpu) -> Vec<(&'static str, usize, usize)> {
    let mut v = vec![("prelu_bwd", K_PRELU_BWD, 1usize)];
    if gpu.caps().workgroup_reductions {
        v.push(("prelu_bwd_wg", K_PRELU_BWD_WG, 64));
    }
    v
}

/// Expensive/physical-GPU work is skipped when the variable is merely PRESENT.
fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Inputs come from [`Lcg::signed`], which is **[-1, 1)** — both signs. That is
/// load-bearing here and nowhere more so: with a one-sided (negative-only)
/// stream the `x > 0` branch is never taken, and a kernel that computed
/// `a[c] * x` unconditionally would pass every forward and backward test in
/// this file. Measured over every seed and shape used below: 50.9% positive,
/// and no exact zeros, so the `x == 0` tie-break never decides an assertion.
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

// ---- CPU reference oracles ----------------------------------------------------
//
// Re-derived from the PReLU definition; matches `wgsl/prelu.wgsl` and
// `wgsl/prelu_bwd.wgsl` by construction, not by shared code.

/// Slope index for channel `c`: `a[c]` per-channel, `a[0]` when shared.
fn slope_idx(c: usize, nslope: usize) -> usize {
    if nslope > 1 {
        c
    } else {
        0
    }
}

/// y = x > 0 ? x : a[.]*x over NCHW. Matches `wgsl/prelu.wgsl`.
fn prelu_ref(x: &[f32], a: &[f32], n: usize, c: usize, h: usize, w: usize, nslope: usize) -> Vec<f32> {
    let hw = h * w;
    let mut y = vec![0.0f32; n * c * hw];
    for ni in 0..n {
        for ci in 0..c {
            let s = a[slope_idx(ci, nslope)];
            let base = (ni * c + ci) * hw;
            for i in 0..hw {
                let v = x[base + i];
                y[base + i] = if v > 0.0 { v } else { s * v };
            }
        }
    }
    y
}

/// (dx, da) for `L` with upstream `dy`. `da` is ALWAYS length `c` — one partial
/// per channel — matching `wgsl/prelu_bwd.wgsl`; the shared-slope gradient is
/// the sum of the returned vector.
fn prelu_bwd_ref(
    x: &[f32],
    a: &[f32],
    dy: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    nslope: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hw = h * w;
    let mut dx = vec![0.0f32; n * c * hw];
    let mut da = vec![0.0f32; c];
    for ci in 0..c {
        let s = a[slope_idx(ci, nslope)];
        let mut acc = 0.0f64; // f64 so the oracle's own summation is not the error
        for ni in 0..n {
            let base = (ni * c + ci) * hw;
            for i in 0..hw {
                let v = x[base + i];
                let g = dy[base + i];
                if v > 0.0 {
                    dx[base + i] = g;
                } else {
                    dx[base + i] = s * g;
                    acc += g as f64 * v as f64;
                }
            }
        }
        da[ci] = acc as f32;
    }
    (dx, da)
}

// ---- dispatch helpers ---------------------------------------------------------

fn prelu_fwd(gpu: &Gpu, x: &[f32], a: &[f32], dims: (usize, usize, usize, usize), nslope: usize) -> Vec<f32> {
    let (n, c, h, w) = dims;
    let total = n * c * h * w;
    let xb = gpu.storage_init("x", x);
    let ab = gpu.storage_init("a", a);
    let yb = gpu.storage(total as u64);
    let params = [n as u32, c as u32, h as u32, w as u32, nslope as u32];
    // Dispatch geometry: one invocation per OUTPUT element.
    let s = gpu.step(K_PRELU, &[&xb, &ab, &yb], &params, total as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    gpu.read(&yb, total)
}

/// Returns (dx, da). `da` is zeroed via the submit clear list, which is the
/// documented contract for this accumulating output.
///
/// `kind` selects the variant and `tpc` its threads-per-channel — 1 for the
/// barrier-free `prelu_bwd`, 64 for the workgroup-per-channel `prelu_bwd_wg`.
/// Both are C-scaled, never the element count: dispatching `N*C*H*W` at either
/// would re-accumulate every channel `N*H*W` times (`dx` still correct, `da`
/// inflated — the `silu_mul` failure mode).
fn prelu_bwd_k(
    gpu: &Gpu,
    kind: usize,
    tpc: usize,
    x: &[f32],
    a: &[f32],
    dy: &[f32],
    dims: (usize, usize, usize, usize),
    nslope: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (n, c, h, w) = dims;
    let total = n * c * h * w;
    let xb = gpu.storage_init("x", x);
    let ab = gpu.storage_init("a", a);
    let dyb = gpu.storage_init("dy", dy);
    let dxb = gpu.storage(total as u64);
    let dab = gpu.storage(c as u64);
    let params = [n as u32, c as u32, h as u32, w as u32, nslope as u32];
    let s = gpu.step(kind, &[&xb, &ab, &dyb, &dxb, &dab], &params, (c * tpc) as u32);
    gpu.submit(&[&dab], &[s]);
    gpu.poll_wait();
    (gpu.read(&dxb, total), gpu.read(&dab, c))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ---- forward ------------------------------------------------------------------

/// Per-channel slopes, forward vs the CPU oracle. Tolerance 1e-6 absolute: the
/// op is one fp32 multiply per element, so anything larger is a real divergence,
/// not reassociation.
#[test]
fn forward_per_channel_matches_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for &(n, c, h, w) in &[(3usize, 5usize, 6usize, 7usize), (2, 3, 9, 15), (1, 7, 1, 1)] {
        let total = n * c * h * w;
        let x = Lcg::new(0x5eed_0001).vec(total);
        let a = Lcg::new(0x5eed_0002).vec(c);
        let got = prelu_fwd(&gpu, &x, &a, (n, c, h, w), c);
        let want = prelu_ref(&x, &a, n, c, h, w, c);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-6, "prelu fwd N{n}C{c}H{h}W{w}: max abs diff {d:.3e}");
    }
}

/// The single-shared-slope form (`nslope == 1`) must read `a[0]` for EVERY
/// channel. A kernel that indexed `a[c]` regardless would read out of bounds or
/// (worse, on a clamped backend) silently reuse `a[0]` only sometimes.
#[test]
fn forward_shared_slope_matches_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2usize, 5usize, 4usize, 11usize);
    let total = n * c * h * w;
    let x = Lcg::new(0x5eed_0011).vec(total);
    let a = vec![0.2437f32];
    let got = prelu_fwd(&gpu, &x, &a, (n, c, h, w), 1);
    let want = prelu_ref(&x, &a, n, c, h, w, 1);
    let d = max_abs_diff(&got, &want);
    assert!(d < 1e-6, "prelu fwd shared slope: max abs diff {d:.3e}");

    // And it must NOT coincide with the per-channel path for a varying `a`,
    // which is what makes the assertion above meaningful.
    let a_pc = Lcg::new(0x5eed_0012).vec(c);
    let pc = prelu_fwd(&gpu, &x, &a_pc, (n, c, h, w), c);
    assert!(max_abs_diff(&pc, &got) > 1e-3, "shared and per-channel paths are indistinguishable");
}

/// Cross-check against the already-gated `leaky_relu`: with every slope equal to
/// the same constant, PReLU IS leaky ReLU. This ties the new family to a kernel
/// that other models already trust.
///
/// The two differ at exactly `x == 0` (`leaky_relu` tests `x >= 0`, PReLU tests
/// `x > 0`, matching torch.prelu) — a measure-zero disagreement the LCG never
/// samples, and one whose only observable effect is which branch's derivative is
/// taken at the kink.
#[test]
fn forward_equals_leaky_relu_at_constant_slope() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2usize, 4usize, 5usize, 5usize);
    let total = n * c * h * w;
    let x = Lcg::new(0x5eed_0021).vec(total);
    let slope = 0.1f32;

    let mine = prelu_fwd(&gpu, &x, &vec![slope; c], (n, c, h, w), c);

    let xb = gpu.storage_init("x", &x);
    let yb = gpu.storage(total as u64);
    let s = gpu.step(K_LEAKY, &[&xb, &yb], &[total as u32, slope.to_bits()], total as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let theirs = gpu.read(&yb, total);

    let d = max_abs_diff(&mine, &theirs);
    assert!(d < 1e-6, "prelu != leaky_relu at constant slope: max abs diff {d:.3e}");
}

/// A wrong channel decode is the silent failure mode here: `idx % C` has the
/// right range and produces entirely plausible output. Perturbing `a[j]` may
/// change output channel `j` and nothing else — a property no wrong decode
/// satisfies, at any C.
#[test]
fn channel_isolation() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (3usize, 5usize, 6usize, 7usize);
    let hw = h * w;
    let total = n * c * hw;
    let x = Lcg::new(0x5eed_0031).vec(total);
    let a = vec![0.25f32; c];
    let base = prelu_fwd(&gpu, &x, &a, (n, c, h, w), c);

    for j in 0..c {
        let mut a2 = a.clone();
        a2[j] += 1.0;
        let alt = prelu_fwd(&gpu, &x, &a2, (n, c, h, w), c);
        let mut touched_elsewhere = 0usize;
        let mut touched_here = 0usize;
        for ni in 0..n {
            for ci in 0..c {
                for i in 0..hw {
                    let idx = (ni * c + ci) * hw + i;
                    if (alt[idx] - base[idx]).abs() > 1e-6 {
                        if ci == j {
                            touched_here += 1;
                        } else {
                            touched_elsewhere += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(touched_elsewhere, 0, "a[{j}] leaked into another channel ({touched_elsewhere} elements)");
        assert!(touched_here > 0, "a[{j}] changed nothing in its own channel");
    }
}

// ---- backward -----------------------------------------------------------------

/// Shapes every backward test sweeps. Chosen to be hostile to the cooperative
/// variant's 64-lane walk AND to the reference's per-plane loop:
///   * `H*W = 42`, `135`, `63`, `65` — partial strides and a 7/1-element tail;
///   * `H*W < 64` with `N > 1` — the shape where a per-PLANE stride-64 walk
///     leaves 64-H*W lanes idle, and at `H*W == 1` collapses onto thread 0;
///   * `H*W == 1` — the flat `[N, C]` activation the family advertises.
///
/// `C` is never a multiple of anything.
const BWD_SHAPES: &[(usize, usize, usize, usize)] = &[
    (3, 5, 6, 7),
    (2, 3, 9, 15),
    (1, 7, 1, 1),
    (5, 3, 2, 3),
    (4, 6, 1, 1),
    (2, 5, 1, 63),
    (2, 5, 1, 65),
];

/// Backward vs the CPU oracle, for EVERY variant this device may run. `dx` is
/// exact (one multiply); `da` is a reduction whose order differs between the
/// oracle, the reference kernel and the cooperative kernel, so it gets a
/// relative tolerance.
#[test]
fn backward_matches_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for &(name, kind, tpc) in bwd_variants(&gpu).iter() {
        for &(n, c, h, w) in BWD_SHAPES {
            let total = n * c * h * w;
            let x = Lcg::new(0x5eed_0041).vec(total);
            let a = Lcg::new(0x5eed_0042).vec(c);
            let dy = Lcg::new(0x5eed_0043).vec(total);
            let (dx, da) = prelu_bwd_k(&gpu, kind, tpc, &x, &a, &dy, (n, c, h, w), c);
            let (wdx, wda) = prelu_bwd_ref(&x, &a, &dy, n, c, h, w, c);

            let d = max_abs_diff(&dx, &wdx);
            assert!(d < 1e-6, "{name} dx N{n}C{c}H{h}W{w}: max abs diff {d:.3e}");
            for ci in 0..c {
                let tol = 1e-4 * wda[ci].abs().max(1.0);
                assert!(
                    (da[ci] - wda[ci]).abs() < tol,
                    "{name} da[{ci}] N{n}C{c}H{h}W{w}: got {} want {}",
                    da[ci],
                    wda[ci]
                );
            }
            // The oracle itself must not be asserting zero == zero. A kernel
            // that never writes `da` (the shape `prelu_bwd_wg` takes on a
            // backend without working barriers) returns all zeros and would
            // sail through the per-channel comparison if the reference were
            // also zero. Individual channels may legitimately be zero at tiny
            // N*H*W, so the guard is on the whole vector.
            assert!(
                wda.iter().any(|v| v.abs() > 1e-3),
                "{name} N{n}C{c}H{h}W{w}: the whole reference da is ~0, the assertion is vacuous"
            );
        }
    }
}

/// The two backward variants must agree. `+` reassociates, so this is a
/// tolerance check, not `assert_eq!` — but a dispatch-geometry or indexing
/// divergence between them is orders of magnitude larger than reassociation
/// and cannot hide under it.
///
/// Skipped, not failed, where `DeviceCaps::workgroup_reductions` is false:
/// there the cooperative kernel is not a legal kernel to run at all.
#[test]
fn backward_variants_agree() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        return; // prelu_bwd_wg is not selectable here; prelu_bwd is the only path
    }
    for &(n, c, h, w) in BWD_SHAPES {
        let total = n * c * h * w;
        let x = Lcg::new(0x5eed_0091).vec(total);
        let a = Lcg::new(0x5eed_0092).vec(c);
        let dy = Lcg::new(0x5eed_0093).vec(total);
        let (dx_r, da_r) = prelu_bwd_k(&gpu, K_PRELU_BWD, 1, &x, &a, &dy, (n, c, h, w), c);
        let (dx_w, da_w) = prelu_bwd_k(&gpu, K_PRELU_BWD_WG, 64, &x, &a, &dy, (n, c, h, w), c);
        // dx is a pure map — no reassociation, so this one IS exact.
        assert_eq!(dx_r, dx_w, "dx differs between prelu_bwd and prelu_bwd_wg at N{n}C{c}H{h}W{w}");
        for ci in 0..c {
            let tol = 1e-4 * da_r[ci].abs().max(1.0);
            assert!(
                (da_r[ci] - da_w[ci]).abs() < tol,
                "da[{ci}] differs at N{n}C{c}H{h}W{w}: prelu_bwd {} vs prelu_bwd_wg {}",
                da_r[ci],
                da_w[ci]
            );
        }
    }
}

/// `dx` against central differences of `L = <y, dy>`.
///
/// PReLU is piecewise-linear, so away from the kink the central difference is
/// EXACT — the tolerance below is round-off, not truncation. Coordinates within
/// `10*eps` of zero are skipped: there the two probes land on different linear
/// pieces and FD measures a quantity the derivative does not have.
#[test]
fn backward_dx_matches_finite_differences() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2usize, 3usize, 5usize, 6usize);
    let total = n * c * h * w;
    let x = Lcg::new(0x5eed_0051).vec(total);
    let a = Lcg::new(0x5eed_0052).vec(c);
    let dy = Lcg::new(0x5eed_0053).vec(total);
    for &(name, kind, tpc) in bwd_variants(&gpu).iter() {
        let (dx, _) = prelu_bwd_k(&gpu, kind, tpc, &x, &a, &dy, (n, c, h, w), c);

        let eps = 1e-3f32;
        let mut probed = 0usize;
        for i in (0..total).step_by(7) {
            if x[i].abs() <= 10.0 * eps {
                continue; // straddles the kink — FD is not the derivative there
            }
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let lp = dot(&prelu_fwd(&gpu, &xp, &a, (n, c, h, w), c), &dy);
            let lm = dot(&prelu_fwd(&gpu, &xm, &a, (n, c, h, w), c), &dy);
            let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let tol = 1e-3 + 1e-2 * num.abs().max(dx[i].abs());
            assert!(
                (num - dx[i]).abs() < tol,
                "{name} dx[{i}]: fd {num} vs analytic {} (x = {})",
                dx[i],
                x[i]
            );
            probed += 1;
        }
        assert!(probed > 20, "{name}: too few dx probes survived the kink filter ({probed})");
    }
}

/// `da` against central differences of `L = <y, dy>` w.r.t. each slope.
///
/// No kink filter is needed: perturbing `a[c]` cannot change the sign of any
/// `x`, so `L` is exactly affine in `a[c]` and the central difference is exact
/// to fp32 round-off at any eps.
#[test]
fn backward_da_matches_finite_differences() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2usize, 5usize, 4usize, 11usize);
    let total = n * c * h * w;
    let x = Lcg::new(0x5eed_0061).vec(total);
    let a = Lcg::new(0x5eed_0062).vec(c);
    let dy = Lcg::new(0x5eed_0063).vec(total);
    for &(name, kind, tpc) in bwd_variants(&gpu).iter() {
        let (_, da) = prelu_bwd_k(&gpu, kind, tpc, &x, &a, &dy, (n, c, h, w), c);

        let eps = 1e-2f32;
        for ci in 0..c {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[ci] += eps;
            am[ci] -= eps;
            let lp = dot(&prelu_fwd(&gpu, &x, &ap, (n, c, h, w), c), &dy);
            let lm = dot(&prelu_fwd(&gpu, &x, &am, (n, c, h, w), c), &dy);
            let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let tol = 1e-3 + 1e-2 * num.abs().max(da[ci].abs());
            // A slope with a zero gradient would make the comparison vacuous.
            assert!(num.abs() > 1e-2, "{name}: da[{ci}] FD is ~0, the assertion proves nothing");
            assert!((num - da[ci]).abs() < tol, "{name} da[{ci}]: fd {num} vs analytic {}", da[ci]);
        }
    }
}

/// With `nslope == 1` the kernel still writes C per-channel partials; the shared
/// slope's true gradient is their SUM. This is the contract most likely to be
/// mis-wired, because the wrong answer (using `da[0]` alone) has the right shape
/// and a plausible magnitude.
#[test]
fn shared_slope_da_is_per_channel_partials() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2usize, 5usize, 4usize, 11usize);
    let total = n * c * h * w;
    let x = Lcg::new(0x5eed_0071).vec(total);
    let a = vec![0.31f32];
    let dy = Lcg::new(0x5eed_0073).vec(total);
    for &(name, kind, tpc) in bwd_variants(&gpu).iter() {
        let (_, da) = prelu_bwd_k(&gpu, kind, tpc, &x, &a, &dy, (n, c, h, w), 1);
        let summed: f64 = da.iter().map(|&v| v as f64).sum();

        let eps = 1e-2f32;
        let lp = dot(&prelu_fwd(&gpu, &x, &[a[0] + eps], (n, c, h, w), 1), &dy);
        let lm = dot(&prelu_fwd(&gpu, &x, &[a[0] - eps], (n, c, h, w), 1), &dy);
        let num = (lp - lm) / (2.0 * eps as f64);
        let tol = 1e-3 + 1e-2 * num.abs().max(summed.abs());
        assert!(
            (num - summed).abs() < tol,
            "{name} shared-slope da: fd {num} vs sum of per-channel partials {summed}"
        );
        // And da[0] alone is NOT the answer, which is why the sum is required.
        assert!(
            (num - da[0] as f64).abs() > tol,
            "{name}: da[0] alone happens to equal the shared gradient — pick a shape where it does not"
        );
    }
}

/// `da` ACCUMULATES: a second dispatch into an unzeroed buffer must double it.
/// Callers zero via `submit`'s clear list. Asserting the contract here is what
/// stops a future caller from quietly getting a running sum across steps.
#[test]
fn da_accumulates_into_a_pre_zeroed_buffer() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2usize, 3usize, 5usize, 6usize);
    let total = n * c * h * w;
    let x = Lcg::new(0x5eed_0081).vec(total);
    let a = Lcg::new(0x5eed_0082).vec(c);
    let dy = Lcg::new(0x5eed_0083).vec(total);
    let params = [n as u32, c as u32, h as u32, w as u32, c as u32];

    for &(name, kind, tpc) in bwd_variants(&gpu).iter() {
        let xb = gpu.storage_init("x", &x);
        let ab = gpu.storage_init("a", &a);
        let dyb = gpu.storage_init("dy", &dy);
        let dxb = gpu.storage(total as u64);
        let dab = gpu.storage(c as u64);
        let s1 = gpu.step(kind, &[&xb, &ab, &dyb, &dxb, &dab], &params, (c * tpc) as u32);
        let s2 = gpu.step(kind, &[&xb, &ab, &dyb, &dxb, &dab], &params, (c * tpc) as u32);
        gpu.submit(&[&dab], &[s1, s2]); // cleared ONCE, dispatched twice
        gpu.poll_wait();
        let twice = gpu.read(&dab, c);

        let (_, once) = prelu_bwd_k(&gpu, kind, tpc, &x, &a, &dy, (n, c, h, w), c);
        for ci in 0..c {
            // `0 == 2*0` passes this test for free, which is precisely how an
            // all-zero `da` survives it. Assert the single dispatch is non-zero
            // first.
            assert!(once[ci].abs() > 1e-3, "{name}: da[{ci}] is ~0 — the doubling check is vacuous");
            let want = 2.0 * once[ci];
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                (twice[ci] - want).abs() < tol,
                "{name}: da[{ci}] did not accumulate: {} vs 2x{}",
                twice[ci],
                once[ci]
            );
        }
    }
}
