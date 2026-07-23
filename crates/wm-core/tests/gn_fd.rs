// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1.gn — finite-difference (gradcheck) entry for the GroupNorm backward
//! chain, written FROM the spec (`docs/models/world-models/specs/p1.gn.md` §10.3).
//!
//! Kernel-level composed-backward FD check in the mse_fd.rs pattern
//! (crates/gradcheck/tests/mse_fd.rs) — NOT a CheckModel registration:
//! wm-core has no trainable model yet. Drives the kernels directly via
//! `gpu_core::Gpu::new(&Gn::kernel_sources())`.
//!
//! Gating runs are `BRAIN_DEVICE=cpu` (+ `MOE_SKIP_GPU_TESTS=1`), both set by
//! `scripts/wm-locked-make.sh`; GPU results never gate (playbook §1).
//!
//! Scalar loss: L = Σ_i dy_i · y_i (host sum of the forward output), whose
//! gradients w.r.t. x and gb are exactly the backward kernels' dx and dgb.
//! Global gradcheck tolerances (playbook §3): h = 5e-3, pass iff
//! |analytic − numeric| ≤ 4e-3 + 8e-2 · max(|a|, |n|). NEVER loosened —
//! failures follow the playbook §3 ladder instead.

use gpu_core::Gpu;
use wm_core::gn::{Gn, GnDims};

/// Deterministic LCG in [-1, 1). Spec §10.3 requires seeded data in [−1,1];
/// mse_fd.rs's `>> 33` variant keeps only 31 bits and thus lands in [−1,0)
/// (its `~[-1,1)` comment is wrong) — copying it here lost all sign coverage,
/// so this takes 32 bits before the 2^31 scale (round-2 adversary fix).
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 32) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Forward pass; returns L = Σ dy_i · y_i (f64 host accumulation).
fn loss(gpu: &Gpu, gn: &Gn, d: &GnDims, x: &[f32], gb: &[f32], dy: &[f32]) -> f64 {
    let xb = gpu.storage_init("x", x);
    let gbb = gpu.storage_init("gb", gb);
    let stats = gpu.storage(d.stats_len());
    let y = gpu.storage(d.elems() as u64);
    let steps = gn.forward(gpu, d, &xb, &gbb, &stats, &y);
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let yv = gpu.read(&y, d.elems() as usize);
    yv.iter().zip(dy.iter()).map(|(&yi, &di)| yi as f64 * di as f64).sum()
}

/// Forward + backward (spec §6 order); returns (dx [N*C*H*W], dgb [2C]).
fn analytic_grads(
    gpu: &Gpu,
    gn: &Gn,
    d: &GnDims,
    x: &[f32],
    gb: &[f32],
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let xb = gpu.storage_init("x", x);
    let gbb = gpu.storage_init("gb", gb);
    let stats = gpu.storage(d.stats_len());
    let y = gpu.storage(d.elems() as u64);
    let fwd = gn.forward(gpu, d, &xb, &gbb, &stats, &y);
    gpu.submit(&[], &fwd);
    gpu.poll_wait();

    let dyb = gpu.storage_init("dy", dy);
    let dyg = gpu.storage(d.elems() as u64);
    let sums = gpu.storage(d.sums_len());
    let dx = gpu.storage(d.elems() as u64);
    let dgb = gpu.storage_init("dgb", &vec![0.0f32; 2 * d.c as usize]); // pre-zeroed
    let bwd = gn.backward(gpu, d, &xb, &gbb, &stats, &dyb, &dyg, &sums, &dx, &dgb);
    gpu.submit(&[], &bwd);
    gpu.poll_wait();

    (gpu.read(&dx, d.elems() as usize), gpu.read(&dgb, 2 * d.c as usize))
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// Spec §10.3: directional FD check of the composed backward, for dx and for
/// dgamma/dbeta. Shape N=2, C=4, G=2, H=3, W=2, eps = 1e-5, LCG-seeded
/// x/gamma/beta/dy in [−1,1] with gamma offset to ~[0.5,1.5).
#[test]
fn gn_fd_backward_directional() {
    let d = GnDims::new(2, 4, 3, 2, 2, 1e-5).expect("valid dims");
    let n_el = d.elems() as usize;
    let n_gb = 2 * d.c as usize;

    let mut seed = 0x5EED_1BADu64;
    let x: Vec<f32> = (0..n_el).map(|_| lcg(&mut seed)).collect();
    // gamma offset to ~[0.5, 1.5) keeps dyg = dy*gamma well-conditioned.
    let mut gb: Vec<f32> = (0..d.c).map(|_| 1.0 + 0.5 * lcg(&mut seed)).collect();
    gb.extend((0..d.c).map(|_| lcg(&mut seed))); // beta in [-1,1)
    let dy: Vec<f32> = (0..n_el).map(|_| lcg(&mut seed)).collect();

    let gpu = Gpu::new(&Gn::kernel_sources());
    let gn = Gn::seq();
    let (dx, dgb) = analytic_grads(&gpu, &gn, &d, &x, &gb, &dy);

    let h = 5e-3f64; // global gradcheck step (playbook §3)
    let tol = |a: f64, n: f64| 4e-3 + 8e-2 * a.abs().max(n.abs());

    // dx: >= 2 random directions v over x; <dx, v> vs central difference.
    let mut dir_seed = 0xD12E_C710_0000_0001u64;
    for dir in 0..2 {
        let v: Vec<f32> = (0..n_el).map(|_| lcg(&mut dir_seed)).collect();
        let xp: Vec<f32> = x.iter().zip(&v).map(|(&xi, &vi)| xi + (h as f32) * vi).collect();
        let xm: Vec<f32> = x.iter().zip(&v).map(|(&xi, &vi)| xi - (h as f32) * vi).collect();
        let numeric = (loss(&gpu, &gn, &d, &xp, &gb, &dy) - loss(&gpu, &gn, &d, &xm, &gb, &dy))
            / (2.0 * h);
        let analytic = dot(&dx, &v);
        let err = (analytic - numeric).abs();
        println!("dx dir {dir}: analytic {analytic:.6e}, numeric {numeric:.6e}, err {err:.3e}");
        assert!(
            err <= tol(analytic, numeric),
            "dx FD dir {dir}: |{analytic} - {numeric}| = {err} > {}",
            tol(analytic, numeric)
        );
    }

    // dgamma/dbeta: >= 2 random directions over ALL 2C entries of gb;
    // <dgb, v> vs central difference of L under gb perturbation.
    for dir in 0..2 {
        let v: Vec<f32> = (0..n_gb).map(|_| lcg(&mut dir_seed)).collect();
        let gp: Vec<f32> = gb.iter().zip(&v).map(|(&gi, &vi)| gi + (h as f32) * vi).collect();
        let gm: Vec<f32> = gb.iter().zip(&v).map(|(&gi, &vi)| gi - (h as f32) * vi).collect();
        let numeric = (loss(&gpu, &gn, &d, &x, &gp, &dy) - loss(&gpu, &gn, &d, &x, &gm, &dy))
            / (2.0 * h);
        let analytic = dot(&dgb, &v);
        let err = (analytic - numeric).abs();
        println!("dgb dir {dir}: analytic {analytic:.6e}, numeric {numeric:.6e}, err {err:.3e}");
        assert!(
            err <= tol(analytic, numeric),
            "dgb FD dir {dir}: |{analytic} - {numeric}| = {err} > {}",
            tol(analytic, numeric)
        );
    }
}
