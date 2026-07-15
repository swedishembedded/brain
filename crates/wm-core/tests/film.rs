// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1.film unit tests — written FROM `docs/world-models/specs/P1.film.md`
//! (NEVER from the implementation): §6 hand-computed references (EXACT `==`
//! comparisons — every value is a small dyadic rational and every reduction
//! has <= 4 exactly-representable terms), §9 kernel-level finite-difference
//! entries (the `crates/gradcheck/tests/mse_fd.rs` pattern — NOT a
//! `CheckModel` registration; this unit has no trainable model), §10 required
//! test list, §11 bitwise determinism.
//!
//! Gating runs are `BRAIN_DEVICE=cpu` under `scripts/wm-locked-make.sh`,
//! which also sets `MOE_SKIP_GPU_TESTS=1`. These tests drive the WGSL
//! kernels on whatever backend `Gpu::new` selects (CPU in gating), exactly
//! like `mse_fd.rs`; they are deliberately NOT skipped under
//! `MOE_SKIP_GPU_TESTS` — spec §10/§12.6 requires all of them to RUN and
//! pass in the CPU-gated environment (a skip guard would make them never
//! gate at all; the guard convention is for tests that need a real GPU).
//!
//! FD tolerances are the GLOBAL gradcheck values (playbook §3, never
//! loosened): step `h = 5e-3`, pass iff
//! `|analytic - numeric| <= 4e-3 + 8e-2 * max(|analytic|, |numeric|)`.
//!
//! All test fn names start with `film_` — the red-check filter is `film_`.

use gpu_core::{Gpu, Step};
use wm_core::film::{Film, FilmChanDims, FilmRowDims};

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// FD step and global gradcheck tolerances (spec §9, playbook §3).
const H_FD: f32 = 5e-3;
const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

/// LCG in [-1, 1). Spec §10.8: use the CORRECT `>> 32` variant —
/// `mse_fd.rs`'s `>> 33` lands in [-1, 0) and must not be copied.
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 32) as f32 / (1u64 << 31) as f32) - 1.0
}

fn vec_seeded(n: usize, state: &mut u64) -> Vec<f32> {
    (0..n).map(|_| lcg(state)).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `a + s*v` element-wise (FD perturbation along a direction).
fn add_scaled(a: &[f32], v: &[f32], s: f32) -> Vec<f32> {
    a.iter().zip(v).map(|(x, y)| x + s * y).collect()
}

fn fd_pass(analytic: f32, numeric: f32) -> bool {
    (analytic - numeric).abs() <= ATOL + RTOL * analytic.abs().max(numeric.abs())
}

/// EXACT f32 equality, element-wise (spec §6: no tolerance on the hand
/// references; `-0.0 == 0.0` makes this safe at signed zeros).
fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            *g == *w,
            "{what}[{i}]: got {g}, want {w} (EXACT == required by spec §6)"
        );
    }
}

/// Build + submit the same step `passes` times over the SAME output buffer.
/// `passes = 2` pins the `=` OVERWRITE contract against an accumulating
/// (`+=`) mutant (spec §10.1 — gn round-2 lesson).
fn run_passes(gpu: &Gpu, mk: &dyn Fn() -> Step, passes: usize) {
    for _ in 0..passes {
        let st = mk();
        gpu.submit(&[], &[st]);
        gpu.poll_wait();
    }
}

/// One `Gpu` holding exactly the nine-family kernel table (spec §10:
/// `gpu_core::Gpu::new(&Film::kernel_sources())`, like `mse_fd.rs`).
fn film_gpu() -> (Gpu, Film) {
    (Gpu::new(&Film::kernel_sources()), Film::seq())
}

// --- channel family (NCHW; sb packed [N,2C]: scale first, shift second) ---

fn chan_fwd(gpu: &Gpu, f: &Film, d: &FilmChanDims, x: &[f32], sb: &[f32], passes: usize) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let sbb = gpu.storage_init("sb", sb);
    let y = gpu.storage(x.len() as u64);
    run_passes(gpu, &|| f.step_chan(gpu, d, &xb, &sbb, &y), passes);
    gpu.read(&y, x.len())
}

fn chan_dx(gpu: &Gpu, f: &Film, d: &FilmChanDims, dy: &[f32], sb: &[f32], passes: usize) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let sbb = gpu.storage_init("sb", sb);
    let dx = gpu.storage(dy.len() as u64);
    run_passes(gpu, &|| f.step_chan_dx(gpu, d, &dyb, &sbb, &dx), passes);
    gpu.read(&dx, dy.len())
}

fn chan_dsb(gpu: &Gpu, f: &Film, d: &FilmChanDims, x: &[f32], dy: &[f32], sb_len: usize, passes: usize) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let dyb = gpu.storage_init("dy", dy);
    let dsb = gpu.storage(sb_len as u64);
    run_passes(gpu, &|| f.step_chan_dsb(gpu, d, &xb, &dyb, &dsb), passes);
    gpu.read(&dsb, sb_len)
}

// --- row family ([R,D]; sb packed [NC,2D]) ---

fn row_fwd(gpu: &Gpu, f: &Film, d: &FilmRowDims, x: &[f32], sb: &[f32], passes: usize) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let sbb = gpu.storage_init("sb", sb);
    let y = gpu.storage(x.len() as u64);
    run_passes(gpu, &|| f.step_row(gpu, d, &xb, &sbb, &y), passes);
    gpu.read(&y, x.len())
}

fn row_dx(gpu: &Gpu, f: &Film, d: &FilmRowDims, dy: &[f32], sb: &[f32], passes: usize) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let sbb = gpu.storage_init("sb", sb);
    let dx = gpu.storage(dy.len() as u64);
    run_passes(gpu, &|| f.step_row_dx(gpu, d, &dyb, &sbb, &dx), passes);
    gpu.read(&dx, dy.len())
}

fn row_dsb(gpu: &Gpu, f: &Film, d: &FilmRowDims, x: &[f32], dy: &[f32], sb_len: usize, passes: usize) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let dyb = gpu.storage_init("dy", dy);
    let dsb = gpu.storage(sb_len as u64);
    run_passes(gpu, &|| f.step_row_dsb(gpu, d, &xb, &dyb, &dsb), passes);
    gpu.read(&dsb, sb_len)
}

// --- gated residual ([R,D]; gate g[NC,D], no packing) ---

fn gate_fwd(gpu: &Gpu, f: &Film, d: &FilmRowDims, x: &[f32], g: &[f32], h: &[f32], passes: usize) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let gb = gpu.storage_init("g", g);
    let hb = gpu.storage_init("h", h);
    let y = gpu.storage(x.len() as u64);
    run_passes(gpu, &|| f.step_gate(gpu, d, &xb, &gb, &hb, &y), passes);
    gpu.read(&y, x.len())
}

fn gate_dh(gpu: &Gpu, f: &Film, d: &FilmRowDims, dy: &[f32], g: &[f32], passes: usize) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let gb = gpu.storage_init("g", g);
    let dh = gpu.storage(dy.len() as u64);
    run_passes(gpu, &|| f.step_gate_dh(gpu, d, &dyb, &gb, &dh), passes);
    gpu.read(&dh, dy.len())
}

fn gate_dg(gpu: &Gpu, f: &Film, d: &FilmRowDims, dy: &[f32], h: &[f32], g_len: usize, passes: usize) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let hb = gpu.storage_init("h", h);
    let dg = gpu.storage(g_len as u64);
    run_passes(gpu, &|| f.step_gate_dg(gpu, d, &dyb, &hb, &dg), passes);
    gpu.read(&dg, g_len)
}

// ---------------------------------------------------------------------------
// §6.1 hand-computed reference — film_chan, N=1, C=2, H=W=2
// ---------------------------------------------------------------------------

// x flat index c*4 + h*2 + w: c0 = [1,2,3,4], c1 = [-1,0,2,3].
const CHAN_X: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 2.0, 3.0];
// sb[1,4] = [s0, s1, b0, b1] (scale first, shift second per row n).
const CHAN_SB: [f32; 4] = [0.5, -1.5, 0.25, 2.0];
const CHAN_Y: [f32; 8] = [1.75, 3.25, 4.75, 6.25, 2.5, 2.0, 1.0, 0.5];
const CHAN_DY: [f32; 8] = [1.0, -1.0, 2.0, 0.0, 0.5, 1.0, -1.0, 2.0];
const CHAN_DX: [f32; 8] = [1.5, -1.5, 3.0, 0.0, -0.25, -0.5, 0.5, -1.0];
// dsb packing [ds0, ds1, db0, db1].
const CHAN_DSB: [f32; 4] = [5.0, 3.5, 2.0, 2.5];

/// Spec §10.1.
#[test]
fn film_chan_forward_matches_hand_reference() {
    let (gpu, f) = film_gpu();
    let d = FilmChanDims::new(1, 2, 2, 2).expect("valid dims");
    let y = chan_fwd(&gpu, &f, &d, &CHAN_X, &CHAN_SB, 1);
    assert_exact(&y, &CHAN_Y, "film_chan y (single dispatch)");
    // Dispatch TWICE over the same y buffer: `=` overwrite, not accumulate.
    let y2 = chan_fwd(&gpu, &f, &d, &CHAN_X, &CHAN_SB, 2);
    assert_exact(&y2, &CHAN_Y, "film_chan y (double dispatch, overwrite contract)");
}

/// Spec §10.2.
#[test]
fn film_chan_backward_matches_hand_reference() {
    let (gpu, f) = film_gpu();
    let d = FilmChanDims::new(1, 2, 2, 2).expect("valid dims");
    let dx = chan_dx(&gpu, &f, &d, &CHAN_DY, &CHAN_SB, 1);
    assert_exact(&dx, &CHAN_DX, "film_chan_dx");
    let dsb = chan_dsb(&gpu, &f, &d, &CHAN_X, &CHAN_DY, CHAN_DSB.len(), 1);
    assert_exact(&dsb, &CHAN_DSB, "film_chan_dsb (single dispatch)");
    let dsb2 = chan_dsb(&gpu, &f, &d, &CHAN_X, &CHAN_DY, CHAN_DSB.len(), 2);
    assert_exact(&dsb2, &CHAN_DSB, "film_chan_dsb (double dispatch, overwrite contract)");
}

// ---------------------------------------------------------------------------
// §6.2 hand-computed reference — film_row, R=4, D=2, rows_per_cond=2 (NC=2)
// ---------------------------------------------------------------------------

// x flat index r*2 + d.
const ROW_X: [f32; 8] = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0, -2.0, 1.0];
// sb[2,4], per group [s_d0, s_d1, b_d0, b_d1].
const ROW_SB: [f32; 8] = [0.5, -1.0, 1.0, 0.5, -0.5, 2.0, 0.0, -1.0];
const ROW_Y: [f32; 8] = [2.5, 0.5, 5.5, 0.5, 0.25, 11.0, -1.0, 2.0];
const ROW_DY: [f32; 8] = [1.0, 0.0, -1.0, 2.0, 0.5, 1.0, 2.0, -1.0];
const ROW_DX: [f32; 8] = [1.5, 0.0, -1.5, 0.0, 0.25, 3.0, 1.0, -3.0];
const ROW_DSB: [f32; 8] = [-2.0, -2.0, 0.0, 2.0, -3.75, 3.0, 2.5, 0.0];

/// Spec §10.3.
#[test]
fn film_row_forward_matches_hand_reference() {
    let (gpu, f) = film_gpu();
    let d = FilmRowDims::new(4, 2, 2).expect("valid dims");
    let y = row_fwd(&gpu, &f, &d, &ROW_X, &ROW_SB, 1);
    assert_exact(&y, &ROW_Y, "film_row y (single dispatch)");
    let y2 = row_fwd(&gpu, &f, &d, &ROW_X, &ROW_SB, 2);
    assert_exact(&y2, &ROW_Y, "film_row y (double dispatch, overwrite contract)");
}

/// Spec §10.4.
#[test]
fn film_row_backward_matches_hand_reference() {
    let (gpu, f) = film_gpu();
    let d = FilmRowDims::new(4, 2, 2).expect("valid dims");
    let dx = row_dx(&gpu, &f, &d, &ROW_DY, &ROW_SB, 1);
    assert_exact(&dx, &ROW_DX, "film_row_dx");
    let dsb = row_dsb(&gpu, &f, &d, &ROW_X, &ROW_DY, ROW_DSB.len(), 1);
    assert_exact(&dsb, &ROW_DSB, "film_row_dsb (single dispatch)");
    let dsb2 = row_dsb(&gpu, &f, &d, &ROW_X, &ROW_DY, ROW_DSB.len(), 2);
    assert_exact(&dsb2, &ROW_DSB, "film_row_dsb (double dispatch, overwrite contract)");
}

// ---------------------------------------------------------------------------
// §6.3 hand-computed reference — gate_row, R=4, D=2, rows_per_cond=2 (NC=2)
// ---------------------------------------------------------------------------

const GATE_X: [f32; 8] = [1.0, 2.0, -1.0, 0.0, 2.0, 1.0, 0.0, -1.0];
const GATE_H: [f32; 8] = [0.5, 1.0, 2.0, -1.0, 1.0, 0.5, -2.0, 1.0];
// g[NC,D] plain (no packing): cond0 = [2, -0.5], cond1 = [0.5, 1].
const GATE_G: [f32; 4] = [2.0, -0.5, 0.5, 1.0];
const GATE_Y: [f32; 8] = [2.0, 1.5, 3.0, 0.5, 2.5, 1.5, -1.0, 0.0];
const GATE_DY: [f32; 8] = [1.0, -1.0, 0.5, 2.0, -1.0, 0.0, 2.0, 1.0];
const GATE_DH: [f32; 8] = [2.0, 0.5, 1.0, -1.0, -0.5, 0.0, 1.0, 1.0];
const GATE_DG: [f32; 4] = [1.5, -3.0, -5.0, 1.0];
// dx of gate_row is the IDENTITY (dx = dy) with NO kernel — spec §1/§3.3;
// validated by the FD test film_fd_gate_backward_directional (§10.10).

/// Spec §10.5.
#[test]
fn film_gate_forward_matches_hand_reference() {
    let (gpu, f) = film_gpu();
    let d = FilmRowDims::new(4, 2, 2).expect("valid dims");
    let y = gate_fwd(&gpu, &f, &d, &GATE_X, &GATE_G, &GATE_H, 1);
    assert_exact(&y, &GATE_Y, "gate_row y (single dispatch)");
    let y2 = gate_fwd(&gpu, &f, &d, &GATE_X, &GATE_G, &GATE_H, 2);
    assert_exact(&y2, &GATE_Y, "gate_row y (double dispatch, overwrite contract)");
}

/// Spec §10.6.
#[test]
fn film_gate_backward_matches_hand_reference() {
    let (gpu, f) = film_gpu();
    let d = FilmRowDims::new(4, 2, 2).expect("valid dims");
    let dh = gate_dh(&gpu, &f, &d, &GATE_DY, &GATE_G, 1);
    assert_exact(&dh, &GATE_DH, "gate_row_dh");
    let dg = gate_dg(&gpu, &f, &d, &GATE_DY, &GATE_H, GATE_DG.len(), 1);
    assert_exact(&dg, &GATE_DG, "gate_row_dg (single dispatch)");
    let dg2 = gate_dg(&gpu, &f, &d, &GATE_DY, &GATE_H, GATE_DG.len(), 2);
    assert_exact(&dg2, &GATE_DG, "gate_row_dg (double dispatch, overwrite contract)");
}

// ---------------------------------------------------------------------------
// §10.7 property — zero modulation is the identity (exact ==)
// ---------------------------------------------------------------------------

/// `s = b = 0` => film_chan / film_row are the identity; `g = 0` =>
/// gate_row is the identity on x. Also exercises both rows_per_cond
/// extremes (spec §8): row uses rows_per_cond=1 (NC=R, per-row cond),
/// gate uses rows_per_cond=R (NC=1, classic adaLN).
#[test]
fn film_identity_when_modulation_zero() {
    let (gpu, f) = film_gpu();
    let mut st = 0xF11A_5EED_0001u64;

    // channel: N=2, C=3, H=W=2 (24 elems), sb = 0 with len 2*N*C = 12.
    let dc = FilmChanDims::new(2, 3, 2, 2).expect("valid dims");
    let xc = vec_seeded(24, &mut st);
    let yc = chan_fwd(&gpu, &f, &dc, &xc, &vec![0.0f32; 12], 1);
    assert_exact(&yc, &xc, "film_chan identity at s=b=0");

    // row: R=4, D=3, rows_per_cond=1 => NC=4 (extreme), sb = 0 len 2*4*3.
    let dr = FilmRowDims::new(4, 3, 1).expect("valid dims");
    let xr = vec_seeded(12, &mut st);
    let yr = row_fwd(&gpu, &f, &dr, &xr, &vec![0.0f32; 24], 1);
    assert_exact(&yr, &xr, "film_row identity at s=b=0 (rows_per_cond=1)");

    // gate: R=4, D=3, rows_per_cond=R=4 => NC=1 (other extreme), g = 0.
    let dg = FilmRowDims::new(4, 3, 4).expect("valid dims");
    let xg = vec_seeded(12, &mut st);
    let hg = vec_seeded(12, &mut st);
    let yg = gate_fwd(&gpu, &f, &dg, &xg, &vec![0.0f32; 3], &hg, 1);
    assert_exact(&yg, &xg, "gate_row identity at g=0 (rows_per_cond=R)");
}

// ---------------------------------------------------------------------------
// §9 / §10.8-10.10 — kernel-level directional finite differences
// (scalar loss L = sum_i w_i * y_i; backward runs with dy := w)
// ---------------------------------------------------------------------------

/// Spec §9.1 / §10.8 — chan, N=2, C=3, H=2, W=2.
#[test]
fn film_fd_chan_backward_directional() {
    let (gpu, f) = film_gpu();
    let d = FilmChanDims::new(2, 3, 2, 2).expect("valid dims");
    let (ne, nsb) = (24usize, 12usize);
    let mut st = 0x00C0_FFEE_D15Cu64;
    let x = vec_seeded(ne, &mut st);
    let sb = vec_seeded(nsb, &mut st);
    let w = vec_seeded(ne, &mut st);

    let loss = |xv: &[f32], sbv: &[f32]| -> f32 { dot(&w, &chan_fwd(&gpu, &f, &d, xv, sbv, 1)) };
    let dx = chan_dx(&gpu, &f, &d, &w, &sb, 1);
    let dsb = chan_dsb(&gpu, &f, &d, &x, &w, nsb, 1);

    for dir in 0..2 {
        // perturb x: <dx, v> vs central difference.
        let v = vec_seeded(ne, &mut st);
        let a = dot(&dx, &v);
        let n = (loss(&add_scaled(&x, &v, H_FD), &sb) - loss(&add_scaled(&x, &v, -H_FD), &sb)) / (2.0 * H_FD);
        assert!(fd_pass(a, n), "chan dx dir {dir}: analytic {a} vs numeric {n}");

        // perturb sb: <dsb, v> vs central difference.
        let vs = vec_seeded(nsb, &mut st);
        let a2 = dot(&dsb, &vs);
        let n2 = (loss(&x, &add_scaled(&sb, &vs, H_FD)) - loss(&x, &add_scaled(&sb, &vs, -H_FD))) / (2.0 * H_FD);
        assert!(fd_pass(a2, n2), "chan dsb dir {dir}: analytic {a2} vs numeric {n2}");
    }
}

/// Spec §9.2 / §10.9 — row, R=6, D=4, rows_per_cond=3 (NC=2; exercises
/// rows_per_cond not in {1, R}).
#[test]
fn film_fd_row_backward_directional() {
    let (gpu, f) = film_gpu();
    let d = FilmRowDims::new(6, 4, 3).expect("valid dims");
    let (ne, nsb) = (24usize, 16usize); // R*D, 2*NC*D
    let mut st = 0x0B0B_5EED_0002u64;
    let x = vec_seeded(ne, &mut st);
    let sb = vec_seeded(nsb, &mut st);
    let w = vec_seeded(ne, &mut st);

    let loss = |xv: &[f32], sbv: &[f32]| -> f32 { dot(&w, &row_fwd(&gpu, &f, &d, xv, sbv, 1)) };
    let dx = row_dx(&gpu, &f, &d, &w, &sb, 1);
    let dsb = row_dsb(&gpu, &f, &d, &x, &w, nsb, 1);

    for dir in 0..2 {
        let v = vec_seeded(ne, &mut st);
        let a = dot(&dx, &v);
        let n = (loss(&add_scaled(&x, &v, H_FD), &sb) - loss(&add_scaled(&x, &v, -H_FD), &sb)) / (2.0 * H_FD);
        assert!(fd_pass(a, n), "row dx dir {dir}: analytic {a} vs numeric {n}");

        let vs = vec_seeded(nsb, &mut st);
        let a2 = dot(&dsb, &vs);
        let n2 = (loss(&x, &add_scaled(&sb, &vs, H_FD)) - loss(&x, &add_scaled(&sb, &vs, -H_FD))) / (2.0 * H_FD);
        assert!(fd_pass(a2, n2), "row dsb dir {dir}: analytic {a2} vs numeric {n2}");
    }
}

/// Spec §9.3 / §10.10 — gate, R=4, D=3, rows_per_cond=2, including the
/// documented `dx = dy` identity direction check with NO kernel dispatched.
#[test]
fn film_fd_gate_backward_directional() {
    let (gpu, f) = film_gpu();
    let d = FilmRowDims::new(4, 3, 2).expect("valid dims");
    let (ne, ng) = (12usize, 6usize); // R*D, NC*D
    let mut st = 0x6A7E_5EED_0003u64;
    let x = vec_seeded(ne, &mut st);
    let g = vec_seeded(ng, &mut st);
    let h = vec_seeded(ne, &mut st);
    let w = vec_seeded(ne, &mut st);

    let loss = |xv: &[f32], gv: &[f32], hv: &[f32]| -> f32 { dot(&w, &gate_fwd(&gpu, &f, &d, xv, gv, hv, 1)) };
    let dh = gate_dh(&gpu, &f, &d, &w, &g, 1);
    let dg = gate_dg(&gpu, &f, &d, &w, &h, ng, 1);

    for dir in 0..2 {
        // perturb h: <dh, v>.
        let v = vec_seeded(ne, &mut st);
        let a = dot(&dh, &v);
        let n = (loss(&x, &g, &add_scaled(&h, &v, H_FD)) - loss(&x, &g, &add_scaled(&h, &v, -H_FD))) / (2.0 * H_FD);
        assert!(fd_pass(a, n), "gate dh dir {dir}: analytic {a} vs numeric {n}");

        // perturb g: <dg, v>.
        let vg = vec_seeded(ng, &mut st);
        let a2 = dot(&dg, &vg);
        let n2 = (loss(&x, &add_scaled(&g, &vg, H_FD), &h) - loss(&x, &add_scaled(&g, &vg, -H_FD), &h)) / (2.0 * H_FD);
        assert!(fd_pass(a2, n2), "gate dg dir {dir}: analytic {a2} vs numeric {n2}");

        // perturb x: dx = dy is the IDENTITY (spec §1/§3.3) — the analytic
        // directional derivative is <dy, v> = <w, v>, NO kernel dispatched.
        let vx = vec_seeded(ne, &mut st);
        let a3 = dot(&w, &vx);
        let n3 = (loss(&add_scaled(&x, &vx, H_FD), &g, &h) - loss(&add_scaled(&x, &vx, -H_FD), &g, &h)) / (2.0 * H_FD);
        assert!(fd_pass(a3, n3), "gate dx=dy identity dir {dir}: analytic {a3} vs numeric {n3}");
    }
}

// ---------------------------------------------------------------------------
// §10.11 — bitwise determinism (fixed seed, run twice, f32::to_bits equal)
// ---------------------------------------------------------------------------

/// Every forward+backward kernel of the family runs twice on identical
/// seeded inputs (fresh buffers each run); ALL outputs — y(x3), dx(x2),
/// dsb(x2), dh, dg — must be BITWISE equal (spec §11: single writer per
/// element, fixed ascending reduction order).
#[test]
fn film_deterministic_bitwise() {
    let (gpu, f) = film_gpu();
    let mut st = 0xDE7E_2814_0004u64;

    // chan fixture: N=2, C=3, H=W=2.
    let dc = FilmChanDims::new(2, 3, 2, 2).expect("valid dims");
    let xc = vec_seeded(24, &mut st);
    let sbc = vec_seeded(12, &mut st);
    let dyc = vec_seeded(24, &mut st);
    // row fixture: R=6, D=4, rows_per_cond=3.
    let dr = FilmRowDims::new(6, 4, 3).expect("valid dims");
    let xr = vec_seeded(24, &mut st);
    let sbr = vec_seeded(16, &mut st);
    let dyr = vec_seeded(24, &mut st);
    // gate fixture: R=4, D=3, rows_per_cond=2.
    let dgm = FilmRowDims::new(4, 3, 2).expect("valid dims");
    let xg = vec_seeded(12, &mut st);
    let gg = vec_seeded(6, &mut st);
    let hg = vec_seeded(12, &mut st);
    let dyg = vec_seeded(12, &mut st);

    let names = ["chan y", "chan dx", "chan dsb", "row y", "row dx", "row dsb", "gate y", "gate dh", "gate dg"];
    let run_all = || -> Vec<Vec<f32>> {
        vec![
            chan_fwd(&gpu, &f, &dc, &xc, &sbc, 1),
            chan_dx(&gpu, &f, &dc, &dyc, &sbc, 1),
            chan_dsb(&gpu, &f, &dc, &xc, &dyc, 12, 1),
            row_fwd(&gpu, &f, &dr, &xr, &sbr, 1),
            row_dx(&gpu, &f, &dr, &dyr, &sbr, 1),
            row_dsb(&gpu, &f, &dr, &xr, &dyr, 16, 1),
            gate_fwd(&gpu, &f, &dgm, &xg, &gg, &hg, 1),
            gate_dh(&gpu, &f, &dgm, &dyg, &gg, 1),
            gate_dg(&gpu, &f, &dgm, &dyg, &hg, 6, 1),
        ]
    };

    let r1 = run_all();
    let r2 = run_all();
    for (k, (a, b)) in r1.iter().zip(&r2).enumerate() {
        assert_eq!(a.len(), b.len(), "{}: length changed between runs", names[k]);
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "{}[{i}]: {va} vs {vb} not bitwise equal across runs",
                names[k]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §10.12 / §8 — validated-constructor error paths and derived values
// ---------------------------------------------------------------------------

#[test]
fn film_dims_validate_rejects() {
    // FilmChanDims: each zero dim is an error.
    assert!(FilmChanDims::new(0, 2, 2, 2).is_err(), "n=0 must be rejected");
    assert!(FilmChanDims::new(1, 0, 2, 2).is_err(), "c=0 must be rejected");
    assert!(FilmChanDims::new(1, 2, 0, 2).is_err(), "h=0 must be rejected");
    assert!(FilmChanDims::new(1, 2, 2, 0).is_err(), "w=0 must be rejected");

    // FilmRowDims: each zero dim is an error.
    assert!(FilmRowDims::new(0, 4, 1).is_err(), "r=0 must be rejected");
    assert!(FilmRowDims::new(6, 0, 3).is_err(), "d=0 must be rejected");
    assert!(FilmRowDims::new(6, 4, 0).is_err(), "rows_per_cond=0 must be rejected");
    // Non-divisible grouping: ragged final group is undefined => Err.
    assert!(FilmRowDims::new(5, 4, 2).is_err(), "r=5 % rows_per_cond=2 != 0 must be rejected");

    // Accepted dims report the spec'd derived values.
    let dr = FilmRowDims::new(6, 4, 3).expect("6 % 3 == 0 is valid");
    assert_eq!(dr.conds(), 2, "conds = r / rows_per_cond");
    assert_eq!(dr.cond_elems(), 8, "cond_elems = conds * d");
    assert_eq!(dr.sb_len(), 16, "sb_len = 2 * conds * d");
    assert_eq!(dr.g_len(), 8, "g_len = conds * d");

    let dc = FilmChanDims::new(2, 3, 2, 2).expect("nonzero dims are valid");
    assert_eq!(dc.elems(), 24, "elems = n*c*h*w");
    assert_eq!(dc.pairs(), 6, "pairs = n*c");
    assert_eq!(dc.sb_len(), 12, "sb_len = 2*n*c");
}
