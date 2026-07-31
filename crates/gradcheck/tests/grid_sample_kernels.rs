// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the `grid_sample` kernel family, driven directly through
//! `gpu_core` like `depth_kernels.rs` / `glue.rs` — no model is built.
//!
//! The contract these tests exist to pin down is
//! `torch.nn.functional.grid_sample(input, grid, mode='bilinear',
//! padding_mode='zeros', align_corners=...)`, for `input [N,C,H,W]` and
//! `grid [N,Ho,Wo,2]` whose last axis is (X, Y) in normalized `[-1,1]`.
//!
//! Three independent techniques, because each catches a class the others miss:
//!
//! 1. **Hand-computed goldens for the conventions.** `align_corners` and the
//!    (X, Y) axis order are the two things that cannot be caught by any
//!    self-consistency check: a kernel that unnormalizes with the wrong
//!    convention, or that reads `grid[...,0]` as a row index, is *perfectly*
//!    self-consistent — its backward is the correct backward of the wrong
//!    forward, so finite differences and adjointness both pass. Only an external
//!    number distinguishes them. `half_pixel_convention_golden` and
//!    `grid_last_axis_is_x_then_y` are those numbers, hand-derived in their own
//!    doc comments from PyTorch's formulas, on a NON-square input so an axis
//!    swap cannot hide.
//!
//! 2. **Adjointness for `dx`.** For a FIXED grid, `grid_sample` is linear in
//!    `x`, so `grid_sample_dx` is exactly `Aᵀ` and `<A(x), dy> == <x, Aᵀ(dy)>`
//!    holds to fp32 round-off for all `x`, `dy`. That is far sharper than FD and
//!    it exercises the atomic-free gather inversion at every out-of-bounds
//!    corner at once — the grids used here deliberately run off the edge.
//!
//! 3. **Central finite differences for `dgrid`.** The output is genuinely
//!    nonlinear in the grid, so FD is the honest check. `grid_sample` is
//!    piecewise-bilinear in `(ix, iy)`, which means FD is only valid while the
//!    perturbation stays inside one cell and inside the frame — crossing a pixel
//!    boundary or the border changes which taps exist and puts a kink under the
//!    difference. `safe_interior_grid` therefore constructs coordinates whose
//!    fractional part is pinned into `[0.15, 0.85]` and which sit at least one
//!    pixel inside the border; the out-of-bounds behaviour is covered instead by
//!    the exact reference comparison in `dgrid_matches_reference` and by
//!    `zeros_padding_drops_out_of_bounds_taps`.
//!
//! The CPU oracle at the bottom is re-derived in `f64` from the PyTorch
//! definition rather than shared with anything the kernels use, per AGENTS.md's
//! gradcheck-oracle exception.
//!
//! Run with `BRAIN_DEVICE=cpu`; `MOE_SKIP_GPU_TESTS` skips (the `dx` gather is
//! O(Ho*Wo) per input element by construction — see grid_sample_dx.wgsl — so the
//! shapes here are deliberately tiny, but the gate is honoured anyway).

use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("grid_sample", kernels::GRID_SAMPLE),              // 0
    ("grid_sample_dx", kernels::GRID_SAMPLE_DX),        // 1
    ("grid_sample_dgrid", kernels::GRID_SAMPLE_DGRID),  // 2
];
const K_FWD: usize = 0;
const K_DX: usize = 1;
const K_DGRID: usize = 2;

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

// ---- shapes -----------------------------------------------------------------

#[derive(Clone, Copy)]
struct Shape {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    ho: usize,
    wo: usize,
    align: bool,
}

impl Shape {
    /// The family's shared 7-word Params: [N, C, H, W, Ho, Wo, align_corners].
    fn params(&self) -> [u32; 7] {
        [
            self.n as u32,
            self.c as u32,
            self.h as u32,
            self.w as u32,
            self.ho as u32,
            self.wo as u32,
            u32::from(self.align),
        ]
    }
    fn xn(&self) -> usize {
        self.n * self.c * self.h * self.w
    }
    fn yn(&self) -> usize {
        self.n * self.c * self.ho * self.wo
    }
    fn gn(&self) -> usize {
        self.n * self.ho * self.wo * 2
    }
}

// ---- deterministic noise (never `rand`) -------------------------------------

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1,1)
}
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n).map(|_| lcg(&mut st)).collect()
}
/// Grid values in ~[-1.3, 1.3): deliberately WIDER than the valid range so that
/// roughly a third of the samples have one or more taps outside the frame. That
/// is the `padding_mode='zeros'` path, and it is where a clamp-to-edge mistake
/// lives.
fn randgrid(seed: u64, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n).map(|_| lcg(&mut st) * 1.3).collect()
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

// ---- dispatch helpers -------------------------------------------------------

fn run_fwd(gpu: &Gpu, s: &Shape, x: &[f32], grid: &[f32]) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let gb = gpu.storage_init("grid", grid);
    let yb = gpu.storage(s.yn() as u64);
    let st = gpu.step(K_FWD, &[&xb, &gb, &yb], &s.params(), s.yn() as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&yb, s.yn())
}

fn run_dx(gpu: &Gpu, s: &Shape, dy: &[f32], grid: &[f32]) -> Vec<f32> {
    let db = gpu.storage_init("dy", dy);
    let gb = gpu.storage_init("grid", grid);
    let xb = gpu.storage(s.xn() as u64);
    let st = gpu.step(K_DX, &[&db, &gb, &xb], &s.params(), s.xn() as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&xb, s.xn())
}

fn run_dgrid(gpu: &Gpu, s: &Shape, dy: &[f32], x: &[f32], grid: &[f32]) -> Vec<f32> {
    let db = gpu.storage_init("dy", dy);
    let xb = gpu.storage_init("x", x);
    let gb = gpu.storage_init("grid", grid);
    let ob = gpu.storage(s.gn() as u64);
    // One invocation per GRID POSITION (n, ho, wo) — it writes BOTH components.
    let threads = (s.n * s.ho * s.wo) as u32;
    let st = gpu.step(K_DGRID, &[&db, &xb, &gb, &ob], &s.params(), threads);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&ob, s.gn())
}

// ---- forward ----------------------------------------------------------------

/// Forward parity against the f64 CPU oracle, with a grid that spills off every
/// edge so the zeros-padding tap dropping is exercised on the same run.
#[test]
fn fwd_matches_reference_half_pixel() {
    if skip() {
        return;
    }
    let s = Shape { n: 2, c: 3, h: 5, w: 7, ho: 4, wo: 6, align: false };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = randvec(1, s.xn());
    let grid = randgrid(2, s.gn());
    let y = run_fwd(&gpu, &s, &x, &grid);
    let r = gs_ref(&s, &x, &grid);
    let d = max_abs_diff(&y, &r);
    assert!(d < 1e-5, "grid_sample (align_corners=false) vs reference: max |diff| = {d:.3e}");
}

/// Same, align_corners=true. Both modes ship because both are needed and neither
/// can be inferred from the other.
#[test]
fn fwd_matches_reference_align_corners() {
    if skip() {
        return;
    }
    let s = Shape { n: 2, c: 3, h: 5, w: 7, ho: 4, wo: 6, align: true };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = randvec(3, s.xn());
    let grid = randgrid(4, s.gn());
    let y = run_fwd(&gpu, &s, &x, &grid);
    let r = gs_ref(&s, &x, &grid);
    let d = max_abs_diff(&y, &r);
    assert!(d < 1e-5, "grid_sample (align_corners=true) vs reference: max |diff| = {d:.3e}");
}

/// THE half-pixel golden. On a 2x2 input `[[a,b],[c,d]]` sampled at the single
/// point `grid = (-1, -1)`:
///
///   align_corners = true : ix = ((-1+1)/2)*(2-1) = 0, iy = 0
///                          -> the NW tap alone, weight 1  -> exactly `a`.
///   align_corners = false: ix = ((-1+1)*2 - 1)/2 = -0.5, iy = -0.5
///                          -> x0 = -1 (dropped), x1 = 0 with fx = 0.5, and the
///                             same in y, so only the SE tap survives with
///                             weight 0.5*0.5 -> `0.25 * a`.
///
/// A factor of four apart, from one flag. Nothing else in this file can tell
/// these two kernels apart.
#[test]
fn half_pixel_convention_golden() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = [2.0f32, -3.0, 5.0, 7.0]; // a, b, c, d
    let grid = [-1.0f32, -1.0];

    let s_t = Shape { n: 1, c: 1, h: 2, w: 2, ho: 1, wo: 1, align: true };
    let y_t = run_fwd(&gpu, &s_t, &x, &grid);
    assert!((y_t[0] - 2.0).abs() < 1e-6, "align_corners=true at (-1,-1) must be x[0,0]=2, got {}", y_t[0]);

    let s_f = Shape { align: false, ..s_t };
    let y_f = run_fwd(&gpu, &s_f, &x, &grid);
    assert!((y_f[0] - 0.5).abs() < 1e-6, "align_corners=false at (-1,-1) must be 0.25*x[0,0]=0.5, got {}", y_f[0]);
}

/// The grid's last axis is (X, Y): component 0 indexes W, component 1 indexes H.
/// On a 2x3 input (H=2, W=3 — deliberately non-square, so a swap cannot land on
/// a valid index and pass by accident), align_corners=false:
///     gx = 0.0  -> ix = ((0+1)*3 - 1)/2 = 1.0
///     gy = -0.5 -> iy = ((-0.5+1)*2 - 1)/2 = 0.0
/// so the sample is exactly the single pixel x[0,1]. Reading the components the
/// other way round asks for x[1, 0.0-ish] and returns a different number.
#[test]
fn grid_last_axis_is_x_then_y() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    // row-major [H=2, W=3]
    let x = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let grid = [0.0f32, -0.5];
    let s = Shape { n: 1, c: 1, h: 2, w: 3, ho: 1, wo: 1, align: false };
    let y = run_fwd(&gpu, &s, &x, &grid);
    assert!((y[0] - 20.0).abs() < 1e-5, "grid (x=0.0, y=-0.5) on a 2x3 input must select x[0,1]=20, got {} (axes swapped?)", y[0]);
}

/// The grid is shared by every channel, and the sample of channel `c` must use
/// plane `c` and nothing else. Zeroing one input channel may change exactly that
/// output channel — a wrong plane stride still produces plausible numbers.
#[test]
fn channels_are_independent() {
    if skip() {
        return;
    }
    let s = Shape { n: 2, c: 4, h: 5, w: 6, ho: 3, wo: 3, align: false };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let grid = randgrid(5, s.gn());
    let x = randvec(6, s.xn());
    let base = run_fwd(&gpu, &s, &x, &grid);

    let zc = 2usize;
    let mut xz = x.clone();
    for n in 0..s.n {
        let off = (n * s.c + zc) * s.h * s.w;
        for v in xz[off..off + s.h * s.w].iter_mut() {
            *v = 0.0;
        }
    }
    let got = run_fwd(&gpu, &s, &xz, &grid);
    for n in 0..s.n {
        for c in 0..s.c {
            let off = (n * s.c + c) * s.ho * s.wo;
            for i in 0..s.ho * s.wo {
                if c == zc {
                    assert!(got[off + i].abs() < 1e-6, "channel {c} should be zeroed, got {}", got[off + i]);
                } else {
                    assert!(
                        (got[off + i] - base[off + i]).abs() < 1e-6,
                        "zeroing input channel {zc} changed output channel {c} (index {i})"
                    );
                }
            }
        }
    }
}

/// `padding_mode='zeros'` is not clamp-to-edge: a sample whose whole 2x2
/// neighbourhood is outside the frame produces 0, and contributes nothing to
/// either gradient. A clamp implementation returns the nearest edge pixel here
/// and passes every self-consistency test in this file.
#[test]
fn zeros_padding_drops_out_of_bounds_taps() {
    if skip() {
        return;
    }
    let s = Shape { n: 1, c: 2, h: 4, w: 4, ho: 2, wo: 2, align: false };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = randvec(7, s.xn());
    // Every sample is far outside [-1,1] in both axes.
    let grid: Vec<f32> = (0..s.gn()).map(|i| if i % 2 == 0 { -4.0 } else { 4.0 }).collect();

    let y = run_fwd(&gpu, &s, &x, &grid);
    assert!(y.iter().all(|v| v.abs() < 1e-6), "out-of-bounds samples must be 0 under padding_mode='zeros', got {y:?}");

    let dy = randvec(8, s.yn());
    let dx = run_dx(&gpu, &s, &dy, &grid);
    assert!(dx.iter().all(|v| v.abs() < 1e-6), "out-of-bounds samples must not deposit any dx");

    let dg = run_dgrid(&gpu, &s, &dy, &x, &grid);
    assert!(dg.iter().all(|v| v.abs() < 1e-6), "out-of-bounds samples must have zero coordinate gradient");
}

// ---- dx ---------------------------------------------------------------------

/// For a fixed grid the forward is LINEAR in x, so `grid_sample_dx` must be its
/// exact transpose. This is the sharp test of the atomic-free gather inversion,
/// and the wide grid means many samples have taps off the edge — precisely the
/// case where a gather that forgets the zeros-padding rule leaks weight.
fn assert_dx_adjoint(s: Shape, seed: u64) {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = randvec(seed, s.xn());
    let grid = randgrid(seed + 100, s.gn());
    let dy = randvec(seed + 200, s.yn());

    let y = run_fwd(&gpu, &s, &x, &grid);
    let dx = run_dx(&gpu, &s, &dy, &grid);

    let lhs = dot(&y, &dy);
    let rhs = dot(&x, &dx);
    let tol = 1e-4 * lhs.abs().max(rhs.abs()).max(1.0);
    assert!(
        (lhs - rhs).abs() < tol,
        "grid_sample_dx adjointness broken (align={}): <A(x),dy> = {lhs}, <x,A^T(dy)> = {rhs} (diff {:.3e})",
        s.align,
        (lhs - rhs).abs()
    );
}

#[test]
fn dx_is_the_adjoint_half_pixel() {
    if skip() {
        return;
    }
    assert_dx_adjoint(Shape { n: 2, c: 3, h: 5, w: 7, ho: 4, wo: 6, align: false }, 11);
}

#[test]
fn dx_is_the_adjoint_align_corners() {
    if skip() {
        return;
    }
    assert_dx_adjoint(Shape { n: 2, c: 3, h: 5, w: 7, ho: 4, wo: 6, align: true }, 21);
}

/// Adjointness is a scalar identity: it can be satisfied by a dx whose mass is
/// in the wrong places as long as the total inner product matches. Compare
/// element-by-element against the oracle's honest CPU scatter as well.
#[test]
fn dx_matches_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for align in [false, true] {
        let s = Shape { n: 2, c: 2, h: 5, w: 6, ho: 4, wo: 4, align };
        let grid = randgrid(31, s.gn());
        let dy = randvec(32, s.yn());
        let got = run_dx(&gpu, &s, &dy, &grid);
        let want = gs_dx_ref(&s, &grid, &dy);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-5, "grid_sample_dx (align={align}) vs reference: max |diff| = {d:.3e}");
    }
}

// ---- dgrid ------------------------------------------------------------------

/// A grid whose unnormalized coordinates are pinned strictly inside the frame
/// and away from pixel boundaries, so a small perturbation stays inside one
/// bilinear cell and central differences are meaningful. See the module doc.
fn safe_interior_grid(s: &Shape, seed: u64) -> Vec<f32> {
    let mut st = seed;
    let mut g = vec![0f32; s.gn()];
    for i in 0..s.n * s.ho * s.wo {
        for (a, size) in [(0usize, s.w), (1usize, s.h)] {
            // uniform-ish in [0.6, size-1.6], then push the fraction into [0.15, 0.85]
            let u = (lcg(&mut st) as f64 + 1.0) * 0.5; // [0,1)
            let mut c = 0.6 + u * ((size as f64) - 2.2).max(0.0);
            let fr = c - c.floor();
            if fr < 0.15 {
                c += 0.15 - fr;
            } else if fr > 0.85 {
                c -= fr - 0.85;
            }
            // invert the unnormalization for this mode
            let gv = if s.align {
                2.0 * c / ((size as f64) - 1.0) - 1.0
            } else {
                (2.0 * c + 1.0) / (size as f64) - 1.0
            };
            g[i * 2 + a] = gv as f32;
        }
    }
    g
}

fn assert_dgrid_fd(s: Shape, seed: u64) {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = randvec(seed, s.xn());
    let grid = safe_interior_grid(&s, seed + 100);
    let dy = randvec(seed + 200, s.yn());

    let ana = run_dgrid(&gpu, &s, &dy, &x, &grid);

    // L(grid) = <grid_sample(x, grid), dy>, evaluated in f64.
    let eps = 1e-3f64;
    for i in 0..s.gn() {
        let mut gp = grid.clone();
        gp[i] = (grid[i] as f64 + eps) as f32;
        let lp = gs_loss(&s, &x, &gp, &dy);
        let mut gm = grid.clone();
        gm[i] = (grid[i] as f64 - eps) as f32;
        let lm = gs_loss(&s, &x, &gm, &dy);
        // The perturbation actually applied, after the f32 round trip.
        let h = (gp[i] as f64) - (gm[i] as f64);
        let num = (lp - lm) / h;
        let a = ana[i] as f64;
        assert!(
            (num - a).abs() < 1e-3 + 2e-3 * num.abs().max(a.abs()),
            "dgrid[{i}] (align={}) num={num} ana={a}",
            s.align
        );
    }
}

#[test]
fn dgrid_matches_finite_differences_half_pixel() {
    if skip() {
        return;
    }
    assert_dgrid_fd(Shape { n: 2, c: 3, h: 6, w: 7, ho: 3, wo: 3, align: false }, 41);
}

#[test]
fn dgrid_matches_finite_differences_align_corners() {
    if skip() {
        return;
    }
    assert_dgrid_fd(Shape { n: 2, c: 3, h: 6, w: 7, ho: 3, wo: 3, align: true }, 51);
}

/// FD can only be run where the function is differentiable, which excludes every
/// sample that straddles the border. The oracle has no such restriction, so run
/// it on the wide grid too.
#[test]
fn dgrid_matches_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for align in [false, true] {
        let s = Shape { n: 2, c: 3, h: 5, w: 6, ho: 4, wo: 4, align };
        let x = randvec(61, s.xn());
        let grid = randgrid(62, s.gn());
        let dy = randvec(63, s.yn());
        let got = run_dgrid(&gpu, &s, &dy, &x, &grid);
        let want = gs_dgrid_ref(&s, &x, &grid, &dy);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-4, "grid_sample_dgrid (align={align}) vs reference: max |diff| = {d:.3e}");
    }
}

// ---- CPU oracle -------------------------------------------------------------
//
// Re-derived in f64 straight from the PyTorch definition (AGENTS.md permits a
// gradcheck oracle to duplicate the math — one that shares code with the thing
// it checks proves nothing). Matches `wgsl/grid_sample.wgsl`,
// `wgsl/grid_sample_dx.wgsl` and `wgsl/grid_sample_dgrid.wgsl`.

/// `grid_sampler_unnormalize`: normalized [-1,1] -> input pixel coordinate.
fn unnormalize(g: f64, size: usize, align: bool) -> f64 {
    if align {
        ((g + 1.0) * 0.5) * (size as f64 - 1.0)
    } else {
        ((g + 1.0) * (size as f64) - 1.0) * 0.5
    }
}

/// d(pixel coordinate)/d(normalized coordinate).
fn unnormalize_slope(size: usize, align: bool) -> f64 {
    if align {
        0.5 * (size as f64 - 1.0)
    } else {
        0.5 * (size as f64)
    }
}

/// The four bilinear taps of (ix, iy), with the out-of-bounds ones DROPPED
/// (padding_mode='zeros'). Returns (row, col, weight).
fn taps(ix: f64, iy: f64, h: usize, w: usize) -> Vec<(usize, usize, f64)> {
    let x0f = ix.floor();
    let y0f = iy.floor();
    let fx = ix - x0f;
    let fy = iy - y0f;
    let x0 = x0f as i64;
    let y0 = y0f as i64;
    let cand = [
        (y0, x0, (1.0 - fx) * (1.0 - fy)),
        (y0, x0 + 1, fx * (1.0 - fy)),
        (y0 + 1, x0, (1.0 - fx) * fy),
        (y0 + 1, x0 + 1, fx * fy),
    ];
    cand.iter()
        .filter(|&&(r, c, _)| r >= 0 && (r as usize) < h && c >= 0 && (c as usize) < w)
        .map(|&(r, c, wt)| (r as usize, c as usize, wt))
        .collect()
}

fn gs_ref(s: &Shape, x: &[f32], grid: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; s.yn()];
    for n in 0..s.n {
        for ho in 0..s.ho {
            for wo in 0..s.wo {
                let gi = ((n * s.ho + ho) * s.wo + wo) * 2;
                let ix = unnormalize(grid[gi] as f64, s.w, s.align);
                let iy = unnormalize(grid[gi + 1] as f64, s.h, s.align);
                let t = taps(ix, iy, s.h, s.w);
                for c in 0..s.c {
                    let plane = (n * s.c + c) * s.h * s.w;
                    let mut acc = 0f64;
                    for &(r, col, wt) in &t {
                        acc += x[plane + r * s.w + col] as f64 * wt;
                    }
                    y[((n * s.c + c) * s.ho + ho) * s.wo + wo] = acc as f32;
                }
            }
        }
    }
    y
}

/// `L(grid) = <grid_sample(x, grid), dy>` accumulated entirely in f64, so the FD
/// difference is not swamped by the f32 rounding of an intermediate `y`.
fn gs_loss(s: &Shape, x: &[f32], grid: &[f32], dy: &[f32]) -> f64 {
    let mut l = 0f64;
    for n in 0..s.n {
        for ho in 0..s.ho {
            for wo in 0..s.wo {
                let gi = ((n * s.ho + ho) * s.wo + wo) * 2;
                let ix = unnormalize(grid[gi] as f64, s.w, s.align);
                let iy = unnormalize(grid[gi + 1] as f64, s.h, s.align);
                let t = taps(ix, iy, s.h, s.w);
                for c in 0..s.c {
                    let plane = (n * s.c + c) * s.h * s.w;
                    let mut acc = 0f64;
                    for &(r, col, wt) in &t {
                        acc += x[plane + r * s.w + col] as f64 * wt;
                    }
                    l += acc * dy[((n * s.c + c) * s.ho + ho) * s.wo + wo] as f64;
                }
            }
        }
    }
    l
}

/// The honest CPU SCATTER the GPU kernel has to reproduce by gathering.
fn gs_dx_ref(s: &Shape, grid: &[f32], dy: &[f32]) -> Vec<f32> {
    let mut dx = vec![0f64; s.xn()];
    for n in 0..s.n {
        for ho in 0..s.ho {
            for wo in 0..s.wo {
                let gi = ((n * s.ho + ho) * s.wo + wo) * 2;
                let ix = unnormalize(grid[gi] as f64, s.w, s.align);
                let iy = unnormalize(grid[gi + 1] as f64, s.h, s.align);
                let t = taps(ix, iy, s.h, s.w);
                for c in 0..s.c {
                    let plane = (n * s.c + c) * s.h * s.w;
                    let go = dy[((n * s.c + c) * s.ho + ho) * s.wo + wo] as f64;
                    for &(r, col, wt) in &t {
                        dx[plane + r * s.w + col] += go * wt;
                    }
                }
            }
        }
    }
    dx.iter().map(|&v| v as f32).collect()
}

fn gs_dgrid_ref(s: &Shape, x: &[f32], grid: &[f32], dy: &[f32]) -> Vec<f32> {
    let mx = unnormalize_slope(s.w, s.align);
    let my = unnormalize_slope(s.h, s.align);
    let mut dg = vec![0f32; s.gn()];
    for n in 0..s.n {
        for ho in 0..s.ho {
            for wo in 0..s.wo {
                let gi = ((n * s.ho + ho) * s.wo + wo) * 2;
                let ix = unnormalize(grid[gi] as f64, s.w, s.align);
                let iy = unnormalize(grid[gi + 1] as f64, s.h, s.align);
                let x0f = ix.floor();
                let y0f = iy.floor();
                let fx = ix - x0f;
                let fy = iy - y0f;
                let x0 = x0f as i64;
                let y0 = y0f as i64;
                // d(weight)/d(ix) and d(weight)/d(iy) per corner.
                let cand = [
                    (y0, x0, -(1.0 - fy), -(1.0 - fx)),
                    (y0, x0 + 1, 1.0 - fy, -fx),
                    (y0 + 1, x0, -fy, 1.0 - fx),
                    (y0 + 1, x0 + 1, fy, fx),
                ];
                let mut gix = 0f64;
                let mut giy = 0f64;
                for c in 0..s.c {
                    let plane = (n * s.c + c) * s.h * s.w;
                    let go = dy[((n * s.c + c) * s.ho + ho) * s.wo + wo] as f64;
                    for &(r, col, dwdx, dwdy) in &cand {
                        if r < 0 || r as usize >= s.h || col < 0 || col as usize >= s.w {
                            continue;
                        }
                        let v = x[plane + (r as usize) * s.w + col as usize] as f64;
                        gix += v * dwdx * go;
                        giy += v * dwdy * go;
                    }
                }
                dg[gi] = (gix * mx) as f32;
                dg[gi + 1] = (giy * my) as f32;
            }
        }
    }
    dg
}
