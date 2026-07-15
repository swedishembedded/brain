// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1.gn — GroupNorm (NCHW) kernel-family tests, written FROM the spec
//! (`docs/world-models/specs/P1.gn.md` §7 hand-computed reference, §9 edge
//! cases, §10 required tests), never from the implementation.
//!
//! Gating runs are `BRAIN_DEVICE=cpu` (+ `MOE_SKIP_GPU_TESTS=1`), both set by
//! `scripts/wm-locked-make.sh`. The FD/gradcheck entry for the backward ops
//! lives in `tests/gn_fd.rs` (mse_fd.rs pattern). All test fn names start
//! with `gn_` so `cargo test gn_` selects exactly this unit's tests.

use gpu_core::Gpu;
use wm_core::gn::{num_groups, Gn, GnDims};

// ---------------------------------------------------------------- harness --

fn gpu() -> Gpu {
    Gpu::new(&Gn::kernel_sources())
}

/// Deterministic LCG in [-1, 1). Spec §10.3 requires seeded data in [−1,1];
/// mse_fd.rs's `>> 33` variant keeps only 31 bits and thus lands in [−1,0)
/// (its `~[-1,1)` comment is wrong) — copying it here lost all sign coverage,
/// so this takes 32 bits before the 2^31 scale (round-2 adversary fix).
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 32) as f32 / (1u64 << 31) as f32) - 1.0
}

struct Fwd {
    stats: Vec<f32>, // [2*N*G] mean|rstd
    y: Vec<f32>,     // [N*C*H*W]
}

struct Bwd {
    dyg: Vec<f32>,  // [N*C*H*W]
    sums: Vec<f32>, // [4*N*G] mean|rstd|S1|S2
    dx: Vec<f32>,   // [N*C*H*W]
    dgb: Vec<f32>,  // [2C] gamma-grads || beta-grads
}

/// Forward only: submit [gn_stats, gn_apply], read back stats and y.
fn run_forward(gpu: &Gpu, gn: &Gn, d: &GnDims, x: &[f32], gb: &[f32]) -> Fwd {
    assert_eq!(x.len(), d.elems() as usize, "test harness: x length");
    assert_eq!(gb.len(), 2 * d.c as usize, "test harness: gb length");
    let xb = gpu.storage_init("x", x);
    let gbb = gpu.storage_init("gb", gb);
    let stats = gpu.storage(d.stats_len());
    let y = gpu.storage(d.elems() as u64);
    let steps = gn.forward(gpu, d, &xb, &gbb, &stats, &y);
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    Fwd {
        stats: gpu.read(&stats, d.stats_len() as usize),
        y: gpu.read(&y, d.elems() as usize),
    }
}

/// Forward then backward (spec §6 order), fresh buffers, dgb pre-zeroed.
fn run_fwd_bwd(gpu: &Gpu, gn: &Gn, d: &GnDims, x: &[f32], gb: &[f32], dy: &[f32]) -> (Fwd, Bwd) {
    assert_eq!(dy.len(), d.elems() as usize, "test harness: dy length");
    let n_el = d.elems() as usize;
    let xb = gpu.storage_init("x", x);
    let gbb = gpu.storage_init("gb", gb);
    let stats = gpu.storage(d.stats_len());
    let y = gpu.storage(d.elems() as u64);
    let fwd_steps = gn.forward(gpu, d, &xb, &gbb, &stats, &y);
    gpu.submit(&[], &fwd_steps);
    gpu.poll_wait();

    let dyb = gpu.storage_init("dy", dy);
    let dyg = gpu.storage(d.elems() as u64);
    let sums = gpu.storage(d.sums_len());
    let dx = gpu.storage(d.elems() as u64);
    // dgb is ACCUMULATED by the kernels; the caller pre-zeroes it (spec §2).
    let dgb = gpu.storage_init("dgb", &vec![0.0f32; 2 * d.c as usize]);
    let bwd_steps = gn.backward(gpu, d, &xb, &gbb, &stats, &dyb, &dyg, &sums, &dx, &dgb);
    gpu.submit(&[], &bwd_steps);
    gpu.poll_wait();

    (
        Fwd {
            stats: gpu.read(&stats, d.stats_len() as usize),
            y: gpu.read(&y, n_el),
        },
        Bwd {
            dyg: gpu.read(&dyg, n_el),
            sums: gpu.read(&sums, d.sums_len() as usize),
            dx: gpu.read(&dx, n_el),
            dgb: gpu.read(&dgb, 2 * d.c as usize),
        },
    )
}

fn assert_close(got: &[f32], want: &[f32], atol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} != {}", got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "{what}[{i}]: got {g}, want {w} (|diff| {} > atol {atol})",
            (g - w).abs()
        );
    }
}

// -------------------------------------------- spec §7 hand-computed fixture --
// N=1, C=4, G=2, H=W=2, eps=1.0 (eps deliberately NOT 1e-5: proves eps is
// read from params). Flat index = c*4 + h*2 + w.

const D7: (u32, u32, u32, u32, u32, f32) = (1, 4, 2, 2, 2, 1.0);

const X7: [f32; 16] = [
    1.0, 2.0, 3.0, 4.0, // c0
    5.0, 6.0, 7.0, 8.0, // c1
    -1.0, 1.0, 1.0, 1.0, // c2
    3.0, 3.0, 3.0, 5.0, // c3
];
// gb = [gamma || beta] (CONCATENATED layout, spec §2).
const GB7: [f32; 8] = [2.0, 0.5, 1.0, -1.0, 0.5, -1.0, 0.0, 2.0];

const STATS7: [f32; 4] = [4.5, 0.4, 2.0, 0.5]; // mean0, rstd0, mean1, rstd1

const Y7: [f32; 16] = [
    -2.3, -1.5, -0.7, 0.1, // c0
    -0.9, -0.7, -0.5, -0.3, // c1
    -1.5, -0.5, -0.5, -0.5, // c2
    1.5, 1.5, 1.5, 0.5, // c3
];

const DY7: [f32; 16] = [
    1.0, -1.0, 2.0, 0.0, // c0
    0.0, 1.0, -1.0, 1.0, // c1
    1.0, 1.0, -1.0, -1.0, // c2
    2.0, 0.0, 0.0, -2.0, // c3
];

const DYG7: [f32; 16] = [
    2.0, -2.0, 4.0, 0.0, // c0 = 2.0*dy
    0.0, 0.5, -0.5, 0.5, // c1 = 0.5*dy
    1.0, 1.0, -1.0, -1.0, // c2 = 1.0*dy
    -2.0, 0.0, 0.0, 2.0, // c3 = -1.0*dy
];

// sums = [mean, rstd, S1, S2] per (n,g).
const SUMS7: [f32; 8] = [4.5, 0.4, 4.5, -2.7, 2.0, 0.5, 0.0, 1.0];

// dgb = [dgamma || dbeta].
const DGB7: [f32; 8] = [-1.6, 1.0, -1.0, -2.0, 2.0, 1.0, 0.0, 0.0];

const DX7: [f32; 16] = [
    0.386, -1.16, 1.294, -0.252, // c0
    -0.198, 0.056, -0.29, 0.164, // c1
    0.59375, 0.53125, -0.46875, -0.46875, // c2
    -1.03125, -0.03125, -0.03125, 0.90625, // c3
];

fn d7() -> GnDims {
    let (n, c, h, w, g, eps) = D7;
    GnDims::new(n, c, h, w, g, eps).expect("spec §7 dims are valid")
}

// ------------------------------------------------------------------- tests --

/// Spec §10.1: §7 input, eps = 1.0 — stats and all 16 y values, atol 1e-5.
///
/// The forward is dispatched TWICE over the SAME stats/y buffers: spec
/// §4.1/§4.2 write with `=` (`stats[2k] = …`, `y[idx] = …`), so pass 2 must
/// reproduce identical values. An accumulating (`+=`) gn_stats or gn_apply
/// mutant survives fresh zero-initialized buffers everywhere else in this
/// suite and doubles here (round-2 adversary mutant).
#[test]
fn gn_forward_matches_hand_reference() {
    let gpu = gpu();
    let gn = Gn::seq();
    let d = d7();
    let xb = gpu.storage_init("x", &X7);
    let gbb = gpu.storage_init("gb", &GB7);
    let stats = gpu.storage(d.stats_len());
    let y = gpu.storage(d.elems() as u64);
    for pass in 1..=2u32 {
        let steps = gn.forward(&gpu, &d, &xb, &gbb, &stats, &y);
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        assert_close(
            &gpu.read(&stats, d.stats_len() as usize),
            &STATS7,
            1e-5,
            &format!("stats (overwrite, pass {pass})"),
        );
        assert_close(
            &gpu.read(&y, d.elems() as usize),
            &Y7,
            1e-5,
            &format!("y (overwrite, pass {pass})"),
        );
    }
}

/// Spec §10.2: §7 upstream dy — dyg, sums (incl. copied-through mean/rstd),
/// dgb (both halves, pre-zeroed buffer), all 16 dx values, atol 1e-5; plus
/// the group-sum invariant |Σ_Ω dx| ≤ 1e-5 for every group.
#[test]
fn gn_backward_matches_hand_reference() {
    let gpu = gpu();
    let gn = Gn::seq();
    let d = d7();
    let (_, b) = run_fwd_bwd(&gpu, &gn, &d, &X7, &GB7, &DY7);
    assert_close(&b.dyg, &DYG7, 1e-5, "dyg");
    assert_close(&b.sums, &SUMS7, 1e-5, "sums");
    assert_close(&b.dgb, &DGB7, 1e-5, "dgb");
    assert_close(&b.dx, &DX7, 1e-5, "dx");

    // Invariant (spec §3): Σ_Ω(k) dx = 0 per group, any eps.
    // (Do NOT assert Σ dx*xhat = 0 — that holds only at eps = 0.)
    let (hw, cpg) = ((d.h * d.w) as usize, (d.c / d.g) as usize);
    for n in 0..d.n as usize {
        for g in 0..d.g as usize {
            let mut s = 0f32;
            for c in g * cpg..(g + 1) * cpg {
                for i in 0..hw {
                    s += b.dx[(n * d.c as usize + c) * hw + i];
                }
            }
            assert!(s.abs() <= 1e-5, "group (n={n},g={g}): |Σ dx| = {} > 1e-5", s.abs());
        }
    }
}

/// Spec §2 + §4.3/§4.4: `dgb` is ACCUMULATED (`+=`) into whatever the buffer
/// already holds, so grads compose with grad-accumulation; spec §4.6: `gn_dx`
/// OVERWRITES `dx` (not accumulate). Neither is distinguishable from the
/// wrong semantics when `dgb` is pre-zeroed and `dx` is fresh (as in the §10.2
/// test), so this test seeds both buffers:
///   - `dgb` starts at a known nonzero vector; one backward must land on
///     seed + dgb_ref, a second backward on seed + 2·dgb_ref;
///   - `dx` starts at a 7.5 sentinel; each backward must leave exactly the
///     §7.3 reference (an accumulating mutant reports 7.5 + dx or 2·dx).
/// The second pass also pins gn_dsum's overwrite semantics (spec §4.5 `=`):
/// an accumulating gn_dsum doubles S1/S2 and corrupts pass-2 dx.
#[test]
fn gn_dgb_accumulates_dx_overwrites() {
    let gpu = gpu();
    let gn = Gn::seq();
    let d = d7();
    let n_el = d.elems() as usize;

    let xb = gpu.storage_init("x", &X7);
    let gbb = gpu.storage_init("gb", &GB7);
    let stats = gpu.storage(d.stats_len());
    let y = gpu.storage(d.elems() as u64);
    let fwd = gn.forward(&gpu, &d, &xb, &gbb, &stats, &y);
    gpu.submit(&[], &fwd);
    gpu.poll_wait();

    // Exact binary fractions, so seed + k·ref stays within the 1e-5 atol.
    const DGB_SEED: [f32; 8] = [0.5, -0.25, 1.0, 2.0, -1.5, 0.75, -0.125, 3.0];
    let dyb = gpu.storage_init("dy", &DY7);
    let dyg = gpu.storage(d.elems() as u64);
    let sums = gpu.storage(d.sums_len());
    let dx = gpu.storage_init("dx", &vec![7.5f32; n_el]); // sentinel: must vanish
    let dgb = gpu.storage_init("dgb", &DGB_SEED);

    for pass in 1..=2u32 {
        let bwd = gn.backward(&gpu, &d, &xb, &gbb, &stats, &dyb, &dyg, &sums, &dx, &dgb);
        gpu.submit(&[], &bwd);
        gpu.poll_wait();

        let want_dgb: Vec<f32> = DGB_SEED
            .iter()
            .zip(DGB7.iter())
            .map(|(&s, &g)| s + pass as f32 * g)
            .collect();
        assert_close(
            &gpu.read(&dgb, 2 * d.c as usize),
            &want_dgb,
            1e-5,
            &format!("dgb (accumulate, pass {pass})"),
        );
        assert_close(
            &gpu.read(&dx, n_el),
            &DX7,
            1e-5,
            &format!("dx (overwrite, pass {pass})"),
        );
    }
}

/// Spec §10.4: eps enters INSIDE the sqrt, added to var, and is read from
/// params. N=1,C=1,G=1,H=W=2, x=[0,1,1,2], eps=0.5, gamma=3, beta=1:
/// mean=1, var=0.5 → rstd = 1/sqrt(0.5+0.5) = 1.0 EXACTLY and
/// y = [-2,1,1,4]. Misplacements land far away (1.914 / 0.828).
#[test]
fn gn_eps_placement() {
    let gpu = gpu();
    let gn = Gn::seq();
    let d = GnDims::new(1, 1, 2, 2, 1, 0.5).expect("valid dims");
    let x = [0.0f32, 1.0, 1.0, 2.0];
    let gb = [3.0f32, 1.0]; // gamma=3 || beta=1
    let f = run_forward(&gpu, &gn, &d, &x, &gb);
    assert!((f.stats[0] - 1.0).abs() <= 1e-6, "mean: got {}, want 1.0", f.stats[0]);
    assert_eq!(f.stats[1], 1.0, "rstd must be exactly 1.0 = 1/sqrt(var+eps); got {} (1.914 => eps added outside sqrt, 0.828 => eps added to sqrt(var))", f.stats[1]);
    assert_close(&f.y, &[-2.0, 1.0, 1.0, 4.0], 1e-6, "y");
}

/// Spec §10.5 property: G = C is InstanceNorm — with gamma=1, beta=0 the
/// output equals host per-(n,c) standardization over (h,w), atol 1e-5.
#[test]
fn gn_g_equals_c_is_instance_norm() {
    let (n, c, h, w) = (2u32, 3u32, 2u32, 2u32);
    let eps = 1e-2f32;
    let d = GnDims::new(n, c, h, w, c, eps).expect("valid dims");
    let mut seed = 0x0A11_5EEDu64;
    let x: Vec<f32> = (0..d.elems()).map(|_| lcg(&mut seed)).collect();
    let mut gb = vec![1.0f32; c as usize]; // gamma = 1
    gb.extend(std::iter::repeat(0.0f32).take(c as usize)); // beta = 0

    let gpu = gpu();
    let gn = Gn::seq();
    let f = run_forward(&gpu, &gn, &d, &x, &gb);

    // Host reference: per-(n,c) biased standardization over H*W.
    let hw = (h * w) as usize;
    let mut want = vec![0f32; x.len()];
    for nc in 0..(n * c) as usize {
        let sl = &x[nc * hw..(nc + 1) * hw];
        let mean = sl.iter().map(|&v| v as f64).sum::<f64>() / hw as f64;
        let var = sl.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / hw as f64;
        let rstd = 1.0 / (var + eps as f64).sqrt();
        for i in 0..hw {
            want[nc * hw + i] = ((sl[i] as f64 - mean) * rstd) as f32;
        }
    }
    assert_close(&f.y, &want, 1e-5, "instance-norm y");
}

/// Spec §10.6: fixed inputs, forward+backward run twice on FRESH buffers —
/// every output (stats, y, dyg, sums, dx, dgb) must be BITWISE equal.
#[test]
fn gn_deterministic_bitwise() {
    let d = GnDims::new(2, 4, 3, 2, 2, 1e-5).expect("valid dims");
    let mut seed = 0xD37E_2814_57A7_E5EEu64;
    let x: Vec<f32> = (0..d.elems()).map(|_| lcg(&mut seed)).collect();
    let mut gb: Vec<f32> = (0..d.c).map(|_| 1.0 + 0.5 * lcg(&mut seed)).collect(); // gamma ~[0.5,1.5)
    gb.extend((0..d.c).map(|_| lcg(&mut seed))); // beta
    let dy: Vec<f32> = (0..d.elems()).map(|_| lcg(&mut seed)).collect();

    let gpu = gpu();
    let gn = Gn::seq();
    let (f1, b1) = run_fwd_bwd(&gpu, &gn, &d, &x, &gb, &dy);
    let (f2, b2) = run_fwd_bwd(&gpu, &gn, &d, &x, &gb, &dy);

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&f1.stats), bits(&f2.stats), "stats not bitwise-deterministic");
    assert_eq!(bits(&f1.y), bits(&f2.y), "y not bitwise-deterministic");
    assert_eq!(bits(&b1.dyg), bits(&b2.dyg), "dyg not bitwise-deterministic");
    assert_eq!(bits(&b1.sums), bits(&b2.sums), "sums not bitwise-deterministic");
    assert_eq!(bits(&b1.dx), bits(&b2.dx), "dx not bitwise-deterministic");
    assert_eq!(bits(&b1.dgb), bits(&b2.dgb), "dgb not bitwise-deterministic");
}

/// Spec §9 edge: M = 1 (G=C, H=W=1) — var = 0, rstd = 1/sqrt(eps), xhat = 0,
/// y = beta broadcast, dx = 0, dgamma = 0, dbeta = dy. No NaN/inf.
#[test]
fn gn_m1_degenerate_group_y_is_beta_dx_zero() {
    let d = GnDims::new(1, 3, 1, 1, 3, 0.25).expect("valid dims");
    let x = [0.5f32, -2.0, 3.0];
    let gb = [2.0f32, -1.0, 0.5, 0.25, -0.5, 1.0]; // gamma || beta
    let dy = [1.0f32, -2.0, 0.5];

    let gpu = gpu();
    let gn = Gn::seq();
    let (f, b) = run_fwd_bwd(&gpu, &gn, &d, &x, &gb, &dy);

    for (i, v) in f.stats.iter().chain(&f.y).chain(&b.sums).chain(&b.dx).chain(&b.dgb).enumerate() {
        assert!(v.is_finite(), "output value #{i} not finite: {v}");
    }
    // mean_k = x_k, rstd = 1/sqrt(0 + 0.25) = 2.0 exactly.
    assert_close(&f.stats, &[0.5, 2.0, -2.0, 2.0, 3.0, 2.0], 1e-6, "stats");
    assert_close(&f.y, &[0.25, -0.5, 1.0], 1e-6, "y (= beta broadcast)");
    assert_close(&b.dx, &[0.0, 0.0, 0.0], 1e-6, "dx (= 0 when M=1)");
    // dgamma = Σ dy·xhat = 0 (xhat = 0); dbeta = dy.
    assert_close(&b.dgb, &[0.0, 0.0, 0.0, 1.0, -2.0, 0.5], 1e-6, "dgb");
}

/// Spec §10.7 + §9: GnDims::new rejections and the num_groups convention.
#[test]
fn gn_dims_validate_rejects() {
    // Non-divisible C % G.
    assert!(GnDims::new(1, 5, 2, 2, 2, 1e-5).is_err(), "c=5,g=2 must be rejected");
    assert!(GnDims::new(1, 6, 2, 2, 4, 1e-5).is_err(), "c=6,g=4 must be rejected");
    assert!(GnDims::new(1, 4, 2, 2, 8, 1e-5).is_err(), "g=8 > c=4 (c%g=4) must be rejected");
    // Zero dims.
    assert!(GnDims::new(0, 4, 2, 2, 2, 1e-5).is_err(), "n=0 must be rejected");
    assert!(GnDims::new(1, 0, 2, 2, 2, 1e-5).is_err(), "c=0 must be rejected");
    assert!(GnDims::new(1, 4, 0, 2, 2, 1e-5).is_err(), "h=0 must be rejected");
    assert!(GnDims::new(1, 4, 2, 0, 2, 1e-5).is_err(), "w=0 must be rejected");
    assert!(GnDims::new(1, 4, 2, 2, 0, 1e-5).is_err(), "g=0 must be rejected");
    // eps must be strictly positive.
    assert!(GnDims::new(1, 4, 2, 2, 2, 0.0).is_err(), "eps=0 must be rejected");
    assert!(GnDims::new(1, 4, 2, 2, 2, -1.0).is_err(), "eps<0 must be rejected");

    // Valid dims: accessors per spec §8.
    let d = GnDims::new(2, 4, 3, 2, 2, 1e-5).expect("valid dims must be accepted");
    assert_eq!(d.elems(), 48, "elems = n*c*h*w");
    assert_eq!(d.groups(), 4, "groups = n*g");
    assert_eq!(d.stats_len(), 8, "stats_len = 2*n*g");
    assert_eq!(d.sums_len(), 16, "sums_len = 4*n*g");
    // G=1 (LayerNorm-over-(C,H,W)) and G=C (InstanceNorm) are valid.
    assert!(GnDims::new(1, 4, 2, 2, 1, 1e-5).is_ok(), "g=1 must be accepted");
    assert!(GnDims::new(1, 4, 2, 2, 4, 1e-5).is_ok(), "g=c must be accepted");

    // num_groups = max(1, c/32), u32 division (spec §9).
    for c in 1..=63u32 {
        assert_eq!(num_groups(c), 1, "num_groups({c})");
    }
    assert_eq!(num_groups(64), 2, "num_groups(64)");
    assert_eq!(num_groups(95), 2, "num_groups(95)");
    assert_eq!(num_groups(96), 3, "num_groups(96)");
}
