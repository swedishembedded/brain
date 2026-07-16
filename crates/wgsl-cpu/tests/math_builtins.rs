// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Numeric gate for the math intrinsics the JIT lowers.
//!
//! `compile_all.rs` proves every registered kernel *translates*; this proves the
//! translation *computes the right thing*. It matters most for the rounding family
//! (`floor`/`ceil`/`trunc`/`round`/`fract`) and `sign`, whose edge cases (negative
//! values, exact halves, -0.0) are exactly where a plausible-looking lowering is
//! silently wrong — a bug no shape/compile test can see.
//!
//! Each case JITs a one-line kernel `y[i] = <expr>(x[i], ...)` and compares against
//! the same expression evaluated by Rust's std, which is IEEE-754 and therefore the
//! same answer WGSL's spec requires.

use wgsl_cpu::Jit;

/// A minimal one-output-per-invocation kernel wrapping a scalar expression in `v`.
/// Shaped exactly like the real kernels (cf. `kernels/wgsl/silu.wgsl`): the
/// 2D-grid-safe linear index, and the `num_workgroups` arg the JIT requires.
fn kernel(expr: &str) -> String {
    format!(
        r#"
struct Params {{ n: u32 }};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) {{ return; }}
    let v = x[i];
    y[i] = {expr};
}}
"#
    )
}

/// JIT `expr` and run it over `xs`.
fn run(expr: &str, xs: &[f32]) -> Vec<f32> {
    let src = kernel(expr);
    let jit = Jit::new(&[("k", &src)]).unwrap_or_else(|e| panic!("JIT failed for `{expr}`: {e}"));
    let n = xs.len();
    let mut ys = vec![0f32; n];
    let uni = [n as u32];
    let bufs: Vec<*mut u8> = vec![xs.as_ptr() as *mut u8, ys.as_mut_ptr() as *mut u8];
    let gx = (n as u32).div_ceil(64).max(1);
    // SAFETY: one output element per invocation, single-threaded here; `uni`/`bufs`
    // outlive the call and match the kernel's binding order (storage bindings only).
    unsafe { jit.run(0, 0, n as u64, gx, 1, uni.as_ptr(), bufs.as_ptr()) };
    ys
}

/// The inputs that actually discriminate: both signs, exact halves (the tie-break),
/// exact integers, and zero.
const XS: &[f32] = &[
    -2.5, -2.0, -1.75, -1.5, -1.25, -0.5, -0.0, 0.0, 0.5, 1.25, 1.5, 1.75, 2.0, 2.5, 3.5, 7.25,
];

fn assert_exact(expr: &str, got: &[f32], want: impl Fn(f32) -> f32) {
    for (i, (&x, &g)) in XS.iter().zip(got).enumerate() {
        let w = want(x);
        assert!(
            g == w || (g.is_nan() && w.is_nan()),
            "{expr} at [{i}] x={x}: got {g}, want {w}"
        );
    }
}

#[test]
fn floor_ceil_trunc_match_ieee() {
    assert_exact("floor(v)", &run("floor(v)", XS), f32::floor);
    assert_exact("ceil(v)", &run("ceil(v)", XS), f32::ceil);
    assert_exact("trunc(v)", &run("trunc(v)", XS), f32::trunc);
}

/// WGSL `round` is round-half-to-EVEN, not round-half-away-from-zero. Rust's
/// `f32::round` is half-away, so it is the WRONG oracle here — `round_ties_even`
/// is the right one. This test exists precisely because `floor(x+0.5)` and
/// `f32::round` both look correct until you feed them an exact .5.
#[test]
fn round_is_ties_to_even() {
    let got = run("round(v)", XS);
    assert_exact("round(v)", &got, |x| x.round_ties_even());

    // Pin the tie-break explicitly so the intent survives a refactor.
    let ties = [-2.5f32, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
    let got = run("round(v)", &ties);
    assert_eq!(got, vec![-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0], "round must break ties to even");
    // The discriminating cases: .5 values whose two rules DISAGREE. 1.5 and 3.5 go
    // to 2 and 4 under both rules, so they prove nothing; 0.5 and 2.5 are the test.
    assert_eq!(got[3], 0.0, "0.5 -> 0 under ties-to-even, 1 under half-away");
    assert_eq!(got[5], 2.0, "2.5 -> 2 under ties-to-even, 3 under half-away");
    assert_eq!(got[0], -2.0, "-2.5 -> -2 under ties-to-even, -3 under half-away");
    // f32::round IS the half-away rule, so it must disagree on exactly those.
    assert_ne!(got[5], 2.5f32.round(), "f32::round is half-away (3.0) and must differ");
}

#[test]
fn fract_matches_x_minus_floor() {
    assert_exact("fract(v)", &run("fract(v)", XS), |x| x - x.floor());
}

#[test]
fn clamp_saturate_mix() {
    assert_exact("clamp(v, -1.0, 2.0)", &run("clamp(v, -1.0, 2.0)", XS), |x| {
        x.max(-1.0).min(2.0)
    });
    assert_exact("saturate(v)", &run("saturate(v)", XS), |x| x.max(0.0).min(1.0));
    // Spec form: e1*(1-e3) + e2*e3.
    assert_exact("mix(v, 10.0, 0.25)", &run("mix(v, 10.0, 0.25)", XS), |x| {
        x * 0.75 + 10.0 * 0.25
    });
}

#[test]
fn sign_is_minus_one_zero_plus_one() {
    // NOT f32::signum: signum(-0.0) == -1.0, but WGSL sign(-0.0) == 0.0 (it is
    // neither < 0 nor > 0). That divergence is the whole reason this is asserted.
    assert_exact("sign(v)", &run("sign(v)", XS), |x| {
        if x < 0.0 {
            -1.0
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        }
    });
    let z = run("sign(v)", &[-0.0, 0.0]);
    assert_eq!(z[0], 0.0, "sign(-0.0) must be 0.0, not -1.0 (f32::signum would say -1.0)");
    assert_eq!(z[1], 0.0);
}

/// The reason 0.1 exists: bilinear resize needs floor + fract + clamp together, and
/// this is the exact expression shape it will use. Pins that the combination lowers.
#[test]
fn bilinear_weight_expression_lowers_and_computes() {
    // src coordinate -> integer tap + fractional weight, clamped to a 4-wide axis.
    let expr = "clamp(floor(v), 0.0, 3.0) + fract(v) * 0.0 + mix(0.0, 1.0, fract(v))";
    let xs: &[f32] = &[-1.0, 0.0, 0.25, 1.5, 2.75, 3.0, 9.0];
    let got = run(expr, xs);
    for (i, &x) in xs.iter().enumerate() {
        let want = x.floor().clamp(0.0, 3.0) + (x - x.floor());
        assert!(
            (got[i] - want).abs() < 1e-6,
            "bilinear expr at x={x}: got {}, want {want}",
            got[i]
        );
    }
}
