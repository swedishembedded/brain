// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pins the `compile_one_wg` fix (the Cranelift-JIT
//! bug where a per-invocation local written before the barrier and read back
//! by a later top-level statement of the SAME segment silently read its
//! zero-initialised value instead) directly, with tiny synthetic fixtures
//! rather than nine per-kernel parity tests.
//!
//! Why one generic compiler test instead of covering each of the 9 real
//! kernels that share the at-risk shape (`max_abs_rows`, `rmsnorm_rows`,
//! `gradnorm_part`, `clip_coef_wg`, `prelu_bwd_wg`, `paged_decode_scores_wg`,
//! `layernorm_rows`, `ln_stats_rows`, `layernorm_dx_rows`): the defect is three
//! lines in one function (`compile_one_wg`'s `carried` handling), so pinning it
//! once here is cheaper and more sensitive than nine parity tests would be —
//! and this one runs unconditionally (direct `Jit::new`/`jit.run`, no device,
//! no `BRAIN_DEVICE`, immune to the `workgroup_reductions` caps-gating that
//! silently drops 8 of those 9 kernels from every model-level CPU test today).
//!
//! Each fixture is dispatched over >=3 workgroups so the ORIGINAL symptom
//! (every workgroup after the first collapsing to workgroup 0's answer)
//! reproduces end to end, not just a single-workgroup smoke case.
//!
//! Mutation-verify: temporarily revert `crates/wgsl-cpu/src/lib.rs`'s
//! `for (seg, c) in [(&seg_before, &no_carried), (&seg_after, &carried)]` back
//! to feeding `carried` to both segments (the pre-fix shape) — every F1-F5
//! fixture below must fail. That is what turns "F1-F6 cover the 9 at-risk
//! kernels' shape" from inference (reading WGSL and reasoning about how naga
//! forms `Statement::Emit` ranges) into a measurement.

use wgsl_cpu::Jit;

/// Deterministic, integer-valued pseudo-random floats in [-125, 125] — small
/// enough that sums over a few dozen of them stay exact in f32 regardless of
/// summation order (well under the 2^24 exact-integer range), so every
/// fixture below can assert bit-for-bit rather than within a tolerance.
fn val(i: usize) -> f32 {
    ((i * 37 + 11) % 251) as f32 - 125.0
}

/// Run a single-kernel `Jit` over `m` workgroups (rows) of 64 lanes each, with
/// `x: [m*k]` bound at binding 1 and up to two read_write outputs at bindings
/// 2 (and 3, if `out2_len` is `Some`), each `m` long. Returns the output
/// buffer(s).
fn run_wg_kernel(src: &str, m: u32, k: u32, out2: bool) -> (Vec<f32>, Vec<f32>) {
    let jit = Jit::new(&[("fixture", src)]).expect("fixture must compile");
    let mut x: Vec<f32> = (0..(m * k) as usize).map(val).collect();
    let mut out1 = vec![-1.0f32; m as usize];
    let mut out2buf = vec![-1.0f32; m as usize];
    let uniform = [m, k];
    let mut bufs: Vec<*mut u8> =
        vec![x.as_mut_ptr() as *mut u8, out1.as_mut_ptr() as *mut u8];
    if out2 {
        bufs.push(out2buf.as_mut_ptr() as *mut u8);
    }
    unsafe {
        jit.run(0, 0, (m * 64) as u64, m, 1, uniform.as_ptr(), bufs.as_ptr());
    }
    (out1, out2buf)
}

/// F1 — the minimal repro this bug was originally found with: no loop at all,
/// just `var a = x[row]; partial[t] = a;`. Proves the defect is not
/// loop-dependent.
const F1: &str = r#"
struct Params { m: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    var a = x[row];
    partial[t] = a;
    workgroupBarrier();
    if (t == 0u) {
        out[row] = partial[0];
    }
}
"#;

#[test]
fn f1_local_read_with_no_loop_is_not_stale() {
    let m = 5u32;
    let (out, _) = run_wg_kernel(F1, m, 1, false);
    for row in 0..m as usize {
        assert_eq!(out[row], val(row), "row {row}");
    }
}

/// F2 — the shape `rmsnorm_rows`/`gradnorm_part`/`clip_coef_wg` share: a local
/// accumulated by a `+` loop, then read by a top-level store.
const F2: &str = r#"
struct Params { m: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    let base = row * p.k;
    var a = 0.0;
    for (var c = t; c < p.k; c = c + 64u) {
        a = a + x[base + c];
    }
    partial[t] = a;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[i];
        }
        out[row] = s;
    }
}
"#;

#[test]
fn f2_sum_reduction_local_matches_host_reference() {
    let (m, k) = (5u32, 37u32);
    let (out, _) = run_wg_kernel(F2, m, k, false);
    for row in 0..m as usize {
        let base = row * k as usize;
        let want: f32 = (0..k as usize).map(|c| val(base + c)).sum();
        assert_eq!(out[row], want, "row {row}");
    }
}

/// F3 — the real `max_abs_rows` shape: a `max` reduction local. `max` is
/// associative and exact on floats, so this is the one fixture that stands in
/// directly for a landed kernel and is worth its own `assert_eq!` (no
/// tolerance needed either way, since F2/F4/F5's inputs are exact integers).
const F3: &str = r#"
struct Params { m: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    let base = row * p.k;
    var a = 0.0;
    for (var c = t; c < p.k; c = c + 64u) {
        a = max(a, abs(x[base + c]));
    }
    partial[t] = a;
    workgroupBarrier();
    if (t == 0u) {
        var mx = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            mx = max(mx, partial[i]);
        }
        out[row] = mx;
    }
}
"#;

#[test]
fn f3_max_reduction_local_matches_host_reference() {
    let (m, k) = (5u32, 37u32);
    let (out, _) = run_wg_kernel(F3, m, k, false);
    for row in 0..m as usize {
        let base = row * k as usize;
        let want = (0..k as usize).map(|c| val(base + c).abs()).fold(0.0f32, f32::max);
        assert_eq!(out[row], want, "row {row}");
    }
}

/// F4 — the layernorm-family shape: TWO locals fed by one loop, then TWO
/// separate top-level stores (`layernorm_rows`/`ln_stats_rows` write two;
/// `layernorm_dx_rows` writes four — two is enough to exercise "more than one
/// carried expression per segment").
const F4: &str = r#"
struct Params { m: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_sum: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_sumsq: array<f32>;
var<workgroup> partial1: array<f32, 64>;
var<workgroup> partial2: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    let base = row * p.k;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var c = t; c < p.k; c = c + 64u) {
        let v = x[base + c];
        s1 = s1 + v;
        s2 = s2 + v * v;
    }
    partial1[t] = s1;
    partial2[t] = s2;
    workgroupBarrier();
    if (t == 0u) {
        var t1 = 0.0;
        var t2 = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            t1 = t1 + partial1[i];
            t2 = t2 + partial2[i];
        }
        out_sum[row] = t1;
        out_sumsq[row] = t2;
    }
}
"#;

#[test]
fn f4_two_locals_two_top_level_stores_match_host_reference() {
    let (m, k) = (5u32, 19u32); // small k: v*v stays well under 2^24 for the exact-sum property
    let (sum, sumsq) = run_wg_kernel(F4, m, k, true);
    for row in 0..m as usize {
        let base = row * k as usize;
        let want_sum: f32 = (0..k as usize).map(|c| val(base + c)).sum();
        let want_sumsq: f32 = (0..k as usize).map(|c| val(base + c).powi(2)).sum();
        assert_eq!(sum[row], want_sum, "row {row} sum");
        assert_eq!(sumsq[row], want_sumsq, "row {row} sumsq");
    }
}

/// F5 — the `prelu_bwd_wg`/`paged_decode_scores_wg` shape: the pre-barrier
/// store is nested under an `if { for { .. } }`, not a bare top-level loop,
/// before the top-level read after it. `k < 64` so some lanes' `if` guard is
/// false (they never enter the loop at all) and some are true — both paths
/// are exercised in the same dispatch.
const F5: &str = r#"
struct Params { m: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    let base = row * p.k;
    var a = 0.0;
    if (t < p.k) {
        for (var c = t; c < p.k; c = c + 64u) {
            a = a + x[base + c];
        }
    }
    partial[t] = a;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[i];
        }
        out[row] = s;
    }
}
"#;

#[test]
fn f5_store_nested_under_if_for_matches_host_reference() {
    let (m, k) = (5u32, 30u32); // < 64: some lanes' `if (t < p.k)` guard is false
    let (out, _) = run_wg_kernel(F5, m, k, false);
    for row in 0..m as usize {
        let base = row * k as usize;
        let want: f32 = (0..k as usize).map(|c| val(base + c)).sum();
        assert_eq!(out[row], want, "row {row}");
    }
}

/// F6 — the invariant `compile_one_wg`'s own doc comment already states but,
/// before this test, nothing checked: "per-invocation `var` locals ... none
/// may cross the barrier." A local stored BEFORE the barrier and loaded AFTER
/// it must be a compile-time error, not a silent read of zero (the same
/// silent-wrong-answer class as the bug this file otherwise pins, one
/// function away). No `var<workgroup>` is needed for this one — a
/// `workgroupBarrier()` alone is enough to route it through `compile_one_wg`.
const F6: &str = r#"
struct Params { m: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.m) { return; }
    var a = x[row];
    workgroupBarrier();
    out[row] = a;
}
"#;

#[test]
fn f6_local_live_across_the_barrier_is_a_compile_error() {
    match Jit::new(&[("fixture", F6)]) {
        Ok(_) => panic!("must be rejected at compile time"),
        Err(err) => assert!(
            err.contains("live across the workgroup synchronisation point"),
            "expected the barrier-crossing-local error, got: {err}"
        ),
    }
}
