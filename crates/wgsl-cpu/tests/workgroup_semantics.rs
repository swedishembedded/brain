// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two WGSL guarantees about work-group execution that the barrier-splitting
//! JIT model does not give for free, and that a kernel cannot tell has been
//! dropped: it keeps computing, with a plausible number.
//!
//! 1. **An early `return` is permanent.** WGSL's rule is that the predicate
//!    guarding a pre-barrier `return` must be work-group-uniform - and when it
//!    is, the returned invocation does no further work AT ALL, including after
//!    the barrier. The JIT compiles the body as two independent per-invocation
//!    loops split at the barrier, so a `return` in the first loop skips only
//!    the rest of THAT loop; the second segment then runs for an invocation
//!    that had already left. Every kernel here that guards on a padded grid
//!    (`if (w >= p.n_wg) { return; }`) writes an output it must not write.
//! 2. **`var<workgroup>` is zero-initialised.** WGSL zero-initialises
//!    work-group memory at the start of every work-group's execution. The JIT
//!    backs it with a stack slot allocated once and reused for every
//!    work-group, so a slot no invocation writes on this pass reads back the
//!    PREVIOUS work-group's value - and a reduction whose tail lanes are
//!    inactive folds that stale value into its sum.
//!
//! Both are exercised over several work-groups, because with one work-group
//! the stale value happens to be the uninitialised-but-zero stack and the
//! surplus-invocation case never arises.

use wgsl_cpu::Jit;

/// Small integer-valued floats: every sum below stays exact in f32, so the
/// assertions are bit-for-bit and no tolerance can hide a wrong addend.
fn val(i: usize) -> f32 {
    ((i * 37 + 11) % 251) as f32 - 125.0
}

/// A reduction that guards on a padded grid, exactly as the real gradient-norm
/// and row-statistics kernels do.
const EARLY_RETURN: &str = r#"
struct Params { numel: u32, n_wg: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    let t = li.x;
    // Work-group-uniform, which is what makes the early return legal.
    if (w >= p.n_wg) { return; }
    var acc = 0.0;
    var i = w * 64u + t;
    for (; i < p.numel; i = i + p.n_wg * 64u) {
        acc = acc + x[i];
    }
    partial[t] = acc;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            s = s + partial[k];
        }
        out[w] = s;
    }
}
"#;

/// A reduction whose tail lanes never write their slot, so the sum is correct
/// only if the unwritten slots read back as zero.
const SPARSE_PARTIALS: &str = r#"
struct Params { live: u32, n_wg: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (t < p.live) {
        partial[t] = x[w * 64u + t];
    }
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            s = s + partial[k];
        }
        out[w] = s;
    }
}
"#;

/// Compile and dispatch `dispatched` work-groups of 64 invocations.
fn run(src: &str, params: &[u32], x: &[f32], out_len: usize, dispatched: u32, fill: f32) -> Vec<f32> {
    let jit = Jit::new(&[("fixture", src)]).expect("fixture must compile");
    let mut xs = x.to_vec();
    let mut out = vec![fill; out_len];
    let bufs = [xs.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];
    // SAFETY: both bindings are large enough for every index the kernel forms
    // under these params, and the uniform stream is the packed `Params`.
    unsafe {
        jit.run(0, 0, (dispatched as u64) * 64, dispatched, 1, params.as_ptr(), bufs.as_ptr());
    }
    out
}

/// An invocation that returned before the barrier must not run the segment
/// after it either - so a surplus work-group writes nothing at all.
#[test]
fn an_invocation_that_returned_before_the_barrier_stays_returned() {
    let (numel, n_wg, dispatched) = (400usize, 2u32, 5u32);
    let x: Vec<f32> = (0..numel).map(val).collect();
    let out = run(EARLY_RETURN, &[numel as u32, n_wg], &x, dispatched as usize, dispatched, -1.0);

    for (w, got) in out.iter().take(n_wg as usize).enumerate() {
        let mut want = 0.0f32;
        for t in 0..64usize {
            let mut i = w * 64 + t;
            while i < numel {
                want += x[i];
                i += n_wg as usize * 64;
            }
        }
        assert_eq!(*got, want, "work-group {w}");
    }
    for (w, got) in out.iter().enumerate().skip(n_wg as usize) {
        assert_eq!(*got, -1.0, "work-group {w} returned before the barrier and must write nothing");
    }
}

/// Work-group memory starts at zero for EVERY work-group, not just the first.
#[test]
fn workgroup_memory_is_zero_at_the_start_of_every_workgroup() {
    let (live, n_wg) = (5u32, 4u32);
    let x: Vec<f32> = (0..(n_wg as usize) * 64).map(val).collect();
    let out = run(SPARSE_PARTIALS, &[live, n_wg], &x, n_wg as usize, n_wg, 0.0);

    for (w, got) in out.iter().enumerate() {
        let want: f32 = (0..live as usize).map(|t| x[w * 64 + t]).sum();
        assert_eq!(
            *got, want,
            "work-group {w} folded a slot no invocation wrote; WGSL says it reads zero"
        );
    }
}
