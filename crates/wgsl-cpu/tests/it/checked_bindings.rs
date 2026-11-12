// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for kernels that fail with a diagnosis instead of a crash for its clients.
// If your team needs expertise in JIT code generation and debuggable accelerator back ends then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A checked [`Jit`] turns a storage buffer shorter than its kernel reaches into an error that names the kernel and
//! the binding, touches no memory outside the buffers, and leaves in-range kernels bit-identical.

use wgsl_cpu::Jit;

/// `out[i] = 2 * x[i]` for `i < p.n`, over two storage bindings.
const SCALE2: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    out[gid.x] = 2.0 * x[gid.x];
}
"#;

/// Run `scale2` over `n` invocations with `x` and `out` bound to the first `x_len` and `out_len` floats of buffers that
/// are really `x_real` and `out_real` floats long. Returns the checked result and `out` afterwards.
fn run_scale2(n: u32, x_len: usize, out_len: usize, x_real: usize, out_real: usize) -> (Result<(), wgsl_cpu::AccessFault>, Vec<f32>) {
    let jit = Jit::new_checked(&[("scale2", SCALE2)]).expect("compile scale2");
    let mut x: Vec<f32> = (0..x_real).map(|i| i as f32 + 1.0).collect();
    // The canary past the bound region of `out`: the failing store must not reach it.
    let mut out = vec![-7.0f32; out_real];
    let uni = [n];
    let bufs = [x.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];
    let lens = [(x_len * 4) as u64, (out_len * 4) as u64];
    // SAFETY: `bufs` point into live vectors at least `*_len` floats long (`*_real >= *_len`); the checked kernel
    // confines every access to `lens`.
    let r = unsafe { jit.run_checked(0, 0, 64, 1, 1, uni.as_ptr(), bufs.as_ptr(), lens.as_ptr()) };
    (r, out)
}

#[test]
fn a_binding_in_range_computes_exactly_what_the_unchecked_kernel_does() {
    let (r, out) = run_scale2(10, 10, 10, 10, 10);
    r.expect("an in-range dispatch must not fault");
    assert_eq!(&out[..], &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
}

#[test]
fn a_too_short_input_is_an_error_naming_the_kernel_and_the_binding() {
    // 10 invocations read x[0..10] but only 6 floats are bound; the real allocation is big enough to not crash either way.
    let (r, out) = run_scale2(10, 6, 10, 16, 10);
    let e = r.expect_err("reading past the bound input must fault");
    assert_eq!((e.kernel.as_str(), e.binding, e.name.as_str()), ("scale2", 1, "x"));
    assert_eq!(e.byte_len, 24);
    assert!(e.byte_offset >= 24, "the recorded offset {} is not past the end of the 24 bound bytes", e.byte_offset);
    let msg = e.to_string();
    assert!(msg.contains("scale2") && msg.contains("`x`"), "message must name kernel and binding: {msg}");
    // The in-range lanes still stored their results; the lanes that read zero stored 2 * 0.
    assert_eq!(&out[..6], &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn a_too_short_output_is_an_error_and_the_store_never_lands() {
    // `out` is bound as 4 floats but backed by 10 sentinels: the six stores past the bound must be dropped.
    let (r, out) = run_scale2(10, 10, 4, 10, 10);
    let e = r.expect_err("writing past the bound output must fault");
    assert_eq!((e.kernel.as_str(), e.binding, e.name.as_str()), ("scale2", 2, "out"));
    assert_eq!(&out[..4], &[2.0, 4.0, 6.0, 8.0]);
    assert!(out[4..].iter().all(|&v| v == -7.0), "a store past the bound reached memory: {out:?}");
}

#[test]
fn a_workgroup_kernel_is_checked_too() {
    let jit = Jit::new_checked(&[("gradnorm_part", kernels::GRADNORM_PART)]).expect("compile gradnorm_part");
    // 500 elements are claimed by the params, 100 are bound.
    let mut grad = vec![1.0f32; 500];
    let mut parts = vec![0.0f32; 2];
    let uni = [500u32, 0, 2];
    let bufs = [grad.as_mut_ptr() as *mut u8, parts.as_mut_ptr() as *mut u8];
    let lens = [400u64, 8];
    // SAFETY: both vectors are at least as long as the lengths passed; the checked kernel confines itself to them.
    let r = unsafe { jit.run_checked(0, 0, 128, 2, 1, uni.as_ptr(), bufs.as_ptr(), lens.as_ptr()) };
    let e = r.expect_err("a workgroup kernel reading past its binding must fault");
    assert_eq!((e.kernel.as_str(), e.binding, e.name.as_str()), ("gradnorm_part", 1, "grad"));
}

/// The range checks are emitted at every place shape the translator supports (scalars, vectors, structs, arrays, in
/// both execution models); the Cranelift verifier is on, so a malformed check fails the build of the kernel.
#[test]
fn every_catalogue_kernel_still_compiles_with_range_checks() {
    let mut failed = Vec::new();
    for (name, src) in kernels::ALL {
        if let Err(e) = Jit::new_checked(&[(name, src)]) {
            failed.push(format!("{name}: {e}"));
        }
    }
    assert!(failed.is_empty(), "{} of {} kernels failed with checks on:\n{}", failed.len(), kernels::ALL.len(), failed.join("\n"));
}
