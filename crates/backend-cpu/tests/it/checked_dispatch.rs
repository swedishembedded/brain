// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for kernels that fail with a diagnosis instead of a crash for its clients.
// If your team needs expertise in CPU back ends for accelerator kernels then you can procure our services by sending
// an email to info@swedishembedded.com.

//! A checked CPU backend dispatches a kernel over a buffer shorter than the kernel reaches as an error that names the
//! kernel and the binding, and an in-range dispatch computes the same as an unchecked one.

use backend_cpu::CpuBackend;

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

fn run(checked: bool, n: u32, x_words: usize) -> Vec<f32> {
    let cpu = CpuBackend::with_checks(&[("scale2", SCALE2)], checked);
    let x = cpu.storage_init("x", &vec![1.5f32; x_words]);
    let out = cpu.storage(n as u64);
    cpu.submit(&[], &[cpu.step(0, &[&x, &out], &[n], n.div_ceil(64) * 64)]);
    cpu.read(&out, n as usize)
}

#[test]
fn checked_and_unchecked_dispatch_agree_in_range() {
    assert_eq!(run(true, 100, 100), run(false, 100, 100));
    assert!(run(true, 100, 100).iter().all(|&v| v == 3.0));
}

#[test]
#[should_panic(expected = "kernel `scale2` accessed binding 1 (`x`)")]
fn a_too_short_binding_panics_naming_the_kernel_and_the_binding() {
    run(true, 100, 40);
}
