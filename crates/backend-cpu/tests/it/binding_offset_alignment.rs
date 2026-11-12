// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for kernels that behave the same on a laptop CPU and on a strict GPU
// for its clients. If your team needs expertise in portable accelerator back ends then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The CPU backend refuses a sub-range binding the strictest supported device would refuse (a storage-binding
//! offset that is not a multiple of 256 bytes), naming the kernel, the binding and the offset, and accepts an
//! aligned one.

use backend_cpu::CpuBackend;

const COPY: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    out[gid.x] = x[gid.x];
}
"#;

fn run_at(offset_words: u64) -> Vec<f32> {
    let cpu = CpuBackend::new(&[("copy_view", COPY)]);
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let x = cpu.storage_init("x", &data);
    let out = cpu.storage(64);
    cpu.submit(&[], &[cpu.step_sliced(0, &[&x, &out], &[(offset_words, 0), (0, 0)], &[64], 64)]);
    cpu.read(&out, 64)
}

#[test]
fn an_offset_on_the_256_byte_grain_binds_the_view() {
    assert_eq!(run_at(64)[0], 64.0);
}

#[test]
#[should_panic(expected = "kernel `copy_view` binding 0 offset 128 bytes (32 words)")]
fn an_offset_off_the_grain_panics_naming_the_kernel_binding_and_offset() {
    run_at(32);
}
