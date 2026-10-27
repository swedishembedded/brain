// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native elementwise kernels (`kernels_cuda`, `cu/elementwise_f32.cu`) that
//! `gpu_core::native_upgrade` substitutes for `bias_add`, `add2` and `gelu_erf` on a CUDA device.
//!
//! Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
//! needs expertise in putting the memory-bound glue of a transformer tower onto the bandwidth roof of the card it
//! runs on then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! Each is an elementwise expression, so nothing is reassociated and the bar is the RAW BITS of the WGSL
//! reference, dispatched on the same device from the same inputs under a name no redirect knows. Shapes are the
//! served towers' (SAM's 4096 x 3072 hidden activations, CLIP's 257-row ones) plus lengths and widths that are not
//! a multiple of four, which the native kernel must hand back to the generated tier.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[
    ("bias_add", kernels::BIAS_ADD),
    ("add2", kernels::ADD2),
    ("gelu_erf", kernels::GELU_ERF),
    ("bias_add_ref", kernels::BIAS_ADD),
    ("add2_ref", kernels::ADD2),
    ("gelu_erf_ref", kernels::GELU_ERF),
];
const BIAS_ADD: usize = 0;
const ADD2: usize = 1;
const GELU: usize = 2;
const REF: usize = 3;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Values across many binades, both signs, zeros and large magnitudes: enough to make an `expf` or a rounding that
/// differs in one bit show.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|i| if i % 97 == 0 { 0.0 } else { r.signed() * 10f32.powi((r.next_u32() % 6) as i32 - 2) }).collect()
}

/// Run slot `k` and its reference `k + REF` on copies of the same buffers and compare buffer `written`.
fn same_bits(gpu: &Gpu, k: usize, inputs: &[&[f32]], written: usize, params: &[u32], threads: u32, what: &str) {
    let run = |slot: usize| {
        let bufs: Vec<_> = inputs.iter().map(|v| gpu.storage_init("in", v)).collect();
        let refs: Vec<&gpu_core::DeviceBuffer> = bufs.iter().collect();
        gpu.submit(&[], &[gpu.step(slot, &refs, params, threads)]);
        gpu.poll_wait();
        gpu.read(&bufs[written], inputs[written].len())
    };
    assert!(bits(&run(k)) == bits(&run(k + REF)), "{what}: native differs from the reference");
}

fn the_native_elementwise_kernels_are_bit_identical_to_the_reference() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(ADD2, &[1024]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(BIAS_ADD, &[4096, 3072]), Some("bias_add_f32"));
    assert_eq!(gpu.native_kernel_for(ADD2, &[4096 * 768]), Some("add2_f32"));
    assert_eq!(gpu.native_kernel_for(GELU, &[4096 * 3072]), Some("gelu_erf_f32"));
    assert_eq!(gpu.native_kernel_for(BIAS_ADD, &[0, 3072]), None, "no rows");
    assert_eq!(gpu.native_kernel_for(ADD2, &[0]), None, "no elements");

    // (rows, width): the towers' shapes, then widths that are not whole float4s and a single row.
    for (i, &(m, n)) in [(4096u32, 3072u32), (4900, 2304), (257, 1024), (101, 4096), (33, 7), (5, 1), (1, 768), (1000, 9)].iter().enumerate() {
        let x = values((m * n) as usize, 10 + i as u64);
        let bias = values(n as usize, 20 + i as u64);
        same_bits(&gpu, BIAS_ADD, &[&x, &bias], 0, &[m, n], m * n, &format!("bias_add {m}x{n}"));
    }
    for (i, &total) in [4096u32 * 768, 4096 * 3072, 12345, 7, 1].iter().enumerate() {
        let (a, b, out) = (values(total as usize, 30 + i as u64), values(total as usize, 40 + i as u64), vec![f32::from_bits(0x7fc0_dead); total as usize]);
        same_bits(&gpu, ADD2, &[&a, &b, &out], 2, &[total], total, &format!("add2 {total}"));
        let x = values(total as usize, 50 + i as u64);
        same_bits(&gpu, GELU, &[&x, &out], 1, &[total], total, &format!("gelu_erf {total}"));
    }
}

gpu_core::card_tests!(
    the_native_elementwise_kernels_are_bit_identical_to_the_reference,
);
