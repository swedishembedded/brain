// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Ops::matmul_group` - several projections of one activation - against one
//! `Ops::matmul` per weight.
//!
//! Swedish Embedded AB implements bit-exact decode kernels for quantised LLM
//! inference. If your team needs expertise in merging a layer's projections
//! into one launch without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The group is an optimisation that must be invisible: every output is
//! BIT-identical to the separate matmuls', on every device (where the native
//! multi GEMV is not offered the group simply IS the separate matmuls), and
//! on a CUDA device an eligible group really is ONE step - without that
//! second assertion the first would hold trivially on a build that never
//! merged anything.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::{Ops, Weight};

/// Whether this device is offered the native kernels: a CUDA device, with
/// native kernels not withheld (`gpu_core::set_native_kernels`, the A/B
/// switch that pins the WGSL tier).
/// Under it these gates skip, and the "offered only where it can run" test
/// asserts the withholding.
fn is_cuda(g: &Gpu) -> bool {
    g.kind() == "cuda" && g.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Run `ns` (one weight each) of `dtypes` against one activation of `m` rows
/// both ways. Returns how many steps the group pushed.
fn check(m: u32, k: u32, ns: &[u32], dtypes: &[Dtype], seed: u64) -> usize {
    let gpu = gpu_core::testgpu::dev(model::ops::kernel_list());
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();
    let mut rng = Lcg::new(seed);
    let x = g.storage_init("x", &rng.vec_scaled((m * k) as usize, 1.0));
    let weights: Vec<Weight> = ns.iter().zip(dtypes).map(|(&n, &dt)| Weight::upload(&ops, &rng.vec_scaled((n * k) as usize, 1.0), n as usize, k as usize, dt)).collect();

    let mut steps = Vec::new();
    let act = ops.act(&mut steps, &x, 0, m, k);
    let grouped: Vec<DeviceBuffer> = ns.iter().map(|&n| g.storage((m * n) as u64)).collect();
    let separate: Vec<DeviceBuffer> = ns.iter().map(|&n| g.storage((m * n) as u64)).collect();

    let pairs: Vec<(&Weight, &DeviceBuffer)> = weights.iter().zip(&grouped).collect();
    let before = steps.len();
    ops.matmul_group(&mut steps, &pairs, &act);
    let pushed = steps.len() - before;
    for (w, y) in weights.iter().zip(&separate) {
        ops.matmul(&mut steps, w, &act, y, 0);
    }
    g.submit(&[], &steps);
    g.poll_wait();
    for (i, &n) in ns.iter().enumerate() {
        assert_eq!(
            bits(g.read(&grouped[i], (m * n) as usize)),
            bits(g.read(&separate[i], (m * n) as usize)),
            "weight {i} differs: m={m} k={k} ns={ns:?}"
        );
    }
    pushed
}

#[test]
fn an_int8_group_is_one_step_on_cuda_and_bit_identical_everywhere() {
    let gpu = gpu_core::testgpu::dev(model::ops::kernel_list());
    let cuda = is_cuda(&gpu) && gpu.caps().numeric.int8_dot;
    drop(gpu);
    for (m, k, ns) in [(1u32, 256u32, vec![96u32, 8, 8, 64]), (1, 256, vec![130, 17, 33]), (1, 128, vec![64, 64]), (4, 256, vec![40, 9])] {
        let dtypes = vec![Dtype::I8; ns.len()];
        let pushed = check(m, k, &ns, &dtypes, 0x9e37 ^ u64::from(m * 131 + ns[0]));
        if cuda {
            assert_eq!(pushed, 1, "an eligible int8 group (m={m} ns={ns:?}) should be ONE launch");
        }
    }
}

#[test]
fn a_group_that_cannot_merge_is_the_separate_matmuls() {
    let gpu = gpu_core::testgpu::dev(model::ops::kernel_list());
    let cuda = is_cuda(&gpu) && gpu.caps().numeric.int8_dot;
    drop(gpu);
    // A mixed tier, a batch past the GEMV's tile, and more than four weights.
    let mixed = check(1, 256, &[64, 64], &[Dtype::I8, Dtype::F32], 77);
    let big_m = check(9, 256, &[64, 64], &[Dtype::I8, Dtype::I8], 78);
    if cuda {
        assert!(mixed >= 2, "a mixed-tier group must not be merged");
        assert!(big_m >= 2, "a batch past the GEMV tile must not be merged");
    }
}
