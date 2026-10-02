// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The placement policies through the `Gpu` facade on a CUDA device: opt-in
//! managed and system buffers that kernels use, and that are charged to the pool
//! that holds them.
//!
//! Swedish Embedded AB implements memory-placement strategies for coherent
//! CPU-GPU systems for its clients. If your team needs expertise in placing
//! tensors across a coherent memory hierarchy, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Skip-if-absent: a box with no CUDA device, or whose device cannot make a given
//! kind of allocation, has nothing to assert for it.

use gpu_core::{AllocPolicy, Gpu, PrefetchTarget};

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const N: u32 = 4096;

#[test]
fn managed_and_system_buffers_run_a_kernel_through_the_facade() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let facts = gpu.placement_facts();
    let src = gpu.storage_init("src", &(0..N).map(|i| i as f32).collect::<Vec<_>>());
    for policy in [AllocPolicy::Managed, AllocPolicy::System] {
        let made = gpu.try_alloc_placed("out", N as u64 * 4, policy);
        assert_eq!(made.is_ok(), facts.supports(policy), "{policy:?}: {:?}", made.as_ref().err());
        let Ok(out) = made else { continue };
        gpu.prefetch(&out, 0, N as u64 * 4, PrefetchTarget::Device).expect("prefetch");
        let step = gpu.step(0, &[&out, &src], &[N, 3.0f32.to_bits()], N);
        gpu.submit(&[], &[step]);
        let got = gpu.read(&out, N as usize);
        assert!(got.iter().enumerate().all(|(i, &v)| v == 3.0 * i as f32), "{policy:?} buffer computed the wrong numbers");
    }
}
