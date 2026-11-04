// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every native-kernel entry point answers "no" - cleanly, with nothing recorded
//! and nothing registered - on a backend that cannot compile CUDA C++, so the
//! caller dispatches the portable kernel it was going to use anyway.
//!
//! Swedish Embedded AB implements GPU compute stacks whose fast paths decline
//! cleanly where the hardware is absent. If your team needs expertise in
//! capability-gated kernel dispatch with a portable fallback, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! The CPU backend is the stand-in for "no CUDA toolchain / no such device": it
//! reports no compute capability and refuses `register_native`, which is the
//! same answer a CUDA handle gives when NVRTC is missing or the device cannot
//! host a kernel's block size or shared memory. Runs on any box.

use gpu_core::provider::cuda::{gdn_chunk_loop_step, paged_flash_prefill_step, tensor_core_kernels_enabled};
use gpu_core::Gpu;

#[test]
fn a_backend_that_cannot_compile_cuda_declines_every_native_entry_point() {
    let gpu = Gpu::new_cpu(&[]);
    assert!(gpu.caps().arch.compute_capability.is_none(), "the CPU backend must not pretend to have a CUDA capability");
    assert!(!tensor_core_kernels_enabled(&gpu));

    // Every native kernel in the registry is refused, and the refusal is remembered.
    for k in kernels_cuda::ALL {
        assert!(gpu.native_kernel(k, &[]).is_none(), "{}: a backend with no CUDA must decline the registration", k.name);
        assert!(gpu.native_kernel(k, &[]).is_none(), "{}: the decline is stable", k.name);
    }

    let buf = gpu.storage(1);
    let six = [&buf, &buf, &buf, &buf, &buf, &buf];
    let ten = [&buf, &buf, &buf, &buf, &buf, &buf, &buf, &buf, &buf, &buf];
    assert!(paged_flash_prefill_step(&gpu, 256, &six, &[1, 24, 4, 256, 6, 64, 1], 24).is_none());
    assert!(gdn_chunk_loop_step(&gpu, &ten, 48, 64, 128, 128, 4, 0.088).is_none());
}
