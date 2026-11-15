// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A test chooses its backend for its own thread, never for the process.
//!
//! Swedish Embedded AB implements test suites whose results do not depend on
//! which other tests happen to run beside them for its clients. If your team
//! needs expertise in isolating GPU tests that share one process and one
//! device, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A test binary runs its tests side by side. A test that set the process's
//! default backend to the CPU moved every device built after it in the binary
//! onto the CPU too, so a later test compared a CUDA engine against a CPU one
//! and failed only in the full run. [`gpu_core::with_backend`] scopes the
//! choice to the calling thread, and the shared test device follows it.

use gpu_core::Backend;

static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL)];

#[test]
fn a_scoped_backend_holds_for_its_own_thread_only() {
    let ambient = gpu_core::backend_name();
    gpu_core::with_backend(Backend::Cpu, || {
        assert_eq!(gpu_core::backend_name(), "cpu");
        let other = std::thread::spawn(gpu_core::backend_name).join().unwrap();
        assert_eq!(other, ambient, "another thread keeps the process's backend");
        gpu_core::with_backend(Backend::Vulkan, || assert_eq!(gpu_core::backend_name(), "vulkan"));
        assert_eq!(gpu_core::backend_name(), "cpu", "an inner scope ending restores the outer one");
    });
    assert_eq!(gpu_core::backend_name(), ambient, "the scope ends with its closure");
    let unwound = std::panic::catch_unwind(|| gpu_core::with_backend(Backend::Cpu, || panic!("inside the scope")));
    assert!(unwound.is_err());
    assert_eq!(gpu_core::backend_name(), ambient, "a panic inside the scope still ends it");
}

#[test]
fn the_shared_test_device_follows_the_scoped_backend() {
    // An ambient device for the same kernel set is alive while the scoped test asks for one.
    let held = (!brain_testutil::gpu_tests_skipped()).then(|| gpu_core::testgpu::dev(KERNELS));
    let cpu = gpu_core::with_backend(Backend::Cpu, || gpu_core::testgpu::dev(KERNELS));
    assert_eq!(cpu.kind(), "cpu", "a CPU-scoped test gets a CPU device, not the pooled ambient one");
    if let Some(held) = held {
        assert_eq!(gpu_core::testgpu::dev(KERNELS).kind(), held.kind(), "an unscoped test still shares the ambient device");
    }
}
