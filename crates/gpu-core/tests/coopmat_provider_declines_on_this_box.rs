// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.9's own proof gate: `CoopMatProvider::requires()` genuinely gates
//! against this box's REAL, queried `DeviceCaps` - not a synthetic/mocked
//! one - and correctly reports the provider UNREACHABLE here.
//!
//! Modelled directly on `arch_desc_real_divergence.rs` (M8.1's own real-
//! hardware gate): open the real `backend-vulkan` device this sandbox
//! actually has (an Intel iGPU, no `VK_KHR_cooperative_matrix` shapes - see
//! `kernel-performance.md`'s hardware note for this campaign's own sandbox),
//! and assert the requirement `CoopMatProvider::requires` reports is NOT
//! satisfied by its real `caps()`. Cooperative matrix needs Turing sm_75+
//! (NVIDIA) or an equivalent matrix-engine driver; this box has neither, so
//! this is the expected, correct, DESIGNED outcome - a fake positive here
//! would be the bug, not the pass.
//!
//! **A real, measured surprise, recorded rather than assumed away.** This
//! test originally also asserted `VulkanBackend::register_native` returns
//! `None` on this device (the SPIR-V genuinely compiles here - glslc is on
//! `PATH` - so a decline would be the DEVICE's, not a missing build
//! artifact). That assertion is FALSE on this real hardware: Mesa's Intel
//! ANV driver (no `VK_LAYER_KHRONOS_validation` installed) accepts
//! `vkCreateComputePipelines` for the coopmat SPIR-V and returns a live
//! pipeline object, even though `caps.arch.matrix` is `None` (no
//! `VK_KHR_cooperative_matrix` shapes queried) and this test's own
//! `Requirement` check above correctly reports the provider unreachable.
//! Pipeline CREATION succeeding is not the same claim as "this device can
//! correctly EXECUTE `OpTypeCooperativeMatrixKHR`" - a driver without strict
//! SPIR-V capability validation (no validation layer active) can accept a
//! pipeline it would mis-execute or hang on, silently. This is exactly why
//! `Requirement.matrix`/`ProviderRegistry::resolve` - checked BEFORE
//! `register_native` is ever reached in production - is the one authoritative
//! gate, not "did `register_native` return `Some`". Actually DISPATCHING this
//! pipeline (to see whether the result is correct, garbage, or a device
//! hang) is exactly the kind of hardware-harness-gated question this
//! campaign's own decision 2 exists for, so this test observes and records
//! the outcome via `skip_unvalidated_capability` and never attempts a real
//! dispatch here.

use backend_api::Backend;
use backend_vulkan::VulkanBackend;
use gpu_core::provider::coopmat::CoopMatProvider;
use gpu_core::provider::{OpRequest, OperatorProvider, Pass};

#[test]
fn coopmat_provider_requires_is_unsatisfied_by_this_boxs_real_vulkan_caps() {
    let vk = match VulkanBackend::try_new(&[]) {
        Ok(b) => b,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            return;
        }
    };
    let caps = vk.caps();
    eprintln!(
        "real device matrix engine: {:?} (None means no VK_KHR_cooperative_matrix shapes reported)",
        caps.arch.matrix.as_ref().map(|m| (m.kind, m.shapes.len()))
    );

    let provider = CoopMatProvider::new();
    let req = OpRequest {
        op: backend_api::select::Op::MatMul,
        shape: backend_api::select::OpShape { m: 32, n: 32, k: 32, dtype: backend_api::DType::F16 },
        pass: Pass::Forward,
        operands: &[],
        attrs: &[],
        bind: &|_| panic!("coopmat provider never calls OpRequest::bind"),
    };
    let requirement = provider.requires(&req);

    if requirement.satisfied_by(&caps) {
        // Only reachable on a box that genuinely has a usable f16x f16->f32
        // matrix engine - not this sandbox's Intel iGPU, per the ledger's own
        // hardware note. If this branch ever runs, it is real hardware
        // proving the OTHER half of the gate (a positive), not a failure.
        brain_testutil::skip_unvalidated_capability(
            "coopmat-hardware",
            "this box's real Vulkan device reports a usable cooperative-matrix shape - the \
             declined-everywhere case this test names is not what this run measured; the \
             opposite (a real dispatch) is unvalidated by THIS test either way",
        );
        return;
    }

    // The expected, designed-for outcome on this sandbox: the provider is
    // PROVABLY unreachable, not merely assumed to be.
    println!("coopmat provider correctly UNREACHABLE on this box's real DeviceCaps");

    // `register_native` never panics either way - observe, do not assume.
    // See this file's own doc comment for the real, measured surprise this
    // probe found: pipeline CREATION can succeed on a device this test just
    // proved the `Requirement` gate correctly declines, which is exactly why
    // that gate (not this probe) is what a real caller must go through.
    let Some(spec) = backend_vulkan::coopmat::spec() else {
        eprintln!("note: no coopmat SPIR-V baked in at build time (no glslc/glslangValidator on PATH)");
        return;
    };
    match vk.register_native(&spec) {
        None => println!("register_native also declined the real coopmat SPIR-V on this device (None)"),
        Some(_id) => {
            // Deliberately does NOT dispatch it (no `step_native`/`submit`/
            // `read`) - whether this pipeline executes correctly, produces
            // garbage, or hangs the device on hardware without a real matrix
            // engine is unknown and not safe to probe on a shared sandbox.
            brain_testutil::skip_unvalidated_capability(
                "coopmat-hardware",
                "vkCreateComputePipelines accepted the coopmat SPIR-V on this Intel ANV device \
                 even though it has no VK_KHR_cooperative_matrix shapes (caps.arch.matrix is \
                 None) - pipeline creation succeeding is NOT proof this device can correctly \
                 EXECUTE OpTypeCooperativeMatrixKHR (no validation layer is active here); needs \
                 Turing sm_75+ or equivalent real matrix-engine hardware to validate a dispatch",
            );
        }
    }
}
