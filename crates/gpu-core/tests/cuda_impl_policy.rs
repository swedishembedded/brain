// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tier ratchet: **an operator the CUDA tier policy declares `Tuned`
//! must never be satisfied by a `Generated` implementation.**
//!
//! Swedish Embedded AB implements verifiable performance contracts for
//! heterogeneous compute stacks. If your team needs expertise in proving
//! that the fast path is the path that actually ran, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # Why this test exists before any tuned CUDA kernel does
//!
//! A generated (WGSL-translated) CUDA kernel and a hand-tuned one are
//! indistinguishable from the outside: both produce the right numbers, and
//! only one of them is why the backend exists. A backend that quietly runs
//! the generated tier where the tuned tier was promised is not a slow
//! backend, it is a backend making a claim nothing checks - the same defect
//! class as a device silently demoting itself to another device.
//!
//! So the instrumentation lands *before* the first tuned kernel, not after:
//! [`gpu_core::provider::ImplChoice`] records which implementation family
//! actually ran and why every other provider was skipped, and
//! `backend_cuda::policy` states - as code, not configuration - which tier
//! an operator is required to reach on a given compute capability. This
//! file is the gate that puts those two together.
//!
//! Nothing here touches a GPU: `resolve_choice` is a pure function of the
//! request, the device capabilities and the provider chain, and the policy
//! check is a pure function of `(op, compute capability, observed tier)`.
//! The one test that does build a device builds the CPU one.

use std::sync::Arc;

use backend_api::select::{self, CachedSelector, Dtype, KernelSelector};
use backend_api::{DType, DeviceCaps, DeviceClass, ImplSource};
use backend_cuda::policy::{self, PolicyEntry};
use gpu_core::provider::{
    DeclineReason, LowerCtx, Lowered, OpRequest, Operand, OperatorProvider, Pass, ProviderRegistry, Role,
};

/// **Test fixture, not a shipped policy.** `backend_cuda::policy::POLICY` is
/// the real table; it is deliberately empty while no hand-written CUDA
/// kernel exists, so this file states its own synthetic requirement in order
/// to exercise the mechanism at all. A compute capability of `(6, 0)` here
/// is an arbitrary fixture *floor*, chosen because every CUDA-capable device
/// this code can run on is at or above it - it is not a claim about any
/// card, and the checks below pass identically on a device with tensor cores.
const FIXTURE_POLICY: &[PolicyEntry] =
    &[PolicyEntry { op: select::Op::MatMul, min_cc: (6, 0), required: ImplSource::Tuned }];

/// A fixture compute capability to evaluate [`FIXTURE_POLICY`] at. Real code
/// gets this from `cuDeviceGetAttribute`; a unit test has no device, so it
/// names one - which is exactly why the policy API takes the capability as
/// an argument instead of querying one behind the caller's back.
const FIXTURE_CC: (u32, u32) = (6, 1);

/// A provider that claims a fixed [`ImplSource`] and accepts everything -
/// the smallest thing that can stand in for a real CUDA provider (which
/// does not exist yet) at the seam the ratchet reads.
struct FixtureProvider {
    name: &'static str,
    source: ImplSource,
    requires: select::Requirement,
    accepts: bool,
}

impl OperatorProvider for FixtureProvider {
    fn name(&self) -> &'static str {
        self.name
    }
    fn source(&self, _req: &OpRequest) -> ImplSource {
        self.source
    }
    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        self.requires
    }
    fn accepts(&self, _req: &OpRequest, _capture: bool) -> bool {
        self.accepts
    }
    fn lower(&self, _ctx: &mut LowerCtx, _req: &OpRequest) -> Result<Lowered, String> {
        Err(format!("{} is a fixture and lowers nothing", self.name))
    }
}

fn fixture(name: &'static str, source: ImplSource) -> Arc<dyn OperatorProvider> {
    Arc::new(FixtureProvider { name, source, requires: select::Requirement::default(), accepts: true })
}

fn bind(v: select::KernelVariant) -> (usize, &'static str) {
    match v {
        select::KernelVariant::Reference => (0, "matmul"),
        _ => (1, "matmul_gemv"),
    }
}

fn request<'a>(operands: &'a [Operand<'a>], attrs: &'a [u32]) -> OpRequest<'a> {
    OpRequest {
        op: select::Op::MatMul,
        shape: select::OpShape { m: 64, n: 64, k: 64, dtype: Dtype::F32 },
        pass: Pass::Forward,
        operands,
        attrs,
        bind: &bind,
    }
}

fn selector() -> Arc<dyn KernelSelector> {
    Arc::new(CachedSelector::new(select::DefaultSelector))
}

/// THE ratchet. An op the policy declares `Tuned` that resolves to a
/// `Generated` implementation is a policy violation, reported with both
/// tiers named - never a silent success.
#[test]
fn a_policy_tuned_op_resolved_to_generated_is_a_violation() {
    let reg = ProviderRegistry::reference(selector()).prefer(fixture("fixture-generated", ImplSource::Generated));
    let caps = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
    let req = request(&[], &[]);

    let (chosen, choice) = reg.resolve_choice(&req, &caps, false);
    assert_eq!(chosen.name(), "fixture-generated");
    assert_eq!(choice.provider, "fixture-generated");
    assert_eq!(choice.source, ImplSource::Generated, "the record must report the tier that actually ran");
    assert_eq!(choice.op, select::Op::MatMul);
    assert_eq!(choice.arch.class, DeviceClass::DiscreteGpu);

    let v = policy::violation_in(FIXTURE_POLICY, choice.op, FIXTURE_CC, choice.source)
        .expect("a Generated impl must NOT satisfy a policy that requires Tuned");
    assert!(v.contains("Generated"), "the violation must name the tier that ran: {v}");
    assert!(v.contains("Tuned"), "the violation must name the tier that was required: {v}");
    assert!(v.contains("6.1"), "the violation must name the compute capability it was judged at: {v}");
}

/// The converse, so the ratchet is not vacuously red: the same op resolved
/// to a genuinely tuned implementation satisfies the same policy.
#[test]
fn the_same_op_resolved_to_a_tuned_impl_satisfies_the_policy() {
    let reg = ProviderRegistry::reference(selector()).prefer(fixture("fixture-tuned", ImplSource::Tuned));
    let caps = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
    let req = request(&[], &[]);

    let (_, choice) = reg.resolve_choice(&req, &caps, false);
    assert_eq!(choice.source, ImplSource::Tuned);
    assert_eq!(policy::violation_in(FIXTURE_POLICY, choice.op, FIXTURE_CC, choice.source), None);
}

/// Falling back to the portable WGSL reference is *also* a violation where a
/// tuned tier was required. "It still computes the right answer" is the
/// property that makes this failure invisible, which is why the tier, not
/// the numerics, is what the gate reads.
#[test]
fn the_portable_reference_does_not_satisfy_a_tuned_requirement_either() {
    let reg = ProviderRegistry::reference(selector());
    let caps = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
    let req = request(&[], &[]);

    let (_, choice) = reg.resolve_choice(&req, &caps, false);
    assert_eq!(choice.provider, "wgsl");
    assert_eq!(choice.source, ImplSource::Reference);
    assert!(policy::violation_in(FIXTURE_POLICY, choice.op, FIXTURE_CC, choice.source).is_some());
}

/// A policy entry only applies at or above the compute capability it names,
/// and the highest applicable entry wins - the same "highest `min_cc` <=
/// device cc" resolution the kernel registry uses, so a policy cannot
/// require of an older device what only a newer one can do.
#[test]
fn a_policy_entry_does_not_apply_below_the_compute_capability_it_names() {
    const TIERED: &[PolicyEntry] = &[
        PolicyEntry { op: select::Op::MatMul, min_cc: (7, 0), required: ImplSource::Generated },
        PolicyEntry { op: select::Op::MatMul, min_cc: (8, 0), required: ImplSource::Tuned },
    ];
    assert_eq!(policy::required_in(TIERED, select::Op::MatMul, (6, 1)), None);
    assert_eq!(policy::required_in(TIERED, select::Op::MatMul, (7, 5)), Some(ImplSource::Generated));
    assert_eq!(policy::required_in(TIERED, select::Op::MatMul, (9, 0)), Some(ImplSource::Tuned));
    // Nothing is required of an op the policy says nothing about.
    assert_eq!(policy::required_in(TIERED, select::Op::RmsNorm, (9, 0)), None);
    assert_eq!(policy::violation_in(TIERED, select::Op::MatMul, (6, 1), ImplSource::Reference), None);
}

/// The datum that did not exist before: *why each provider that did not run
/// was skipped*. Without it, "the tuned provider did not run" and "the tuned
/// provider was never asked" look identical from the outside.
#[test]
fn every_skipped_provider_records_why_it_declined() {
    // SAFETY (test-only): this test binary is its own process and nothing
    // else in it reads BRAIN_NO_PROVIDER concurrently.
    unsafe {
        std::env::set_var("BRAIN_NO_PROVIDER", "fixture-disabled");
    }
    let needs_f16 = select::Requirement { f16_compute: true, ..Default::default() };
    let reg = ProviderRegistry::reference(selector())
        .prefer(Arc::new(FixtureProvider {
            name: "fixture-caps-unmet",
            source: ImplSource::Tuned,
            requires: needs_f16,
            accepts: true,
        }))
        .prefer(Arc::new(FixtureProvider {
            name: "fixture-declines",
            source: ImplSource::Tuned,
            requires: select::Requirement::default(),
            accepts: false,
        }))
        .prefer(fixture("fixture-disabled", ImplSource::Tuned));
    let caps = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
    let req = request(&[], &[]);

    let (_, choice) = reg.resolve_choice(&req, &caps, false);
    unsafe {
        std::env::remove_var("BRAIN_NO_PROVIDER");
    }

    assert_eq!(choice.provider, "wgsl", "every preferred provider declined, so the reference runs");
    let declined: Vec<(&str, DeclineReason)> =
        choice.declined.iter().map(|d| (d.provider, d.reason.clone())).collect();
    assert_eq!(
        declined,
        vec![
            ("fixture-caps-unmet", DeclineReason::CapsUnmet(needs_f16)),
            ("fixture-declines", DeclineReason::NotAccepted),
            ("fixture-disabled", DeclineReason::Disabled),
        ],
        "each skipped provider must be recorded with its OWN reason, in chain order \
         (`prefer` inserts ahead of the reference provider, so the chain is in call order)"
    );
}

/// A provider that accepted a request and then failed to lower it is the
/// worst case of all - the work still happens, on the fallback, and today
/// the only trace is a `tracing::warn` nobody has a subscriber for. It must
/// appear on the record as a decline carrying the error.
#[test]
fn a_failed_lower_is_recorded_on_the_choice_not_only_warned() {
    gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL)];
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let caps = gpu.caps();

    let (m, n, k) = (8u32, 8u32, 8u32);
    let x = gpu.storage_init("x", &vec![0.5f32; (m * k) as usize]);
    let w = gpu.storage_init("w", &vec![0.25f32; (n * k) as usize]);
    let y = gpu.storage((m * n) as u64);
    let operands = [
        Operand { role: Role::Act, buf: &x, range: (0, (m * k) as u64), dtype: DType::F32 },
        Operand { role: Role::Weight, buf: &w, range: (0, (n * k) as u64), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &y, range: (0, (m * n) as u64), dtype: DType::F32 },
    ];
    let attrs = [m, k, n];
    let req = request(&operands, &attrs);

    let reg = ProviderRegistry::reference(selector()).prefer(fixture("fixture-broken", ImplSource::Tuned));
    let mut steps = Vec::new();
    let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
    let lowered = reg.dispatch(&mut ctx, &req);

    let choice = lowered.choice.expect("every dispatch must carry its ImplChoice");
    assert_eq!(choice.provider, "wgsl", "the record must name the provider that RAN, not the one that was chosen");
    assert_eq!(choice.source, ImplSource::Reference);
    match choice.declined.iter().find(|d| d.provider == "fixture-broken").map(|d| &d.reason) {
        Some(DeclineReason::LowerFailed(msg)) => {
            assert!(msg.contains("fixture"), "the decline must carry the provider's own error: {msg}")
        }
        other => panic!("a failed lower must be recorded as a LowerFailed decline, got {other:?}"),
    }
}

/// The ratchet over the SHIPPED policy, modelled on `gpu_core::cost`'s
/// coverage ratchet: every operator the real policy declares must be
/// backed by a real entry in `kernels_cuda`'s registry at that tier, for
/// the capability the policy names. This is what stops the policy table
/// from becoming a wish list - and what stops a tuned kernel from being
/// deleted while the claim that it exists stays behind.
#[test]
fn the_shipped_policy_is_backed_by_the_shipped_cuda_registry() {
    // Measured 2026-09-13: 0. No hand-written CUDA kernel exists yet, so the
    // shipped policy honestly demands nothing of any device.
    //
    // An equality rather than `cost.rs`'s floor, on purpose: a kernel cost
    // formula lands whenever a kernel does and must not cost a test edit,
    // whereas a tier requirement is a deliberate performance contract for one
    // operator. Raise this with the contract and the kernel that satisfies
    // it; a DROP means a contract was deleted, which has to be a reviewed
    // edit rather than a side effect of deleting the kernel under it.
    const CONTRACTS: usize = 0;
    let mut backed = 0usize;
    for e in policy::POLICY {
        let k = kernels_cuda::best_for(kernels_cuda::ALL, e.op, e.min_cc).unwrap_or_else(|| {
            panic!(
                "policy requires {:?} at cc {}.{} to be {:?}, but kernels-cuda has no implementation for it",
                e.op, e.min_cc.0, e.min_cc.1, e.required
            )
        });
        assert!(
            k.source >= e.required,
            "policy requires {:?} at cc {}.{} to be {:?}, but the best kernels-cuda entry ({}) is {:?}",
            e.op,
            e.min_cc.0,
            e.min_cc.1,
            e.required,
            k.name,
            k.source
        );
        backed += 1;
    }
    println!("cuda tier policy: {backed}/{} entries backed by a real kernel", policy::POLICY.len());
    assert_eq!(
        backed, CONTRACTS,
        "the number of operators under a CUDA tier contract changed. Update CONTRACTS in the \
         same commit as the contract itself - this test is what keeps the policy table a \
         statement about what IS, not about what should be"
    );
}
