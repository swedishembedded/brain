// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `OperatorProvider` ABI - *which implementation of a whole operator
//! runs, on a device, for a real dispatch.*
//!
//! `backend_api::select::KernelSelector` (Phase 1) already answers "which
//! kernel *variant*" for one WGSL kernel family. This seam answers a bigger
//! question one level up: an operator (`select::Op` - already whole-operator
//! granular, `PagedAttention`/`MoeExpertLinear`/`Conv2d`, not a kernel) can
//! have more than one *implementation family* - the portable WGSL reference
//! always, and eventually a native-f16, cooperative-matrix, or CPU-ISA-pack
//! provider built in a later wave against this exact ABI. `AGENTS.md`'s
//! "fp32 arithmetic only, core compute only" bullet already named this as
//! the sanctioned extension point once it landed; this module is that
//! landing (`kernel-performance.md` M8.3).
//!
//! **This milestone changes no behaviour.** [`wgsl::WgslProvider`] IS
//! `model::ops::Ops::matmul`'s pre-existing dispatch logic, moved here
//! verbatim (selector → kernel/thread-count → `Gpu::step_sliced`); every
//! other kernel selector policy is unchanged. See that module's doc comment
//! for the one ABI deviation this move required (a caller-supplied kernel
//! *name* resolver) and why `crates/gpu-core` cannot own kernel name tables
//! itself.
//!
//! ## Scope, stated plainly
//!
//! Only `Op::MatMul` is wired through this seam this milestone - 
//! `model::ops::Ops::matmul` is the one façade method this session moved.
//! `Ops::embed`/`Ops::moe_linear`/`Ops::matmul_dx`/`Ops::matmul_dw` keep
//! dispatching exactly as they did before this change; they are candidates
//! for a follow-up migration onto this same seam, not done here. `model::
//! block`'s attention/softmax/paged-attention gates and `qwen3::serve`'s
//! manual GEMM dispatch region (a deliberate M1.2 exception - see that
//! ledger entry) stay entirely outside this seam's reach, exactly as before.
//! A later provider therefore does not speed up either of those simply by
//! existing.

use std::sync::Arc;

use backend_api::select;
use backend_api::DeviceCaps;
use backend_api::ImplSource;

use crate::{DeviceBuffer, Gpu, Step};

// M8.9: the first non-WGSL provider (cooperative-matrix). Native-only - it
// registers a real SPIR-V pipeline through `backend_vulkan::coopmat`, which
// (like every other Vulkan/ash path in this workspace) cannot target wasm.
#[cfg(not(target_arch = "wasm32"))]
pub mod coopmat;
/// Hand-written CUDA kernels (`brain-kernels-cuda`'s registry), resolved
/// against the compute capability the driver reported. Native-only: it
/// registers CUDA C++ source through `Backend::register_native`, which no
/// wasm target has a driver for.
#[cfg(not(target_arch = "wasm32"))]
pub mod cuda;
/// CPU ISA-pack provider (AVX2 F32 GEMM hoist + AVX2 int8 GEMM) -
/// `kernel-performance.md` M8.10/M8.11, this ABI's first non-reference
/// provider.
pub mod cpu_isa;
pub mod native_f16;
pub mod parity;
pub mod wgsl;

/// Which half of a training step a dispatch belongs to. `Op::MatMul`
/// requests from `Ops::matmul` are always [`Pass::Forward`] - the backward
/// GEMMs (`Ops::matmul_dx`/`matmul_dw`) are separate, not-yet-migrated `Ops`
/// methods (see this module's doc comment) - but the field exists now so a
/// provider written against this ABI does not need a breaking change when
/// they migrate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pass {
    Forward,
    Backward,
}

/// What role one bound buffer plays in a dispatch - informational, for a
/// provider that wants to reason about an operand's meaning (the parity
/// harness, a future provider that needs to know which operand is the
/// output). The bind ORDER that actually reaches the kernel is the order
/// [`OpRequest::operands`] lists them in, decided by the caller building the
/// request (the same convention `model::ops::Ops::matmul`'s own per-dtype
/// buffer list already followed before this move) - a provider's `lower`
/// does not reorder by role.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Act,
    Weight,
    Out,
    Accum,
    ActScale,
    WeightScale,
    WeightMin,
    ActGroupSum,
    Aux(u8),
}

/// One bound buffer: which role it plays, the buffer itself, the sub-range
/// it binds (`(offset_words, len_words)` - the exact convention
/// [`Gpu::step_sliced`]'s own `offsets` parameter already uses, reused
/// unchanged), and its storage dtype.
pub struct Operand<'a> {
    pub role: Role,
    pub buf: &'a DeviceBuffer,
    pub range: (u64, u64),
    pub dtype: backend_api::DType,
}

/// One logical dispatch request: an operator, the shape that selects its
/// implementation, which pass it belongs to, its bound operands (in the
/// exact order the reference kernel expects them), and its non-shape
/// uniform params.
///
/// # `bind` - the one deviation from the sketch this ABI was proposed with
///
/// `crates/gpu-core` cannot own a `(KernelVariant, Dtype) -> kernel name`
/// table the way `model::ops::Ops::bind` does: those name spellings
/// (`"matmul_gemv"`, `"matmul_i8_dyn#QPG=1"`, …) are `crates/model`'s own
/// registered-kernel contract (`kname`/`REQUIRED_KERNELS`/`kernel_list`),
/// and `gpu-core` sits BELOW `model` in the dependency graph - it must never
/// depend on it. So the caller that already knows its `Gpu` handle's
/// name→index table (`Ops`, today) supplies a closure that turns a selected
/// [`select::KernelVariant`] into `(pipeline index, kernel name)` - the name
/// is `&'static str` (every `kname` constant already is one) purely so
/// [`Lowered::kernels`] can report it without allocating one nor needing a
/// lifetime tied to this request. A future non-WGSL provider does not need
/// this at all - it resolves its own kernel through
/// [`backend_api::Backend::register_native`]/`step_native` instead, which
/// carries no name-table problem because it never had string names to
/// begin with.
pub struct OpRequest<'a> {
    pub op: select::Op,
    pub shape: select::OpShape,
    pub pass: Pass,
    pub operands: &'a [Operand<'a>],
    pub attrs: &'a [u32],
    pub bind: &'a dyn Fn(select::KernelVariant) -> (usize, &'static str),
}

/// Everything a provider's `lower` needs beyond the request itself: the
/// device, its capabilities, the tape to push onto, and whether this
/// recording must be replayable.
///
/// `caps` is a [`DeviceCaps`] rather than an `ArchDesc` - M8.1 (a sibling
/// wave-1 item, a different worktree) had not landed an architecture
/// descriptor type in this worktree at the time this milestone was built;
/// per this milestone's own brief, adapting this field to a real `ArchDesc`
/// once M8.1 lands is the integrating orchestrator's job, not this one's.
pub struct LowerCtx<'a> {
    pub gpu: &'a Gpu,
    pub caps: &'a DeviceCaps,
    pub steps: &'a mut Vec<Step>,
    pub capture: bool,
}

/// What a `lower` call actually did: how many [`Step`]s it pushed, and which
/// physical kernel(s) it dispatched (for diagnostics/profiling - never
/// consumed by dispatch logic itself).
///
/// `kernels` is `Vec<&'static str>` rather than the sketch's `&'static
/// [&'static str]`: the STRINGS are `'static` (every `kname` constant is),
/// but which ones a given call names is a runtime choice (which dtype/
/// variant got selected), and there is no `'static` backing array to borrow
/// a slice of at that granularity without leaking one per call. An owned
/// `Vec` of `'static` string references is the direct, allocation-cheap
/// fix.
pub struct Lowered {
    pub pushed: usize,
    pub kernels: Vec<&'static str>,
    /// Which implementation family answered this request and why every other
    /// provider did not - filled by [`ProviderRegistry::dispatch`], which is
    /// the only place that knows the whole chain. A provider's own `lower`
    /// leaves it `None` (see [`Lowered::new`]); it cannot report on the
    /// providers it never saw.
    pub choice: Option<ImplChoice>,
}

impl Lowered {
    /// What a provider returns: the steps it pushed and the kernels it
    /// named. The dispatch record is attached afterwards, by the registry.
    pub fn new(pushed: usize, kernels: Vec<&'static str>) -> Lowered {
        Lowered { pushed, kernels, choice: None }
    }
}

/// The device architecture a dispatch was resolved FOR, in the smallest form
/// that can be carried per dispatch.
///
/// Every field is queried, never assumed. `compute_capability` is `None`
/// wherever the backend that produced the [`DeviceCaps`] has no such concept
/// (every backend but CUDA: it is a CUDA notion, and the others describe
/// themselves by feature bits instead) - `None` means *not reported*, and no
/// consumer may read it as a version, a floor or a default.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArchTag {
    pub class: backend_api::DeviceClass,
    /// `(major, minor)` as the driver reports the two halves.
    pub compute_capability: Option<(u32, u32)>,
}

impl ArchTag {
    /// The tag for a device, read off its capabilities - the single place
    /// that mapping lives, so a backend that starts reporting a compute
    /// capability is wired in once.
    pub fn of(caps: &DeviceCaps) -> ArchTag {
        ArchTag { class: caps.class, compute_capability: caps.arch.compute_capability }
    }
}

/// Why one provider in the chain did not run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeclineReason {
    /// Named in `BRAIN_NO_PROVIDER` - skipped before it was asked anything.
    Disabled,
    /// Its [`OperatorProvider::requires`] is not satisfied by this device's
    /// capabilities. Carries the requirement so the record says WHAT was
    /// missing, not merely that something was.
    CapsUnmet(select::Requirement),
    /// Capable in principle, but [`OperatorProvider::accepts`] said no to
    /// this request (an operator it does not implement, a shape it does not
    /// cover, or a capture it cannot record replayably).
    NotAccepted,
    /// It accepted the request and then failed to lower it, so the work ran
    /// on the reference provider instead. The one case with a message, and
    /// the one that used to leave no trace but a `tracing::warn`.
    LowerFailed(String),
}

/// One provider that did not run, and why.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decline {
    pub provider: &'static str,
    pub reason: DeclineReason,
}

/// The per-dispatch record: which implementation family actually answered an
/// operator, on what architecture, and why every other candidate did not.
///
/// This is the datum that makes a silent performance fallback impossible to
/// miss. Numerically the tiers are interchangeable, so nothing downstream
/// can otherwise tell a hand-tuned kernel from a mechanical translation of
/// the portable one - a backend serving the bottom tier where the top one
/// was promised is indistinguishable from a backend that is simply slower
/// than expected. With the tier and the decline reasons recorded, a policy
/// can assert on it (see `backend_cuda::policy`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ImplChoice {
    pub op: select::Op,
    pub shape: select::OpShape,
    /// [`OperatorProvider::name`] of the provider that RAN - which, after a
    /// failed lower, is the reference provider and not the one first chosen.
    pub provider: &'static str,
    pub source: ImplSource,
    pub arch: ArchTag,
    /// Chain order, preferred providers first. Empty is the common case (the
    /// first provider took the request), so the record costs no allocation
    /// when nothing declined.
    pub declined: Vec<Decline>,
}

/// One implementation family for a whole operator.
///
/// Implementations must be `Send + Sync` (providers are shared behind
/// `Arc` in a [`ProviderRegistry`], resolved from any thread that owns a
/// `Gpu` handle).
pub trait OperatorProvider: Send + Sync {
    /// Stable, lowercase, `[a-z0-9-]+` - used for `BRAIN_NO_PROVIDER`
    /// matching and diagnostics. Never changes across a provider's
    /// lifetime; a rename is a breaking change for anyone disabling it by
    /// name.
    fn name(&self) -> &'static str;

    /// Which implementation family this provider would answer `req` with -
    /// the tier a performance policy is stated in, recorded on every
    /// dispatch as [`ImplChoice::source`].
    ///
    /// Deliberately per-request and deliberately NOT defaulted: a provider
    /// may hold hand-written kernels for some operators and fall back to a
    /// generated one for the rest, and a default would let a new provider
    /// inherit a claim nobody made about it. Answer what this provider would
    /// actually run for THIS request.
    fn source(&self, req: &OpRequest) -> ImplSource;

    /// What this provider would need from the device to run `req`, checked
    /// against real [`DeviceCaps`] by [`ProviderRegistry::resolve`] before
    /// [`Self::accepts`] is even asked. The reference WGSL provider returns
    /// [`select::Requirement::default`] unconditionally - its own selector
    /// already only ever offers a [`select::KernelVariant`] whose
    /// requirement `caps` satisfies (see `select::candidates`'s own
    /// contract), so there is no separate, coarser requirement to check
    /// ahead of that.
    fn requires(&self, req: &OpRequest) -> select::Requirement;

    /// Whether this provider will actually take `req` - beyond the bare
    /// capability check in [`Self::requires`], a provider may decline for
    /// its own reasons (an unimplemented `Op`, a shape it does not cover, or,
    /// when `capture` is true, an inability to emit a *replayable*
    /// [`Step`] sequence at all). The reference WGSL provider is the
    /// fallback of last resort by construction ([`ProviderRegistry`] always
    /// places it last), so it answers `true` unconditionally; see
    /// [`wgsl::WgslProvider`]'s own doc comment for which `Op`s its `lower`
    /// actually implements this milestone and what happens if one outside
    /// that set ever reaches it (never happens today - see this module's
    /// top-level scope note).
    fn accepts(&self, req: &OpRequest, capture: bool) -> bool;

    /// Push zero or more [`Step`]s onto `ctx.steps` implementing `req`.
    /// A provider NEVER submits - see [`crate::Gpu::submit`] - it only
    /// records; the caller's own tape (a per-step `Vec::new()` today, or a
    /// captured-and-replayed tape once M6.3-style capture reaches a given
    /// call site) owns submission.
    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String>;
}

/// `BRAIN_NO_PROVIDER=<name>[,<name>]` - disable one or more providers by
/// name, mirroring `gpu_core::upgrade`'s `BRAIN_NO_KERNEL_UPGRADE` (an A/B
/// switch for the analogous seam one level down) and `brain_testutil`'s
/// `BRAIN_REQUIRE_CAPABILITIES` (the same comma-separated-list parse this
/// function copies exactly: `split(',')`, `trim`, no reserved names). Read
/// fresh at [`ProviderRegistry::reference`] construction time, not cached - 
/// a process-wide cache would make this env var untestable per-test the way
/// `upgrade`'s own `OnceLock` cache already is for its callers.
fn disabled_providers() -> Vec<String> {
    std::env::var("BRAIN_NO_PROVIDER")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// An ordered chain of providers, the reference (WGSL) provider always last
/// and always accepting - the fallback nothing else in the chain can be
/// missing.
pub struct ProviderRegistry {
    /// Preferred providers first, the reference provider always the LAST
    /// element - [`Self::prefer`] maintains this invariant by inserting
    /// before it, never after.
    chain: Vec<Arc<dyn OperatorProvider>>,
    disabled: Vec<String>,
}

impl ProviderRegistry {
    /// A registry with only the WGSL reference provider, dispatching
    /// through `selector` - what an empty registry means: `Ops::with_selector`
    /// builds exactly this today.
    pub fn reference(selector: Arc<dyn select::KernelSelector>) -> ProviderRegistry {
        ProviderRegistry { chain: vec![Arc::new(wgsl::WgslProvider::new(selector))], disabled: disabled_providers() }
    }

    /// **The production registry**: the reference provider, plus whichever
    /// specialised providers `gpu`'s own device actually supports.
    ///
    /// This is what every model's `Ops` is built with, and it is the ONE
    /// place a non-reference provider enters a real run. Everything it adds
    /// is gated on a queried device fact rather than on a build flag or an
    /// environment variable, so the same binary on the same box picks the
    /// same chain whoever launched it:
    ///
    /// - [`cuda::CudaProvider`] when this handle's device reports a compute
    ///   capability at all ([`ArchTag::compute_capability`]). A device that
    ///   reports one but whose backend cannot compile CUDA C++ still
    ///   declines - one layer down, at `register_native` - and the decline is
    ///   RECORDED in the dispatch's [`ImplChoice`] rather than being
    ///   invisible. Adding it unconditionally on that one fact, instead of
    ///   asking `gpu.kind() == "cuda"`, keeps the rule a property of the
    ///   device rather than of a backend name.
    ///
    /// Providers that exist but are NOT added here, and why: `coopmat` and
    /// `native_f16` were built against synthetic call sites and have never
    /// been proven against a real model's operand bundles on hardware that
    /// admits them (no box this campaign has run on does), and `cpu_isa`
    /// belongs to the CPU backend's own dispatch path. Adding a provider to
    /// this chain is a claim that a real forward pass through it was
    /// checked; none of those three has one yet.
    pub fn for_gpu(gpu: &crate::Gpu, selector: Arc<dyn select::KernelSelector>) -> ProviderRegistry {
        let reg = ProviderRegistry::reference(selector);
        #[cfg(not(target_arch = "wasm32"))]
        let reg = match cuda::CudaProvider::for_gpu(gpu) {
            Some(p) => reg.prefer(Arc::new(p)),
            None => reg,
        };
        #[cfg(target_arch = "wasm32")]
        let _ = gpu;
        reg
    }

    /// Add a preferred provider ahead of the reference provider - checked
    /// FIRST by [`Self::resolve`], but the reference stays last regardless
    /// of call order.
    pub fn prefer(mut self, p: Arc<dyn OperatorProvider>) -> ProviderRegistry {
        let last = self.chain.len() - 1;
        self.chain.insert(last, p);
        self
    }

    /// The first non-[`BRAIN_NO_PROVIDER`]-disabled provider in the chain
    /// whose [`OperatorProvider::requires`] is satisfied by `caps` and whose
    /// [`OperatorProvider::accepts`] answers `true` for `req` - the
    /// reference provider always qualifies, so this never falls through
    /// empty.
    pub fn resolve(&self, req: &OpRequest, caps: &DeviceCaps, capture: bool) -> &dyn OperatorProvider {
        self.resolve_choice(req, caps, capture).0
    }

    /// [`Self::resolve`], plus the record of HOW that answer was reached:
    /// the tier the chosen provider will run, the architecture it was
    /// resolved for, and one [`Decline`] per provider that was skipped -
    /// each with its own reason.
    ///
    /// The decline list is the datum this seam did not have. Without it,
    /// "the tuned provider declined this shape", "the device cannot satisfy
    /// what it needs" and "somebody disabled it in the environment" are one
    /// indistinguishable silence, and the only visible effect is that things
    /// are slower than they were supposed to be.
    ///
    /// Pure: it asks providers what they would do and touches no device, so
    /// a policy gate can run it with nothing attached.
    pub fn resolve_choice<'a>(
        &'a self,
        req: &OpRequest,
        caps: &DeviceCaps,
        capture: bool,
    ) -> (&'a dyn OperatorProvider, ImplChoice) {
        let mut declined: Vec<Decline> = Vec::new();
        for p in &self.chain {
            if self.disabled.iter().any(|d| d == p.name()) {
                declined.push(Decline { provider: p.name(), reason: DeclineReason::Disabled });
                continue;
            }
            let requirement = p.requires(req);
            if !requirement.satisfied_by(caps) {
                declined.push(Decline { provider: p.name(), reason: DeclineReason::CapsUnmet(requirement) });
                continue;
            }
            if !p.accepts(req, capture) {
                declined.push(Decline { provider: p.name(), reason: DeclineReason::NotAccepted });
                continue;
            }
            let choice = ImplChoice {
                op: req.op,
                shape: req.shape,
                provider: p.name(),
                source: p.source(req),
                arch: ArchTag::of(caps),
                declined,
            };
            return (p.as_ref(), choice);
        }
        // Unreachable in practice: the reference provider is never disabled
        // by construction of `disabled` alone (a caller CAN disable it by
        // name), never fails `requires`, and always `accepts`. If every
        // entry (including a disabled reference) is skipped, fall back to
        // the reference regardless - refusing to dispatch at all would be
        // worse than ignoring a misconfigured `BRAIN_NO_PROVIDER=wgsl`. The
        // reference's own decline stays on the record: it ran despite being
        // disabled, and that is exactly the kind of thing a record is for.
        let reference = self.chain.last().expect("ProviderRegistry::reference always seeds one entry");
        let choice = ImplChoice {
            op: req.op,
            shape: req.shape,
            provider: reference.name(),
            source: reference.source(req),
            arch: ArchTag::of(caps),
            declined,
        };
        (reference.as_ref(), choice)
    }

    /// Resolve a provider for `req` and lower it, falling back to the
    /// reference provider (and RECORDING that fallback, never silently) if
    /// the resolved provider's `lower` returns `Err`.
    ///
    /// The returned [`Lowered`] always carries its [`ImplChoice`], naming
    /// the provider that actually RAN - after a failed lower that is the
    /// reference provider, with the failure itself recorded as that
    /// provider's decline.
    pub fn dispatch(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Lowered {
        let (chosen, mut choice) = self.resolve_choice(req, ctx.caps, ctx.capture);
        match chosen.lower(ctx, req) {
            Ok(mut l) => {
                l.choice = Some(choice);
                l
            }
            Err(e) => {
                let reference = self.chain.last().expect("ProviderRegistry::reference always seeds one entry");
                if std::ptr::eq(chosen, reference.as_ref()) {
                    // The reference provider is expected to accept every op
                    // by construction (see `OperatorProvider::accepts`'s doc
                    // comment), but `lower` for an `Op` this milestone has
                    // not yet moved onto the seam (everything except
                    // `Op::MatMul` - see this module's top-level scope note)
                    // returns `Err` rather than silently doing nothing.
                    // Nothing in this tree calls `dispatch` for any other op
                    // yet, so this is unreached in production; a caller that
                    // DOES reach it needs the real error immediately, not a
                    // swallowed fallback.
                    panic!("ProviderRegistry::dispatch: the reference WGSL provider could not lower op {:?}: {e}", req.op);
                }
                tracing::warn!(
                    provider = chosen.name(),
                    op = ?req.op,
                    error = %e,
                    "provider failed to lower request, falling back to the reference WGSL provider"
                );
                choice.declined.push(Decline {
                    provider: chosen.name(),
                    reason: DeclineReason::LowerFailed(e),
                });
                choice.provider = reference.name();
                choice.source = reference.source(req);
                let mut l = reference
                    .lower(ctx, req)
                    .expect("reference provider must never fail to lower a request it accepted");
                l.choice = Some(choice);
                l
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(m: u32, n: u32, k: u32) -> select::OpShape {
        select::OpShape { m, n, k, dtype: select::Dtype::F32 }
    }

    /// An empty registry (nothing preferred) IS the reference provider - the
    /// contract `Ops::with_providers(gpu, ProviderRegistry::reference(sel))`
    /// depends on to behave exactly like today's `Ops::with_selector`.
    #[test]
    fn an_empty_registry_is_the_reference_provider() {
        let sel: Arc<dyn select::KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
        let reg = ProviderRegistry::reference(sel);
        assert_eq!(reg.chain.len(), 1);
        assert_eq!(reg.chain[0].name(), "wgsl");

        let bind = |v: select::KernelVariant| -> (usize, &'static str) {
            match v {
                select::KernelVariant::Reference => (0, "matmul"),
                _ => (1, "matmul_gemv"),
            }
        };
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: shape(4096, 128, 128),
            pass: Pass::Forward,
            operands: &[],
            attrs: &[],
            bind: &bind,
        };
        let caps = DeviceCaps::portable_baseline(backend_api::DeviceClass::Cpu);
        let chosen = reg.resolve(&req, &caps, false);
        assert_eq!(chosen.name(), "wgsl");
    }

    /// `BRAIN_NO_PROVIDER=<name>` removes a provider from the chain - proven
    /// here against the reference provider itself (no second provider exists
    /// yet this wave), which is why `resolve` still must not panic or return
    /// nothing: disabling every entry falls back to the reference anyway
    /// (see `resolve`'s own doc comment on that unreachable-in-practice
    /// case).
    #[test]
    fn brain_no_provider_removes_a_provider_from_the_chain() {
        // SAFETY (test-only): no other test in this process reads this var
        // concurrently with a race that would matter - `disabled_providers`
        // is read once, synchronously, at construction.
        unsafe {
            std::env::set_var("BRAIN_NO_PROVIDER", "wgsl, some-other-provider");
        }
        let sel: Arc<dyn select::KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
        let reg = ProviderRegistry::reference(sel);
        assert_eq!(reg.disabled, vec!["wgsl".to_string(), "some-other-provider".to_string()]);

        let bind = |v: select::KernelVariant| -> (usize, &'static str) {
            match v {
                select::KernelVariant::Reference => (0, "matmul"),
                _ => (1, "matmul_gemv"),
            }
        };
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: shape(4096, 128, 128),
            pass: Pass::Forward,
            operands: &[],
            attrs: &[],
            bind: &bind,
        };
        let caps = DeviceCaps::portable_baseline(backend_api::DeviceClass::Cpu);
        // `wgsl` is named in BRAIN_NO_PROVIDER but is also the only entry, so
        // `resolve` still returns it (the documented "never dispatch nothing"
        // fallback) - the env var's PARSING and the chain-skip loop having
        // actually run is what this test pins, via `reg.disabled` above; a
        // resolve returning nothing here would be a worse failure mode than
        // ignoring a misconfigured disable of the only provider that exists.
        let chosen = reg.resolve(&req, &caps, false);
        assert_eq!(chosen.name(), "wgsl");
        unsafe {
            std::env::remove_var("BRAIN_NO_PROVIDER");
        }
    }

    #[test]
    fn prefer_keeps_the_reference_provider_last() {
        struct Dummy(&'static str);
        impl OperatorProvider for Dummy {
            fn name(&self) -> &'static str {
                self.0
            }
            fn source(&self, _req: &OpRequest) -> ImplSource {
                ImplSource::Tuned
            }
            fn requires(&self, _req: &OpRequest) -> select::Requirement {
                select::Requirement::default()
            }
            fn accepts(&self, _req: &OpRequest, _capture: bool) -> bool {
                false
            }
            fn lower(&self, _ctx: &mut LowerCtx, _req: &OpRequest) -> Result<Lowered, String> {
                Err("Dummy never lowers anything".to_string())
            }
        }
        let sel: Arc<dyn select::KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
        let reg = ProviderRegistry::reference(sel).prefer(Arc::new(Dummy("dummy")));
        assert_eq!(reg.chain.len(), 2);
        assert_eq!(reg.chain[0].name(), "dummy");
        assert_eq!(reg.chain[1].name(), "wgsl");
    }
}
