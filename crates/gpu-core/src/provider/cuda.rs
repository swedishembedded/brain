// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CudaProvider` - the [`super::OperatorProvider`] that answers an operator
//! with a **hand-written CUDA kernel** from `kernels_cuda`'s registry,
//! resolved against the compute capability the driver reported for the device
//! it is about to run on.
//!
//! Swedish Embedded AB implements architecture-specialised GPU operator
//! libraries and the dispatch machinery that keeps them honest. If your team
//! needs expertise in getting a hand-tuned kernel onto real silicon without
//! losing the portable path that proves it right, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # What decides whether this provider runs
//!
//! Three questions, all answered by the DEVICE, none by this file:
//!
//! 1. what compute capability did the driver report
//!    (`DeviceCaps::arch::compute_capability`)? That is met against each
//!    kernel's own declared floor by `kernels_cuda::best_for`, highest
//!    eligible floor first - so a card gets the most specialised kernel it
//!    can run and a card below every floor gets none.
//! 2. can the device host the kernel's launch geometry - threads per block
//!    and `__shared__` bytes, both queried capabilities? `register_native`
//!    checks this and answers `None`, which is a decline, not a failure.
//! 3. is the backend one that can compile CUDA C++ at all? Every other
//!    backend answers `None` to `register_native` for a
//!    `NativeSpec::Cuda`, so this provider declines there by the same
//!    mechanism rather than by asking what backend it is on.
//!
//! Nothing in this module names a card, an architecture or a capability
//! value. [`CudaProvider::new`] takes the capability as an argument for the
//! same reason the tier policy does: a value that is queried cannot be a
//! value that is assumed.
//!
//! # Scope
//!
//! `Op::MatMul`, forward, f32, `[Act, Weight, Out]` - the operand bundle
//! `model::ops::Ops::matmul`'s own F32 arm builds, with the same uniform
//! (`[m, k, n]`) the portable `matmul.wgsl` reads. Anything else (a
//! quantized weight tier, a backward pass, another operator) is declined by
//! [`CudaProvider::accepts`] and served by the WGSL reference provider
//! exactly as before - a decline, never a forced failure and never a
//! silently different answer.

use std::sync::Mutex;

use backend_api::select::{self, Dtype};
use backend_api::{BindKind, DType, ImplSource, NativeId, NativeSpec};
use kernels_cuda::{CudaKernel, Cc};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Pass, Role};

/// How one hand-written kernel binds its arguments: the uniform first, then
/// the storage pointers in the order the reference kernel expects them.
/// Mirrors `matmul.wgsl`'s own `@binding` order exactly, which is what lets
/// the same [`super::OpRequest`] feed either provider.
const MATMUL_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// Whether the device has taken this kernel, cached per provider instance -
/// the same one-provider-per-device assumption `coopmat::CoopMatProvider` and
/// `select::CachedSelector` already make (one `ProviderRegistry` is built per
/// `Ops`/`Gpu`, never shared across unrelated devices).
#[derive(Clone, Copy)]
enum Registration {
    Unattempted,
    /// The backend cannot compile CUDA C++, cannot host this kernel's launch
    /// geometry, or has no NVRTC. An expected outcome, not a fault.
    Declined,
    Registered(NativeId),
}

pub struct CudaProvider {
    /// The capability the DRIVER reported for the device this provider was
    /// built for. Never a default: a provider with no capability to resolve
    /// against cannot be constructed (see [`CudaProvider::for_gpu`]).
    cc: Cc,
    matmul: Mutex<Registration>,
}

impl CudaProvider {
    /// A provider that resolves kernels against compute capability `cc`.
    pub fn new(cc: Cc) -> CudaProvider {
        CudaProvider { cc, matmul: Mutex::new(Registration::Unattempted) }
    }

    /// The provider for `gpu`, or `None` when this handle's device reports no
    /// compute capability at all.
    ///
    /// "Reports none" is the honest gate, rather than "is not the CUDA
    /// backend": the capability is a queried device fact, and a backend that
    /// does not publish one has nothing for `kernels_cuda::best_for` to
    /// resolve against. A handle that publishes one but cannot compile CUDA
    /// C++ still declines - one layer down, at `register_native` - so no
    /// dispatch can reach a kernel the device never took.
    pub fn for_gpu(gpu: &crate::Gpu) -> Option<CudaProvider> {
        gpu.caps().arch.compute_capability.map(CudaProvider::new)
    }

    /// The compute capability this provider resolves against.
    pub fn compute_capability(&self) -> Cc {
        self.cc
    }

    /// The registry entry this provider would run for `op` on its device, or
    /// `None` when the registry offers nothing for that operator at this
    /// capability.
    pub fn kernel(&self, op: select::Op) -> Option<&'static CudaKernel> {
        kernels_cuda::find(op, self.cc)
    }

    /// [`Self::kernel`]'s registry name - what a test or a diagnostic reports
    /// without having to reach into the registry itself.
    pub fn kernel_name(&self, op: select::Op) -> Option<&'static str> {
        self.kernel(op).map(|k| k.name)
    }

    /// Register the matmul kernel on `gpu`'s backend if this is the first
    /// call, else reuse the cached id/decline.
    fn matmul_id(&self, gpu: &crate::Gpu) -> Option<NativeId> {
        let mut reg = self.matmul.lock().unwrap_or_else(|e| e.into_inner());
        match *reg {
            Registration::Registered(id) => Some(id),
            Registration::Declined => None,
            Registration::Unattempted => {
                let k = self.kernel(select::Op::MatMul)?;
                let id = gpu.register_native(&NativeSpec::Cuda {
                    src: k.src,
                    entry: k.entry,
                    block_dim: k.block_dim,
                    bindings: MATMUL_BINDINGS,
                    shared_bytes: k.shared_bytes,
                });
                *reg = match id {
                    Some(id) => Registration::Registered(id),
                    None => Registration::Declined,
                };
                id
            }
        }
    }

    /// The operand bundle this provider's matmul kernel reads: exactly
    /// `[Act, Weight, Out]`, all f32.
    ///
    /// Checked structurally rather than trusted: the request's `shape.dtype`
    /// names the WEIGHT tier, and a bundle for a quantized tier carries five
    /// operands with two scale planes in the middle. Binding those to a
    /// three-pointer kernel would read scales as weights - a wrong answer, not
    /// a crash.
    fn is_plain_f32_matmul(req: &OpRequest) -> bool {
        req.shape.dtype == Dtype::F32
            && req.operands.len() == 3
            && req.operands[0].role == Role::Act
            && req.operands[1].role == Role::Weight
            && req.operands[2].role == Role::Out
            && req.operands.iter().all(|o| o.dtype == DType::F32)
    }
}

impl OperatorProvider for CudaProvider {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn source(&self, _req: &OpRequest) -> ImplSource {
        // Hand-written CUDA C++ selected against a queried capability. The
        // tier is a property of who wrote the kernel, not of how fast it
        // turned out to be - which is why the speedup is asserted by a test
        // rather than implied by this answer.
        ImplSource::Tuned
    }

    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        // Nothing `select::Requirement` can express. What this provider
        // actually needs - a backend that compiles CUDA C++, a device whose
        // reported threads-per-block and shared-memory limits fit the
        // kernel - is asked of the backend itself at `register_native`, which
        // is the only place those answers exist. Claiming a requirement here
        // that does not gate anything would be worse than claiming none: a
        // decline would then be attributed to the wrong cause in the
        // `ImplChoice` record.
        select::Requirement::default()
    }

    fn accepts(&self, req: &OpRequest, _capture: bool) -> bool {
        // `capture` never disqualifies: this provider pushes an ordinary
        // `Step` onto the caller's tape, replayable exactly like a WGSL one.
        req.op == select::Op::MatMul
            && req.pass == Pass::Forward
            && Self::is_plain_f32_matmul(req)
            && self.kernel(select::Op::MatMul).is_some()
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        if req.op != select::Op::MatMul || !Self::is_plain_f32_matmul(req) {
            return Err(format!(
                "cuda::CudaProvider::lower: {:?} at dtype {:?} is not implemented (plain f32 MatMul only)",
                req.op, req.shape.dtype
            ));
        }
        let k = self
            .kernel(select::Op::MatMul)
            .ok_or_else(|| format!("cuda::CudaProvider::lower: no native MatMul kernel at compute capability {}.{}", self.cc.0, self.cc.1))?;
        let Some(id) = self.matmul_id(ctx.gpu) else {
            return Err(
                "cuda::CudaProvider::lower: this device declined the native matmul kernel \
                 (register_native returned None) - falling back to the WGSL reference provider"
                    .to_string(),
            );
        };

        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let offsets: Vec<(u64, u64)> = req.operands.iter().map(|o| o.range).collect();
        // The uniform is the reference kernel's own `[m, k, n]`, passed
        // through untouched - the hand-written kernel reads the identical
        // layout on purpose, so there is no second place a shape could be
        // spelled differently.
        let blocks = k.blocks_for(req.shape.m, req.shape.n);
        let step = ctx
            .gpu
            .step_native_sliced(id, &bufs, &offsets, req.attrs, blocks)
            .ok_or_else(|| {
                "cuda::CudaProvider::lower: step_native_sliced declined a NativeId this provider \
                 itself registered - the backend has no native slicing path"
                    .to_string()
            })?;
        ctx.steps.push(step);
        // Deliberately not spelled like a WGSL catalogue name: this is a
        // kernel from a different registry, and a bare `[a-z0-9_]+` string
        // here would look to a catalogue-cross-referencing gate like an
        // unregistered WGSL kernel. Same convention `coopmat` uses.
        Ok(Lowered::new(1, vec![k.reported]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Operand;
    use backend_api::DeviceBuffer;

    fn shape(m: u32, n: u32, k: u32, dt: Dtype) -> select::OpShape {
        select::OpShape { m, n, k, dtype: dt }
    }

    fn req<'a>(pass: Pass, operands: &'a [Operand<'a>], shape: select::OpShape) -> OpRequest<'a> {
        OpRequest {
            op: select::Op::MatMul,
            shape,
            pass,
            operands,
            attrs: &[],
            bind: &|_| panic!("the CUDA provider never calls OpRequest::bind - it has no WGSL name table"),
        }
    }

    fn f32_bundle<'a>(b: &'a DeviceBuffer) -> [Operand<'a>; 3] {
        [
            Operand { role: Role::Act, buf: b, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: b, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: b, range: (0, 1), dtype: DType::F32 },
        ]
    }

    /// A capability below every declared floor resolves to no kernel at all,
    /// and a provider with no kernel accepts nothing - it never falls through
    /// to "the closest thing available". Stated at a capability no device
    /// here has, which is the point: the rule has to be right for hardware
    /// this code cannot be run against.
    #[test]
    fn a_capability_below_every_floor_offers_nothing_and_accepts_nothing() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let x = gpu.storage(1);
        let operands = f32_bundle(&x);
        let below = CudaProvider::new((1, 0));
        assert!(below.kernel(select::Op::MatMul).is_none());
        assert!(!below.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32)), false));
    }

    /// A quantized weight tier binds FIVE operands with two scale planes; a
    /// three-pointer f32 kernel would read a scale plane as weights. The
    /// refusal is structural, on the operand bundle, not merely on the
    /// declared dtype.
    #[test]
    fn a_quantized_or_backward_request_is_declined() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let x = gpu.storage(1);
        // A capability above every declared floor, so the decline below can
        // only come from the request and never from kernel resolution.
        let p = CudaProvider::new((99, 0));
        assert!(p.kernel(select::Op::MatMul).is_some());

        let operands = f32_bundle(&x);
        assert!(p.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32)), false));
        assert!(!p.accepts(&req(Pass::Backward, &operands, shape(64, 64, 64, Dtype::F32)), false));
        assert!(!p.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::I8)), false));

        let quantized = [
            Operand { role: Role::Act, buf: &x, range: (0, 1), dtype: DType::I8 },
            Operand { role: Role::Weight, buf: &x, range: (0, 0), dtype: DType::I8 },
            Operand { role: Role::ActScale, buf: &x, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::WeightScale, buf: &x, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &x, range: (0, 1), dtype: DType::F32 },
        ];
        assert!(!p.accepts(&req(Pass::Forward, &quantized, shape(64, 64, 64, Dtype::I8)), false));
    }

    /// On a backend that cannot compile CUDA C++ (every backend but the CUDA
    /// one), `lower` returns `Err` rather than dispatching something else -
    /// which is what makes `ProviderRegistry::dispatch` record the decline
    /// and fall back to the reference provider. Run against the pooled test
    /// device, whatever backend that is.
    #[test]
    fn a_backend_that_cannot_compile_cuda_declines_at_lower() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        if gpu.kind() == "cuda" {
            return; // this box's test device IS the CUDA backend; nothing to decline
        }
        let x = gpu.storage(64 * 64);
        let operands = f32_bundle(&x);
        let p = CudaProvider::new((99, 0));
        let caps = gpu.caps();
        let mut steps = Vec::new();
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        assert!(p.lower(&mut ctx, &req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32))).is_err());
        assert!(steps.is_empty(), "a declining provider must push no Step");
    }
}
