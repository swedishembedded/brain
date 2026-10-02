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
//! Two requests, both `Op::MatMul`, forward only:
//!
//! * plain f32, `[Act, Weight, Out]` - the operand bundle
//!   `model::ops::Ops::matmul`'s own F32 arm builds, with the same uniform
//!   (`[m, k, n]`) the portable `matmul.wgsl` reads;
//! * the int8 dynamic-activation GEMM, `[Act, Weight, ActScale, WeightScale,
//!   Out]` with group-32 weight scales - the prefill GEMM of every int8
//!   resident model, run on int8 tensor cores (`matmul_i8_mma`). It is taken
//!   only above the decode regime ([`select::DECODE_REGIME_MAX_ROWS`] rows -
//!   below that the portable GEMV family is the right shape), only when K is a
//!   whole number of 64-element tiles, and only on a device whose capability
//!   reaches the kernel's own floor.
//!
//! Anything else (another weight tier or scale group, a backward pass,
//! another operator) is declined by [`CudaProvider::accepts`] and served by
//! the WGSL reference provider exactly as before - a decline, never a forced
//! failure and never a silently different answer.

use backend_api::select::{self, Dtype};
use backend_api::{BindKind, DType, ImplSource};
use kernels_cuda::{CudaKernel, Cc};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Pass, Role};

/// `matmul.wgsl`'s own `@binding` order, which the f32 GEMM mirrors: uniform,
/// activations, weights, output.
const MATMUL_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `matmul_i8_dyn.wgsl`'s own `@binding` order: uniform, packed activations,
/// packed weights, per-token activation scale, group weight scale, output.
const MATMUL_I8_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `paged_flash_prefill_hd256.wgsl`'s own `@binding` order: uniform, queries,
/// K pool, V pool, block tables, sequence lengths, context output.
const PAGED_FLASH_PREFILL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// Elements of K per weight scale the int8 tensor-core kernel folds at (one
/// `k32` MMA). A layout with a different group is declined, not approximated.
const I8_MMA_GROUP: u32 = 32;

/// Elements of K per staged tile of the int8 tensor-core kernel (two groups).
const I8_MMA_K_TILE: u32 = 64;

pub struct CudaProvider {
    /// The capability the DRIVER reported for the device this provider was
    /// built for. Never a default: a provider with no capability to resolve
    /// against cannot be constructed (see [`CudaProvider::for_gpu`]).
    cc: Cc,
}

impl CudaProvider {
    /// A provider that resolves kernels against compute capability `cc`.
    pub fn new(cc: Cc) -> CudaProvider {
        CudaProvider { cc }
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

    /// The registry entry this provider would run for `op` at weight tier
    /// `dtype` on its device, or `None` when the registry offers nothing for
    /// that operator and tier at this capability.
    pub fn kernel(&self, op: select::Op, dtype: Dtype) -> Option<&'static CudaKernel> {
        if dtype == Dtype::I8 {
            // The int8 tensor-core GEMM is asked for by name: the registry's
            // (MatMul, I8) answer stays the decode GEMV `native_upgrade` resolves.
            return kernels_cuda::get("matmul_i8_mma").filter(|k| k.min_cc <= self.cc);
        }
        kernels_cuda::find(op, dtype, self.cc)
    }

    /// [`Self::kernel`]'s registry name - what a test or a diagnostic reports
    /// without having to reach into the registry itself.
    pub fn kernel_name(&self, op: select::Op, dtype: Dtype) -> Option<&'static str> {
        self.kernel(op, dtype).map(|k| k.name)
    }

    /// The weight tier of the kernel `req` is shaped for, or `None` when it
    /// is neither request this provider implements.
    ///
    /// Checked structurally rather than trusted: `shape.dtype` names the
    /// WEIGHT tier, and the operand bundle is what the kernel actually binds.
    /// A quantized bundle has five operands with two scale planes in the
    /// middle; binding those to a three-pointer kernel would read scales as
    /// weights - a wrong answer, not a crash. The same holds for the uniform:
    /// the int8 kernel reads `[m, k/4, n]`, and a request whose uniform
    /// disagrees with its own shape is refused rather than launched.
    fn shape_of(req: &OpRequest) -> Option<Dtype> {
        let roles = |want: &[(Role, DType)]| {
            req.operands.len() == want.len() && req.operands.iter().zip(want).all(|(o, (r, d))| o.role == *r && o.dtype == *d)
        };
        match req.shape.dtype {
            Dtype::F32 if roles(&[(Role::Act, DType::F32), (Role::Weight, DType::F32), (Role::Out, DType::F32)]) => Some(Dtype::F32),
            Dtype::I8
                if roles(&[
                    (Role::Act, DType::I8),
                    (Role::Weight, DType::I8),
                    (Role::ActScale, DType::F32),
                    (Role::WeightScale, DType::F32),
                    (Role::Out, DType::F32),
                ]) && req.group == I8_MMA_GROUP
                    && req.shape.m > select::DECODE_REGIME_MAX_ROWS
                    && req.shape.k % I8_MMA_K_TILE == 0
                    && req.attrs == [req.shape.m, req.shape.k / 4, req.shape.n] =>
            {
                Some(Dtype::I8)
            }
            _ => None,
        }
    }
}

/// Whether `gpu` runs this provider's tensor-core kernels: its device reports a
/// compute capability at the int8 / fp16 MMA floor
/// ([`kernels_cuda::MMA_S8_MIN_CC`]) and `cuda` is not named in
/// `BRAIN_NO_PROVIDER`. A caller whose numerics depend on the tier (a test
/// stating a tolerance, a parity gate) asks this rather than guessing.
pub fn tensor_core_kernels_enabled(gpu: &crate::Gpu) -> bool {
    !super::disabled_providers().iter().any(|d| d == "cuda") && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::MMA_S8_MIN_CC)
}

/// The native tensor-core paged flash-prefill step for `head_dim = 256`, or
/// `None` when the caller should dispatch the portable
/// `paged_flash_prefill_hd256` it was going to anyway.
///
/// This is not a provider request: attention's call site
/// (`model::block::gqa_chunk_step`) already chooses its kernel by registered
/// pipeline index, and the native kernel is a drop-in for that index - same
/// buffers, same uniform, same launch geometry - so the seam is "try native
/// first" at that one choice. `None` means any of: the head width is not
/// `256`, `cuda` is named in `BRAIN_NO_PROVIDER`, the device's compute
/// capability is below the kernel's floor (or not reported at all), or the
/// backend declined to take the kernel. Each is a decline, never a failure.
///
/// `bufs` is the portable kernel's own order - queries, K pool, V pool, block
/// tables, sequence lengths, context - bound whole, and `params` its
/// `[bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt]`; `blocks`
/// is its workgroup count.
pub fn paged_flash_prefill_step(gpu: &crate::Gpu, head_dim: u32, bufs: &[&backend_api::DeviceBuffer; 6], params: &[u32], blocks: u32) -> Option<crate::Step> {
    if head_dim != 256 || !tensor_core_kernels_enabled(gpu) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::find(select::Op::PagedAttentionFused, Dtype::F32, cc)?;
    let id = gpu.native_kernel(kernel, PAGED_FLASH_PREFILL_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 6], params, blocks)
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
            && match Self::shape_of(req) {
                Some(dt) => self.kernel(select::Op::MatMul, dt).is_some(),
                None => false,
            }
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        let dt = match (req.op, Self::shape_of(req)) {
            (select::Op::MatMul, Some(dt)) => dt,
            _ => {
                return Err(format!(
                    "cuda::CudaProvider::lower: {:?} at dtype {:?} is not implemented (plain f32 MatMul and group-32 int8 MatMul only)",
                    req.op, req.shape.dtype
                ))
            }
        };
        let k = self.kernel(select::Op::MatMul, dt).ok_or_else(|| {
            format!("cuda::CudaProvider::lower: no native {dt:?} MatMul kernel at compute capability {}.{}", self.cc.0, self.cc.1)
        })?;
        let bindings = if dt == Dtype::I8 { MATMUL_I8_BINDINGS } else { MATMUL_BINDINGS };
        let Some(id) = ctx.gpu.native_kernel(k, bindings) else {
            return Err(format!(
                "cuda::CudaProvider::lower: this device declined the native {dt:?} matmul kernel \
                 (register_native returned None) - falling back to the WGSL reference provider"
            ));
        };

        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let offsets: Vec<(u64, u64)> = req.operands.iter().map(|o| o.range).collect();
        // The uniform is the reference kernel's own (`[m, k, n]` for f32,
        // `[m, k/4, n]` for int8), passed through untouched - the hand-written
        // kernel reads the identical layout on purpose, so there is no second
        // place a shape could be spelled differently.
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
            group: 32,
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
        assert!(below.kernel(select::Op::MatMul, Dtype::F32).is_none() && below.kernel(select::Op::MatMul, Dtype::I8).is_none());
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
        assert!(p.kernel(select::Op::MatMul, Dtype::F32).is_some() && p.kernel(select::Op::MatMul, Dtype::I8).is_some());

        let operands = f32_bundle(&x);
        assert!(p.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32)), false));
        assert!(!p.accepts(&req(Pass::Backward, &operands, shape(64, 64, 64, Dtype::F32)), false));
        // Three f32 operands under an I8 shape: the bundle, not the label, decides.
        assert!(!p.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::I8)), false));
    }

    /// The int8 tensor-core kernel is taken for exactly the layout it
    /// implements: group-32 scales, a K that is a whole number of 64-wide
    /// tiles, a row count above the decode regime, and a uniform that agrees
    /// with the shape. Each deviation is a decline (the portable kernel then
    /// serves it), never a launch that reads the wrong plane.
    #[test]
    fn the_int8_tensor_core_kernel_is_taken_only_for_the_layout_it_implements() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let x = gpu.storage(1);
        let p = CudaProvider::new((99, 0));
        let quantized = [
            Operand { role: Role::Act, buf: &x, range: (0, 1), dtype: DType::I8 },
            Operand { role: Role::Weight, buf: &x, range: (0, 0), dtype: DType::I8 },
            Operand { role: Role::ActScale, buf: &x, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::WeightScale, buf: &x, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &x, range: (0, 1), dtype: DType::F32 },
        ];
        let with = |m: u32, n: u32, k: u32, group: u32, attrs: &[u32]| {
            let sh = shape(m, n, k, Dtype::I8);
            let mut r = req(Pass::Forward, &quantized, sh);
            r.group = group;
            r.attrs = attrs;
            p.accepts(&r, false)
        };
        assert!(with(256, 5120, 5120, 32, &[256, 1280, 5120]), "the real prefill shape is the kernel's own");
        assert!(!with(256, 5120, 5120, 16, &[256, 1280, 5120]), "group-16 (Q6_K) scales are a different layout");
        assert!(!with(8, 5120, 5120, 32, &[8, 1280, 5120]), "decode-regime row counts belong to the GEMV family");
        assert!(!with(256, 5120, 5152, 32, &[256, 1288, 5120]), "K must be a whole number of 64-wide tiles");
        assert!(!with(256, 5120, 5120, 32, &[256, 5120, 5120]), "the uniform carries K/4, not K");
        assert!(!with(256, 5120, 5120, 32, &[]), "a request without its uniform is malformed");
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
