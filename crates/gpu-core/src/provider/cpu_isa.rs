// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CpuIsaProvider` - `kernel-performance.md` M8.10: the first real
//! [`OperatorProvider`] built on the ABI [`super::wgsl::WgslProvider`] proved
//! zero-delta, and the first one this engine ships that is not the WGSL
//! reference.
//!
//! Swedish Embedded AB implements CPU-native SIMD fast paths for neural-net
//! inference on machines with no GPU at all. If your team needs an
//! `OperatorProvider`/ISA-pack seam like this one for your own edge-AI
//! engine, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ## Hoisting the existing AVX2 F32 GEMM into the ABI
//!
//! `backend_cpu::CpuBackend::dispatch` already intercepts the `matmul`/
//! `matmul_tiled`/`matmul_reg{,2,3}` kernel NAMES with a hidden if-ladder and
//! calls `fast_ops::matmul_abt` directly - correct, but reached only by a
//! backend-internal name match, invisible to the `OperatorProvider` seam.
//! `Self::lower_matmul_f32` reaches the SAME function through
//! `backend_api::Backend::register_native`/`step_native` instead - see
//! `crates/gpu-core/tests/cpu_isa_provider_zero_delta.rs`'s
//! `cpu_isa_f32_matmul_is_bit_identical_to_the_hidden_fastpath` for the
//! zero-delta proof this milestone's own brief demanded (same call, same
//! numeric output, reached two different ways).
//!
//! ## Why this provider never touches `select::Requirement`
//!
//! [`backend_api::Backend::register_native`] already refuses (`None`) under
//! the identical AVX2-availability condition `backend_cpu::CpuBackend`'s own
//! hidden fast-path if-ladder refuses under - see that method's own doc on
//! `fast_native_enabled`. So [`Self::lower_matmul_f32`] fails loudly (an
//! `Err`, caught by [`super::ProviderRegistry::dispatch`]'s own documented
//! fallback) exactly when the native path is genuinely unavailable, with no
//! need to duplicate that gate through a `Requirement` field this provider
//! would have to keep in sync by hand. `Self::accepts` therefore answers
//! purely from `req` (which `Op`/`Dtype`), never from `caps` - the ONE thing
//! this provider cannot get wrong is claiming to run something
//! [`super::LowerCtx::gpu`]'s own backend cannot actually register, and
//! `register_native` returning `None` is exactly that claim being refused at
//! the source, not guessed at here.
//!
//! **Deliberately NOT wired into any production `Ops::with_providers` call
//! site this milestone** - `Ops::matmul` on `backend-cpu` already gets the
//! identical AVX2 F32 GEMM through the pre-existing hidden if-ladder, so this
//! provider is the ABI-native reachability proof, not a behavioural change to
//! any live model.

use std::sync::OnceLock;

use backend_api::{select, DType, NativeSpec};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Role};

/// The native kernel this provider knows how to reach, resolved (and its
/// [`backend_api::NativeId`] cached) lazily on first use against whichever
/// `Gpu`/backend a given request's [`LowerCtx::gpu`] carries - `None` cached
/// once and for all means "this backend refused to register it", never
/// re-probed per dispatch.
#[derive(Default)]
pub struct CpuIsaProvider {
    matmul_f32: OnceLock<Option<backend_api::NativeId>>,
}

impl CpuIsaProvider {
    pub fn new() -> CpuIsaProvider {
        CpuIsaProvider::default()
    }

    fn native_id(cell: &OnceLock<Option<backend_api::NativeId>>, gpu: &crate::Gpu, name: &'static str) -> Option<backend_api::NativeId> {
        *cell.get_or_init(|| gpu.register_native(&NativeSpec::HostFn(name)))
    }

    /// `Op::MatMul` at `Dtype::F32` - `fast_ops::matmul_abt`, the same AVX2
    /// GEMM `CpuBackend::dispatch`'s hidden `f.matmul` arm already calls.
    fn lower_matmul_f32(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        let id = Self::native_id(&self.matmul_f32, ctx.gpu, "cpu_matmul_abt").ok_or_else(|| {
            "CpuIsaProvider::lower_matmul_f32: backend has no native \"cpu_matmul_abt\" HostFn path \
             (not the CPU backend, or AVX2/BRAIN_NO_FASTCONV disabled it)"
                .to_string()
        })?;
        if req.operands.last().map(|o| o.role) != Some(Role::Out) {
            return Err("CpuIsaProvider::lower_matmul_f32: operands must end with Role::Out".to_string());
        }
        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let threads = req.shape.m * req.shape.n;
        let step = ctx
            .gpu
            .step_native(id, &bufs, req.attrs, threads)
            .ok_or_else(|| "CpuIsaProvider::lower_matmul_f32: step_native returned None".to_string())?;
        ctx.steps.push(step);
        Ok(Lowered { pushed: 1, kernels: vec!["cpu_matmul_abt"] })
    }
}

impl OperatorProvider for CpuIsaProvider {
    fn name(&self) -> &'static str {
        "cpu-isa"
    }

    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        // See this module's own doc comment ("why this provider never
        // touches `select::Requirement`") - `register_native` returning
        // `None` is the real gate; there is nothing left for a `Requirement`
        // to check ahead of it.
        select::Requirement::default()
    }

    fn accepts(&self, req: &OpRequest, _capture: bool) -> bool {
        req.op == select::Op::MatMul && req.shape.dtype == DType::F32
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        match (req.op, req.shape.dtype) {
            (select::Op::MatMul, DType::F32) => self.lower_matmul_f32(ctx, req),
            (op, dt) => Err(format!("CpuIsaProvider::lower: {op:?}/{dt:?} not implemented")),
        }
    }
}
