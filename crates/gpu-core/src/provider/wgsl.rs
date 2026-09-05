// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The WGSL reference provider - the portable correctness oracle every other
//! [`super::OperatorProvider`] is gated against (`AGENTS.md`'s "fp32
//! arithmetic only, core compute only" bullet), and, until a second provider
//! ever lands, the only one that runs at all.
//!
//! `Self::threads` below is `model::ops::Ops::threads`, moved verbatim: a
//! pure function of `(variant, dtype, m, n)` with no dependency on any
//! model-crate kernel-name table, so it could move down into `gpu-core`
//! unchanged. `Self::lower_matmul`'s dispatch shape - ask the selector, bind
//! the kernel, compute the thread count, push one `step_sliced` - is
//! `Ops::matmul`'s own dispatch body; what moved is *which operand bundle it
//! reads its buffers from* (an [`super::Operand`] slice instead of a `&Weight`/
//! `&Act` pair), not the arithmetic or the dispatch shape itself. See
//! `super::OpRequest::bind`'s doc comment for why kernel NAME resolution
//! could not move down the same way and stays a caller-supplied closure.
//!
//! ## Scope this milestone
//!
//! `lower` implements `Op::MatMul` only - the operator `Ops::matmul` itself
//! covers, including every weight tier that method dispatches (`F32`/`BF16`/
//! `F16`/`I8`/`Q4`/affine K-quant). Every other `select::Op` variant returns
//! `Err` from `lower` (never reached in this tree today - nothing yet builds
//! an [`super::OpRequest`] for any of them); widening this provider to
//! `Ops::embed`/`moe_linear`/`matmul_dx`/`matmul_dw` is a real follow-up, not
//! attempted here (see the crate-level `provider` module doc for why).

use std::sync::Arc;

use backend_api::select::{self, Dtype, KernelSelector, KernelVariant};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Role};

pub struct WgslProvider {
    selector: Arc<dyn KernelSelector>,
}

impl WgslProvider {
    pub fn new(selector: Arc<dyn KernelSelector>) -> WgslProvider {
        WgslProvider { selector }
    }

    /// Dispatch invocation count for `(variant, dtype, m, n)` - 
    /// `model::ops::Ops::threads`, moved verbatim (see this module's doc
    /// comment). `Op::MatMul`'s own `candidates()` never returns
    /// `SplitReduction`/`FusedFlash`, so those arms stay `unreachable!`
    /// exactly as they were at the old call site.
    fn threads(v: KernelVariant, dt: Dtype, m: u32, n: u32) -> u32 {
        let tile = || m.div_ceil(128) * n.div_ceil(128) * 256;
        match v {
            KernelVariant::Reference => m * n,
            KernelVariant::WorkgroupPerOutput => n * 64,
            KernelVariant::RegisterTiled => tile(),
            KernelVariant::PackedInt8 => match dt {
                // `NF4`/`F4E2M1` (M8.5) are W4A8, physically packed exactly
                // like `Q4` - a future register-tiled `PackedInt8` kernel for
                // either would need the identical tile geometry, so they fold
                // into this arm now even though no such kernel exists yet
                // (only `matmul_q4_gemv_nf4`/`matmul_q4_gemv_f4e2m1`,
                // `WorkgroupPerOutput`, do).
                Dtype::I8 | Dtype::Q4 | Dtype::Q4K | Dtype::Q8K | Dtype::NF4 | Dtype::F4E2M1 => tile(),
                // `select::candidates` never offers `PackedInt8` for an
                // F32-family dtype (see this method's own doc comment) -
                // this arm is unreachable in practice, kept only so the
                // match stays exhaustive over `Dtype` without silently
                // absorbing a real future `PackedInt8` dtype into the wrong
                // formula the way the pre-M5.5 blanket `_ => m*n` did for
                // `Dtype::Q4`.
                Dtype::F32 | Dtype::BF16 | Dtype::F16 => m * n,
            },
            KernelVariant::SplitReduction => {
                unreachable!("Op::MatMul's candidates() never returns SplitReduction")
            }
            KernelVariant::FusedFlash => {
                unreachable!("Op::MatMul's candidates() never returns FusedFlash")
            }
        }
    }

    fn lower_matmul(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        let variant = self.selector.select(req.op, req.shape, ctx.caps);
        let (kind, name) = (req.bind)(variant);
        let threads = Self::threads(variant, req.shape.dtype, req.shape.m, req.shape.n);

        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let offsets: Vec<(u64, u64)> = req.operands.iter().map(|o| o.range).collect();

        if req.operands.last().map(|o| o.role) != Some(Role::Out) {
            return Err(
                "wgsl::WgslProvider::lower_matmul: OpRequest::operands must end with the Role::Out \
                 operand -- Gpu::step_sliced binds the output buffer last"
                    .to_string(),
            );
        }

        ctx.steps.push(ctx.gpu.step_sliced(kind, &bufs, &offsets, req.attrs, threads));
        Ok(Lowered { pushed: 1, kernels: vec![name] })
    }
}

impl OperatorProvider for WgslProvider {
    fn name(&self) -> &'static str {
        "wgsl"
    }

    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        // The selector this provider dispatches through already only ever
        // offers a `KernelVariant` whose own `requires(dtype)` is satisfied
        // by the `DeviceCaps` it was asked with (`select::candidates`'s own
        // contract) - there is no coarser, provider-level requirement ahead
        // of that to report.
        select::Requirement::default()
    }

    fn accepts(&self, _req: &OpRequest, _capture: bool) -> bool {
        // The reference provider is the fallback of last resort by
        // construction (`ProviderRegistry` always places it last) and must
        // never refuse a request the way a specialised provider legitimately
        // can - see this module's doc comment for which `Op`s `lower`
        // actually implements today; an `Op` outside that set still
        // `accepts` here (this invariant), then fails loudly inside `lower`
        // instead (see `ProviderRegistry::dispatch`'s own doc comment on
        // that path). Every WGSL dispatch is a plain `Step` pushed onto the
        // caller's tape, so `capture` never disqualifies it either.
        true
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        match req.op {
            select::Op::MatMul => self.lower_matmul(ctx, req),
            other => Err(format!(
                "wgsl::WgslProvider::lower: {other:?} is not yet migrated onto the OperatorProvider \
                 seam (M8.3 wired Op::MatMul only -- see the `provider` module's top-level doc comment)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Operand, Pass};
    use crate::testgpu;
    use backend_api::DType;

    static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL), ("matmul_gemv", kernels::MATMUL_GEMV)];

    fn bind(v: KernelVariant) -> (usize, &'static str) {
        match v {
            KernelVariant::Reference => (0, "matmul"),
            KernelVariant::WorkgroupPerOutput => (1, "matmul_gemv"),
            other => panic!("test bind: no kernel registered for {other:?}"),
        }
    }

    /// The reference provider dispatches `Op::MatMul` end to end: same
    /// output a direct `gpu.step_sliced` call over the same buffers would
    /// have produced (the plain `F32` weight tier, `Ops::matmul`'s simplest
    /// arm).
    #[test]
    fn wgsl_provider_lowers_matmul_f32_to_the_right_output() {
        let gpu = testgpu::dev(KERNELS);
        let (m, n, k) = (2u32, 3u32, 4u32);
        let x = gpu.storage_init("x", &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let w = gpu.storage_init(
            "w",
            &[
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = gpu.storage((m * n) as u64);

        let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::AlwaysReference));
        let provider = WgslProvider::new(selector);
        let operands = [
            Operand { role: Role::Act, buf: &x, range: (0, (m * k) as u64), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &y, range: (0, (m * n) as u64), dtype: DType::F32 },
        ];
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: select::OpShape { m, n, k, dtype: Dtype::F32 },
            pass: Pass::Forward,
            operands: &operands,
            attrs: &[m, k, n],
            bind: &bind,
        };
        let mut steps = Vec::new();
        let caps = gpu.caps();
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        let lowered = provider.lower(&mut ctx, &req).unwrap();
        assert_eq!(lowered.pushed, 1);
        assert_eq!(lowered.kernels, vec!["matmul"]);

        gpu.submit(&[], &steps);
        // x[0,:]=[1,0,0,0] picks out column 0 of every W row -> [1,5,9];
        // x[1,:]=[0,1,0,0] picks out column 1 -> [2,6,10].
        assert_eq!(gpu.read(&y, (m * n) as usize), vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0]);
    }

    #[test]
    fn wgsl_provider_reports_the_kernel_it_actually_bound() {
        let gpu = testgpu::dev(KERNELS);
        let x = gpu.storage(1);
        let w = gpu.storage(1);
        let y = gpu.storage(1);
        let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
        let provider = WgslProvider::new(selector);
        let operands = [
            Operand { role: Role::Act, buf: &x, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &y, range: (0, 1), dtype: DType::F32 },
        ];
        // A decode-shaped (m=1) request: DefaultSelector picks
        // WorkgroupPerOutput -> "matmul_gemv" on any real GPU cap set, but
        // the CPU JIT here reports `workgroup_reductions: false`, so it
        // stays `Reference` -> "matmul" - either is a valid, honest
        // assertion of "the name Lowered reports matches what was bound",
        // which is all this test claims.
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: select::OpShape { m: 1, n: 1, k: 1, dtype: Dtype::F32 },
            pass: Pass::Forward,
            operands: &operands,
            attrs: &[1, 1, 1],
            bind: &bind,
        };
        let mut steps = Vec::new();
        let caps = gpu.caps();
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        let lowered = provider.lower(&mut ctx, &req).unwrap();
        assert_eq!(lowered.pushed, 1);
        assert_eq!(lowered.kernels.len(), 1);
        assert!(lowered.kernels[0] == "matmul" || lowered.kernels[0] == "matmul_gemv");
    }

    #[test]
    fn wgsl_provider_refuses_an_operand_list_not_ending_in_out() {
        let gpu = testgpu::dev(KERNELS);
        let x = gpu.storage(1);
        let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::AlwaysReference));
        let provider = WgslProvider::new(selector);
        let operands = [Operand { role: Role::Act, buf: &x, range: (0, 1), dtype: DType::F32 }];
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: select::OpShape { m: 1, n: 1, k: 1, dtype: Dtype::F32 },
            pass: Pass::Forward,
            operands: &operands,
            attrs: &[1, 1, 1],
            bind: &bind,
        };
        let mut steps = Vec::new();
        let caps = gpu.caps();
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        assert!(provider.lower(&mut ctx, &req).is_err());
    }

    /// *** The specific bug `Self::threads`'s own doc comment warns about
    /// *** (moved verbatim from `model::ops::Ops::threads`'s own regression
    /// test, `kq_dtypes_dispatch_the_tiled_formula_not_m_times_n`, M8.3):
    /// `Weight::KQuant`'s affine dtypes, and `Dtype::Q4` (M5.5's
    /// `matmul_q4_dyn_reg`), MUST dispatch the TILED formula
    /// (`matmul_kq_dyn`/`matmul_q4_dyn_reg` are 128x128 register-tiled
    /// kernels, `matmul_i8_dyn`'s own siblings), not a naive `m*n` count -
    /// under-dispatching leaves real output elements never written (silent
    /// corruption, not a crash).
    #[test]
    fn kq_dtypes_dispatch_the_tiled_formula_not_m_times_n() {
        let (m, n) = (513u32, 257u32);
        let expected_tile = m.div_ceil(128) * n.div_ceil(128) * 256;
        assert_ne!(expected_tile, m * n, "test shape must distinguish tile() from m*n");
        for dt in [Dtype::Q4, Dtype::Q4K, Dtype::Q8K] {
            assert_eq!(
                WgslProvider::threads(KernelVariant::PackedInt8, dt, m, n),
                expected_tile,
                "{dt:?} must dispatch the tile formula, not m*n -- under-dispatching leaves real \
                 output elements never written (silent corruption, not a crash)"
            );
        }
    }

    #[test]
    fn wgsl_provider_declines_ops_it_does_not_implement_yet() {
        let gpu = testgpu::dev(KERNELS);
        let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::AlwaysReference));
        let provider = WgslProvider::new(selector);
        let req = OpRequest {
            op: select::Op::RmsNorm,
            shape: select::OpShape { m: 1, n: 1, k: 1, dtype: Dtype::F32 },
            pass: Pass::Forward,
            operands: &[],
            attrs: &[],
            bind: &bind,
        };
        let mut steps = Vec::new();
        let caps = gpu.caps();
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        // `accepts` is unconditionally true (the reference-provider
        // invariant), but `lower` for an unimplemented op still errors.
        assert!(provider.accepts(&req, false));
        assert!(provider.lower(&mut ctx, &req).is_err());
    }
}
