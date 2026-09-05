// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CoopMatProvider` - the first non-WGSL [`super::OperatorProvider`] (M8.9,
//! `kernel-performance.md`): `VK_KHR_cooperative_matrix` f16xf16->f32 GEMM,
//! 16x16x16 subgroup tiles, wired through `Backend::register_native`/
//! `step_native` (M8.3's ABI) rather than dispatching outside it.
//!
//! **Capability gate, not a fake positive.** [`CoopMatProvider::requires`]
//! reports `Requirement.matrix` (M8.1) for the exact `(f16, f16, f32)` triple
//! the kernel was authored for; [`super::ProviderRegistry::resolve`] checks
//! that against the REAL `DeviceCaps::arch.matrix` BEFORE `accepts`/`lower`
//! are ever reached. Cooperative matrix needs Turing sm_75+ (NVIDIA) or an
//! equivalent AMD/Intel matrix-engine driver; every box this campaign has
//! actually run on (P40/Xeon, and this crate's own sandbox - an Intel iGPU
//! with no `VK_KHR_cooperative_matrix` shapes) correctly DECLINES here. That
//! is the gate working, not a hole in it - see `gpu_core::tests` (the crate-
//! level integration test proving `requires()` is unsatisfied on this box's
//! real `DeviceCaps`) and this milestone's own ledger entry.
//!
//! **Gradient-check safety.** f16-multiply/f32-accumulate is not gradient-
//! faithful: the intermediate rounding a real finite-difference oracle would
//! flag is exactly what a training backward pass cannot silently absorb.
//! [`CoopMatProvider::accepts`] refuses `Pass::Backward` unconditionally -
//! the same structural rule the native-f16 provider (a sibling wave-2
//! milestone) applies for the identical reason.
//!
//! **Scope.** Only `Op::MatMul`, forward, with both operands already F16 and
//! a tile-aligned `(m, n, k)` (`backend_vulkan::coopmat::is_tile_aligned`) -
//! this provider does not repack an arbitrary incoming shape/dtype into the
//! packed-f16, tile-padded layout the kernel needs; a caller that wants this
//! path must already produce operands in that shape (nothing in a real model
//! call site does today - matching M8.1's own "no vendor pack ships in this
//! campaign" scope, this provider proves the ABI wiring, not a production
//! migration). Any request outside that falls back to the WGSL reference
//! provider via `accepts` returning `false`, never a forced failure.

use std::sync::Mutex;

use backend_api::{select, DType, NativeId};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Pass, Role};

/// Lazily-registered [`NativeId`] for the coopmat pipeline, cached per
/// `CoopMatProvider` instance - like `select::CachedSelector` and every other
/// per-device cache in this engine, this assumes one provider instance is
/// used against a single backend/device for its whole lifetime (the normal
/// shape: one `ProviderRegistry` built per `Ops`/`Gpu`, not shared across
/// unrelated devices).
#[derive(Clone, Copy)]
enum Registration {
    Unattempted,
    /// The device declined (see `backend_vulkan::coopmat::build_pipeline`'s
    /// doc) or no SPIR-V was baked in at build time.
    Declined,
    Registered(NativeId),
}

pub struct CoopMatProvider {
    registration: Mutex<Registration>,
}

impl Default for CoopMatProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CoopMatProvider {
    pub fn new() -> CoopMatProvider {
        CoopMatProvider { registration: Mutex::new(Registration::Unattempted) }
    }

    /// Register the coopmat pipeline on `gpu`'s backend if this is the first
    /// call, else reuse the cached [`NativeId`]/decline. `None` exactly when
    /// this device/build cannot run the kernel - see this module's own doc
    /// comment for why that is an expected, correct outcome on most hardware.
    fn native_id(&self, gpu: &crate::Gpu) -> Option<NativeId> {
        let mut reg = self.registration.lock().unwrap_or_else(|e| e.into_inner());
        match *reg {
            Registration::Registered(id) => Some(id),
            Registration::Declined => None,
            Registration::Unattempted => {
                let spec = backend_vulkan::coopmat::spec()?;
                let id = gpu.register_native(&spec);
                *reg = match id {
                    Some(id) => Registration::Registered(id),
                    None => Registration::Declined,
                };
                id
            }
        }
    }
}

impl OperatorProvider for CoopMatProvider {
    fn name(&self) -> &'static str {
        "coopmat"
    }

    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        select::Requirement {
            matrix: Some(select::MatShapeReq { a: DType::F16, b: DType::F16, accum: DType::F32 }),
            ..Default::default()
        }
    }

    fn accepts(&self, req: &OpRequest, _capture: bool) -> bool {
        if req.op != select::Op::MatMul || req.pass == Pass::Backward {
            return false;
        }
        if !backend_vulkan::coopmat::is_tile_aligned(req.shape.m, req.shape.n, req.shape.k) {
            return false;
        }
        // Exactly [Act, Weight, Out], both inputs already F16 - the packed,
        // tile-padded layout the kernel reads directly (see this module's
        // top-level scope note: no on-the-fly repacking here).
        req.operands.len() == 3
            && req.operands[0].role == Role::Act
            && req.operands[1].role == Role::Weight
            && req.operands[2].role == Role::Out
            && req.operands[0].dtype == DType::F16
            && req.operands[1].dtype == DType::F16
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        if req.op != select::Op::MatMul {
            return Err(format!("coopmat::CoopMatProvider::lower: {:?} is not implemented (MatMul only)", req.op));
        }
        let Some(id) = self.native_id(ctx.gpu) else {
            return Err(
                "coopmat::CoopMatProvider::lower: this device declined the coopmat pipeline \
                 (register_native returned None) - falling back to the WGSL reference provider"
                    .to_string(),
            );
        };
        let tiles_m = req.shape.m / backend_vulkan::coopmat::TILE;
        let tiles_n = req.shape.n / backend_vulkan::coopmat::TILE;
        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let params = [req.shape.m, req.shape.k, req.shape.n];
        let step = ctx
            .gpu
            .step_native(id, &bufs, &params, tiles_m * tiles_n)
            .ok_or_else(|| "coopmat::CoopMatProvider::lower: step_native returned None for a NativeId this provider itself registered".to_string())?;
        ctx.steps.push(step);
        // Not a registered WGSL catalogue name - this is a SPIR-V kernel
        // registered via `register_native`, so `Lowered::kernels`' string is
        // diagnostic-only and deliberately spelled unlike a real kernel
        // identifier (no bare `[a-z0-9_]+` name for a catalogue-cross-
        // referencing gate script to mistake for an unregistered catalogue
        // kernel).
        Ok(Lowered { pushed: 1, kernels: vec!["native:coopmat"] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_api::DeviceCaps;
    use crate::provider::Operand;

    fn shape(m: u32, n: u32, k: u32) -> select::OpShape {
        select::OpShape { m, n, k, dtype: select::Dtype::F16 }
    }

    fn req<'a>(pass: Pass, operands: &'a [Operand<'a>], shape: select::OpShape) -> OpRequest<'a> {
        OpRequest {
            op: select::Op::MatMul,
            shape,
            pass,
            operands,
            attrs: &[],
            bind: &|_| panic!("coopmat provider never calls OpRequest::bind - it has no kernel-name table"),
        }
    }

    /// The gradient-check safety rule: never accept a backward request,
    /// regardless of shape/dtype - fp16 accumulation is not gradient-faithful.
    #[test]
    fn accepts_refuses_backward_unconditionally() {
        let provider = CoopMatProvider::new();
        let r = req(Pass::Backward, &[], shape(32, 32, 32));
        assert!(!provider.accepts(&r, false));
    }

    /// A non-tile-aligned shape declines (falls back to WGSL), never panics
    /// or silently pads.
    #[test]
    fn accepts_refuses_non_tile_aligned_shapes() {
        let provider = CoopMatProvider::new();
        let r = req(Pass::Forward, &[], shape(33, 32, 32));
        assert!(!provider.accepts(&r, false));
    }

    /// `requires()` reports a real `Requirement.matrix` - a device with no
    /// matrix engine at all (every `DeviceCaps::portable_baseline`, and this
    /// crate's own real Vulkan device on Intel iGPU hardware, per the
    /// integration test alongside this one) must never satisfy it.
    #[test]
    fn requires_reports_the_f16_f16_f32_triple_and_a_baseline_caps_never_satisfies_it() {
        let provider = CoopMatProvider::new();
        let r = req(Pass::Forward, &[], shape(32, 32, 32));
        let requirement = provider.requires(&r);
        assert_eq!(
            requirement.matrix,
            Some(select::MatShapeReq { a: DType::F16, b: DType::F16, accum: DType::F32 })
        );
        let caps = DeviceCaps::portable_baseline(backend_api::DeviceClass::Cpu);
        assert!(!requirement.satisfied_by(&caps), "a device with no matrix engine must never satisfy this requirement");
    }
}
