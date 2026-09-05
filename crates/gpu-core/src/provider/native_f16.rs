// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `NativeF16Provider` - the first non-reference [`super::OperatorProvider`]
//! (`kernel-performance.md` M8.7), narrowing `Op::MatMul`'s forward GEMM to
//! real `f16`-typed registers for the multiply (f32 accumulate) instead of
//! `matmul_reg3`'s all-f32 arithmetic. Built on top of the M8.3
//! `OperatorProvider` ABI and M8.1's [`backend_api::arch::ArchDesc`] - read
//! both before touching this file.
//!
//! **The capability gate, and why it lives here, not in [`Self::requires`].**
//! [`super::OperatorProvider::requires`] returns a [`select::Requirement`],
//! checked by [`super::ProviderRegistry::resolve`] against the AMBIENT
//! [`backend_api::DeviceCaps`] every provider in the chain shares.
//! `Requirement::f16_compute`'s own `satisfied_by` check reads
//! `caps.numeric.f16` - which stays a hard-coded `false` in every backend's
//! production `query_caps` today (`backend-wgpu`'s own B11/M8.1 doc comment:
//! a roofline-grade measurement does not belong on the per-construction hot
//! path `query_caps` runs on, the same reason `peak_gflops`/
//! `peak_bandwidth_gbs` stay `None` until `gpu_core::roof::ensure` measures
//! them lazily). If this provider's `requires()` set `f16_compute: true`, it
//! would be permanently unselectable via the shared `DeviceCaps` channel
//! regardless of what THIS provider itself measures - a worse bug than the
//! "availability alone is enough" trap this milestone exists to avoid, not a
//! fix for it. So `requires()` stays [`select::Requirement::default`]
//! (imposes no constraint through that channel), and the real, measured gate
//! lives entirely in [`Self::accepts`] instead, using
//! [`backend_api::arch::ArchDesc::is_fast`] against this provider's own
//! `arch` snapshot (built once at [`Self::probe`] time). `accepts` has no
//! restriction to a shared channel - a provider may decline for its own
//! reasons, per [`super::OperatorProvider::accepts`]'s own doc.
//!
//! **What is narrowed, and what is not.** Only the multiply: both operands
//! convert from the existing plain-`f32` shared-memory tile to `f16`
//! registers immediately before each of the 64 per-thread products, and the
//! product widens back to `f32` before accumulating - see
//! `kernels::template::native_f16_matmul::MATMUL_REG3_F16N`'s own doc
//! comment for the exact diff against `matmul_reg3.wgsl` and why no buffer
//! layout changes. This means [`select::Op::MatMul`] requests this provider
//! accepts still declare `Dtype::F32` (the buffers really are plain f32
//! words) - this provider is an ALTERNATIVE, measured-faster implementation
//! of the SAME fp32 forward GEMM `matmul_reg3` already serves, not a new
//! storage tier. A caller integrating this behind `Ops::matmul`'s existing
//! `Dtype::F16` (the storage-tier decode - packed f16 BYTES, a totally
//! different buffer layout) would need a new, distinct dtype tag so a
//! `lower` failure's WGSL-reference fallback could never reinterpret one
//! buffer layout as the other - not attempted this milestone (`model::ops`
//! is untouched, matching M8.3's own "each future provider needs its own
//! migration onto the seam" scope boundary).
//!
//! **Gradient-check safety, structural.** [`Self::accepts`] refuses
//! `Pass::Backward` unconditionally - `f16` multiply is not gradient-faithful
//! at this repo's finite-difference gradcheck tolerances (`crates/gradcheck`),
//! and nothing in this engine has a backward GEMM migrated onto the
//! `OperatorProvider` seam yet regardless (M8.3's own scope note), so this is
//! a structural guarantee against a future caller that DOES migrate one,
//! never merely "nothing calls this backward today."

use backend_api::arch::{ArchDesc, TierLevel, TierSupport};
use backend_api::{select, DType};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Pass, Role};
use crate::Gpu;

/// The one kernel this provider dispatches - `enable f16;`-wrapped by
/// [`kernel`] below, registered under this name on any `Gpu` handle that
/// wants to use this provider.
pub const KERNEL_NAME: &str = "matmul_reg3_f16n";

/// The `(name, source)` pair a caller passes to `Gpu::new*`'s kernel list -
/// wraps [`kernels::template::native_f16_matmul::MATMUL_REG3_F16N`] with the
/// `enable f16;` directive via [`kernels::template::native_f16_variant`], the
/// one textual step every native-f16 kernel in this tree needs (see that
/// function's own doc for why it is not folded into the constant itself).
pub fn kernel() -> (&'static str, &'static str) {
    kernels::template::native_f16_variant(KERNEL_NAME, kernels::template::native_f16_matmul::MATMUL_REG3_F16N)
}

/// A native-f16 `Op::MatMul` (forward only) provider, gated on a REAL,
/// measured `speedup_vs_f32` for this device - see this module's own doc
/// comment for the full gate, and [`Self::probe`]/[`Self::from_arch`] for the
/// two ways to build one.
pub struct NativeF16Provider {
    arch: ArchDesc,
}

impl NativeF16Provider {
    /// Build from an already-known [`ArchDesc`] - the seam a test uses to
    /// construct a synthetic fast/slow/unmeasured device without a real GPU
    /// (see this module's test suite below).
    pub fn from_arch(arch: ArchDesc) -> NativeF16Provider {
        NativeF16Provider { arch }
    }

    /// Measure THIS device's real native-f16-vs-fp32 GEMM-shaped throughput
    /// and build an [`ArchDesc`] snapshot from the result - the one place
    /// this provider's gate becomes a real number instead of a guess.
    ///
    /// Checks [`Gpu::supports_native_f16`] FIRST and returns a snapshot whose
    /// `F16` tier is `Absent` (the default) if it is false - compiling
    /// `enable f16;` source on a device that never asked for the feature is a
    /// hard device-fault panic on every backend this engine has (see
    /// [`backend_api::Backend::supports_native_f16`]'s own doc comment), so
    /// this check is load-bearing, not a nicety. Reuses
    /// `gpu_core::roof::measure_compute`/`measure_f16` (the exact fp32/f16
    /// GFLOP/s probes B11's own roofline comparison used) rather than a third
    /// reimplementation of the same calibration loop - on `gpu_core::roof
    /// ::PROBE_KERNELS`'s own dedicated probe device (built fresh via
    /// `gpu.new_like`, warmed up, THEN timed), never on `gpu` directly: `gpu`
    /// is the CALLER's own handle, whose kernel list is whatever THAT caller
    /// registered (e.g. this provider's own `matmul_reg3_f16n`), not
    /// `roof_fma` at the index `measure_compute` assumes - reusing `gpu` as
    /// the probe device produced a real wgpu bind-group-layout mismatch panic
    /// the first time this was tried, exactly because index 0 was some other
    /// kernel's pipeline. An aborted measurement (either probe returns
    /// `None`, or the fp32 probe reports a nonsensical `<= 0` rate) leaves
    /// the tier at its default `Absent`/unmeasured state - "unmeasured" must
    /// never read as "measured and fast", the same rule
    /// [`TierSupport::speedup_vs_f32`]'s own doc states.
    pub fn probe(gpu: &Gpu) -> NativeF16Provider {
        let mut arch = ArchDesc::default();
        if !gpu.supports_native_f16() {
            return NativeF16Provider { arch };
        }
        let g = gpu.new_like(crate::roof::PROBE_KERNELS);
        crate::roof::warm_up(&g);
        if let (Some(fp32_gflops), Some(f16_gflops)) =
            (crate::roof::measure_compute(&g), crate::roof::measure_f16(&g))
        {
            if fp32_gflops > 0.0 {
                arch.set_tier(
                    DType::F16,
                    TierSupport {
                        level: TierLevel::Native,
                        rate_gops: Some(f16_gflops),
                        speedup_vs_f32: Some(f16_gflops / fp32_gflops),
                    },
                );
            }
        }
        NativeF16Provider { arch }
    }

    /// The real, measured speedup this provider's own snapshot carries -
    /// `None` if never measured (feature unsupported, or the probe aborted).
    /// For diagnostics/reporting only - dispatch decisions go through
    /// [`OperatorProvider::accepts`], never this directly.
    pub fn measured_speedup(&self) -> Option<f32> {
        self.arch.tier(DType::F16).speedup_vs_f32
    }

    /// Dispatch invocation count - identical tile formula to
    /// `wgsl::WgslProvider::threads`'s own `KernelVariant::RegisterTiled`
    /// arm, since this kernel is byte-identical in tiling to `matmul_reg3`
    /// (see `MATMUL_REG3_F16N`'s own doc comment).
    fn threads(m: u32, n: u32) -> u32 {
        m.div_ceil(128) * n.div_ceil(128) * 256
    }

    fn lower_matmul(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        let kind = ctx.gpu.kernel_index(KERNEL_NAME).ok_or_else(|| {
            format!(
                "native_f16::NativeF16Provider::lower_matmul: kernel '{KERNEL_NAME}' is not \
                 registered on this Gpu handle -- a caller using this provider must include \
                 `native_f16::kernel()`'s (name, source) pair in its kernel list"
            )
        })?;
        let threads = Self::threads(req.shape.m, req.shape.n);

        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let offsets: Vec<(u64, u64)> = req.operands.iter().map(|o| o.range).collect();

        if req.operands.last().map(|o| o.role) != Some(Role::Out) {
            return Err(
                "native_f16::NativeF16Provider::lower_matmul: OpRequest::operands must end with \
                 the Role::Out operand -- Gpu::step_sliced binds the output buffer last"
                    .to_string(),
            );
        }

        ctx.steps.push(ctx.gpu.step_sliced(kind, &bufs, &offsets, req.attrs, threads));
        Ok(Lowered { pushed: 1, kernels: vec![KERNEL_NAME] })
    }
}

impl OperatorProvider for NativeF16Provider {
    fn name(&self) -> &'static str {
        "native-f16"
    }

    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        // See this module's own doc comment for why the real gate is NOT
        // expressed here.
        select::Requirement::default()
    }

    fn accepts(&self, req: &OpRequest, _capture: bool) -> bool {
        req.op == select::Op::MatMul
            && req.pass == Pass::Forward
            && req.shape.dtype == DType::F32
            && self.arch.is_fast(DType::F16)
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        match (req.op, req.pass) {
            (select::Op::MatMul, Pass::Forward) => self.lower_matmul(ctx, req),
            (op, pass) => Err(format!(
                "native_f16::NativeF16Provider::lower: {op:?}/{pass:?} is not supported by this \
                 provider -- OperatorProvider::accepts should have refused this request before \
                 lower was ever called"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_arch() -> ArchDesc {
        let mut arch = ArchDesc::default();
        arch.set_tier(
            DType::F16,
            TierSupport {
                level: TierLevel::Native,
                rate_gops: Some(100.0),
                speedup_vs_f32: Some(backend_api::arch::FAST_TIER_MIN_SPEEDUP as f32 + 0.5),
            },
        );
        arch
    }

    fn shape(m: u32, n: u32, k: u32, dtype: DType) -> select::OpShape {
        select::OpShape { m, n, k, dtype }
    }

    fn bind(_v: select::KernelVariant) -> (usize, &'static str) {
        panic!("NativeF16Provider never calls OpRequest::bind -- it resolves its own fixed kernel")
    }

    fn matmul_forward_req(dtype: DType) -> OpRequest<'static> {
        OpRequest {
            op: select::Op::MatMul,
            shape: shape(128, 128, 128, dtype),
            pass: Pass::Forward,
            operands: &[],
            attrs: &[],
            bind: &bind,
        }
    }

    /// The measured-fast gate, precisely: `Native` level alone (no
    /// measurement) must NOT be enough to accept - only a real
    /// `speedup_vs_f32` clearing `FAST_TIER_MIN_SPEEDUP` may. Synthetic
    /// `ArchDesc`s only - no real (or real-slow) hardware required, per this
    /// milestone's own explicit test requirement.
    #[test]
    fn declines_when_f16_is_not_measured_fast() {
        // Tier absent entirely (the default, unmeasured/unsupported state).
        let never_measured = NativeF16Provider::from_arch(ArchDesc::default());
        assert!(!never_measured.accepts(&matmul_forward_req(DType::F32), false));

        // Structurally Native (queried hardware executes it) but NEVER
        // MEASURED for speed - availability alone must not be enough.
        let mut native_unmeasured = ArchDesc::default();
        native_unmeasured.set_tier(DType::F16, TierSupport { level: TierLevel::Native, ..Default::default() });
        let p = NativeF16Provider::from_arch(native_unmeasured);
        assert!(!p.accepts(&matmul_forward_req(DType::F32), false));

        // Measured, but BELOW the margin (a device where f16 is real but not
        // worth it) -- still declines.
        let mut native_slow = ArchDesc::default();
        native_slow.set_tier(
            DType::F16,
            TierSupport { level: TierLevel::Native, speedup_vs_f32: Some(1.0), ..Default::default() },
        );
        let p = NativeF16Provider::from_arch(native_slow);
        assert!(!p.accepts(&matmul_forward_req(DType::F32), false));
    }

    /// The positive case, symmetric with the above: a REAL measurement at or
    /// above the margin accepts a forward, F32-shaped `Op::MatMul` request.
    #[test]
    fn accepts_a_forward_matmul_when_f16_is_measured_fast() {
        let p = NativeF16Provider::from_arch(fast_arch());
        assert!(p.accepts(&matmul_forward_req(DType::F32), false));
    }

    /// Gradient-check safety, structural: `Pass::Backward` is refused even
    /// when the device measures f16 comfortably fast - narrow-multiply f16
    /// is not gradient-faithful at this repo's finite-difference tolerances,
    /// so this must never depend on which pass happens to call it today.
    #[test]
    fn declines_backward_pass_even_when_f16_is_fast() {
        let p = NativeF16Provider::from_arch(fast_arch());
        let mut req = matmul_forward_req(DType::F32);
        req.pass = Pass::Backward;
        assert!(!p.accepts(&req, false));
    }

    /// This provider only ever claims the plain-fp32-shaped forward GEMM
    /// `matmul_reg3` already serves (see this module's own doc comment on
    /// why) - a request tagged with a different nominal dtype (the
    /// storage-tier `Dtype::F16` decode, a totally different buffer layout)
    /// must not be silently accepted.
    #[test]
    fn declines_a_non_f32_shaped_request_even_when_f16_is_fast() {
        let p = NativeF16Provider::from_arch(fast_arch());
        assert!(!p.accepts(&matmul_forward_req(DType::F16), false));
    }

    /// `Self::threads` mirrors `wgsl::WgslProvider::threads`'s own
    /// `RegisterTiled` tile formula exactly - this kernel is tiled
    /// byte-identically to `matmul_reg3`, so under-dispatching (the specific
    /// bug that formula's own regression test guards) would silently corrupt
    /// output the same way here.
    #[test]
    fn threads_uses_the_tile_formula_not_m_times_n() {
        let (m, n) = (513u32, 257u32);
        let expected = m.div_ceil(128) * n.div_ceil(128) * 256;
        assert_ne!(expected, m * n, "test shape must distinguish tile() from m*n");
        assert_eq!(NativeF16Provider::threads(m, n), expected);
    }

    /// `kernel()` wraps `MATMUL_REG3_F16N` with `enable f16;` and reports it
    /// under `KERNEL_NAME` - the exact pairing `lower_matmul` looks up by
    /// name, pinned so the two can never silently drift apart.
    #[test]
    fn kernel_registers_under_kernel_name_with_enable_f16_prepended() {
        let (name, src) = kernel();
        assert_eq!(name, KERNEL_NAME);
        assert!(src.starts_with("enable f16;\n"));
        assert!(src.ends_with(kernels::template::native_f16_matmul::MATMUL_REG3_F16N));
    }
}
