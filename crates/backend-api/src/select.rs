// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kernel selection — *which implementation of an op runs, given the shape and
//! the device.*
//!
//! Before this seam existed, every call site hand-wrote its regime test
//! (`if m <= 32 && gpu.kind() != "cpu" { … }`), which meant the policy was
//! scattered, untestable without a device, and keyed on the backend's *name*
//! rather than on what the device can actually do.
//!
//! The seam is deliberately small:
//!
//! * [`Op`] + [`OpShape`] name a logical operation and the shape that decides
//!   which variant wins.
//! * [`KernelVariant`] is the selected implementation *family*. Kernel sets
//!   (and their pipeline indices) are per-model, so the selector cannot return
//!   an index — the call site maps the family to its own index. A family the
//!   call site has no kernel for falls back to `Reference`, which every model
//!   ships by construction.
//! * [`KernelSelector`] is a trait so tests can install a deterministic policy
//!   ([`AlwaysReference`]) and a future autotuner can implement the same seam
//!   with measured choices (S5).
//!
//! The default policy ([`DefaultSelector`]) is a pure function of its inputs —
//! unit-testable with no device at all — and its device inputs come from
//! [`DeviceCaps`](crate::DeviceCaps), never from backend names.

use crate::DeviceCaps;

/// One logical operation, independent of how it is implemented.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    MatMul,
    RmsNorm,
    ArgMaxRow,
}

/// Element type an op runs over.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dtype {
    F32,
    /// Packed int8 (weights quantised per `qwen::q8` / `matmul_i8`).
    I8,
}

/// The shape that decides which variant wins. For a GEMM, `m×k @ k×n`; for a
/// row-wise op, `m` rows of `n` elements (`k` unused, 0).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OpShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: Dtype,
}

/// The selected implementation family. The call site maps this to its own
/// pipeline index; a family it has no kernel for falls back to `Reference`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelVariant {
    /// The portable per-element / per-row kernel — always present, always
    /// correct, the baseline every other family is checked against.
    Reference,
    /// Workgroup-cooperative variant shaped for FEW rows (the decode regime):
    /// one workgroup per output row/column, threads split the reduction axis
    /// behind a single barrier. Requires `caps.workgroup_reductions`.
    WorkgroupPerOutput,
    /// Two-dispatch split reduction (`*_part` → `*_final`); no barrier at all,
    /// so it runs on every backend.
    SplitReduction,
    /// The packed-int8 register-tiled GEMM (`matmul_i8*`). Requires
    /// `caps.numeric.int8_dot`.
    PackedInt8,
}

/// Picks a [`KernelVariant`] for an op/shape on a device.
///
/// Implementations must be pure with respect to their inputs — selection is
/// memoised per distinct `(Op, OpShape)` (see [`CachedSelector`]), so a
/// stateful answer would be frozen at first use anyway.
pub trait KernelSelector: Send + Sync {
    fn select(&self, op: Op, shape: OpShape, caps: &DeviceCaps) -> KernelVariant;
}

/// Rows at or below this run in the decode regime: the per-element reference
/// kernels degenerate there (`rmsnorm` = one thread per row — 8 threads on a
/// 3840-core card at batch 8), and the workgroup-cooperative variants win.
/// Above it, M is large enough that the reference kernels saturate the device.
pub const DECODE_REGIME_MAX_ROWS: u32 = 32;

/// Below this vocabulary the single-pass argmax wins: two dispatches cost more
/// than they save on a short row.
pub const ARGMAX_SPLIT_MIN_VOCAB: u32 = 4096;

/// The static default policy — the measured rules from the decode-regime work,
/// expressed once. Every rule is either a correctness gate (a capability the
/// variant requires) or a measured regime boundary; nothing keys on a backend
/// name.
pub struct DefaultSelector;

impl KernelSelector for DefaultSelector {
    fn select(&self, op: Op, shape: OpShape, caps: &DeviceCaps) -> KernelVariant {
        match op {
            Op::MatMul => match shape.dtype {
                // Int8 GEMM only where the packed-dot kernels execute; a
                // device without them gets the fp32 reference (the caller
                // keeps fp32 weights in that case — see qwen::serve).
                Dtype::I8 if caps.numeric.int8_dot => KernelVariant::PackedInt8,
                Dtype::I8 => KernelVariant::Reference,
                Dtype::F32
                    if shape.m <= DECODE_REGIME_MAX_ROWS && caps.workgroup_reductions =>
                {
                    KernelVariant::WorkgroupPerOutput
                }
                Dtype::F32 => KernelVariant::Reference,
            },
            Op::RmsNorm => {
                if shape.m <= DECODE_REGIME_MAX_ROWS && caps.workgroup_reductions {
                    KernelVariant::WorkgroupPerOutput
                } else {
                    KernelVariant::Reference
                }
            }
            // Device-independent: the split kernels have no barrier, so the
            // boundary is purely the row length.
            Op::ArgMaxRow => {
                if shape.n >= ARGMAX_SPLIT_MIN_VOCAB {
                    KernelVariant::SplitReduction
                } else {
                    KernelVariant::Reference
                }
            }
        }
    }
}

/// Always the reference kernel — the deterministic policy tests install to pin
/// behaviour independent of device or tuning.
pub struct AlwaysReference;

impl KernelSelector for AlwaysReference {
    fn select(&self, _op: Op, _shape: OpShape, _caps: &DeviceCaps) -> KernelVariant {
        KernelVariant::Reference
    }
}

/// Memoises an inner selector per distinct `(Op, OpShape)`.
///
/// Decode shapes are few and fixed, so after warm-up every selection is one
/// `HashMap` hit — never a per-dispatch policy walk. This matters little for
/// the pure default policy and a lot for a future autotuner, which is exactly
/// why the memo lives here in the seam rather than in each policy.
pub struct CachedSelector<S> {
    inner: S,
    memo: std::sync::Mutex<std::collections::HashMap<(Op, OpShape), KernelVariant>>,
}

impl<S: KernelSelector> CachedSelector<S> {
    pub fn new(inner: S) -> CachedSelector<S> {
        CachedSelector { inner, memo: std::sync::Mutex::new(std::collections::HashMap::new()) }
    }
}

impl<S: KernelSelector> KernelSelector for CachedSelector<S> {
    fn select(&self, op: Op, shape: OpShape, caps: &DeviceCaps) -> KernelVariant {
        let mut memo = self.memo.lock().unwrap_or_else(|e| e.into_inner());
        *memo.entry((op, shape)).or_insert_with(|| self.inner.select(op, shape, caps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceClass, NumericSupport};

    fn gpu_caps() -> DeviceCaps {
        let mut c = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
        c.numeric = NumericSupport { f32: true, int8_dot: true, f16: false, coop_matrix: false };
        c
    }

    fn cpu_caps() -> DeviceCaps {
        let mut c = DeviceCaps::portable_baseline(DeviceClass::Cpu);
        c.workgroup_reductions = false;
        c
    }

    fn shape(m: u32, n: u32, k: u32, dtype: Dtype) -> OpShape {
        OpShape { m, n, k, dtype }
    }

    /// The decode regime picks the cooperative kernels on a GPU and must NOT
    /// pick them where workgroup reductions mis-execute (the CPU JIT).
    #[test]
    fn decode_regime_is_capability_gated() {
        let s = DefaultSelector;
        let decode = shape(8, 512, 512, Dtype::F32);
        assert_eq!(s.select(Op::MatMul, decode, &gpu_caps()), KernelVariant::WorkgroupPerOutput);
        assert_eq!(s.select(Op::RmsNorm, decode, &gpu_caps()), KernelVariant::WorkgroupPerOutput);
        assert_eq!(s.select(Op::MatMul, decode, &cpu_caps()), KernelVariant::Reference);
        assert_eq!(s.select(Op::RmsNorm, decode, &cpu_caps()), KernelVariant::Reference);
    }

    /// Training-sized M keeps the reference kernels — the regime boundary is
    /// the whole point of having a shape input.
    #[test]
    fn large_m_keeps_reference() {
        let s = DefaultSelector;
        let train = shape(4096, 512, 512, Dtype::F32);
        assert_eq!(s.select(Op::MatMul, train, &gpu_caps()), KernelVariant::Reference);
        assert_eq!(s.select(Op::RmsNorm, train, &gpu_caps()), KernelVariant::Reference);
    }

    /// Int8 selects the packed GEMM only where the packed-dot kernels execute;
    /// claiming it elsewhere would dispatch a kernel the device cannot run.
    #[test]
    fn int8_requires_the_capability() {
        let s = DefaultSelector;
        let sh = shape(8, 512, 512, Dtype::I8);
        assert_eq!(s.select(Op::MatMul, sh, &gpu_caps()), KernelVariant::PackedInt8);
        assert_eq!(s.select(Op::MatMul, sh, &cpu_caps()), KernelVariant::Reference);
    }

    /// Argmax splits by row length alone — the split kernels are barrier-free,
    /// so every backend takes the same boundary.
    #[test]
    fn argmax_splits_on_vocab() {
        let s = DefaultSelector;
        for caps in [gpu_caps(), cpu_caps()] {
            assert_eq!(
                s.select(Op::ArgMaxRow, shape(4, 151_936, 0, Dtype::F32), &caps),
                KernelVariant::SplitReduction
            );
            assert_eq!(
                s.select(Op::ArgMaxRow, shape(4, 64, 0, Dtype::F32), &caps),
                KernelVariant::Reference
            );
        }
    }

    /// The memo consults its inner policy once per distinct (op, shape).
    #[test]
    fn cache_memoises_per_shape() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counting(AtomicUsize);
        impl KernelSelector for Counting {
            fn select(&self, _: Op, _: OpShape, _: &DeviceCaps) -> KernelVariant {
                self.0.fetch_add(1, Ordering::Relaxed);
                KernelVariant::Reference
            }
        }
        let s = CachedSelector::new(Counting(AtomicUsize::new(0)));
        let sh = shape(8, 512, 512, Dtype::F32);
        let caps = gpu_caps();
        s.select(Op::MatMul, sh, &caps);
        s.select(Op::MatMul, sh, &caps);
        s.select(Op::MatMul, shape(9, 512, 512, Dtype::F32), &caps);
        assert_eq!(s.inner.0.load(Ordering::Relaxed), 2, "one call per distinct shape");
    }
}
