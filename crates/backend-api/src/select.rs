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

/// The int8 GEMV/tile crossover sits LOWER than fp32's: the packed GEMV
/// accumulates through workgroup memory (one read-modify-write per row per
/// K-group), so its per-row cost grows faster than the register-tiled GEMM's.
/// Measured on a P40 (qwen-synth 8x512x8, decode_heavy): GEMV 299 vs tile 225
/// tok/s at m=4, tile 753 vs GEMV 592 at m=16. A per-device autotuner (S5)
/// owns refining this boundary.
pub const I8_GEMV_MAX_ROWS: u32 = 8;

/// Every variant that can EXECUTE for `(op, shape)` on this device, with the
/// static best guess FIRST. Never empty: `Reference` is always executable.
///
/// This is both the default policy (its head) and the autotuner's probe list
/// (its tail): a measuring selector times exactly these — never a variant the
/// device cannot run, which is what keeps tuning a refinement rather than a
/// correctness risk. Ordering rules are either correctness gates (a capability
/// the variant requires) or measured regime boundaries; nothing keys on a
/// backend name.
pub fn candidates(op: Op, shape: OpShape, caps: &DeviceCaps) -> Vec<KernelVariant> {
    use KernelVariant::*;
    match op {
        Op::MatMul => match shape.dtype {
            // Int8 GEMMs only where the packed-dot kernels execute; a device
            // without them gets the fp32 reference (the caller keeps fp32
            // weights in that case — see qwen::serve). Within int8, the split
            // mirrors fp32: the 128x128 tile is mostly idle at decode row
            // counts, but the packed GEMV's workgroup-memory accumulation
            // grows per-row — the measured P40 crossover is m≈8, and refining
            // it per device is exactly what the autotuner probes this tail
            // for. The GEMV requires m <= 32 (its accumulator bound).
            Dtype::I8 if caps.numeric.int8_dot => {
                if shape.m > DECODE_REGIME_MAX_ROWS || !caps.workgroup_reductions {
                    vec![PackedInt8]
                } else if shape.m <= I8_GEMV_MAX_ROWS {
                    vec![WorkgroupPerOutput, PackedInt8]
                } else {
                    vec![PackedInt8, WorkgroupPerOutput]
                }
            }
            Dtype::I8 => vec![Reference],
            Dtype::F32 if shape.m <= DECODE_REGIME_MAX_ROWS && caps.workgroup_reductions => {
                vec![WorkgroupPerOutput, Reference]
            }
            Dtype::F32 => vec![Reference],
        },
        Op::RmsNorm => {
            if shape.m <= DECODE_REGIME_MAX_ROWS && caps.workgroup_reductions {
                vec![WorkgroupPerOutput, Reference]
            } else {
                vec![Reference]
            }
        }
        // Device-independent: the split kernels have no barrier, so the
        // boundary is purely the row length.
        Op::ArgMaxRow => {
            if shape.n >= ARGMAX_SPLIT_MIN_VOCAB {
                vec![SplitReduction, Reference]
            } else {
                vec![Reference, SplitReduction]
            }
        }
    }
}

/// The static default policy — the head of [`candidates`], BY CONSTRUCTION:
/// there is one list, so the default choice and the tuner's probe set can
/// never drift apart.
pub struct DefaultSelector;

impl KernelSelector for DefaultSelector {
    fn select(&self, op: Op, shape: OpShape, caps: &DeviceCaps) -> KernelVariant {
        candidates(op, shape, caps)[0]
    }
}

impl KernelVariant {
    /// Stable name for persistence (the tune cache stores these).
    pub fn as_str(self) -> &'static str {
        match self {
            KernelVariant::Reference => "reference",
            KernelVariant::WorkgroupPerOutput => "workgroup_per_output",
            KernelVariant::SplitReduction => "split_reduction",
            KernelVariant::PackedInt8 => "packed_int8",
        }
    }
    pub fn from_str(s: &str) -> Option<KernelVariant> {
        Some(match s {
            "reference" => KernelVariant::Reference,
            "workgroup_per_output" => KernelVariant::WorkgroupPerOutput,
            "split_reduction" => KernelVariant::SplitReduction,
            "packed_int8" => KernelVariant::PackedInt8,
            _ => return None,
        })
    }
}

/// Persistence for measured kernel choices — a trait so this crate stays
/// dependency-free; the file-backed implementation lives in `gpu-core`
/// (`tune::FileTuneStore`), keyed per adapter.
pub trait TuneStore: Send + Sync {
    fn load(&self, key: &str) -> Option<String>;
    fn save(&self, key: &str, value: &str);
}

/// Measure-once-and-remember kernel selection (S5): the selector's model of a
/// device is always approximate, so the right choice among [`candidates`] is
/// empirical.
///
/// `resolve` consults its memo, then the persistent store, and only then
/// measures — each candidate once via the caller's closure (the caller owns
/// dispatch; this type owns policy). The winner is remembered and persisted.
/// A measurement that fails (`None`) removes that candidate from contention;
/// if nothing measures, the static best guess stands. `BRAIN_NO_AUTOTUNE=1`
/// (read once at construction) forces the static selector everywhere — what
/// CI and reproducible benchmarking use, so an autotuned result can never make
/// a benchmark unreproducible. A persisted value that is not among today's
/// candidates is IGNORED, never trusted — the stale-cache rule.
pub struct AutoTuner {
    enabled: bool,
    store: Option<Box<dyn TuneStore>>,
    memo: std::sync::Mutex<std::collections::HashMap<(Op, OpShape), KernelVariant>>,
}

impl AutoTuner {
    pub fn new(store: Option<Box<dyn TuneStore>>) -> AutoTuner {
        let enabled = std::env::var("BRAIN_NO_AUTOTUNE").map(|v| v == "0").unwrap_or(true);
        AutoTuner { enabled, store, memo: Default::default() }
    }

    /// A tuner that never measures — the static policy with the same API.
    pub fn disabled() -> AutoTuner {
        AutoTuner { enabled: false, store: None, memo: Default::default() }
    }

    fn key(op: Op, shape: OpShape) -> String {
        format!("{:?}/{:?}/{}x{}x{}", op, shape.dtype, shape.m, shape.n, shape.k)
    }

    /// The measured-best variant for `(op, shape)`. `measure` returns the cost
    /// (lower is better; typically milliseconds) of running one candidate, or
    /// `None` if it could not be measured.
    pub fn resolve(
        &self,
        op: Op,
        shape: OpShape,
        caps: &DeviceCaps,
        measure: &mut dyn FnMut(KernelVariant) -> Option<f64>,
    ) -> KernelVariant {
        let cands = candidates(op, shape, caps);
        if !self.enabled || cands.len() < 2 {
            return cands[0];
        }
        if let Some(&hit) = self.memo.lock().unwrap().get(&(op, shape)) {
            return hit;
        }
        let key = Self::key(op, shape);
        if let Some(stored) = self.store.as_ref().and_then(|s| s.load(&key)) {
            if let Some(v) = KernelVariant::from_str(&stored).filter(|v| cands.contains(v)) {
                self.memo.lock().unwrap().insert((op, shape), v);
                return v;
            }
        }
        let mut best = cands[0];
        let mut best_cost = f64::INFINITY;
        for &c in &cands {
            if let Some(cost) = measure(c) {
                if cost < best_cost {
                    best_cost = cost;
                    best = c;
                }
            }
        }
        self.memo.lock().unwrap().insert((op, shape), best);
        if let Some(s) = &self.store {
            s.save(&key, best.as_str());
        }
        best
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

    /// Int8 selects the packed kernels only where they execute — and splits
    /// by regime exactly like fp32: the packed GEMV at decode row counts (a
    /// 128x128 tile is mostly idle at m=8), the tile GEMM at prefill shapes.
    #[test]
    fn int8_requires_the_capability() {
        let s = DefaultSelector;
        let decode = shape(8, 512, 512, Dtype::I8);
        let prefill = shape(512, 512, 512, Dtype::I8);
        assert_eq!(s.select(Op::MatMul, decode, &gpu_caps()), KernelVariant::WorkgroupPerOutput);
        assert_eq!(s.select(Op::MatMul, prefill, &gpu_caps()), KernelVariant::PackedInt8);
        assert_eq!(s.select(Op::MatMul, decode, &cpu_caps()), KernelVariant::Reference);
        assert_eq!(s.select(Op::MatMul, prefill, &cpu_caps()), KernelVariant::Reference);
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

    /// The candidate list is never empty and its head IS the default policy —
    /// one list, so the static choice and the tuner's probe set cannot drift.
    #[test]
    fn candidates_head_is_the_default_policy() {
        let s = DefaultSelector;
        for caps in [gpu_caps(), cpu_caps()] {
            for op in [Op::MatMul, Op::RmsNorm, Op::ArgMaxRow] {
                for m in [1u32, 8, 9, 33, 4096] {
                    for dtype in [Dtype::F32, Dtype::I8] {
                        let sh = shape(m, 8192, 512, dtype);
                        let c = candidates(op, sh, &caps);
                        assert!(!c.is_empty());
                        assert_eq!(s.select(op, sh, &caps), c[0]);
                    }
                }
            }
        }
    }

    /// The tuner picks the measured minimum, measures each candidate exactly
    /// once per shape (memoised after), persists the winner, and trusts a
    /// stored value only if it is still a valid candidate today.
    #[test]
    fn autotuner_measures_once_and_persists() {
        use std::sync::Mutex;
        #[derive(Default)]
        struct MapStore(Mutex<std::collections::HashMap<String, String>>);
        impl TuneStore for MapStore {
            fn load(&self, k: &str) -> Option<String> {
                self.0.lock().unwrap().get(k).cloned()
            }
            fn save(&self, k: &str, v: &str) {
                self.0.lock().unwrap().insert(k.into(), v.into());
            }
        }
        let store = Box::leak(Box::new(MapStore::default()));
        // (Deliberately reborrow as a plain reference impl for the Box.)
        struct Ref(&'static MapStore);
        impl TuneStore for Ref {
            fn load(&self, k: &str) -> Option<String> {
                self.0.load(k)
            }
            fn save(&self, k: &str, v: &str) {
                self.0.save(k, v)
            }
        }
        let t = AutoTuner { enabled: true, store: Some(Box::new(Ref(store))), memo: Default::default() };
        let sh = shape(4, 512, 512, Dtype::I8); // static best: WorkgroupPerOutput
        let caps = gpu_caps();
        let mut calls = 0;
        // The tile GEMM measures FASTER here, overriding the static guess.
        let mut measure = |v: KernelVariant| {
            calls += 1;
            Some(if v == KernelVariant::PackedInt8 { 1.0 } else { 2.0 })
        };
        assert_eq!(t.resolve(Op::MatMul, sh, &caps, &mut measure), KernelVariant::PackedInt8);
        assert_eq!(t.resolve(Op::MatMul, sh, &caps, &mut measure), KernelVariant::PackedInt8);
        assert_eq!(calls, 2, "both candidates measured once; the second resolve is a memo hit");
        // A fresh tuner sharing the store trusts the persisted winner without
        // measuring at all.
        let t2 = AutoTuner { enabled: true, store: Some(Box::new(Ref(store))), memo: Default::default() };
        let mut no_measure = |_: KernelVariant| -> Option<f64> { panic!("stored winner must be reused") };
        assert_eq!(t2.resolve(Op::MatMul, sh, &caps, &mut no_measure), KernelVariant::PackedInt8);
        // Disabled tuner = the static policy, zero measurements.
        let td = AutoTuner::disabled();
        let mut no_measure2 =
            |_: KernelVariant| -> Option<f64> { panic!("disabled tuner must not measure") };
        assert_eq!(
            td.resolve(Op::MatMul, sh, &caps, &mut no_measure2),
            KernelVariant::WorkgroupPerOutput,
            "disabled = the static best guess"
        );
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
