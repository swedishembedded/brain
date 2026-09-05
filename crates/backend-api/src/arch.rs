// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ArchDesc` - what a device's arithmetic actually is, per [`DType`] tier,
//! replacing the bool-flattened semantics [`NumericSupport`] used to carry on
//! its own.
//!
//! **The conflation this fixes.** `NumericSupport::int8_dot` was one bool for
//! two different facts: "the packed-int8 kernels execute" (true on every wgpu
//! target - `dot4I8Packed` is core WGSL, naga lowers it to hardware DP4A or a
//! polyfill) and "this runs on dedicated int8 hardware" (true only where a
//! backend queried a real DP4A property). `backend-wgpu` hardcoded the first
//! meaning; `backend-vulkan` reported the second. A selector reading one bool
//! could not tell a real DP4A P40 from a polyfilling browser. [`TierLevel`]
//! separates the two facts into distinct levels instead of one bool per
//! dtype: `Emulated` ("executes, no claim about speed") and `Native` ("a
//! real device query says dedicated hardware runs this") are different
//! levels, and a selector that only needs "does it run at all"
//! ([`ArchDesc::executes`]) gets the same answer as before, while a selector
//! that needs to prefer real hardware over a polyfill now has something to
//! read.
//!
//! [`ArchDesc::numeric_view`] is the ONLY function that may ever produce a
//! [`NumericSupport`] value from a queried device: it is the single derived
//! view every backend's `query_caps`/`caps` builds through
//! (`DeviceCaps::numeric` and `DeviceCaps::arch` are filled from the exact
//! same [`ArchDesc`] this way), so the two views can never independently
//! drift the way the standalone bools used to invite.

use crate::{DType, NumericSupport};

/// The margin a measured `speedup_vs_f32` must clear before
/// [`ArchDesc::is_fast`] calls a tier fast - moved here (from wherever a
/// single backend used to define its own f16 margin) because "measured, not
/// marketed" is a property of the tier system itself, not of one backend's
/// f16 gate. `1.2` is the same bar `backend_wgpu::WgpuBackend::
/// F16_COMPUTE_MIN_SPEEDUP` already used (that constant is now DEFINED from
/// this one, with no cast, so the two can never independently drift):
/// comfortably outside the run-to-run noise a warmed, idle device shows on
/// repeated identical dispatches, while still requiring a genuinely fast
/// path, not merely "not slower".
///
/// `f64`, not `f32`, even though [`TierSupport::speedup_vs_f32`] is `f32`:
/// `backend-wgpu`'s own gate (and its test suite) is `f64`-typed, and `1.2`
/// is not exactly representable in either width - going `f64 -> f32` at the
/// comparison site in [`ArchDesc::is_fast`] is an exact, lossless narrowing
/// of an already-decided value, while the reverse (`f32 -> f64`) would have
/// resurrected a value that was never actually `1.2`, just the nearest
/// `f32` to it. Widest-first is what makes "one source" true bit for bit
/// instead of true up to a rounding error - the discovered failure mode
/// this comment exists to head off.
pub const FAST_TIER_MIN_SPEEDUP: f64 = 1.2;

/// How a [`DType`] sits on one device, from "cannot even be read" to "a
/// matrix engine consumes it". Ordered so a selector can write `level >=
/// TierLevel::Emulated` instead of matching every variant that "runs" - see
/// [`ArchDesc::executes`]/[`ArchDesc::holds`].
///
/// **None of these levels are speed claims** except implicitly through
/// [`TierSupport::speedup_vs_f32`] - `Native` means a real device query
/// found dedicated hardware for this tier, not that it is fast (Pascal's
/// `f16` is real, queried, native... at 1/64 rate). Whether a tier is
/// actually worth preferring over fp32 is answered by [`ArchDesc::is_fast`]
/// alone, which never looks at `level`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub enum TierLevel {
    /// The device cannot hold or execute this tier at all.
    #[default]
    Absent,
    /// Bytes held/decoded; arithmetic actually happens in a WIDER tier (the
    /// storage-tier `#w=bf16`/`#w=f16` decode: plain integer/bitcast WGSL,
    /// no device feature, computes in fp32).
    Storage,
    /// Arithmetic executes AT this tier, by polyfill or software emulation -
    /// deliberately not a speed claim (the naga-polyfilled `dot4I8Packed`
    /// case this type exists to separate from `Native`).
    Emulated,
    /// Arithmetic executes at this tier on hardware dedicated to it,
    /// established by a real device query (a queried DP4A property, a
    /// queried `shaderFloat16` bit) - not a measured rate.
    Native,
    /// A matrix/tensor engine consumes this tier directly (implies
    /// `Native` - see [`ArchDesc::matrix`] for the shapes it accepts).
    Matrix,
}

/// One [`DType`]'s tier on a device: the qualitative [`TierLevel`] plus
/// whatever has actually been MEASURED for it. Both measured fields start
/// `None` - "unmeasured" is a distinct state from "measured and slow", since
/// a caller that skips measuring must never read `None` as a negative
/// result.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TierSupport {
    pub level: TierLevel,
    /// Measured throughput, GOP/s. `None` until something measures it -
    /// never derived from a marketing figure, matching
    /// `DeviceCaps::peak_gflops`'s own rule.
    pub rate_gops: Option<f32>,
    /// Measured speedup over the fp32 baseline at the same shape. `None`
    /// until measured; [`ArchDesc::is_fast`] treats `None` as "not fast",
    /// never as a guess.
    pub speedup_vs_f32: Option<f32>,
}

/// CPU SIMD/matrix-extension bits this device's core detected, reusing
/// whatever runtime feature probes already exist (`backend-cpu::fast_conv`'s
/// `is_x86_feature_detected!` calls) rather than re-probing CPUID here.
/// Every field defaults `false`/`None` - "not detected" - since this engine
/// has no VNNI/AMX/NEON probe of its own yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IsaFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512_vnni: bool,
    pub avx512_bf16: bool,
    pub amx_bf16: bool,
    pub amx_int8: bool,
    pub neon: bool,
    pub neon_dotprod: bool,
    pub sve_bits: Option<u32>,
}

/// One matrix-engine shape: the `(a, b, accum)` dtype triple it multiplies
/// and the `m×k @ k×n` tile it is queried to support, mirroring
/// `brain_vulkan::context::CoopMatShape`'s fields (the only real enumeration
/// this engine has today) without duplicating its query.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: DType,
    pub b: DType,
    pub accum: DType,
    /// Subgroup/SIMD width this shape is scoped to, where the query
    /// distinguishes scopes (`None` where the query does not expose one).
    pub scope_width: Option<u32>,
}

/// Which physical matrix engine [`MatrixEngine::shapes`] were enumerated
/// from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixKind {
    CoopMatrix,
    Amx,
    Npu,
}

/// A device's matrix/tensor engine and the shapes it was queried to accept.
/// `None` on [`ArchDesc::matrix`] means no matrix engine was found at all -
/// not "unqueried" (every backend that populates an [`ArchDesc`] queries
/// this at construction time).
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixEngine {
    pub kind: MatrixKind,
    pub shapes: Vec<MatShape>,
}

/// Elements [`DType`] declares, in exactly its declaration order - the one
/// place this file assumes that order, so a new `DType` variant only ever
/// needs a new match arm plus incrementing this constant, never a silent
/// array-bounds trap. [`tests::every_dtype_round_trips_its_own_tier`] below
/// asserts the same count against this file's own indexing.
const DTYPE_COUNT: usize = 7;

/// What one device's arithmetic actually is, per [`DType`]: the ONE seam a
/// selector reads, replacing [`NumericSupport`]'s flattened bools. Filled
/// once at backend construction from real device queries only - no
/// measurement, no timing - and cached on [`crate::DeviceCaps`] alongside the
/// [`NumericSupport`] view [`Self::numeric_view`] derives from it.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchDesc {
    tiers: [TierSupport; DTYPE_COUNT],
    pub matrix: Option<MatrixEngine>,
    pub isa: IsaFeatures,
}

impl Default for ArchDesc {
    /// Every tier `Absent`, no matrix engine, no ISA feature - the same
    /// portable floor [`NumericSupport::BASELINE`] describes, and
    /// [`Self::numeric_view`] maps this value to exactly that constant (see
    /// [`tests::default_arch_numeric_view_is_the_baseline`]).
    fn default() -> Self {
        ArchDesc { tiers: [TierSupport::default(); DTYPE_COUNT], matrix: None, isa: IsaFeatures::default() }
    }
}

impl ArchDesc {
    pub fn tier(&self, dt: DType) -> TierSupport {
        self.tiers[dt as usize]
    }

    pub fn set_tier(&mut self, dt: DType, t: TierSupport) {
        self.tiers[dt as usize] = t;
    }

    /// `dt`'s arithmetic runs on this device at all - by polyfill or by
    /// dedicated hardware, no speed claim either way. The direct replacement
    /// for reading `NumericSupport::int8_dot` as "does this execute".
    pub fn executes(&self, dt: DType) -> bool {
        self.tier(dt).level >= TierLevel::Emulated
    }

    /// `dt`'s bytes can be held/read on this device, even if arithmetic on
    /// them happens in a wider tier.
    pub fn holds(&self, dt: DType) -> bool {
        self.tier(dt).level >= TierLevel::Storage
    }

    /// `dt` cleared [`FAST_TIER_MIN_SPEEDUP`] in a REAL measurement -
    /// "measured, not marketed", the same rule `NumericSupport::f16`'s own
    /// doc already stated. `None` (never measured) reads as not-fast, never
    /// as a guess.
    pub fn is_fast(&self, dt: DType) -> bool {
        self.tier(dt).speedup_vs_f32.is_some_and(|s| s >= FAST_TIER_MIN_SPEEDUP as f32)
    }

    /// The backward-compatible derived view every backend's `caps()` must go
    /// through instead of constructing a [`NumericSupport`] by hand.
    ///
    /// This is NOT simply "`executes`/`holds`/`is_fast` for every field" -
    /// two of the seven fields are deliberately narrower than their generic
    /// `ArchDesc` counterpart, because this function's job is fidelity to
    /// what each real backend has always reported, not to be the general
    /// "does this device have `dt`" answer (that answer is [`Self::holds`]
    /// itself, directly, for any FUTURE caller that wants it).
    ///
    /// `f16_storage`/`bf16_storage` check the tier is EXACTLY `Storage`, not
    /// `>= Storage`. No backend has ever reported both "this tier is
    /// storage-only" and "this tier is native" at once for the same dtype -
    /// `backend-vulkan`'s real `F16` tier is `Native` (a queried
    /// `shaderFloat16` bit) while its `NumericSupport::f16_storage` has
    /// always stayed `false` (nothing there ever set it); an exact-match
    /// keeps reproducing that `false` instead of flipping it true the
    /// moment `Native` also implies "and it can obviously be stored".
    ///
    /// `int8_dot` and `coop_matrix`, in contrast, ARE genuinely
    /// `executes(I8)` / `matrix.is_some()` - these two are the fields this
    /// milestone fixes the meaning of (see this module's doc comment), so
    /// unlike the storage pair above they are deliberately the general form.
    pub fn numeric_view(&self) -> NumericSupport {
        NumericSupport {
            f32: true,
            int8_dot: self.executes(DType::I8),
            f16: self.is_fast(DType::F16),
            bf16: self.is_fast(DType::BF16),
            f16_storage: self.tier(DType::F16).level == TierLevel::Storage,
            bf16_storage: self.tier(DType::BF16).level == TierLevel::Storage,
            coop_matrix: self.matrix.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_arch_numeric_view_is_the_baseline() {
        assert_eq!(ArchDesc::default().numeric_view(), NumericSupport::BASELINE);
    }

    #[test]
    fn every_dtype_round_trips_its_own_tier() {
        let dtypes =
            [DType::F32, DType::F16, DType::BF16, DType::I8, DType::Q4, DType::Q4K, DType::Q8K];
        assert_eq!(dtypes.len(), DTYPE_COUNT, "DType grew a variant this file's indexing did not follow");
        let mut arch = ArchDesc::default();
        for (i, &dt) in dtypes.iter().enumerate() {
            let t = TierSupport {
                level: TierLevel::Native,
                rate_gops: Some(i as f32),
                speedup_vs_f32: Some(i as f32),
            };
            arch.set_tier(dt, t);
        }
        for (i, &dt) in dtypes.iter().enumerate() {
            let t = arch.tier(dt);
            assert_eq!(t.level, TierLevel::Native);
            assert_eq!(t.rate_gops, Some(i as f32));
            assert_eq!(t.speedup_vs_f32, Some(i as f32));
        }
    }

    #[test]
    fn tier_level_orders_absent_below_storage_below_emulated_below_native_below_matrix() {
        assert!(TierLevel::Absent < TierLevel::Storage);
        assert!(TierLevel::Storage < TierLevel::Emulated);
        assert!(TierLevel::Emulated < TierLevel::Native);
        assert!(TierLevel::Native < TierLevel::Matrix);
    }

    #[test]
    fn executes_is_true_at_emulated_and_every_level_above_it() {
        for level in [TierLevel::Emulated, TierLevel::Native, TierLevel::Matrix] {
            let mut arch = ArchDesc::default();
            arch.set_tier(DType::I8, TierSupport { level, ..Default::default() });
            assert!(arch.executes(DType::I8), "{level:?} must execute");
        }
        for level in [TierLevel::Absent, TierLevel::Storage] {
            let mut arch = ArchDesc::default();
            arch.set_tier(DType::I8, TierSupport { level, ..Default::default() });
            assert!(!arch.executes(DType::I8), "{level:?} must not execute");
        }
    }

    #[test]
    fn is_fast_requires_a_real_measurement_at_or_above_the_threshold() {
        let mut arch = ArchDesc::default();
        arch.set_tier(
            DType::F16,
            TierSupport { level: TierLevel::Native, speedup_vs_f32: None, ..Default::default() },
        );
        assert!(!arch.is_fast(DType::F16), "Native alone (no measurement) must not read as fast");

        arch.set_tier(
            DType::F16,
            TierSupport { level: TierLevel::Native, speedup_vs_f32: Some(1.0), ..Default::default() },
        );
        assert!(!arch.is_fast(DType::F16), "below the margin must not read as fast");

        arch.set_tier(
            DType::F16,
            TierSupport {
                level: TierLevel::Native,
                speedup_vs_f32: Some(FAST_TIER_MIN_SPEEDUP as f32),
                ..Default::default()
            },
        );
        assert!(arch.is_fast(DType::F16), "at the margin must read as fast");
    }

    /// This is the M8.1 fix, pinned directly: a device whose `I8` tier is
    /// `Native` (real DP4A hardware) and one whose `I8` tier is `Emulated`
    /// (naga polyfill) now produce DIFFERENT `TierLevel`s even though
    /// `numeric_view().int8_dot` is `true` for both - the whole point being
    /// that a selector reading `ArchDesc::tier` directly can finally tell
    /// them apart, where reading the old flattened `NumericSupport::int8_dot`
    /// bool alone could not.
    #[test]
    fn native_and_emulated_i8_are_distinguishable_even_though_both_execute() {
        let mut native = ArchDesc::default();
        native.set_tier(DType::I8, TierSupport { level: TierLevel::Native, ..Default::default() });
        let mut emulated = ArchDesc::default();
        emulated.set_tier(DType::I8, TierSupport { level: TierLevel::Emulated, ..Default::default() });

        assert_ne!(native.tier(DType::I8).level, emulated.tier(DType::I8).level);
        assert!(native.executes(DType::I8) && emulated.executes(DType::I8));
        assert_eq!(native.numeric_view().int8_dot, emulated.numeric_view().int8_dot);
    }

    #[test]
    fn f16_storage_view_is_exact_match_not_at_least_storage() {
        let mut arch = ArchDesc::default();
        arch.set_tier(DType::F16, TierSupport { level: TierLevel::Storage, ..Default::default() });
        assert!(arch.numeric_view().f16_storage);

        // A backend whose F16 tier is Native (arithmetic genuinely executes,
        // established by a real query) does NOT get f16_storage flipped on
        // by that fact alone - see `Self::numeric_view`'s doc for why this
        // is deliberate, not an oversight.
        let mut native = ArchDesc::default();
        native.set_tier(DType::F16, TierSupport { level: TierLevel::Native, ..Default::default() });
        assert!(!native.numeric_view().f16_storage);
        assert!(native.holds(DType::F16), "holds() itself still says yes - only the legacy view is narrow");
    }

    #[test]
    fn coop_matrix_view_follows_matrix_presence() {
        let mut arch = ArchDesc::default();
        assert!(!arch.numeric_view().coop_matrix);
        arch.matrix = Some(MatrixEngine {
            kind: MatrixKind::CoopMatrix,
            shapes: vec![MatShape {
                m: 16,
                n: 16,
                k: 16,
                a: DType::F16,
                b: DType::F16,
                accum: DType::F32,
                scope_width: None,
            }],
        });
        assert!(arch.numeric_view().coop_matrix);
    }
}
