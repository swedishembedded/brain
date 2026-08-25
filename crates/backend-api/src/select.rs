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
    /// LayerNorm (forward `layernorm`, and the backward helpers `ln_stats` /
    /// `layernorm_dx` that share its row mapping). Selected by the same rule as
    /// [`Op::RmsNorm`] — see [`candidates`].
    LayerNorm,
    ArgMaxRow,
    /// Per-parameter sum-of-squares for the optimiser's global grad-norm clip
    /// (`gradnorm_sq` vs `gradnorm_part` + `clip_coef_wg`). Shape is `m = 1`
    /// row of `n = numel` — see [`candidates`] for why `n` does not gate it.
    GradNorm,
    /// Per-row max|x| for the int8 dynamic activation quant (`max_abs_row` vs
    /// `max_abs_rows`). `m` rows of `n = k` — see [`candidates`] for why
    /// neither gates it.
    MaxAbsRow,
    /// Forward 1D convolution, NCL (`conv1d` direct vs the GEMM lowering).
    /// Shape is the LOWERED GEMM's, not the conv's: `m` = output positions
    /// (`N*Lo`), `n` = `Cout`, `k` = the contraction (`Cin*K`). See
    /// [`candidates`] and [`GEMM_CONV1D_MIN_COUT`].
    Conv1d,
    /// Forward transposed 1D convolution, NCL (`convtr1d` direct vs the GEMM
    /// lowering). Shape is again the lowered GEMM's: `m` = INPUT positions
    /// (`N*L` - the transposed lowering does `L` rows, not `Lo`), `n` =
    /// `Cout`, `k` = `Cin`.
    ConvTranspose1d,
}

/// Element type an op runs over - an alias for the engine's ONE dtype enum
/// ([`crate::DType`]). This used to be a separate `{F32, I8}` enum; unifying
/// it means a checkpoint's declared width and a kernel's selection key are
/// the same type by construction, not two enums a caller could let drift.
pub type Dtype = crate::DType;

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
    /// The fp32/storage-tier 128×128 register-tiled GEMM (`matmul_reg`/
    /// `matmul_reg2`/`matmul_reg3` - call sites map this to whichever of the
    /// three they registered; all three are bit-identical by construction,
    /// see `matmul_reg3.wgsl`'s header comment). Wins once there is enough
    /// work to fill a tile - see [`GEMM_TILE_MIN_ROWS`]/[`GEMM_TILE_MIN_COLS`]
    /// for the measured crossover. Requires `caps.workgroup_reductions` (the
    /// kernel stages its tile in workgroup memory behind
    /// `workgroupBarrier()`).
    RegisterTiled,
}

/// What a [`KernelVariant`] needs from the device to correctly execute a
/// given [`Dtype`] - data, checked in ONE place ([`Requirement::satisfied_by`]),
/// rather than scattered `if caps.numeric.int8_dot` conditions inline in
/// [`candidates`]'s match arms. Every field defaults to `false` (no
/// constraint); [`KernelVariant::requires`] sets only the ones that matter
/// for a given (variant, dtype) pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Requirement {
    /// The packed-int8 dot kernels execute (`caps.numeric.int8_dot`).
    pub int8_dot: bool,
    /// *Fast* f16 arithmetic (`caps.numeric.f16`).
    pub f16_compute: bool,
    /// *Fast* bf16 arithmetic (`caps.numeric.bf16`).
    pub bf16_compute: bool,
    /// f16 bytes merely storable (`caps.numeric.f16_storage`) - also
    /// satisfied by fast f16 compute, since a device that computes f16 can
    /// certainly hold it (see [`Requirement::satisfied_by`]).
    pub f16_storage: bool,
    /// bf16 bytes merely storable (`caps.numeric.bf16_storage`), same rule.
    pub bf16_storage: bool,
    /// Workgroup-barrier reductions execute *correctly* on this device
    /// (`caps.workgroup_reductions`) - the CPU JIT's split-at-barrier
    /// execution model mis-executes these, so this is a correctness gate,
    /// not a preference.
    pub workgroup_reductions: bool,
}

impl Requirement {
    /// Whether every flag this requirement sets is actually true on `caps`.
    /// An unset flag imposes no constraint, so [`Requirement::default`]
    /// (nothing set) is always satisfied.
    pub fn satisfied_by(&self, caps: &DeviceCaps) -> bool {
        let n = &caps.numeric;
        (!self.int8_dot || n.int8_dot)
            && (!self.f16_compute || n.f16)
            && (!self.bf16_compute || n.bf16)
            && (!self.f16_storage || n.f16 || n.f16_storage)
            && (!self.bf16_storage || n.bf16 || n.bf16_storage)
            && (!self.workgroup_reductions || caps.workgroup_reductions)
    }
}

/// What it takes to merely HOLD/read `dt`'s bytes - the capability a
/// storage-tiled variant (today: `WorkgroupPerOutput` reusing F32's tiling,
/// see [`candidates`]'s `Op::MatMul` arm) needs regardless of which variant
/// carries it. `F32` needs nothing (the universal floor). `I8`/`Q4` key on
/// `int8_dot`: `Q4` is W4A8 (see `model::int4`'s module doc - activations
/// stay on the existing int8 dynamic-quant path, only weights narrow
/// further), so it rides the exact same capability as `I8`.
fn dtype_storage_requirement(dt: Dtype) -> Requirement {
    match dt {
        Dtype::F32 => Requirement::default(),
        Dtype::BF16 => Requirement { bf16_storage: true, ..Requirement::default() },
        Dtype::F16 => Requirement { f16_storage: true, ..Requirement::default() },
        Dtype::I8 | Dtype::Q4 => Requirement { int8_dot: true, ..Requirement::default() },
    }
}

impl KernelVariant {
    /// What this variant needs from the device to correctly execute over
    /// `dt` - the single source [`candidates`]'s uniform filter consults.
    ///
    /// `Reference` never requires anything (the always-correct portable
    /// baseline, by construction). `SplitReduction`'s BLANKET requirement is
    /// also empty - its barrier-or-not correctness varies by OP, which a
    /// (variant, dtype) pair alone cannot express: `Op::ArgMaxRow`'s split
    /// kernels are genuinely barrier-free (no `caps.workgroup_reductions`
    /// check anywhere in that arm), while `Op::GradNorm`'s are not (that
    /// arm keeps its own explicit `caps.workgroup_reductions` guard rather
    /// than being force-fit into this table). `WorkgroupPerOutput` always
    /// needs `workgroup_reductions` PLUS whatever it takes to merely hold
    /// `dt`'s bytes (today's storage-tiled variants reuse F32's tiling).
    /// `PackedInt8` always needs `int8_dot` - it is, physically, the
    /// packed-int8 kernel regardless of the shape's nominal dtype tag.
    pub fn requires(self, dt: Dtype) -> Requirement {
        match self {
            KernelVariant::Reference => Requirement::default(),
            KernelVariant::SplitReduction => Requirement::default(),
            KernelVariant::WorkgroupPerOutput => {
                Requirement { workgroup_reductions: true, ..dtype_storage_requirement(dt) }
            }
            KernelVariant::PackedInt8 => Requirement { int8_dot: true, ..Requirement::default() },
            KernelVariant::RegisterTiled => {
                Requirement { workgroup_reductions: true, ..dtype_storage_requirement(dt) }
            }
        }
    }
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
/// Swept with `brain perf run` (qwen-synth 8x512x8, `decode_heavy`) rather
/// than assumed: the packed GEMV still leads at `m = 4`, and by `m = 16` the
/// register-tiled GEMM has taken the lead. A per-device autotuner (S5) owns
/// refining this boundary.
pub const I8_GEMV_MAX_ROWS: u32 = 8;

/// The fp32/storage-tier register-tiled GEMM ([`KernelVariant::RegisterTiled`])
/// needs at least this many rows to be worth it - below it the 128×128 tile is
/// mostly idle and the naive one-thread-per-output kernel wins outright.
///
/// Migrated from `model::block::pick_gemm`'s doc comment (B2), which measured
/// it directly rather than assuming the tile's own 128-row dimension is the
/// threshold: requiring a full tile was far more expensive than the sweep
/// supports. Swept at `k = 2048`, `n = 2560` by `crates/gpu-core/tests/
/// bench_matmul.rs`, which is what to re-run on another card. The shape of
/// the result is what matters: the naive kernel's time grows with `m` while
/// the tile's is nearly flat, so the naive kernel wins at `m = 1..4`, they
/// are level at `m = 8`, and the tile's lead then widens without bound.
///
/// So the crossover is `m = 8`, not 128.
pub const GEMM_TILE_MIN_ROWS: u32 = 8;

/// Companion to [`GEMM_TILE_MIN_ROWS`]: below this output width the tile's
/// columns are mostly idle too, so the naive kernel wins regardless of `m`.
/// Also migrated from `model::block::pick_gemm`'s old `m < 8 || n < 128` rule.
pub const GEMM_TILE_MIN_COLS: u32 = 128;

/// Minimum output channels for the GEMM-lowered 1D convolution
/// ([`Op::Conv1d`]). `matmul_reg3` computes a 128-wide column tile, so a conv
/// with too few output channels pays for a full tile and loses.
///
/// # 16, swept - NOT the 2D lowering's 32, and the difference is the baseline
///
/// `vae::blocks`'s `GEMM_CONV_MIN_COUT` measured 32 for the same GEMM, but its
/// "direct" side is `conv_bias_reg`, an `@opt 5` register-tiled conv that
/// reaches a large fraction of the card's compute roof. The 1D direct side is
/// `conv1d`, one thread per output element with a serial reduction, which the
/// same sweep puts at a low single-digit percent of that roof. A much weaker
/// baseline crosses over much earlier, so copying the 2D number would have
/// left the whole 16..32 band unclaimed - a selection rule is as much a
/// bottleneck as a kernel, so each pair earns its own measured crossover
/// rather than inheriting another pair's.
///
/// Swept by `crates/audio/tests/bench_conv1d_lowering.rs` (`--ignored`), best
/// of 5, warm-up excluded, box otherwise idle; that test is what to re-run on
/// another card. `lowered` is the WHOLE lowering including its im2col and
/// epilogue passes. The sub-threshold half needs `BRAIN_CONV1D_GEMM=force`:
/// without it the selector answers "direct" for both columns and the ratio
/// reads a meaningless dead heat, which is how a threshold gets "confirmed"
/// by a measurement that never tested it.
///
/// The sweep walks `Cout` at N = 2, Cin = Cout, L = 44096 for both forms
/// (`k=7 pad=3`, im2col + `matmul_reg3`; and `k=1`, `matmul_dx_reg` with no
/// im2col). Direct wins at the narrow end, the lowering takes over between 12
/// and 16 in BOTH forms - so one number serves both - and its lead then grows
/// monotonically with width.
///
/// The lowered `conv1d` was also **bit-identical** to the direct kernel at
/// every shape in that sweep (`max|delta|` exactly 0.0, against a f32 host
/// reference that differs from both by ~1e-6). That is not luck and not a
/// guarantee: the `matmul_reg*` family accumulates strictly in increasing `k`,
/// one FMA at a time, and the col operand is laid out `ci*K + kw` - the same
/// order `conv1d.wgsl`'s nested loops sum in. Treat it as a measured property
/// of this driver, not a contract; the transposed lowering below genuinely
/// does reassociate.
pub const GEMM_CONV1D_MIN_COUT: u32 = 16;

/// Minimum output channels for the GEMM-lowered TRANSPOSED 1D convolution
/// ([`Op::ConvTranspose1d`]) - a SEPARATE threshold from
/// [`GEMM_CONV1D_MIN_COUT`], because it is a different GEMM against a
/// different baseline and the measurement says they part company.
///
/// `convtr1d` is worse than `conv1d` at the same width: every thread walks all
/// `K` taps and the `(lo + pad - kw·d) % stride != 0` test discards all but
/// `K/stride` of them, so at an upsampling stride most of its work is thrown
/// away (a fraction of one percent of the card's compute roof, measured by
/// the same sweep). The lowering does not pay that - its GEMM does `L` rows
/// rather than `Lo` - so at stride=4, K=8, L=11024, Cin=2·Cout it wins at
/// every width the sweep tried, including the narrowest, and its lead grows
/// with width from there.
///
/// So this is 4 rather than 16 - and 4 rather than 1 because 4 is the
/// narrowest width actually measured, not because anything is known to break
/// below it.
pub const GEMM_CONVTR1D_MIN_COUT: u32 = 4;

/// `BRAIN_NO_COOP_LN=1` pins LayerNorm to the per-element kernels — the A/B
/// switch the end-to-end speedup was measured with, and the fallback if a
/// driver ever mishandles the cooperative variant. Read once (the policy must
/// stay a pure function of its inputs for a given process).
fn no_coop_layernorm() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_NO_COOP_LN").map(|v| v != "0").unwrap_or(false))
}

/// `BRAIN_CONV1D_GEMM` pins [`Op::Conv1d`]/[`Op::ConvTranspose1d`] to one
/// side: `0` to the direct kernels, `force` to the GEMM lowering even below
/// [`GEMM_CONV1D_MIN_COUT`].
///
/// This is the A/B switch that threshold was measured with - without it a
/// sweep cannot see below the threshold at all, because the selector answers
/// "direct" on both sides and the ratio reads a meaningless dead heat. It is
/// also
/// the fallback if a driver ever mishandles the register-tiled GEMMs. Read
/// once (the policy must stay a pure function of its inputs for a given
/// process).
fn conv1d_gemm_override() -> Option<bool> {
    static V: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("BRAIN_CONV1D_GEMM").ok()?.as_str() {
        "0" | "off" => Some(false),
        "1" | "force" => Some(true),
        _ => None,
    })
}

/// `BRAIN_NO_COOP_GRADNORM=1` pins the optimiser's grad-norm to the
/// single-threaded `gradnorm_sq` — the A/B switch the speedup and the
/// trajectory-equivalence run were measured with, and the fallback if a driver
/// ever mishandles the cooperative reduction. Read once (see above).
fn no_coop_gradnorm() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_NO_COOP_GRADNORM").map(|v| v != "0").unwrap_or(false))
}

/// Every variant that can EXECUTE for `(op, shape)` on this device, with the
/// static best guess FIRST. Never empty: `Reference` is always executable.
///
/// This is both the default policy (its head) and the autotuner's probe list
/// (its tail): a measuring selector times exactly these — never a variant the
/// device cannot run, which is what keeps tuning a refinement rather than a
/// correctness risk.
///
/// Two layers: the match below enumerates what variants COULD apply for an
/// (op, dtype) shape-regime - ordering preference (measured regime
/// boundaries) only, no capability checks - and the uniform filter at the
/// end (`v.requires(shape.dtype).satisfied_by(caps)`) is the SINGLE
/// correctness gate, replacing what used to be scattered inline
/// `if caps.numeric.int8_dot` conditions per match arm. A shape-regime whose
/// every variant gets filtered out (e.g. int8 dtype on a device without
/// `int8_dot`) falls back to `Reference`, preserving the "never empty"
/// invariant. The one exception is `Op::GradNorm`'s own
/// `caps.workgroup_reductions` guard, kept inline rather than folded into
/// [`KernelVariant::requires`] - see that type's doc for why a (variant,
/// dtype)-only table cannot express it.
pub fn candidates(op: Op, shape: OpShape, caps: &DeviceCaps) -> Vec<KernelVariant> {
    use KernelVariant::*;
    let raw: Vec<KernelVariant> = match op {
        Op::MatMul => match shape.dtype {
            // F32 and the storage-tier dtypes (BF16/F16) share the SAME
            // regime split today: they use the SAME tiling as F32, just a
            // different load once a real bf16/f16 decode path lands. The
            // GEMV requires m <= 32 (its accumulator bound) and, where
            // registered, always wins the decode regime - the HEAD of both
            // branches below is therefore unchanged from before B2.
            //
            // `RegisterTiled` fills two gaps `WorkgroupPerOutput`/`Reference`
            // alone could not express (B2): within the decode regime, a
            // caller with no GEMV kernel at all (`model::block::pick_gemm`'s
            // callers - see `GEMM_TILE_MIN_ROWS`'s doc) still needs a
            // tiled-vs-naive answer once `m` clears the tile-fill threshold;
            // above the decode regime, EVERY caller needs it - that used to
            // fall through to `Reference` unconditionally, which is exactly
            // the "every prefill chunk above 32 rows takes the naive kernel"
            // hole `qwen3::serve::Engine::gemm_tier`'s old doc comment
            // flagged. Appending it after `WorkgroupPerOutput` (rather than
            // before) keeps every existing `DefaultSelector`/`AutoTuner`
            // consumer's head choice identical - only a caller that walks
            // past `WorkgroupPerOutput` (because it has no GEMV kernel) ever
            // reaches it.
            Dtype::F32 | Dtype::BF16 | Dtype::F16 => {
                if shape.m <= DECODE_REGIME_MAX_ROWS {
                    if shape.m >= GEMM_TILE_MIN_ROWS && shape.n >= GEMM_TILE_MIN_COLS {
                        vec![WorkgroupPerOutput, RegisterTiled, Reference]
                    } else {
                        vec![WorkgroupPerOutput, Reference]
                    }
                } else if shape.n >= GEMM_TILE_MIN_COLS {
                    vec![RegisterTiled, Reference]
                } else {
                    vec![Reference]
                }
            }
            // Packed/quantized weight tiers. Q4 is W4A8 (`model::int4`'s
            // module doc: activations stay on the existing int8
            // dynamic-quant path, only weights narrow further), so it
            // mirrors I8's shape exactly - same regime split, same
            // `int8_dot` requirement via `KernelVariant::requires`. Within
            // this regime, the 128x128 tile is mostly idle at decode row
            // counts, but the packed GEMV's workgroup-memory accumulation
            // grows per-row - the measured P40 crossover is m≈8, and
            // refining it per device is exactly what the autotuner probes
            // this tail for.
            Dtype::I8 | Dtype::Q4 => {
                if shape.m > DECODE_REGIME_MAX_ROWS || !caps.workgroup_reductions {
                    vec![PackedInt8]
                } else if shape.m <= I8_GEMV_MAX_ROWS {
                    vec![WorkgroupPerOutput, PackedInt8]
                } else {
                    vec![PackedInt8, WorkgroupPerOutput]
                }
            }
        },
        // RmsNorm's crossover is NOT a row count. The per-element kernel gives
        // thread t row t, so a warp's loads are `n` floats apart and each
        // 32-byte sector fetched serves one useful float; the cooperative
        // kernel walks a row with 64 threads and is coalesced by construction.
        // That penalty does not go away as rows grow: swept at a fixed total
        // element count, the cooperative variant wins at EVERY width, by the
        // widest margin at the narrow-row end where the per-element kernel's
        // reads are most scattered and by the least where a row is long
        // enough to amortise them (`crates/gpu-core/tests/bench_layernorm.rs`
        // runs the same comparison for the LayerNorm half). The old
        // `m <= DECODE_REGIME_MAX_ROWS` gate was therefore leaving that win
        // on the table for every prefill/encoder shape.
        // Reference stays in the list (and first without workgroup barriers,
        // via the uniform filter below) because the CPU JIT cannot run the
        // barrier.
        // LayerNorm is the same kernel family with the same bug: `layernorm`,
        // `ln_stats` and `layernorm_dx` all give thread t row t (and
        // `layernorm_dx` walks its row FOUR times that way). The `*_rows`
        // variants walk a row with 64 threads. Measured on a P40 — see
        // `layernorm_rows.wgsl` and `brain-gpu-core`'s `bench_layernorm`.
        Op::RmsNorm => vec![WorkgroupPerOutput, Reference],
        Op::LayerNorm => {
            if no_coop_layernorm() {
                vec![Reference]
            } else {
                vec![WorkgroupPerOutput, Reference]
            }
        }
        // The optimiser's grad-norm. `gradnorm_sq` is not merely uncoalesced —
        // it runs the WHOLE tensor on ONE invocation, so its cost is
        // `numel` dependent scalar loads however big the device is. There is
        // therefore NO tensor size at which the reference wins: even a
        // 768-element bias costs the same single dispatch either way, and the
        // cooperative variant does it with 64 lanes instead of 1. Measured on
        // a P40 the split reduction wins at every size in the GPT/Qwen
        // distribution (see `bench_gradnorm`), so `n` must never gate this —
        // exactly the mistake `Op::RmsNorm`'s old `m <= 32` gate made.
        // `workgroup_reductions` is a correctness gate here, not a
        // preference - but, unlike `Op::ArgMaxRow`'s split kernels, this
        // op's `SplitReduction` genuinely is barrier-bound, so this stays an
        // inline guard rather than moving into `KernelVariant::requires`
        // (see that type's doc comment for why).
        Op::GradNorm => {
            if caps.workgroup_reductions && !no_coop_gradnorm() {
                vec![SplitReduction, Reference]
            } else {
                vec![Reference]
            }
        }
        // The int8 activation quant's per-row max. `max_abs_row` is
        // `Op::RmsNorm`'s bug verbatim — thread `t` owns row `t` and walks the
        // whole row — so it takes RmsNorm's rule, and for the same measured
        // reason: the loss is per-access efficiency (each 32-byte sector serves
        // one useful float) plus a serial chain of `k` dependent loads, and
        // neither is fixed by having more rows. `max_abs_rows` is BIT-identical
        // (max is exact under reassociation), so there is no accuracy side to
        // trade either. `workgroup_reductions` is the correctness gate: the CPU
        // JIT mis-executes the barrier.
        Op::MaxAbsRow => vec![WorkgroupPerOutput, Reference],
        // The 1D convolutions. `conv1d`/`convtr1d` are one-thread-per-output
        // kernels with a serial `Cin*K` reduction, i.e. the classic "wrong
        // kernel, not a slow one": profiled in the MiniMax-Music-3 vocoder,
        // `conv1d` ran at a low single-digit percent of the card's compute
        // roof and `convtr1d` at a fraction of one percent, and between them
        // they accounted for essentially all of that stage's time.
        // `RegisterTiled` here means the whole
        // lowering (`im2col1d_at` + `matmul_reg3` + `nlc_bias_nchw` for
        // `conv1d`; `matmul_dw_reg_splitk` + `col2im1d_bias` for `convtr1d`),
        // so it inherits `workgroup_reductions` from
        // [`KernelVariant::requires`] - which is exactly the correctness gate
        // the register-tiled GEMMs need, and what keeps the CPU JIT on the
        // direct kernels.
        //
        // Only `n` (Cout) gates it, for the same reason
        // [`GEMM_TILE_MIN_COLS`] gates a plain GEMM: the GEMM's column tile is
        // 128 wide and a conv with too few output channels pays for a full
        // tile. `m` does NOT gate it - a 1D conv's position count is `Lo`,
        // which is in the thousands even for a short signal, so there is no
        // decode-shaped regime here to protect.
        //
        // The two ops get their OWN thresholds
        // ([`GEMM_CONV1D_MIN_COUT`] / [`GEMM_CONVTR1D_MIN_COUT`]) because the
        // sweep says they differ: the transposed direct kernel is so much
        // worse that the lowering wins at every width measured, while the
        // plain one loses below 16. Sharing one number would have picked the
        // wrong side of that for one of them.
        Op::Conv1d | Op::ConvTranspose1d => {
            let min_cout = if op == Op::Conv1d { GEMM_CONV1D_MIN_COUT } else { GEMM_CONVTR1D_MIN_COUT };
            match conv1d_gemm_override() {
                Some(false) => vec![Reference],
                Some(true) => vec![RegisterTiled, Reference],
                None if shape.n >= min_cout => vec![RegisterTiled, Reference],
                None => vec![Reference],
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
    };
    let filtered: Vec<KernelVariant> =
        raw.into_iter().filter(|v| v.requires(shape.dtype).satisfied_by(caps)).collect();
    if filtered.is_empty() {
        vec![Reference]
    } else {
        filtered
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
            KernelVariant::RegisterTiled => "register_tiled",
        }
    }
    /// Inverse of [`KernelVariant::as_str`]. Deliberately NOT `from_str`:
    /// that name shadows `std::str::FromStr::from_str` at the call site, and
    /// this returns `Option` rather than the trait's `Result`.
    pub fn parse_str(s: &str) -> Option<KernelVariant> {
        Some(match s {
            "reference" => KernelVariant::Reference,
            "workgroup_per_output" => KernelVariant::WorkgroupPerOutput,
            "split_reduction" => KernelVariant::SplitReduction,
            "packed_int8" => KernelVariant::PackedInt8,
            "register_tiled" => KernelVariant::RegisterTiled,
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
            if let Some(v) = KernelVariant::parse_str(&stored).filter(|v| cands.contains(v)) {
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
        c.numeric = NumericSupport { int8_dot: true, ..NumericSupport::BASELINE };
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

    /// The 1D convolutions: lowered above the `Cout` threshold on a GPU,
    /// direct below it, and direct on ANY device without workgroup
    /// reductions - the register-tiled GEMMs the lowering is built from are
    /// mis-executed by the CPU JIT's split-at-barrier model, so that is a
    /// correctness gate rather than a preference.
    ///
    /// `m` (the position count) must NOT gate it: a 1D conv's `Lo` is in the
    /// thousands even for a short signal, and a row threshold copied from the
    /// GEMM's decode regime would exclude nothing while looking careful.
    #[test]
    fn the_1d_convolutions_are_gated_on_cout_and_on_workgroup_reductions() {
        let s = DefaultSelector;
        for op in [Op::Conv1d, Op::ConvTranspose1d] {
            let min_cout = if op == Op::Conv1d { GEMM_CONV1D_MIN_COUT } else { GEMM_CONVTR1D_MIN_COUT };
            let wide = shape(4096, min_cout, 512, Dtype::F32);
            let narrow = shape(4096, min_cout - 1, 512, Dtype::F32);
            assert_eq!(s.select(op, wide, &gpu_caps()), KernelVariant::RegisterTiled, "{op:?}");
            assert_eq!(s.select(op, narrow, &gpu_caps()), KernelVariant::Reference, "{op:?}");
            assert_eq!(s.select(op, wide, &cpu_caps()), KernelVariant::Reference, "{op:?}");
            // Few positions, wide output: still lowered. `m` is not a gate.
            assert_eq!(s.select(op, shape(8, 256, 512, Dtype::F32), &gpu_caps()), KernelVariant::RegisterTiled, "{op:?}");
        }
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

    /// Training-sized M takes the register-tiled GEMM once `n` clears the
    /// tile-fill threshold (B2) - the naive kernel used to win this shape by
    /// default, which was the exact "every prefill chunk above 32 rows takes
    /// the naive kernel" hole `RegisterTiled` exists to close. A narrow `n`
    /// (below [`GEMM_TILE_MIN_COLS`]) still keeps the naive kernel - the tile
    /// would be mostly idle there too.
    #[test]
    fn large_m_and_wide_n_gets_the_register_tiled_gemm() {
        let s = DefaultSelector;
        let wide = shape(4096, 512, 512, Dtype::F32);
        assert_eq!(s.select(Op::MatMul, wide, &gpu_caps()), KernelVariant::RegisterTiled);
        let narrow = shape(4096, 64, 512, Dtype::F32);
        assert_eq!(s.select(Op::MatMul, narrow, &gpu_caps()), KernelVariant::Reference);
        // No workgroup barriers (the CPU JIT) -> the reference, always, even
        // at a shape that would otherwise fill a tile.
        assert_eq!(s.select(Op::MatMul, wide, &cpu_caps()), KernelVariant::Reference);
    }

    /// …but RmsNorm does NOT: the per-element kernel's loss is uncoalesced
    /// global reads, which large M does not fix. The cooperative kernel
    /// measured faster at every width, so it is the choice at prefill/encoder
    /// row counts too; the regression this test exists to prevent is silently
    /// reverting to `m <= 32`.
    #[test]
    fn rmsnorm_prefers_cooperative_at_every_row_count() {
        let s = DefaultSelector;
        for m in [1u32, 32, 1536, 36864] {
            let sh = shape(m, 128, 0, Dtype::F32);
            assert_eq!(s.select(Op::RmsNorm, sh, &gpu_caps()), KernelVariant::WorkgroupPerOutput, "m={m}");
            // No workgroup barriers (the CPU JIT) -> the reference, always.
            assert_eq!(s.select(Op::RmsNorm, sh, &cpu_caps()), KernelVariant::Reference, "m={m}");
        }
    }

    /// LayerNorm inherits RmsNorm's rule, for the same measured reason: the
    /// loss is per-access efficiency, not thread count, so no row count makes
    /// the per-element kernel competitive. The regression this prevents is
    /// someone re-introducing a `m <= 32` gate for LayerNorm.
    #[test]
    fn layernorm_prefers_cooperative_at_every_row_count() {
        let s = DefaultSelector;
        for m in [1u32, 32, 512, 2048, 36864] {
            for n in [128u32, 768, 3072] {
                let sh = shape(m, n, 0, Dtype::F32);
                assert_eq!(
                    s.select(Op::LayerNorm, sh, &gpu_caps()),
                    KernelVariant::WorkgroupPerOutput,
                    "m={m} n={n}"
                );
                // No workgroup barriers (the CPU JIT) -> the reference, always.
                assert_eq!(
                    s.select(Op::LayerNorm, sh, &cpu_caps()),
                    KernelVariant::Reference,
                    "m={m} n={n}"
                );
            }
        }
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

    /// The grad-norm has NO size gate. `gradnorm_sq` walks the whole tensor on
    /// one invocation, so there is no tensor small enough for it to win; the
    /// only thing that can deselect the cooperative variant is a device that
    /// cannot execute a workgroup barrier. This test exists so a `numel <= X`
    /// gate cannot creep back the way `Op::RmsNorm`'s `m <= 32` one did.
    #[test]
    fn gradnorm_is_cooperative_at_every_size() {
        let s = DefaultSelector;
        for numel in [1u32, 64, 768, 8192, 1_769_472, 38_597_376] {
            assert_eq!(
                s.select(Op::GradNorm, shape(1, numel, 0, Dtype::F32), &gpu_caps()),
                KernelVariant::SplitReduction,
                "grad-norm at numel {numel} must not fall back to the serial walk"
            );
            // The CPU JIT cannot execute the barrier — a correctness gate.
            assert_eq!(
                s.select(Op::GradNorm, shape(1, numel, 0, Dtype::F32), &cpu_caps()),
                KernelVariant::Reference
            );
        }
    }

    /// The int8 activation quant's per-row max has NO shape gate either: it is
    /// `Op::RmsNorm`'s one-thread-per-row bug on the int8 path, so neither the
    /// row count nor the row width can make the per-element kernel win. This
    /// test exists so an `m <= 32` gate cannot creep in the way `Op::RmsNorm`'s
    /// did.
    #[test]
    fn max_abs_row_is_cooperative_at_every_shape() {
        let s = DefaultSelector;
        for m in [1u32, 8, 32, 512, 4096, 36864] {
            for n in [128u32, 1024, 3072, 12288] {
                let sh = shape(m, n, 0, Dtype::F32);
                assert_eq!(
                    s.select(Op::MaxAbsRow, sh, &gpu_caps()),
                    KernelVariant::WorkgroupPerOutput,
                    "m={m} k={n}"
                );
                // The CPU JIT cannot execute the barrier — a correctness gate.
                assert_eq!(
                    s.select(Op::MaxAbsRow, sh, &cpu_caps()),
                    KernelVariant::Reference,
                    "m={m} k={n}"
                );
            }
        }
    }

    /// The candidate list is never empty and its head IS the default policy —
    /// one list, so the static choice and the tuner's probe set cannot drift.
    #[test]
    fn candidates_head_is_the_default_policy() {
        let s = DefaultSelector;
        for caps in [gpu_caps(), cpu_caps()] {
            for op in
                [Op::MatMul, Op::RmsNorm, Op::LayerNorm, Op::ArgMaxRow, Op::GradNorm, Op::MaxAbsRow]
            {
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

    /// The exhaustive proof: for every reachable `NumericSupport` combination
    /// (all 6 independent bools it carries, `f32` is always true) crossed
    /// with `workgroup_reductions`, every `Op`, every [`Dtype`] and a
    /// representative row-count sample, no variant [`candidates`] returns
    /// ever fails its own [`KernelVariant::requires`] check against the caps
    /// it was computed from. This is the whole point of the (variant, dtype)
    /// `Requirement` table: capability-gating becomes something a test can
    /// PROVE across the whole input space, not something a reviewer has to
    /// trust from reading scattered `if caps.numeric.X` conditions.
    #[test]
    fn no_candidate_ever_requires_an_unsupported_capability() {
        let ops =
            [Op::MatMul, Op::RmsNorm, Op::LayerNorm, Op::ArgMaxRow, Op::GradNorm, Op::MaxAbsRow];
        let dtypes = [Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::I8, Dtype::Q4];
        // 6 independent bools -> 64 combinations (f32 is always true, not
        // varied) crossed with workgroup_reductions (2), covering baseline
        // (all false), int8_dot-only, full-everything, and every
        // single-flag-true combination along the way.
        for bits in 0u8..64 {
            let numeric = NumericSupport {
                f32: true,
                int8_dot: bits & 1 != 0,
                f16: bits & 2 != 0,
                bf16: bits & 4 != 0,
                f16_storage: bits & 8 != 0,
                bf16_storage: bits & 16 != 0,
                coop_matrix: bits & 32 != 0,
            };
            for workgroup_reductions in [false, true] {
                let mut caps = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
                caps.numeric = numeric;
                caps.workgroup_reductions = workgroup_reductions;
                for &op in &ops {
                    for &dtype in &dtypes {
                        for &m in &[1u32, 8, 9, 33, 4096] {
                            let shape = OpShape { m, n: 8192, k: 512, dtype };
                            for v in candidates(op, shape, &caps) {
                                assert!(
                                    v.requires(dtype).satisfied_by(&caps),
                                    "{op:?}/{dtype:?}/m={m} on {numeric:?} \
                                     (workgroup_reductions={workgroup_reductions}) -> {v:?} \
                                     but capability not satisfied"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// B2: `candidates()`'s default policy (its head) must agree with the two
    /// pre-B2 GEMM pickers this phase unifies - `model::block::pick_gemm`
    /// (training-shaped: `m < GEMM_TILE_MIN_ROWS || n < GEMM_TILE_MIN_COLS`
    /// picks the naive kernel, else the register-tiled one - no GEMV concept
    /// at all) and `model::block::gemm_variant` (inference-shaped: `m <=
    /// DECODE_REGIME_MAX_ROWS` picks the GEMV kernel when the model registered
    /// one, else the tiled kernel at every M > 32 regardless of N). This crate
    /// cannot depend on `brain-model`, so the two rules are reproduced inline -
    /// the measured (m, n) pairs are `block::pick_gemm`'s own doc table (P40,
    /// k=2048, n=2560): naive wins at m in {1,2,4}, the tile wins from m=8.
    /// Before `KernelVariant::RegisterTiled` exists this does not even
    /// compile - `candidates()` has no way to express the tiled choice, which
    /// is exactly the gap `qwen3::serve::Engine::gemm_tier`'s doc comment
    /// complains about.
    #[test]
    fn candidates_agrees_with_the_pre_b2_gemm_pickers() {
        let caps = gpu_caps();

        // `pick_gemm`'s callers never register a GEMV kernel (the function
        // has no such parameter), so its adapter skips `WorkgroupPerOutput`
        // and takes the first variant it CAN express (naive or tiled).
        for &(m, n) in &[
            (1u32, 2560u32),
            (2, 2560),
            (4, 2560),
            (7, 2560),
            (8, 2560),
            (9, 2560),
            (12, 2560),
            (32, 2560),
            (33, 2560),
            (77, 2560),
            (512, 2560),
            (8, 64),
            (33, 64),
            (512, 127),
            (512, 128),
        ] {
            let want = if m < GEMM_TILE_MIN_ROWS || n < GEMM_TILE_MIN_COLS {
                KernelVariant::Reference
            } else {
                KernelVariant::RegisterTiled
            };
            let sh = shape(m, n, 512, Dtype::F32);
            let got = candidates(Op::MatMul, sh, &caps)
                .into_iter()
                .find(|v| *v != KernelVariant::WorkgroupPerOutput)
                .expect("candidates() is never empty, and Reference is always in it");
            assert_eq!(got, want, "pick_gemm parity at m={m} n={n}");
        }

        // `gemm_variant`'s Fast tier: GEMV owns `m <= 32` when registered,
        // the tiled kernel owns every M above that regardless of N (no naive
        // kernel exists in that API shape at all).
        for &m in &[1u32, 8, 32, 33, 512] {
            let sh = shape(m, 3072, 512, Dtype::F32);
            let head = candidates(Op::MatMul, sh, &caps)[0];
            let want = if m <= DECODE_REGIME_MAX_ROWS {
                KernelVariant::WorkgroupPerOutput
            } else {
                KernelVariant::RegisterTiled
            };
            assert_eq!(head, want, "gemm_variant parity at m={m}");
        }
    }

    /// Persistence round-trips every variant - a variant added to the enum
    /// without a matching `as_str`/`parse_str` pair would silently lose tuned
    /// choices back to the static default on the next process start.
    #[test]
    fn every_kernel_variant_round_trips_through_persistence() {
        for v in [
            KernelVariant::Reference,
            KernelVariant::WorkgroupPerOutput,
            KernelVariant::SplitReduction,
            KernelVariant::PackedInt8,
            KernelVariant::RegisterTiled,
        ] {
            assert_eq!(KernelVariant::parse_str(v.as_str()), Some(v), "{v:?}");
        }
    }

    /// The persisted tune-cache key must not collide across dtype tiers -
    /// `AutoTuner::key` formats `{:?}` of the (now-unified) `DType`/`Dtype`
    /// into the key string, so this pins that BF16 and F16 (same byte width,
    /// no natural ordering between them per `DType`'s doc) still produce
    /// distinct keys, and so do every other pair.
    #[test]
    fn cache_key_distinguishes_every_dtype_tier() {
        let dtypes = [Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::I8, Dtype::Q4];
        let mut seen = std::collections::HashSet::new();
        for dtype in dtypes {
            let key = AutoTuner::key(Op::MatMul, shape(4, 512, 512, dtype));
            assert!(seen.insert(key.clone()), "duplicate cache key for {dtype:?}: {key}");
        }
        assert_eq!(seen.len(), dtypes.len());
    }
}
