// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Transparent kernel upgrades — *a faster kernel a model inherits without
//! knowing it exists.*
//!
//! The most expensive mistake in this repo is not a slow kernel, it is a fast
//! kernel nobody knew about: `gn_stats` was fixed for DIAMOND in 2025 and
//! `crates/vae`, written afterwards against the same kernel *name*, silently
//! inherited the slow one — 2262 ms of a 6.5 s decode, 159x when finally
//! selected. The fix belongs in **selection**, never in a copy each model
//! opts into by hand.
//!
//! Two selection seams already existed:
//!
//! * `backend_api::select` — the call site asks for a variant and maps it to
//!   its own pipeline index (`qwen3::serve`, `model::block`).
//! * `Gpu::kernel_index` by name — the call site probes for an optional
//!   kernel (`model::vit`, `optim::Optim::coop_gradnorm`).
//!
//! Both still require **editing every dispatch site**, which is exactly the
//! step the next model forgets. This seam removes that step for the case where
//! the fast variant is a drop-in: same params, same bindings, same result, only
//! a different *thread count*. `Gpu` appends the fast kernel to the pipeline
//! set at construction and, when [`backend_api::select`] says so for this
//! device, rewrites `step`/`step_sliced` to dispatch it with the scaled thread
//! count. A model that registers `max_abs_row` and dispatches `m` threads gets
//! the workgroup-per-row kernel with `m * 64` threads and no source change.
//!
//! ## The bar for adding an entry here
//!
//! This is invisible machinery, so it is deliberately hard to qualify for:
//!
//! 1. **Identical contract** — same `Params` struct, same binding order, same
//!    output layout. The only difference may be the thread count.
//! 2. **Identical results**, not "close". `max_abs_rows` reduces with `max`,
//!    which is exact and associative, so splitting a row across 64 lanes is
//!    bit-identical. A *sum* reduction would reassociate and change the last
//!    bits — for those, keep the explicit seams above so the trajectory /
//!    parity gate is visible at the call site.
//! 3. **Wins at every shape**, proven by a microbenchmark, so there is no
//!    regime the caller would have wanted to opt out of.
//! 4. **Capability-gated through `backend_api::select`** — the policy lives
//!    there with everything else, never in a backend-name test here.
//!
//! `BRAIN_NO_KERNEL_UPGRADE=1` disables the whole table (the A/B switch every
//! measurement below was taken with).
//!
//! ## Shape-specialised rows
//!
//! A row may carry a `kernels::template` KNOB instead of a single fast kernel
//! (`buckets`/`knob`/`shape_param`). The one such row today is the decode-regime
//! GEMV, whose faster body holds its accumulators in registers and therefore
//! needs the row count as a COMPILE-TIME constant: one specialised pipeline is
//! appended per bucket, and the dispatch picks the smallest bucket that covers
//! the caller's own `params[shape_param]`. The four bars above are unchanged -
//! in particular "wins at every shape" is what forces a bucket LADDER rather
//! than one worst-case specialisation, because a variant compiled for 32 rows
//! is SLOWER than the workgroup-accumulator kernel it replaces when only 1 row
//! is asked for. The ladder is a measurement, never a guess (checklist §F.6).
//!
//! A shape-specialised row cannot resolve its bucket from
//! [`crate::Gpu::step_buf`] alone - its shape lives in a caller-owned uniform
//! buffer this seam cannot read, so that path keeps the kernel the caller
//! registered: correct, just not upgraded. A caller that already holds the
//! values it wrote into that buffer can hand them back through
//! [`crate::Gpu::step_buf_shaped`] instead, which reaches [`apply`] exactly
//! as [`crate::Gpu::step`]'s own `params` slice does
//! (`tests/gemv_reg_upgrade_step_buf.rs`).

use backend_api::{select, DeviceCaps};

/// One drop-in replacement: dispatch `fast` instead of `slow`, with the thread
/// count multiplied by `thread_mul`.
pub(crate) struct Upgrade {
    /// The kernel name models register and dispatch by index.
    pub slow: &'static str,
    /// The faster, contract-identical variant appended to the pipeline set.
    /// For a shape-specialised row (`knob` set) this is the variant-name STEM
    /// `kernels::template::variant_name` builds each bucket's name from.
    pub fast: &'static str,
    /// `fast`'s WGSL.
    pub src: &'static str,
    /// `fast`'s dispatch size as a multiple of `slow`'s (64 for the
    /// one-thread-per-row -> one-workgroup-per-row rewrites).
    pub thread_mul: u32,
    /// The policy that decides whether this device wants it.
    pub op: select::Op,
    /// The shape this row probes [`select`] with. A row whose kernels are not
    /// shape-gated can use any shape (the policy is constant over it); a
    /// decode-regime row must probe IN that regime or it reads the policy for
    /// a shape it never runs at.
    pub probe: select::OpShape,
    /// `Some((template const, `params` index))` for a shape-specialised row:
    /// the `kernels::template` constant to rewrite, and which of the caller's
    /// own uniform params carries the value the buckets are indexed by.
    pub knob: Option<(&'static str, usize)>,
    /// Ascending bucket values for `knob`. `&[]` for a plain row.
    pub buckets: &'static [u32],
    /// Additional template constants FIXED for this row (applied to every
    /// bucket alongside `knob`), for a `fast` kernel that is ALREADY
    /// specialised along a second axis before the bucket ladder ever runs -
    /// the affine K-quant GEMV pair, whose `slow` kernel is only ever
    /// registered as `matmul_kq_gemv#CODE_BITS={4,8}` (`kernels::template::
    /// interned` names it can never be the bare, unspecialised stem name -
    /// see `model::ops::kernel_list`, the only place that ever registers it).
    /// `&[]` for every plain row - the `variants()` params list is simply
    /// `extra` followed by `(knob, bucket)`.
    pub extra: &'static [(&'static str, u32)],
    /// `true` when `fast` cannot be compiled by the CPU JIT, so it must not be
    /// appended to a kernel set destined for `backend-cpu`: `wgsl_cpu::Jit`
    /// only *skips* a kernel it cannot express, and every skipped kernel costs
    /// a parse plus a warning line for something that backend can never
    /// dispatch anyway (`backend-cpu` reports `workgroup_reductions: false`,
    /// so [`resolve`] would not activate the row there in any case).
    pub gpu_only: bool,
}

/// A plain row's probe shape: these kernels are gated by capability only, never
/// by shape (see `select::candidates`), so any shape reads the same policy.
const ANY_SHAPE: select::OpShape =
    select::OpShape { m: 1024, n: 1024, k: 0, dtype: select::Dtype::F32 };

/// The GEMV row's probe shape: `m = 1` is IN the decode regime
/// (`select::DECODE_REGIME_MAX_ROWS`), which is the only regime where
/// `candidates` heads with `WorkgroupPerOutput` for a MatMul. Probing at
/// `ANY_SHAPE` would read the tiled-GEMM policy and never activate.
const DECODE_SHAPE: select::OpShape =
    select::OpShape { m: 1, n: 1024, k: 0, dtype: select::Dtype::F32 };

/// [`DECODE_SHAPE`]'s int8 twin. A separate constant rather than a reused one
/// because the two dtypes reach `WorkgroupPerOutput` through DIFFERENT arms of
/// `select::candidates` - the int8 arm additionally requires `int8_dot`, so
/// probing with the f32 shape would activate the int8 row on a device that
/// cannot run its kernel at all.
const DECODE_SHAPE_I8: select::OpShape =
    select::OpShape { m: 1, n: 1024, k: 0, dtype: select::Dtype::I8 };

/// [`DECODE_SHAPE`]'s q4 twin, for the same reason [`DECODE_SHAPE_I8`] is
/// separate from it: `select::candidates` reaches `WorkgroupPerOutput`
/// through the `Dtype::I8 | Dtype::Q4` arm, which the plain f32 shape would
/// never probe.
const DECODE_SHAPE_Q4: select::OpShape =
    select::OpShape { m: 1, n: 1024, k: 0, dtype: select::Dtype::Q4 };

/// [`DECODE_SHAPE_I8`]'s affine K-quant (Q4_K/Q5_K) twin. A separate constant
/// per this file's own convention (one probe per dtype family) even though
/// `backend_api::select::candidates`'s `Op::MatMul` arm folds `Q4K`/`Q8K` into
/// the IDENTICAL branch `I8` takes today (same `int8_dot` requirement, same
/// regime split) - reusing `DECODE_SHAPE_I8` here would silently start
/// reading the wrong dtype's future policy the day that arm ever splits, and
/// costs nothing today since the two constants resolve to the same
/// `candidates()` answer either way. `Dtype::Q4K` stands in for both `Q4K`
/// and `Q8K`: the two GEMV rows below share this one probe because
/// `select::candidates` does not distinguish them (`M12`'s own
/// `affine_kquant_dtypes_select_exactly_like_i8` test is what pins that).
const DECODE_SHAPE_KQ: select::OpShape =
    select::OpShape { m: 1, n: 1024, k: 0, dtype: select::Dtype::Q4K };

/// The `MREG` ladder for `matmul_gemv_reg`. Powers of two, so "the smallest
/// bucket covering `m`" carries at most twice the rows actually needed, and the
/// ladder is complete by construction for `matmul_gemv`'s own `m <= 32`
/// contract. Measured at the depth decoder's shapes: every `m` in 1..=32 beats
/// `matmul_gemv` through its own bucket, where a single `MREG = 32`
/// specialisation would LOSE outright at `m = 1`.
const GEMV_MREG_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 32];

/// The table. Keep it short; see the bar above.
pub(crate) const UPGRADES: &[Upgrade] = &[
    Upgrade {
        // The int8 dynamic-activation-quant path: every int8 linear in
        // `qwen3::q8`, `s3dit::int8`/`block`, and the FLUX.2 int8 DiT quantizes
        // its activations with `max_abs_row` -> `quant_pack` -> `matmul_i8_dyn`.
        // `max_abs_row` walks a whole row from one invocation (checklist §C2).
        slow: "max_abs_row",
        fast: "max_abs_rows",
        src: kernels::MAX_ABS_ROWS,
        thread_mul: 64,
        op: select::Op::MaxAbsRow,
        probe: ANY_SHAPE,
        knob: None,
        buckets: &[],
        extra: &[],
        gpu_only: false,
    },
    Upgrade {
        // The decode-regime GEMM every LM, DiT and depth decoder in this tree
        // dispatches for `m <= 32` (`model::block::gemm_variant`).
        // `matmul_gemv` accumulates in workgroup memory sized for the WORST
        // case, which costs it both occupancy (8 KB/workgroup at every `m`)
        // and a shared-memory dependency chain per `(k, m)`; `matmul_gemv_reg`
        // is the same arithmetic in registers. Same `Params`, same bindings,
        // same `n * 64` thread count, byte-identical results
        // (`tests/gemv_reg_upgrade.rs` asserts the BITS, not a tolerance).
        slow: "matmul_gemv",
        fast: "matmul_gemv_reg",
        src: kernels::MATMUL_GEMV_REG,
        thread_mul: 1,
        op: select::Op::MatMul,
        probe: DECODE_SHAPE,
        knob: Some(("MREG", 0)), // Params { m, k, n } -> m
        buckets: GEMV_MREG_BUCKETS,
        extra: &[],
        gpu_only: true,
    },
    Upgrade {
        // The int8 twin of the row above, and it is here because the fix was
        // made for one dtype and its sibling never got it: `matmul_i8_gemv`
        // still accumulates in workgroup memory sized for `m = 32`, paying the
        // same 8 KB-per-workgroup occupancy cap and the same per-`(k, m)`
        // shared-memory dependency chain the fp32 kernel was rescued from.
        // Measured on a Tesla P40 at Qwen3-VL-4B's decode shape, it streamed
        // its weights at about half the card's DRAM roof where the fp32
        // register kernel reached essentially all of it - so int8's four-fold
        // smaller weights were returning a little over two-fold in time.
        //
        // Same `Params`, same bindings, same `n * 64` thread count. Results
        // are bit-identical BY CONSTRUCTION rather than by care: the
        // accumulator is `i32`, and integer addition is exact and associative,
        // so no regrouping of the same terms can differ.
        slow: "matmul_i8_gemv",
        fast: "matmul_i8_gemv_reg",
        src: kernels::MATMUL_I8_GEMV_REG,
        thread_mul: 1,
        op: select::Op::MatMul,
        probe: DECODE_SHAPE_I8,
        knob: Some(("MREG", 0)), // Params { m, kg, n } -> m
        buckets: GEMV_MREG_BUCKETS,
        extra: &[],
        gpu_only: true,
    },
    Upgrade {
        // The q4 twin of the row above (`matmul_i8_gemv` -> `_reg`), added
        // once the shape that mattered was actually measured. An earlier
        // pass benchmarked ONLY the un-templated, worst-case MREG=32 build
        // at one generic k=n=2048 shape and found it 15-29% SLOWER at
        // m=1..16 - the same "compiled for 32 rows, asked for 1" pathology
        // `matmul_gemv_reg`'s own bucket ladder exists to avoid, and exactly
        // what running the WORST-CASE build at every m would produce. Re-run
        // with the per-m bucket the production dispatch actually picks
        // (`crates/model/tests/matmul_q4_speed_bench.rs::
        // gemv_vs_gemv_reg_at_qwen35_decode_shapes`), at qwen35's own real
        // decode shapes (not a generic stand-in), `_reg` won at every one:
        // 1.55-1.88x, m=1. Bit-identity is `tests/q4_gemv_reg_upgrade.rs`
        // (this crate) - a fresh gate, mirroring `i8_gemv_reg_upgrade.rs`,
        // because the earlier decision to leave this row out meant no such
        // gate existed to inherit.
        slow: "matmul_q4_gemv",
        fast: "matmul_q4_gemv_reg",
        src: kernels::MATMUL_Q4_GEMV_REG,
        thread_mul: 1,
        op: select::Op::MatMul,
        probe: DECODE_SHAPE_Q4,
        knob: Some(("MREG", 0)), // Params { m, k, n } -> m
        buckets: GEMV_MREG_BUCKETS,
        extra: &[],
        gpu_only: true,
    },
    Upgrade {
        // M13: the affine K-quant (Q4_K) GEMV's register-accumulator sibling.
        // `matmul_kq_gemv` carries exactly the same two costs
        // `matmul_i8_gemv_reg`'s own row above was added to fix -
        // worst-case-sized workgroup-memory accumulators and a per-`(k, m)`
        // shared-memory read-modify-write - and `matmul_kq_gemv_reg.wgsl`'s
        // own header derives the fix the identical way. `slow` is the
        // `CODE_BITS=4` (Q4_K) specialisation `model::ops::kernel_list`
        // registers; `extra` fixes the SAME axis on `fast` before the `MREG`
        // bucket ladder runs, so the two vary together the way `CODE_BITS`
        // and `MREG` are independent knobs on one source file.
        slow: "matmul_kq_gemv#CODE_BITS=4",
        fast: "matmul_kq_gemv_reg",
        src: kernels::MATMUL_KQ_GEMV_REG,
        thread_mul: 1,
        op: select::Op::MatMul,
        probe: DECODE_SHAPE_KQ,
        knob: Some(("MREG", 0)), // Params { m, k, n } -> m
        buckets: GEMV_MREG_BUCKETS,
        extra: &[("CODE_BITS", 4)],
        gpu_only: true,
    },
    Upgrade {
        // Q5_K's twin of the row above - identical reasoning, `CODE_BITS=8`.
        slow: "matmul_kq_gemv#CODE_BITS=8",
        fast: "matmul_kq_gemv_reg",
        src: kernels::MATMUL_KQ_GEMV_REG,
        thread_mul: 1,
        op: select::Op::MatMul,
        probe: DECODE_SHAPE_KQ,
        knob: Some(("MREG", 0)), // Params { m, k, n } -> m
        buckets: GEMV_MREG_BUCKETS,
        extra: &[("CODE_BITS", 8)],
        gpu_only: true,
    },
];

/// `BRAIN_NO_KERNEL_UPGRADE=1` pins every model to the kernel it registered —
/// the A/B switch, and the fallback if a driver ever mishandles a cooperative
/// variant. Read once (the policy must stay fixed for a given process).
fn disabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_NO_KERNEL_UPGRADE").map(|v| v != "0").unwrap_or(false))
}

/// `kernels` plus every applicable fast variant, **appended** so existing
/// pipeline indices are untouched. `None` when there is nothing to add, which
/// is the common case and keeps the allocation off that path.
///
/// The variants are compiled unconditionally (device caps are not known until
/// the backend exists); whether they are *dispatched* is decided per handle by
/// [`resolve`]. One extra shader compile is cheap next to a device init.
pub(crate) fn expand<'a>(
    kernels: &[(&'a str, &'a str)],
    cpu_jit: bool,
) -> Option<Vec<(&'a str, &'a str)>> {
    if disabled() {
        return None;
    }
    let mut add: Vec<(&'static str, &'static str)> = Vec::new();
    for u in UPGRADES {
        if (u.gpu_only && cpu_jit) || !kernels.iter().any(|(n, _)| *n == u.slow) {
            continue;
        }
        for (name, src) in u.variants() {
            if !kernels.iter().any(|(n, _)| *n == name) {
                add.push((name, src));
            }
        }
    }
    if add.is_empty() {
        return None;
    }
    let mut out = kernels.to_vec();
    out.extend(add);
    Some(out)
}

impl Upgrade {
    /// Every `(name, source)` this row contributes: one pair for a plain row,
    /// one per bucket for a shape-specialised one. The specialised sources come
    /// from `kernels::template`, so the WGSL stays a SINGLE file (one source,
    /// tunable constants) and each variant carries the `stem#KNOB=value` name
    /// that machinery already gives them - which is also what `BRAIN_PROFILE=1`
    /// prints, so a profile says which bucket ran.
    fn variants(&self) -> Vec<(&'static str, &'static str)> {
        let Some((knob, _)) = self.knob else {
            return vec![(self.fast, self.src)];
        };
        self.buckets
            .iter()
            .map(|&b| {
                let mut params: Vec<(&str, u32)> = self.extra.to_vec();
                params.push((knob, b));
                kernels::template::interned(self.fast, self.src, &params).unwrap_or_else(|e| {
                    // A malformed row is a programming error in THIS table, not
                    // a runtime condition: failing here beats silently running
                    // the kernel the caller was trying to leave behind.
                    panic!("upgrade: {} cannot be specialised {params:?}: {e}", self.fast)
                })
            })
            .collect()
    }
}

/// One resolved redirect for a handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Active {
    /// The pipeline slot the model registered and dispatches by index.
    pub slow: usize,
    /// `slow`'s thread count is multiplied by this.
    pub thread_mul: u32,
    /// `Some(i)` for a shape-specialised row: which of the caller's uniform
    /// params selects the bucket. `None` means [`Active::slots`] has exactly
    /// one entry and it always applies.
    pub shape_param: Option<usize>,
    /// `(largest value this bucket covers, pipeline slot)`, ascending.
    pub slots: Vec<(u32, usize)>,
}

/// The active redirects for a handle.
///
/// Empty for a model that registered neither kernel, and empty on a device
/// whose [`select`] policy prefers the reference (the CPU JIT, which cannot
/// execute a workgroup barrier). Computed once per handle so `step` costs one
/// integer compare against a usually-empty list, never a policy walk.
pub(crate) fn resolve(names: &[String], caps: &DeviceCaps) -> Vec<Active> {
    if disabled() {
        return Vec::new();
    }
    UPGRADES
        .iter()
        .filter_map(|u| {
            let slow = names.iter().position(|n| n == u.slow)?;
            // Probing the policy once with this row's own representative shape
            // is exact, not an approximation: `select::candidates` gates these
            // variants on capability, and on the ROW COUNT only through the
            // decode-regime threshold that `Upgrade::probe` is chosen to sit
            // inside. The unit tests in `select.rs` are what hold that.
            match select::candidates(u.op, u.probe, caps).first() {
                Some(select::KernelVariant::WorkgroupPerOutput) => {}
                _ => return None,
            }
            let variants = u.variants();
            let mut slots = Vec::with_capacity(variants.len());
            for (i, (name, _)) in variants.iter().enumerate() {
                // A bucket the handle does not carry (a model that registered
                // the fast kernel by hand, or a handle built through a path
                // that bypasses `expand`) is skipped, not faked.
                let idx = names.iter().position(|n| n == name)?;
                let cap = if u.knob.is_some() { u.buckets[i] } else { u32::MAX };
                slots.push((cap, idx));
            }
            Some(Active { slow, thread_mul: u.thread_mul, shape_param: u.knob.map(|(_, p)| p), slots })
        })
        .collect()
}

/// The `(pipeline slot, thread count)` to actually DISPATCH for a caller's
/// `(kind, threads)`. Identity when `kind` is not an upgraded slot, when a
/// shape-specialised row has no `params` to read (`Gpu::step_buf`), or when the
/// caller's shape is past the last bucket.
///
/// Only the dispatch moves: the caller's `StepMeta` keeps the caller's own
/// `kind`/`threads`, because profilers and cost harnesses index `meta.kernel`
/// through *their* kernel list and an appended slot would run off the end of it
/// (`crates/flux2/src/bin/flux2_bench.rs` does exactly this). See `Gpu::step`.
#[inline]
pub(crate) fn apply(
    active: &[Active],
    kind: usize,
    params: Option<&[u32]>,
    threads: u32,
) -> (usize, u32) {
    for a in active {
        if a.slow != kind {
            continue;
        }
        let slot = match a.shape_param {
            None => a.slots[0].1,
            Some(p) => {
                let Some(v) = params.and_then(|q| q.get(p).copied()) else { break };
                match a.slots.iter().find(|(cap, _)| v <= *cap) {
                    Some(&(_, idx)) => idx,
                    None => break,
                }
            }
        };
        return (slot, threads.saturating_mul(a.thread_mul));
    }
    (kind, threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// A model that registers the slow kernel gets the fast one appended — at
    /// the END, so every index it already hard-codes still resolves.
    #[test]
    fn expand_appends_without_moving_indices() {
        let ks = [("add2", "a"), ("max_abs_row", "b"), ("quant_pack", "c")];
        let out = expand(&ks, false).expect("max_abs_row must be upgradable");
        assert_eq!(&out[..3], &ks[..]);
        assert_eq!(out[3].0, "max_abs_rows");
        assert_eq!(out.len(), 4);
    }

    /// A model that does not use the op pays nothing (no allocation, no extra
    /// shader compile) — and one that already registered the fast kernel by
    /// hand is left exactly as it is.
    #[test]
    fn expand_is_a_no_op_when_not_applicable() {
        assert!(expand(&[("add2", "a")], false).is_none());
        assert!(expand(&[("max_abs_row", "b"), ("max_abs_rows", "c")], false).is_none());
    }

    /// The redirect is capability-gated through `backend_api::select`: a device
    /// with workgroup barriers takes the cooperative kernel and scales the
    /// dispatch; one without (the CPU JIT) keeps the kernel the model chose.
    #[test]
    fn resolve_is_capability_gated_and_scales_threads() {
        use backend_api::DeviceClass;
        let n = names(&["add2", "max_abs_row", "max_abs_rows"]);

        let gpu = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
        assert!(gpu.workgroup_reductions);
        let active = resolve(&n, &gpu);
        assert_eq!(
            active,
            vec![Active { slow: 1, thread_mul: 64, shape_param: None, slots: vec![(u32::MAX, 2)] }]
        );
        assert_eq!(apply(&active, 1, None, 512), (2, 512 * 64), "one workgroup per row");
        assert_eq!(apply(&active, 0, None, 512), (0, 512), "other kernels untouched");

        let mut cpu = DeviceCaps::portable_baseline(DeviceClass::Cpu);
        cpu.workgroup_reductions = false;
        assert!(resolve(&n, &cpu).is_empty());
    }

    /// A model that never registered the fast kernel (an old handle built
    /// before `expand`, or one built through a path that bypasses it) is not
    /// redirected into a pipeline slot it does not have.
    #[test]
    fn resolve_needs_both_kernels_present() {
        let gpu = DeviceCaps::portable_baseline(backend_api::DeviceClass::DiscreteGpu);
        assert!(resolve(&names(&["max_abs_row"]), &gpu).is_empty());
        assert!(resolve(&names(&["max_abs_rows"]), &gpu).is_empty());
    }

    // --- the shape-specialised GEMV row -----------------------------------

    fn gemv_names() -> Vec<String> {
        expand(&[("add2", "a"), ("matmul_gemv", "b")], false)
            .unwrap()
            .iter()
            .map(|(n, _)| (*n).to_string())
            .collect()
    }

    /// One appended pipeline per `MREG` bucket, each a `kernels::template`
    /// variant of the ONE source file - never a hand-written per-`m` kernel.
    #[test]
    fn a_shape_specialised_row_appends_one_variant_per_bucket() {
        let out = expand(&[("matmul_gemv", "b")], false).unwrap();
        assert_eq!(out.len(), 1 + GEMV_MREG_BUCKETS.len());
        assert_eq!(out[1].0, "matmul_gemv_reg#MREG=1");
        assert_eq!(out[out.len() - 1].0, "matmul_gemv_reg#MREG=32");
        // The knob really is rewritten in the variant's SOURCE, not just its
        // name - a name-only variant would silently run 32 accumulators.
        assert!(out[2].1.contains("const MREG: u32 = 2u;"), "{}", out[2].1);
    }

    /// The dispatch picks the SMALLEST bucket covering the caller's own row
    /// count, and falls back to the caller's kernel when there is no shape to
    /// read (`Gpu::step_buf`) or the shape is past the last bucket.
    #[test]
    fn a_shape_specialised_row_picks_the_bucket_from_the_callers_params() {
        let gpu = DeviceCaps::portable_baseline(backend_api::DeviceClass::DiscreteGpu);
        let n = gemv_names();
        let active = resolve(&n, &gpu);
        let gemv = active.iter().find(|a| a.slow == 1).expect("the GEMV row must be active");
        assert_eq!(gemv.shape_param, Some(0), "Params {{ m, k, n }} -> m");
        assert_eq!(gemv.thread_mul, 1, "same n*64 dispatch as matmul_gemv");

        let name = |m: u32| n[apply(&active, 1, Some(&[m, 4096, 4096]), 4096 * 64).0].as_str();
        assert_eq!(name(1), "matmul_gemv_reg#MREG=1");
        assert_eq!(name(2), "matmul_gemv_reg#MREG=2");
        assert_eq!(name(3), "matmul_gemv_reg#MREG=4", "smallest bucket that covers it");
        assert_eq!(name(17), "matmul_gemv_reg#MREG=32");
        assert_eq!(name(32), "matmul_gemv_reg#MREG=32");
        // Thread count is untouched (mul 1) - the whole point of the drop-in.
        assert_eq!(apply(&active, 1, Some(&[2, 4096, 4096]), 4096 * 64).1, 4096 * 64);
        // Past the last bucket: `matmul_gemv` itself requires m <= 32, so this
        // is already a caller error - keep its kernel rather than truncate.
        assert_eq!(apply(&active, 1, Some(&[33, 4096, 4096]), 64).0, 1);
        // No params to read: `step_buf`. Identity, never a guessed bucket.
        assert_eq!(apply(&active, 1, None, 64), (1, 64));
    }

    // --- M13: the affine K-quant (Q4_K/Q5_K) GEMV register-accumulator rows

    fn gpu_caps_int8() -> DeviceCaps {
        let mut c = DeviceCaps::portable_baseline(backend_api::DeviceClass::DiscreteGpu);
        c.numeric = backend_api::NumericSupport { int8_dot: true, ..backend_api::NumericSupport::BASELINE };
        c
    }

    fn kq_names() -> Vec<String> {
        expand(&[("add2", "a"), ("matmul_kq_gemv#CODE_BITS=4", "b"), ("matmul_kq_gemv#CODE_BITS=8", "c")], false)
            .unwrap()
            .iter()
            .map(|(n, _)| (*n).to_string())
            .collect()
    }

    /// Two independent bucket ladders, one per `CODE_BITS` - the row for
    /// `CODE_BITS=4` never appends a `CODE_BITS=8` variant or vice versa, so
    /// a Q4_K model's `Gpu` never carries dead Q5_K pipelines and vice versa.
    #[test]
    fn kq_gemv_reg_appends_one_bucket_ladder_per_code_bits() {
        let out = expand(&[("matmul_kq_gemv#CODE_BITS=4", "b")], false).unwrap();
        assert_eq!(out.len(), 1 + GEMV_MREG_BUCKETS.len());
        assert_eq!(out[1].0, "matmul_kq_gemv_reg#CODE_BITS=4,MREG=1");
        assert_eq!(out[out.len() - 1].0, "matmul_kq_gemv_reg#CODE_BITS=4,MREG=32");
        // Both knobs are rewritten in the SOURCE, not just the name.
        assert!(out[2].1.contains("const CODE_BITS: u32 = 4u;"), "{}", out[2].1);
        assert!(out[2].1.contains("const MREG: u32 = 2u;"), "{}", out[2].1);

        let out8 = expand(&[("matmul_kq_gemv#CODE_BITS=8", "b")], false).unwrap();
        assert_eq!(out8[1].0, "matmul_kq_gemv_reg#CODE_BITS=8,MREG=1");
        assert!(out8[2].1.contains("const CODE_BITS: u32 = 8u;"), "{}", out8[2].1);
    }

    /// Both `CODE_BITS` rows resolve independently and each picks its own
    /// smallest-covering `MREG` bucket - the same dispatch-time behaviour
    /// `matmul_gemv`'s own row already proves, duplicated across the two
    /// affine specialisations a real Q4_K+Q5_K model can carry at once.
    #[test]
    fn kq_gemv_reg_resolves_independently_per_code_bits() {
        let caps = gpu_caps_int8();
        let n = kq_names();
        let active = resolve(&n, &caps);
        let kq4 = active.iter().find(|a| n[a.slow] == "matmul_kq_gemv#CODE_BITS=4").expect("CODE_BITS=4 row active");
        let kq8 = active.iter().find(|a| n[a.slow] == "matmul_kq_gemv#CODE_BITS=8").expect("CODE_BITS=8 row active");
        assert_eq!(kq4.shape_param, Some(0), "Params {{ m, k, n }} -> m");
        assert_eq!(kq4.thread_mul, 1, "same n*64 dispatch as matmul_kq_gemv");

        let name4 = |m: u32| n[apply(&active, kq4.slow, Some(&[m, 4096, 4096]), 4096 * 64).0].as_str();
        assert_eq!(name4(1), "matmul_kq_gemv_reg#CODE_BITS=4,MREG=1");
        assert_eq!(name4(3), "matmul_kq_gemv_reg#CODE_BITS=4,MREG=4", "smallest bucket that covers it");
        assert_eq!(name4(32), "matmul_kq_gemv_reg#CODE_BITS=4,MREG=32");

        let name8 = |m: u32| n[apply(&active, kq8.slow, Some(&[m, 4096, 4096]), 4096 * 64).0].as_str();
        assert_eq!(name8(1), "matmul_kq_gemv_reg#CODE_BITS=8,MREG=1");
        assert_eq!(name8(17), "matmul_kq_gemv_reg#CODE_BITS=8,MREG=32");

        // Past the last bucket: `matmul_kq_gemv` itself requires m <= 32, so
        // this is already a caller error - keep its kernel rather than
        // truncate, exactly like the plain GEMV row.
        assert_eq!(apply(&active, kq4.slow, Some(&[33, 4096, 4096]), 64).0, kq4.slow);
    }

    /// The CPU JIT never receives a `gpu_only` row's variants: it cannot
    /// compile them and could not dispatch them if it could
    /// (`workgroup_reductions: false`).
    #[test]
    fn a_gpu_only_row_is_not_appended_for_the_cpu_jit() {
        assert!(expand(&[("matmul_gemv", "b")], true).is_none());
        // ...and a non-`gpu_only` row in the same table still is.
        let out = expand(&[("matmul_gemv", "b"), ("max_abs_row", "c")], true).unwrap();
        assert_eq!(out.iter().filter(|(n, _)| n.starts_with("matmul_gemv_reg")).count(), 0);
        assert_eq!(out[out.len() - 1].0, "max_abs_rows");
    }

    /// Every entry must name a real kernel and a real fast variant, and the
    /// fast source must be the one the name resolves to in `kernels::src`.
    #[test]
    fn table_entries_name_real_kernels() {
        // `u.slow` may be a `kernels::template` specialisation
        // (`"matmul_kq_gemv#CODE_BITS=4"`, never registered under that exact
        // string in `kernels::ALL` - only the bare stem is), so resolve the
        // CONTRACT (Params/bindings never move under a `#K=V` specialisation
        // - only a `const` literal's value does) against the base name.
        let base_src = |n: &str| kernels::src(n.split('#').next().unwrap());
        for u in UPGRADES {
            assert_eq!(base_src(u.slow), base_src(u.slow), "{} is not a kernel", u.slow);
            assert_eq!(kernels::src(u.fast), u.src, "{} source mismatch", u.fast);
            assert!(u.thread_mul >= 1);
            assert_eq!(u.knob.is_some(), !u.buckets.is_empty(), "{}: knob and buckets pair up", u.fast);
            assert!(u.buckets.windows(2).all(|w| w[0] < w[1]), "{}: buckets must ascend", u.fast);
            // Same contract: identical `Params` struct, so a caller's params
            // need no rewriting (checklist §B).
            let params = |s: &str| {
                s.lines().find(|l| l.contains("struct Params")).map(|l| l.trim().to_string())
            };
            assert_eq!(
                params(base_src(u.slow)),
                params(u.src),
                "{} and {} must take the same Params",
                u.slow,
                u.fast
            );
            // ...and identical BINDINGS, in the same order. A silently permuted
            // binding list is the `silu_mul` defect class (checklist §B):
            // wrong answers, not a crash.
            let bindings = |s: &str| {
                s.lines()
                    .filter(|l| l.trim_start().starts_with("@group("))
                    .map(|l| l.trim().to_string())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bindings(base_src(u.slow)),
                bindings(u.src),
                "{} and {} must take the same bindings",
                u.slow,
                u.fast
            );
        }
    }

    /// A `gpu_only` row's fast kernel must DECLARE `@cpu no` - that header is
    /// what `wgsl-cpu`'s skip list and `kernelmeta.py`'s `@cpu` derivation are
    /// cross-checked against, so a row claiming GPU-only over a kernel the JIT
    /// would happily run is a table bug.
    #[test]
    fn gpu_only_rows_declare_cpu_no() {
        for u in UPGRADES.iter().filter(|u| u.gpu_only) {
            assert!(
                u.src.lines().any(|l| l.trim_start().starts_with("// @cpu") && l.contains("no")),
                "{} is marked gpu_only but does not declare `@cpu no`",
                u.fast
            );
        }
    }
}
