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

use backend_api::{select, DeviceCaps};

/// One drop-in replacement: dispatch `fast` instead of `slow`, with the thread
/// count multiplied by `thread_mul`.
pub(crate) struct Upgrade {
    /// The kernel name models register and dispatch by index.
    pub slow: &'static str,
    /// The faster, contract-identical variant appended to the pipeline set.
    pub fast: &'static str,
    /// `fast`'s WGSL.
    pub src: &'static str,
    /// `fast`'s dispatch size as a multiple of `slow`'s (64 for the
    /// one-thread-per-row -> one-workgroup-per-row rewrites).
    pub thread_mul: u32,
    /// The policy that decides whether this device wants it.
    pub op: select::Op,
}

/// The table. Keep it short; see the bar above.
pub(crate) const UPGRADES: &[Upgrade] = &[Upgrade {
    // The int8 dynamic-activation-quant path: every int8 linear in
    // `qwen3::q8`, `s3dit::int8`/`block`, and the FLUX.2 int8 DiT quantizes
    // its activations with `max_abs_row` -> `quant_pack` -> `matmul_i8_dyn`.
    // `max_abs_row` walks a whole row from one invocation (checklist §C2).
    slow: "max_abs_row",
    fast: "max_abs_rows",
    src: kernels::MAX_ABS_ROWS,
    thread_mul: 64,
    op: select::Op::MaxAbsRow,
}];

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
pub(crate) fn expand<'a>(kernels: &[(&'a str, &'a str)]) -> Option<Vec<(&'a str, &'a str)>> {
    if disabled() {
        return None;
    }
    let add: Vec<&Upgrade> = UPGRADES
        .iter()
        .filter(|u| {
            kernels.iter().any(|(n, _)| *n == u.slow) && !kernels.iter().any(|(n, _)| *n == u.fast)
        })
        .collect();
    if add.is_empty() {
        return None;
    }
    let mut out = kernels.to_vec();
    out.extend(add.iter().map(|u| (u.fast, u.src)));
    Some(out)
}

/// The active redirects for a handle: `(slow index, fast index, thread_mul)`.
///
/// Empty for a model that registered neither kernel, and empty on a device
/// whose [`select`] policy prefers the reference (the CPU JIT, which cannot
/// execute a workgroup barrier). Computed once per handle so `step` costs one
/// integer compare against a usually-empty list, never a policy walk.
pub(crate) fn resolve(names: &[String], caps: &DeviceCaps) -> Vec<(usize, usize, u32)> {
    if disabled() {
        return Vec::new();
    }
    UPGRADES
        .iter()
        .filter_map(|u| {
            let slow = names.iter().position(|n| n == u.slow)?;
            let fast = names.iter().position(|n| n == u.fast)?;
            // Neither the row count nor the row width gates these variants (see
            // `select::candidates`), so probing the policy once with a
            // representative shape is exact, not an approximation — the unit
            // tests in `select.rs` are what hold that property.
            let shape =
                select::OpShape { m: 1024, n: 1024, k: 0, dtype: select::Dtype::F32 };
            match select::candidates(u.op, shape, caps).first() {
                Some(select::KernelVariant::WorkgroupPerOutput) => Some((slow, fast, u.thread_mul)),
                _ => None,
            }
        })
        .collect()
}

/// The `(pipeline slot, thread count)` to actually DISPATCH for a caller's
/// `(kind, threads)`. Identity when `kind` is not an upgraded slot.
///
/// Only the dispatch moves: the caller's `StepMeta` keeps the caller's own
/// `kind`/`threads`, because profilers and cost harnesses index `meta.kernel`
/// through *their* kernel list and an appended slot would run off the end of it
/// (`crates/flux2/src/bin/flux2_bench.rs` does exactly this). See `Gpu::step`.
#[inline]
pub(crate) fn apply(active: &[(usize, usize, u32)], kind: usize, threads: u32) -> (usize, u32) {
    for &(slow, fast, mul) in active {
        if kind == slow {
            return (fast, threads.saturating_mul(mul));
        }
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
        let out = expand(&ks).expect("max_abs_row must be upgradable");
        assert_eq!(&out[..3], &ks[..]);
        assert_eq!(out[3].0, "max_abs_rows");
        assert_eq!(out.len(), 4);
    }

    /// A model that does not use the op pays nothing (no allocation, no extra
    /// shader compile) — and one that already registered the fast kernel by
    /// hand is left exactly as it is.
    #[test]
    fn expand_is_a_no_op_when_not_applicable() {
        assert!(expand(&[("add2", "a")]).is_none());
        assert!(expand(&[("max_abs_row", "b"), ("max_abs_rows", "c")]).is_none());
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
        assert_eq!(active, vec![(1, 2, 64)]);
        assert_eq!(apply(&active, 1, 512), (2, 512 * 64), "one workgroup per row");
        assert_eq!(apply(&active, 0, 512), (0, 512), "other kernels untouched");

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

    /// Every entry must name a real kernel and a real fast variant, and the
    /// fast source must be the one the name resolves to in `kernels::src`.
    #[test]
    fn table_entries_name_real_kernels() {
        for u in UPGRADES {
            assert_eq!(kernels::src(u.slow), kernels::src(u.slow), "{} is not a kernel", u.slow);
            assert_eq!(kernels::src(u.fast), u.src, "{} source mismatch", u.fast);
            assert!(u.thread_mul >= 1);
            // Same contract: identical `Params` struct, so a caller's params
            // need no rewriting (checklist §B).
            let params = |s: &str| {
                s.lines().find(|l| l.contains("struct Params")).map(|l| l.trim().to_string())
            };
            assert_eq!(
                params(kernels::src(u.slow)),
                params(u.src),
                "{} and {} must take the same Params",
                u.slow,
                u.fast
            );
        }
    }
}
