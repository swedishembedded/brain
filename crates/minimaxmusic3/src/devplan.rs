// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which physical card each of the DiT's two CFG branches runs on.
//!
//! Swedish Embedded AB implements multi-accelerator placement and scheduling
//! for production inference pipelines. If your team needs expertise in
//! turning an idle card into wall-clock without giving up bit-exact
//! reproducibility, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # The idle card this exists to remove
//!
//! A generation is three sequential stages - the AR stage, the chunked
//! denoise, the vocoder - and the denoise stage is ~all of the wall clock
//! (roughly an hour for a four-minute track). Inside it,
//! [`crate::denoise::denoise_chunk`] evaluates the DiT **twice per Euler
//! step**: once against this chunk's condition and once against a zeroed
//! copy of it (`denoise.py`'s `zeros_like(condition)`). The two forwards
//! share `latents`, the timestep and `length`, and share no intermediate
//! value at all - the only thing that reads both is the host-side fold
//! `u + (c - u) * GUIDANCE_SCALE` after both have returned.
//!
//! Meanwhile `crate::generate::stage_devices` hands the denoise stage one
//! card and the vocoder stage the other, and the two stages never overlap,
//! so on a two-card box the second card is at 0% for the entire denoise
//! stage. Two independent forwards is exactly the shape two cards want.
//!
//! # Why this is bit-identical, not merely "close enough"
//!
//! The two forwards are separate computations over separate inputs, not two
//! halves of one reduction. Nothing is split across the cards and
//! recombined, so no sum is reassociated and no accumulator is partitioned;
//! each card computes exactly the same sequence of dispatches over exactly
//! the same bytes it would have computed on its own, and the fold that
//! consumes them runs on the host in the same order either way. Two
//! identical GP102 cards running identical WGSL over identical inputs must
//! therefore agree bit for bit, and
//! [`crate::denoise`]'s own gate (`the_concurrent_cfg_pair_is_bit_identical_
//! to_the_sequential_one`) checks that rather than assuming it - a
//! disagreement would be a real bug (a kernel reading uninitialised memory,
//! a nondeterministic reduction) and is worth failing on, not worth
//! widening a tolerance for.
//!
//! # What decides the default
//!
//! [`DevicePlan::Auto`] reads `gpu_core::devices::ambient_compute_set()`,
//! the same `--device`/`BRAIN_DEVICE` resolution every other placement
//! decision in this workspace goes through - never `gpus()` directly, which
//! would ignore a `--device gpu0` restriction and schedule onto a card the
//! operator excluded. With fewer than two schedulable cards, or on the CPU
//! backend, it resolves to [`Placement::single`] - byte-for-byte the
//! previous behaviour, no threads spawned.
//!
//! The base card is whatever is already selected on this thread
//! (`current_gpu()`), so a generation running under the residency
//! executor's `with_gpu`-scoped lane keeps its assigned card as `cond` and
//! borrows only the *other* one. The second card is the next schedulable
//! index after it, wrapping - so an operator who pinned the run to gpu1 on
//! a two-card box gets `cond = 1, uncond = 0`, not a silent relocation onto
//! gpu0.
//!
//! An explicit in-process `GenOpts::device` token is an instruction, not a
//! hint (`crate::generate::stage_devices` treats it the same way), so it
//! resolves [`Placement::single`] on that one device rather than quietly
//! spreading the stage over a card the caller did not name.
//!
//! # The one honest gap: borrowing a card residency assigned to someone else
//!
//! `Auto` borrows the second card without asking `crates/residency` whether
//! another model is resident on it. This is not a NEW claim on the box -
//! `crate::generate::ar_branch_devices` already puts the AR stage's two
//! Global LLM instances on both cards unconditionally, and the residency
//! adapter scopes a whole generation to one assigned card
//! (`crates/cli/src/resident_minimaxmusic3.rs`) that this plan then reads as
//! its `cond` - but it is the same gap, and it is worth stating rather than
//! taking the card quietly. There is no seam to ask through today -
//! a `residency::Instance` is handed the device it was placed on, not a
//! view of the whole placement - and inventing one for this is a different
//! milestone's work. What bounds the damage meanwhile is real rather than
//! hoped for: `memauth`'s process-wide `--limit-vram-total` ceiling is
//! enforced at `gpu_core::Gpu`'s allocation facade, so an over-subscribed
//! box that published a ceiling gets a refusal rather than a driver-level
//! out-of-memory abort. Note that ceiling is a TOTAL across all cards, not
//! a per-card one, so a concurrent pair charges it twice: an operator who
//! sized `--limit-vram-total` for one card should either raise it or set
//! `BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0`. A [`crate::dit::Resident`] at
//! `DitConfig::real()` dims is ~9.7 GB, so the pair is ~19.4 GB across two
//! cards, not on one.

use gpu_core::devices;

/// Where one chunk's two CFG branches run.
///
/// `Single` is the pre-existing behaviour and is what every CPU-backend,
/// single-card and explicitly-pinned run resolves to. `Split` is the
/// two-card shape. `Auto` picks between them from the machine's real,
/// `--device`-restricted schedulable set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DevicePlan {
    /// Both forwards on the ambient/stage device, one after the other - no
    /// second handle, no threads. Exactly what this pipeline did before
    /// device plans existed.
    Single,
    /// The conditional forward on `cond` and the zero-condition one
    /// concurrently on `uncond`.
    Split { cond: u32, uncond: u32 },
    /// `Split` across two schedulable cards when the machine has them,
    /// `Single` otherwise. The default.
    #[default]
    Auto,
}

/// A [`DevicePlan`] resolved against the machine this process can really
/// schedule on - what [`crate::denoise::CfgDevices`] opens handles from.
/// `None` means "the stage's own device", i.e. do not name a card at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub cond: Option<u32>,
    pub uncond: Option<u32>,
}

impl Placement {
    /// One device, named by the caller rather than by a plan: no second
    /// handle, no concurrency.
    pub fn single() -> Placement {
        Placement { cond: None, uncond: None }
    }

    /// True when the two CFG branches land on two DIFFERENT cards, which is
    /// the only case worth opening a second device and spawning a thread
    /// for.
    pub fn cfg_is_parallel(&self) -> bool {
        match (self.cond, self.uncond) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        }
    }
}

/// Opt out of concurrent CFG dispatch without changing any code path:
/// `BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0` forces [`DevicePlan::Auto`] to
/// resolve `Single`. Deliberately an opt-OUT: the concurrent path is gated
/// bit-identical, so the two-card shape is the right default on hardware
/// that has two cards, and the escape hatch exists for bisecting a
/// suspected driver problem or for an operator whose
/// `--limit-vram-total` was sized for one card, not for routine use.
///
/// Matches `ltxv::devplan`'s `BRAIN_LTXV_CFG_PARALLEL` convention, spelling
/// included.
pub fn cfg_parallel_enabled() -> bool {
    !matches!(std::env::var("BRAIN_MINIMAXMUSIC3_CFG_PARALLEL").as_deref(), Ok("0") | Ok("false") | Ok("off"))
}

impl DevicePlan {
    /// Resolve against this machine's schedulable set and this thread's
    /// current selection.
    ///
    /// `device` is `GenOpts::device` - an in-process caller's explicit
    /// per-stage token (`Some("cpu")` / `Some("gpu")` / `Some("gpu1")`), or
    /// `None` to inherit the ambient `--device`/`BRAIN_DEVICE` selection. A
    /// CPU-backend run has no cards to split across, and a caller who named
    /// one device asked for that device, so both resolve to
    /// [`Placement::single`] rather than scoping onto a card the run was
    /// never pointed at.
    pub fn resolve(&self, device: Option<&str>) -> Placement {
        if device.is_some() {
            return Placement::single();
        }
        match *self {
            DevicePlan::Single => Placement::single(),
            DevicePlan::Split { cond, uncond } => Placement { cond: Some(cond), uncond: Some(uncond) },
            DevicePlan::Auto => Self::auto(),
        }
    }

    /// The two-card shape when there are two schedulable cards, else
    /// `Single`. See this module's doc for why the base card is the CURRENT
    /// selection rather than a hardcoded 0.
    fn auto() -> Placement {
        if !cfg_parallel_enabled() {
            return Placement::single();
        }
        let set = devices::ambient_compute_set();
        if set.backend == devices::Backend::Cpu || set.gpus.len() < 2 {
            return Placement::single();
        }
        let base = devices::current_gpu().unwrap_or(set.gpus[0]);
        let at = set.gpus.iter().position(|&g| g == base).unwrap_or(0);
        Placement { cond: Some(set.gpus[at]), uncond: Some(set.gpus[(at + 1) % set.gpus.len()]) }
    }
}

/// Run `f` on GPU `index`, or straight through when `index` is `None`.
///
/// The one place this crate turns a [`Placement`] field into a real device
/// scope, so "None means the stage's own device" is decided once rather
/// than at every call site. Errors from `with_gpu` (an out-of-range index)
/// are propagated, never clamped - a plan naming a card this machine does
/// not have is a bug in the plan, and silently running on a different card
/// would hide it.
pub fn on_gpu<R>(index: Option<u32>, f: impl FnOnce() -> R) -> Result<R, String> {
    match index {
        Some(i) => devices::with_gpu(i, f),
        None => Ok(f()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The explicit shapes resolve to themselves, and a run the caller
    /// pointed at ONE device refuses to be split no matter what it was
    /// asked for - an explicit token is an instruction, and a CPU-backend
    /// run has no cards to split across at all.
    #[test]
    fn explicit_plans_resolve_to_themselves_and_a_pinned_device_never_splits() {
        assert_eq!(DevicePlan::Single.resolve(None), Placement::single());
        let split = DevicePlan::Split { cond: 0, uncond: 1 };
        assert_eq!(split.resolve(None), Placement { cond: Some(0), uncond: Some(1) });
        assert!(split.resolve(None).cfg_is_parallel());
        assert_eq!(split.resolve(Some("cpu")), Placement::single());
        assert_eq!(split.resolve(Some("gpu1")), Placement::single());
        assert_eq!(DevicePlan::Auto.resolve(Some("cpu")), Placement::single());
    }

    /// `cfg_is_parallel` is the predicate that decides whether a second
    /// `Resident` is uploaded and a thread spawned at all, so "same card
    /// twice" must read as NOT parallel - a plan that put both branches on
    /// one card and still split would hold two ~9.7 GB weight copies on one
    /// 24 GB board.
    #[test]
    fn two_branches_on_one_card_is_not_parallel() {
        let same = DevicePlan::Split { cond: 1, uncond: 1 }.resolve(None);
        assert!(!same.cfg_is_parallel());
        assert!(!Placement::single().cfg_is_parallel());
    }

    /// `Auto` must never name a card outside the schedulable set, and must
    /// degrade to `Single` rather than to a half-formed split when there is
    /// only one card (or none). Machine-shape-independent: it asserts the
    /// INVARIANT against whatever this box really has, so it is meaningful
    /// on a two-card box, a one-card box and a GPU-less CI runner alike.
    #[test]
    fn auto_stays_inside_the_schedulable_set() {
        let p = DevicePlan::Auto.resolve(None);
        let set = devices::ambient_compute_set();
        if set.backend == devices::Backend::Cpu || set.gpus.len() < 2 || !cfg_parallel_enabled() {
            assert_eq!(p, Placement::single(), "fewer than two schedulable cards must resolve Single");
            return;
        }
        for card in [p.cond, p.uncond].into_iter().flatten() {
            assert!(set.gpus.contains(&card), "auto named gpu{card}, outside the schedulable set {:?}", set.gpus);
        }
        assert!(p.cfg_is_parallel(), "two schedulable cards must produce a genuinely split plan");
    }

    /// `on_gpu(None, ..)` is a plain call - the "no card named" case must
    /// not depend on a GPU existing at all.
    #[test]
    fn on_gpu_none_is_a_straight_call() {
        assert_eq!(on_gpu(None, || 41 + 1), Ok(42));
    }
}
