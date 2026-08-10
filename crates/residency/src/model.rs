// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The residency unit: a [`ResidentModel`] the manager can build on a device
//! (promote to Hot), run, and drop (demote), plus the built [`Instance`] that
//! executes actions.
//!
//! A `ResidentModel` is one model *family* (e.g. z-image); a specific build — a
//! size/precision/adapter combination — is identified by an [`InstanceKey`] and,
//! once built, is an `Instance`. The trait stays model-agnostic (it only speaks
//! `capability` types + [`MemCost`]/[`Device`]), so the residency crate never
//! depends on any model crate; the concrete adapters live next to the models.

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::{Device, InstanceKey, MemCost, Tier};

/// A model family that can be made resident on demand.
pub trait ResidentModel: Send + Sync {
    /// Discovery manifest (same content as `capability::Provider::manifest`).
    fn manifest(&self) -> Manifest;

    /// The config fingerprint that fixes a build for `(action, inv)` — two jobs with
    /// the same key share one hot instance (and may batch). E.g. z-image keys on
    /// `WxH:precision:adapter`; a stateless model keys on a constant.
    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey;

    /// Estimated Hot footprint of the instance named by `key`, *before* building —
    /// what the manager budgets against (reuse `model::plan` where a shape is known).
    fn estimate(&self, key: &InstanceKey) -> MemCost;

    /// Build the instance on `device` (blocking: weight load + upload). The manager
    /// has already reserved room for `estimate(key)` on `device`.
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String>;

    /// Estimated footprint at `tier` — what the manager would budget against
    /// *before* calling [`Instance::demote`]/[`promote`](Instance::promote).
    /// Defaulted to [`Self::estimate`] regardless of `tier`: a model that
    /// never overrides `demote` never demotes, so this is never consulted
    /// for anything but `Tier::Hot` unless a model opts into both together.
    fn estimate_at(&self, key: &InstanceKey, _tier: Tier) -> MemCost {
        self.estimate(key)
    }
}

/// A built, Hot model instance. Dropping it frees the device memory (RAII), so the
/// manager demotes simply by dropping. Actions run on the manager's worker thread,
/// never on an async/bus thread.
pub trait Instance: Send {
    /// Run one action; `progress` streams updates for streaming actions.
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult;

    /// Run a batch of same-key invocations. Default = sequential loop; a model with
    /// real batch support overrides this (P4). Results align with `invs`. `progress`
    /// is called with the batch index so per-sequence token streams stay separate.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }

    /// Model-specific observability metrics (e.g. a paged-KV serving engine's
    /// prefix-cache hit rate) — polled by the dispatcher between runs and
    /// exposed via `Executor::stats().metrics`. Default: nothing extra to
    /// report, which is correct for every model that doesn't override it.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        Vec::new()
    }

    /// Move this instance down to `tier` (`Warm`: release device buffers,
    /// keep host/mapped bytes; `Cold`: additionally `madvise(DONTNEED)` the
    /// mapping) — an alternative to the manager dropping and later fully
    /// rebuilding the instance from the checkpoint. Defaulted to
    /// `Err("demote: unsupported")`: **every existing model's behaviour is
    /// unchanged** — the manager's eviction still just drops the instance —
    /// unless a model deliberately opts in by overriding this (and
    /// [`ResidentModel::estimate_at`] with real per-tier numbers).
    /// `tier == Hot` is never a valid `demote` target (that is
    /// [`Self::promote`]'s job); implementations may debug-assert this but
    /// need not check it in release.
    fn demote(&mut self, _tier: Tier) -> Result<(), String> {
        Err("demote: unsupported".to_string())
    }

    /// The inverse of [`Self::demote`]: rebuild whatever `demote` released
    /// on `device`, returning to `Tier::Hot`. Only ever called on an
    /// instance a prior `demote` call returned `Ok(())` for — defaulted to
    /// `Err` for the same reason `demote` is.
    fn promote(&mut self, _device: Device) -> Result<(), String> {
        Err("promote: unsupported".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{ActionSpec, Blob, Media, Outcome};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// A fake resident model: `activate` bumps a shared "live GPU instances" counter,
    /// dropping the instance decrements it — so a test can assert promote/evict frees
    /// device memory without a GPU.
    struct FakeModel {
        vram: u64,
        live: Arc<AtomicU32>,
    }
    struct FakeInstance {
        live: Arc<AtomicU32>,
        runs: u32,
    }
    impl Drop for FakeInstance {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl ResidentModel for FakeModel {
        fn manifest(&self) -> Manifest {
            Manifest::new("fake", "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("fake", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInstance { live: self.live.clone(), runs: 0 }))
        }
    }
    impl Instance for FakeInstance {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            self.runs += 1;
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![self.runs as u8])))
        }
    }

    #[test]
    fn activate_run_batch_and_drop_frees() {
        let live = Arc::new(AtomicU32::new(0));
        let m = FakeModel { vram: 1 << 30, live: live.clone() };
        let key = m.instance_key("run", &Invocation::new());
        assert_eq!(m.estimate(&key).vram, 1 << 30);

        let mut inst = m.activate(&key, Device::Gpu(0)).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 1);
        // single run
        assert!(inst.run("run", &Invocation::new(), &mut |_| {}).is_ok());
        // default batch = sequential loop
        let res = inst.run_batch("run", &[Invocation::new(), Invocation::new()], &mut |_, _| {});
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|r| r.is_ok()));

        drop(inst); // demote
        assert_eq!(live.load(Ordering::SeqCst), 0, "dropping the instance must free the device");
    }

    /// The default `demote`/`promote`/`estimate_at` must be a complete no-op
    /// for a model that never overrides them — this is the "every existing
    /// model's behaviour is unchanged" guarantee the whole design depends
    /// on, verified rather than merely claimed.
    #[test]
    fn demote_promote_and_estimate_at_default_to_unsupported_and_hot_only() {
        let live = Arc::new(AtomicU32::new(0));
        let m = FakeModel { vram: 1 << 30, live: live.clone() };
        let key = m.instance_key("run", &Invocation::new());

        assert_eq!(m.estimate_at(&key, Tier::Warm), m.estimate(&key), "default estimate_at ignores tier");
        assert_eq!(m.estimate_at(&key, Tier::Cold), m.estimate(&key));

        let mut inst = m.activate(&key, Device::Gpu(0)).unwrap();
        assert!(inst.demote(Tier::Warm).is_err(), "default demote must refuse, not silently pretend to succeed");
        assert!(inst.promote(Device::Gpu(0)).is_err());
        assert_eq!(live.load(Ordering::SeqCst), 1, "a refused demote must not have freed anything");
    }

    /// A model that DOES opt in: `demote(Warm)` releases the simulated
    /// device resource (the shared counter) while the `Instance` itself
    /// stays alive (unlike a full evict+drop, which the FakeModel/
    /// FakeInstance pair above already covers) — `promote` re-acquires it.
    /// This is the contract [`ResidencyManager`](crate::manager::
    /// ResidencyManager) can build real Warm-tier eviction on, proven here
    /// at the trait level.
    struct DemotableInstance {
        live: Arc<AtomicU32>,
        resident: bool,
    }
    impl Instance for DemotableInstance {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new())
        }
        fn demote(&mut self, tier: Tier) -> Result<(), String> {
            if tier == Tier::Hot {
                return Err("demote: Hot is not a demotion target".to_string());
            }
            if self.resident {
                self.live.fetch_sub(1, Ordering::SeqCst);
                self.resident = false;
            }
            Ok(())
        }
        fn promote(&mut self, _device: Device) -> Result<(), String> {
            if !self.resident {
                self.live.fetch_add(1, Ordering::SeqCst);
                self.resident = true;
            }
            Ok(())
        }
    }

    #[test]
    fn a_model_that_opts_in_can_be_demoted_to_warm_then_promoted_back() {
        let live = Arc::new(AtomicU32::new(1)); // starts Hot/resident
        let mut inst: Box<dyn Instance> = Box::new(DemotableInstance { live: live.clone(), resident: true });

        assert!(inst.demote(Tier::Warm).is_ok());
        assert_eq!(live.load(Ordering::SeqCst), 0, "demote(Warm) must release the device resource");

        assert!(inst.promote(Device::Gpu(0)).is_ok());
        assert_eq!(live.load(Ordering::SeqCst), 1, "promote must rebuild what demote released");

        // Idempotent in the direction that matters: demoting an
        // already-Warm instance again does not double-release.
        assert!(inst.demote(Tier::Cold).is_ok());
        assert_eq!(live.load(Ordering::SeqCst), 0);
        assert!(inst.demote(Tier::Cold).is_ok());
        assert_eq!(live.load(Ordering::SeqCst), 0, "demoting an already-non-resident instance must not underflow");
    }
}
