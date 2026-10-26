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

use crate::{AdapterIdentity, Device, InstanceKey, MemCost, Tier};

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

    /// How many instances of this model the host can run at once, each on its own devices. The
    /// executor spreads jobs over the replicas (a job goes to the least loaded one); replica `i` is
    /// the build named by `instance_key(..).replica(i)`, and [`Self::estimate`] and
    /// [`Self::activate`] receive that key and place replica `i` where this model decided at
    /// construction. A model that can use spare capacity (a free card) says how many it can hold;
    /// the default is the one instance every model has had. Constant for the life of the model.
    fn replicas(&self) -> usize {
        1
    }

    /// The most queued jobs one [`Instance::run_batch`] call of this model
    /// should receive, when that differs from the scheduler policy's
    /// `max_batch`. A model that decodes many requests per step (a
    /// continuous-batching LLM with a wide slot pool) declares its slot count
    /// here, so the dispatcher hands it everything queued up to its capacity
    /// rather than the policy's small default. `None` (the default) defers to
    /// the policy.
    fn batch_limit(&self) -> Option<usize> {
        None
    }

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

    /// Mark an invocation the executor accepted for this model, before it is
    /// grouped and run. The one seam through which a model that serves several
    /// selectable ids from one built instance (a base and the adapters riding
    /// on it, each its own model id sharing an [`InstanceKey`]) tells the
    /// instance WHICH id the caller asked for: the instance only ever sees
    /// [`Invocation`]s, never the model string.
    ///
    /// Runs on the dispatcher thread, after the caller's params were
    /// validated and never before: a param written here is the host's word,
    /// not the caller's, so it must not be a name the model's manifest
    /// declares (validation refuses an undeclared param from a caller, which
    /// is what keeps a reserved param unforgeable). A model that serves more
    /// than one id and reads such a param must also override this on its
    /// base id to remove it, or a caller reaching the base through a path
    /// that skips validation could select a sibling.
    ///
    /// Defaulted to a no-op: a model serving one id has nothing to say.
    fn stamp(&self, _inv: &mut Invocation) {}

    /// The fine-tune this model id serves, when it is an adapter view: its
    /// card id and content digest, the facts a model card reports. The
    /// executor records it under the model id when the model is registered,
    /// so a model listing names the adapter without a second call; a later
    /// change of the served release is recorded with
    /// [`Executor::set_served_adapter`](crate::Executor::set_served_adapter).
    ///
    /// Defaulted to `None`: no adapter is known, and none is invented.
    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        None
    }
}

/// A model served under a second id by the SAME resident: every request for
/// `id` runs on the inner model's instance (one set of weights, one batch),
/// exactly as if it had named the inner model. Use it where two ids must mean
/// one engine instead of loading a second copy of the checkpoint.
pub struct Alias<M: ResidentModel + ?Sized> {
    id: String,
    inner: std::sync::Arc<M>,
}

impl<M: ResidentModel + ?Sized> Alias<M> {
    pub fn new(id: impl Into<String>, inner: std::sync::Arc<M>) -> Alias<M> {
        Alias { id: id.into(), inner }
    }
}

impl<M: ResidentModel + ?Sized> ResidentModel for Alias<M> {
    fn manifest(&self) -> Manifest {
        let mut manifest = self.inner.manifest();
        manifest.model = self.id.clone();
        manifest
    }
    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.inner.instance_key(action, inv)
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.inner.estimate(key)
    }
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        self.inner.estimate_at(key, tier)
    }
    fn batch_limit(&self) -> Option<usize> {
        self.inner.batch_limit()
    }
    fn replicas(&self) -> usize {
        self.inner.replicas()
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        self.inner.activate(key, device)
    }
    fn stamp(&self, inv: &mut Invocation) {
        self.inner.stamp(inv)
    }
    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        self.inner.adapter_identity()
    }
}

/// The live end of a running job group: how an [`Instance::run_open`] call
/// learns of jobs that arrived AFTER the group started, and hands each job's
/// result back the moment it is done instead of when the whole group is.
///
/// Indices are the order jobs joined the group: the `invs` given to
/// `run_open` are `0..invs.len()`, each job [`take`](Self::take) returns
/// follows, and the `progress`/`finish` calls name a job by that index.
pub trait JobFeed {
    /// Jobs queued for this group since the last call, at most `max`.
    /// Non-blocking; empty when nothing has arrived or `max` is 0.
    fn take(&mut self, max: usize) -> Vec<Invocation>;

    /// Job `index` is finished: deliver its result to its caller now.
    /// Frees the slot it held, so a queued job can take it.
    fn finish(&mut self, index: usize, result: ActionResult);

    /// Turn away jobs that are waiting to join this group but can never run,
    /// without giving them a slot. `check` is asked once about each waiting job
    /// that has not been asked yet and answers with the error to deliver, or
    /// `None` to leave the job waiting. A refused job's caller is answered at
    /// once, however long the running jobs will take: refusing a request is not
    /// work that should wait for a free slot. The default refuses nothing.
    fn screen(&mut self, check: &mut dyn FnMut(&Invocation) -> Option<String>) {
        let _ = check;
    }

    /// Stop accepting further jobs because the instance has nothing running.
    /// `false` means a job slipped in between the last `take` and now - it is
    /// waiting to be taken and the group must not end yet. After `true`, jobs
    /// for this key wait for the next group.
    fn close(&mut self) -> bool;
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

    /// Run `invs` as an OPEN group: a model that decodes many requests per step
    /// admits jobs that arrive while the group runs, between its steps, and
    /// finishes each job as soon as it completes - continuous batching across
    /// everything queued for this instance, not only what was queued when the
    /// call started. Returns `true` when it handled the group (every job's
    /// result delivered through `feed`); `false`, having done nothing, when
    /// the model only batches a closed group ([`Self::run_batch`]) - the
    /// default.
    fn run_open(&mut self, _action: &str, _invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress), _feed: &mut dyn JobFeed) -> bool {
        false
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
    struct One;
    impl ResidentModel for One {
        fn manifest(&self) -> Manifest {
            Manifest::new("one", "a model", Vec::new())
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("one", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(1, 0)
        }
        fn batch_limit(&self) -> Option<usize> {
            Some(8)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Err("not built in this test".into())
        }
    }

    /// A second id for the same resident must be listed under its own name yet
    /// run on the inner model's instance, or every id would load its own copy
    /// of the weights and never share a batch.
    #[test]
    fn an_alias_is_listed_under_its_own_id_and_shares_the_inner_instance() {
        let inner = Arc::new(One);
        let alias = Alias::new("other", inner.clone());
        assert_eq!(alias.manifest().model, "other");
        assert_eq!(inner.manifest().model, "one");
        let inv = Invocation::new();
        assert_eq!(alias.instance_key("generate", &inv), inner.instance_key("generate", &inv));
        assert_eq!(alias.batch_limit(), Some(8));
    }
}
