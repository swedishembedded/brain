// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`ResidencyManager`]: given `(model, action, invocation)`, ensure the right
//! model instance is **Hot** on a device — placing it on the emptiest GPU that fits,
//! or evicting least-recently-used instances to make room — then run it. Dropping an
//! evicted [`Instance`] frees its device memory (RAII). This is the "use all the
//! memory automatically" core; the scheduler (next) drives concurrency and batching
//! on top of it.
//!
//! Single-device instances are handled directly. A model that spans multiple devices
//! (e.g. z-image's encoder + DiT on two cards) reports the extra footprint via
//! [`ResidentModel::estimate`]'s `ram`/secondary accounting and pins its own cards;
//! the manager still tracks the primary-device budget.

use std::collections::HashMap;
use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::budget::Budgets;
use crate::lru::Residents;
use crate::place::{pick_device, plan_eviction};
use crate::{Device, InstanceKey, ResidentModel, Tier};

/// Owns the resident model instances, their budgets, and the LRU. Not thread-safe by
/// itself — the scheduler owns one behind its worker(s).
pub struct ResidencyManager {
    models: HashMap<String, Arc<dyn ResidentModel>>,
    budgets: Budgets,
    residents: Residents,
    instances: HashMap<InstanceKey, Box<dyn crate::Instance>>,
    /// Eviction/promotion audit log (most recent last) for reporting/tests.
    pub events: Vec<String>,
}

impl ResidencyManager {
    pub fn new(budgets: Budgets) -> ResidencyManager {
        ResidencyManager { models: HashMap::new(), budgets, residents: Residents::new(), instances: HashMap::new(), events: Vec::new() }
    }

    pub fn register(&mut self, model: Arc<dyn ResidentModel>) {
        self.models.insert(model.manifest().model.clone(), model);
    }

    pub fn manifests(&self) -> Vec<Manifest> {
        self.models.values().map(|m| m.manifest()).collect()
    }

    /// The instance key for `(model, action, inv)`, or `None` if the model is unknown.
    pub fn instance_key_for(&self, model: &str, action: &str, inv: &Invocation) -> Option<InstanceKey> {
        self.models.get(model).map(|m| m.instance_key(action, inv))
    }

    pub fn models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.models.keys().cloned().collect();
        v.sort();
        v
    }

    /// Current tier of each resident instance (for `Residency` reporting).
    pub fn residency(&self) -> Vec<(InstanceKey, Device, Tier)> {
        self.residents.iter().map(|(k, e)| (k.clone(), e.device, Tier::Hot)).collect()
    }

    pub fn budgets(&self) -> &Budgets {
        &self.budgets
    }

    /// Ensure the instance for `(model, action, inv)` is Hot, then run the action.
    /// Promotes (evicting LRU as needed) automatically. Pins the instance while it
    /// runs so a concurrent request can't evict it mid-job.
    pub fn run(&mut self, model: &str, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let m = self.models.get(model).ok_or_else(|| format!("no model '{model}'"))?.clone();
        let key = m.instance_key(action, inv);
        self.ensure_hot(&key, &m)?;
        self.residents.set_pinned(&key, true);
        let out = self.instances.get_mut(&key).expect("hot").run(action, inv, progress);
        self.residents.set_pinned(&key, false);
        self.residents.touch(&key);
        out
    }

    /// Run several same-key invocations of one action on a single hot instance —
    /// the hot-path-reuse batch. The instance is promoted once and pinned for the
    /// whole group (so it can't be evicted between jobs), then its `run_batch` runs
    /// them (a model with real batch support does one forward; others loop).
    pub fn run_batch(&mut self, model: &str, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(Progress)) -> Result<Vec<ActionResult>, String> {
        let m = self.models.get(model).ok_or_else(|| format!("no model '{model}'"))?.clone();
        let key = m.instance_key(action, invs.first().ok_or("empty batch")?);
        self.ensure_hot(&key, &m)?;
        self.residents.set_pinned(&key, true);
        let out = self.instances.get_mut(&key).expect("hot").run_batch(action, invs, progress);
        self.residents.set_pinned(&key, false);
        self.residents.touch(&key);
        Ok(out)
    }

    /// Promote `key` to Hot if not already: pick a device (evicting LRU victims when
    /// nothing fits), build the instance, and account its budget.
    pub fn ensure_hot(&mut self, key: &InstanceKey, m: &Arc<dyn ResidentModel>) -> Result<(), String> {
        if self.instances.contains_key(key) {
            self.residents.touch(key);
            return Ok(());
        }
        let cost = m.estimate(key);
        let device = match pick_device(&cost, &self.budgets) {
            Some(d) => d,
            None => {
                let plan = plan_eviction(&cost, &self.budgets, &self.residents, &[key.clone()])
                    .ok_or_else(|| format!("{key} ({} MiB) is too large for any device", cost.vram >> 20))?;
                for victim in &plan.victims {
                    self.evict(victim);
                }
                plan.device
            }
        };
        let inst = m.activate(key, device)?;
        self.budgets.alloc(device, cost.on(device));
        self.residents.insert(key.clone(), cost, device);
        self.instances.insert(key.clone(), inst);
        self.events.push(format!("promote {key} -> {device:?}"));
        Ok(())
    }

    /// Demote (drop) an instance, freeing its device memory.
    fn evict(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.residents.remove(key) {
            self.budgets.release(entry.device, entry.cost.on(entry.device));
            self.instances.remove(key); // drops the Instance → frees the GPU
            self.events.push(format!("evict {key} <- {:?}", entry.device));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemCost;
    use capability::{ActionResult, ActionSpec, Blob, Media, Outcome};
    use std::sync::atomic::{AtomicU32, Ordering};

    const GB: u64 = 1 << 30;

    /// A fake model whose instances count live GPU builds (via a shared counter), so
    /// a test can watch automatic swap free memory without a GPU.
    struct Fake {
        name: String,
        vram: u64,
        live: Arc<AtomicU32>,
    }
    struct FakeInst {
        live: Arc<AtomicU32>,
    }
    impl Drop for FakeInst {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl ResidentModel for Fake {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }
    impl crate::Instance for FakeInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    #[test]
    fn three_models_on_one_gpu_swap_by_lru() {
        // One 24 GB card, 2 GB reserved → 22 usable. Three 10 GB models: two fit, the
        // third forces the LRU one out.
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB);
        let mut mgr = ResidencyManager::new(budgets);
        for n in ["a", "b", "c"] {
            mgr.register(Arc::new(Fake { name: n.into(), vram: 10 * GB, live: live.clone() }));
        }

        // Run a, then b — both fit (20 GB <= 22).
        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 2);
        assert_eq!(mgr.residency().len(), 2);

        // Run c → needs 10, only 2 free → evict LRU (a) → 2 resident, c hot, a gone.
        mgr.run("c", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 2, "one instance evicted, memory freed");
        let hot: Vec<String> = mgr.residency().into_iter().map(|(k, _, _)| k.model).collect();
        assert!(hot.contains(&"c".to_string()) && hot.contains(&"b".to_string()) && !hot.contains(&"a".to_string()));
        assert!(mgr.events.iter().any(|e| e.contains("evict a")), "events: {:?}", mgr.events);

        // Re-running b (still hot) is a no-op promotion.
        let before = live.load(Ordering::SeqCst);
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), before);
    }

    #[test]
    fn balances_across_two_gpus() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        for n in ["a", "b"] {
            mgr.register(Arc::new(Fake { name: n.into(), vram: 20 * GB, live: live.clone() }));
        }
        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        // Both resident, one per card (b placed on the emptier GPU 1).
        let devs: Vec<Device> = mgr.residency().into_iter().map(|(_, d, _)| d).collect();
        assert!(devs.contains(&Device::Gpu(0)) && devs.contains(&Device::Gpu(1)));
        assert_eq!(live.load(Ordering::SeqCst), 2);
    }
}
