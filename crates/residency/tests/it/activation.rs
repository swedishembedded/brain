// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for placing models on GPUs with
// room for what they allocate while they run for its clients. If your team
// needs expertise in GPU memory planning then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A model's peak activation in its placement (`MemCost::activation`): what a
//! model declares is charged beside its resident bytes, and the operator's
//! measured fallback is charged only to a model that declares nothing.

use std::collections::HashSet;
use std::sync::Arc;

use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::budget::Budgets;
use residency::manager::{ClaimError, ResidencyManager};
use residency::{Device, Instance, InstanceKey, MemCost, MultiDeviceCost, MultiDeviceResidentModel, ResidentModel};

const GIB: u64 = 1 << 30;

struct Fake {
    name: &'static str,
    cost: MemCost,
    multi: Option<MultiDeviceCost>,
}

struct FakeInst;

impl Instance for FakeInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        Ok(Outcome::new())
    }
}

impl ResidentModel for Fake {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name, "fake", vec![ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.name, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        self.cost
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeInst))
    }
}

impl MultiDeviceResidentModel for Fake {
    fn estimate_multi(&self, _k: &InstanceKey) -> MultiDeviceCost {
        self.multi.clone().unwrap_or_default()
    }
    fn activate_multi(&self, _k: &InstanceKey, _d: &[Device]) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeInst))
    }
}

fn model(name: &'static str, cost: MemCost) -> Arc<Fake> {
    Arc::new(Fake { name, cost, multi: None })
}

/// Claim `name`, adopt a built instance and release it, as a finished run would.
fn run_once(mgr: &mut ResidencyManager, name: &str) -> Result<u64, ClaimError> {
    let (_, device, key) = mgr.claim(name, "run", &Invocation::new(), 0, &HashSet::new())?;
    mgr.adopt(&key, Arc::new(std::sync::Mutex::new(Box::new(FakeInst) as Box<dyn Instance>)));
    mgr.release(&key);
    Ok(mgr.report().placements.iter().find(|p| p.key == key && p.device == device).map(|p| p.mem).expect("placed"))
}

#[test]
fn a_declared_activation_is_charged_and_the_fallback_only_where_none_is_declared() {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 10 * GIB, 0).set_activation_fallback(3 * GIB);
    let mut mgr = ResidencyManager::new(budgets);
    mgr.register(model("declares", MemCost::new(4 * GIB, 0).with_activation(GIB)));
    mgr.register(model("silent", MemCost::new(4 * GIB, 0)));
    mgr.register(model("host", MemCost::new(0, GIB)));
    mgr.register(model("too-big", MemCost::new(8 * GIB, 0).with_activation(3 * GIB)));

    assert_eq!(run_once(&mut mgr, "declares").unwrap(), 5 * GIB, "its own estimate, not the fallback");
    // 4 GiB resident plus the 3 GiB fallback no longer fits beside the 5 GiB one: it is evicted.
    assert_eq!(run_once(&mut mgr, "silent").unwrap(), 7 * GIB, "the measured fallback stands in for a missing estimate");
    assert_eq!(mgr.report().placements.len(), 1, "the declared one gave way: {:?}", mgr.report().placements);
    assert!(matches!(mgr.claim("too-big", "run", &Invocation::new(), 0, &HashSet::new()), Err(ClaimError::TooLarge(_))), "a model whose peak does not fit any card is refused before anything is evicted");
    assert_eq!(mgr.report().placements.len(), 1);

    let mut host_only = Budgets::new();
    host_only.set(Device::Cpu, 8 * GIB, 0).set_activation_fallback(3 * GIB);
    let mut mgr = ResidencyManager::new(host_only);
    mgr.register(model("host", MemCost::new(0, GIB)));
    assert_eq!(run_once(&mut mgr, "host").unwrap(), GIB, "a model holding no device bytes runs no device batch: no fallback");
}

#[test]
fn a_multi_device_model_is_charged_its_activation_on_every_device() {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 10 * GIB, 0).set(Device::Gpu(1), 10 * GIB, 0).set_activation_fallback(3 * GIB);
    let mut mgr = ResidencyManager::new(budgets);
    let per_device = vec![(Device::Gpu(0), 4 * GIB), (Device::Gpu(1), 2 * GIB)];
    let declares = Fake { name: "staged", cost: MemCost::default(), multi: Some(MultiDeviceCost::new(per_device.clone(), 0).with_activation(GIB)) };
    let silent = Fake { name: "silent", cost: MemCost::default(), multi: Some(MultiDeviceCost::new(per_device, 0)) };
    mgr.register_multi(Arc::new(declares));
    mgr.register_multi(Arc::new(silent));
    let used = |mgr: &ResidencyManager| -> Vec<u64> { mgr.report().budgets.iter().filter(|b| matches!(b.device, Device::Gpu(_))).map(|b| b.used).collect() };

    let (_, _, key) = mgr.claim_multi("staged", "run", &Invocation::new(), 0, &HashSet::new()).unwrap();
    let mut by_device = used(&mgr);
    by_device.sort_unstable();
    assert_eq!(by_device, vec![3 * GIB, 5 * GIB]);
    mgr.build_failed_multi(&key);
    assert_eq!(used(&mgr), vec![0, 0], "the whole charge is given back");

    mgr.claim_multi("silent", "run", &Invocation::new(), 0, &HashSet::new()).unwrap();
    let mut by_device = used(&mgr);
    by_device.sort_unstable();
    assert_eq!(by_device, vec![5 * GIB, 7 * GIB], "the fallback on each device a stage runs on");
}
