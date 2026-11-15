// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping models sharded across
// several accelerators available through driver faults for its clients. If
// your team needs expertise in supervised multi-device execution then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A model across several cards, hosted whole by one worker: its activation
//! opens every card of the placement in that worker (`open gpu0,gpu1 <spec>`),
//! and a fault the worker attributes (`fault <device>`) quarantines that card
//! alone, where a silent death quarantines every card it held.

use std::sync::Arc;

use capability::{Invocation, Manifest};
use gpu_supervisor::{Health, WorkerCommand};

use super::{Deadlines, HostSpec, SupervisedInstance, WorkerStart, WORKER_DEVICE_CONTEXT_BYTES};
use crate::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use crate::{AdapterIdentity, Device, Instance, InstanceKey, MemCost, ResidentModel};

/// A multi-device model served by one supervised worker process that owns
/// every card of an instance. Register it with `Executor::register_multi`.
pub struct SupervisedMultiModel {
    inner: Arc<dyn MultiDeviceResidentModel>,
    spec: HostSpec,
    command: WorkerCommand,
    health: Arc<Health<Device>>,
    deadlines: Deadlines,
}

impl SupervisedMultiModel {
    /// `inner` answers everything the scheduler asks without a device; a worker started from `command` rebuilds it
    /// from `spec` and runs it across the cards of its placement.
    pub fn new(inner: Arc<dyn MultiDeviceResidentModel>, spec: HostSpec, command: WorkerCommand, health: Arc<Health<Device>>, deadlines: Deadlines) -> SupervisedMultiModel {
        SupervisedMultiModel { inner, spec, command, health, deadlines }
    }
}

impl ResidentModel for SupervisedMultiModel {
    fn manifest(&self) -> Manifest {
        self.inner.manifest()
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.inner.instance_key(action, inv)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.inner.estimate(key)
    }

    fn batch_limit(&self) -> Option<usize> {
        self.inner.batch_limit()
    }

    fn stamp(&self, inv: &mut Invocation) {
        self.inner.stamp(inv)
    }

    fn check_job(&self, action: &str, inv: &Invocation) -> Result<(), String> {
        self.inner.check_job(action, inv)
    }

    fn transient(&self) -> bool {
        self.inner.transient()
    }

    fn adapter_identity(&self) -> Option<AdapterIdentity> {
        self.inner.adapter_identity()
    }

    /// A multi-device model is built only across its placement ([`MultiDeviceResidentModel::activate_multi`]).
    fn activate(&self, key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{key} spans several devices; it is registered with register_multi and built across them"))
    }
}

impl MultiDeviceResidentModel for SupervisedMultiModel {
    /// The model's own cost plus its worker's device context on every card it holds memory on.
    fn estimate_multi(&self, key: &InstanceKey) -> MultiDeviceCost {
        self.inner.estimate_multi(key).with_device_context(WORKER_DEVICE_CONTEXT_BYTES)
    }

    fn unavailable_reason(&self, key: &InstanceKey) -> Option<String> {
        self.inner.unavailable_reason(key)
    }

    /// One worker opens every card of the placement.
    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let start = WorkerStart::new(&self.spec, key, &self.command, &self.health, self.deadlines);
        SupervisedInstance::open(start, devices, None)
    }
}
