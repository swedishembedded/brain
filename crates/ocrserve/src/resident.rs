// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An OCR model behind the residency scheduler.
//!
//! `activate_multi` builds the whole model ONCE - the vision encoder's weight
//! upload, the decoder engine's weight upload and its paged KV pool - and the
//! [`Instance`] it returns is a `llmserve::ScheduledInstance`, so dropping it
//! frees every buffer. One action, `generate`; its schema and all of its work
//! come from [`crate::caps`] and [`crate::frontend`], so a model holds no second
//! copy of the batching, the token accounting or the placement.
//!
//! # This model spans TWO devices, and names both
//!
//! The vision encoder builds on the ambient card and the decoder on whichever
//! device its [`DecoderDevice`] names - a second card, the vision card, or the
//! CPU. [`MultiDeviceCost`] is the honest expression: every device the instance
//! touches is NAMED, with its own byte count, checked against its own budget.
//!
//! * decoder on a second card - `(Gpu(v), vision)`, `(Gpu(d), decoder)` and the
//!   host working set as the cost's `ram` (not a named CPU placement: naming
//!   `Device::Cpu` made the model unplaceable under `--device gpu0`).
//! * decoder on the vision card - one GPU entry holding both, plus the same host figure.
//! * decoder on the CPU - `(Gpu(v), vision)` and `(Cpu, host_cpu_decoder)`.
//!
//! Nothing here touches the environment: placement is a scoped registry
//! selection (`gpu_core::devices::with_gpu`). The placement is computed ONCE, at
//! construction, and passed down, so the reservation and the allocation cannot
//! name different cards.
//!
//! # Replicas
//!
//! When two or more cards each hold the whole model, one replica runs on each
//! ([`plan_replicas`]); the scheduler spreads requests across them.

use std::marker::PhantomData;

use capability::{Assembly, Invocation, Manifest};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::placement::{plan_replicas, DecoderDevice, Placement};
use crate::{caps, dir_from_assembly, OcrModel};

/// An OCR model `M` behind the scheduler. `dir` is the checkpoint directory the
/// model-store resolver picked.
pub struct OcrResident<M: OcrModel> {
    dir: String,
    /// One placement per replica, decided ONCE, at construction.
    plans: Vec<Placement>,
    _model: PhantomData<M>,
}

impl<M: OcrModel> OcrResident<M> {
    /// The `catalog::MultiCtor` shape: `dir` comes from an already-resolved
    /// [`Assembly`]'s `dir` role.
    pub fn from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Option<OcrResident<M>> {
        match dir_from_assembly(assembly) {
            Ok(dir) => Self::new(dir, gpus, reserved),
            Err(e) => {
                residency::log::not_served(&format!("brain: {} not served ({e})", M::ID));
                None
            }
        }
    }

    /// Direct constructor: `None` when `dir` holds no checkpoint the model loads,
    /// rather than a resident that fails every call.
    pub fn new(dir: impl Into<String>, gpus: &[(u32, u64)], reserved: u64) -> Option<OcrResident<M>> {
        let dir = dir.into();
        match M::locate(&dir) {
            Ok(()) => {
                let cards: Vec<(u32, u64)> = gpus.iter().map(|&(i, total)| (i, total.saturating_sub(reserved))).collect();
                Some(OcrResident { dir, plans: plan_replicas(&cards, &M::FOOTPRINT), _model: PhantomData })
            }
            Err(e) => {
                residency::log::not_served(&format!("brain: {} not served ({e})", M::ID));
                None
            }
        }
    }

    /// A resident at explicit placements, for accounting tests that must not
    /// depend on the box's own device registry.
    pub fn with_plans(dir: impl Into<String>, plans: Vec<Placement>) -> OcrResident<M> {
        OcrResident { dir: dir.into(), plans, _model: PhantomData }
    }

    /// The placement of the replica `key` names.
    pub fn plan_of(&self, key: &InstanceKey) -> Placement {
        self.plans[key.replica_index().min(self.plans.len() - 1)]
    }
}

impl<M: OcrModel> ResidentModel for OcrResident<M> {
    fn manifest(&self) -> Manifest {
        caps::manifest_resident::<M>()
    }

    /// One replica per card that holds the whole model when there are two or more.
    fn replicas(&self) -> usize {
        self.plans.len()
    }

    /// The pages the engine decodes together: the dispatcher hands the instance everything queued up to this, and admits later
    /// arrivals into the running group, instead of the scheduler policy's small default (eight at a time, the next eight only
    /// when the whole group had finished).
    fn batch_limit(&self) -> Option<usize> {
        Some(crate::policy::config().batch() as usize)
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One model serves every request: nothing in an invocation can fork the
        // graph. Keying on anything else would duplicate a multi-GiB build.
        InstanceKey::new(M::ID, self.dir.clone())
    }

    /// Deliberately unusable: this model is registered via `register_multi` and
    /// claimed via `claim_multi`, so the single-device estimate is never consulted.
    /// A real figure here would invite the plain `register` path, whose single
    /// `budgets.alloc` can only charge ONE of the two devices this instance occupies.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{}: single-device activate is not supported -- this model spans a GPU (vision) and a second device (decoder), so it must be claimed via ResidencyManager::claim_multi", M::ID))
    }
}

impl<M: OcrModel> MultiDeviceResidentModel for OcrResident<M> {
    fn estimate_multi(&self, key: &InstanceKey) -> MultiDeviceCost {
        // Cheap and panic-free by construction: this runs on the dispatcher
        // thread on every scheduling round, and a panic there kills serving for
        // every OTHER model too.
        let fp = M::FOOTPRINT;
        let Placement { vision, decoder } = self.plan_of(key);
        let Some(v) = vision else {
            // With no card big enough for the encoder everything this model holds is host bytes.
            return MultiDeviceCost::new(vec![(Device::Cpu, fp.composite_peak())], 0);
        };
        let (cost, ram) = match decoder {
            DecoderDevice::Cpu => (vec![(Device::Gpu(v), fp.vision_device), (Device::Cpu, fp.host_cpu_decoder)], 0),
            // A `Card(i)` that names the vision card is the same pool as `SameCard`, charged once.
            DecoderDevice::SameCard => (vec![(Device::Gpu(v), fp.vision_device + fp.decoder_device)], fp.host_gpu_decoder),
            DecoderDevice::Card(i) if i == v => (vec![(Device::Gpu(v), fp.vision_device + fp.decoder_device)], fp.host_gpu_decoder),
            DecoderDevice::Card(i) => (vec![(Device::Gpu(v), fp.vision_device), (Device::Gpu(i), fp.decoder_device)], fp.host_gpu_decoder),
        };
        MultiDeviceCost::new(cost, ram)
    }

    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        // `claim_multi` reserves against exactly the devices `estimate_multi`
        // named. Insisting on it here makes the reservation and the allocation
        // describe the same bytes.
        let planned: Vec<Device> = self.estimate_multi(key).devices().collect();
        if devices.len() != planned.len() || !devices.iter().all(|d| planned.contains(d)) {
            return Err(format!("{}: activate_multi got devices {devices:?} but the plan placed {planned:?}", M::ID));
        }
        let Placement { vision, decoder } = self.plan_of(key);
        let dir = key.base().config;
        match vision {
            Some(i) => gpu_core::devices::with_gpu(i, || M::load(&dir, decoder))?,
            None => gpu_core::devices::with_host_tier(|| M::load(&dir, decoder)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Footprint;

    const GB: u64 = 1 << 30;

    struct Toy;
    impl OcrModel for Toy {
        const ID: &'static str = "toy/ocr";
        const SUMMARY: &'static str = "a toy";
        const FOOTPRINT: Footprint = Footprint { vision_device: 7 * GB, decoder_device: 14 * GB, host_gpu_decoder: 3 * GB, host_cpu_decoder: 14 * GB };
        fn locate(_dir: &str) -> Result<(), String> {
            Err("no checkpoint".into())
        }
        fn load(_dir: &str, _placement: DecoderDevice) -> Result<Box<dyn Instance>, String> {
            Err("no checkpoint".into())
        }
    }

    fn at(vision: u32, decoder: DecoderDevice) -> OcrResident<Toy> {
        OcrResident::with_plans("/tmp", vec![Placement { vision: Some(vision), decoder }])
    }

    fn key(r: &OcrResident<Toy>) -> InstanceKey {
        r.instance_key("generate", &Invocation::new())
    }

    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(OcrResident::<Toy>::new("/definitely/not/a/dir", &[(0, 24 * GB)], 2 * GB).is_none());
    }

    /// Every device this instance really holds bytes on must be named and
    /// budgeted. A cost that omits one is how another model gets placed on top
    /// of memory this one already holds.
    #[test]
    fn every_placement_names_every_device_it_occupies() {
        let fp = Toy::FOOTPRINT;
        let r = at(1, DecoderDevice::Cpu);
        let cost = r.estimate_multi(&key(&r));
        assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Gpu(1), Device::Cpu]);
        assert_eq!(cost.on(Device::Gpu(1)), fp.vision_device);
        assert_eq!(cost.on(Device::Cpu), fp.host_cpu_decoder);

        // Decoder on its own card: two cards, each with its own figure; the host
        // working set is staging (`ram`), never a named CPU placement.
        let r = at(1, DecoderDevice::Card(0));
        let cost = r.estimate_multi(&key(&r));
        assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Gpu(1), Device::Gpu(0)]);
        assert_eq!(cost.on(Device::Gpu(0)), fp.decoder_device);
        assert_eq!(cost.ram(), fp.host_gpu_decoder);

        // Decoder on the encoder's own card: ONE entry holding both.
        for placement in [DecoderDevice::SameCard, DecoderDevice::Card(1)] {
            let r = at(1, placement);
            let cost = r.estimate_multi(&key(&r));
            assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Gpu(1)], "{placement:?}");
            assert_eq!(cost.on(Device::Gpu(1)), fp.vision_device + fp.decoder_device, "{placement:?}");
            assert_eq!(cost.ram(), fp.host_gpu_decoder, "{placement:?}");
        }
    }

    /// Moving the decoder onto a card must MOVE its weights off the host, not
    /// merely add a card entry beside an unchanged host claim.
    #[test]
    fn putting_the_decoder_on_a_card_takes_its_weights_off_the_host() {
        let (cpu, gpu) = (at(1, DecoderDevice::Cpu), at(1, DecoderDevice::Card(0)));
        let (a, b) = (cpu.estimate_multi(&key(&cpu)), gpu.estimate_multi(&key(&gpu)));
        let freed = a.on(Device::Cpu) - b.ram();
        assert!(freed >= 10 * GB, "moving the decoder to a card freed only {freed} host bytes");
        assert!(b.on(Device::Gpu(0)) >= 10 * GB, "and those bytes must be charged to the card that now holds them");
        assert_eq!(a.ram(), 0, "the CPU decoder names the host as a placement");
    }

    /// A model with no card for its encoder is all host bytes.
    #[test]
    fn no_card_for_the_encoder_is_all_host() {
        let r = OcrResident::<Toy>::with_plans("/tmp", vec![Placement { vision: None, decoder: DecoderDevice::Cpu }]);
        let cost = r.estimate_multi(&key(&r));
        assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Cpu]);
        assert_eq!(cost.on(Device::Cpu), Toy::FOOTPRINT.composite_peak());
    }

    /// The single-device path is structurally refused, so this model cannot be
    /// registered through `Executor::register` by mistake and end up with one of
    /// its two devices unbudgeted; `activate_multi` refuses a device set it did not plan.
    #[test]
    fn only_the_multi_device_path_activates() {
        let r = at(1, DecoderDevice::Cpu);
        assert_eq!(r.estimate(&key(&r)), MemCost::new(0, 0));
        assert!(r.activate(&key(&r), Device::Gpu(0)).err().unwrap_or_default().contains("claim_multi"));
        assert!(r.activate_multi(&key(&r), &[Device::Cpu]).err().unwrap_or_default().contains("but the plan placed"));
    }

    /// Two cards that each hold the whole model run one replica each, each
    /// charged to its own card alone.
    #[test]
    fn the_resident_asks_the_dispatcher_for_a_group_as_wide_as_the_engine_decodes() {
        let r = at(0, DecoderDevice::SameCard);
        assert_eq!(r.batch_limit(), Some(crate::MAX_BATCH as usize), "without it the dispatcher runs eight pages at a time and starts the next eight only when all have finished");
    }

    #[test]
    fn two_whole_cards_are_two_replicas_each_charged_to_its_own_card() {
        let plans = plan_replicas(&[(0, 90 * GB), (1, 90 * GB)], &Toy::FOOTPRINT);
        let r = OcrResident::<Toy>::with_plans("/tmp", plans);
        assert_eq!(r.replicas(), 2);
        let base = key(&r);
        for i in 0..2usize {
            let cost = r.estimate_multi(&base.replica(i));
            assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Gpu(i as u32)], "replica {i}");
        }
    }
}
