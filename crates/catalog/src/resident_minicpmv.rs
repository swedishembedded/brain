// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V behind the residency scheduler.
//!
//! The whole composite - vision tower, merger and the Qwen3.5 decoder with
//! its KV pool - lives on ONE card: at registration the checkpoint's
//! footprint (`minicpmv::model::footprint`, fp32 weights plus the pool for
//! `SERVE_CONCURRENT` sequences of `SERVE_CONTEXT` tokens plus the tower's
//! scratch) is placed on the budgeted card with the most room for it. All
//! request work lives in `minicpmv::caps`; this file holds placement and
//! budgeting only.

use capability::{ActionResult, Assembly, Invocation, Manifest, Progress};
use minicpmv::caps::{Session, MODEL, SERVE_CONCURRENT, SERVE_CONTEXT};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

pub struct MiniCpmVResident {
    dir: String,
    bytes: u64,
    card: u32,
}

/// The card among `gpus` (`(index, total bytes)`, `reserved` kept free on
/// each) with the most room left after `bytes`, if any fits.
fn pick_card(gpus: &[(u32, u64)], reserved: u64, bytes: u64) -> Option<u32> {
    gpus.iter().filter(|&&(_, total)| total.saturating_sub(reserved) >= bytes).max_by_key(|&&(_, total)| total).map(|&(i, _)| i)
}

impl MiniCpmVResident {
    pub fn from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Option<MiniCpmVResident> {
        let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned())?;
        let cfg = match minicpmv::MiniCpmVConfig::from_dir(std::path::Path::new(&dir)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("brain: minicpmv not served ({e})");
                return None;
            }
        };
        let bytes = minicpmv::model::footprint(&cfg, SERVE_CONTEXT, SERVE_CONCURRENT);
        match pick_card(gpus, reserved, bytes) {
            Some(card) => Some(MiniCpmVResident { dir, bytes, card }),
            None => {
                eprintln!("brain: minicpmv not served (needs {} MiB on one card)", bytes >> 20);
                None
            }
        }
    }
}

impl ResidentModel for MiniCpmVResident {
    fn manifest(&self) -> Manifest {
        minicpmv::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, self.dir.clone())
    }

    /// Unusable by design: the model is claimed through `claim_multi` on the
    /// card registration placed it on.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{MODEL}: placed on card {} at registration, so it must be claimed via ResidencyManager::claim_multi", self.card))
    }
}

impl MultiDeviceResidentModel for MiniCpmVResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        MultiDeviceCost::new(vec![(Device::Gpu(self.card), self.bytes)], 0)
    }

    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        if devices != [Device::Gpu(self.card)] {
            return Err(format!("{MODEL}: activate_multi got devices {devices:?} but the plan placed Gpu({})", self.card));
        }
        Ok(Box::new(MiniCpmVInstance { session: minicpmv::caps::load_session(&key.config, Some(self.card))? }))
    }
}

struct MiniCpmVInstance {
    session: Session,
}

impl Instance for MiniCpmVInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("minicpmv: unknown action '{action}'"));
        }
        self.session.generate(inv, progress)
    }

    /// Requests dispatched together decode as one batch on the shared KV pool.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        if action != "generate" {
            return invs.iter().map(|_| Err(format!("minicpmv: unknown action '{action}'"))).collect();
        }
        self.session.generate_batch(invs, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_roomiest_card_that_fits_is_chosen_and_none_when_none_fits() {
        let gib = 1u64 << 30;
        let gpus = [(0, 16 * gib), (1, 24 * gib), (2, 8 * gib)];
        assert_eq!(pick_card(&gpus, gib, 10 * gib), Some(1));
        assert_eq!(pick_card(&gpus, gib, 30 * gib), None);
        assert_eq!(pick_card(&[(3, 12 * gib)], 3 * gib, 9 * gib), Some(3), "the reserve counts against the card");
    }

    #[test]
    fn an_unplanned_device_is_refused() {
        let r = MiniCpmVResident { dir: "/nonexistent".into(), bytes: 1, card: 1 };
        let e = r.activate_multi(&r.instance_key("generate", &Invocation::new()), &[Device::Gpu(0)]).err().unwrap_or_default();
        assert!(e.contains("plan placed"), "{e}");
        let cost = r.estimate_multi(&r.instance_key("generate", &Invocation::new()));
        assert_eq!((cost.devices().collect::<Vec<_>>(), cost.on(Device::Gpu(1))), (vec![Device::Gpu(1)], 1));
    }
}
