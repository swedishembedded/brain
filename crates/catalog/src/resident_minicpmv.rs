// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V behind the residency scheduler.
//!
//! The whole composite - vision tower, merger and the Qwen3.5 decoder with
//! its KV pool - lives on ONE card: at registration the checkpoint's
//! footprint (`minicpmv::model::footprint`, fp32 weights plus the pool for
//! `SERVE_CONCURRENT` sequences of `SERVE_CONTEXT` tokens plus the tower's
//! scratch) is placed on the budgeted card with the most room for it. Each
//! fine-tune stored beside the checkpoint (`adapters/<owner>/<name>/<tag>/`,
//! what `brain minicpmv finetune` writes) is a model of its own,
//! `brain/minicpmv:<owner>:<name>:<tag>`, placed the same way. All request
//! work lives in `minicpmv::caps`; this file holds placement and budgeting
//! only.

use capability::{ActionResult, Assembly, Invocation, Manifest, Progress};
use minicpmv::caps::{Session, MODEL, SERVE_CONCURRENT, SERVE_CONTEXT};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

pub struct MiniCpmVResident {
    /// The model id this resident answers to: the base's, or the base's with
    /// a stored fine-tune's `:owner:name:tag`.
    id: String,
    dir: String,
    /// The stored fine-tune applied when the model loads.
    tune: Option<std::path::PathBuf>,
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
            Some(card) => Some(MiniCpmVResident { id: MODEL.to_string(), dir, tune: None, bytes, card }),
            None => {
                eprintln!("brain: minicpmv not served (needs {} MiB on one card)", bytes >> 20);
                None
            }
        }
    }

    /// This resident again as each fine-tune stored beside the checkpoint.
    pub fn stored_fine_tunes(&self) -> Vec<MiniCpmVResident> {
        minicpmv::finetune::stored_tunes(std::path::Path::new(&self.dir))
            .into_iter()
            .map(|t| MiniCpmVResident { id: format!("{MODEL}:{}", t.label), dir: self.dir.clone(), tune: Some(t.dir), bytes: self.bytes, card: self.card })
            .collect()
    }

    /// The base and every fine-tune stored beside it, as the model ids they serve.
    pub fn family_from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Vec<MiniCpmVResident> {
        let Some(base) = Self::from_assembly(assembly, gpus, reserved) else { return Vec::new() };
        let mut family = base.stored_fine_tunes();
        family.insert(0, base);
        family
    }
}

impl ResidentModel for MiniCpmVResident {
    fn manifest(&self) -> Manifest {
        Manifest { model: self.id.clone(), ..minicpmv::caps::manifest_resident() }
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.clone(), self.dir.clone())
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
        Ok(Box::new(MiniCpmVInstance { session: minicpmv::caps::load_session(&key.config, self.tune.as_deref(), Some(self.card))? }))
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
        let r = MiniCpmVResident { id: MODEL.into(), dir: "/nonexistent".into(), tune: None, bytes: 1, card: 1 };
        let e = r.activate_multi(&r.instance_key("generate", &Invocation::new()), &[Device::Gpu(0)]).err().unwrap_or_default();
        assert!(e.contains("plan placed"), "{e}");
        let cost = r.estimate_multi(&r.instance_key("generate", &Invocation::new()));
        assert_eq!((cost.devices().collect::<Vec<_>>(), cost.on(Device::Gpu(1))), (vec![Device::Gpu(1)], 1));
    }

    #[test]
    fn a_stored_fine_tune_is_a_model_of_its_own() {
        let root = std::env::temp_dir().join(format!("brain-minicpmv-resident-tuned-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let dir = minicpmv::finetune::store_dir(&root, "acme:receipts:v2").unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(minicpmv::train::TUNE_MANIFEST), b"{}").unwrap();
        let base = MiniCpmVResident { id: MODEL.into(), dir: root.to_string_lossy().into_owned(), tune: None, bytes: 7, card: 1 };
        let tuned = base.stored_fine_tunes();
        assert_eq!(tuned.len(), 1);
        assert_eq!(tuned[0].manifest().model, "brain/minicpmv:acme:receipts:v2");
        assert_eq!(tuned[0].tune.as_deref(), Some(dir.as_path()));
        let key = |r: &MiniCpmVResident| r.instance_key("generate", &Invocation::new());
        assert_ne!(key(&tuned[0]), key(&base), "the base and its fine-tune are separate instances");
        assert_eq!((tuned[0].bytes, tuned[0].card), (base.bytes, base.card), "same weights, same card");
        std::fs::remove_dir_all(&root).ok();
    }
}
