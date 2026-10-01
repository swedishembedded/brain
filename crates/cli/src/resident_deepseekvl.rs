// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-VL behind the residency scheduler.
//!
//! The composite does not fit one 24 GB card: its towers hold about 8 GB
//! (SAM's global attention at 1024 pixels) and the fp16 decoder about 15 GB
//! before any KV cache. So it is a multi-device model: at registration
//! `deepseekvl::model::place` decides, once, which card holds the towers and
//! which the decoder, and how long a context the decoder's card leaves room
//! for (the checkpoint's own maximum when it fits). `estimate_multi` charges
//! exactly those bytes to exactly those cards, and `activate_multi` builds
//! each part inside its card's scope. All request work lives in
//! `deepseekvl::caps`; this file holds placement and budgeting only.

use capability::{ActionResult, Assembly, Invocation, Manifest, Progress};
use deepseekvl::caps::{Session, MODEL};
use deepseekvl::model::{Footprint, Placement};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

pub struct DeepseekVlResident {
    /// The model id this resident answers to: the base's, or the base's with
    /// a stored fine-tune's `:owner:name:tag`.
    id: String,
    dir: String,
    /// A fine-tune to serve (`BRAIN_DEEPSEEKVL_TUNED`: the directory
    /// `brain deepseekvl finetune` wrote), applied when the model loads.
    tuned: Option<std::path::PathBuf>,
    footprint: Footprint,
    placement: Placement,
}

impl DeepseekVlResident {
    /// Resolve the checkpoint and place it over `gpus` (`(index, total
    /// bytes)`), keeping `reserved` bytes free on each card.
    pub fn from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Option<DeepseekVlResident> {
        let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned())?;
        let placed = deepseekvl::model::footprint(std::path::Path::new(&dir)).and_then(|fp| {
            let cards: Vec<(u32, u64)> = gpus.iter().map(|&(i, total)| (i, total.saturating_sub(reserved))).collect();
            deepseekvl::model::place(&fp, &cards).map(|p| (fp, p))
        });
        match placed {
            Ok((footprint, placement)) => Some(DeepseekVlResident { id: MODEL.to_string(), dir, tuned: std::env::var_os("BRAIN_DEEPSEEKVL_TUNED").map(Into::into), footprint, placement }),
            Err(e) => {
                eprintln!("brain: deepseekvl not served ({e})");
                None
            }
        }
    }

    /// This resident again as each fine-tune stored beside the checkpoint,
    /// under the id `<base id>:<owner>:<name>:<tag>`, so a fine-tune is a model
    /// of its own next to the base, as a text adapter is.
    pub fn stored_fine_tunes(&self) -> Vec<DeepseekVlResident> {
        deepseekvl::tuned::scan(std::path::Path::new(&self.dir), deepseekvl::tuned::is_vl_fine_tune)
            .into_iter()
            .map(|t| DeepseekVlResident { id: format!("{MODEL}:{}", t.label), dir: self.dir.clone(), tuned: Some(t.dir), footprint: self.footprint, placement: self.placement })
            .collect()
    }

    /// The base and every fine-tune stored beside it, as the model ids they serve.
    pub fn family_from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Vec<DeepseekVlResident> {
        let Some(base) = Self::from_assembly(assembly, gpus, reserved) else { return Vec::new() };
        let mut family = base.stored_fine_tunes();
        family.insert(0, base);
        family
    }

    /// Bytes per device: the towers on theirs, the decoder with its KV cache
    /// on its, summed when they share one.
    fn cost(&self) -> Vec<(Device, u64)> {
        let (t, d) = (self.placement.tower, self.placement.decoder);
        let decoder = self.footprint.decoder_placed(&self.placement);
        if t == d {
            vec![(Device::Gpu(d), self.footprint.tower + decoder)]
        } else {
            vec![(Device::Gpu(t), self.footprint.tower), (Device::Gpu(d), decoder)]
        }
    }
}

impl ResidentModel for DeepseekVlResident {
    fn manifest(&self) -> Manifest {
        Manifest { model: self.id.clone(), ..deepseekvl::caps::manifest_resident() }
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.clone(), self.dir.clone())
    }

    /// Unusable by design: the model spans the devices `estimate_multi` names
    /// and is claimed through `claim_multi`; a single-device figure could
    /// charge only one of them.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{MODEL}: spans the towers' card and the decoder's, so it must be claimed via ResidencyManager::claim_multi"))
    }
}

impl MultiDeviceResidentModel for DeepseekVlResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        MultiDeviceCost::new(self.cost(), 0)
    }

    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let planned: Vec<Device> = self.cost().into_iter().map(|(d, _)| d).collect();
        if devices.len() != planned.len() || !devices.iter().all(|d| planned.contains(d)) {
            return Err(format!("{MODEL}: activate_multi got devices {devices:?} but the plan placed {planned:?}"));
        }
        Ok(Box::new(DeepseekVlInstance { session: deepseekvl::caps::load_session_tuned(&key.config, self.placement, self.tuned.as_deref())? }))
    }
}

struct DeepseekVlInstance {
    session: Session,
}

impl Instance for DeepseekVlInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("deepseekvl: unknown action '{action}'"));
        }
        self.session.generate(inv, progress)
    }

    /// Requests dispatched together decode as one batch on the shared KV pool.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        if action != "generate" {
            return invs.iter().map(|_| Err(format!("deepseekvl: unknown action '{action}'"))).collect();
        }
        self.session.generate_batch(invs, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident(placement: Placement) -> DeepseekVlResident {
        let cfg = qwen3::hf::decoder_config_as(r#"{"max_position_embeddings":16384,"model_type":"llama","num_hidden_layers":30,"vocab_size":102400}"#, "llama").unwrap();
        DeepseekVlResident { id: MODEL.into(), dir: "/tmp".into(), tuned: None, footprint: Footprint::of(&cfg, deepseekvl::model::HYBRID_TOWER_BYTES), placement }
    }

    #[test]
    fn each_part_is_charged_to_its_own_card() {
        let r = resident(Placement { tower: 0, decoder: 1, context: 4096, int8: false });
        let cost = r.cost();
        assert_eq!(cost.iter().map(|(d, _)| *d).collect::<Vec<_>>(), vec![Device::Gpu(0), Device::Gpu(1)]);
        assert_eq!(cost[1].1, r.footprint.decoder_at(4096), "the decoder carries its KV cache");
        let shared = resident(Placement { tower: 0, decoder: 0, context: 4096, int8: false }).cost();
        assert_eq!(shared, vec![(Device::Gpu(0), cost[0].1 + cost[1].1)], "one card is charged once, for both");
    }

    #[test]
    fn a_stored_fine_tune_is_a_model_of_its_own() {
        let root = std::env::temp_dir().join(format!("brain-vl-resident-tuned-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let dir = root.join("adapters/acme/puppies/v1");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(deepseekvl::train::ADAPTER_FILE), b"x").unwrap();
        let mut base = resident(Placement { tower: 0, decoder: 1, context: 4096, int8: false });
        base.dir = root.to_string_lossy().into_owned();
        let tuned = base.stored_fine_tunes();
        assert_eq!(tuned.len(), 1);
        assert_eq!(tuned[0].manifest().model, "brain/deepseekvl:acme:puppies:v1");
        assert_eq!(tuned[0].tuned.as_deref(), Some(dir.as_path()));
        assert_ne!(tuned[0].instance_key("generate", &Invocation::new()), base.instance_key("generate", &Invocation::new()), "the base and its fine-tune are separate instances");
        assert_eq!(tuned[0].cost(), base.cost(), "same weights, same bytes");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_unplanned_device_set_is_refused() {
        let r = resident(Placement { tower: 0, decoder: 1, context: 4096, int8: false });
        let e = r.activate_multi(&r.instance_key("generate", &Invocation::new()), &[Device::Gpu(0)]).err().unwrap_or_default();
        assert!(e.contains("the plan placed"), "{e}");
    }
}
