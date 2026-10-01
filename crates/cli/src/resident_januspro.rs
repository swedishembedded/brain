// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro behind the residency scheduler.
//!
//! The two actions need two different builds of the checkpoint: `generate`
//! the understanding composite (SigLIP-L tower, bf16 decoder), `text2image`
//! the batched serving engine with the generation heads and the VQ-16
//! decoder. Each is its own instance key and each lives on ONE card, so the
//! scheduler swaps them by ordinary eviction (it never evicts a multi-device
//! resident to make room for another, which a two-card chat build would be).
//! Every context is what the build's card has room for, decided once at
//! registration from the roomiest card, never a fixed figure. All request
//! work lives in `januspro::caps`; this file holds placement and budgeting
//! only.

use capability::{ActionResult, Assembly, Invocation, Manifest, Progress};
use deepseekvl::model::{Footprint, Placement};
use januspro::caps::MODEL;
use januspro::t2i::TextToImage;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// A build's plan: its bytes on its card and its context.
#[derive(Clone, Copy, Debug)]
struct Plan {
    bytes: u64,
    context: u32,
}

pub struct JanusProResident {
    dir: String,
    /// A fine-tune of the understanding path to serve
    /// (`BRAIN_JANUSPRO_TUNED`: what `brain januspro finetune --mode
    /// understanding` wrote), applied when the chat build loads.
    tuned: Option<std::path::PathBuf>,
    understanding: Option<Plan>,
    generation: Option<Plan>,
}

impl JanusProResident {
    /// Resolve the checkpoint and plan both builds for the roomiest of
    /// `gpus` (`(index, free bytes)`), keeping `reserved` bytes free. A build
    /// that fits no card is left unplanned; its requests then fail cleanly.
    pub fn from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Option<JanusProResident> {
        let dir = assembly.roles.get("dir").map(|p| p.to_string_lossy().into_owned())?;
        let cfg = match januspro::model::decoder_config(std::path::Path::new(&dir)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("brain: januspro not served ({e})");
                return None;
            }
        };
        let Some(room) = gpus.iter().map(|&(_, free)| free.saturating_sub(reserved)).max() else {
            eprintln!("brain: januspro not served (no GPU)");
            return None;
        };
        let card = [(0, room)];
        let fp = Footprint::of(&cfg, januspro::model::SIGLIP_TOWER_BYTES);
        let understanding = deepseekvl::model::place(&fp, &card)
            .map(|p| Plan { bytes: fp.tower + fp.decoder_at(p.context), context: p.context })
            .map_err(|e| eprintln!("brain: januspro chat not served ({e})"))
            .ok();
        let gen_fp = Footprint::of(&cfg, 0);
        let generation = januspro::t2i::place(&gen_fp, 1, &card)
            .map(|(_, context)| Plan { bytes: gen_fp.decoder_at(2 * context) + januspro::t2i::GENERATION_EXTRA_BYTES, context })
            .map_err(|e| eprintln!("brain: januspro text2image not served ({e})"))
            .ok();
        if understanding.is_none() && generation.is_none() {
            return None;
        }
        Some(JanusProResident { dir, tuned: std::env::var_os("BRAIN_JANUSPRO_TUNED").map(Into::into), understanding, generation })
    }

    fn is_generation(key: &InstanceKey) -> bool {
        key.config.ends_with("|generation")
    }

    fn plan(&self, key: &InstanceKey) -> Option<Plan> {
        if Self::is_generation(key) {
            self.generation
        } else {
            self.understanding
        }
    }
}

impl ResidentModel for JanusProResident {
    fn manifest(&self) -> Manifest {
        januspro::caps::manifest_resident()
    }

    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        let build = if action == "text2image" { "generation" } else { "understanding" };
        InstanceKey::new(MODEL, format!("{}|{build}", self.dir))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        MemCost::new(self.plan(key).map_or(0, |p| p.bytes), 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let plan = self.plan(key).ok_or_else(|| format!("{MODEL}: this build fits none of the cards"))?;
        let Device::Gpu(card) = device else {
            return Err(format!("{MODEL}: assigned {device:?}, but both builds need a GPU"));
        };
        let (dir, _) = key.config.rsplit_once('|').ok_or("januspro: malformed instance key")?;
        if Self::is_generation(key) {
            Ok(Box::new(Generation { t2i: januspro::caps::load_t2i(dir, card, plan.context)? }))
        } else {
            let placement = Placement { tower: card, decoder: card, context: plan.context };
            Ok(Box::new(Understanding { session: januspro::caps::load_understanding(dir, placement, self.tuned.as_deref())? }))
        }
    }
}

struct Understanding {
    session: deepseekvl::caps::Session,
}

impl Instance for Understanding {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "generate" => self.session.generate(inv, progress),
            other => Err(format!("januspro: action '{other}' does not run on the understanding build")),
        }
    }
}

struct Generation {
    t2i: TextToImage,
}

impl Instance for Generation {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "text2image" => januspro::caps::text2image(&mut self.t2i, inv, progress),
            other => Err(format!("januspro: action '{other}' does not run on the generation build")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident(generation: Option<Plan>) -> JanusProResident {
        JanusProResident { dir: "/tmp".into(), tuned: None, understanding: Some(Plan { bytes: 18 << 30, context: 2048 }), generation }
    }

    #[test]
    fn each_build_is_its_own_single_card_instance() {
        let r = resident(Some(Plan { bytes: 20 << 30, context: 1024 }));
        let (chat, draw) = (r.instance_key("generate", &Invocation::new()), r.instance_key("text2image", &Invocation::new()));
        assert_ne!(chat, draw, "the two builds are separate instances");
        assert_eq!((r.estimate(&chat).vram, r.estimate(&draw).vram), (18 << 30, 20 << 30));
    }

    #[test]
    fn an_unplanned_build_fails_cleanly() {
        let r = resident(None);
        let key = r.instance_key("text2image", &Invocation::new());
        assert!(r.activate(&key, Device::Gpu(0)).err().unwrap_or_default().contains("fits none"));
    }

    #[test]
    fn an_assembly_without_its_dir_is_not_served() {
        let assembly = Assembly { id: "local/januspro".into(), arch: "januspro".into(), variant: None, roles: Default::default(), provenance: Vec::new() };
        assert!(JanusProResident::from_assembly(&assembly, &[(0, 24 << 30)], 2 << 30).is_none(), "no dir role, nothing to serve");
    }
}
