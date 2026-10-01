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

/// The most images the drawing build batches, memory permitting.
const MAX_IMAGES_PER_BATCH: u32 = 4;

/// A build's plan: its bytes on its card and its context.
#[derive(Clone, Copy, Debug)]
struct Plan {
    bytes: u64,
    context: u32,
    /// The chat build's decoder has int8 linears (see `deepseekvl::model::place`).
    int8: bool,
    /// Images the drawing build makes in one batch.
    parallel: u32,
}

pub struct JanusProResident {
    /// The model id this resident answers to: the base's, or the base's with
    /// a stored fine-tune's `:owner:name:tag`.
    id: String,
    /// The one action a stored fine-tune serves (a fine-tune trains one of
    /// the two builds); `None` for the base, which serves both.
    only: Option<&'static str>,
    dir: String,
    /// A fine-tune of the understanding path to serve
    /// (`BRAIN_JANUSPRO_TUNED`: what `brain januspro finetune --mode
    /// understanding` wrote), applied when the chat build loads.
    tuned: Option<std::path::PathBuf>,
    /// A generation fine-tune for the drawing build (`BRAIN_JANUSPRO_TUNED_GENERATION`).
    tuned_generation: Option<std::path::PathBuf>,
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
            .map(|p| Plan { bytes: fp.tower + fp.decoder_placed(&p), context: p.context, int8: p.int8, parallel: 1 })
            .map_err(|e| eprintln!("brain: januspro chat not served ({e})"))
            .ok();
        let gen_fp = Footprint::of(&cfg, 0);
        // As many images per batch as leave a useful context on the card.
        let generation = [MAX_IMAGES_PER_BATCH, 2, 1]
            .into_iter()
            .find_map(|parallel| januspro::t2i::place(&gen_fp, parallel, &card).ok().map(|(_, context)| Plan { bytes: gen_fp.decoder_at(2 * parallel * context) + januspro::t2i::GENERATION_EXTRA_BYTES, context, int8: false, parallel }));
        if generation.is_none() {
            eprintln!("brain: januspro text2image not served (no card holds the decoder, the heads and a useful context)");
        }
        if understanding.is_none() && generation.is_none() {
            return None;
        }
        Some(JanusProResident { id: MODEL.to_string(), only: None, dir, tuned: std::env::var_os("BRAIN_JANUSPRO_TUNED").map(Into::into), tuned_generation: std::env::var_os("BRAIN_JANUSPRO_TUNED_GENERATION").map(Into::into), understanding, generation })
    }

    /// The base and every fine-tune stored beside it, as the model ids they
    /// serve. A fine-tune holding `generation.safetensors` trains the drawing
    /// build and serves `text2image`; any other trained the chat build and
    /// serves `generate`.
    pub fn family_from_assembly(assembly: &Assembly, gpus: &[(u32, u64)], reserved: u64) -> Vec<JanusProResident> {
        let Some(base) = Self::from_assembly(assembly, gpus, reserved) else { return Vec::new() };
        let mut family = base.stored_fine_tunes();
        family.insert(0, base);
        family
    }

    fn stored_fine_tunes(&self) -> Vec<JanusProResident> {
        deepseekvl::tuned::scan(std::path::Path::new(&self.dir), deepseekvl::tuned::is_vl_fine_tune)
            .into_iter()
            .map(|t| {
                let drawing = t.dir.join(januspro::train::GENERATION_FILE).is_file();
                JanusProResident {
                    id: format!("{MODEL}:{}", t.label),
                    only: Some(if drawing { "text2image" } else { "generate" }),
                    dir: self.dir.clone(),
                    tuned: (!drawing).then(|| t.dir.clone()),
                    tuned_generation: drawing.then_some(t.dir),
                    understanding: self.understanding,
                    generation: self.generation,
                }
            })
            .collect()
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
        let mut manifest = Manifest { model: self.id.clone(), ..januspro::caps::manifest_resident() };
        if let Some(only) = self.only {
            manifest.actions.retain(|a| a.name == only);
        }
        manifest
    }

    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        let build = if action == "text2image" { "generation" } else { "understanding" };
        InstanceKey::new(self.id.clone(), format!("{}|{build}", self.dir))
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
            Ok(Box::new(Generation { t2i: januspro::caps::load_t2i(dir, card, plan.context, plan.parallel, self.tuned_generation.as_deref())? }))
        } else {
            let placement = Placement { tower: card, decoder: card, context: plan.context, int8: plan.int8 };
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

    /// Requests dispatched together decode as one batch on the shared KV pool.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        match action {
            "generate" => self.session.generate_batch(invs, progress),
            other => invs.iter().map(|_| Err(format!("januspro: action '{other}' does not run on the understanding build"))).collect(),
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

    /// Drawings dispatched together share the decoder's batch, up to the
    /// build's image count; further ones follow in the next batch.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        match action {
            "text2image" => januspro::caps::text2image_batch(&mut self.t2i, invs, progress),
            other => invs.iter().map(|_| Err(format!("januspro: action '{other}' does not run on the generation build"))).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident(generation: Option<Plan>) -> JanusProResident {
        JanusProResident { id: MODEL.into(), only: None, dir: "/tmp".into(), tuned: None, tuned_generation: None, understanding: Some(Plan { bytes: 18 << 30, context: 2048, int8: false, parallel: 1 }), generation }
    }

    #[test]
    fn each_build_is_its_own_single_card_instance() {
        let r = resident(Some(Plan { bytes: 20 << 30, context: 1024, int8: false, parallel: 1 }));
        let (chat, draw) = (r.instance_key("generate", &Invocation::new()), r.instance_key("text2image", &Invocation::new()));
        assert_ne!(chat, draw, "the two builds are separate instances");
        assert_eq!((r.estimate(&chat).vram, r.estimate(&draw).vram), (18 << 30, 20 << 30));
    }

    /// A fine-tune stored beside the checkpoint is a model of its own that
    /// serves the one action it trained, with the id of the base plus its label.
    #[test]
    fn stored_fine_tunes_each_serve_their_own_action() {
        let root = std::env::temp_dir().join(format!("brain-janus-resident-tuned-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        for (rel, files) in [("adapters/acme/chat/v1", vec![deepseekvl::train::ALIGNER_FILE]), ("adapters/acme/draw/v1", vec![deepseekvl::train::ADAPTER_FILE, januspro::train::GENERATION_FILE])] {
            std::fs::create_dir_all(root.join(rel)).unwrap();
            for f in files {
                std::fs::write(root.join(rel).join(f), b"x").unwrap();
            }
        }
        let mut base = resident(Some(Plan { bytes: 20 << 30, context: 1024, int8: false, parallel: 1 }));
        base.dir = root.to_string_lossy().into_owned();
        let family: Vec<(String, Vec<String>)> = base.stored_fine_tunes().iter().map(|r| (r.manifest().model, r.manifest().actions.into_iter().map(|a| a.name).collect())).collect();
        assert_eq!(family, [("brain/januspro:acme:chat:v1".to_string(), vec!["generate".to_string()]), ("brain/januspro:acme:draw:v1".to_string(), vec!["text2image".to_string()])]);
        assert_eq!(base.manifest().actions.len(), 2, "the base serves both");
        std::fs::remove_dir_all(&root).ok();
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
