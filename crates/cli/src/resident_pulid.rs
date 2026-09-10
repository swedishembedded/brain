// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PuLID-conditioned FLUX.1 behind the residency scheduler.
//!
//! `activate` builds a [`pulid::caps::Session`] rooted at the five checkpoint
//! directories `pulid::caps::PulidProvider::from_env` requires; each
//! (variant, size) bundle - FLUX.1, ArcFace, BiSeNet, EVA-CLIP, IDFormer, the
//! `PulidAdapter` - is built lazily inside it. All of the work comes from
//! `pulid::caps`, so this file holds no second copy of param decoding, the
//! ID-conditioning composition, or the generation call.
//!
//! # No batching, same reasoning as `resident_flux1.rs`
//!
//! Every request is its own multi-step sample with an ID-conditioned DiT -
//! there is no batch axis a residency-level grouping could fill.

use capability::{ActionResult, Invocation, Manifest, Progress};
use pulid::caps::Session;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// PuLID-conditioned FLUX.1 behind the scheduler. Five directories, all
/// required: `BRAIN_FLUX1_DIR` (the backbone, same as `resident_flux1.rs`),
/// `BRAIN_PULID_DIR` (`pulid_flux_v0.9.1.safetensors` or its directory),
/// `BRAIN_ARCFACE_DIR` (same as `resident_arcface.rs`), `BRAIN_CLIP_DIR` (for
/// the EVA-CLIP-L/336 file, same as `resident_clip.rs`), `BRAIN_BISENET_DIR`
/// (a directory holding `parsing_bisenet.safetensors` - see
/// `pulid::caps::BISENET_FILE`'s doc for how to produce one).
pub struct PulidResident {
    flux1_root: String,
    pulid_root: String,
    arcface_root: String,
    clip_root: String,
    bisenet_root: String,
}

impl PulidResident {
    /// `None` unless every directory is set and the FLUX.1 root holds a
    /// released `transformer/` - registering a model whose every call would
    /// fail is worse than not serving it.
    pub fn from_env() -> Option<PulidResident> {
        let get = |k: &str| std::env::var(k).ok().filter(|p| !p.is_empty());
        let (flux1_root, pulid_root, arcface_root, clip_root, bisenet_root) = (
            get("BRAIN_FLUX1_DIR")?,
            get("BRAIN_PULID_DIR")?,
            get("BRAIN_ARCFACE_DIR")?,
            get("BRAIN_CLIP_DIR")?,
            get("BRAIN_BISENET_DIR")?,
        );
        Self::new(flux1_root, pulid_root, arcface_root, clip_root, bisenet_root)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(
        flux1_root: impl Into<String>,
        pulid_root: impl Into<String>,
        arcface_root: impl Into<String>,
        clip_root: impl Into<String>,
        bisenet_root: impl Into<String>,
    ) -> Option<PulidResident> {
        let flux1_root = flux1_root.into();
        if !std::path::Path::new(&flux1_root).join("transformer").exists() {
            eprintln!("brain: flux1-pulid not served ({flux1_root} holds no transformer/)");
            return None;
        }
        Some(PulidResident {
            flux1_root,
            pulid_root: pulid_root.into(),
            arcface_root: arcface_root.into(),
            clip_root: clip_root.into(),
            bisenet_root: bisenet_root.into(),
        })
    }
}

impl ResidentModel for PulidResident {
    fn manifest(&self) -> Manifest {
        pulid::caps::manifest()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // `(variant, precision, size)` - EXACTLY the tuple
        // `pulid::caps::Session` keys its own built bundles on. A single
        // opaque "default" key made the two disagree: one scheduler-priced
        // instance could hold an unbounded number of bundles built inside it,
        // none of which the budget ever saw. Same shape (and same reason) as
        // `resident_flux2.rs`'s key.
        let variant = inv.get_str("variant").unwrap_or_else(|| "dev".into());
        let precision = inv.get_str("precision").unwrap_or_else(|| "int8".into());
        let w = inv.get_i64("width").unwrap_or(1024);
        let h = inv.get_i64("height").unwrap_or(1024);
        InstanceKey::new(pulid::caps::MODEL, format!("{variant}:{precision}:{w}x{h}"))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // Read back out of "{variant}:{precision}:{w}x{h}" (see `instance_key`).
        // The precision is what decides servability, and a flat figure sized
        // for fp32 made EVERY request fail admission with
        // `ClaimError::TooLarge` on any card smaller than the fp32 DiT -
        // including the int8 build this action defaults to precisely because
        // it is the one that fits a 24 GiB card (`pulid::caps`'s `precision`
        // param doc).
        let int8 = key.config.contains(":int8:");
        // Measured at 512x512 int8: 21.9 GiB resident on the DiT's card. That
        // is the int8 FLUX.1-dev DiT plus PuLID's whole identity stack -
        // ArcFace, BiSeNet, EVA-CLIP-L, IDFormer and `PulidCa`, all of which
        // `pulid::caps::Bundle::load` deliberately pins to the DiT's own card
        // - plus the VAE and one sample's activations. The T5-XXL and CLIP-L
        // text encoders are NOT in this figure: `flux1::pipeline::plan_flux1`
        // places them itself, on their own device.
        // fp32 is that same stack over the DiT's own ~47.6 GB of f32 weights;
        // it genuinely does not fit a 24 GiB card, so it must keep being
        // refused up front rather than admitted and then OOM mid-generation.
        let base = if int8 { 22u64 << 30 } else { 54u64 << 30 };
        // The baseline above is a 512x512 sample (1024 image tokens). A larger
        // canvas only grows the activations, which scale with the joint token
        // count - `MAX_TXT_LEN` text tokens plus (w/16)*(h/16) image tokens -
        // over the DiT's 16 [n, hidden] + 3 [n, mlp] f32 working buffers
        // (`resident_flux2.rs::estimate` prices its own the same way).
        let (w, h) = size_from_key(&key.config);
        let n = flux1::pipeline::MAX_TXT_LEN as u64 + (w / 16) * (h / 16);
        let n_base = flux1::pipeline::MAX_TXT_LEN as u64 + (512 / 16) * (512 / 16);
        let per_token = (16 * 3072u64 + 3 * 12288u64) * 4;
        MemCost::new(base + n.saturating_sub(n_base) * per_token, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Every model `pulid::caps::Bundle::load` builds constructs its own
        // `Gpu` lazily (the same shape `resident_flux1.rs`/`resident_sdxl.rs`
        // document) - every `run` call, not just `activate`, is device-scoped.
        Ok(Box::new(PulidInstance {
            session: Session::new(
                self.flux1_root.clone(),
                self.pulid_root.clone(),
                self.arcface_root.clone(),
                self.clip_root.clone(),
                self.bisenet_root.clone(),
            ),
            device,
        }))
    }
}

/// `(w, h)` out of a `"{variant}:{precision}:{w}x{h}"` instance key, falling
/// back to the action's own 1024x1024 default when the key does not parse (an
/// unrecognisable key must not be priced at zero tokens).
fn size_from_key(config: &str) -> (u64, u64) {
    let parsed = config.rsplit(':').next().and_then(|wh| wh.split_once('x')).and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)));
    parsed.unwrap_or((1024, 1024))
}

struct PulidInstance {
    session: Session,
    device: Device,
}

impl Instance for PulidInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        crate::resident_llm::on_device(self.device, || self.session.run(action, inv))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A `PulidResident` over a directory that merely LOOKS like a released
    /// FLUX.1 checkpoint - `new` only probes for `transformer/`, and neither
    /// `instance_key` nor `estimate` reads a single weight.
    fn resident(tag: &str) -> (PulidResident, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("brain-cli-pulid-resident-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        let r = PulidResident::new(root.to_str().unwrap(), "/pulid", "/arcface", "/clip", "/bisenet").unwrap();
        (r, root)
    }

    /// The servability gate. `int8` is what this action DEFAULTS to, for the
    /// stated reason that it is the tier which fits a 24 GiB card; a flat,
    /// precision-blind estimate sized for the fp32 DiT therefore made every
    /// served request - including that default one - fail admission outright
    /// with `residency::ClaimError::TooLarge` (checked against each device's
    /// usable budget individually, never their sum), so the model was
    /// registered but structurally unreachable over every serving transport.
    #[test]
    fn the_default_int8_request_fits_one_24_gib_card() {
        let (r, root) = resident("fits");
        // What `brain serve --dbus` leaves usable on a 24 GiB card at its own
        // default `--reserve-gb 2`.
        let usable = (24u64 << 30) - (2u64 << 30);
        let inv = Invocation::new().set("prompt", json!("x")).set("precision", json!("int8")).set("width", json!(512)).set("height", json!(512));
        let cost = r.estimate(&r.instance_key("text2image", &inv));
        assert!(cost.vram <= usable, "int8 512x512 estimates {} MiB, more than a 24 GiB card's {} MiB usable budget", cost.vram >> 20, usable >> 20);

        // fp32 genuinely does not fit and must stay refused, not quietly
        // shrunk to make this test pass.
        let fp32 = Invocation::new().set("prompt", json!("x")).set("precision", json!("fp32")).set("width", json!(512)).set("height", json!(512));
        assert!(r.estimate(&r.instance_key("text2image", &fp32)).vram > usable, "fp32 must remain correctly unservable on a 24 GiB card");
        std::fs::remove_dir_all(&root).ok();
    }

    /// `pulid::caps::Session` caches one built bundle per
    /// `(variant, height, width, precision)`. The residency key must carry the
    /// same tuple, or the scheduler prices one instance while that instance
    /// silently accumulates several bundles' worth of weights.
    #[test]
    fn the_instance_key_carries_variant_precision_and_size() {
        let (r, root) = resident("key");
        let base = Invocation::new().set("prompt", json!("x")).set("precision", json!("int8")).set("width", json!(512)).set("height", json!(512));
        let k = r.instance_key("text2image", &base);
        assert_eq!(k.config, "dev:int8:512x512");

        for differing in [
            base.clone().set("precision", json!("fp32")),
            base.clone().set("width", json!(1024)),
            base.clone().set("variant", json!("schnell")),
        ] {
            assert_ne!(k.config, r.instance_key("text2image", &differing).config, "a differing bundle field must not share one instance");
        }

        // Repeat calls with the SAME bundle fields must share one instance -
        // this is what makes a batch of generations pay the weight load once.
        let same = base.clone().set("prompt", json!("a different prompt")).set("seed", json!(7));
        assert_eq!(k.config, r.instance_key("text2image", &same).config);
        std::fs::remove_dir_all(&root).ok();
    }
}
