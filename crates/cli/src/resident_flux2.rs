// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein behind the residency scheduler (`resident::build_executor`).
//!
//! A resident instance is a built [`flux2::Pipeline`] for one
//! `(variant, size, ref-tokens[, adapter])` fingerprint — DiT + text encoder +
//! VAE held together; dropping the instance frees the memory. `lora_train`
//! runs on a pipeline-less instance (the host f32 trainer builds and drops its
//! own encoders — see `flux2::finetune`). All action execution goes through
//! the shared helpers in `flux2::caps` — ONE implementation for the provider
//! and this adapter.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// FLUX.2 Klein resident model family, gated on the four weight env vars
/// (`BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}`).
pub struct Flux2Resident {
    paths: flux2::Paths,
}

impl Flux2Resident {
    /// `None` (not registered) unless all four `BRAIN_FLUX2_*` vars are set.
    pub fn from_env() -> Option<Flux2Resident> {
        flux2::Paths::from_env().ok().map(|paths| Flux2Resident { paths })
    }
}

/// Reference latent tokens declared by an invocation's input blobs, from their
/// `{w,h}` metadata after the /16 center-crop — used for the instance key
/// without decoding any pixels.
fn ref_tokens_from_meta(inv: &Invocation) -> u32 {
    ["image", "image0", "image1", "image2"]
        .iter()
        .filter_map(|n| inv.get_blob(n))
        .map(|b| {
            let dim = |k: &str| b.meta.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            (dim("h") / 16) * (dim("w") / 16)
        })
        .sum()
}

impl ResidentModel for Flux2Resident {
    fn manifest(&self) -> Manifest {
        flux2::caps::manifest()
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        let variant = inv.get_str("variant").unwrap_or_else(|| "klein-4b".into());
        if action == "lora_train" {
            return InstanceKey::new(flux2::caps::MODEL, format!("train:{variant}"));
        }
        let w = inv.get_i64("width").unwrap_or(512);
        let h = inv.get_i64("height").unwrap_or(512);
        let nref = ref_tokens_from_meta(inv);
        // "{variant}:{w}x{h}:{nref}" fixes the built graphs; a folded LoRA
        // changes the weights, so it is appended when present.
        let adapter = inv.get_str("adapter").filter(|s| !s.is_empty());
        let config = match adapter {
            Some(a) => format!("{variant}:{w}x{h}:{nref}:{a}"),
            None => format!("{variant}:{w}x{h}:{nref}"),
        };
        InstanceKey::new(flux2::caps::MODEL, config)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // Measured-single-run placeholders (fp32, 512×512 t2i on the dev box) —
        // TODO: re-measure via `brain perf run sweep --target flux2 …` once the
        // perf target lands and replace with per-variant curves.
        if key.config.starts_with("train:") {
            // The LoRA trainer is host f32 (model::hostmath) — RAM, not VRAM.
            return MemCost::new(0, 20u64 << 30);
        }
        let vram = if key.config.starts_with("klein-9b") || key.config.starts_with("base-9b") {
            // 9B fp32 DiT + Qwen3-8B encoder — roughly 2× the 4B build.
            36u64 << 30
        } else {
            // 4B fp32: ~15.5 GB DiT + encoder/VAE working set ≈ 18 GiB.
            18u64 << 30
        };
        MemCost::new(vram, 2u64 << 30)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Some(variant) = key.config.strip_prefix("train:") {
            flux2::caps::check_license(variant)?;
            // Training builds (and drops) its own encoders + host trainer per
            // run — no resident pipeline to hold.
            return Ok(Box::new(Flux2Instance { pipe: None, paths: clone_paths(&self.paths) }));
        }
        // "{variant}:{w}x{h}:{nref}[:{adapter}]" — adapter may contain ':'.
        let mut it = key.config.splitn(4, ':');
        let variant = it.next().ok_or("flux2: bad instance key")?;
        let wh = it.next().ok_or("flux2: bad instance key")?;
        let nref: u32 = it.next().and_then(|s| s.parse().ok()).ok_or("flux2: bad instance key")?;
        let adapter = it.next().filter(|s| !s.is_empty());
        let (w, h) = wh.split_once('x').ok_or("flux2: bad instance key")?;
        let (w, h): (u32, u32) = (w.parse().map_err(|_| "flux2: bad width")?, h.parse().map_err(|_| "flux2: bad height")?);
        flux2::caps::check_license(variant)?;
        let cfg = flux2::Flux2Config::from_name(variant)?;
        let n_gen = (h / 16) * (w / 16);
        // Place the pipeline on the assigned card (scoped registry selection;
        // the TE card is flux2's own BRAIN_FLUX2_TE_DEVICE and left as configured).
        let pipe = crate::resident_llm::on_device(device, || {
            flux2::Pipeline::build_adapted(&cfg, &self.paths, n_gen + nref, adapter)
        })??;
        Ok(Box::new(Flux2Instance { pipe: Some(pipe), paths: clone_paths(&self.paths) }))
    }
}

/// `flux2::Paths` derives no `Clone`; the fields are plain strings.
fn clone_paths(p: &flux2::Paths) -> flux2::Paths {
    flux2::Paths { dit: p.dit.clone(), vae: p.vae.clone(), te: p.te.clone(), tokenizer: p.tokenizer.clone() }
}

/// A resident FLUX.2 instance: `pipe` for generation keys, `None` for the
/// training key.
struct Flux2Instance {
    pipe: Option<flux2::Pipeline>,
    paths: flux2::Paths,
}

impl Instance for Flux2Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "text2image" | "edit" => {
                let pipe = self.pipe.as_ref().ok_or("flux2: generation on a training instance")?;
                let p = flux2::caps::gen_params_from(inv)?;
                let refs = flux2::caps::refs_from(inv, action == "edit")?;
                flux2::caps::generate_on(pipe, inv, &refs, &p.opts, progress)
            }
            "lora_train" => flux2::caps::train_action(&self.paths, inv, progress),
            other => Err(format!("flux2-klein: unknown action '{other}'")),
        }
    }

    /// Documented-sequential (serving-contract §3, explicit-reason path): a
    /// TRUE batched forward would run one joint MMDiT forward over N latents,
    /// but the pipeline's device graphs are built for a single joint sequence
    /// (txt+img slab layout) and per-request seeds/steps/CFG make the denoise
    /// trajectories diverge — batching the DiT forward is a separate change to
    /// `flux2::model`. The scheduler already groups same-key jobs into one
    /// `run_batch` call, so that follow-up only touches this method.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(Progress)) -> Vec<ActionResult> {
        invs.iter().map(|inv| self.run(action, inv, progress)).collect()
    }
}
