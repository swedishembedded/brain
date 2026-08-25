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

/// How many concurrent same-key generations one instance batches into a single
/// denoise loop (`BRAIN_FLUX2_MAX_BATCH`, default 4). Only the DiT activation
/// scratch scales with it — measured VRAM per sample determines
/// the point where latency stops paying for throughput.
/// The scheduler's own `Policy::max_batch` caps the group size on top of this.
pub fn max_batch() -> u32 {
    std::env::var("BRAIN_FLUX2_MAX_BATCH")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(4)
        .max(1)
}

/// Image + reference latent tokens from an instance key
/// (`"{variant}:{precision}:{w}x{h}:{nref}[:{adapter}]"`), for the memory
/// estimate. 0 if the key does not parse (an unknown key costs nothing extra).
fn tokens_from_key(config: &str) -> u64 {
    let mut it = config.splitn(5, ':');
    let (_, _) = (it.next(), it.next());
    let Some((w, h)) = it.next().and_then(|wh| wh.split_once('x')) else { return 0 };
    let (Ok(w), Ok(h)) = (w.parse::<u64>(), h.parse::<u64>()) else { return 0 };
    let nref: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (w / 16) * (h / 16) + nref
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
        let precision = inv.get_str("precision").unwrap_or_else(|| "fp32".into());
        let w = inv.get_i64("width").unwrap_or(512);
        let h = inv.get_i64("height").unwrap_or(512);
        let nref = ref_tokens_from_meta(inv);
        // "{variant}:{precision}:{w}x{h}:{nref}" fixes the built graphs; a
        // folded LoRA changes the weights, so it is appended when present.
        let adapter = inv.get_str("adapter").filter(|s| !s.is_empty());
        let config = match adapter {
            Some(a) => format!("{variant}:{precision}:{w}x{h}:{nref}:{a}"),
            None => format!("{variant}:{precision}:{w}x{h}:{nref}"),
        };
        InstanceKey::new(flux2::caps::MODEL, config)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // Measured-single-run placeholders (512×512 t2i on the dev box, two
        // P40s) — TODO: re-measure via `brain perf run sweep --target flux2 …`
        // once the perf target lands and replace with per-variant curves.
        if key.config.starts_with("train:") {
            // The LoRA trainer is host f32 (model::hostmath) — RAM, not VRAM.
            return MemCost::new(0, 20u64 << 30);
        }
        let nine_b = key.config.starts_with("klein-9b") || key.config.starts_with("base-9b");
        let int8 = key.config.contains(":int8:");
        let vram = match (nine_b, int8) {
            // 9B fp32 DiT + Qwen3-8B encoder - roughly twice the 4B build.
            (true, false) => 36u64 << 30,
            // int8 9B DiT ≈ 8.8 GiB + encoder — unmeasured, scaled from 4B.
            (true, true) => 16u64 << 30,
            // 4B fp32: ~15.5 GB DiT + encoder/VAE working set ≈ 18 GiB.
            (false, false) => 18u64 << 30,
            // 4B int8 DiT ≈ 3.9 GiB weights (~6 GiB resident with scratch/VAE;
            // the TE is placed separately via BRAIN_FLUX2_TE_DEVICE).
            (false, true) => 6u64 << 30,
        };
        // A batched instance holds one activation slab per batch slot; the
        // weights are shared. The DiT scratch is 16 [n, hidden] + 3 [n, mlp]
        // f32 buffers (+ a quarter of that again for the int8 packed
        // activations), n = txt_len + image/reference tokens — 472 MiB per slot
        // at 512² klein-4B, which the estimates above already include for slot
        // 0. Only the EXTRA slots are added here.
        let (hidden, mlp, txt_len) = if nine_b { (4096u64, 12288u64, 512u64) } else { (3072u64, 9216u64, 512u64) };
        let n_joint = txt_len + tokens_from_key(&key.config);
        let mut per_slot = n_joint * (16 * hidden + 3 * mlp) * 4;
        if int8 {
            per_slot += n_joint * (hidden + mlp); // packed int8 activations
        }
        MemCost::new(vram + per_slot * (max_batch() as u64 - 1), 2u64 << 30)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Some(variant) = key.config.strip_prefix("train:") {
            flux2::caps::check_license(variant)?;
            // Training builds (and drops) its own encoders + host trainer per
            // run — no resident pipeline to hold.
            return Ok(Box::new(Flux2Instance { pipe: None, paths: clone_paths(&self.paths) }));
        }
        // "{variant}:{precision}:{w}x{h}:{nref}[:{adapter}]" — adapter may
        // contain ':'.
        let mut it = key.config.splitn(5, ':');
        let variant = it.next().ok_or("flux2: bad instance key")?;
        let precision = flux2::Precision::from_name(it.next().ok_or("flux2: bad instance key")?)?;
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
            flux2::Pipeline::build_batched(&cfg, &self.paths, n_gen + nref, adapter, precision, max_batch())
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

    /// TRUE batched generation (serving-contract §3): the N same-key jobs the
    /// scheduler grouped share ONE denoise loop, each step of which is a single
    /// batched MMDiT forward over all their latents
    /// (`flux2::Flux2Model::forward_batch`, bit-identical to N single
    /// forwards — `crates/flux2/tests/batch_parity.rs`).
    ///
    /// Per-request **seed, steps, guidance and prompt** are honoured inside the
    /// batch: the instance key already fixes variant/precision/size/refs/adapter
    /// (so the weights and the slab layout are shared), and differing step
    /// counts simply put samples at different timesteps — free, because
    /// modulation is a per-sample condition group. CFG rides as a second sample.
    /// `inv.cancel` is polled per request per step; a cancelled request leaves
    /// the batch and the rest continue.
    ///
    /// `lora_train` has no batchable form (one host trainer, one dataset, one
    /// adapter out), so it stays the sequential loop.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        if action == "lora_train" || invs.len() < 2 {
            return invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect();
        }
        let Some(pipe) = self.pipe.as_ref() else {
            return invs.iter().map(|_| Err("flux2: generation on a training instance".to_string())).collect();
        };
        // Decode every request first; a request that fails validation reports
        // its own error and does not sink the batch.
        let mut reqs: Vec<Option<flux2::BatchRequest>> = Vec::with_capacity(invs.len());
        let mut out: Vec<ActionResult> = Vec::with_capacity(invs.len());
        for inv in invs {
            out.push(Err("not run".to_string()));
            reqs.push(match build_request(action, inv) {
                Ok(r) => Some(r),
                Err(e) => {
                    *out.last_mut().unwrap() = Err(e);
                    None
                }
            });
        }
        let live: Vec<usize> = reqs.iter().enumerate().filter_map(|(i, r)| r.is_some().then_some(i)).collect();
        if live.is_empty() {
            return out;
        }
        // `take`, not `clone` — a request's reference images are megabytes.
        let batch: Vec<flux2::BatchRequest> = live.iter().map(|&i| reqs[i].take().unwrap()).collect();
        // Denoising progress is batch-level (all samples step together); broadcast
        // each update to every job's sink, matching the prior fan-to-all behavior.
        let n = invs.len();
        let mut prog = |step: u32, total: u32, msg: &str| {
            for i in 0..n {
                progress(i, Progress::step(step, total, msg));
            }
        };
        let results = pipe.generate_batch(&batch, &mut prog);
        for (&i, r) in live.iter().zip(results) {
            out[i] = r.map(|(rgb, w, h)| flux2::caps::image_outcome(&rgb, w, h));
        }
        out
    }
}

/// One invocation → a [`flux2::BatchRequest`] (params + references + its cancel
/// token), through the same shared `flux2::caps` decoders the single-request
/// path uses — no second copy of the param contract.
fn build_request(action: &str, inv: &Invocation) -> Result<flux2::BatchRequest, String> {
    let p = flux2::caps::gen_params_from(inv)?;
    let refs = flux2::caps::refs_from(inv, action == "edit")?;
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    Ok(flux2::BatchRequest { prompt, refs, opts: p.opts, cancel: inv.cancel.clone() })
}
