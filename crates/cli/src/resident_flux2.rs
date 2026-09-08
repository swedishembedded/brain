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
///
/// `id`/`variant` are BOUND at construction time, from the DiT weights
/// themselves ([`flux2::sniff_dit_size`]) - never re-derived per request from
/// an `Invocation`. This is what closes the compliance hole a per-request
/// "variant" param used to open: a real 9B checkpoint keyed and
/// license-gated as whatever the caller happened to claim (defaulting to
/// "klein-4b" when the caller claimed nothing at all).
pub struct Flux2Resident {
    /// What this instance's keys and manifest register under -
    /// `flux2::caps::MODEL` for the env-configured build, but a fetched
    /// checkpoint would register under its own reference (same reason
    /// `WanResident::from_paths` takes one).
    id: String,
    /// The full variant name bound to the real weights: sniffed SIZE
    /// (`4b`/`9b`, from [`flux2::sniff_dit_size`]) combined with an explicit
    /// klein/base FAMILY (`BRAIN_FLUX2_FAMILY`, default `"klein"`) - klein vs
    /// base is not recoverable from tensor shapes (see [`flux2::DitSize`]'s
    /// doc), so it can never be sniffed, only stated. `flux2::caps::
    /// check_license` reads this string's own size suffix, so binding it here
    /// (not per request) is what makes that gate trustworthy.
    variant: String,
    paths: flux2::Paths,
}

impl Flux2Resident {
    /// `None` (not registered) unless all four `BRAIN_FLUX2_*` vars are set
    /// AND the named DiT's own header sniffs cleanly - a misconfigured or
    /// unreadable DiT must not silently masquerade as "not registered".
    pub fn from_env() -> Option<Flux2Resident> {
        let paths = flux2::Paths::from_env().ok()?;
        match Flux2Resident::from_paths(flux2::caps::MODEL.to_string(), paths) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("brain: flux2-klein not served over the scheduler (BRAIN_FLUX2_DIT: {e})");
                None
            }
        }
    }

    /// Bind a resident to real, explicitly-named weights. `id` is the
    /// registration name (see the struct doc); `variant` is fixed HERE, from
    /// `paths.dit`'s own tensor shapes combined with `BRAIN_FLUX2_FAMILY`
    /// (default `"klein"`).
    pub fn from_paths(id: String, paths: flux2::Paths) -> Result<Flux2Resident, String> {
        let family = std::env::var("BRAIN_FLUX2_FAMILY").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "klein".to_string());
        if family != "klein" && family != "base" {
            return Err(format!("BRAIN_FLUX2_FAMILY must be 'klein' or 'base' (got {family:?})"));
        }
        let size = flux2::sniff_dit_size(&paths.dit)?;
        let variant = format!("{family}-{}", size.as_str());
        flux2::Flux2Config::from_name(&variant)?; // the combination must actually exist
        Ok(Flux2Resident { id, variant, paths })
    }
}

/// A request's `variant` param, if present, must name exactly the resident's
/// own bound identity - it can describe the run, never redirect it. Silently
/// honoring a different string would key/estimate correctly (both already
/// ignore it) but generate under a caller's false belief about which weights
/// ran; silently ignoring a wrong one would hide a caller's mistake instead
/// of surfacing it.
fn check_variant_matches(bound: &str, inv: &Invocation) -> Result<(), String> {
    match inv.get_str("variant") {
        Some(v) if v != bound => Err(format!(
            "flux2: request named variant '{v}', but this resident is bound to '{bound}' from its actual weights - the bound variant cannot be overridden per request"
        )),
        _ => Ok(()),
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
/// (`"{variant}:{precision}:{w}x{h}:{nref}[:{lora_scale}:{adapter}]"`), for the
/// memory estimate. Only the size field is read, so the optional adapter tail
/// does not affect it. 0 if the key does not parse (an unknown key costs
/// nothing extra).
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
        // The shared schema, under THIS instance's own id - see `id`'s doc.
        Manifest { model: self.id.clone(), ..flux2::caps::manifest() }
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        // `self.variant` ALWAYS - never `inv`'s "variant": the bound identity
        // is fixed at construction from the real weights (see the struct
        // doc), so a per-request variant can only be checked against it
        // (`check_variant_matches`, applied where the request can still be
        // refused - `Instance::run`/`run_batch`), never used to key.
        if action == "lora_train" {
            return InstanceKey::new(&self.id, format!("train:{}", self.variant));
        }
        let precision = inv.get_str("precision").unwrap_or_else(|| "fp32".into());
        let w = inv.get_i64("width").unwrap_or(512);
        let h = inv.get_i64("height").unwrap_or(512);
        let nref = ref_tokens_from_meta(inv);
        // "{variant}:{precision}:{w}x{h}:{nref}" fixes the built graphs; a
        // folded LoRA changes the weights, so its path AND strength are
        // appended when present (a different strength is different weights).
        // The path goes last because it is the only field that may contain ':'.
        let adapter = inv.get_str("adapter").filter(|s| !s.is_empty());
        let variant = &self.variant;
        let config = match adapter {
            Some(a) => {
                let sc = inv.get_f64("lora_scale").unwrap_or(1.0);
                format!("{variant}:{precision}:{w}x{h}:{nref}:{sc}:{a}")
            }
            None => format!("{variant}:{precision}:{w}x{h}:{nref}"),
        };
        InstanceKey::new(&self.id, config)
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
        if key.config.strip_prefix("train:").is_some() {
            // `self.variant` - the resident's own BOUND identity (sniffed
            // from the real weights at construction, see the struct doc),
            // never a string parsed back out of `key.config`.
            flux2::caps::check_license(&self.variant)?;
            // Training builds (and drops) its own encoders + host trainer per
            // run — no resident pipeline to hold.
            return Ok(Box::new(Flux2Instance { pipe: None, paths: clone_paths(&self.paths), variant: self.variant.clone() }));
        }
        // "{variant}:{precision}:{w}x{h}:{nref}[:{lora_scale}:{adapter}]" -
        // the adapter path is last because it may contain ':'. The variant
        // field is always `self.variant` (see `instance_key`), so parsing it
        // back out here is just recovering what this resident itself wrote.
        let mut it = key.config.splitn(6, ':');
        let variant = it.next().ok_or("flux2: bad instance key")?;
        let precision = flux2::Precision::from_name(it.next().ok_or("flux2: bad instance key")?)?;
        let wh = it.next().ok_or("flux2: bad instance key")?;
        let nref: u32 = it.next().and_then(|s| s.parse().ok()).ok_or("flux2: bad instance key")?;
        let lora_scale: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        // The instance key carries at most ONE adapter: `Pipeline` folds a
        // whole stack, but the residency key is a flat `:`-separated string
        // with no list encoding, so a served instance stays single-adapter
        // (stacking is reachable from `brain flux2 generate --adapter ...`).
        let adapters: Vec<flux2::AdapterSpec> = it
            .next()
            .filter(|s| !s.is_empty())
            .map(|path| flux2::AdapterSpec { path: path.to_string(), scale: lora_scale })
            .into_iter()
            .collect();
        let (w, h) = wh.split_once('x').ok_or("flux2: bad instance key")?;
        let (w, h): (u32, u32) = (w.parse().map_err(|_| "flux2: bad width")?, h.parse().map_err(|_| "flux2: bad height")?);
        flux2::caps::check_license(&self.variant)?;
        let cfg = flux2::Flux2Config::from_name(variant)?;
        let n_gen = (h / 16) * (w / 16);
        // Place the pipeline on the assigned card (scoped registry selection;
        // the TE card is flux2's own BRAIN_FLUX2_TE_DEVICE and left as configured).
        let pipe = crate::resident_llm::on_device(device, || {
            flux2::Pipeline::build_sized(&cfg, &self.paths, n_gen + nref, n_gen, &adapters, precision, max_batch())
        })??;
        Ok(Box::new(Flux2Instance { pipe: Some(pipe), paths: clone_paths(&self.paths), variant: self.variant.clone() }))
    }
}

/// `flux2::Paths` derives no `Clone`; the fields are plain strings.
fn clone_paths(p: &flux2::Paths) -> flux2::Paths {
    flux2::Paths { dit: p.dit.clone(), vae: p.vae.clone(), te: p.te.clone(), tokenizer: p.tokenizer.clone() }
}

/// A resident FLUX.2 instance: `pipe` for generation keys, `None` for the
/// training key. `variant` is the resident's own bound identity, carried down
/// so every request can be checked against it (`check_variant_matches`).
struct Flux2Instance {
    pipe: Option<flux2::Pipeline>,
    paths: flux2::Paths,
    variant: String,
}

impl Instance for Flux2Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        check_variant_matches(&self.variant, inv)?;
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
            reqs.push(match check_variant_matches(&self.variant, inv).and_then(|()| build_request(action, inv)) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A minimal single-file safetensors fixture carrying only the five
    /// tensors `flux2::sniff_dit_size` actually reads, at real `cfg`
    /// dimensions - `U8` (1 byte/element) content of all zeros, because the
    /// sniff never decodes a value, only the header's declared shapes.
    fn write_fake_dit(cfg: &flux2::Flux2Config, tag: &str) -> std::path::PathBuf {
        let entries = [
            ("img_in.weight".to_string(), vec![cfg.hidden, cfg.in_channels]),
            ("txt_in.weight".to_string(), vec![cfg.hidden, cfg.context_in_dim]),
            ("double_blocks.0.img_attn.norm.query_norm.scale".to_string(), vec![cfg.head_dim()]),
            (format!("double_blocks.{}.marker", cfg.depth_double - 1), vec![1]),
            (format!("single_blocks.{}.marker", cfg.depth_single - 1), vec![1]),
        ];
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (name, shape) in &entries {
            let n: usize = shape.iter().product();
            let start = blob.len();
            blob.resize(start + n, 0u8);
            header.insert(name.clone(), json!({"dtype": "U8", "shape": shape, "data_offsets": [start, blob.len()]}));
        }
        let mut hbytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        hbytes.resize(hbytes.len().next_multiple_of(8), b' ');
        let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&hbytes);
        file.extend_from_slice(&blob);
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("brain-cli-flux2-resident-{tag}-{}-{n}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        path
    }

    fn paths_with_dit(dit: String) -> flux2::Paths {
        flux2::Paths { dit, vae: "/vae".into(), te: "/te".into(), tokenizer: "/tok".into() }
    }

    /// The whole point of binding at construction: two residents pointed at
    /// DIFFERENT real checkpoints must key differently, and a per-request
    /// "variant" must never move that key - not even to the truth.
    #[test]
    fn the_resident_binds_its_variant_from_the_real_weights_not_a_default() {
        let p4 = write_fake_dit(&flux2::Flux2Config::klein_4b(), "4b");
        let p9 = write_fake_dit(&flux2::Flux2Config::klein_9b(), "9b");
        let r4 = Flux2Resident::from_paths(flux2::caps::MODEL.to_string(), paths_with_dit(p4.to_str().unwrap().to_string())).unwrap();
        let r9 = Flux2Resident::from_paths(flux2::caps::MODEL.to_string(), paths_with_dit(p9.to_str().unwrap().to_string())).unwrap();
        assert_eq!(r4.variant, "klein-4b", "a 4B-shaped checkpoint must bind to klein-4b, not default there by luck");
        assert_eq!(r9.variant, "klein-9b", "a 9B-shaped checkpoint must bind to klein-9b, never klein-4b's default");

        let base = Invocation::new().set("prompt", json!("a"));
        let k4 = r4.instance_key("text2image", &base);
        let k9 = r9.instance_key("text2image", &base);
        assert_ne!(k4.config, k9.config, "two different underlying checkpoints must key differently");
        assert!(k4.config.starts_with("klein-4b:"), "{}", k4.config);
        assert!(k9.config.starts_with("klein-9b:"), "{}", k9.config);

        // A caller cannot redirect the key by naming a variant on the
        // request - not even the "true" one for a DIFFERENT resident.
        let spoofed = r4.instance_key("text2image", &base.clone().set("variant", json!("klein-9b")));
        assert_eq!(k4.config, spoofed.config, "instance_key must never key on the per-request variant");

        std::fs::remove_file(&p4).ok();
        std::fs::remove_file(&p9).ok();
    }

    /// This is the enforcement half of the bound identity: a request whose
    /// own "variant" disagrees with what the resident is actually running
    /// must be refused, not silently honored (which would run the real
    /// weights under the caller's false belief) or silently ignored (which
    /// would hide the caller's mistake). Exercised through `Instance::run`
    /// on the `lora_train` key so no real GPU/weights are needed - the
    /// contradiction must be caught before anything else runs.
    #[test]
    fn a_request_naming_a_contradicting_variant_is_a_hard_error() {
        let p4 = write_fake_dit(&flux2::Flux2Config::klein_4b(), "contradiction");
        let r = Flux2Resident::from_paths(flux2::caps::MODEL.to_string(), paths_with_dit(p4.to_str().unwrap().to_string())).unwrap();
        assert_eq!(r.variant, "klein-4b");
        let key = r.instance_key("lora_train", &Invocation::new());
        let mut inst = r.activate(&key, Device::Cpu).unwrap();

        let bad = Invocation::new().set("variant", json!("klein-9b"));
        let err = inst.run("lora_train", &bad, &mut |_| {}).unwrap_err();
        assert!(err.contains("klein-9b") && err.contains("klein-4b"), "{err}");

        // Agreeing with the bound variant (or omitting it) must pass this
        // gate - the run then fails downstream for an unrelated reason (no
        // real dataset), never on the variant check. `seed` is a required
        // param (see `fix: seed is a required param, never a silent
        // default`) enforced by `ActionSpec::validate` before a served
        // invocation ever reaches `Instance::run` - this test calls `run`
        // directly, bypassing that gate, so it must supply `seed` itself to
        // keep exercising the downstream failure this test is actually about.
        let ok = Invocation::new().set("variant", json!("klein-4b")).set("data", json!("/nonexistent")).set("save", json!("/nonexistent/out")).set("seed", json!(0));
        let downstream_err = inst.run("lora_train", &ok, &mut |_| {}).unwrap_err();
        assert!(!downstream_err.contains("bound to"), "{downstream_err}");

        std::fs::remove_file(&p4).ok();
    }
}
