// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end FLUX.2 Klein: prompt → text conditioning → 4-step (klein) or
//! CFG (base) rectified-flow Euler → FLUX.2 VAE decode; image editing via
//! reference-image token concatenation (RoPE t-offsets 10·(i+1)).
//!
//! Weight locations come from env only (`Paths::from_env`) — never baked-in
//! paths. The text encoder runs the parity-proven masked-pad path
//! (`Qwen::encode_hiddens_padded`, layers 9/18/27 concatenated).

use crate::config::Flux2Config;
use crate::model::{position_ids, Flux2Model};
use data::Tokenizer;

/// Qwen3 hidden-state taps concatenated per token (also used by
/// [`crate::finetune`]'s standalone caption encoder).
pub const TAP_LAYERS: [usize; 3] = [9, 18, 27];
/// Right-pad token for the masked-pad text-encoder path.
pub const PAD_TOKEN: u32 = 151643;

/// Weight locations, env-only:
/// `BRAIN_FLUX2_DIT` (diffusers `transformer/` dir, BFL single-file
/// safetensors, or BF16 GGUF), `BRAIN_FLUX2_VAE` (diffusers `vae/` dir or
/// file), `BRAIN_FLUX2_TE` (HF text-encoder dir), `BRAIN_FLUX2_TOKENIZER`
/// (`tokenizer.json`).
pub struct Paths {
    pub dit: String,
    pub vae: String,
    pub te: String,
    pub tokenizer: String,
}

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        let get = |k: &str| std::env::var(k).map_err(|_| format!("{k} not set"));
        Ok(Paths {
            dit: get("BRAIN_FLUX2_DIT")?,
            vae: get("BRAIN_FLUX2_VAE")?,
            te: get("BRAIN_FLUX2_TE")?,
            tokenizer: get("BRAIN_FLUX2_TOKENIZER")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct GenOpts {
    pub width: u32,
    pub height: u32,
    /// Image-to-image init strength in `(0, 1]`. `None` (or 1.0) starts the
    /// denoise from pure noise — the reference images then only *condition*
    /// via their tokens, so the result keeps the composition but is a fresh
    /// generation (see `docs/models/flux2/status.md`: this is why a
    /// reference-only "colorize" reinterprets the scene). With `Some(s)` the
    /// first reference is VAE-encoded and the trajectory starts partway down
    /// the schedule from `x_σ = (1−σ)·x₀ + σ·ε` — the rectified-flow forward
    /// process — so structure is anchored to the source. Small `s` = faithful.
    pub strength: Option<f32>,
    /// None → the variant default (4 distilled / 50 base).
    pub steps: Option<u32>,
    /// CFG scale — only meaningful for the undistilled base variants.
    pub guidance: f32,
    pub seed: u64,
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts { width: 1024, height: 1024, strength: None, steps: None, guidance: 4.0, seed: 0 }
    }
}

/// Read DiT weights from a diffusers `transformer/` dir, a BFL single-file
/// safetensors, or a BF16 GGUF, onto the canonical BFL names. Public so
/// [`crate::finetune`] loads the frozen base through the same importer.
pub fn read_dit_tensors(path: &str, cfg: &Flux2Config) -> Result<crate::Tensors, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("{path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors under {path}"));
        }
        let mut ts = Vec::new();
        for f in files {
            ts.extend(checkpoint::safetensors::read(f.to_str().unwrap())?);
        }
        crate::import_diffusers(ts, cfg)
    } else if p.extension().is_some_and(|x| x == "gguf") {
        crate::import_bfl(checkpoint::gguf::read(path)?, cfg)
    } else {
        let ts = checkpoint::safetensors::read(path)?;
        // single-file releases use BFL names; a consolidated diffusers dump
        // would carry `transformer_blocks.` — probe and route
        if ts.iter().any(|t| t.name.starts_with("transformer_blocks.")) {
            crate::import_diffusers(ts, cfg)
        } else {
            crate::import_bfl(ts, cfg)
        }
    }
}

/// A ready-to-generate model: DiT + VAE + text encoder held together.
pub struct Pipeline {
    pub cfg: Flux2Config,
    model: Flux2Model,
    tok: data::qwen_tokenizer::QwenBpe,
    te: qwen::Qwen,
    vae_cfg: vae::VaeConfig,
    vae_tensors: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
}

impl Pipeline {
    /// Build for a maximum joint sequence (txt + generated + reference
    /// tokens). `n_img_max` in latent tokens, e.g. 4096 for 1024×1024.
    pub fn build(cfg: &Flux2Config, paths: &Paths, n_img_max: u32) -> Result<Pipeline, String> {
        Pipeline::build_adapted(cfg, paths, n_img_max, None)
    }

    /// [`Pipeline::build`] with an optional trained LoRA adapter
    /// ([`crate::finetune`] output) folded into the DiT tensors before the
    /// model is built — a plain generation run then produces
    /// adapter-conditioned images with no model change.
    pub fn build_adapted(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&str>) -> Result<Pipeline, String> {
        Pipeline::build_with(cfg, paths, n_img_max, adapter, crate::Precision::F32)
    }

    /// [`Pipeline::build_adapted`] with a DiT numeric tier: `Precision::Int8`
    /// builds the DP4A DiT (~3.9 GiB of weights instead of ~15.5 GiB — DiT +
    /// int8 TE fit ONE 24 GB card). A LoRA adapter (if any) is folded into the
    /// f32 tensors BEFORE quantization, so adapters work at either tier.
    pub fn build_with(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&str>, precision: crate::Precision) -> Result<Pipeline, String> {
        Pipeline::build_batched(cfg, paths, n_img_max, adapter, precision, 1)
    }

    /// [`Pipeline::build_with`] sized for up to `max_batch` concurrent
    /// generations sharing one denoise loop ([`Pipeline::generate_batch`]).
    /// Only the DiT activation scratch grows (~0.5 GiB per extra sample at
    /// 512² klein-4B); the text encoder and VAE stay single-stream.
    pub fn build_batched(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&str>, precision: crate::Precision, max_batch: u32) -> Result<Pipeline, String> {
        let mut dit_ts = read_dit_tensors(&paths.dit, cfg)?;
        if let Some(ap) = adapter {
            // The adapter's tensor shapes depend only on the architecture, not
            // the latent grid — any (lh, lw) loads it.
            let tcfg = crate::modelgrad::Cfg::from_flux2(cfg, 1, 1);
            let ad = crate::lora::load_adapter(ap, &tcfg)?;
            ad.fold_into_tensors(&mut dit_ts)?;
        }
        let gpu = gpu_core::Gpu::new(crate::model::KERNELS);
        let model = Flux2Model::new_batched(cfg, &dit_ts, gpu, cfg.txt_len as u32 + n_img_max, max_batch.max(1), precision);
        drop(dit_ts);

        let tok = data::qwen_tokenizer::QwenBpe::from_file(&paths.tokenizer)?;
        let te_cfg = if cfg.context_in_dim == 12288 {
            qwen::QwenConfig::qwen3_8b()
        } else {
            qwen::QwenConfig::qwen3_4b()
        };
        let te_ts = checkpoint::safetensors::read_model_dir(std::path::Path::new(&paths.te))?;
        let init = qwen::import::brain_init_from_hf(te_ts, &te_cfg)?;
        // TE placement: default = ambient device; `BRAIN_FLUX2_TE_DEVICE=gpu<i>`
        // builds a truncated fp32 shard on that card (layers 0..=deepest tap —
        // res[27] needs no more; drops 9 layers + the head, ~12 GiB resident,
        // so the DiT can own the other card whole). A `:i8` suffix
        // (`gpu<i>:i8`) uses the int8 (DP4A) shard instead (~4× smaller —
        // truncated TE ~4.4 GiB resident, so int8 DiT + int8 TE share ONE
        // card). The masked-pad kmask path is shared by both shard graphs, so
        // parity is unchanged (int8 is the lossy tier, gated in its own test).
        let deepest = *TAP_LAYERS.iter().max().unwrap();
        // `BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8]` is user input, parsed to a
        // canonical card index at this edge; the shard's gpu_index is what
        // places the build (device registry) — later device creation (VAE)
        // stays on the ambient card beside the DiT.
        let te = match std::env::var("BRAIN_FLUX2_TE_DEVICE").ok().as_deref() {
            Some(s) if s.starts_with("gpu") => {
                let (idx_s, te_i8) = match s[3..].strip_suffix(":i8") {
                    Some(p) => (p, true),
                    None => (&s[3..], false),
                };
                let idx: usize = idx_s.parse().map_err(|_| format!("bad BRAIN_FLUX2_TE_DEVICE {s} (gpu<i>[:i8])"))?;
                let shard = qwen::Shard { start: 0, end: deepest, embed: true, head: false, gpu_index: idx };
                if te_i8 {
                    qwen::Qwen::new_shard_i8(te_cfg, 1, cfg.txt_len as u32, &init, shard)
                } else {
                    qwen::Qwen::new_shard(te_cfg, 1, cfg.txt_len as u32, &init, false, shard)
                }
            }
            _ => qwen::Qwen::new(te_cfg, 1, cfg.txt_len as u32, &init),
        };

        let vp = std::path::Path::new(&paths.vae);
        let (vae_file, vae_json) = if vp.is_dir() {
            (
                vp.join("diffusion_pytorch_model.safetensors"),
                std::fs::read_to_string(vp.join("config.json")).ok(),
            )
        } else {
            (vp.to_path_buf(), None)
        };
        let vae_cfg = match vae_json {
            Some(j) => vae::VaeConfig::from_json(
                &serde_json::from_str(&j).map_err(|e| e.to_string())?,
            ),
            None => vae::VaeConfig::flux2(),
        };
        let vae_ts = checkpoint::safetensors::read(vae_file.to_str().unwrap())?;
        let mut map = std::collections::HashMap::new();
        let (mut bn_mean, mut bn_var) = (Vec::new(), Vec::new());
        for t in vae_ts {
            if t.name == "bn.running_mean" {
                bn_mean = t.data.clone();
            }
            if t.name == "bn.running_var" {
                bn_var = t.data.clone();
            }
            map.insert(t.name, (t.shape, t.data));
        }
        if bn_mean.is_empty() || bn_var.is_empty() {
            return Err("vae checkpoint missing bn.running_{mean,var}".into());
        }

        Ok(Pipeline { cfg: cfg.clone(), model, tok, te, vae_cfg, vae_tensors: map, bn_mean, bn_var })
    }

    /// Prompt → `[txt_len, context_in_dim]` conditioning (masked-pad,
    /// layers 9/18/27 concatenated per token).
    pub fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
        let templated = self.tok.apply_chat_template_no_think(&[("user", prompt)]);
        let mut ids = self.tok.encode(&templated);
        ids.truncate(self.cfg.txt_len);
        let content = ids.len();
        ids.resize(self.cfg.txt_len, PAD_TOKEN);
        let taps = self.te.encode_hiddens_padded(&ids, content, &TAP_LAYERS);
        let d = taps[0].len() / self.cfg.txt_len;
        let mut ctx = Vec::with_capacity(self.cfg.txt_len * 3 * d);
        for row in 0..self.cfg.txt_len {
            for tap in &taps {
                ctx.extend_from_slice(&tap[row * d..(row + 1) * d]);
            }
        }
        ctx
    }

    /// VAE-encode an RGB image (`[-1,1]` CHW) to packed+normalized latent
    /// tokens `[lh*lw, 128]` (row-major, matching `position_ids`).
    pub fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
        let enc = vae::VaeEncoder::from_diffusers(self.vae_cfg.clone(), &self.vae_tensors, h, w, None);
        let (lh8, lw8) = ((h / 8) as usize, (w / 8) as usize);
        let mean = enc.encode_mean(chw, lh8 as u32, lw8 as u32);
        let eps = self.vae_cfg.batch_norm_eps;
        let packed = vae::latent::pack(&mean, 32, lh8, lw8, &self.bn_mean, &self.bn_var, eps);
        // [128, lh, lw] -> tokens [lh*lw, 128]
        let (lh, lw) = (lh8 / 2, lw8 / 2);
        let mut tokens = vec![0.0f32; lh * lw * 128];
        for c in 0..128 {
            for y in 0..lh {
                for x in 0..lw {
                    tokens[(y * lw + x) * 128 + c] = packed[(c * lh + y) * lw + x];
                }
            }
        }
        Ok(tokens)
    }

    /// Latent tokens `[lh*lw, 128]` → RGB u8 HWC.
    pub fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
        // tokens -> [128, lh, lw]
        let mut packed = vec![0.0f32; 128 * lh * lw];
        for c in 0..128 {
            for y in 0..lh {
                for x in 0..lw {
                    packed[(c * lh + y) * lw + x] = tokens[(y * lw + x) * 128 + c];
                }
            }
        }
        let eps = self.vae_cfg.batch_norm_eps;
        let unpacked = vae::latent::unpack(&packed, 32, lh * 2, lw * 2, &self.bn_mean, &self.bn_var, eps);
        let (h, w) = ((lh * 16) as u32, (lw * 16) as u32);
        let dec = vae::VaeDecoder::from_diffusers(self.vae_cfg.clone(), &self.vae_tensors, (lh * 2) as u32, (lw * 2) as u32, None);
        let chw = dec.decode(&unpacked);
        // clamp FIRST, then rescale (reference order — reversed produces artifacts)
        let n = (h * w) as usize;
        let mut out = vec![0u8; n * 3];
        for c in 0..3 {
            for i in 0..n {
                let v = chw[c * n + i].clamp(-1.0, 1.0);
                out[i * 3 + c] = (127.5 * (v + 1.0)) as u8;
            }
        }
        Ok(out)
    }

    /// The largest batch [`Pipeline::generate_batch`] can put in one DiT
    /// forward (what the model's scratch was sized for at build time).
    pub fn max_batch(&self) -> u32 {
        self.model.max_batch()
    }

    /// Text-to-image (optionally with reference images for editing).
    /// `refs`: RGB `[-1,1]` CHW images, each with its (h, w) — pre-cropped to
    /// multiples of 16 (see [`ref_from_hwc`]). Returns (rgb8 HWC, width,
    /// height). `cancel` is polled once per denoise step (multi-minute CPU
    /// runs must be abortable); a `Default` token never fires.
    ///
    /// This is [`Pipeline::generate_batch`] with one request — the two share
    /// one denoise loop, so there is no second sampling implementation to
    /// drift.
    pub fn generate(
        &self,
        prompt: &str,
        refs: &[(Vec<f32>, u32, u32)],
        o: &GenOpts,
        cancel: &capability::CancelToken,
        mut progress: impl FnMut(u32, u32, &str),
    ) -> BatchOutcome {
        let req = BatchRequest { prompt: prompt.to_string(), refs: refs.to_vec(), opts: o.clone(), cancel: cancel.clone() };
        self.generate_batch(std::slice::from_ref(&req), &mut progress)
            .pop()
            .expect("one request in, one result out")
    }

    /// Generate `reqs.len()` images through ONE denoise loop: at every step the
    /// still-running requests are packed into a single batched DiT forward
    /// ([`Flux2Model::forward_batch`]).
    ///
    /// Per-request **seed, steps, guidance and prompt** are all honoured:
    ///
    /// * the seed only picks the initial latent, which is per-sample anyway;
    /// * different `steps` mean different sigma schedules, so at loop index `i`
    ///   two samples sit at *different timesteps* — which costs nothing because
    ///   modulation is a per-sample condition group. A request that runs out of
    ///   steps simply leaves the batch, which shrinks for the remainder;
    /// * CFG (undistilled `base` variants) enters as a **second slot** in the
    ///   same batch — the conditional and unconditional evaluations of one
    ///   request are two samples at the same timestep with different `ctx`,
    ///   which used to be two sequential forwards;
    /// * `cancel` is polled per request per step; a cancelled request leaves
    ///   the batch immediately with `Err("cancelled")` and the others continue.
    ///
    /// Requests whose **position ids** differ (a different reference-image
    /// layout at the same total token count) cannot share a slab, so they are
    /// partitioned into id-groups and the groups run one after another. The
    /// text encoder and the VAE stay per request (they are separate models with
    /// their own single-sequence graphs) — only the DiT, which is the whole
    /// denoise cost, batches.
    ///
    /// Results are returned in request order.
    pub fn generate_batch(
        &self,
        reqs: &[BatchRequest],
        progress: &mut dyn FnMut(u32, u32, &str),
    ) -> Vec<BatchOutcome> {
        let mut out: Vec<BatchOutcome> = (0..reqs.len()).map(|_| Err("not run".to_string())).collect();
        // Partition by position ids: one slab layout per group.
        let mut groups: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
        for (i, r) in reqs.iter().enumerate() {
            match self.plan(r) {
                Err(e) => out[i] = Err(e),
                Ok(ids) => match groups.iter_mut().find(|(g, _)| *g == ids) {
                    Some((_, v)) => v.push(i),
                    None => groups.push((ids, vec![i])),
                },
            }
        }
        for (ids, members) in groups {
            self.denoise_group(reqs, &ids, &members, &mut out, progress);
        }
        out
    }

    /// Validate one request and return its joint position ids (the key that
    /// decides which requests can share a batched forward).
    fn plan(&self, r: &BatchRequest) -> Result<Vec<u32>, String> {
        let o = &r.opts;
        if !o.width.is_multiple_of(16) || !o.height.is_multiple_of(16) {
            return Err(format!("width/height must be multiples of 16 (got {}×{})", o.width, o.height));
        }
        let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
        // Keep in step with the token builder: under `strength` the first
        // reference is consumed as the init latent, so it contributes no
        // reference tokens and therefore no reference position ids.
        let ref_skip = if o.strength.is_some_and(|st| st < 1.0) { 1 } else { 0 };
        let ref_dims: Vec<(usize, usize)> = r
            .refs
            .iter()
            .skip(ref_skip)
            .map(|(_, rh, rw)| ((rh / 16) as usize, (rw / 16) as usize))
            .collect();
        Ok(position_ids(self.cfg.txt_len, lh, lw, &ref_dims))
    }

    /// One id-group's shared denoise loop.
    fn denoise_group(
        &self,
        reqs: &[BatchRequest],
        ids: &[u32],
        members: &[usize],
        out: &mut [BatchOutcome],
        progress: &mut dyn FnMut(u32, u32, &str),
    ) {
        let cfg = &self.cfg;
        // Per-member state; a member that fails to encode drops out here.
        struct Lane {
            idx: usize,
            lh: usize,
            lw: usize,
            n_gen: usize,
            steps: usize,
            guidance: f32,
            ctx: Vec<f32>,
            ctx_uncond: Option<Vec<f32>>,
            ref_tokens: Vec<f32>,
            sigmas: Vec<f32>,
            lat: Vec<f32>,
            /// First schedule index this lane runs; > 0 for img2img inits.
            start: usize,
        }
        let max_steps_hint = members.iter().map(|&i| reqs[i].steps_for(cfg.distilled)).max().unwrap_or(0) as u32;
        let mut lanes: Vec<Lane> = Vec::new();
        for &i in members {
            let r = &reqs[i];
            let o = &r.opts;
            let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
            let n_gen = lh * lw;
            let steps = r.steps_for(cfg.distilled);
            progress(0, max_steps_hint + 2, "encoding prompt");
            let ctx = self.encode_prompt(&r.prompt);
            let cf = !cfg.distilled && o.guidance > 1.0;
            let ctx_uncond = if cf { Some(self.encode_prompt("")) } else { None };
            let mut ref_tokens: Vec<f32> = Vec::new();
            let mut failed = None;
            // With `strength`, the first reference IS the init latent — passing
            // it again as conditioning tokens would double-anchor it (and its
            // greyscale evidence), which is not what img2img means. Standard
            // img2img: the init image is the conditioning. Extra references
            // (2nd onward) still ride along as edit context.
            let ref_skip = if o.strength.is_some_and(|s| s < 1.0) { 1 } else { 0 };
            for (chw, rh, rw) in r.refs.iter().skip(ref_skip) {
                progress(0, max_steps_hint + 2, "encoding reference");
                match self.encode_image(chw, *rh, *rw) {
                    Ok(t) => ref_tokens.extend(t),
                    Err(e) => failed = Some(e),
                }
            }
            if let Some(e) = failed {
                out[i] = Err(e);
                continue;
            }
            let sigmas = diffusion::scheduler::klein_sigmas(steps, n_gen);
            let noise = model::hostmath::randn(n_gen * cfg.in_channels, o.seed);
            // img2img: start partway down the schedule from the source latent.
            // `x_σ = (1−σ)·x₀ + σ·ε` is the same forward process the trainer
            // uses (`modelgrad::make_flow_batch`), so the model sees exactly
            // the distribution it was trained on at that σ.
            let (lat, start, sigmas) = match o.strength {
                Some(st) if st < 1.0 => {
                    let st = st.clamp(1e-3, 1.0);
                    // Do NOT slice the distilled schedule: `klein_sigmas` is
                    // shifted so hard for few-step sampling that its lowest
                    // non-zero entry is 0.56 at 8 steps (0.75 at 4) — there is
                    // no low-noise entry point to start an img2img from, and
                    // starting at 0.84 with 3 steps left resolves to noise.
                    // The velocity field is defined at every σ, so integrate
                    // the requested number of Euler steps over [strength, 0]
                    // instead; `strength` IS the starting noise level.
                    let sigmas: Vec<f32> =
                        (0..=steps).map(|k| st * (1.0 - k as f32 / steps as f32)).collect();
                    let start = 0usize;
                    let Some((chw, rh, rw)) = r.refs.first() else {
                        out[i] = Err("strength needs a reference image".into());
                        continue;
                    };
                    if (*rh as usize / 16) * (*rw as usize / 16) != n_gen {
                        out[i] = Err(format!(
                            "strength needs the reference at the output size ({}x{}, got {rw}x{rh})",
                            o.width, o.height
                        ));
                        continue;
                    }
                    match self.encode_image(chw, *rh, *rw) {
                        Ok(x0) => {
                            let lat: Vec<f32> = x0
                                .iter()
                                .zip(&noise)
                                .map(|(&a, &e)| (1.0 - st) * a + st * e)
                                .collect();
                            (lat, start, sigmas)
                        }
                        Err(e) => {
                            out[i] = Err(e);
                            continue;
                        }
                    }
                }
                _ => (noise, 0, sigmas),
            };
            lanes.push(Lane {
                idx: i,
                lh,
                lw,
                n_gen,
                steps,
                guidance: o.guidance,
                ctx,
                ctx_uncond,
                ref_tokens,
                sigmas,
                lat,
                start,
            });
        }
        if lanes.is_empty() {
            return;
        }
        let max_steps = lanes.iter().map(|l| l.steps).max().unwrap_or(0);
        let cap = self.model.max_batch() as usize;

        for i in 0..max_steps {
            // Cancellation is per request: a cancelled lane leaves the batch and
            // the others keep going (the scheduler handed us N independent jobs).
            lanes.retain(|l| {
                if reqs[l.idx].cancel.is_cancelled() {
                    out[l.idx] = Err("cancelled".into());
                    false
                } else {
                    true
                }
            });
            let active: Vec<usize> =
                (0..lanes.len()).filter(|&k| i >= lanes[k].start && i < lanes[k].steps).collect();
            if active.is_empty() {
                break;
            }
            progress(i as u32 + 1, max_steps as u32 + 2, "denoising");

            // Build one slot per DiT evaluation: (lane, ctx, t). CFG adds the
            // unconditional pass as a second slot at the same timestep.
            let mut joints: Vec<Vec<f32>> = Vec::with_capacity(active.len());
            let mut slots: Vec<(usize, bool, f32)> = Vec::new(); // (active index, is_uncond, t)
            for (a, &k) in active.iter().enumerate() {
                let l = &lanes[k];
                let mut joint = Vec::with_capacity(l.lat.len() + l.ref_tokens.len());
                joint.extend_from_slice(&l.lat);
                joint.extend_from_slice(&l.ref_tokens);
                joints.push(joint);
                slots.push((a, false, l.sigmas[i]));
                if l.ctx_uncond.is_some() {
                    slots.push((a, true, l.sigmas[i]));
                }
            }
            // One forward per chunk of at most `max_batch` slots.
            let mut preds: Vec<Vec<f32>> = Vec::with_capacity(slots.len());
            for chunk in slots.chunks(cap) {
                let samples: Vec<crate::model::Sample<'_>> = chunk
                    .iter()
                    .map(|&(a, unc, t)| {
                        let l = &lanes[active[a]];
                        let ctx = if unc { l.ctx_uncond.as_ref().unwrap() } else { &l.ctx };
                        crate::model::Sample { img_tokens: &joints[a], ctx, t }
                    })
                    .collect();
                preds.extend(self.model.forward_batch(&samples, ids, lanes[active[0]].n_gen));
            }
            // Fold CFG and take the Euler step, per lane.
            for (a, &k) in active.iter().enumerate() {
                let cond = slots.iter().position(|&(sa, unc, _)| sa == a && !unc).expect("cond slot");
                let pred: Vec<f32> = match slots.iter().position(|&(sa, unc, _)| sa == a && unc) {
                    None => preds[cond].clone(),
                    Some(u) => preds[cond].iter().zip(&preds[u]).map(|(&c, &un)| un + lanes[k].guidance * (c - un)).collect(),
                };
                let dt = lanes[k].sigmas[i + 1] - lanes[k].sigmas[i];
                for (x, v) in lanes[k].lat.iter_mut().zip(&pred) {
                    *x += dt * v;
                }
            }
        }

        progress(max_steps as u32 + 2, max_steps as u32 + 2, "decoding");
        for l in &lanes {
            let o = &reqs[l.idx].opts;
            out[l.idx] = self.decode_tokens(&l.lat, l.lh, l.lw).map(|rgb| (rgb, o.width, o.height));
        }
    }
}


/// One generated image `(rgb8, width, height)`, or why it failed. Named because
/// it appears in the batch entry point, its per-group helper and the
/// single-image wrapper, which must not drift apart.
pub type BatchOutcome = Result<(Vec<u8>, u32, u32), String>;

/// One generation in a [`Pipeline::generate_batch`] call: everything
/// `Pipeline::generate` takes, owned, plus its cancellation token.
#[derive(Clone)]
pub struct BatchRequest {
    pub prompt: String,
    /// RGB `[-1,1]` CHW reference images with their (h, w), pre-cropped to /16.
    pub refs: Vec<(Vec<f32>, u32, u32)>,
    pub opts: GenOpts,
    /// Polled once per denoise step; a `Default` token never fires.
    pub cancel: capability::CancelToken,
}

impl BatchRequest {
    /// Resolved step count (`opts.steps` or the variant default).
    fn steps_for(&self, distilled: bool) -> usize {
        self.opts.steps.unwrap_or(if distilled { 4 } else { 50 }) as usize
    }
}

/// Convert an interleaved HWC RGB image in `[0,1]` (the shared
/// `capability::blob` wire format, also what the CLI's PPM loader produces) to
/// the reference-image layout [`Pipeline::generate`] expects: `[-1,1]` CHW,
/// **center-cropped** to multiples of 16. Returns `(chw, h, w)` with the
/// cropped dims — the ONE implementation shared by the CLI and the capability
/// provider.
pub fn ref_from_hwc(hwc: &[f32], w: u32, h: u32) -> Result<(Vec<f32>, u32, u32), String> {
    let (cw, ch) = (w - w % 16, h - h % 16);
    if cw == 0 || ch == 0 {
        return Err(format!("reference image {w}×{h} is smaller than 16×16"));
    }
    let (x0, y0) = (((w - cw) / 2) as usize, ((h - ch) / 2) as usize);
    let mut chw = vec![0.0f32; 3 * (cw * ch) as usize];
    for c in 0..3usize {
        for y in 0..ch as usize {
            for x in 0..cw as usize {
                let v = hwc[((y + y0) * w as usize + (x + x0)) * 3 + c];
                chw[(c * ch as usize + y) * cw as usize + x] = 2.0 * v - 1.0;
            }
        }
    }
    Ok((chw, ch, cw))
}
