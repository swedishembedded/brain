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
    /// None → the variant default (4 distilled / 50 base).
    pub steps: Option<u32>,
    /// CFG scale — only meaningful for the undistilled base variants.
    pub guidance: f32,
    pub seed: u64,
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts { width: 1024, height: 1024, steps: None, guidance: 4.0, seed: 0 }
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
        let mut dit_ts = read_dit_tensors(&paths.dit, cfg)?;
        if let Some(ap) = adapter {
            // The adapter's tensor shapes depend only on the architecture, not
            // the latent grid — any (lh, lw) loads it.
            let tcfg = crate::modelgrad::Cfg::from_flux2(cfg, 1, 1);
            let ad = crate::lora::load_adapter(ap, &tcfg)?;
            ad.fold_into_tensors(&mut dit_ts)?;
        }
        let gpu = gpu_core::Gpu::new(crate::model::KERNELS);
        let model = Flux2Model::new_with(cfg, &dit_ts, gpu, cfg.txt_len as u32 + n_img_max, precision);
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

    /// Text-to-image (optionally with reference images for editing).
    /// `refs`: RGB `[-1,1]` CHW images, each with its (h, w) — pre-cropped to
    /// multiples of 16 (see [`ref_from_hwc`]). Returns (rgb8 HWC, width,
    /// height). `cancel` is polled once per denoise step (multi-minute CPU
    /// runs must be abortable); a `Default` token never fires.
    pub fn generate(
        &self,
        prompt: &str,
        refs: &[(Vec<f32>, u32, u32)],
        o: &GenOpts,
        cancel: &capability::CancelToken,
        mut progress: impl FnMut(u32, u32, &str),
    ) -> Result<(Vec<u8>, u32, u32), String> {
        let cfg = &self.cfg;
        assert!(o.width % 16 == 0 && o.height % 16 == 0, "H,W must be /16");
        let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
        let n_gen = lh * lw;
        let steps = o.steps.unwrap_or(if cfg.distilled { 4 } else { 50 }) as usize;
        let cf = !cfg.distilled && o.guidance > 1.0;

        progress(0, steps as u32 + 2, "encoding prompt");
        let ctx = self.encode_prompt(prompt);
        let ctx_uncond = if cf { Some(self.encode_prompt("")) } else { None };

        // reference images -> tokens + ids
        let mut ref_tokens: Vec<f32> = Vec::new();
        let mut ref_dims: Vec<(usize, usize)> = Vec::new();
        for (chw, rh, rw) in refs {
            progress(0, steps as u32 + 2, "encoding reference");
            ref_tokens.extend(self.encode_image(chw, *rh, *rw)?);
            ref_dims.push(((rh / 16) as usize, (rw / 16) as usize));
        }
        let ids = position_ids(cfg.txt_len, lh, lw, &ref_dims);

        let sigmas = diffusion::scheduler::klein_sigmas(steps, n_gen);
        let mut lat = model::hostmath::randn(n_gen * cfg.in_channels, o.seed);

        for i in 0..steps {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            progress(i as u32 + 1, steps as u32 + 2, "denoising");
            let t = sigmas[i];
            let dt = sigmas[i + 1] - sigmas[i];
            let mut joint: Vec<f32> = Vec::with_capacity((n_gen + ref_tokens.len() / cfg.in_channels) * cfg.in_channels);
            joint.extend_from_slice(&lat);
            joint.extend_from_slice(&ref_tokens);
            let pred = self.model.forward(&joint, &ctx, t, &ids, n_gen);
            let pred = match &ctx_uncond {
                None => pred,
                Some(cu) => {
                    let pu = self.model.forward(&joint, cu, t, &ids, n_gen);
                    pred.iter()
                        .zip(&pu)
                        .map(|(&c, &u)| u + o.guidance * (c - u))
                        .collect()
                }
            };
            for (x, v) in lat.iter_mut().zip(&pred) {
                *x += dt * v;
            }
        }
        progress(steps as u32 + 2, steps as u32 + 2, "decoding");
        let rgb = self.decode_tokens(&lat, lh, lw)?;
        Ok((rgb, o.width, o.height))
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
