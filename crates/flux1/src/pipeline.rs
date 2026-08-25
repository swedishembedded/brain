// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FLUX.1 text-to-image pipeline: a prompt in, an image out.
//!
//! Mirrors `sdxlunet::pipeline::Sdxl` and `flux2::pipeline::Pipeline`'s shape
//! (build the conditioning towers, denoise, VAE decode) but is its own loop:
//! FLUX.1 conditions on T5-XXL context + a CLIP-L pooled vector (not FLUX.2's
//! Qwen3 taps), has no undistilled base variant (so no CFG branch at all -
//! `dev`/`kontext-dev` fold a guidance SCALAR into the conditioning via
//! `guidance_in`, `schnell` ignores it), and its 16-channel VAE uses the
//! model's own scalar `(shift_factor, scaling_factor)` affine, NOT
//! `vae::latent::pack`/`unpack` (that module is explicitly FLUX.2's
//! BatchNorm-based packing - reusing it here would silently apply the wrong
//! normalization).
//!
//! # The schedule is FLUX.1's own, not FLUX.2's
//!
//! `diffusion::scheduler::empirical_mu` is FLUX.2 Klein's own empirical fit
//! (its doc says so) - wrong constants for FLUX.1. FLUX.1's `dev`/
//! `kontext-dev` use BFL's `calculate_shift`/`get_lin_function`
//! (`flux/sampling.py`, verbatim in every diffusers `FluxPipeline`): a
//! LINEAR `mu(image_seq_len)` between `(256, 0.5)` and `(4096, 1.15)`, fed
//! through the same [`diffusion::scheduler::time_shift_exponential`] FLUX.2
//! reuses. `schnell` applies **no shift at all** (`shift=(not is_schnell)` in
//! BFL's own CLI) - a plain `linspace(1, 0, steps+1)`.
//!
//! The denoise loop steps this schedule directly (`dt = sigmas[i+1] -
//! sigmas[i]`), the same manual style `flux2::pipeline::Pipeline` uses,
//! rather than through `FlowMatchEulerScheduler` (that wrapper appends its
//! own terminal `0` unconditionally, which double-counts against a
//! `steps+1`-length input like this one - `flux2` avoids it for the same
//! reason).
//!
//! # Not yet in scope
//!
//! Kontext reference-image editing, img2img (`strength`), LoRA adapters, and
//! `int8`/batched serving are all deferred - this is a single-image
//! text-to-image loop. `flux2::pipeline` is the fuller reference for what
//! each of those needs when they land here.
//!
//! # An honest note on verification
//!
//! Every piece this composes (the DiT forward, the T5/CLIP towers, the VAE)
//! is independently parity-gated elsewhere in this workspace. The GLUE
//! written here - patchify layout, position ids, the schedule, the affine
//! latent normalization - is NOT: there is no FLUX.1 checkpoint or reference
//! pipeline dump in this environment to run it against end to end. Treat a
//! first real generation as the actual test of this file.

use std::path::Path;

use clip::config::ClipTextConfig;
use clip::model::ClipText;
use data::unigram::UnigramTokenizer;
use diffusion::scheduler::time_shift_exponential;
use gpu_core::Gpu;
use t5encoder::config::T5Config;
use t5encoder::model::T5Encoder;
use vae::config::VaeConfig;
use vae::VaeDecoder;

use crate::config::Flux1Config;
use crate::model::{position_ids, Flux1Model, KERNELS};

/// How the latent is seeded and how many steps to take.
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// `None` -> the variant default (4 for schnell, 50 for dev/kontext-dev -
    /// matching BFL's own CLI defaults).
    pub steps: Option<usize>,
    /// `guidance_in`'s scalar. Only meaningful for `dev`/`kontext-dev`
    /// (`guidance_embed = true`); `schnell` ignores it. BFL's CLI default is
    /// 3.5.
    pub guidance: f32,
    pub seed: u64,
    /// Generated size in pixels; must be a multiple of 16 (the VAE's 8x
    /// downscale composed with the DiT's 2x2 patchify).
    pub height: u32,
    pub width: u32,
}

impl Default for GenerateOptions {
    fn default() -> GenerateOptions {
        GenerateOptions { steps: None, guidance: 3.5, seed: 0, height: 1024, width: 1024 }
    }
}

/// A loaded FLUX.1 stack.
///
/// # Only the DiT stays resident, for the same reason as SDXL
///
/// FLUX.1-dev is ~12 B params (~48 GB fp32) - even more VRAM-constrained than
/// SDXL. The T5-XXL/CLIP-L towers and the VAE are built for one encode/decode
/// and dropped, the same tiering `sdxlunet::pipeline::Sdxl` uses and for the
/// same reason (documented on its `Sdxl` struct).
pub struct Flux1 {
    gpu: Gpu,
    root: String,
    cfg: Flux1Config,
    variant: String,
    dit: Flux1Model,
    vae_cfg: VaeConfig,
    hw: (u32, u32),
}

/// Inverse of the DiT's token patchify: predicted/denoised tokens
/// `[lh*lw, 4c]` (row-major, matching [`position_ids`]'s (h, w) order) back
/// to the VAE's `[c, h, w]` latent mean, undoing FLUX.1's own affine
/// normalization on the way. `h, w` are the UNPACKED (VAE-latent) dims.
///
/// Released `ae.safetensors`/diffusers `vae/` carries no `bn.running_{mean,
/// var}` (that is FLUX.2's scheme, `vae::latent::pack`/`unpack` is wrong
/// here); just a scalar `(shift_factor, scaling_factor)` affine, the inverse
/// of BFL's `AutoEncoder.encode`: `z = (posterior.mean - shift) * scale`.
/// A `pack_tokens` (forward direction) has no caller yet, per the module
/// docs' "not yet in scope" list, and is deliberately not written until
/// img2img needs it, rather than shipped untested.
fn unpack_tokens(tokens: &[f32], c: usize, h: usize, w: usize, shift: f32, scale: f32) -> Vec<f32> {
    assert!(h.is_multiple_of(2) && w.is_multiple_of(2), "flux1: latent {h}x{w} must be even");
    let (lh, lw) = (h / 2, w / 2);
    let mut out = vec![0.0f32; c * h * w];
    for ci in 0..c {
        for pi in 0..2 {
            for pj in 0..2 {
                let oc = ci * 4 + pi * 2 + pj;
                for y in 0..lh {
                    for x in 0..lw {
                        out[(ci * h + 2 * y + pi) * w + 2 * x + pj] =
                            tokens[(y * lw + x) * (4 * c) + oc] / scale + shift;
                    }
                }
            }
        }
    }
    out
}

/// Box-Muller normal samples from the workspace's deterministic LCG, so a
/// seed reproduces a picture - the same construction `sdxlunet::pipeline`
/// and `controlnet::caps` use.
fn gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = (rng.next_f32().abs()).max(1e-7);
        let u2 = rng.next_f32().abs();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f32::consts::TAU * u2;
        out.push(r * th.cos());
        if out.len() < n {
            out.push(r * th.sin());
        }
    }
    out
}

/// BFL's `linspace(1, 0, steps+1)`, optionally shifted by the linear
/// `calculate_shift` mu - see the module docs for why this is not
/// `diffusion::scheduler::klein_sigmas`/`empirical_mu`.
fn flux1_sigmas(steps: usize, image_seq_len: usize, dynamic_shift: bool) -> Vec<f32> {
    let base: Vec<f32> = (0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect();
    if !dynamic_shift {
        return base;
    }
    // BFL `get_lin_function(base_seq_len=256, max_seq_len=4096, base_shift=0.5,
    // max_shift=1.15)` - the exact constants `flux/sampling.py` and every
    // diffusers `FluxPipeline` use for `dev`/`kontext-dev`.
    let (base_seq_len, max_seq_len, base_shift, max_shift) = (256.0f32, 4096.0f32, 0.5f32, 1.15f32);
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    let b = base_shift - m * base_seq_len;
    let mu = image_seq_len as f32 * m + b;
    time_shift_exponential(mu, &base)
}

fn read_json(p: &Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("flux1: reading {}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("flux1: parsing {}: {e}", p.display()))
}

/// Read the DiT weights from a diffusers `transformer/` dir, a BFL
/// single-file safetensors, or a GGUF, onto the canonical BFL names - the
/// same probe `flux2::pipeline::read_dit_tensors` uses.
fn read_dit_tensors(path: &str, cfg: &Flux1Config) -> Result<crate::import::Tensors, String> {
    let p = Path::new(path);
    if p.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("flux1: {path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("flux1: no .safetensors under {path}"));
        }
        let mut ts = Vec::new();
        for f in files {
            ts.extend(checkpoint::safetensors::read(f.to_str().ok_or("flux1: non-UTF8 path")?)?);
        }
        crate::import_diffusers(ts, cfg)
    } else if p.extension().is_some_and(|x| x == "gguf") {
        crate::import_bfl(checkpoint::gguf::read(path)?, cfg)
    } else {
        let ts = checkpoint::safetensors::read(path)?;
        if ts.iter().any(|t| t.name.starts_with("transformer_blocks.")) {
            crate::import_diffusers(ts, cfg)
        } else {
            crate::import_bfl(ts, cfg)
        }
    }
}

impl Flux1 {
    /// Load from a released FLUX.1 directory (the HF `black-forest-labs/
    /// FLUX.1-*` layout: `transformer/`, `vae/`, `text_encoder/`+`tokenizer/`
    /// for CLIP-L, `text_encoder_2/`+`tokenizer_2/` for T5-XXL - the same
    /// `text_encoder_2/`+`tokenizer_2/` shape `t5encoder::caps`'s `flux_xxl`
    /// variant already expects, since it is the same checkpoint family).
    ///
    /// `variant` is `dev` | `kontext-dev` | `schnell`
    /// ([`Flux1Config::from_name`]). `h`/`w` are the generated size: the DiT's
    /// max joint-token budget is sized for exactly this latent, so a
    /// different size needs a different `Flux1`.
    pub fn load(root: &str, variant: &str, h: u32, w: u32) -> Result<Flux1, String> {
        let cfg = Flux1Config::from_name(variant)?;
        let scale = 16u32; // VAE downscale 8 * DiT 2x2 patchify
        if !h.is_multiple_of(scale) || !w.is_multiple_of(scale) {
            return Err(format!("flux1: {w}x{h} is not a multiple of {scale}"));
        }
        let r = Path::new(root);
        let (lh, lw) = ((h / 16) as usize, (w / 16) as usize);
        let n_max = (lh * lw) as u32; // no txt/refs headroom yet: text2image only

        let gpu = Gpu::new(KERNELS);

        let dit_dir = r.join("transformer");
        let dit_path = if dit_dir.exists() { dit_dir } else { r.to_path_buf() };
        let ts = read_dit_tensors(dit_path.to_str().ok_or("flux1: non-UTF8 transformer path")?, &cfg)?;
        // txt_len is a pipeline argument, not baked into the config - the
        // model is sized for the WORST case this pipeline ever calls it with
        // (image tokens only, today; +txt_len when conditioning is threaded
        // through `n_max` here matches `Flux1Model::new`'s own doc: "at most
        // n_max joint tokens (txt + image + reference)"). Text2image submits
        // `ctx` and `img_tokens` as separate arguments to `forward`, so
        // `n_max` only needs to cover the image tokens this pipeline builds.
        let dit = Flux1Model::new(&cfg, &ts, gpu.share(), n_max);

        let vae_json = r.join("vae").join("config.json");
        let vae_cfg = if vae_json.exists() {
            VaeConfig::from_json(&read_json(&vae_json)?)
        } else {
            // BFL's released `ae.safetensors` (as opposed to the diffusers
            // `vae/config.json` release layout) ships no config at all.
            // `VaeConfig::from_json`'s own fallbacks already ARE FLUX.1's
            // architecture (16 latent channels, [128,256,512,512], the
            // published scaling_factor/shift_factor - this crate's own doc
            // names FLUX.1 as the reference case) with ONE exception:
            // `use_quant_conv`/`use_post_quant_conv` default true (the
            // SDXL/SD1.x-family default) because a real config.json only
            // carries the keys it OVERRIDES - but FLUX.1/Z-Image's real
            // released configs explicitly set both false, so an empty json
            // here must too.
            VaeConfig {
                use_quant_conv: false,
                use_post_quant_conv: false,
                ..VaeConfig::from_json(&serde_json::json!({}))
            }
        };

        Ok(Flux1 { gpu, root: root.into(), cfg, variant: variant.into(), dit, vae_cfg, hw: (h, w) })
    }

    fn clip_l(&self) -> Result<ClipText, String> {
        let cfg = ClipTextConfig::clip_l();
        let t = clip::import::read_text_encoder(&Path::new(&self.root).join("text_encoder"))?;
        let init = clip::import::import_text(t, &cfg)?;
        let map: std::collections::HashMap<String, Vec<f32>> =
            init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        Ok(ClipText::new_on(self.gpu.new_like(clip::model::TEXT_PIPELINES), cfg, 1, 77, &map))
    }

    fn t5_xxl(&self, max_len: usize) -> Result<T5Encoder, String> {
        let cfg = T5Config::xxl();
        let dir = Path::new(&self.root).join("text_encoder_2");
        let tensors = t5encoder::import::read_encoder(&dir)?;
        let init: std::collections::HashMap<String, Vec<f32>> =
            t5encoder::import::import_hf(tensors, &cfg)?.into_iter().map(|(k, (_, d))| (k, d)).collect();
        Ok(T5Encoder::new_on(self.gpu.new_like(t5encoder::model::PIPELINES), cfg, 1, max_len as u32, &init))
    }

    /// `(pooled[768], ctx[max_len*4096])` - CLIP-L's pooled EOS row (it does
    /// not project; only OpenCLIP-bigG does) and T5-XXL's unmasked context
    /// (FLUX passes no `attention_mask`, so right-pad positions are
    /// attended as ordinary keys - `t5encoder::caps`'s `flux_xxl` variant
    /// documents the same choice).
    fn encode(&self, prompt: &str, max_len: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let clip_tok = data::clip_bpe::ClipBpe::from_dir(&Path::new(&self.root).join("tokenizer"))
            .map_err(|e| format!("flux1: CLIP tokenizer: {e}"))?;
        let clip = self.clip_l()?;
        clip.set_tokens(&clip_tok.encode_with_context(prompt, 77).ids);
        clip.forward();
        let pooled = clip.read_pooled();
        drop(clip);

        let t5_tok = UnigramTokenizer::from_dir(Path::new(&self.root).join("tokenizer_2").to_str().ok_or("flux1: non-UTF8 path")?)
            .map_err(|e| format!("flux1: T5 tokenizer: {e}"))?;
        let (ids, _mask) = t5_tok.encode_padded(prompt, max_len);
        let t5 = self.t5_xxl(max_len)?;
        t5.set_tokens(&ids);
        t5.forward();
        let ctx = t5.read_hidden(); // unmasked: no pad-row zeroing (see `encode`'s doc)

        Ok((pooled, ctx))
    }

    /// Generate one image. Returns HWC RGB in `[0,1]`.
    pub fn generate(&self, prompt: &str, o: &GenerateOptions, max_len: usize) -> Result<Vec<f32>, String> {
        self.generate_injected(prompt, o, max_len, None)
    }

    /// [`Flux1::generate`] with every DiT step routed through
    /// `Flux1Model::forward_injected` when `inject` is `Some` - the seam
    /// `pulid::caps` uses to condition on an identity, and `crates/flux1`'s
    /// own `inject::BlockInject` trait so this needs no dependency on
    /// `pulid` (or any other adapter crate) to exist.
    pub fn generate_injected(
        &self,
        prompt: &str,
        o: &GenerateOptions,
        max_len: usize,
        inject: Option<&dyn crate::inject::BlockInject>,
    ) -> Result<Vec<f32>, String> {
        let (h, w) = self.hw;
        let (lh, lw) = ((h / 16) as usize, (w / 16) as usize);
        let n_gen = lh * lw;

        let (pooled, ctx) = self.encode(prompt, max_len)?;

        let dynamic_shift = self.variant != "schnell";
        let steps = o.steps.unwrap_or(if self.variant == "schnell" { 4 } else { 50 });
        let sigmas = flux1_sigmas(steps, n_gen, dynamic_shift);

        let ids = position_ids(max_len, lh, lw, &[]);
        let mut lat = gaussian(n_gen * self.cfg.in_channels, o.seed);

        for i in 0..steps {
            let t = sigmas[i];
            let pred = match inject {
                None => self.dit.forward(&lat, &ctx, &pooled, t, o.guidance, &ids, n_gen),
                Some(inj) => self.dit.forward_injected(&lat, &ctx, &pooled, t, o.guidance, &ids, n_gen, inj),
            };
            let dt = sigmas[i + 1] - t;
            for (x, v) in lat.iter_mut().zip(&pred) {
                *x += dt * v;
            }
        }

        // `in_channels = 16*2*2`; unpack back to the VAE's [16, h_lat, w_lat]
        // latent, undo the affine, then decode.
        let c = self.cfg.in_channels / 4;
        let unpacked = unpack_tokens(&lat, c, lh * 2, lw * 2, self.vae_cfg.shift_factor, self.vae_cfg.scaling_factor);
        let vt = read_any_safetensors(&Path::new(&self.root).join("vae"))?;
        let vmap: vae::blocks::Tensors = vt.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
        let vdev = std::env::var("BRAIN_FLUX1_VAE_DEVICE").unwrap_or_else(|_| "cpu".into());
        let dec = VaeDecoder::from_diffusers(self.vae_cfg.clone(), &vmap, (lh * 2) as u32, (lw * 2) as u32, Some(&vdev));
        let chw = dec.decode(&unpacked);
        // diffusers maps the decoder's [-1,1] output to [0,1].
        let rgb: Vec<f32> = chw.iter().map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0)).collect();
        Ok(imaging::pixels::chw_to_hwc(&rgb, 3, h as usize, w as usize))
    }
}

fn read_any_safetensors(dir: &Path) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("flux1: reading {}: {e}", dir.display()))?;
    let mut files: Vec<std::path::PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("flux1: no *.safetensors under {}", dir.display()));
    }
    let mut out = Vec::new();
    for f in files {
        out.extend(checkpoint::safetensors::read(f.to_str().ok_or("flux1: non-UTF8 path")?)?);
    }
    Ok(out)
}

