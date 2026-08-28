// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The full SUPIR restoration loop: a degraded image in, a restored one out.
//!
//! Everything this assembles was already parity-gated on its own -
//! [`crate::model::Supir`] (the trunk + adaptors + frozen UNet, one graph),
//! `diffusion::restore` (`RestoreEDMSampler`'s scalar math), the frozen SDXL
//! VAE and dual-CLIP conditioning ([`sdxlunet::textenc`]). What was missing
//! was the loop that puts them together - dual encode, conditioning,
//! sampling, decode, colour fix - which is why nothing built on this crate so
//! far could produce a picture. Mirrors [`sdxlunet::pipeline::Sdxl`]'s own
//! "thin glue, no re-implemented math" shape.
//!
//! ```text
//! _z       = 0.13025 . quant_conv(denoise_encoder(lq)).mean        # the HINT
//! x_stage1 = decoder(_z)                                            # frozen VAE decode
//! x_center = 0.13025 . quant_conv(encoder(x_stage1)).mean           # clean re-encode, guidance target
//! c, uc    = dual-CLIP(caption + positive_suffix) / (negative_prompt); c.control = uc.control = _z
//! x_T      = randn . SIGMA_MAX                                      # pure noise, not a noised LQ latent
//! x_0      = RestoreEDMSampler(model, x_T, c, uc, x_center)
//! out      = decoder(x_0), then colour-fixed against x_stage1
//! ```
//!
//! # `quant_conv`/`post_quant_conv`/the decoder are reused FROZEN
//!
//! `denoise_encoder` (SUPIR's own delta) is byte-identical topology to the
//! frozen SDXL VAE encoder - only its `conv_in`/down/mid/`conv_norm_out`/
//! `conv_out` weights differ (see [`crate::config::denoise_encoder_manifest`]'s
//! doc). [`crate::import::denoise_encoder_diffusers_names`] renames its
//! CompVis-named keys to the diffusers keys [`vae::VaeEncoder`] reads; this
//! module merges that renamed map with the frozen VAE's own `quant_conv`
//! weight (never SUPIR's own - it has none) before building the encoder
//! graph. The clean re-encode of `x_stage1` uses the frozen encoder
//! UNCHANGED - a second [`vae::VaeEncoder`] built straight from the backbone
//! checkpoint's `vae/` weights, same as [`sdxlunet::pipeline::Sdxl`]'s decode
//! half.
//!
//! # `control_scale` is baked into the graph, so the linear ramp is deferred
//!
//! [`crate::model::Supir::new`] records `control_scale` as a graph CONSTANT
//! (see that function's own doc) - there is no per-step device buffer to
//! write, unlike CodeFormer's fidelity dial. `linear_s_stage2` (upstream's
//! optional per-step control-scale ramp) is therefore NOT implemented here:
//! doing it faithfully would mean rebuilding - and re-uploading - the whole
//! trunk+adaptors+backbone graph every denoise step, which is not a
//! performance regression to accept quietly. Upstream's own CLI default
//! (`linear_s_stage2 = False`) already runs the constant path this module
//! implements; a rewritable control-scale buffer (`CodeFormer`'s `w`/
//! `scale_add` is the precedent) is the real fix, filed as a follow-up
//! alongside batch > 1 and tiled sampling.
//!
//! # Tiling is not wired in here
//!
//! `imaging::tiling`'s blended `TilePlan` variant exists (built for this
//! port), but composing it into this loop - splitting the sampler's per-step
//! forward across overlapping windows with a shared per-step noise field
//! sliced per tile - is independent, sizeable work this module leaves for a
//! follow-up. Every call here runs the whole image through one graph, which
//! is also why a real end-to-end call on this machine is expected to hit the
//! device-memory ceiling `crates/supir/tests/parity.rs` already documents.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use diffusion::discrete::{DiscreteConfig, Sigmas};
use diffusion::restore::{
    apply_churn_noise, churn_gamma, euler_step, linear_cfg_scale, restore_guidance, sigma_hat as sigma_hat_of,
    DiscreteDenoiserWithControl, RestoreEDMSamplerConfig, SIGMA_MAX,
};
use vae::config::VaeConfig;
use vae::{VaeDecoder, VaeEncoder};

use crate::config::SupirConfig;
use crate::model::Supir;
use sdxlunet::textenc::{read_any_safetensors, read_json, TextEncoders, CONTEXT};

/// SDXL's spatial downscale - the same 8x every diffusers `AutoencoderKL`
/// uses.
const VAE_SCALE: u32 = 8;
/// The pixel grid SUPIR snaps a resized LQ image to (upstream's own
/// `check_upscale`/`ImageProcessor` step).
const PIXEL_SNAP: u32 = 64;
/// Short side floor before snapping, per the spec's own resize rule.
const SHORT_SIDE_FLOOR: u32 = 1024;
/// `0.13025` - SDXL's own VAE `scaling_factor`, applied to BOTH encodes
/// (the hint and the clean re-encode), per the spec.
const SCALING_FACTOR: f32 = 0.13025;

/// SUPIR's own default positive-prompt suffix (`p_p`, appended to the
/// caption with no separator) and negative prompt (`n_p`, used alone) -
/// upstream's `options/SUPIR_v0.yaml` defaults, reproduced so a caller who
/// names only a caption (or none at all) gets what upstream's own CLI would
/// produce.
pub const DEFAULT_POSITIVE_SUFFIX: &str = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R \
camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous \
detailing, skin pore detailing, hyper sharpness, perfect without deformations.";
pub const DEFAULT_NEGATIVE_PROMPT: &str = "painting, oil painting, illustration, drawing, art, sketch, oil \
painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, \
frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth";

/// Checkpoint roots this pipeline loads from - `BRAIN_SDXL_DIR` (the frozen
/// backbone, same layout `sdxlunet`/`controlnet` already load) and
/// `BRAIN_SUPIR_DIR` (SUPIR's own delta: a directory holding the released
/// `SUPIR-v0*.safetensors`, or that file directly). No `default_ref`/auto-fetch
/// for either - SUPIR's weights carry a non-commercial licence, so a user
/// points brain at weights they obtained themselves (see this crate's own
/// module doc).
#[derive(Clone, Debug)]
pub struct Paths {
    pub backbone_root: String,
    pub supir_ckpt: String,
}

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        let backbone_root = std::env::var("BRAIN_SDXL_DIR").map_err(|_| "supir: set BRAIN_SDXL_DIR to a released diffusers SDXL checkpoint root".to_string())?;
        let supir_dir = std::env::var("BRAIN_SUPIR_DIR").map_err(|_| "supir: set BRAIN_SUPIR_DIR to a SUPIR delta checkpoint (file or directory)".to_string())?;
        let supir_ckpt = resolve_supir_checkpoint(&supir_dir)?;
        Ok(Paths { backbone_root, supir_ckpt })
    }
}

/// `dir` is either the checkpoint file itself, or a directory holding
/// exactly one `*.safetensors` file (preferring a name that contains
/// "supir", case-insensitively, when several exist - the released layout
/// ships `SUPIR-v0Q_fp32.safetensors` alongside nothing else, but a user's
/// own directory may not).
fn resolve_supir_checkpoint(dir: &str) -> Result<String, String> {
    let p = Path::new(dir);
    if p.is_file() {
        return Ok(dir.to_string());
    }
    let rd = std::fs::read_dir(p).map_err(|e| format!("supir: reading {dir}: {e}"))?;
    let mut candidates: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
    if candidates.is_empty() {
        return Err(format!("supir: no *.safetensors under {dir}"));
    }
    candidates.sort();
    if candidates.len() > 1 {
        if let Some(named) = candidates.iter().find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.to_lowercase().contains("supir"))) {
            return named.to_str().map(str::to_string).ok_or_else(|| "supir: non-UTF8 path".to_string());
        }
        return Err(format!("supir: {dir} holds {} candidate checkpoints; name the one to use directly", candidates.len()));
    }
    candidates[0].to_str().map(str::to_string).ok_or_else(|| "supir: non-UTF8 path".to_string())
}

/// One `restore` request's tunables - `RestoreEDMSamplerConfig`'s fields plus
/// the text conditioning, reproduced from upstream's own CLI defaults
/// (see `diffusion::restore::RestoreEDMSamplerConfig::default`).
#[derive(Clone, Debug)]
pub struct RestoreOptions {
    pub steps: usize,
    /// `s_cfg` - CFG scale at sigma -> 0.
    pub cfg_scale: f32,
    /// `spt_linear_CFG` - CFG scale at sigma_max.
    pub spt_linear_cfg: f32,
    /// `s_stage2` - the control scale (baked into the graph - see the module doc).
    pub control_scale: f32,
    pub s_churn: f32,
    pub s_noise: f32,
    /// `s_stage1` - restoration guidance. Negative is OFF (upstream's own default).
    pub restore_cfg: f32,
    pub seed: u64,
    /// The image caption (already resolved - auto-captioning through a
    /// `capability::Registry` is a serving-layer concern, see `crate::caps`).
    pub caption: String,
    pub positive_suffix: String,
    pub negative_prompt: String,
}

impl Default for RestoreOptions {
    fn default() -> RestoreOptions {
        let d = RestoreEDMSamplerConfig::default();
        RestoreOptions {
            steps: d.edm_steps,
            cfg_scale: d.s_cfg,
            spt_linear_cfg: d.spt_linear_cfg,
            control_scale: d.s_stage2,
            s_churn: d.s_churn,
            s_noise: d.s_noise,
            restore_cfg: d.s_stage1,
            seed: 0,
            caption: String::new(),
            positive_suffix: DEFAULT_POSITIVE_SUFFIX.to_string(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_string(),
        }
    }
}

/// The working pixel size for an `(w, h)` LQ image: resize so the short side
/// is at least [`SHORT_SIDE_FLOOR`], keeping aspect ratio, then snap both
/// axes UP to the nearest [`PIXEL_SNAP`] multiple (never down - shrinking
/// after the floor could drop back under it).
pub fn target_size(w: u32, h: u32) -> (u32, u32) {
    let (w, h) = (w.max(1), h.max(1));
    let short = w.min(h);
    let scale = if short >= SHORT_SIDE_FLOOR { 1.0 } else { SHORT_SIDE_FLOOR as f64 / short as f64 };
    let snap = |v: u32| -> u32 {
        let scaled = (v as f64 * scale).round() as u32;
        scaled.div_ceil(PIXEL_SNAP) * PIXEL_SNAP
    };
    (snap(w), snap(h))
}

/// A resident restoration graph for one `(pixel size, control_scale)` - the
/// per-step-reusable half of a request. Cheap to keep several of (the trunk
/// is what is expensive; the encoders/decoder below are built fresh per call,
/// same tiering as [`sdxlunet::pipeline::Sdxl`]).
pub struct Restorer {
    backbone_root: String,
    supir_ckpt: String,
    cfg: SupirConfig,
    model: Supir,
    hw: (u32, u32),
}

impl Restorer {
    /// `h`/`w` are the WORKING PIXEL size (already snapped via
    /// [`target_size`]) - the model graph is recorded at the matching latent
    /// size, `h/8 x w/8`, and a different pixel size needs a different
    /// `Restorer`.
    pub fn load(backbone_root: &str, supir_ckpt: &str, h: u32, w: u32, control_scale: f32) -> Result<Restorer, String> {
        if !h.is_multiple_of(VAE_SCALE) || !w.is_multiple_of(VAE_SCALE) {
            return Err(format!("supir: {w}x{h} is not a multiple of the VAE's {VAE_SCALE}x downscale"));
        }
        let cfg = SupirConfig::sdxl();
        let backbone_tensors = sdxlunet::import::load(&Path::new(backbone_root).join("unet").to_string_lossy(), &cfg.backbone)?;
        let delta_tensors = crate::import::load(supir_ckpt, &cfg)?;
        let mut tensors = backbone_tensors;
        tensors.extend(delta_tensors);

        let gpu = gpu_core::Gpu::new(&crate::model::KERNELS);
        let (lh, lw) = (h / VAE_SCALE, w / VAE_SCALE);
        let model = Supir::new(gpu, cfg.clone(), &tensors, lh, lw, CONTEXT as u32, false, control_scale);
        Ok(Restorer { backbone_root: backbone_root.to_string(), supir_ckpt: supir_ckpt.to_string(), cfg, model, hw: (h, w) })
    }

    pub fn hw(&self) -> (u32, u32) {
        self.hw
    }

    /// Restore `lq_hwc` (HWC RGB `f32` in `[0,1]`, `lq_w x lq_h`, any size -
    /// resized to this `Restorer`'s own `(h, w)` internally). Returns HWC RGB
    /// `f32` in `[0,1]` at `(h, w)`. Polls `cancel` once per denoise step.
    pub fn restore(&self, lq_hwc: &[f32], lq_w: u32, lq_h: u32, o: &RestoreOptions, cancel: &capability::CancelToken, progress: &mut dyn FnMut(u32, u32)) -> Result<Vec<f32>, String> {
        let (h, w) = self.hw;
        let resized_hwc = if (lq_w, lq_h) == (w, h) { lq_hwc.to_vec() } else { imaging::host::resize_bilinear_hwc(lq_hwc, 3, lq_w, lq_h, w, h) };
        // [0,1] -> [-1,1], HWC -> CHW: the range/layout every VAE encode in
        // this codebase expects (see `sdxlunet::pipeline::Sdxl::generate`'s
        // decode-side inverse of this same map).
        let signed: Vec<f32> = resized_hwc.iter().map(|&v| v * 2.0 - 1.0).collect();
        let lq_chw = imaging::pixels::hwc_to_chw(&signed, 3, h as usize, w as usize);

        let (lh, lw) = (h / VAE_SCALE, w / VAE_SCALE);
        let n_latent = (self.cfg.backbone.in_channels * lh * lw) as usize;

        // ---- dual encode ----------------------------------------------------
        let denc_tensors = crate::import::load(&self.supir_ckpt, &self.cfg)?;
        let vae_cfg = VaeConfig::from_json(&read_json(&Path::new(&self.backbone_root).join("vae/config.json"))?);
        let frozen_vae_tensors: vae::Tensors = read_any_safetensors(&Path::new(&self.backbone_root).join("vae"))?
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect();

        let denc_prefixed: HashMap<String, (Vec<usize>, Vec<f32>)> =
            denc_tensors.into_iter().filter_map(|(k, v)| k.strip_prefix("denoise_encoder.").map(|s| (s.to_string(), v))).collect();
        let mut hint_tensors = crate::import::denoise_encoder_diffusers_names(&denc_prefixed, &self.cfg.denoise_encoder);
        for key in ["quant_conv.weight", "quant_conv.bias"] {
            if let Some(v) = frozen_vae_tensors.get(key) {
                hint_tensors.insert(key.to_string(), v.clone());
            }
        }
        // Every auxiliary encode/decode below runs on the CPU backend by
        // default, same as `sdxlunet::pipeline::Sdxl`'s own VAE decode: the
        // main model already holds the ONE device this process opens for the
        // multi-step loop, so a second live `wgpu` device would either fail
        // outright or fight it for the same shared iGPU memory - see that
        // module's own comment for the measured OOM this avoids.
        let hint_encoder = VaeEncoder::from_diffusers(self.cfg.denoise_encoder.clone(), &hint_tensors, h, w, Some("cpu"));
        let hint_moments = hint_encoder.encode(&lq_chw);
        let lc = self.cfg.denoise_encoder.latent_channels as usize;
        let hint: Vec<f32> = hint_moments[..lc * (lh * lw) as usize].iter().map(|&v| v * SCALING_FACTOR).collect();

        let x_stage1_chw = {
            let vae_decoder = VaeDecoder::from_diffusers(vae_cfg.clone(), &frozen_vae_tensors, lh, lw, Some("cpu"));
            let z: Vec<f32> = hint.iter().map(|&v| v / vae_cfg.scaling_factor).collect();
            vae_decoder.decode(&z)
        };
        let x_center: Vec<f32> = {
            let clean_encoder = VaeEncoder::from_diffusers(vae_cfg.clone(), &frozen_vae_tensors, h, w, Some("cpu"));
            let moments = clean_encoder.encode(&x_stage1_chw);
            moments[..lc * (lh * lw) as usize].iter().map(|&v| v * SCALING_FACTOR).collect()
        };

        // ---- dual-CLIP conditioning ------------------------------------------
        // `share()`: a fresh handle over the SAME device the model's own
        // graph is resident on - `TextEncoders::load` only uses this as the
        // source for `Gpu::new_like` (the towers' own kernel set), so no
        // second device is opened.
        let te = TextEncoders::load(self.model.gpu().share(), &self.backbone_root)?;
        let positive = format!("{}{}", o.caption, o.positive_suffix);
        let mut enc = te.encode_all(&[positive.as_str(), o.negative_prompt.as_str()])?;
        let (cond_enc, cond_pooled) = enc.remove(0);
        let (uncond_enc, uncond_pooled) = enc.remove(0);

        // ---- sample -----------------------------------------------------------
        let time_ids = vec![h as f32, w as f32, 0.0, 0.0, h as f32, w as f32];
        let x0 = self.sample(n_latent, &hint, &cond_enc, &cond_pooled, &uncond_enc, &uncond_pooled, &time_ids, &x_center, o, cancel, progress)?;

        // ---- decode + colour fix -----------------------------------------------
        let out_chw = {
            let vae_decoder = VaeDecoder::from_diffusers(vae_cfg.clone(), &frozen_vae_tensors, lh, lw, Some("cpu"));
            let z: Vec<f32> = x0.iter().map(|&v| v / vae_cfg.scaling_factor).collect();
            vae_decoder.decode(&z)
        };
        let fixed_chw = imaging::colorfix::wavelet_reconstruction(&out_chw, &x_stage1_chw, 3, h as usize, w as usize);
        let rgb: Vec<f32> = fixed_chw.iter().map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0)).collect();
        Ok(imaging::pixels::chw_to_hwc(&rgb, 3, h as usize, w as usize))
    }

    /// `RestoreEDMSampler`'s loop: `diffusion::restore`'s scalar math driving
    /// [`crate::model::Supir::run`] per step, CFG combined in eps-space
    /// (`sdxlunet::sampler::sample`'s own convention).
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        n_latent: usize,
        hint: &[f32],
        cond_enc: &[f32],
        cond_pooled: &[f32],
        uncond_enc: &[f32],
        uncond_pooled: &[f32],
        time_ids: &[f32],
        x_center: &[f32],
        o: &RestoreOptions,
        cancel: &capability::CancelToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        let steps = o.steps.max(2);
        let grid = DiscreteDenoiserWithControl::new();
        let sigmas = Sigmas::new(&DiscreteConfig::sdxl(), steps, 0.0).sigmas;
        let do_cfg = o.cfg_scale > 1.0 || o.spt_linear_cfg > 1.0;
        let gamma = churn_gamma(o.s_churn, steps);

        let mut x: Vec<f32> = model::hostmath::gaussian(n_latent, o.seed).into_iter().map(|v| v * SIGMA_MAX).collect();

        for step in 0..steps {
            if cancel.is_cancelled() {
                return Err("cancelled".to_string());
            }
            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let s_hat = sigma_hat_of(sigma, gamma);
            if gamma > 0.0 {
                let noise = model::hostmath::gaussian(n_latent, o.seed.wrapping_add(1 + step as u64));
                x = apply_churn_noise(&x, &noise, sigma, s_hat, o.s_noise);
            }
            let c_in = 1.0 / (s_hat * s_hat + 1.0).sqrt();
            let scaled: Vec<f32> = x.iter().map(|&v| v * c_in).collect();
            let t = grid.index(s_hat) as f32;

            let eps_c = self.model.run(&scaled, hint, t, cond_enc, cond_pooled, time_ids);
            let eps = if do_cfg {
                let eps_u = self.model.run(&scaled, hint, t, uncond_enc, uncond_pooled, time_ids);
                let g = linear_cfg_scale(o.spt_linear_cfg, o.cfg_scale, s_hat, SIGMA_MAX);
                eps_u.iter().zip(&eps_c).map(|(&u, &c)| u + g * (c - u)).collect()
            } else {
                eps_c
            };

            let mut denoised: Vec<f32> = x.iter().zip(&eps).map(|(&xi, &e)| xi - s_hat * e).collect();
            denoised = restore_guidance(&denoised, x_center, sigma, SIGMA_MAX, o.restore_cfg, sigma_next);
            x = euler_step(&x, &denoised, s_hat, sigma_next);
            progress(step as u32 + 1, steps as u32);
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_snaps_up_to_64_and_lifts_a_small_short_side() {
        // Already above the floor and already 64-aligned: unchanged.
        assert_eq!(target_size(1024, 1536), (1024, 1536));
        // Short side below the floor: scaled up, then snapped.
        let (w, h) = target_size(512, 768);
        assert!(w % PIXEL_SNAP == 0 && h % PIXEL_SNAP == 0);
        assert!(w.min(h) >= SHORT_SIDE_FLOOR);
        // Above the floor but not 64-aligned: snapped up, never down.
        let (w2, h2) = target_size(1030, 2000);
        assert_eq!(w2 % PIXEL_SNAP, 0);
        assert_eq!(h2 % PIXEL_SNAP, 0);
        assert!(w2 >= 1030 && h2 >= 2000);
    }

    #[test]
    fn default_options_leave_restoration_guidance_off_and_match_upstream() {
        let o = RestoreOptions::default();
        assert_eq!(o.steps, 50);
        assert_eq!(o.cfg_scale, 4.0);
        assert_eq!(o.spt_linear_cfg, 1.0);
        assert_eq!(o.control_scale, 1.0);
        assert!(o.restore_cfg < 0.0, "restoration guidance must default off");
        assert_eq!(o.positive_suffix, DEFAULT_POSITIVE_SUFFIX);
        assert_eq!(o.negative_prompt, DEFAULT_NEGATIVE_PROMPT);
    }
}
