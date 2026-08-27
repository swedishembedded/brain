// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL + ControlNet behind the generalized [`capability`] interface - what
//! makes `brain caps sdxl-controlnet` / `brain do sdxl-controlnet text2image
//! ...`, the D-Bus `Run` method and `brain perf`'s `CapabilityTarget` work
//! with no ControlNet-specific plumbing in the CLI or the transports.
//!
//! One action: **`text2image`** - a prompt plus a conditioning image in, an
//! HWC RGB image out. This is `sdxlunet::sampler::sample`'s loop with
//! `sdxlunet::model::Unet::new_controlled` in place of `Unet::new` and one
//! [`ControlNet::run`] per step, whose [`Residuals`] are ordered by
//! [`adapter::order_for`] and threaded into `Unet::run_with_control` - the
//! [`sdxlunet::sampler::Denoiser`] seam `sdxlunet::pipeline::Sdxl` also
//! implements, over the plain forward. Nothing here re-implements the
//! sampling math - the two CLIP towers (`sdxlunet::textenc`), the discrete
//! scheduler and the VAE decode are the same calls `pipeline::Sdxl` makes.
//!
//! # No batching, for the same reason as plain SDXL
//!
//! Every request is its own multi-step sample; see
//! `crates/cli/src/resident_controlnet.rs`'s module docs.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use gpu_core::Gpu;
use serde_json::json;
use vae::config::VaeConfig;
use vae::VaeDecoder;

use crate::adapter::order_for;
use crate::config::ControlNetConfig;
use crate::model::ControlNet;
use sdxlunet::config::UNetConfig;
use sdxlunet::model::Unet;
use sdxlunet::sampler::{sample, Denoiser, SamplerOptions, StepCtx};
use sdxlunet::textenc::{read_any_safetensors, read_json, TextEncoders, CONTEXT};

/// The model id used on the CLI (`brain do sdxl-controlnet ...`), over D-Bus
/// and in the residency manifest.
pub const MODEL: &str = "brain/sdxl-controlnet";

fn text2image_spec() -> ActionSpec {
    ActionSpec::new("text2image", "Generate an image from a text prompt, conditioned on a control image (SDXL ControlNet).")
        .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("negative", ParamType::Str, "negative prompt (only used when guidance > 1.0)").default(json!("")))
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 8)").default(json!(1024)).min(256.0).max(2048.0).step(8.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 8)").default(json!(1024)).min(256.0).max(2048.0).step(8.0))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps").default(json!(30)).min(1.0).max(150.0).step(1.0))
        .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance scale; 1.0 disables CFG").default(json!(5.0)).min(1.0).max(30.0).step(0.1))
        .param(ParamSpec::new("conditioning_scale", ParamType::Float, "how strongly the control image steers the result").default(json!(1.0)).min(0.0).max(2.0).step(0.05))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for 0)"))
        .input(BlobSpec::new("control_image", Media::Image, "the conditioning image (resized to the output size)").required())
        .output(BlobSpec::new("image", Media::Image, "the generated image"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "SDXL text-to-image conditioned on a control image (edge map, depth map, pose, ...) via a ControlNet.",
        vec![text2image_spec()],
    )
}

struct Req {
    prompt: String,
    negative: String,
    steps: usize,
    guidance: f32,
    conditioning_scale: f32,
    seed: u64,
    height: u32,
    width: u32,
}

fn req_from(inv: &Invocation) -> Req {
    Req {
        prompt: inv.get_str("prompt").unwrap_or_default(),
        negative: inv.get_str("negative").unwrap_or_default(),
        steps: inv.get_i64("steps").unwrap_or(30).max(1) as usize,
        guidance: inv.get_f64("guidance").unwrap_or(5.0) as f32,
        conditioning_scale: inv.get_f64("conditioning_scale").unwrap_or(1.0) as f32,
        seed: inv.get_i64("seed").unwrap_or(0) as u64,
        height: inv.get_i64("height").unwrap_or(1024).max(8) as u32,
        width: inv.get_i64("width").unwrap_or(1024).max(8) as u32,
    }
}

// ===================== the pipeline =====================

/// One (SDXL + ControlNet) pair, recorded for one `(h, w)`.
struct Controlled {
    gpu: Gpu,
    sdxl_root: String,
    unet: Unet,
    control: ControlNet,
    vae_cfg: VaeConfig,
    ucfg: UNetConfig,
    hw: (u32, u32),
}

impl Controlled {
    fn load(sdxl_root: &str, control_root: &str, h: u32, w: u32) -> Result<Controlled, String> {
        let r = std::path::Path::new(sdxl_root);
        let scale = 8u32;
        if !h.is_multiple_of(scale) || !w.is_multiple_of(scale) {
            return Err(format!("controlnet: {w}x{h} is not a multiple of the VAE's {scale}x downscale"));
        }
        let (lh, lw) = (h / scale, w / scale);

        let gpu = Gpu::new(&sdxlunet::model::KERNELS);

        let ucfg = UNetConfig::sdxl_base();
        let udir = r.join("unet");
        let utensors = sdxlunet::import::load(udir.to_str().ok_or("controlnet: non-UTF8 unet path")?, &ucfg)?;
        let unet = Unet::new_controlled(gpu.share(), ucfg.clone(), &utensors, lh, lw, CONTEXT as u32, false, true);

        let ccfg = ControlNetConfig::sdxl();
        let ctensors = crate::import::load(control_root, &ccfg)?;
        let control = ControlNet::new(gpu.share(), ccfg, &ctensors, lh, lw, CONTEXT as u32, false);

        let vae_cfg = VaeConfig::from_json(&read_json(&r.join("vae/config.json"))?);

        Ok(Controlled { gpu, sdxl_root: sdxl_root.into(), unet, control, vae_cfg, ucfg, hw: (h, w) })
    }

    /// `cond_chw` is `[3, h, w]`, `[0,1]` - already at this pipeline's `(h, w)`.
    fn generate(&self, req: &Req, cond_chw: &[f32]) -> Result<Vec<f32>, String> {
        let (h, w) = self.hw;
        let (lh, lw) = (h / 8, w / 8);
        let n = (self.ucfg.in_channels * lh * lw) as usize;

        let te = TextEncoders::load(self.gpu.share(), &self.sdxl_root)?;
        let do_cfg = req.guidance > 1.0;
        let mut enc =
            if do_cfg { te.encode_all(&[req.prompt.as_str(), req.negative.as_str()])? } else { te.encode_all(&[req.prompt.as_str()])? };
        let uncond = do_cfg.then(|| enc.pop().expect("negative encoded"));
        let cond = enc.pop().expect("prompt encoded");

        let denoiser = ControlledDenoiser { unet: &self.unet, control: &self.control, cond_chw, conditioning_scale: req.conditioning_scale };
        let so = SamplerOptions { steps: req.steps, guidance: req.guidance, seed: req.seed, height: h, width: w };
        let lat = sample(&denoiser, n, &cond, uncond.as_ref(), &so)?;

        let sf = self.vae_cfg.scaling_factor;
        let z: Vec<f32> = lat.iter().map(|v| v / sf).collect();
        let vt = read_any_safetensors(&std::path::Path::new(&self.sdxl_root).join("vae"))?;
        let vmap: vae::blocks::Tensors = vt.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
        let vdev = std::env::var("BRAIN_SDXL_VAE_DEVICE").unwrap_or_else(|_| "cpu".into());
        let vae = VaeDecoder::from_diffusers(self.vae_cfg.clone(), &vmap, lh, lw, Some(&vdev));
        let chw = vae.decode(&z);
        let rgb: Vec<f32> = chw.iter().map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0)).collect();
        Ok(imaging::pixels::chw_to_hwc(&rgb, 3, h as usize, w as usize))
    }
}

/// The controlled SDXL forward as a [`Denoiser`]: a [`ControlNet::run`] ahead
/// of `Unet::run_with_control`, the residuals ordered by [`order_for`].
struct ControlledDenoiser<'a> {
    unet: &'a Unet,
    control: &'a ControlNet,
    cond_chw: &'a [f32],
    conditioning_scale: f32,
}

impl Denoiser for ControlledDenoiser<'_> {
    fn eval(&self, ctx: &StepCtx<'_>, enc: &[f32], pooled: &[f32], time_ids: &[f32]) -> Result<Vec<f32>, String> {
        let cres = self.control.run(ctx.scaled, ctx.timestep, enc, pooled, time_ids, self.cond_chw, self.conditioning_scale);
        let ordered = order_for(self.unet, &cres)?;
        Ok(self.unet.run_with_control(ctx.scaled, ctx.timestep, enc, pooled, time_ids, &ordered))
    }
}

// ===================== the shared work =====================

/// The pipelines on one device, keyed by `(h, w)` - shared by
/// [`ControlnetProvider`] and the residency adapter
/// (`crates/cli/src/resident_controlnet.rs`).
pub struct Session {
    sdxl_root: String,
    control_root: String,
    built: Mutex<std::collections::HashMap<(u32, u32), Controlled>>,
}

impl Session {
    pub fn new(sdxl_root: impl Into<String>, control_root: impl Into<String>) -> Session {
        Session { sdxl_root: sdxl_root.into(), control_root: control_root.into(), built: Mutex::new(std::collections::HashMap::new()) }
    }

    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "text2image" => self.text2image(inv),
            other => Err(format!("controlnet: unknown action '{other}'")),
        }
    }

    fn text2image(&self, inv: &Invocation) -> ActionResult {
        let req = req_from(inv);
        let (h, w) = (req.height, req.width);
        let (hwc, cw, ch, c) = capability::blob::decode_hwc(inv, "control_image")?;
        if c != 3 {
            return Err(format!("controlnet: control_image must be RGB (3 channels), got {c}"));
        }
        let chw_src = imaging::pixels::hwc_to_chw(&hwc, 3, ch as usize, cw as usize);

        let mut guard = self.built.lock().map_err(|_| "controlnet: pipeline lock poisoned")?;
        let p = match guard.entry((h, w)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Controlled::load(&self.sdxl_root, &self.control_root, h, w)?)
            }
        };

        let cond_chw = if (cw, ch) == (w, h) {
            chw_src
        } else {
            // Resize on the device to exactly the pipeline's (h, w) - a caller
            // should not have to pre-size the conditioning image to match. Uses
            // the pipeline's own GPU handle rather than a throwaway one - one
            // device per process, the same rule every other resident follows.
            let ctx = imaging::Ctx::new(&p.gpu);
            let src = ctx.upload("controlnet.cond", &chw_src);
            let (dst, _) = ctx.resize(
                &src,
                imaging::Shape::new(1, 3, ch, cw),
                w,
                h,
                imaging::Filter::Bilinear,
                imaging::AlignCorners::HalfPixel,
            );
            ctx.download(&dst, 3 * h * w)
        };

        let hwc_out = p.generate(&req, &cond_chw)?;
        Ok(Outcome::new().blob("image", capability::blob::image_blob(&hwc_out, w, h, 3)))
    }
}

// ===================== the provider =====================

type HotSession = Arc<Mutex<Option<((String, String), Arc<Session>)>>>;

/// The executable SDXL+ControlNet stack behind the manifest.
pub struct ControlnetProvider {
    sdxl_root: String,
    control_root: String,
    hot: HotSession,
}

impl ControlnetProvider {
    pub fn new(sdxl_root: impl Into<String>, control_root: impl Into<String>) -> ControlnetProvider {
        ControlnetProvider { sdxl_root: sdxl_root.into(), control_root: control_root.into(), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_SDXL_DIR` (the backbone) + `BRAIN_CONTROLNET_DIR` (the control
    /// model) - `None` unless both are set and the backbone directory holds a
    /// released `unet/`, since without either no action can run.
    pub fn from_env() -> Option<ControlnetProvider> {
        let sdxl_root = std::env::var("BRAIN_SDXL_DIR").ok().filter(|p| !p.is_empty())?;
        let control_root = std::env::var("BRAIN_CONTROLNET_DIR").ok().filter(|p| !p.is_empty())?;
        std::path::Path::new(&sdxl_root).join("unet").exists().then(|| ControlnetProvider::new(sdxl_root, control_root))
    }
}

impl Provider for ControlnetProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "text2image").then(|| {
            Arc::new(ControlnetAction { sdxl_root: self.sdxl_root.clone(), control_root: self.control_root.clone(), hot: self.hot.clone() })
                as Arc<dyn Action>
        })
    }
}

struct ControlnetAction {
    sdxl_root: String,
    control_root: String,
    hot: HotSession,
}

impl Action for ControlnetAction {
    fn spec(&self) -> ActionSpec {
        text2image_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let session = {
            let mut guard = self.hot.lock().map_err(|_| "controlnet: hot session lock poisoned")?;
            let key = (self.sdxl_root.clone(), self.control_root.clone());
            if !matches!(&*guard, Some((k, _)) if *k == key) {
                *guard = None;
                *guard = Some((key.clone(), Arc::new(Session::new(key.0.clone(), key.1.clone()))));
            }
            guard.as_ref().expect("built above").1.clone()
        };
        session.run("text2image", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;

    #[test]
    fn manifest_declares_text2image() {
        let m = manifest();
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["text2image"]);
    }

    #[test]
    fn an_unknown_action_is_named_not_ignored() {
        let p = ControlnetProvider::new("/nonexistent", "/nonexistent");
        assert!(p.action("edit").is_none());
    }

    #[test]
    fn from_env_declines_without_both_directories() {
        assert!(
            ControlnetProvider::from_env().is_none()
                || (std::env::var("BRAIN_SDXL_DIR").is_ok() && std::env::var("BRAIN_CONTROLNET_DIR").is_ok())
        );
    }

    #[test]
    fn conditioning_scale_carries_ui_range() {
        let spec = text2image_spec();
        let p = spec.params.iter().find(|p| p.name == "conditioning_scale").expect("param");
        assert_eq!(p.min, Some(0.0));
        assert_eq!(p.max, Some(2.0));
    }
}
