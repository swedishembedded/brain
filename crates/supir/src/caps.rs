// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR behind the generalized [`capability`] interface - what makes
//! `brain caps brain/supir` / `brain do brain/supir restore ...`, the D-Bus
//! `Run` method and `brain perf`'s `CapabilityTarget` work with no
//! SUPIR-specific plumbing in the CLI or the transports.
//!
//! One action: **`restore`** - a degraded image in, a restored one out
//! (`crate::pipeline::Restorer::restore`; see that module's docs for the
//! dual-encode / dual-CLIP / `RestoreEDMSampler` / colour-fix pipeline this
//! wraps). Image I/O goes through `capability::blob` exactly like every other
//! image model in the tree (`sdxlunet::caps`, `llava::caps`) - never a local
//! codec.
//!
//! # Cancellable per denoise step
//!
//! `inv.cancel` rides down into [`crate::pipeline::Restorer::restore`], which
//! polls it once per step of the sampler loop - the same
//! `inv.cancel.is_cancelled()` contract `wan::caps`/`crate::pipeline`'s own
//! module doc describes. A 50-step restoration at native resolution is not a
//! sub-second call, so it must be abortable.
//!
//! # LLaVA auto-captioning goes through a [`capability::Registry`], not a dependency
//!
//! Upstream SUPIR optionally captions the LQ image with LLaVA before
//! building the prompt (`--no_llava` turns it off). This crate links no VLM:
//! when the `caption` param is empty, [`RestoreAction::run`] calls
//! [`LLAVA_MODEL`]'s `caption` action through an OPTIONAL
//! [`capability::Registry`] the caller supplies (mirrors `crates/imgpipe`'s
//! own "the registry is supplied by the caller" precedent - see that crate's
//! `Pipeline::new`). No registry, or a registry with no `LLAVA_MODEL` entry,
//! means the caption stays empty - upstream's own `--no_llava` path, not an
//! error.
//!
//! # `run_batch` stays the serial default
//!
//! `crate::pipeline::Restorer::restore` is a full multi-step sample per call,
//! exactly the shape `sdxlunet::caps`/`controlnet::caps` document - there is
//! no `[B, ...]` axis a residency-level grouping could fill. The residency
//! adapter (`crates/cli/src/resident_supir.rs`) uses the serial default and
//! says so, the same way `resident_sdxl.rs`/`resident_controlnet.rs` do.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider, Registry};
use serde_json::json;

use crate::pipeline::{target_size, Paths, RestoreOptions, Restorer};

/// The model id used on the CLI (`brain do brain/supir restore ...`), over
/// D-Bus and in the residency manifest.
pub const MODEL: &str = "brain/supir";

/// The catalog id the optional auto-caption call dispatches to, when a
/// [`Registry`] carrying it is supplied - a plain string, not
/// `llava::caps::MODEL`, for the same reason `crates/imgpipe`'s stage ids are
/// strings: this crate links no VLM. `crates/cli`'s own catalog tests cross-
/// check it against the real constant (the `imgpipe_stage_ids_match_the_catalog`
/// precedent).
pub const LLAVA_MODEL: &str = "brain/llava";

fn restore_spec() -> ActionSpec {
    let d = RestoreOptions::default();
    ActionSpec::new(
        "restore",
        "photo-realistic blind image restoration: a frozen SDXL 1.0 base UNet, a 1.24B GLVControl trunk and 12 ZeroSFT/ZeroCrossAttn adaptors, RestoreEDMSampler",
    )
    .streaming()
    .param(ParamSpec::new("caption", ParamType::Str, "image caption; empty auto-captions via a registered brain/llava when one is available, else stays empty (upstream's --no_llava path)"))
    .param(ParamSpec::new("positive_suffix", ParamType::Str, "appended to the caption with no separator").default(json!(d.positive_suffix)))
    .param(ParamSpec::new("negative_prompt", ParamType::Str, "the negative prompt, used alone").default(json!(d.negative_prompt)))
    .param(ParamSpec::new("steps", ParamType::Int, "edm_steps: denoising steps").default(json!(d.steps as i64)).min(1.0).max(200.0).step(1.0))
    .param(ParamSpec::new("cfg_scale", ParamType::Float, "s_cfg: classifier-free guidance scale at sigma -> 0").default(json!(d.cfg_scale)).min(1.0).max(30.0).step(0.1))
    .param(ParamSpec::new("spt_linear_cfg", ParamType::Float, "CFG scale at sigma_max (LinearCFG's other endpoint)").default(json!(d.spt_linear_cfg)).min(1.0).max(30.0).step(0.1))
    .param(ParamSpec::new("control_scale", ParamType::Float, "s_stage2: the control trunk's contribution").default(json!(d.control_scale)).min(0.0).max(2.0).step(0.05))
    .param(ParamSpec::new("s_churn", ParamType::Float, "stochastic churn strength").default(json!(d.s_churn)).min(0.0).max(40.0))
    .param(ParamSpec::new("s_noise", ParamType::Float, "churn noise scale").default(json!(d.s_noise)).min(0.5).max(1.5))
    .param(ParamSpec::new("restore_cfg", ParamType::Float, "s_stage1: restoration guidance strength toward the clean re-encode; negative is OFF (upstream's own default)").default(json!(d.restore_cfg)))
    .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for 0)"))
    .input(capability::BlobSpec::new("image", capability::Media::Image, "the degraded (LQ) image, HWC f32 in [0,1]").required())
    .output(capability::BlobSpec::new("image", capability::Media::Image, "the restored image"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "SUPIR photo-realistic blind image restoration (SDXL + GLVControl + ZeroSFT/ZeroCrossAttn).", vec![restore_spec()])
}

fn opts_from(inv: &Invocation, caption: String) -> RestoreOptions {
    let d = RestoreOptions::default();
    RestoreOptions {
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        cfg_scale: inv.get_f64("cfg_scale").unwrap_or(d.cfg_scale as f64) as f32,
        spt_linear_cfg: inv.get_f64("spt_linear_cfg").unwrap_or(d.spt_linear_cfg as f64) as f32,
        control_scale: inv.get_f64("control_scale").unwrap_or(d.control_scale as f64) as f32,
        s_churn: inv.get_f64("s_churn").unwrap_or(d.s_churn as f64) as f32,
        s_noise: inv.get_f64("s_noise").unwrap_or(d.s_noise as f64) as f32,
        restore_cfg: inv.get_f64("restore_cfg").unwrap_or(d.restore_cfg as f64) as f32,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        caption,
        positive_suffix: inv.get_str("positive_suffix").unwrap_or(d.positive_suffix),
        negative_prompt: inv.get_str("negative_prompt").unwrap_or(d.negative_prompt),
    }
}

/// Resolve the `caption` param: if the caller supplied one, use it verbatim;
/// otherwise, when `registry` carries [`LLAVA_MODEL`], run its `caption`
/// action on the same image and use that; otherwise stay empty (upstream's
/// `--no_llava`).
fn resolve_caption(inv: &Invocation, image: &capability::Blob, registry: Option<&Registry>) -> Result<String, String> {
    if let Some(c) = inv.get_str("caption").filter(|s| !s.is_empty()) {
        return Ok(c);
    }
    let Some(reg) = registry else { return Ok(String::new()) };
    if reg.provider(LLAVA_MODEL).is_none() {
        return Ok(String::new());
    }
    let cap_inv = Invocation::new().blob("image", image.clone());
    let out = reg.run(LLAVA_MODEL, "caption", cap_inv, &mut |_| {})?;
    Ok(out.outputs.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string())
}

// ===================== the shared work =====================

/// The restoration graphs on one device, keyed by `(pixel h, pixel w,
/// control_scale bits)` - `control_scale` is baked into
/// [`crate::model::Supir`]'s graph (see [`crate::pipeline`]'s module doc), so
/// a distinct value is a distinct cache entry, same as size.
pub struct Session {
    backbone_root: String,
    supir_ckpt: String,
    built: Mutex<std::collections::HashMap<(u32, u32, u32), Arc<Restorer>>>,
}

impl Session {
    pub fn new(backbone_root: impl Into<String>, supir_ckpt: impl Into<String>) -> Session {
        Session { backbone_root: backbone_root.into(), supir_ckpt: supir_ckpt.into(), built: Mutex::new(std::collections::HashMap::new()) }
    }

    pub fn run(&self, action: &str, inv: &Invocation, registry: Option<&Registry>, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "restore" => self.restore(inv, registry, progress),
            other => Err(format!("supir: unknown action '{other}'")),
        }
    }

    fn restore(&self, inv: &Invocation, registry: Option<&Registry>, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (px, lq_w, lq_h) = capability::blob::decode_image(inv, "image")?;
        let control_scale = inv.get_f64("control_scale").unwrap_or(RestoreOptions::default().control_scale as f64) as f32;
        let caption = resolve_caption(inv, inv.blobs.get("image").ok_or("supir restore: 'image' blob missing")?, registry)?;
        let o = opts_from(inv, caption);

        let (h, w) = target_size(lq_w, lq_h);
        let key = (h, w, control_scale.to_bits());
        let restorer = {
            let mut guard = self.built.lock().map_err(|_| "supir: session lock poisoned")?;
            match guard.get(&key) {
                Some(r) => r.clone(),
                None => {
                    let r = Arc::new(Restorer::load(&self.backbone_root, &self.supir_ckpt, h, w, control_scale)?);
                    guard.insert(key, r.clone());
                    r
                }
            }
        };

        let cancel = inv.cancel.clone();
        let total = o.steps as u32;
        let out_hwc = restorer.restore(&px, lq_w, lq_h, &o, &cancel, &mut |step, steps| progress(Progress::step(step, steps.max(total), "denoising")))?;
        Ok(Outcome::new().set("width", json!(w)).set("height", json!(h)).blob("image", capability::blob::image_blob(&out_hwc, w, h, 3)))
    }
}

// ===================== the provider =====================

/// The executable SUPIR stack behind the manifest. Construction is free -
/// the restoration graph imports lazily on first use, per requested size.
pub struct RestoreProvider {
    session: Arc<Session>,
    registry: Option<Arc<Registry>>,
}

impl RestoreProvider {
    pub fn new(backbone_root: impl Into<String>, supir_ckpt: impl Into<String>) -> RestoreProvider {
        RestoreProvider { session: Arc::new(Session::new(backbone_root, supir_ckpt)), registry: None }
    }

    /// [`RestoreProvider::new`], with a [`Registry`] the `caption` auto-fill
    /// can dispatch [`LLAVA_MODEL`] through - see this module's doc.
    pub fn with_registry(backbone_root: impl Into<String>, supir_ckpt: impl Into<String>, registry: Arc<Registry>) -> RestoreProvider {
        RestoreProvider { session: Arc::new(Session::new(backbone_root, supir_ckpt)), registry: Some(registry) }
    }

    /// `BRAIN_SDXL_DIR` + `BRAIN_SUPIR_DIR` - `None` when either is unset or
    /// the backbone directory holds no released `unet/`, since without one no
    /// action can run.
    pub fn from_env() -> Option<RestoreProvider> {
        let paths = Paths::from_env().ok()?;
        std::path::Path::new(&paths.backbone_root).join("unet").exists().then(|| RestoreProvider::new(paths.backbone_root, paths.supir_ckpt))
    }
}

impl Provider for RestoreProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "restore").then(|| Arc::new(RestoreAction { session: self.session.clone(), registry: self.registry.clone() }) as Arc<dyn Action>)
    }
}

struct RestoreAction {
    session: Arc<Session>,
    registry: Option<Arc<Registry>>,
}

impl Action for RestoreAction {
    fn spec(&self) -> ActionSpec {
        restore_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        self.session.run("restore", inv, self.registry.as_deref(), progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, CancelToken, Media};

    #[test]
    fn manifest_validates_without_weights() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let a = &m.actions[0];
        assert_eq!(a.name, "restore");
        assert!(a.streaming, "a 50-step sample must report progress");
        assert!(a.inputs.iter().any(|b| b.name == "image"));
        assert!(a.outputs.iter().any(|b| b.name == "image"));
        // The advertised defaults must match upstream's own CLI defaults
        // (`diffusion::restore::RestoreEDMSamplerConfig::default`), including
        // the trap: restoration guidance off.
        let def = |name: &str| a.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no param {name}")).default.clone();
        assert_eq!(def("steps"), Some(json!(50)));
        assert_eq!(def("cfg_scale"), Some(json!(4.0)));
        assert_eq!(def("spt_linear_cfg"), Some(json!(1.0)));
        assert_eq!(def("control_scale"), Some(json!(1.0)));
        assert_eq!(def("restore_cfg"), Some(json!(-1.0)));
        let j = m.to_json();
        assert_eq!(j["model"], MODEL);
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let session = Session::new("/nonexistent/sdxl", "/nonexistent/supir.safetensors");
        let inv = Invocation::new().blob("image", Blob::new(Media::Image, vec![0u8; 3 * 4 * 4 * 4]).with_meta(json!({"w": 4, "h": 4})));
        let r = session.run("restore", &inv, None, &mut |_| {});
        let err = r.err().unwrap_or_default();
        assert!(!err.is_empty());
    }

    #[test]
    fn an_unknown_action_is_named_not_ignored() {
        let session = Session::new("/nonexistent", "/nonexistent");
        let inv = Invocation::new();
        let err = session.run("edit", &inv, None, &mut |_| {}).unwrap_err();
        assert!(err.contains("edit"), "{err}");
    }

    #[test]
    fn from_env_declines_a_directory_with_no_unet() {
        assert!(RestoreProvider::from_env().is_none() || std::env::var("BRAIN_SDXL_DIR").is_ok());
    }

    /// An unarmed cancel token never fires; an armed, cancelled one aborts
    /// before touching a single weight - the same contract `wan::caps` pins.
    #[test]
    fn a_cancelled_invocation_is_refused_without_touching_the_weights() {
        let provider = RestoreProvider::new("/nonexistent/sdxl", "/nonexistent/supir.safetensors");
        let action = provider.action("restore").expect("restore is advertised");
        let mut inv = Invocation::new().blob("image", Blob::new(Media::Image, vec![0u8; 3 * 4 * 4 * 4]).with_meta(json!({"w": 4, "h": 4})));
        inv.cancel = CancelToken::armed();
        inv.cancel.cancel();
        let err = action.run(&inv, &mut |_| {}).unwrap_err();
        assert_eq!(err, "cancelled");
    }

    /// No `caption` param and no registry: the caption stays empty rather
    /// than erroring - upstream's own `--no_llava` path.
    #[test]
    fn resolve_caption_with_no_registry_and_no_param_is_empty() {
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1}));
        let inv = Invocation::new().blob("image", img.clone());
        assert_eq!(resolve_caption(&inv, &img, None).unwrap(), "");
    }

    /// An explicit `caption` param wins even when a registry is supplied -
    /// auto-captioning only fills a GAP, it never overrides a caller's own text.
    #[test]
    fn resolve_caption_prefers_an_explicit_param() {
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1}));
        let inv = Invocation::new().set("caption", json!("a red bicycle")).blob("image", img.clone());
        assert_eq!(resolve_caption(&inv, &img, None).unwrap(), "a red bicycle");
    }

    /// A registry with no `brain/llava` entry is the same as no registry -
    /// caption stays empty, not an error naming a model this crate never asked for.
    #[test]
    fn resolve_caption_with_an_unrelated_registry_is_still_empty() {
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1}));
        let inv = Invocation::new().blob("image", img.clone());
        let reg = Registry::new();
        assert_eq!(resolve_caption(&inv, &img, Some(&reg)).unwrap(), "");
    }

    #[test]
    fn size_params_carry_ui_ranges() {
        let spec = restore_spec();
        let steps = spec.params.iter().find(|p| p.name == "steps").expect("steps param");
        assert_eq!(steps.min, Some(1.0));
        assert_eq!(steps.max, Some(200.0));
    }
}
