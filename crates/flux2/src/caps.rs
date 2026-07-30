// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein's capabilities, declared through the generalized
//! [`capability`] interface — what makes `brain caps flux2-klein`, `brain do
//! flux2-klein <action> …`, the event API, and the D-Bus surface work with no
//! FLUX.2-specific plumbing in the CLI or runtime.
//!
//! The manifest is **static** (no weights needed) so capability *discovery* is
//! free; only [`Flux2Provider`] (execution) loads the model. Actions:
//! text-to-image, reference-image editing, and LoRA personalisation.
//!
//! The **9B variants are NC-licensed** (FLUX.2 \[Non-Commercial\] License,
//! Black Forest Labs — see `docs/models/flux2/readme.md`): they refuse to run
//! unless `BRAIN_FLUX2_ALLOW_NC=1`, and print the attribution notice once when
//! enabled.

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

use crate::config::Flux2Config;
use crate::pipeline::{GenOpts, Paths, Pipeline};

/// The model id used on the CLI (`brain do flux2-klein …`) and the event API.
pub const MODEL: &str = "flux2-klein";

/// The variant enum, in manifest order.
const VARIANTS: [&str; 4] = ["klein-4b", "klein-9b", "base-4b", "base-9b"];

/// The DiT numeric-tier enum ([`crate::Precision`] names, fp32 first).
const PRECISIONS: [&str; 2] = ["fp32", "int8"];

/// Optional extra reference-image blob names accepted by `edit` (the primary
/// is `image`; the manifest must declare every accepted name — validation
/// rejects undeclared blobs).
const EXTRA_REFS: [&str; 3] = ["image0", "image1", "image2"];

/// Shared generation params (size / steps / seed / guidance / variant / adapter).
fn gen_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 16)").default(json!(512)))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 16)").default(json!(512)))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps; 0 = variant default (4 distilled / 50 base)").default(json!(0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for 0)"))
        .param(ParamSpec::new("guidance", ParamType::Float, "CFG scale — base variants only (klein is guidance-distilled)").default(json!(4.0)))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "model variant; 9B needs BRAIN_FLUX2_ALLOW_NC=1 (FLUX Non-Commercial license)").default(json!("klein-4b")))
        .param(ParamSpec::new("precision", ParamType::Enum(PRECISIONS.iter().map(|s| s.to_string()).collect()), "DiT numeric tier: fp32 (parity reference) or int8 (DP4A, ~4x smaller weights; GPU only)").default(json!("fp32")))
        .param(ParamSpec::new("adapter", ParamType::Str, "server-side path to a trained LoRA adapter (from lora_train) to apply"))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let image_out = || BlobSpec::new("image", Media::Image, "the generated image");

    let text2image = gen_params(ActionSpec::new("text2image", "generate an image from a text prompt (4-step distilled rectified flow; base variants add CFG)").streaming())
        .output(image_out());

    let mut edit = gen_params(ActionSpec::new("edit", "regenerate toward a prompt conditioned on reference image(s) (token-concatenation editing)").streaming())
        .input(BlobSpec::new("image", Media::Image, "the reference image to edit (center-cropped to /16)").required());
    for r in EXTRA_REFS {
        edit = edit.input(BlobSpec::new(r, Media::Image, "additional reference image"));
    }
    let edit = edit.output(image_out());

    let lora_train = ActionSpec::new("lora_train", "fine-tune a LoRA adapter on a folder of captioned images (personalise a person/object/style; host f32 trainer)")
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "server-side folder with images + captions (see data::imageset)").required())
        .param(ParamSpec::new("save", ParamType::Str, "server-side output path for the trained adapter").required())
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").default(json!(16)))
        .param(ParamSpec::new("steps", ParamType::Int, "training steps").default(json!(200)))
        .param(ParamSpec::new("size", ParamType::Int, "training square size, px (multiple of 16)").default(json!(512)))
        .param(ParamSpec::new("lr", ParamType::Float, "learning rate").default(json!(1e-4)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for 0)"))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "base model to adapt; 9B needs BRAIN_FLUX2_ALLOW_NC=1").default(json!("klein-4b")))
        .output(BlobSpec::new("adapter", Media::Bytes, "the trained LoRA adapter checkpoint"));

    Manifest::new(
        MODEL,
        "FLUX.2 Klein (Black Forest Labs) — MMDiT text-to-image + reference-image editing (4B/9B, distilled klein + undistilled base), with LoRA personalisation.",
        vec![text2image, edit, lora_train],
    )
}

// ===================== shared execution helpers =====================
//
// Both the hot-cache [`Flux2Provider`] and the residency adapter
// (`crates/cli/src/resident_flux2.rs`) run actions through these — ONE
// implementation of param decoding, license gating, and generation.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, Invocation, Outcome, Progress, Provider};

/// Decoded generation request: the variant config + name, [`GenOpts`], and an
/// optional LoRA adapter path. Validates sizes (/16) and the 9B license gate.
pub struct GenParams {
    pub cfg: Flux2Config,
    pub variant: String,
    pub opts: GenOpts,
    pub adapter: Option<String>,
    /// DiT numeric tier (fp32 default; int8 = DP4A, GPU only).
    pub precision: crate::Precision,
}

/// Decode + validate the shared generation params from an invocation.
pub fn gen_params_from(inv: &Invocation) -> Result<GenParams, String> {
    let variant = inv.get_str("variant").unwrap_or_else(|| "klein-4b".into());
    let cfg = Flux2Config::from_name(&variant)?;
    check_license(&variant)?;
    let precision = crate::Precision::from_name(
        &inv.get_str("precision").unwrap_or_else(|| "fp32".into()),
    )?;
    let width = inv.get_i64("width").unwrap_or(512).max(16) as u32;
    let height = inv.get_i64("height").unwrap_or(512).max(16) as u32;
    if width % 16 != 0 || height % 16 != 0 {
        return Err(format!("width/height must be multiples of 16 (got {width}×{height})"));
    }
    let steps = inv.get_i64("steps").unwrap_or(0).max(0) as u32;
    let opts = GenOpts {
        width,
        height,
        steps: (steps > 0).then_some(steps),
        guidance: inv.get_f64("guidance").unwrap_or(4.0) as f32,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
    };
    Ok(GenParams { cfg, variant, adapter: inv.get_str("adapter").filter(|s| !s.is_empty()), opts, precision })
}

/// The 9B weights are released under the FLUX.2 \[Non-Commercial\] License —
/// refuse them unless the operator opted in, and print the attribution notice
/// once per process when enabled.
pub fn check_license(variant: &str) -> Result<(), String> {
    if !variant.ends_with("9b") {
        return Ok(());
    }
    if std::env::var("BRAIN_FLUX2_ALLOW_NC").ok().as_deref() != Some("1") {
        return Err(format!(
            "variant '{variant}' uses the FLUX.2 9B weights, released under the FLUX.2 [Non-Commercial] License (Black Forest Labs). Set BRAIN_FLUX2_ALLOW_NC=1 to confirm non-commercial use."
        ));
    }
    static NOTICE: std::sync::Once = std::sync::Once::new();
    NOTICE.call_once(|| {
        eprintln!("flux2: 9B weights enabled — FLUX.2 [Non-Commercial] License © Black Forest Labs; non-commercial use only");
    });
    Ok(())
}

/// Collect reference images from an invocation (`image`, then `image0..`):
/// each decoded through the shared `capability::blob` codec and converted to
/// the `[-1,1]` CHW /16-cropped layout by [`crate::pipeline::ref_from_hwc`].
/// `require_primary` = the action needs `image` (edit); text2image passes
/// none.
pub fn refs_from(inv: &Invocation, require_primary: bool) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
    let mut refs = Vec::new();
    if inv.get_blob("image").is_some() {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        refs.push(crate::pipeline::ref_from_hwc(&hwc, w, h)?);
    } else if require_primary {
        return Err("edit: input blob 'image' is required".into());
    }
    for name in EXTRA_REFS {
        if inv.get_blob(name).is_some() {
            let (hwc, w, h) = capability::blob::decode_image(inv, name)?;
            refs.push(crate::pipeline::ref_from_hwc(&hwc, w, h)?);
        }
    }
    Ok(refs)
}

/// Total reference latent tokens for a set of pre-cropped refs.
pub fn ref_tokens(refs: &[(Vec<f32>, u32, u32)]) -> u32 {
    refs.iter().map(|(_, h, w)| (h / 16) * (w / 16)).sum()
}

/// Run one generation on a built pipeline and wrap the result as an
/// image-output [`Outcome`] (the shared `capability::blob` wire format).
/// Cancellation rides in `inv.cancel` — [`Pipeline::generate`] polls it per
/// denoise step.
pub fn generate_on(
    pipe: &Pipeline,
    inv: &Invocation,
    refs: &[(Vec<f32>, u32, u32)],
    opts: &GenOpts,
    progress: &mut dyn FnMut(Progress),
) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let (rgb, w, h) = pipe.generate(&prompt, refs, opts, &inv.cancel, |step, total, msg| {
        progress(Progress { step, total, message: msg.to_string() })
    })?;
    Ok(image_outcome(&rgb, w, h))
}

/// Wrap a generated RGB8 HWC image as an image-output [`Outcome`] (the shared
/// `capability::blob` wire format) — ONE implementation, shared by the
/// single-request path above and the batched
/// `resident_flux2::Flux2Instance::run_batch`.
pub fn image_outcome(rgb: &[u8], w: u32, h: u32) -> Outcome {
    let hwc: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
    Outcome::new()
        .set("width", json!(w))
        .set("height", json!(h))
        .blob("image", capability::blob::image_blob(&hwc, w, h, 3))
}

/// Run `lora_train` from an invocation: train via [`crate::finetune::run`]
/// (which polls `inv.cancel` every step), then return the trained artifact
/// itself as an output blob — a remote client has no access to the server's
/// filesystem (the post-hardening zimage pattern).
pub fn train_action(paths: &Paths, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let dir = inv.get_str("data").ok_or("lora_train: 'data' folder is required")?;
    let save = inv.get_str("save").ok_or("lora_train: 'save' path is required")?;
    let variant = inv.get_str("variant").unwrap_or_else(|| "klein-4b".into());
    let cfg = Flux2Config::from_name(&variant)?;
    check_license(&variant)?;
    let opts = crate::finetune::TrainOpts {
        steps: inv.get_i64("steps").unwrap_or(200).max(1) as u32,
        rank: inv.get_i64("rank").unwrap_or(16).max(1) as usize,
        lr: inv.get_f64("lr").unwrap_or(1e-4) as f32,
        size: inv.get_i64("size").unwrap_or(512).max(16) as u32,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        save_path: save.clone(),
        ckpt_every: 100,
    };
    let mut prog = |step: u32, total: u32, message: String| progress(Progress { step, total, message });
    let adapter = crate::finetune::run(&cfg, paths, std::path::Path::new(&dir), &opts, &inv.cancel, &mut prog)?;
    use capability::Blob;
    let bytes = std::fs::read(&save).map_err(|e| format!("read trained adapter '{save}': {e}"))?;
    Ok(Outcome::new()
        .set("adapter", json!(save))
        .set("steps", json!(opts.steps))
        .set("rank", json!(adapter.rank()))
        .blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"path": save}))))
}

// ===================== execution (hot-pipeline provider) =====================

/// Cache key for a resident pipeline: everything that fixes the built graphs —
/// (variant, precision, width, height, reference latent tokens) plus the
/// folded adapter.
type HotKey = (String, &'static str, u32, u32, u32, Option<String>);

/// The executable FLUX.2 model behind the manifest. Holds a **hot pipeline
/// cache** so a long-lived process (`brain run` / the event server) loads the
/// weights once per (variant, size, refs, adapter) and reuses them across
/// `ActionRequest`s. Weight paths come from the environment
/// (`BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}`).
pub struct Flux2Provider {
    hot: Arc<Mutex<Option<(HotKey, Pipeline)>>>,
}

impl Flux2Provider {
    pub fn new() -> Flux2Provider {
        Flux2Provider { hot: Arc::new(Mutex::new(None)) }
    }
}

impl Default for Flux2Provider {
    fn default() -> Self {
        Flux2Provider::new()
    }
}

impl Provider for Flux2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest()
            .actions
            .iter()
            .any(|a| a.name == name)
            .then(|| Arc::new(Flux2Action { name: name.to_string(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

/// One FLUX.2 action, dispatched through the shared helpers above.
struct Flux2Action {
    name: String,
    hot: Arc<Mutex<Option<(HotKey, Pipeline)>>>,
}

impl Action for Flux2Action {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.name.as_str() {
            "text2image" | "edit" => {
                // Params (incl. the 9B license gate) before the weights-env
                // check — a licensing refusal must not hide behind "not set".
                let p = gen_params_from(inv)?;
                let paths = Paths::from_env()?;
                let refs = refs_from(inv, self.name == "edit")?;
                let n_gen = (p.opts.height / 16) * (p.opts.width / 16);
                let n_ref = ref_tokens(&refs);
                let key: HotKey = (p.variant.clone(), p.precision.name(), p.opts.width, p.opts.height, n_ref, p.adapter.clone());

                let mut guard = self.hot.lock().map_err(|_| "hot pipeline lock poisoned")?;
                if !matches!(&*guard, Some((k, _)) if *k == key) {
                    *guard = None; // free the old resident weights before building new
                    progress(Progress { step: 0, total: 1, message: "loading weights (first call for this variant/size)".into() });
                    let pipe = Pipeline::build_with(&p.cfg, &paths, n_gen + n_ref, p.adapter.as_deref(), p.precision)?;
                    *guard = Some((key, pipe));
                }
                generate_on(&guard.as_ref().unwrap().1, inv, &refs, &p.opts, progress)
            }
            "lora_train" => {
                let variant = inv.get_str("variant").unwrap_or_else(|| "klein-4b".into());
                check_license(&variant)?;
                train_action(&Paths::from_env()?, inv, progress)
            }
            other => Err(format!("flux2-klein '{other}': unknown action")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["text2image", "edit", "lora_train"]);
        // text2image: prompt required, size defaults 512 (/16), steps 0 = variant
        // default, variant enum with klein-4b default, produces an image.
        let t2i = &m.actions[0];
        assert!(t2i.params.iter().any(|p| p.name == "prompt" && p.required));
        assert_eq!(t2i.params.iter().find(|p| p.name == "width").unwrap().default, Some(json!(512)));
        assert_eq!(t2i.params.iter().find(|p| p.name == "steps").unwrap().default, Some(json!(0)));
        let variant = t2i.params.iter().find(|p| p.name == "variant").unwrap();
        assert_eq!(variant.default, Some(json!("klein-4b")));
        assert!(matches!(&variant.ty, ParamType::Enum(v) if v == &VARIANTS.map(String::from).to_vec()));
        let precision = t2i.params.iter().find(|p| p.name == "precision").unwrap();
        assert_eq!(precision.default, Some(json!("fp32")));
        assert!(matches!(&precision.ty, ParamType::Enum(v) if v == &PRECISIONS.map(String::from).to_vec()));
        assert!(t2i.streaming);
        assert_eq!(t2i.outputs[0].media, Media::Image);
        // edit requires the primary reference image and accepts extra refs.
        let edit = m.actions.iter().find(|a| a.name == "edit").unwrap();
        assert!(edit.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        for r in EXTRA_REFS {
            assert!(edit.inputs.iter().any(|b| b.name == r && !b.required));
        }
        // lora_train declares the trained adapter as a retrievable output blob.
        let lt = m.actions.iter().find(|a| a.name == "lora_train").unwrap();
        assert!(lt.params.iter().any(|p| p.name == "data" && p.required));
        assert!(lt.params.iter().any(|p| p.name == "save" && p.required));
        assert!(lt.outputs.iter().any(|b| b.name == "adapter" && b.media == Media::Bytes));
        // the whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["actions"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn nine_b_variants_are_license_gated() {
        // The gate reads the env var per call; only assert the refusing path here
        // (setting env vars in tests races other tests in the binary).
        if std::env::var("BRAIN_FLUX2_ALLOW_NC").ok().as_deref() != Some("1") {
            let err = check_license("klein-9b").unwrap_err();
            assert!(err.contains("Non-Commercial"), "error must name the license: {err}");
            assert!(err.contains("BRAIN_FLUX2_ALLOW_NC"), "error must name the opt-in: {err}");
        }
        assert!(check_license("klein-4b").is_ok());
        assert!(check_license("base-4b").is_ok());
    }
}
