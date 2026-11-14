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
//! Black Forest Labs): a request refuses to run them
//! unless it carries `allow_non_commercial: true` (the CLI's
//! `--allow-non-commercial`), and the attribution notice prints once per
//! process when one does.

use capability::presentation::maturity;
use capability::ModelPresentation;
use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

use crate::config::Flux2Config;
use crate::pipeline::{Paths, Pipeline};

/// The model id used on the CLI (`brain do flux2-klein …`) and the event API.
pub const MODEL: &str = "brain/flux2-klein";

/// The variant enum, in manifest order.
const VARIANTS: [&str; 4] = ["klein-4b", "klein-9b", "base-4b", "base-9b"];

/// The DiT numeric-tier enum ([`crate::Precision`] names, fp32 first).
const PRECISIONS: [&str; 2] = ["fp32", "int8"];

/// The per-request confirmation of non-commercial use, declared by every
/// action that can load the 9B weights; off unless the request says so.
fn allow_non_commercial_param() -> ParamSpec {
    ParamSpec::new("allow_non_commercial", ParamType::Bool, "confirm your use of the FLUX.2 9B weights is non-commercial (FLUX.2 [Non-Commercial] License, Black Forest Labs); a 9B variant is refused without it").default(json!(false))
}

/// Shared generation params (size / steps / seed / guidance / variant / adapter).
fn gen_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 16); unset, 512 - or, for an edit that starts from its image (strength or mask), that image's").min(64.0).max(capability::bounds::SIDE_PX).step(16.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 16); unset, as width").min(64.0).max(capability::bounds::SIDE_PX).step(16.0))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps; 0 = variant default (4 distilled / 50 base)").default(json!(0)).min(0.0).max(crate::pipeline::MAX_STEPS as f64).step(1.0))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random; the outcome reports the one used)"))
        .param(ParamSpec::new("guidance", ParamType::Float, "CFG scale -- base variants only (klein is guidance-distilled)").default(json!(4.0)).min(0.0).max(30.0).step(0.1))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "model variant; 9B needs allow_non_commercial (FLUX Non-Commercial license)").default(json!("klein-4b")))
        .param(allow_non_commercial_param())
        // perf-number: 4 bytes per weight to 1 is the definition of int8, not a measured speedup
        .param(ParamSpec::new("precision", ParamType::Enum(PRECISIONS.iter().map(|s| s.to_string()).collect()), "DiT numeric tier: fp32 (parity reference) or int8 (DP4A, ~4x smaller weights; GPU only). Unset: the checkpoint's own - int8 for a GGUF DiT, else fp32"))
        .param(ParamSpec::new("adapter", ParamType::Str, model::adapter::ADAPTER_LIST_HELP).host_resolved())
        .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier (ComfyUI strength_model); 1.0 = the reference default").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
        .param(ParamSpec::new("experimental_steps", ParamType::Bool, "let `steps` override a distilled variant's fixed 4-step sampler (off: a distilled variant ignores `steps`)").default(json!(false)))
        .param(ParamSpec::new("tile_size", ParamType::Int, "denoise a large canvas in windows of this many px (multiple of 16; 0 or unset: one forward over the whole canvas). The DiT is built for one window, the VAE decodes the whole canvas").min(0.0).max(capability::bounds::SIDE_PX))
        .param(ParamSpec::new("tile_overlap", ParamType::Int, "px two neighbouring windows share; needs tile_size (default: the tiling's own)").min(0.0).max(capability::bounds::SIDE_PX))
        .param(ParamSpec::new("te_device", ParamType::Str, "pin the text encoder to a card: gpu<i> builds a truncated f32 shard there, gpu<i>:i8 the int8 one (unset: the automatic placement plan)"))
        .param(ParamSpec::new("no_stream", ParamType::Bool, "build a non-GGUF DiT from a whole fp32 tensor map instead of streaming each tensor (same bytes; refused for a .gguf DiT)").default(json!(false)))
        .param(ParamSpec::new("te_no_stream", ParamType::Bool, "import the text encoder as one whole fp32 map instead of streaming it per tensor (same bytes)").default(json!(false)))
        .param(ParamSpec::new("lora_fold", ParamType::Bool, "int8 only: bake an adapter into the weights and requantize instead of applying it as a runtime correction (the lossy deployment, for comparison)").default(json!(false)))
        .param(ParamSpec::new("vae_tiling", ParamType::Bool, "force the VAE encode/decode onto the tiled path (true) or the whole-image path (false); unset, the device's free memory decides"))
}

/// The build [`Tuning`] a generating request asks for; each unset param is the
/// automatic behaviour. A malformed `te_device` is refused by name.
pub fn tuning_of(inv: &Invocation) -> Result<crate::pipeline::Tuning, String> {
    Ok(crate::pipeline::Tuning {
        te_device: inv.get_str("te_device").filter(|s| !s.is_empty()).map(|s| crate::pipeline::parse_te_device(&s)).transpose()?,
        dit_whole_map: inv.get_bool("no_stream").unwrap_or(false),
        te_whole_map: inv.get_bool("te_no_stream").unwrap_or(false),
        lora_fold: inv.get_bool("lora_fold").unwrap_or(false),
        vae_tiling: inv.get_bool("vae_tiling"),
    })
}

/// The reference-image knobs `edit` adds.
fn ref_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("ref_size", ParamType::Int, "long edge, px, a reference is encoded at (0: its own size; unset: 512). The image an edit starts from, and a tiled run's anchor, are never bounded").min(0.0).max(capability::bounds::SIDE_PX))
        .param(ParamSpec::new("ref_resolution_scale", ParamType::Float, "the share of its own resolution the starting image conditions at (0..1): lower is cheaper and looser").min(0.0).max(1.0).step(0.05).default(json!(crate::pipeline::DEFAULT_REF_RESOLUTION_SCALE)))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let image_out = || BlobSpec::new("image", Media::Image, "the generated image");

    let text2image = gen_params(ActionSpec::new("text2image", "generate an image from a text prompt (4-step distilled rectified flow; base variants add CFG)").streaming().cancellable())
        .output(image_out());

    let edit = gen_params(ActionSpec::new("edit", "regenerate toward a prompt conditioned on reference image(s) (token-concatenation editing)").streaming().cancellable())
        // `strength` belongs to `edit` alone: it seeds the trajectory from the
        // FIRST input image's latent, so it is meaningless without one.
        .param(ParamSpec::new(
            "strength",
            ParamType::Float,
            "img2img init strength in [0,1] = the starting noise level (0: the image back through the VAE round trip): the first input image seeds the              trajectory instead of pure noise. Low = keep the source, high = redraw it.              Does NOT add colour, and does NOT turn conditioning off - every input image              conditions the model at every strength. Omit to start from pure noise.",
        ).min(0.0).max(1.0).step(0.01))
        .input(BlobSpec::new("image", Media::Image, "the reference image to edit (center-cropped to /16); also the img2img init when `strength` or `mask` is set").required())
        .input(BlobSpec::new("mask", Media::Mask, "inpaint: white regenerates, black keeps `image`; over the output canvas, at any resolution"));
    let mut edit = ref_params(edit);
    for r in crate::request::EXTRA_REFS {
        edit = edit.input(BlobSpec::new(r, Media::Image, "additional reference image"));
    }
    let edit = edit.output(image_out());

    let lora_train = ActionSpec::new("lora_train", "fine-tune a LoRA adapter on a folder of captioned images (personalise a person/object/style)")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("data", ParamType::Str, "server-side folder with images + captions (see data::imageset)").required().host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "server-side output path for the trained adapter").required().host_resolved())
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").min(0.0).max(capability::bounds::LORA_RANK).default(json!(16)))
        .param(ParamSpec::new("steps", ParamType::Int, "training steps").min(0.0).max(capability::bounds::TRAIN_STEPS).default(json!(200)))
        .param(ParamSpec::new("size", ParamType::Int, "training square size, px (multiple of 16)").min(0.0).max(capability::bounds::SIZE_PX).default(json!(512)))
        .param(ParamSpec::new("lr", ParamType::Float, "PEAK learning rate: a run warms up to it and cosine-decays away from it, so it is not the rate every step runs at").default(json!(1e-4)))
        .param(ParamSpec::new("warmup", ParamType::Int, "steps of linear learning-rate warmup; absent = none, which is what every mainstream FLUX LoRA trainer defaults to (a LoRA starts at B=0, so there is no early instability for a warmup to protect against). A function of the global step, so a resumed run continues the curve").max(capability::bounds::TRAIN_COUNT).min(0.0))
        .param(ParamSpec::new("min_lr", ParamType::Float, "the rate the cooldown lands on at the last step; absent = lr/10. The peak is held flat until the last fifth of the run and then cosine-cooled to this. Adam at a constant rate orbits the minimum at a radius set by the rate times the gradient noise - at batch size 1 that is wide, and the cooldown is what closes it. Set equal to lr for a flat rate").min(0.0))
        .param(ParamSpec::new("edit_weight", ParamType::Float, "region-aware flow loss for a PAIRED run: weight each token by 1 + this * its normalised |target - reference| change, rescaled to mean 1. 0 (default) is the plain mean; 2 is the published value. Only meaningful for spatially aligned pairs (declutter, removal, relight) - not for subject-driven ones").default(json!(0.0)).min(0.0).max(8.0))
        .param(ParamSpec::new("detail_weight", ParamType::Float, "detail-aware flow loss: weight each token by 1 + this * its normalised local detail, rescaled to mean 1. 0 (default) is off; 2 is the value to reach for on a SMOOTHING objective (remove clutter, remove noise), where the cheapest prediction under a uniform loss is the flattest one and a flat latent lifts the frozen VAE decoder's own 4-pixel cell structure into the picture as a grid. Needs no reference, so it applies to unpaired runs too").default(json!(0.0)).min(0.0).max(8.0))
        .param(ParamSpec::new("ref_dropout", ParamType::Float, "probability a PAIRED step trains with its reference tokens blanked (InstructPix2Pix's null image conditioning). 0 (default) never drops. Denies the adapter the 'just copy the reference' solution a near-identical pair makes available, at the cost of steps spent on a mode a guidance-distilled sampler never runs; 0.1 is the rate to reach for").default(json!(0.0)).min(0.0).max(1.0))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random; the outcome reports the one used)"))
        .param(ParamSpec::new("cards", ParamType::Int, "GPUs the device trainer spreads the block stack over (klein-9b's fp32 frozen base does not fit one 24 GiB card)").min(0.0).max(64.0).default(json!(1)))
        .param(ParamSpec::new("trainer", ParamType::Enum(vec!["device".into(), "host".into()]), "gradient implementation: 'device' runs the WGSL kernels with the frozen base on the card; 'host' is the finite-difference-gradchecked reference it is validated against").default(json!("device")))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "base model to adapt; 9B needs allow_non_commercial").default(json!("klein-4b")))
        .param(allow_non_commercial_param())
        .param(ParamSpec::new("method", ParamType::Enum(vec!["lora".into(), "rslora".into()]), "adapter scale: 'lora' is alpha/rank, 'rslora' is alpha/sqrt(rank) (rank-stabilized, does not suppress the update as rank grows)").default(json!("lora")))
        .param(ParamSpec::new("lr_ratio", ParamType::Float, "LoRA+ : B's effective learning rate is lr_ratio*lr; 1.0 (the default) is plain LoRA").default(json!(1.0)).min(1.0).max(64.0))
        .param(ParamSpec::new("freeze_a", ParamType::Bool, "LoRA-FA: freeze A at its random init, train only B").default(json!(false)))
        .param(ParamSpec::new("ckpt_every", ParamType::Int, "checkpoint the adapter every N steps (0 = only at the end); each write is atomic, so an interrupted one cannot damage the last good checkpoint").min(0.0).max(capability::bounds::TRAIN_COUNT).default(json!(100)))
        .param(ParamSpec::new("resume", ParamType::Bool, "continue from the adapter already at 'save' instead of starting over - a cancelled multi-hour run then costs only the steps since its last checkpoint").default(json!(false)))
        .param(ParamSpec::new("precision", ParamType::Enum(PRECISIONS.iter().map(|s| s.to_string()).collect()), "the frozen base's tier in the host trainer").default(json!("fp32")))
        .param(ParamSpec::new("train_base", ParamType::Enum(PRECISIONS.iter().map(|s| s.to_string()).collect()), "the device trainer's frozen base: fp32, or int8 (smaller, fits a 9B on one card); unset, the checkpoint's own"))
        .param(ParamSpec::new("cache_dir", ParamType::Str, "server-side directory caching the encoded dataset between runs (unset: none)").host_resolved())
        .param(ParamSpec::new("activations", ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.iter().map(|s| s.to_string()).collect()), "what the host trainer's backward keeps of the forward: 'keep' every block's activations or 'recompute' them from each block's input (less memory, one extra forward); the device trainer always rebuilds from block inputs; default: keep"))
        .output(BlobSpec::new("adapter", Media::Bytes, "the trained LoRA adapter checkpoint"));

    Manifest::new(
        MODEL,
        "FLUX.2 Klein (Black Forest Labs) — MMDiT text-to-image + reference-image editing (4B/9B, distilled klein + undistilled base), with LoRA personalisation.",
        vec![text2image, edit, lora_train],
    )
    .with_presentation(
        ModelPresentation::default()
            .display_name("FLUX.2 Klein")
            .vendor("Black Forest Labs")
            .family("FLUX.2")
            .version("Klein")
            .task_tags(&["image-generation", "image-editing", "lora-training"])
            .maturity(maturity::STABLE)
            .default_action("text2image")
            // The 9B weights are non-commercial; a caller deploying this
            // has to know before it runs, not after.
            .license("flux-2-nc"),
    )
}

// ===================== shared execution helpers =====================
//
// Both the hot-cache [`Flux2Provider`] and the residency adapter
// (`crates/catalog/src/resident_flux2.rs`) run actions through these - ONE
// implementation of param decoding, license gating, and generation.

use std::sync::Arc;

use capability::{hot::Hot, Action, ActionResult, Invocation, Outcome, Progress, Provider};

/// Combine a caller-STATED variant's klein/base FAMILY with the DiT at
/// `dit_path`'s ACTUAL, sniffed size ([`crate::sniff_dit_size`]), so the
/// returned variant string always names the real weights - the size half of a
/// request's `variant` is a claim, never a fact, and must not be trusted
/// as-is. This is what every [`check_license`] call site not already holding
/// a resident's own bound identity (`crates/catalog/src/resident_flux2.rs`) must
/// run its variant through FIRST: a real 9B checkpoint requested/labeled
/// "klein-4b" still comes back "klein-9b" here, so the license gate sees the
/// truth regardless of the mislabel.
///
/// Errors if `requested`'s family half is not `klein`/`base`, or if
/// `dit_path`'s own shapes match neither known size.
pub fn bind_variant(dit_path: &str, requested: &str) -> Result<String, String> {
    let family = requested.strip_suffix("-4b").or_else(|| requested.strip_suffix("-9b")).unwrap_or(requested);
    if family != "klein" && family != "base" {
        return Err(format!("flux2: unknown variant '{requested}' (klein-4b|klein-9b|base-4b|base-9b)"));
    }
    let size = crate::sniff_dit_size(dit_path)?;
    Ok(format!("{family}-{}", size.as_str()))
}

/// The 9B weights are released under the FLUX.2 \[Non-Commercial\] License:
/// refuse them unless the request confirmed non-commercial use
/// (`allow_non_commercial`, the CLI's `--allow-non-commercial`), and print the
/// attribution notice once per process when it did. `accepted` is that
/// confirmation as the request carried it, so each request answers for itself.
///
/// `variant` must be a TRUSTED variant string - a resident's own bound
/// identity, or the result of [`bind_variant`] - never a raw, caller-supplied
/// one: this function itself has no weights to check the claim against, so
/// passing it an unverified string reopens exactly the hole `bind_variant`
/// closes.
pub fn check_license(variant: &str, accepted: bool) -> Result<(), String> {
    if !variant.ends_with("9b") {
        return Ok(());
    }
    if !accepted {
        return Err(format!(
            "variant '{variant}' uses the FLUX.2 9B weights, released under the FLUX.2 [Non-Commercial] License (Black Forest Labs). Pass --allow-non-commercial (allow_non_commercial: true in a request) to confirm non-commercial use."
        ));
    }
    static NOTICE: std::sync::Once = std::sync::Once::new();
    NOTICE.call_once(|| {
        eprintln!("flux2: 9B weights enabled — FLUX.2 [Non-Commercial] License © Black Forest Labs; non-commercial use only");
    });
    Ok(())
}

/// A request's `allow_non_commercial`; absent means not confirmed.
pub fn non_commercial_accepted(inv: &Invocation) -> bool {
    inv.get_bool("allow_non_commercial").unwrap_or(false)
}

/// Run a planned generation on a built pipeline.
pub fn generate_on(pipe: &Pipeline, inv: &Invocation, plan: &crate::request::Plan, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let refs = plan.decode(inv)?;
    let (rgb, w, h) = pipe.generate(&plan.prompt, &refs, &plan.opts, &inv.cancel, |step, total, msg| progress(Progress::step(step, total, msg.to_string())))?;
    Ok(image_outcome(&rgb, w, h, plan.opts.seed))
}

/// Wrap a generated RGB8 HWC image, and the seed it was sampled with, as an
/// image-output [`Outcome`] (the shared `capability::blob` wire format): ONE
/// implementation, shared by the single-request path above and the batched
/// `resident_flux2::Flux2Instance::run_batch`.
pub fn image_outcome(rgb: &[u8], w: u32, h: u32, seed: u64) -> Outcome {
    let hwc: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
    Outcome::new()
        .set("width", json!(w))
        .set("height", json!(h))
        .set("seed", json!(seed))
        .blob("image", capability::blob::image_blob(&hwc, w, h, 3))
}

/// What `lora_train`'s `activations` names for the host trainer's backward; absent keeps every block's activations,
/// the trainer's own default.
fn saved_policy(inv: &Invocation) -> Result<model::saved::SavedPolicy, String> {
    match inv.get_str("activations") {
        Some(value) => model::saved::SavedPolicy::parse_activations(&value).map_err(|e| format!("lora_train: {e}")),
        None => Ok(model::saved::SavedPolicy::Keep),
    }
}

/// Run `lora_train` from an invocation: train via [`crate::finetune::run`]
/// (which polls `inv.cancel` every step), then return the trained artifact
/// itself as an output blob — a remote client has no access to the server's
/// filesystem (the post-hardening zimage pattern).
pub fn train_action(paths: &Paths, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let dir = inv.get_str("data").ok_or("lora_train: 'data' folder is required")?;
    let save = inv.get_str("save").ok_or("lora_train: 'save' path is required")?;
    let requested = inv.get_str("variant").unwrap_or_else(|| "klein-4b".into());
    // Bound against `paths.dit`'s own shapes, not trusted as stated - see
    // `bind_variant`'s doc: this is the ONE place a served `lora_train`
    // decides which weights it is actually about to train against.
    let variant = bind_variant(&paths.dit, &requested)?;
    let cfg = Flux2Config::from_name(&variant)?;
    check_license(&variant, non_commercial_accepted(inv))?;
    let opts = crate::finetune::TrainOpts {
        steps: inv.get_i64("steps").unwrap_or(200).max(1) as u32,
        rank: inv.get_i64("rank").unwrap_or(16).max(1) as usize,
        lr: inv.get_f64("lr").unwrap_or(1e-4) as f32,
        // The LR SCHEDULE's two knobs. Absent means the library default (no
        // warmup, the peak held until the last fifth, then cooled to a tenth
        // of it) rather than a number spelled out here: this is the second
        // place a `TrainOpts` is built, and defaults that live at the
        // construction sites are defaults that drift between them.
        warmup: inv.get_i64("warmup").map(|v| v.max(0) as u32),
        min_lr: inv.get_f64("min_lr").map(|v| v as f32),
        edit_weight: inv.get_f64("edit_weight").unwrap_or(0.0).max(0.0) as f32,
        detail_weight: inv.get_f64("detail_weight").unwrap_or(0.0).max(0.0) as f32,
        ref_dropout: inv.get_f64("ref_dropout").unwrap_or(0.0).clamp(0.0, 1.0) as f32,
        // Served runs default to the device trainer for the same reason the
        // CLI does: the host one is the oracle, not a production path. A
        // caller can still name it, and the choice is echoed in the log.
        trainer: crate::finetune::Trainer::from_name(&inv.get_str("trainer").unwrap_or_else(|| "device".into()))?,
        cards: inv.get_i64("cards").unwrap_or(1).max(1) as usize,
        // A served request is already placed by the residency layer.
        card: None,
        train_base: match inv.get_str("train_base").as_deref() {
            Some("int8") => Some(crate::devgrad::BasePrecision::Int8),
            Some("fp32") => Some(crate::devgrad::BasePrecision::F32),
            Some(other) => return Err(format!("lora_train: unknown train_base {other} (fp32|int8)")),
            None => None,
        },
        cache_dir: inv.get_str("cache_dir").filter(|d| !d.is_empty()).map(std::path::PathBuf::from),
        size: inv.get_i64("size").unwrap_or(512).max(16) as u32,
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        save_path: save.clone(),
        ckpt_every: inv.get_i64("ckpt_every").unwrap_or(100).max(0) as u32,
        // A served run is the one most likely to be cancelled and re-issued,
        // so the same request repeated continues rather than restarting.
        resume: inv.get_bool("resume").unwrap_or(false),
        rank_stabilized: inv.get_str("method").unwrap_or_else(|| "lora".into()) == "rslora",
        lr_ratio: inv.get_f64("lr_ratio").unwrap_or(1.0) as f32,
        freeze_a: inv.get_bool("freeze_a").unwrap_or(false),
        // The tier this adapter will be SERVED at, which is what selects the
        // text encoder the captions are embedded through. `fp32` is the
        // served pipeline's own default request; a `.gguf` DiT overrides it
        // to int8 the same way `generate` does.
        precision: match inv.get_str("precision").as_deref() {
            Some("int8") => crate::Precision::Int8,
            Some("fp32") | None => crate::Precision::F32,
            Some(other) => return Err(format!("lora_train: unknown precision {other} (fp32|int8)")),
        },
        saved: saved_policy(inv)?,
    };
    let mut prog = |step: u32, total: u32, message: String| progress(Progress::step(step, total, message));
    let adapter = crate::finetune::run(&cfg, paths, std::path::Path::new(&dir), &opts, &inv.cancel, &mut prog)?;
    use capability::Blob;
    let bytes = std::fs::read(&save).map_err(|e| format!("read trained adapter '{save}': {e}"))?;
    Ok(Outcome::new()
        .set("adapter", json!(save))
        .set("steps", json!(opts.steps))
        .set("rank", json!(adapter.rank()))
        .set("seed", json!(opts.seed))
        .blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"path": save}))))
}

// ===================== execution (hot-pipeline provider) =====================

/// What a built pipeline is fixed to: (variant, precision, tokens per
/// forward, canvas tokens, the adapters folded in, the build tuning).
type HotKey = (String, &'static str, u32, u32, Vec<crate::AdapterSpec>, crate::pipeline::Tuning);

/// The executable FLUX.2 model behind the manifest. Holds a **hot pipeline
/// cache** so a long-lived process (`brain run` / the event server) loads the
/// weights once per (variant, size, refs, adapter) and reuses them across
/// `ActionRequest`s. Weight paths are BOUND at construction (see
/// [`Flux2Provider::new`]) - never re-resolved per request.
pub struct Flux2Provider {
    hot: Arc<Hot<HotKey, Pipeline>>,
    paths: Paths,
}

impl Flux2Provider {
    pub fn new(paths: Paths) -> Flux2Provider {
        Flux2Provider { hot: Arc::new(Hot::new()), paths }
    }
}

/// The provider over a resolved [`capability::Assembly`] (its four roles).
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(Flux2Provider::new(Paths::from_assembly(assembly)?)))
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
            .then(|| Arc::new(Flux2Action { name: name.to_string(), hot: self.hot.clone(), paths: self.paths.clone() }) as Arc<dyn Action>)
    }
}

/// One FLUX.2 action, dispatched through the shared helpers above. `paths` is
/// the provider's own bound weights (see [`Flux2Provider::new`]) - never
/// re-resolved per request.
struct Flux2Action {
    name: String,
    hot: Arc<Hot<HotKey, Pipeline>>,
    paths: Paths,
}

impl Action for Flux2Action {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        match self.name.as_str() {
            "text2image" | "edit" => {
                // The plan before the license gate: a malformed request must
                // not hide behind a licensing message. `resolve` (not
                // `build_resolved`) because the variant and precision are the
                // cache key, needed before deciding whether to build at all;
                // it also corrects the precision a served `.gguf` DiT needs.
                let plan = crate::request::plan(inv, self.name == "edit")?;
                let paths = &self.paths;
                let (cfg, precision, variant) = crate::build::resolve(crate::build::VariantRequest::new(crate::build::VariantSource::Sniff(&plan.variant), non_commercial_accepted(inv)), paths, plan.precision, plan.precision_explicit)?;
                for note in &plan.notes {
                    tracing::info!(target: "flux2", "{note}");
                }
                let (fwd, canvas) = (plan.forward_tokens(), plan.canvas_tokens());
                let key: HotKey = (variant, precision.name(), fwd, canvas, plan.adapters.clone(), plan.tuning);
                // Both closures report through the same sink, so it is shared
                // rather than borrowed twice.
                let sink = std::cell::RefCell::new(progress);
                self.hot.try_get(
                    key,
                    || {
                        (**sink.borrow_mut())(Progress::step(0, 1, "loading weights (first call for this variant/size)"));
                        Pipeline::build_sized(&cfg, paths, crate::pipeline::Sizing { n_img_max: fwd, n_out_max: canvas, max_batch: 1 }, &plan.adapters, precision, &plan.tuning)
                    },
                    |pipe| generate_on(pipe, inv, &plan, &mut **sink.borrow_mut()),
                )
            }
            "lora_train" => train_action(&self.paths, inv, progress),
            other => Err(format!("flux2-klein '{other}': unknown action")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build knobs that used to be environment variables are params: unset
    /// is the automatic behaviour, a set one reaches the build [`Tuning`], and a
    /// malformed card pin is refused by name instead of silently ignored.
    #[test]
    fn build_params_reach_the_tuning() {
        use crate::pipeline::{parse_te_device, TeDevice, Tuning};
        assert_eq!(tuning_of(&Invocation::new()).unwrap(), Tuning::default());
        let inv = Invocation::new().set("te_device", json!("gpu1:i8")).set("no_stream", json!(true)).set("te_no_stream", json!(true)).set("lora_fold", json!(true)).set("vae_tiling", json!(false));
        let want = Tuning { te_device: Some(TeDevice { gpu_index: 1, int8: true }), dit_whole_map: true, te_whole_map: true, lora_fold: true, vae_tiling: Some(false) };
        assert_eq!(tuning_of(&inv).unwrap(), want);
        assert_eq!(parse_te_device("gpu0").unwrap(), TeDevice { gpu_index: 0, int8: false });
        for bad in ["cpu", "gpu", "gpux", "gpu1:i4"] {
            assert!(tuning_of(&Invocation::new().set("te_device", json!(bad))).unwrap_err().contains("te_device"), "{bad}");
        }
        // The key a resident names an instance by round-trips every field and has no ':'.
        for t in [Tuning::default(), want] {
            assert!(!t.key().contains(':'));
            assert_eq!(Tuning::from_key(&t.key()), Some(t));
        }
    }

    #[test]
    fn lora_train_declares_activations_and_routes_the_choice_to_the_trainer() {
        use model::saved::SavedPolicy;
        let lt = manifest().actions.into_iter().find(|a| a.name == "lora_train").unwrap();
        let p = lt.params.iter().find(|p| p.name == "activations").expect("lora_train declares activations");
        assert_eq!(p.ty, ParamType::Enum(vec!["keep".into(), "recompute".into()]));
        assert!(!p.required && p.default.is_none());
        assert_eq!(saved_policy(&Invocation::new()), Ok(SavedPolicy::Keep));
        assert_eq!(saved_policy(&Invocation::new().set("activations", json!("keep"))), Ok(SavedPolicy::Keep));
        assert_eq!(saved_policy(&Invocation::new().set("activations", json!("recompute"))), Ok(SavedPolicy::Recompute));
        let e = saved_policy(&Invocation::new().set("activations", json!("recompute+bf16"))).unwrap_err();
        assert!(e.contains("activations"), "{e}");
    }

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
        // seed is optional with no default: an unseeded request draws one and
        // reports it in its outcome.
        assert!(t2i.params.iter().any(|p| p.name == "seed" && !p.required && p.default.is_none()));
        assert_eq!(t2i.params.iter().find(|p| p.name == "width").unwrap().default, None, "unset width is 512, or an edit's init image's");
        assert_eq!(t2i.params.iter().find(|p| p.name == "steps").unwrap().default, Some(json!(0)));
        let variant = t2i.params.iter().find(|p| p.name == "variant").unwrap();
        assert_eq!(variant.default, Some(json!("klein-4b")));
        assert!(matches!(&variant.ty, ParamType::Enum(v) if v == &VARIANTS.map(String::from).to_vec()));
        let precision = t2i.params.iter().find(|p| p.name == "precision").unwrap();
        assert_eq!(precision.default, None, "unset precision is the checkpoint's own; a default would read as an explicit choice");
        assert!(matches!(&precision.ty, ParamType::Enum(v) if v == &PRECISIONS.map(String::from).to_vec()));
        assert!(t2i.streaming);
        assert_eq!(t2i.outputs[0].media, Media::Image);
        // edit requires the primary reference image and accepts extra refs.
        let edit = m.actions.iter().find(|a| a.name == "edit").unwrap();
        assert!(edit.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        for r in crate::request::EXTRA_REFS {
            assert!(edit.inputs.iter().any(|b| b.name == r && !b.required));
        }
        assert!(edit.inputs.iter().any(|b| b.name == "mask" && b.media == Media::Mask && !b.required));
        // lora_train declares the trained adapter as a retrievable output blob.
        let lt = m.actions.iter().find(|a| a.name == "lora_train").unwrap();
        assert!(lt.params.iter().any(|p| p.name == "data" && p.required));
        assert!(lt.params.iter().any(|p| p.name == "save" && p.required));
        assert!(lt.params.iter().any(|p| p.name == "seed" && !p.required && p.default.is_none()));
        assert!(lt.outputs.iter().any(|b| b.name == "adapter" && b.media == Media::Bytes));
        // method/lr_ratio/freeze_a are params on this SAME action, not a new
        // one - the action list assertion above already pins that.
        let method = lt.params.iter().find(|p| p.name == "method").unwrap();
        assert_eq!(method.default, Some(json!("lora")));
        assert!(matches!(&method.ty, ParamType::Enum(v) if v == &["lora".to_string(), "rslora".to_string()]));
        assert_eq!(lt.params.iter().find(|p| p.name == "lr_ratio").unwrap().default, Some(json!(1.0)));
        assert_eq!(lt.params.iter().find(|p| p.name == "freeze_a").unwrap().default, Some(json!(false)));
        for p in ["ckpt_every", "resume", "precision", "train_base", "cache_dir"] {
            assert!(lt.params.iter().any(|q| q.name == p), "lora_train takes {p}");
            assert!(!t2i.params.iter().any(|q| q.name == p && p != "precision"), "{p} is a training knob, not a generation one");
        }
        // the whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["actions"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn generation_params_carry_ui_ranges() {
        let t2i = &manifest().actions[0];
        let p = |name: &str| t2i.params.iter().find(|p| p.name == name).unwrap();
        assert_eq!((p("width").min, p("width").max, p("width").step), (Some(64.0), Some(capability::bounds::SIDE_PX), Some(16.0)));
        assert_eq!((p("height").min, p("height").max, p("height").step), (Some(64.0), Some(capability::bounds::SIDE_PX), Some(16.0)));
        assert_eq!((p("steps").min, p("steps").max, p("steps").step), (Some(0.0), Some(150.0), Some(1.0)));
        assert_eq!((p("guidance").min, p("guidance").max), (Some(0.0), Some(30.0)));

        let edit = manifest().actions.iter().find(|a| a.name == "edit").cloned().unwrap();
        let strength = edit.params.iter().find(|p| p.name == "strength").unwrap().clone();
        assert_eq!((strength.min, strength.max, strength.step), (Some(0.0), Some(1.0), Some(0.01)));
    }

    #[test]
    fn nine_b_variants_are_license_gated() {
        let err = check_license("klein-9b", false).unwrap_err();
        assert!(err.contains("Non-Commercial"), "error must name the license: {err}");
        assert!(err.contains("--allow-non-commercial") && err.contains("allow_non_commercial"), "error must name the opt-in on both surfaces: {err}");
        assert!(check_license("klein-9b", true).is_ok());
        assert!(check_license("klein-4b", false).is_ok());
        assert!(check_license("base-4b", false).is_ok());
    }

    #[test]
    fn every_action_that_can_load_9b_declares_the_opt_in_off_by_default() {
        for a in manifest().actions {
            let p = a.params.iter().find(|p| p.name == "allow_non_commercial").unwrap_or_else(|| panic!("{} declares allow_non_commercial", a.name));
            assert_eq!((&p.ty, &p.default), (&ParamType::Bool, &Some(json!(false))), "{}", a.name);
        }
        assert!(!non_commercial_accepted(&Invocation::new()));
        assert!(!non_commercial_accepted(&Invocation::new().set("allow_non_commercial", json!(false))));
        assert!(non_commercial_accepted(&Invocation::new().set("allow_non_commercial", json!(true))));
    }

    /// An unseeded request is valid, and the image it produces reports the
    /// seed it was drawn with, so it can be reproduced.
    #[test]
    fn text2image_takes_an_unseeded_request_and_reports_the_seed() {
        let t2i = manifest().actions.into_iter().find(|a| a.name == "text2image").unwrap();
        assert!(t2i.validate(capability::Invocation::new().set("prompt", json!("a cat"))).is_ok());
        assert_eq!(image_outcome(&[0, 0, 0], 1, 1, 42).outputs["seed"], json!(42));
    }

    #[test]
    fn text2image_accepts_two_invocations_differing_only_in_seed() {
        let t2i = manifest().actions.into_iter().find(|a| a.name == "text2image").unwrap();
        let inv1 = capability::Invocation::new().set("prompt", json!("a cat")).set("seed", json!(1));
        let inv2 = capability::Invocation::new().set("prompt", json!("a cat")).set("seed", json!(2));
        assert!(t2i.validate(inv1).is_ok());
        assert!(t2i.validate(inv2).is_ok());
    }

    /// A minimal single-file safetensors fixture carrying only the five
    /// tensors `flux2::sniff_dit_size` reads, at real `cfg` dimensions - `U8`
    /// content of all zeros (the sniff never decodes a value, only the
    /// header's declared shapes). Mirrors `import`'s own test fixture.
    fn write_fake_dit(cfg: &Flux2Config, tag: &str) -> std::path::PathBuf {
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
        let path = std::env::temp_dir().join(format!("brain-flux2-caps-dit-{tag}-{}-{n}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        path
    }

    /// THE regression test for the compliance hole: a real 9B-shaped
    /// checkpoint, requested/labeled "klein-4b" (the mislabel that used to
    /// skip the NC gate entirely, since the old `check_license` trusted the
    /// string verbatim), must still be recognized as 9B and refused.
    /// `bind_variant` is what closes it - it never guesses, only sniffs.
    #[test]
    fn a_mislabeled_9b_checkpoint_is_still_license_gated() {
        let path = write_fake_dit(&Flux2Config::klein_9b(), "mislabel");
        let bound = bind_variant(path.to_str().unwrap(), "klein-4b").expect("a real 9B checkpoint must sniff cleanly");
        assert_eq!(bound, "klein-9b", "the mislabel must not survive binding");
        let err = check_license(&bound, false).unwrap_err();
        assert!(err.contains("Non-Commercial"), "{err}");
        assert!(err.contains("allow_non_commercial"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// `bind_variant` keeps the caller's stated FAMILY (klein vs base is not
    /// sniffable) while overriding only the SIZE half from the real weights.
    #[test]
    fn bind_variant_keeps_the_stated_family_but_sniffs_the_size() {
        let path = write_fake_dit(&Flux2Config::klein_4b(), "family");
        assert_eq!(bind_variant(path.to_str().unwrap(), "base-9b").unwrap(), "base-4b");
        assert_eq!(bind_variant(path.to_str().unwrap(), "klein-4b").unwrap(), "klein-4b");
        assert!(bind_variant(path.to_str().unwrap(), "not-a-family").is_err());
        std::fs::remove_file(&path).ok();
    }
}
