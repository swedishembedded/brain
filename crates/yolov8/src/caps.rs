// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8's served actions: `detect` (with the optional second stage that
//! tells WHICH person a box holds), `train` (from scratch), `finetune` (from a
//! checkpoint, appending classes and freezing what must not move) and `eval`
//! (the mAP report).
//!
//! `detect` is [`Yolo::detect`] (letterbox, forward, DFL decode, NMS) over the
//! image at its own resolution; boxes come back in original-image pixels. With
//! an `identity_ref` embedding it becomes detect-then-verify: SCRFD finds every
//! face in the full-resolution frame, each face is attributed to the person box
//! its centre falls inside and embedded by ArcFace (a [`FaceVerifier`] the host
//! supplies),
//! and a box whose cosine against the reference clears `identity_threshold` is
//! labelled `identity_name`. SCRFD and ArcFace come from the model store
//! through the host's model store, and the `components` param names them
//! outright.
//!
//! The detector stays resident across calls (keyed by checkpoint and input side).
//!
//! Swedish Embedded AB implements object detection and person re-identification
//! on its clients' own cameras and hardware. If your team needs expertise in
//! detectors that are trained, verified and served on-premises then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use capability::{
    hot::Hot, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use model::saved::SavedPolicy;
use serde_json::{json, Value};

use crate::evaluate::{self, Split};
use crate::identity::{parse_embedding, FaceVerifier, IdentityGate, Labeled, VerifyRequest};
use crate::model::Yolo;
use crate::train::{self, Arch, TrainOptions};

/// The model id the catalog serves YOLOv8 under.
pub const MODEL: &str = "brain/yolov8";

/// Most images one training step may hold.
const MAX_BATCH: f64 = 1_024.0;
/// Most classes a detector may have (a class id is never above this).
const MAX_CLASSES: f64 = 10_000.0;

fn detect_spec() -> ActionSpec {
    ActionSpec::new("detect", "detect objects in an image (letterbox, forward, DFL decode, NMS); with identity_ref, also tell which person each person box holds (SCRFD + ArcFace)").cancellable()
        .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.25)))
        .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.45)))
        .param(ParamSpec::new("input", ParamType::Int, "side the image is letterboxed to, x32; 0 = the checkpoint's own. The net is fully convolutional, so a fine-tune's cheap training side need not be the pretrained classes' inference side").min(0.0).max(capability::bounds::SIZE_PX).default(json!(0)))
        .param(ParamSpec::new("identity_ref_path", ParamType::Str, "embedding file (little-endian f32, what `arcface embed` writes as `embedding`) on the serving machine, when identity_ref is not sent as a blob").host_resolved())
        .param(ParamSpec::new("identity_name", ParamType::Str, "label of a person box whose face matches the reference").default(json!("identity")))
        .param(ParamSpec::new("identity_class_name", ParamType::Str, "label of a verified-class box that does not match").default(json!("person")))
        .param(ParamSpec::new("identity_threshold", ParamType::Float, "cosine floor for a match; insightface's antelopev2 recipe verifies around 0.4 on photographs, and the floor is a choice because the population sets it: generated or heavily compressed imagery separates lower than camera originals").min(-1.0).max(1.0).default(json!(0.35)))
        .param(ParamSpec::new("identity_class", ParamType::Int, "the detected class that is verified (0 = COCO person); other classes pass through unlabelled").min(0.0).max(MAX_CLASSES).default(json!(0)))
        .param(ParamSpec::new("identity_margin", ParamType::Float, "fraction of a box's own size its face search widens by; a tight person box clips the hairline, where the landmarker is least reliable").min(0.0).max(capability::bounds::UNIT).default(json!(0.05)))
        .param(ParamSpec::new("components", ParamType::Str, "the face models' files named outright, `<model>.<role>=<path>` comma-separated: arcface.weights (glintr100.onnx; scrfd_10g_bnkps.onnx is read from its directory)").host_resolved())
        .input(BlobSpec::new("image", Media::Image, "the image to run detection on, at its own resolution").required())
        .input(BlobSpec::new("identity_ref", Media::Bytes, "the reference identity's embedding (little-endian f32, what `arcface embed` writes as `embedding`); its presence turns identity verification on"))
        .output(BlobSpec::new("detections", Media::Text, "JSON array of {bbox:[x1,y1,x2,y2], conf, class} in image coords, plus label and similarity under identity verification"))
}

/// The knobs `train` and `finetune` share.
fn training_params(spec: ActionSpec) -> ActionSpec {
    spec.training()
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "detection dataset directory on the serving machine (images.f32, boxes.bin, meta.json: what `brain data gen detect` writes)").required().host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "where to write the trained checkpoint (.safetensors) on the serving machine").required().host_resolved())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(200)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("batch", ParamType::Int, "images per step").default(json!(4)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(1e-3)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("wd", ParamType::Float, "AdamW weight decay; 0 keeps classes gated off by train_classes bit-exact").default(json!(1e-2)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("nc", ParamType::Int, "class count; 0 = the dataset's (from scratch) or the checkpoint's (finetune). Above the checkpoint's, classes are appended: the pretrained ones keep their indices and weights").default(json!(0)).min(0.0).max(MAX_CLASSES))
        .param(ParamSpec::new("input", ParamType::Int, "model input side, x32; 0 = the dataset's own, so its images upload with no letterbox").default(json!(0)).min(0.0).max(capability::bounds::SIZE_PX))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the fresh weights (omit for random)").min(0.0))
        .param(ParamSpec::new("arch", ParamType::Enum(Arch::NAMES.map(String::from).to_vec()), "layout: auto = tiny from scratch and the checkpoint's own for a fine-tune (so its tensors fit), or tiny / yolov8n outright").default(json!("auto")))
        .param(ParamSpec::new("freeze_backbone", ParamType::Bool, "freeze the backbone and neck: no gradients and BatchNorm running statistics pinned, so they are bit-for-bit unchanged").default(json!(false)))
        .param(ParamSpec::new("freeze_reg_head", ParamType::Bool, "freeze the box/DFL head, which is class-agnostic").default(json!(false)))
        .param(ParamSpec::new("freeze_cls_hidden", ParamType::Bool, "freeze the class branch's shared hidden convs so only the final per-class 1x1 trains: the untouched classes are then a function of frozen inputs and weights end to end").default(json!(false)))
        .param(ParamSpec::new("train_classes", ParamType::Str, "class ids allowed to learn, comma-separated (e.g. 80,81); every other class's gradient is gated off, which a fine-tune on data without all the classes needs").default(json!("")))
        .param(ParamSpec::new("activations", ParamType::Enum(SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()), "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward for the same gradients at lower memory (default keep)"))
}

fn train_spec() -> ActionSpec {
    training_params(ActionSpec::new("train", "train a detector from freshly initialized weights on a detection dataset (the real detection loss, AdamW)").cancellable()).from_scratch()
}

fn finetune_spec() -> ActionSpec {
    training_params(ActionSpec::new("finetune", "continue training the resolved checkpoint on a detection dataset, on the checkpoint's own architecture; a larger nc appends classes").cancellable())
}

fn eval_spec() -> ActionSpec {
    ActionSpec::new("eval", "score the checkpoint on a detection dataset: mAP@0.5, mAP@0.5:0.95, precision, recall and per-class AP").cancellable()
        .param(ParamSpec::new("data", ParamType::Str, "detection dataset directory on the serving machine (images.f32, boxes.bin, meta.json)").required().host_resolved())
        .param(ParamSpec::new("split", ParamType::Enum(Split::NAMES.map(String::from).to_vec()), "images to score: val = the last 10% that training never sees, all = every image (for a checkpoint not trained on this dataset)").default(json!("val")))
        .param(ParamSpec::new("conf", ParamType::Float, "confidence floor; use 0.001 for mAP").min(0.0).max(capability::bounds::UNIT).default(json!(0.25)))
        .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.45)))
        .param(ParamSpec::new("dump_preds", ParamType::Str, "path of a JSONL file to write on the serving machine: one line per image (index, ground truth, predictions)").host_resolved())
        .output(BlobSpec::new("report", Media::Text, "the metric table"))
}

/// The full, static capability manifest: safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "YOLOv8-style anchor-free object detector: detection with optional identity verification, training, fine-tuning and evaluation.", vec![detect_spec(), train_spec(), finetune_spec(), eval_spec()])
}

/// The executable YOLOv8 behind the manifest, over one checkpoint (none for a
/// model trained from scratch). Construction is free: the checkpoint loads on
/// the first call that needs it and stays resident.
#[derive(Default)]
pub struct YoloProvider {
    weights: Option<String>,
    faces: Option<Arc<dyn FaceVerifier>>,
    detector: Arc<Hot<(String, u32), Yolo>>,
}

impl YoloProvider {
    /// `faces` runs identity verification; without it `detect` still detects
    /// and refuses only an identity.
    pub fn new(weights: Option<String>, faces: Option<Arc<dyn FaceVerifier>>) -> YoloProvider {
        YoloProvider { weights, faces, ..YoloProvider::default() }
    }
}

/// YOLOv8 over `assembly`'s `weights` role (none serves only `train`), with the
/// identity verification done by `faces`.
pub fn provider(assembly: &capability::Assembly, faces: Arc<dyn FaceVerifier>) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(YoloProvider::new(assembly.role_path("weights").ok(), Some(faces))))
}

impl Provider for YoloProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let weights = self.weights.clone();
        match name {
            "detect" => Some(Arc::new(DetectAction { weights, faces: self.faces.clone(), detector: self.detector.clone() })),
            "train" => Some(Arc::new(TrainAction { weights: None, finetune: false })),
            "finetune" => Some(Arc::new(TrainAction { weights, finetune: true })),
            "eval" => Some(Arc::new(EvalAction { weights })),
            _ => None,
        }
    }
}

fn need(weights: &Option<String>, action: &str) -> Result<String, String> {
    weights.clone().ok_or_else(|| format!("yolov8 {action}: no YOLOv8 checkpoint (pull one, or name it with --weights)"))
}

struct DetectAction {
    weights: Option<String>,
    faces: Option<Arc<dyn FaceVerifier>>,
    detector: Arc<Hot<(String, u32), Yolo>>,
}

/// The reference embedding the invocation names: the blob, else the file.
fn reference_of(inv: &Invocation) -> Result<Option<Vec<f32>>, String> {
    if let Some(blob) = inv.get_blob("identity_ref") {
        return parse_embedding(&blob.bytes, "identity_ref").map(Some);
    }
    match inv.get_str("identity_ref_path").filter(|p| !p.is_empty()) {
        Some(path) => {
            let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
            parse_embedding(&bytes, &path).map(Some)
        }
        None => Ok(None),
    }
}

/// The identity gate the invocation's knobs describe, over `reference`.
fn gate_of(inv: &Invocation, reference: Vec<f32>) -> IdentityGate {
    IdentityGate {
        reference,
        threshold: inv.get_f64("identity_threshold").unwrap_or(0.35) as f32,
        name: inv.get_str("identity_name").unwrap_or_else(|| "identity".into()),
        class_name: inv.get_str("identity_class_name").unwrap_or_else(|| "person".into()),
        class: inv.get_i64("identity_class").unwrap_or(0).max(0) as u32,
        margin: inv.get_f64("identity_margin").unwrap_or(0.05) as f32,
    }
}

impl DetectAction {
    /// Stage two for `dets`: who each gated-class box holds.
    fn verify(&self, inv: &Invocation, model: &Yolo, gate: &IdentityGate, dets: &[[f32; 6]], hwc: &[f32], w: u32, h: u32) -> Result<Vec<Labeled>, String> {
        let faces = self.faces.as_ref().ok_or("yolov8 detect: identity verification needs the face models, and this provider was built without them")?;
        let named = capability::component_overrides(&inv.get_str("components").unwrap_or_default())?;
        let arcface_roles = named.get("arcface").cloned().unwrap_or_default();
        faces.verify(&VerifyRequest { gate, arcface_roles: &arcface_roles, gpu: &model.gpu, dets, hwc, w, h }).map_err(|e| format!("yolov8 detect: the face models: {e}"))
    }
}

impl Action for DetectAction {
    fn spec(&self) -> ActionSpec {
        detect_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = need(&self.weights, "detect")?;
        let conf = inv.get_f64("conf").unwrap_or(0.25) as f32;
        let iou = inv.get_f64("iou").unwrap_or(0.45) as f32;
        let input = inv.get_i64("input").unwrap_or(0).max(0) as u32;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let reference = reference_of(inv)?;

        self.detector.try_get((weights.clone(), input), || Yolo::try_load_at(&weights, 1, input), |model| {
            let dets = model.detect(&hwc, w, h, conf, iou);
            let labeled = match reference {
                // Without a reference this is exactly the detector.
                None => unlabeled(&dets),
                Some(reference) => self.verify(inv, model, &gate_of(inv, reference), &dets, &hwc, w, h)?,
            };
            detect_outcome(&labeled)
        })
    }
}

/// What `detect` returns for `labeled`: the count, how many were verified as
/// the reference identity, and the detections as one JSON array (also as the
/// `detections` text blob). Shared by every device a `detect` runs on.
pub fn detect_outcome(labeled: &[Labeled]) -> ActionResult {
    let objects = crate::identity::to_json(labeled);
    let text = serde_json::to_string(&objects).map_err(|e| e.to_string())?;
    Ok(Outcome::new()
        .set("count", json!(labeled.len()))
        .set("verified", json!(labeled.iter().filter(|l| l.verified).count()))
        .set("detections", objects)
        .blob("detections", Blob::new(Media::Text, text.into_bytes())))
}

/// Detections with no identity stage: every box unlabelled.
pub fn unlabeled(dets: &[[f32; 6]]) -> Vec<Labeled> {
    dets.iter().map(|&det| Labeled { det, label: None, similarity: None, verified: false }).collect()
}

struct TrainAction {
    /// The checkpoint a fine-tune builds on; `None` for `train`, which starts
    /// from fresh weights whatever the store holds.
    weights: Option<String>,
    finetune: bool,
}

/// Class ids from a comma-separated list; an id that is not a number is named.
fn parse_class_list(list: &str) -> Result<Vec<u32>, String> {
    list.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse::<u32>().map_err(|_| format!("train_classes: {s:?} is not a class id"))).collect()
}

fn options_of(inv: &Invocation, seed: u64) -> Result<TrainOptions, String> {
    let defaults = TrainOptions::default();
    Ok(TrainOptions {
        steps: inv.get_i64("steps").unwrap_or(defaults.steps as i64).max(1) as u32,
        batch: inv.get_i64("batch").unwrap_or(defaults.batch as i64).max(1) as u32,
        lr: inv.get_f64("lr").unwrap_or(defaults.lr as f64) as f32,
        wd: inv.get_f64("wd").unwrap_or(defaults.wd as f64) as f32,
        seed,
        nc: inv.get_i64("nc").unwrap_or(0).max(0) as u32,
        input: inv.get_i64("input").unwrap_or(0).max(0) as u32,
        arch: Arch::parse(&inv.get_str("arch").unwrap_or_default())?,
        freeze_backbone: inv.get_bool("freeze_backbone").unwrap_or(false),
        freeze_reg_head: inv.get_bool("freeze_reg_head").unwrap_or(false),
        freeze_cls_hidden: inv.get_bool("freeze_cls_hidden").unwrap_or(false),
        train_classes: parse_class_list(&inv.get_str("train_classes").unwrap_or_default())?,
        activations: inv.get_str("activations").map(|a| SavedPolicy::parse_activations(&a)).transpose()?,
    })
}

impl Action for TrainAction {
    fn spec(&self) -> ActionSpec {
        if self.finetune {
            finetune_spec()
        } else {
            train_spec()
        }
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let verb = if self.finetune { "finetune" } else { "train" };
        let data = inv.get_str("data").ok_or_else(|| format!("yolov8 {verb}: missing required param 'data'"))?;
        let save = inv.get_str("save").ok_or_else(|| format!("yolov8 {verb}: missing required param 'save'"))?;
        let pretrained = if self.finetune { Some(need(&self.weights, verb)?) } else { None };
        let seed = inv.get_i64("seed").map_or_else(data::rng::random_seed, |s| s as u64);
        let opts = options_of(inv, seed).map_err(|e| format!("yolov8 {verb}: {e}"))?;
        let report = train::run(Path::new(&data), pretrained.as_deref(), &save, &opts, &mut |step, steps, loss| {
            progress(Progress::step(step + 1, steps, format!("loss {loss:.4}")));
            !inv.cancel.is_cancelled()
        })
        .map_err(|e| format!("yolov8 {verb}: {e}"))?;
        Ok(Outcome::new()
            .set("first_loss", json!(report.first_loss))
            .set("last_loss", json!(report.last_loss))
            .set("steps_run", json!(report.steps_run))
            .set("nc", json!(report.nc))
            .set("input", json!(report.input))
            .set("seed", json!(seed))
            .set("from", json!(pretrained.unwrap_or_else(|| "fresh weights".into())))
            .set("init", json!(report.init))
            .set("notes", json!(report.notes))
            .set("saved", json!(save)))
    }
}

struct EvalAction {
    weights: Option<String>,
}

/// The metric table `eval` prints, one metric per line then per-class AP.
fn report_table(r: &crate::detection_report::DetectionReport, split: Split, conf: f32, iou: f32) -> String {
    let mut t = format!("metric         value\nmAP@0.5        {:.4}\nmAP@0.5:0.95   {:.4}\nprecision@0.5  {:.4}\nrecall@0.5     {:.4}\nclass  AP@0.5  AP@0.5:0.95\n", r.map50, r.map50_95, r.precision50, r.recall50);
    for c in &r.per_class {
        t += &format!("{:<5}  {:.4}   {:.4}\n", c.class, c.ap50, c.ap50_95);
    }
    t + &format!("conf {conf}  iou {iou}  split {}  preds {}  gts {}  (images {})\n", split.name(), r.n_preds, r.n_gts, r.n_images)
}

impl Action for EvalAction {
    fn spec(&self) -> ActionSpec {
        eval_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = need(&self.weights, "eval")?;
        let data = inv.get_str("data").ok_or("yolov8 eval: missing required param 'data'")?;
        let split = Split::parse(&inv.get_str("split").unwrap_or_else(|| "val".into())).map_err(|e| format!("yolov8 eval: {e}"))?;
        let conf = inv.get_f64("conf").unwrap_or(0.25) as f32;
        let iou = inv.get_f64("iou").unwrap_or(0.45) as f32;
        let dump = inv.get_str("dump_preds").filter(|p| !p.is_empty());
        let r = evaluate::run(&weights, Path::new(&data), split, conf, iou, dump.as_deref().map(Path::new)).map_err(|e| format!("yolov8 eval: {e}"))?;
        let per_class: Vec<Value> = r.per_class.iter().map(|c| json!({"class": c.class, "ap50": c.ap50, "ap50_95": c.ap50_95})).collect();
        Ok(Outcome::new()
            .set("map50", json!(r.map50))
            .set("map50_95", json!(r.map50_95))
            .set("precision50", json!(r.precision50))
            .set("recall50", json!(r.recall50))
            .set("per_class", json!(per_class))
            .set("preds", json!(r.n_preds))
            .set("gts", json!(r.n_gts))
            .set("images", json!(r.n_images))
            .set("split", json!(split.name()))
            .blob("report", Blob::new(Media::Text, report_table(&r, split, conf, iou).into_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YoloConfig;
    use capability::Registry;
    use data::gen_detect::{write_dataset, Preset};
    use model::ModelConfig;

    fn tiny_image(w: u32, h: u32) -> Blob {
        let px = vec![0.5f32; (w * h * 3) as usize];
        let bytes: Vec<u8> = px.iter().flat_map(|f| f.to_le_bytes()).collect();
        Blob::new(Media::Image, bytes).with_meta(json!({"w": w, "h": h, "c": 3}))
    }

    fn registry(weights: Option<&str>) -> Registry {
        let mut reg = Registry::new();
        reg.register(Arc::new(YoloProvider::new(weights.map(str::to_string), None)));
        reg
    }

    /// A tiny 64px, 1-class checkpoint written the way `Yolo::save` writes it.
    fn tiny_checkpoint(tag: &str) -> (std::path::PathBuf, String) {
        let mut cfg = YoloConfig::tiny(1);
        cfg.input = 64; // fully convolutional: a small input keeps the test fast
        let init = crate::init::init_model(&cfg, 3);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
                (name, vec![n as u64], v)
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("yolo-caps-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        let p = path.to_string_lossy().into_owned();
        (dir, p)
    }

    #[test]
    fn manifest_declares_the_four_actions_with_their_kinds() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["detect", "train", "finetune", "eval"]);
        let (detect, train, finetune, eval) = (&m.actions[0], &m.actions[1], &m.actions[2], &m.actions[3]);
        assert!(!detect.streaming && !detect.training, "detect is one-shot");
        assert!(train.training && train.streaming && train.from_scratch);
        assert!(finetune.training && finetune.streaming && !finetune.from_scratch, "a fine-tune needs its checkpoint");
        assert!(!eval.training && !eval.from_scratch);
        // Every knob the CLI verbs had is a param.
        let has = |a: &ActionSpec, ps: &[&str]| ps.iter().all(|p| a.params.iter().any(|q| q.name == *p));
        assert!(has(detect, &["conf", "iou", "input", "identity_name", "identity_threshold", "identity_class", "identity_class_name", "identity_margin", "components", "identity_ref_path"]));
        assert!(detect.inputs.iter().any(|b| b.name == "identity_ref") && detect.inputs.iter().any(|b| b.name == "image" && b.required));
        let training = ["data", "save", "steps", "batch", "lr", "wd", "nc", "input", "seed", "arch", "freeze_backbone", "freeze_reg_head", "freeze_cls_hidden", "train_classes", "activations"];
        assert!(has(train, &training) && has(finetune, &training));
        assert!(has(eval, &["data", "split", "conf", "iou", "dump_preds"]));
        assert_eq!(m.to_json()["actions"][0]["name"], "detect");
    }

    #[test]
    fn detect_validation_fills_defaults_and_refuses_a_missing_image() {
        let d = detect_spec();
        let inv = d.validate(Invocation::new().blob("image", tiny_image(1, 1))).unwrap();
        assert_eq!((inv.get_f64("iou"), inv.get_f64("identity_threshold"), inv.get_str("identity_name").as_deref()), (Some(0.45), Some(0.35), Some("identity")));
        assert!(d.validate(Invocation::new()).is_err());
    }

    #[test]
    fn a_missing_checkpoint_is_a_clean_error() {
        let err = registry(None).run(MODEL, "detect", Invocation::new().blob("image", tiny_image(1, 1)), &mut |_| {}).unwrap_err();
        assert!(err.contains("no YOLOv8 checkpoint"), "{err}");
        let err = registry(Some("/nonexistent/yolo.safetensors")).run(MODEL, "detect", Invocation::new().blob("image", tiny_image(1, 1)), &mut |_| {}).unwrap_err();
        assert!(!err.is_empty());
    }

    /// End to end on a tiny checkpoint: load, letterbox, forward, NMS. A
    /// random-init model detects nothing meaningful; the assertion is that the
    /// whole path runs and emits the outputs, without an identity stage.
    #[test]
    fn tiny_checkpoint_detects_end_to_end() {
        let (dir, path) = tiny_checkpoint("detect");
        let out = registry(Some(&path)).run(MODEL, "detect", Invocation::new().blob("image", tiny_image(48, 32)), &mut |_| {}).unwrap();
        assert!(out.outputs["count"].is_u64());
        assert_eq!(out.outputs["verified"], 0);
        assert!(out.outputs["detections"].is_array());
        let text = String::from_utf8(out.blobs["detections"].bytes.clone()).unwrap();
        assert!(serde_json::from_str::<Value>(&text).unwrap().is_array());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identity_verification_without_the_face_models_is_refused_by_name() {
        let (dir, path) = tiny_checkpoint("identity");
        // A permissive detector so the person box exists to be verified.
        let reference: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let inv = Invocation::new().blob("image", tiny_image(48, 32)).blob("identity_ref", Blob::new(Media::Bytes, reference)).set("conf", json!(0.0));
        let err = registry(Some(&path)).run(MODEL, "detect", inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("face models"), "{err}");
        let bad = Invocation::new().blob("image", tiny_image(48, 32)).blob("identity_ref", Blob::new(Media::Bytes, vec![1, 2, 3]));
        assert!(registry(Some(&path)).run(MODEL, "detect", bad, &mut |_| {}).unwrap_err().contains("identity_ref"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The served loop: `train` from scratch writes a checkpoint, `eval` scores
    /// it, `finetune` extends it by a class, and `detect` runs the result.
    #[test]
    fn train_eval_finetune_and_detect_chain_through_the_registry() {
        let dir = std::env::temp_dir().join(format!("yolo-caps-chain-{}", std::process::id()));
        write_dataset(&dir, Preset::MultiObject, 10, 64, 64, 2, 5).unwrap();
        let (data, scratch, tuned) = (dir.to_string_lossy().into_owned(), dir.join("a.safetensors").to_string_lossy().into_owned(), dir.join("b.safetensors").to_string_lossy().into_owned());
        let mut steps = Vec::new();
        let out = registry(None)
            .run(MODEL, "train", Invocation::new().set("data", json!(data)).set("save", json!(scratch)).set("steps", json!(2)).set("batch", json!(2)).set("seed", json!(4)), &mut |p| steps.push(p.step))
            .unwrap();
        assert_eq!((out.outputs["nc"].as_u64(), out.outputs["steps_run"].as_u64(), out.outputs["seed"].as_u64()), (Some(2), Some(2), Some(4)));
        assert_eq!(steps, [1, 2], "one progress event per step");

        let eval = registry(Some(&scratch)).run(MODEL, "eval", Invocation::new().set("data", json!(data)).set("split", json!("all")).set("conf", json!(0.001)), &mut |_| {}).unwrap();
        assert_eq!(eval.outputs["images"], 10);
        assert!(String::from_utf8(eval.blobs["report"].bytes.clone()).unwrap().contains("mAP@0.5"));

        let inv = Invocation::new().set("data", json!(data)).set("save", json!(tuned)).set("steps", json!(1)).set("batch", json!(2)).set("nc", json!(3)).set("freeze_backbone", json!(true)).set("train_classes", json!("2"));
        let ft = registry(Some(&scratch)).run(MODEL, "finetune", inv, &mut |_| {}).unwrap();
        assert_eq!(ft.outputs["nc"], 3);
        assert!(ft.outputs["init"].as_str().unwrap().contains("expanded"));
        let det = registry(Some(&tuned)).run(MODEL, "detect", Invocation::new().blob("image", tiny_image(64, 64)), &mut |_| {}).unwrap();
        assert!(det.outputs["detections"].is_array());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_finetune_needs_a_checkpoint_and_a_cancelled_training_stops_early() {
        let dir = std::env::temp_dir().join(format!("yolo-caps-cancel-{}", std::process::id()));
        write_dataset(&dir, Preset::MultiObject, 10, 64, 64, 1, 5).unwrap();
        let (data, save) = (dir.to_string_lossy().into_owned(), dir.join("w.safetensors").to_string_lossy().into_owned());
        let base = Invocation::new().set("data", json!(data)).set("save", json!(save)).set("batch", json!(2)).set("steps", json!(50));
        let err = registry(None).run(MODEL, "finetune", base.clone(), &mut |_| {}).unwrap_err();
        assert!(err.contains("no YOLOv8 checkpoint"), "{err}");
        let mut inv = base;
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        let out = registry(None).run(MODEL, "train", inv, &mut |_| {}).unwrap();
        assert_eq!(out.outputs["steps_run"], 1, "a cancelled run stops after the step in flight and still saves");
        assert!(Path::new(&save).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_class_list_names_what_is_not_an_id() {
        assert_eq!(parse_class_list("80, 81,").unwrap(), vec![80, 81]);
        assert!(parse_class_list("80,cat").unwrap_err().contains("cat"));
        assert!(parse_class_list("").unwrap().is_empty());
    }
}
