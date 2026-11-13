// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training and fine-tuning the detector on a `Dataset::Detect` directory (CHW
//! `images.f32` + `boxes.bin` + `meta.json`, as `brain data gen detect` writes).
//!
//! The model is trained with the real detection loss ([`LossMode::Detection`]):
//! per step one image batch is uploaded, its ground-truth boxes are set, then
//! forward / backward / AdamW. [`run`] is both verbs: from scratch when
//! `pretrained` is `None`, a fine-tune of a checkpoint when it names one.
//!
//! A fine-tune builds on the CHECKPOINT'S OWN architecture, and a class count
//! above the checkpoint's APPENDS classes: the pretrained ones keep their
//! indices and weights, the new ones get a fresh head ([`crate::finetune`]). The
//! freezes and [`TrainOptions::train_classes`] are what keep the pretrained
//! classes working while a new one is taught - see [`Yolo::freeze_backbone`],
//! [`Yolo::freeze_cls_hidden`] and [`Yolo::train_only_classes`] for what each
//! one protects and why gradients alone are not enough.
//!
//! Swedish Embedded AB implements object detectors trained on a client's own
//! imagery, on the client's own hardware. If your team needs expertise in
//! adapting a detector to new classes without losing the old ones then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::BTreeSet;
use std::path::Path;

use data::gen_detect::{load_dataset, DetectData};
use model::saved::SavedPolicy;

use crate::model::{GtBox, LossMode, Yolo};
use crate::YoloConfig;

/// The architecture a run starts from, before the class count and input size
/// are applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// From scratch: `tiny`. Fine-tuning: the checkpoint's own layout, so the
    /// pretrained tensors fit by construction. `YoloConfig::tiny` deliberately
    /// uses different channel widths from the canonical `yolov8n`, so building
    /// `tiny` and then copying a real COCO checkpoint into it would match
    /// almost nothing and silently train from scratch.
    Auto,
    Tiny,
    Yolov8n,
}

impl Arch {
    pub const NAMES: [&'static str; 3] = ["auto", "tiny", "yolov8n"];

    pub fn parse(name: &str) -> Result<Arch, String> {
        match name {
            "auto" | "" => Ok(Arch::Auto),
            "tiny" => Ok(Arch::Tiny),
            "yolov8n" => Ok(Arch::Yolov8n),
            other => Err(format!("unknown arch {other:?} (expected auto|tiny|yolov8n)")),
        }
    }
}

/// Every knob of a training run.
#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub steps: u32,
    /// Images per step.
    pub batch: u32,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    /// Class count; 0 = the dataset's (from scratch) or the checkpoint's
    /// (fine-tune). Above the checkpoint's, classes are appended.
    pub nc: u32,
    /// Model input side; 0 = the dataset's own, so its CHW blob uploads
    /// directly with no letterbox.
    pub input: u32,
    pub arch: Arch,
    pub freeze_backbone: bool,
    pub freeze_reg_head: bool,
    pub freeze_cls_hidden: bool,
    /// Class ids allowed to learn; empty = all of them.
    pub train_classes: Vec<u32>,
    /// `None` keeps the detector's own default.
    pub activations: Option<SavedPolicy>,
}

impl Default for TrainOptions {
    fn default() -> TrainOptions {
        TrainOptions {
            steps: 200,
            batch: 4,
            lr: 1e-3,
            wd: 1e-2,
            seed: data::rng::random_seed(),
            nc: 0,
            input: 0,
            arch: Arch::Auto,
            freeze_backbone: false,
            freeze_reg_head: false,
            freeze_cls_hidden: false,
            train_classes: Vec::new(),
            activations: None,
        }
    }
}

/// What a run did.
#[derive(Debug)]
pub struct TrainReport {
    pub first_loss: f32,
    pub last_loss: f32,
    /// Steps taken; fewer than asked when the run was stopped.
    pub steps_run: u32,
    pub nc: u32,
    pub input: u32,
    /// What the fine-tune initialization copied from the checkpoint.
    pub init: Option<String>,
    /// What each freeze covered, and the things worth a second look.
    pub notes: Vec<String>,
}

/// 90/10 train/val split (chronological): the first index of the val split.
pub fn val_start(n: usize) -> usize {
    ((n as f64) * 0.9) as usize
}

/// Image `i`'s boxes as loss targets for position `img` in the batch (the loss
/// reads normalized centre-xywh + class).
fn targets_for(data: &DetectData, i: usize, img: u32) -> Vec<GtBox> {
    data.boxes[i].iter().map(|b| GtBox { img, cls: b.class, cx: b.cx, cy: b.cy, w: b.w, h: b.h }).collect()
}

/// The distinct class ids that appear in the dataset's boxes.
fn used_classes(data: &DetectData) -> BTreeSet<u32> {
    data.boxes.iter().flatten().map(|b| b.class).collect()
}

/// The architecture to build, before the class count and input size apply.
fn base_config(opts: &TrainOptions, pretrained: Option<&YoloConfig>, data: &DetectData) -> YoloConfig {
    // From scratch there is no checkpoint to inherit a class count from, so the
    // dataset's own is the fallback (`nc` still overrides, in `build_model`).
    let scratch_nc = if opts.nc > 0 { opts.nc } else { data.nc.max(1) };
    match (opts.arch, pretrained) {
        (Arch::Yolov8n, _) => YoloConfig::yolov8n(),
        (Arch::Tiny, _) => YoloConfig::tiny(scratch_nc),
        (Arch::Auto, Some(cfg)) => cfg.clone(),
        (Arch::Auto, None) => YoloConfig::tiny(scratch_nc),
    }
}

/// A `Yolo` for the dataset's geometry from `base`, at the requested class
/// count, with seeded random weights ([`crate::finetune::load_pretrained`]
/// overwrites them from a checkpoint afterwards).
fn build_model(base: YoloConfig, opts: &TrainOptions, data: &DetectData) -> Yolo {
    let mut cfg = base;
    if opts.nc > 0 {
        cfg.nc = opts.nc;
    }
    cfg.nc = cfg.nc.max(1);
    cfg.input = if opts.input > 0 { opts.input } else { data.w };
    let init = <Yolo as model::Model>::init_weights(&cfg, opts.seed);
    Yolo::new(cfg, opts.batch, 0, &init)
}

/// Apply the requested freezes and class gate, reporting what each covered.
fn apply_freezes(model: &mut Yolo, opts: &TrainOptions, data: &DetectData) -> Result<Vec<String>, String> {
    let mut notes = Vec::new();
    if opts.freeze_backbone {
        let n = model.freeze_backbone();
        notes.push(format!("froze {n} backbone+neck tensors (gradients AND BatchNorm running stats)"));
    }
    if opts.freeze_reg_head {
        let n = model.freeze_reg_head();
        notes.push(format!("froze {n} box/DFL head tensors (the box head is class-agnostic)"));
    }
    if opts.freeze_cls_hidden {
        let n = model.freeze_cls_hidden();
        notes.push(format!("froze {n} class-branch hidden tensors (only the final per-class 1x1 trains)"));
    }
    if !opts.train_classes.is_empty() {
        if let Some(c) = opts.train_classes.iter().find(|&&c| c >= model.cfg.nc) {
            return Err(format!("train_classes: class {c} is out of range for a model with {} classes", model.cfg.nc));
        }
        model.train_only_classes(&opts.train_classes);
        notes.push(format!("training only class(es) {:?}; every other class's gradient is gated off", opts.train_classes));
        if opts.wd != 0.0 {
            notes.push(format!(
                "WARNING train_classes with wd {} - AdamW's decoupled weight decay is not a gradient and still shrinks the gated classes. Pass wd 0 to keep them bit-exact.",
                opts.wd
            ));
        }
    } else if model.cfg.nc as usize > used_classes(data).len() {
        notes.push(format!(
            "WARNING this model has {} classes but the dataset only uses {}. The detection loss scores BCE over ALL classes against an all-zero target for the absent ones, so they will be trained to never fire. Pass train_classes to gate them off.",
            model.cfg.nc,
            used_classes(data).len()
        ));
    }
    Ok(notes)
}

/// Per-kernel device time of the step just run, largest first (`--kernel-profile`).
fn print_kernel_table(gpu: &gpu_core::Gpu) {
    let Some(mut rows) = gpu.kernel_times() else {
        eprintln!("kernel table: this backend cannot time kernels");
        return;
    };
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    eprintln!("kernel table: {:.1} ms device time over {} kernel kinds", total, rows.len());
    for (name, ms, calls) in rows.iter().take(25) {
        eprintln!("  {name:<44} {ms:>9.2} ms {calls:>6} calls {:>5.1}%", 100.0 * ms / total);
    }
}

/// The loop over the train split. `on_step(step, steps, loss)` is called after
/// each step; returning `false` stops the run there.
fn train_loop(model: &Yolo, data: &DetectData, opts: &TrainOptions, on_step: &mut dyn FnMut(u32, u32, f32) -> bool) -> (f32, f32, u32) {
    let n_train = val_start(data.n).max(1);
    let b = opts.batch as usize;
    let stride = data.image_stride();
    let image_len = 3 * model.cfg.input as usize * model.cfg.input as usize;

    model.set_mode(LossMode::Detection);
    model.set_eval(false);
    // Accumulate BN running mean/var during training so eval-mode inference
    // (the saved checkpoint -> `Yolo::detect`) reads usable running stats.
    model.set_update_running(true);

    let (mut first, mut last, mut done) = (f32::NAN, f32::NAN, 0u32);
    let mut batch = vec![0.0f32; b * image_len];
    for step in 0..opts.steps {
        // Round-robin contiguous mini-batch over the train split.
        let base = (step as usize * b) % n_train;
        let mut gts: Vec<GtBox> = Vec::new();
        for j in 0..b {
            let idx = (base + j) % n_train;
            batch[j * image_len..(j + 1) * image_len].copy_from_slice(&data.images[idx * stride..idx * stride + image_len]);
            gts.extend(targets_for(data, idx, j as u32));
        }
        let t_step = std::time::Instant::now();
        // Under --kernel-profile, attribute the second step's DEVICE time per
        // kernel (the first step pays one-time kernel compilation).
        let kernel_table = gpu_core::profile::enabled() && step == 1;
        if kernel_table {
            model.gpu.reset_kernel_times();
            model.gpu.set_kernel_timing(true);
        }
        model.set_image(&batch);
        model.set_targets(&gts);
        model.zero_grads();
        let t = std::time::Instant::now();
        let loss = model.forward();
        gpu_core::profile::stage_time("yolov8 step.forward+loss", t);
        let t = std::time::Instant::now();
        model.backward();
        gpu_core::profile::stage_time("yolov8 step.backward", t);
        let t = std::time::Instant::now();
        model.adamw_step(step + 1, opts.lr, opts.wd, Default::default(), Some(1.0), 1.0);
        model.poll_wait();
        gpu_core::profile::stage_time("yolov8 step.adamw+sync", t);
        gpu_core::profile::stage_time("yolov8 step.total", t_step);
        if kernel_table {
            print_kernel_table(&model.gpu);
            model.gpu.set_kernel_timing(false);
        }
        if first.is_nan() {
            first = loss;
        }
        last = loss;
        done = step + 1;
        if !on_step(step, opts.steps, loss) {
            break;
        }
    }
    (first, last, done)
}

/// Train on the detection dataset in `data_dir`, from scratch or - when
/// `pretrained` names a checkpoint - as a fine-tune of it, and write the result
/// to `save`. `on_step(step, steps, loss)` reports progress and may stop the
/// run by returning `false` (the weights so far are still saved).
pub fn run(data_dir: &Path, pretrained: Option<&str>, save: &str, opts: &TrainOptions, on_step: &mut dyn FnMut(u32, u32, f32) -> bool) -> Result<TrainReport, String> {
    if opts.steps == 0 || opts.batch == 0 {
        return Err("steps and batch must be at least 1".into());
    }
    let data = load_dataset(data_dir).map_err(|e| format!("loading {}: {e}", data_dir.display()))?;
    if data.n == 0 {
        return Err(format!("{} holds no images", data_dir.display()));
    }
    let checkpoint = pretrained.map(crate::model::read_checkpoint).transpose()?;
    let mut model = build_model(base_config(opts, checkpoint.as_ref().map(|(cfg, _)| cfg), &data), opts, &data);
    let side = model.cfg.input;
    if (data.w, data.h) != (side, side) {
        return Err(format!(
            "dataset {}x{} != model input {side}: training needs matching geometry (regenerate the dataset at {side}px, or leave input unset to train at {})",
            data.w, data.h, data.w
        ));
    }
    if let Some(policy) = opts.activations {
        model.set_saved_policy(policy)?;
    }
    let mut notes = Vec::new();
    let mut init = None;
    if let (Some((_, src)), Some(path)) = (&checkpoint, pretrained) {
        let rep = crate::finetune::load_pretrained(&model, src, opts.seed);
        if !rep.transferred_anything() {
            return Err(format!(
                "{path} shares no tensor with this model: that is a from-scratch run, not a fine-tune (architecture mismatch: set arch to match the checkpoint, or train without it)"
            ));
        }
        if !rep.mismatched.is_empty() {
            notes.push(format!("{} tensor(s) left at random init:", rep.mismatched.len()));
            notes.extend(rep.mismatched.iter().take(10).map(|(n, want, got)| format!("  {n}: model wants {want} elements, checkpoint has {got}")));
        }
        init = Some(rep.summary());
    }
    notes.extend(apply_freezes(&mut model, opts, &data)?);
    let (first_loss, last_loss, steps_run) = train_loop(&model, &data, opts, on_step);
    model.try_save(save)?;
    Ok(TrainReport { first_loss, last_loss, steps_run, nc: model.cfg.nc, input: side, init, notes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::gen_detect::{write_dataset, Preset};

    fn dataset(tag: &str, nc: u32) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("yolov8-train-{tag}-{}", std::process::id()));
        write_dataset(&dir, Preset::MultiObject, 10, 64, 64, nc, 7).unwrap();
        dir
    }

    fn quick(steps: u32) -> TrainOptions {
        TrainOptions { steps, batch: 2, seed: 3, ..TrainOptions::default() }
    }

    fn tensor(path: &str, name: &str) -> Vec<f32> {
        crate::model::read_checkpoint(path).unwrap().1.remove(name).unwrap_or_else(|| panic!("{name} missing"))
    }

    #[test]
    fn the_split_keeps_the_last_tenth_held_out() {
        assert_eq!((val_start(10), val_start(100), val_start(1)), (9, 90, 0));
    }

    /// A from-scratch run trains at the dataset's geometry, takes the dataset's
    /// class count, writes a checkpoint that loads back, and reports each step.
    #[test]
    fn a_scratch_run_saves_a_checkpoint_that_loads_back() {
        let dir = dataset("scratch", 2);
        let save = dir.join("w.safetensors");
        let mut losses = Vec::new();
        let rep = run(&dir, None, save.to_str().unwrap(), &quick(2), &mut |s, total, l| {
            assert_eq!(total, 2);
            losses.push((s, l));
            true
        })
        .unwrap();
        assert_eq!((rep.steps_run, rep.nc, rep.input), (2, 2, 64));
        assert_eq!(losses.len(), 2);
        assert!(rep.first_loss.is_finite() && rep.last_loss.is_finite());
        let loaded = Yolo::try_load_at(save.to_str().unwrap(), 1, 0).unwrap();
        assert_eq!((loaded.cfg.nc, loaded.cfg.input), (2, 64));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stopped_run_still_saves_what_it_learned() {
        let dir = dataset("stop", 1);
        let save = dir.join("w.safetensors");
        let rep = run(&dir, None, save.to_str().unwrap(), &quick(5), &mut |step, _, _| step < 1).unwrap();
        assert_eq!(rep.steps_run, 2, "the callback that answers false after step 2 ends the run there");
        assert!(save.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fine-tune contract: a class count above the checkpoint's appends
    /// classes, and `freeze_backbone` leaves the trunk bit-for-bit unchanged.
    #[test]
    fn a_fine_tune_appends_classes_and_a_frozen_backbone_does_not_move() {
        let dir = dataset("ft", 1);
        let base = dir.join("base.safetensors");
        run(&dir, None, base.to_str().unwrap(), &quick(1), &mut |_, _, _| true).unwrap();
        let tuned = dir.join("tuned.safetensors");
        let opts = TrainOptions { nc: 3, freeze_backbone: true, ..quick(2) };
        let rep = run(&dir, Some(base.to_str().unwrap()), tuned.to_str().unwrap(), &opts, &mut |_, _, _| true).unwrap();
        assert_eq!(rep.nc, 3, "classes are appended to the checkpoint's");
        assert!(rep.init.as_deref().is_some_and(|s| s.contains("expanded")), "{:?}", rep.init);
        assert!(rep.notes.iter().any(|n| n.starts_with("froze")), "{:?}", rep.notes);
        let trunk = "backbone.0.conv.weight";
        let (before, after) = (tensor(base.to_str().unwrap(), trunk), tensor(tuned.to_str().unwrap(), trunk));
        assert_eq!(before, after, "a frozen trunk is bit-exact");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A checkpoint of another layout than the one asked for is reported tensor
    /// by tensor, never silently half-loaded.
    #[test]
    fn a_fine_tune_names_the_tensors_the_checkpoint_could_not_fill() {
        let dir = dataset("nomatch", 1);
        let (save, base) = (dir.join("w.safetensors"), dir.join("base.safetensors"));
        run(&dir, None, base.to_str().unwrap(), &quick(1), &mut |_, _, _| true).unwrap();
        // yolov8n's layout against a tiny checkpoint: the trunk does not line up.
        let opts = TrainOptions { arch: Arch::Yolov8n, ..quick(1) };
        let rep = run(&dir, Some(base.to_str().unwrap()), save.to_str().unwrap(), &opts, &mut |_, _, _| true).unwrap();
        assert!(rep.notes.iter().any(|n| n.contains("left at random init")), "{:?}", rep.notes);
        assert!(rep.notes.iter().any(|n| n.contains("model wants")), "{:?}", rep.notes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_class_gate_outside_the_model_is_refused_and_unused_classes_are_flagged() {
        let dir = dataset("gate", 1);
        let save = dir.join("w.safetensors");
        let err = run(&dir, None, save.to_str().unwrap(), &TrainOptions { train_classes: vec![5], ..quick(1) }, &mut |_, _, _| true).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
        let rep = run(&dir, None, save.to_str().unwrap(), &TrainOptions { nc: 4, ..quick(1) }, &mut |_, _, _| true).unwrap();
        assert!(rep.notes.iter().any(|n| n.contains("trained to never fire")), "{:?}", rep.notes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dataset_at_another_size_than_the_model_is_refused() {
        let dir = dataset("geom", 1);
        let err = run(&dir, None, "/nonexistent/w.safetensors", &TrainOptions { input: 96, ..quick(1) }, &mut |_, _, _| true).unwrap_err();
        assert!(err.contains("matching geometry"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
