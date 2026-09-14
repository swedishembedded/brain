// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain yolov8 …` - train / evaluate / run the from-scratch YOLOv8-style
//! detector. Mirrors the `gpt_cli` flag-parsing idiom; respects the global
//! `--device cpu|gpu` flag handled in `main.rs` (the YOLO model itself only ever
//! instantiates the CPU backend today, see `Yolo::new`).
//!
//!   brain yolov8 train <data_dir> --out F [--steps N --batch B --lr X --nc C
//!                                          --input S --seed S --arch tiny|yolov8n]
//!   brain yolov8 eval  --weights F --data <dir> [--conf X --iou X]
//!   brain yolov8 detect --weights F --image <path> [--conf X --iou X --input S]
//!         [--identity-ref <embedding.bin> --identity-name NAME
//!          --identity-threshold X --identity-class N --identity-class-name NAME
//!          --identity-margin X --arcface-dir DIR]
//!   brain yolov8 fine-tune <data_dir> --weights F --out F [--nc C
//!         --freeze-backbone --freeze-reg-head --freeze-cls-hidden
//!         --train-classes a,b,...]
//!
//! `detect` takes the image at its OWN resolution and never wants a
//! pre-letterboxed one: `Yolo::detect` letterboxes internally and inverts the
//! transform, so its boxes come back in original-image pixel coordinates.
//! `--input` overrides the side it letterboxes TO, which the checkpoint
//! otherwise dictates - see `Yolo::load_at` for why a fine-tune's cheap
//! training resolution must not become the pretrained classes' inference one.
//!
//! `--identity-ref` turns `detect` into the **two-stage detect-then-verify**
//! recognizer: the detector finds any `--identity-class` (COCO `person`), SCRFD
//! finds every face in the ORIGINAL full-resolution frame in one pass, each
//! face is attributed to the person box its centre falls inside, and that face
//! is aligned and embedded by ArcFace (`arcface::caps::ArcFaceSession`'s own
//! `detect_faces` + `embed_face_chw`, the two halves `embed_raw_chw` composes -
//! the same path `brain arcface embed` uses). A box whose face's cosine against
//! the reference clears `--identity-threshold` is labelled `--identity-name`;
//! every other person box keeps `--identity-class-name`. This is where personal
//! identity belongs: a COCO-preserving class head is a linear probe on features
//! trained to separate CATEGORIES, and cannot express WHICH person - see
//! `yolov8::identity`'s module docs.
//!
//! `fine-tune` builds on the CHECKPOINT'S own architecture, and an `--nc` above
//! the checkpoint's class count APPENDS classes: the pretrained ones keep their
//! indices and their weights, the new ones get a fresh head (`yolov8::finetune`).
//! The freeze flags and `--train-classes` are what keep the pretrained classes
//! working while a new one is taught - see `Yolo::freeze_backbone` /
//! `Yolo::freeze_cls_hidden` / `Yolo::train_only_classes` for what each one
//! protects and why gradients alone are not enough.
//!
//! `infer` is accepted as an alias for `detect` - the canonical verb every
//! architecture answers to.
//!
//! Datasets are the synthetic `Dataset::Detect` dirs produced by
//! `brain data gen detect` (CHW `images.f32` + `boxes.bin` + `meta.json`). The
//! model is trained with the real detection loss (`LossMode::Detection`):
//! per step we upload one image batch, set its ground-truth boxes, then
//! forward/backward/adamw_step.

use std::path::Path;

use data::gen_detect::{load_dataset, DetectData};
use eval::detection::{self, GtBox as EvalGt};
use yolov8::model::{GtBox, LossMode, Yolo};
use yolov8::YoloConfig;

pub fn run_yolo(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("train") => train(&args[1..], None),
        Some("fine-tune") | Some("finetune") => fine_tune(&args[1..]),
        Some("eval") => eval(&args[1..]),
        Some("detect") | Some("infer") => detect(&args[1..]),
        other => eprintln!(
            "usage: brain yolov8 <train|fine-tune|eval|detect> ...  (got {other:?})"
        ),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// 90/10 train/val split index over a dataset (chronological, like the text
/// datasets). Returns the first index of the val split.
fn split_at(n: usize) -> usize {
    ((n as f64) * 0.9) as usize
}

/// Convert image `i`'s normalized `DetectBox` list to `GtBox` targets for image
/// index `img` in the batch (the loss reads normalized center-xywh + class).
fn gts_for(data: &DetectData, i: usize, img: u32) -> Vec<GtBox> {
    data.boxes[i]
        .iter()
        .map(|b| GtBox { img, cls: b.class, cx: b.cx, cy: b.cy, w: b.w, h: b.h })
        .collect()
}

/// Training options shared by `train` and `fine-tune`.
struct TrainCfg {
    steps: usize,
    batch: u32,
    lr: f32,
    wd: f32,
    seed: u64,
    nc: u32,
    input: u32,
    out: String,
    weights: String,
    /// `""` = auto: `tiny` when training from scratch, the CHECKPOINT'S OWN
    /// layout when fine-tuning (so the pretrained tensors fit by construction).
    arch: String,
    freeze_backbone: bool,
    freeze_reg_head: bool,
    freeze_cls_hidden: bool,
    /// Class ids allowed to learn; empty = all of them.
    train_classes: Vec<u32>,
}

impl Default for TrainCfg {
    fn default() -> TrainCfg {
        TrainCfg {
            steps: 200,
            batch: 4,
            lr: 1e-3,
            wd: 1e-2,
            seed: 1337,
            nc: 0,
            input: 0,
            out: String::new(),
            weights: String::new(),
            arch: String::new(),
            freeze_backbone: false,
            freeze_reg_head: false,
            freeze_cls_hidden: false,
            train_classes: Vec::new(),
        }
    }
}

fn parse_train_flags(args: &[String], start: usize) -> TrainCfg {
    let mut cfg = TrainCfg::default();
    let mut i = start;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => cfg.out = val(args, &mut i, "--out"),
            "--weights" => cfg.weights = val(args, &mut i, "--weights"),
            "--steps" => cfg.steps = val(args, &mut i, "--steps").parse().unwrap_or(cfg.steps),
            "--batch" => cfg.batch = val(args, &mut i, "--batch").parse().unwrap_or(cfg.batch),
            "--lr" => cfg.lr = val(args, &mut i, "--lr").parse().unwrap_or(cfg.lr),
            "--wd" => cfg.wd = val(args, &mut i, "--wd").parse().unwrap_or(cfg.wd),
            "--seed" => cfg.seed = val(args, &mut i, "--seed").parse().unwrap_or(cfg.seed),
            "--nc" => cfg.nc = val(args, &mut i, "--nc").parse().unwrap_or(cfg.nc),
            "--input" => cfg.input = val(args, &mut i, "--input").parse().unwrap_or(cfg.input),
            "--arch" => cfg.arch = val(args, &mut i, "--arch"),
            "--freeze-backbone" => cfg.freeze_backbone = true,
            "--freeze-reg-head" => cfg.freeze_reg_head = true,
            "--freeze-cls-hidden" => cfg.freeze_cls_hidden = true,
            "--train-classes" => {
                cfg.train_classes = val(args, &mut i, "--train-classes")
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        s.trim().parse::<u32>().unwrap_or_else(|_| {
                            eprintln!("--train-classes: {s:?} is not a class id");
                            std::process::exit(2);
                        })
                    })
                    .collect()
            }
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if !args.iter().any(|a| a == "--seed") {
        cfg.seed = data::rng::random_seed();
        println!("yolov8 train: no --seed given, using random seed {} (pass --seed {} to reproduce)", cfg.seed, cfg.seed);
    }
    cfg
}

/// The architecture to build, before the class count / input size are applied.
///
/// Fine-tuning reads the layout off the PRETRAINED CHECKPOINT rather than
/// guessing: `YoloConfig::tiny` deliberately uses different channel widths from
/// the canonical `yolov8n`, so building `tiny` and then copying a real COCO
/// checkpoint into it matches almost nothing and silently trains from scratch.
fn base_config(cfg: &TrainCfg, pretrained: Option<&str>, data: &DetectData) -> YoloConfig {
    // From scratch there is no checkpoint to inherit a class count from, so the
    // dataset's own is the fallback (`--nc` still overrides, in `build_model`).
    let scratch_nc = if cfg.nc > 0 { cfg.nc } else { data.nc.max(1) };
    match cfg.arch.as_str() {
        "yolov8n" => return YoloConfig::yolov8n(),
        "tiny" => return YoloConfig::tiny(scratch_nc),
        "" => {}
        other => {
            eprintln!("brain yolov8: unknown --arch {other:?} (expected tiny|yolov8n)");
            std::process::exit(2);
        }
    }
    match pretrained {
        // The checkpoint carries its own config (`brain.config` metadata).
        Some(path) => YoloConfig::from_json(&checkpoint::read_config(path)),
        None => YoloConfig::tiny(scratch_nc),
    }
}

/// Build a `Yolo` for the dataset's geometry from `base`, at the requested class
/// count. Weights are random-seeded; `fine_tune` overwrites them from the
/// checkpoint afterwards.
fn build_model(base: YoloConfig, cfg: &TrainCfg, data: &DetectData) -> Yolo {
    let mut ycfg = base;
    if cfg.nc > 0 {
        ycfg.nc = cfg.nc;
    }
    ycfg.nc = ycfg.nc.max(1);
    if cfg.input > 0 {
        ycfg.input = cfg.input;
    } else {
        // Train at the dataset's own resolution so the CHW blob uploads directly
        // (no letterbox) - the synthetic generator's default is 128, matching
        // tiny's default input.
        ycfg.input = data.w;
    }
    let init = <Yolo as model::Model>::init_weights(&ycfg, cfg.seed);
    Yolo::new(ycfg, cfg.batch, 0, &init)
}

/// Apply the requested freezes/gates, reporting what each one covered.
fn apply_freezes(model: &mut Yolo, cfg: &TrainCfg, data: &DetectData) {
    if cfg.freeze_backbone {
        let n = model.freeze_backbone();
        println!("froze {n} backbone+neck tensors (gradients AND BatchNorm running stats)");
    }
    if cfg.freeze_reg_head {
        let n = model.freeze_reg_head();
        println!("froze {n} box/DFL head tensors (the box head is class-agnostic)");
    }
    if cfg.freeze_cls_hidden {
        let n = model.freeze_cls_hidden();
        println!("froze {n} class-branch hidden tensors (only the final per-class 1x1 trains)");
    }
    if !cfg.train_classes.is_empty() {
        model.train_only_classes(&cfg.train_classes);
        println!("training only class(es) {:?}; every other class's gradient is gated off", cfg.train_classes);
        if cfg.wd != 0.0 {
            eprintln!(
                "brain yolov8: WARNING --train-classes with --wd {} - AdamW's decoupled weight \
                 decay is not a gradient and still shrinks the gated classes. Pass --wd 0 to keep \
                 them bit-exact.",
                cfg.wd
            );
        }
    } else if model.cfg.nc as usize > used_classes(data).len() {
        eprintln!(
            "brain yolov8: WARNING this model has {} classes but the dataset only uses {}. The \
             detection loss scores BCE over ALL classes against an all-zero target for the absent \
             ones, so they will be trained to never fire. Pass --train-classes to gate them off.",
            model.cfg.nc,
            used_classes(data).len()
        );
    }
}

/// The distinct class ids that actually appear in the dataset's boxes.
fn used_classes(data: &DetectData) -> std::collections::BTreeSet<u32> {
    data.boxes.iter().flatten().map(|b| b.class).collect()
}

/// Run the shared training loop over the train split. Prints loss periodically
/// and returns `(first_loss, last_loss)`.
fn run_train_loop(model: &Yolo, data: &DetectData, cfg: &TrainCfg) -> (f32, f32) {
    let n_train = split_at(data.n).max(1);
    let b = cfg.batch as usize;
    let stride = data.image_stride();
    let side = model.cfg.input as usize;
    let want = b * 3 * side * side;
    let dims_match = (data.w as usize == side) && (data.h as usize == side);
    if !dims_match {
        eprintln!(
            "brain yolov8 train: WARNING dataset {}x{} != model input {side}; training only \
             supports matching geometry (regenerate the dataset at {side}px or pass \
             --input {})",
            data.w, data.h, data.w
        );
    }

    model.set_mode(LossMode::Detection);
    model.set_eval(false);
    // Accumulate BN running mean/var during training so eval-mode inference
    // (the saved checkpoint -> `Yolo::detect`) reads usable running stats.
    model.set_update_running(true);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    let mut img_batch = vec![0.0f32; want];

    for step in 0..cfg.steps {
        // Round-robin contiguous mini-batch over the train split.
        let base = (step * b) % n_train;
        let mut gts: Vec<GtBox> = Vec::new();
        for j in 0..b {
            let idx = (base + j) % n_train;
            let src = &data.images[idx * stride..idx * stride + stride.min(3 * side * side)];
            let dst = &mut img_batch[j * 3 * side * side..(j + 1) * 3 * side * side];
            let n = src.len().min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
            gts.extend(gts_for(data, idx, j as u32));
        }
        model.set_image(&img_batch);
        model.set_targets(&gts);
        model.zero_grads();
        let loss = model.forward();
        model.backward();
        model.adamw_step((step + 1) as u32, cfg.lr, cfg.wd, Some(1.0), 1.0);
        model.poll_wait();

        if first.is_nan() {
            first = loss;
        }
        last = loss;
        if step == 0 || (step + 1) % 10 == 0 || step + 1 == cfg.steps {
            println!("step {:>5}/{}  loss {:.4}", step + 1, cfg.steps, loss);
        }
    }
    (first, last)
}

fn train(args: &[String], pretrained: Option<&str>) {
    let Some(dir) = args.first().cloned() else {
        eprintln!("usage: brain yolov8 train <data_dir> --out F [--steps N --batch B --lr X --nc C --input S --seed S --arch tiny|yolov8n]");
        return;
    };
    let cfg = parse_train_flags(args, 1);
    if cfg.out.is_empty() {
        eprintln!("brain yolov8 train: --out <weights> is required");
        return;
    }

    let data = match load_dataset(Path::new(&dir)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("brain yolov8 train: loading {dir}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "training yolo on {dir}: n={} (train {}) {}x{} nc={} | steps={} batch={} lr={}",
        data.n, split_at(data.n), data.w, data.h, data.nc, cfg.steps, cfg.batch, cfg.lr
    );

    let mut model = build_model(base_config(&cfg, pretrained, &data), &cfg, &data);
    // Optional pretrained init for fine-tune: copy each matching tensor in,
    // widening the class head when this run adds classes.
    if let Some(path) = pretrained {
        let src = checkpoint::load(path).by_role("");
        let rep = yolov8::finetune::load_pretrained(&model, &src, cfg.seed);
        println!("fine-tune init from {path}: {}", rep.summary());
        if !rep.transferred_anything() {
            eprintln!(
                "brain yolov8 fine-tune: the checkpoint shares NO tensor with this model - \
                 that is a from-scratch run, not a fine-tune (architecture mismatch: pass \
                 --arch matching the checkpoint, or drop --weights)"
            );
            std::process::exit(1);
        }
        if !rep.mismatched.is_empty() {
            eprintln!("brain yolov8 fine-tune: {} tensor(s) left at random init:", rep.mismatched.len());
            for (n, want, got) in rep.mismatched.iter().take(10) {
                eprintln!("  {n}: model wants {want} elements, checkpoint has {got}");
            }
        }
    }
    apply_freezes(&mut model, &cfg, &data);
    let (i0, i1) = run_train_loop(&model, &data, &cfg);
    model.save(&cfg.out);
    println!("done: train loss {i0:.4} -> {i1:.4}; saved {}", cfg.out);
}

/// Fine-tune: load pretrained weights, then continue training on a new dataset.
///
/// Builds on the CHECKPOINT'S OWN architecture (so a real COCO `yolov8n`
/// actually loads), and `--nc` above the checkpoint's class count APPENDS
/// classes - the pretrained ones keep their indices and their weights, the new
/// ones get a fresh head (see `yolov8::finetune`).
///
/// `--freeze-backbone` / `--freeze-reg-head` are real: the named tensors are
/// withdrawn from the optimiser AND pinned to eval-mode BatchNorm, so they are
/// bit-for-bit unchanged by the run.
fn fine_tune(args: &[String]) {
    // `fine-tune <data_dir> --weights <pretrained> --out F ...`
    if args.is_empty() {
        eprintln!("usage: brain yolov8 fine-tune <data_dir> --weights <pretrained> --out F [--nc C --freeze-backbone --freeze-reg-head ...]");
        return;
    }
    let cfg = parse_train_flags(args, 1);
    if cfg.weights.is_empty() || cfg.out.is_empty() {
        eprintln!("brain yolov8 fine-tune: --weights <pretrained> and --out <weights> are required");
        return;
    }
    if !Path::new(&cfg.weights).exists() {
        eprintln!("brain yolov8 fine-tune: --weights {} not found", cfg.weights);
        std::process::exit(1);
    }
    // Reuse the train path, seeding the model from the pretrained checkpoint.
    let weights = cfg.weights.clone();
    train(args, Some(&weights));
}

fn eval(args: &[String]) {
    let mut weights = String::new();
    let mut data_dir = String::new();
    let mut conf = 0.25f32;
    let mut iou = 0.45f32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--data" => data_dir = val(args, &mut i, "--data"),
            "--conf" => conf = val(args, &mut i, "--conf").parse().unwrap_or(conf),
            "--iou" => iou = val(args, &mut i, "--iou").parse().unwrap_or(iou),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || data_dir.is_empty() {
        eprintln!("usage: brain yolov8 eval --weights F --data <dir> [--conf X --iou X]");
        return;
    }
    let data = match load_dataset(Path::new(&data_dir)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("brain yolov8 eval: loading {data_dir}: {e}");
            std::process::exit(1);
        }
    };
    // Batch=1 inference over the val split.
    let model = Yolo::load(&weights, 1);
    let side = model.cfg.input as usize;
    let nc = model.cfg.nc;
    let val0 = split_at(data.n);

    let mut all_preds: Vec<[f32; 6]> = Vec::new();
    let mut all_gts: Vec<EvalGt> = Vec::new();
    // Each image is scored in its OWN pixel coordinate frame; to score them
    // jointly with the model-free `map50`, we offset every image's boxes into a
    // disjoint horizontal strip so they never cross-match across images.
    let stride = data.image_stride();
    for (k, i) in (val0..data.n).enumerate() {
        let off = (k as f32) * (data.w as f32 + 16.0);
        // CHW -> HWC for detect (it expects interleaved RGB).
        let chw = &data.images[i * stride..(i + 1) * stride];
        let hwc = imaging::pixels::chw_to_hwc(chw, 3, data.h as usize, data.w as usize);
        let dets = model.detect(&hwc, data.w, data.h, conf, iou);
        for mut d in dets {
            d[0] += off;
            d[2] += off;
            all_preds.push(d);
        }
        for b in &data.boxes[i] {
            let cx = b.cx * data.w as f32 + off;
            let cy = b.cy * data.h as f32;
            let bw = b.w * data.w as f32;
            let bh = b.h * data.h as f32;
            all_gts.push(EvalGt {
                class: b.class,
                bbox: [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5],
            });
        }
    }
    let _ = side;
    let map = detection::map50(&all_preds, &all_gts, nc);
    let (p, r) = detection::precision_recall(&all_preds, &all_gts, 0.5);
    println!("metric        value");
    println!("mAP@0.5       {map:.4}");
    println!("precision@0.5 {p:.4}");
    println!("recall@0.5    {r:.4}");
    println!("preds {}  gts {}  (val images {})", all_preds.len(), all_gts.len(), data.n - val0);
}

/// The identity half of `detect`'s flags - all optional, and inert unless
/// `--identity-ref` names a reference embedding.
struct IdentityCfg {
    reference: String,
    name: String,
    class_name: String,
    threshold: f32,
    class: u32,
    margin: f32,
    arcface_dir: String,
}

impl Default for IdentityCfg {
    fn default() -> IdentityCfg {
        IdentityCfg {
            reference: String::new(),
            name: "identity".into(),
            class_name: "person".into(),
            // insightface's own antelopev2 recipe verifies at a cosine around
            // 0.4 on real photographs. The floor is exposed rather than pinned
            // because the population being verified sets it: a run over
            // generated or heavily compressed imagery separates lower than one
            // over camera originals, and the honest way to pick it is to look
            // at the two distributions on YOUR data.
            threshold: 0.35,
            class: 0,
            // A detector box is tight around the subject; a few percent of
            // context makes the face landmarker markedly more reliable at the
            // top edge, where a tight person box clips the hairline.
            margin: 0.05,
            arcface_dir: String::new(),
        }
    }
}

/// `arcface::caps::ArcFaceSession` behind `yolov8`'s embedder seam.
///
/// The face search runs ONCE for the frame and its result is reused for every
/// box - both because it is N times cheaper than a detector pass per box, and
/// because it is more accurate: SCRFD must see a face at its native capture
/// scale, and a per-box crop is upscaled onto the detector's canvas (see
/// `yolov8::identity`'s module docs for the measured numbers). Attribution is
/// then geometric - a face belongs to the person box its CENTRE falls inside -
/// which is what "the largest face in the frame" cannot express once a frame
/// holds several people.
///
/// No second face-embedding path lives here: detection is
/// `ArcFaceSession::detect_faces` and alignment + embedding is
/// `embed_face_chw`, the two halves `embed_raw_chw` itself composes.
struct ArcFaceBoxes {
    session: arcface::caps::ArcFaceSession,
    /// The frame's faces, found lazily on the first box and reused after.
    /// `detect` handles one image per process, so one slot is the whole cache.
    faces: std::cell::RefCell<Option<Vec<scrfd::Face>>>,
}

impl yolov8::identity::BoxEmbedder for ArcFaceBoxes {
    fn embed_in_box(&self, chw: &[f32], w: u32, h: u32, bbox: [f32; 4]) -> Result<Vec<f32>, String> {
        let mut cache = self.faces.borrow_mut();
        if cache.is_none() {
            let found = self.session.detect_faces(chw, w, h)?;
            eprintln!("  arcface: {} face(s) in the {w}x{h} frame", found.len());
            *cache = Some(found);
        }
        let inside: Vec<scrfd::Face> = cache
            .as_ref()
            .expect("filled above")
            .iter()
            .filter(|f| {
                let (cx, cy) = ((f.bbox[0] + f.bbox[2]) * 0.5, (f.bbox[1] + f.bbox[3]) * 0.5);
                cx >= bbox[0] && cx <= bbox[2] && cy >= bbox[1] && cy <= bbox[3]
            })
            .copied()
            .collect();
        if inside.is_empty() {
            return Err("no face inside this box".into());
        }
        // Several faces inside one person box (a crowd behind the subject) is
        // resolved the reference recipe's way: the largest is the subject.
        self.session.embed_face_chw(chw, w, h, &arcface::caps::primary_face(&inside))
    }
}

/// Read a `.bin` of little-endian f32 - the shape `brain arcface embed --out
/// embedding=…` writes, so a reference identity is produced by the embedder
/// itself rather than by a bespoke serializer.
fn read_f32_le(path: &str) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(format!("{path}: {} bytes is not a little-endian f32 vector", bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect())
}

/// Build the detect-then-verify gate + embedder, or exit with a real message.
///
/// The embedder rides on the detector's OWN device with its own kernel set
/// (`Gpu::new_like`): one process, one device (`AGENTS.md`), two kernel lists,
/// because each crate resolves its pipeline indices against the list it
/// registered and a shared handle would bind the wrong ones.
fn build_identity_gate(cfg: &IdentityCfg, model: &Yolo) -> (yolov8::identity::IdentityGate, ArcFaceBoxes) {
    let reference = read_f32_le(&cfg.reference).unwrap_or_else(|e| {
        eprintln!("brain yolov8 detect --identity-ref: {e}");
        std::process::exit(1);
    });
    let dir = if cfg.arcface_dir.is_empty() {
        crate::resident_arcface::dir_from_env().unwrap_or_else(|| {
            eprintln!(
                "brain yolov8 detect --identity-ref: no ArcFace weights - pass --arcface-dir <dir> \
                 (holding glintr100.onnx and scrfd_10g_bnkps.onnx) or set BRAIN_ARCFACE_DIR"
            );
            std::process::exit(1);
        })
    } else {
        cfg.arcface_dir.clone()
    };
    let gpu = model.gpu.new_like(&arcface::caps::SERVING_PIPELINES);
    let session = arcface::caps::ArcFaceSession::load(&dir, gpu).unwrap_or_else(|e| {
        eprintln!("brain yolov8 detect --identity-ref: loading ArcFace from {dir}: {e}");
        std::process::exit(1);
    });
    let gate = yolov8::identity::IdentityGate {
        reference,
        threshold: cfg.threshold,
        name: cfg.name.clone(),
        class_name: cfg.class_name.clone(),
        class: cfg.class,
        margin: cfg.margin,
    };
    (gate, ArcFaceBoxes { session, faces: std::cell::RefCell::new(None) })
}

fn detect(args: &[String]) {
    let mut weights = String::new();
    let mut image = String::new();
    let mut conf = 0.25f32;
    let mut iou = 0.45f32;
    let mut input = 0u32;
    let mut id = IdentityCfg::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--image" => image = val(args, &mut i, "--image"),
            "--conf" => conf = val(args, &mut i, "--conf").parse().unwrap_or(conf),
            "--iou" => iou = val(args, &mut i, "--iou").parse().unwrap_or(iou),
            "--input" => input = val(args, &mut i, "--input").parse().unwrap_or(input),
            "--identity-ref" => id.reference = val(args, &mut i, "--identity-ref"),
            "--identity-name" => id.name = val(args, &mut i, "--identity-name"),
            "--identity-class-name" => id.class_name = val(args, &mut i, "--identity-class-name"),
            "--identity-threshold" => id.threshold = val(args, &mut i, "--identity-threshold").parse().unwrap_or(id.threshold),
            "--identity-class" => id.class = val(args, &mut i, "--identity-class").parse().unwrap_or(id.class),
            "--identity-margin" => id.margin = val(args, &mut i, "--identity-margin").parse().unwrap_or(id.margin),
            "--arcface-dir" => id.arcface_dir = val(args, &mut i, "--arcface-dir"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || image.is_empty() {
        eprintln!("usage: brain yolov8 detect --weights F --image <path> [--conf X --iou X --input S]");
        eprintln!("         [--identity-ref <embedding.bin> --identity-name NAME --identity-threshold X]");
        eprintln!("         [--identity-class N --identity-class-name NAME --identity-margin X --arcface-dir DIR]");
        eprintln!("  <path> is a binary PPM (P6) or a detection dataset dir (uses image 0)");
        eprintln!("  add --device npu to compile+run on the Intel NPU via OpenVINO");
        return;
    }
    // `--device npu` routes through the OpenVINO NPU path (export fp32 -> compile).
    if crate::npu_explicit() {
        if !id.reference.is_empty() {
            eprintln!("brain yolov8 detect: --identity-ref is not available on --device npu (the ArcFace embedder has no NPU path)");
            std::process::exit(2);
        }
        return detect_via_npu(&weights, &image, conf, iou);
    }
    let model = Yolo::load_at(&weights, 1, input);

    // Accept either a binary PPM (P6) file or a detection-dataset directory (in
    // which case image 0 is used). PPM/raw decoding reuses the `events` codec.
    //
    // Deliberately the ORIGINAL image, at its own resolution: `Yolo::detect`
    // letterboxes internally and inverts the transform on the way out, so a
    // caller must NOT pre-letterbox - and identity verification searches these
    // same full-resolution pixels, where the face still has its detail.
    let (hwc, w, h) = match crate::image_io::load_image(&image) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain yolov8 detect: {e}");
            std::process::exit(1);
        }
    };
    let dets = model.detect(&hwc, w, h, conf, iou);
    print_dets(&dets);

    // Stage 2: who is each detected person? Only when a reference was named -
    // without it this is exactly the detector it has always been.
    let labeled = if id.reference.is_empty() {
        dets.iter().map(|&det| yolov8::identity::Labeled { det, label: None, similarity: None, verified: false }).collect()
    } else {
        let (gate, embedder) = build_identity_gate(&id, &model);
        let out = yolov8::identity::verify_detections(&gate, &embedder, &dets, &hwc, w, h);
        for l in &out {
            if l.det[5] as u32 != gate.class {
                continue;
            }
            match l.similarity {
                Some(s) => eprintln!(
                    "  class-{} box [{:.0},{:.0},{:.0},{:.0}] conf {:.3}: cosine {s:.4} vs {} -> {}",
                    gate.class, l.det[0], l.det[1], l.det[2], l.det[3], l.det[4], gate.name,
                    if l.verified { gate.name.as_str() } else { gate.class_name.as_str() }
                ),
                None => eprintln!(
                    "  class-{} box [{:.0},{:.0},{:.0},{:.0}] conf {:.3}: no face inside the box -> {}",
                    gate.class, l.det[0], l.det[1], l.det[2], l.det[3], l.det[4], gate.class_name
                ),
            }
        }
        out
    };
    // One machine-readable line, in `imageops draw_boxes`' own `boxes` shape -
    // matching `brain scrfd detect`'s `faces:` convention, so a caller pipes it
    // straight through instead of reformatting the per-detection lines above.
    println!("detections: {}", yolov8::identity::to_json(&labeled));
    eprintln!("brain yolov8 detect: {} detection(s) on {w}x{h}", dets.len());
}

/// `--device npu` route for `detect`: auto-export the weights to an fp32 ONNX and
/// run it on the Intel NPU via OpenVINO (host DFL-decode + NMS). For INT8, use
/// `brain npu quantize` + `brain npu run`.
fn detect_via_npu(weights: &str, image: &str, conf: f32, iou: f32) {
    let (hwc, w, h) = match crate::image_io::load_image(image) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain yolov8 detect: {e}");
            std::process::exit(1);
        }
    };
    let cfg = npu::openvino::NpuConfig { device: npu::openvino::NpuDevice::Npu, ..Default::default() };
    match npu::detect_weights_on_npu(weights, &hwc, w, h, conf, iou, &cfg, None) {
        Ok(dets) => {
            print_dets(&dets);
            eprintln!("brain yolov8 detect (--device npu): {} detection(s) on {w}x{h}", dets.len());
        }
        Err(e) => {
            eprintln!("brain yolov8 detect --device npu: {e}");
            std::process::exit(1);
        }
    }
}

/// Print one JSON line per detection: `[x1,y1,x2,y2,conf,class]`.
fn print_dets(dets: &[[f32; 6]]) {
    for d in dets {
        println!("[{:.2},{:.2},{:.2},{:.2},{:.4},{}]", d[0], d[1], d[2], d[3], d[4], d[5] as u32);
    }
}
