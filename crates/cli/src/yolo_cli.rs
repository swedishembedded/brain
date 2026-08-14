// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain yolov8 …` - train / evaluate / run the from-scratch YOLOv8-style
//! detector. Mirrors the `gpt_cli` flag-parsing idiom; respects the global
//! `--device cpu|gpu` flag handled in `main.rs` (the YOLO model itself only ever
//! instantiates the CPU backend today, see `Yolo::new`).
//!
//!   brain yolov8 train <data_dir> --out F [--steps N --batch B --lr X --nc C
//!                                          --input S --seed S]
//!   brain yolov8 eval  --weights F --data <dir> [--conf X --iou X]
//!   brain yolov8 detect --weights F --image <path> [--conf X --iou X]
//!   brain yolov8 fine-tune <data_dir> --weights F --out F [--freeze-backbone ...]
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
}

impl Default for TrainCfg {
    fn default() -> TrainCfg {
        TrainCfg { steps: 200, batch: 4, lr: 1e-3, wd: 1e-2, seed: 1337, nc: 0, input: 0 }
    }
}

fn parse_train_flags(args: &[String], start: usize, cfg: &mut TrainCfg, out: &mut String, weights: &mut String, freeze: &mut bool) {
    let mut i = start;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => *out = val(args, &mut i, "--out"),
            "--weights" => *weights = val(args, &mut i, "--weights"),
            "--steps" => cfg.steps = val(args, &mut i, "--steps").parse().unwrap_or(cfg.steps),
            "--batch" => cfg.batch = val(args, &mut i, "--batch").parse().unwrap_or(cfg.batch),
            "--lr" => cfg.lr = val(args, &mut i, "--lr").parse().unwrap_or(cfg.lr),
            "--wd" => cfg.wd = val(args, &mut i, "--wd").parse().unwrap_or(cfg.wd),
            "--seed" => cfg.seed = val(args, &mut i, "--seed").parse().unwrap_or(cfg.seed),
            "--nc" => cfg.nc = val(args, &mut i, "--nc").parse().unwrap_or(cfg.nc),
            "--input" => cfg.input = val(args, &mut i, "--input").parse().unwrap_or(cfg.input),
            "--freeze-backbone" => *freeze = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
}

/// Build a tiny `Yolo` from a (possibly pretrained) init for the dataset's
/// geometry. When `init` is `None`, weights are random-seeded.
fn build_model(cfg: &TrainCfg, data: &DetectData) -> Yolo {
    let nc = if cfg.nc > 0 { cfg.nc } else { data.nc.max(1) };
    let mut ycfg = YoloConfig::tiny(nc);
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
        eprintln!("usage: brain yolov8 train <data_dir> --out F [--steps N --batch B --lr X --nc C --input S --seed S]");
        return;
    };
    let mut cfg = TrainCfg::default();
    let mut out = String::new();
    let mut weights = String::new();
    let mut freeze = false;
    parse_train_flags(args, 1, &mut cfg, &mut out, &mut weights, &mut freeze);
    if out.is_empty() {
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

    let model = build_model(&cfg, &data);
    // Optional pretrained init for fine-tune: copy each matching tensor in.
    if let Some(path) = pretrained {
        load_pretrained_into(&model, path);
    }
    let (i0, i1) = run_train_loop(&model, &data, &cfg);
    model.save(&out);
    println!("done: train loss {i0:.4} -> {i1:.4}; saved {out}");
}

/// Fine-tune: load pretrained weights, then continue training the (whole)
/// network on a new dataset. `--freeze-backbone` is accepted but, since the
/// model exposes no per-parameter freeze API, it currently has no effect beyond
/// a one-line notice (documented limitation).
fn fine_tune(args: &[String]) {
    // `fine-tune <data_dir> --weights <pretrained> --out F ...`
    if args.is_empty() {
        eprintln!("usage: brain yolov8 fine-tune <data_dir> --weights <pretrained> --out F [flags]");
        return;
    }
    let mut cfg = TrainCfg::default();
    let mut out = String::new();
    let mut weights = String::new();
    let mut freeze = false;
    parse_train_flags(args, 1, &mut cfg, &mut out, &mut weights, &mut freeze);
    if weights.is_empty() || out.is_empty() {
        eprintln!("brain yolov8 fine-tune: --weights <pretrained> and --out <weights> are required");
        return;
    }
    if freeze {
        eprintln!(
            "brain yolov8 fine-tune: --freeze-backbone has no effect (no per-param freeze \
             API); fine-tuning the whole network from the pretrained init"
        );
    }
    // Reuse the train path, seeding the model from the pretrained checkpoint.
    train(args, Some(&weights));
}

/// Copy every tensor present in BOTH the checkpoint and the model into the
/// model's weights (shape-matched by element count). Tensors that do not match
/// (e.g. a different class count `nc`) are left at their random init.
fn load_pretrained_into(model: &Yolo, path: &str) {
    let c = checkpoint::load(path);
    let init = c.by_role("");
    let mut copied = 0usize;
    for name in <Yolo as model::Model>::param_names(model) {
        if let Some(w) = init.get(&name) {
            let cur = model.read_weight(&name);
            if cur.len() == w.len() {
                model.write_weight(&name, w);
                copied += 1;
            }
        }
    }
    eprintln!("brain yolov8 fine-tune: loaded {copied} pretrained tensors from {path}");
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

fn detect(args: &[String]) {
    let mut weights = String::new();
    let mut image = String::new();
    let mut conf = 0.25f32;
    let mut iou = 0.45f32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--image" => image = val(args, &mut i, "--image"),
            "--conf" => conf = val(args, &mut i, "--conf").parse().unwrap_or(conf),
            "--iou" => iou = val(args, &mut i, "--iou").parse().unwrap_or(iou),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || image.is_empty() {
        eprintln!("usage: brain yolov8 detect --weights F --image <path> [--conf X --iou X]");
        eprintln!("  <path> is a binary PPM (P6) or a detection dataset dir (uses image 0)");
        eprintln!("  add --device npu to compile+run on the Intel NPU via OpenVINO");
        return;
    }
    // `--device npu` routes through the OpenVINO NPU path (export fp32 -> compile).
    if crate::npu_explicit() {
        return detect_via_npu(&weights, &image, conf, iou);
    }
    let model = Yolo::load(&weights, 1);

    // Accept either a binary PPM (P6) file or a detection-dataset directory (in
    // which case image 0 is used). PPM/raw decoding reuses the `events` codec.
    let (hwc, w, h) = match crate::image_io::load_image(&image) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain yolov8 detect: {e}");
            std::process::exit(1);
        }
    };
    let dets = model.detect(&hwc, w, h, conf, iou);
    print_dets(&dets);
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
