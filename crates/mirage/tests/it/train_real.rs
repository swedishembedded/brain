// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Acceptance experiments on the released weights and the prepared fine-tuning subsets (COCO val2017 objects,
//! synthetic text lines). Each takes minutes of host compute, so they are `#[ignore]`d; run them with
//! `make test/one P=brain-mirage T=it ARGS="train_real --ignored --nocapture"`. Every number they print is
//! recorded in the crate's ledger.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mirage::import::{load_dir, Checkpoint, ImportOptions};
use mirage::device::train::trainer::DevTrainer;
use mirage::device::Device;
use mirage::train::data::{load_picture, read_jsonl, Record};
use mirage::train::eval::{average_precision_50, character_error_rate, mask_iou};
use mirage::train::objective::{rasterise, TrainWeights};
use mirage::train::prepare::{prepare, Resources};
use mirage::train::trainer::{FineTuner, Head, Mode, Prepared, TrainConfig, Trainer};
use model::saved::SavedPolicy;
use mirage::vision::masks::{coefficient_cell, object_field, round_box};
use mirage::vision::preprocess::{canvas, letterbox, CANVAS};
use mirage::vision::reader::crop_rect;
use mirage::vision::{ProposalOptions, VisionModel};

/// Threads for the experiments (the machine is shared; the unit tests use 8).
const THREADS: usize = 32;

struct Env {
    dir: PathBuf,
    ck: Checkpoint,
    res: Resources,
    coco: PathBuf,
    ocr: PathBuf,
}

fn env(test: &str) -> Option<Env> {
    let _ = backend_cpu::par::set_global_threads(THREADS);
    let dir = PathBuf::from(brain_testutil::model_dir(crate::common::REFERENCE)?);
    let shared = dir.ancestors().nth(4)?.to_path_buf(); // <shared>/containers/brain-models/<org>/<name> is four levels below <shared>
    let coco = shared.join("resources/datasets/coco2017/mirage-subset");
    let ocr = shared.join("resources/datasets/mirage-ocr-synth");
    if !coco.join("train.jsonl").is_file() || !ocr.join("train.jsonl").is_file() {
        brain_testutil::skip(&format!("{test}: the fine-tuning subsets are not under {}", shared.display()));
        return None;
    }
    let ck = load_dir(&dir, &ImportOptions::default()).ok()?;
    let res = Resources::open(&dir, &ck).ok()?;
    Some(Env { dir, ck, res, coco, ocr })
}

fn records(root: &Path, split: &str, n: usize) -> Vec<Record> {
    let mut r = read_jsonl(&root.join(format!("{split}.jsonl"))).expect("the subset parses");
    r.truncate(n);
    r
}

fn prepared(e: &Env, root: &Path, recs: &[Record]) -> Vec<Prepared> {
    recs.iter().map(|r| prepare(r, root, &e.ck.config, &e.res).expect("record prepares")).collect()
}

/// Held-out AP50 of class-agnostic object proposals.
fn detection_ap50(ck: &Checkpoint, root: &Path, recs: &[Record]) -> f64 {
    let vm = VisionModel::new(ck);
    let (mut dets, mut truths) = (Vec::new(), Vec::new());
    for rec in recs {
        let pic = load_picture(root, rec).unwrap();
        let img = letterbox(&pic.rgb, pic.h, pic.w).unwrap();
        let opts = ProposalOptions { object_threshold: 0.05, text_threshold: 0.05, max_objects: 64, max_text: 32 };
        let a = vm.analyze(&img, opts);
        let (sx, sy) = (CANVAS as f64 / img.w as f64 * 1000.0, CANVAS as f64 / img.h as f64 * 1000.0);
        dets.push(a.objects.iter().map(|p| (p.score, [f64::from(p.bbox[0]) * sx, f64::from(p.bbox[1]) * sy, f64::from(p.bbox[2]) * sx, f64::from(p.bbox[3]) * sy])).collect());
        truths.push(rec.objects.iter().flatten().map(|o| o.bbox).collect());
    }
    average_precision_50(&dets, &truths)
}

/// Held-out character error rate of the reader on the labelled lines.
fn reader_cer(ck: &Checkpoint, root: &Path, recs: &[Record], charset: &[char]) -> f64 {
    let vm = VisionModel::new(ck);
    let mut pairs = Vec::new();
    for rec in recs {
        let pic = load_picture(root, rec).unwrap();
        for t in rec.text.iter().flatten() {
            let rect = crop_rect(t.bbox.map(|v| v as f32), pic.w, pic.h);
            let read = vm.reader.read_line(&pic.rgb, pic.h, pic.w, rect, charset).unwrap_or_default();
            pairs.push((t.value.clone(), read));
        }
    }
    character_error_rate(&pairs)
}

/// Held-out mask IoU of the labelled objects' masks, predicted inside the labelled boxes.
fn mask_miou(ck: &Checkpoint, root: &Path, recs: &[Record]) -> f64 {
    let vm = VisionModel::new(ck);
    let (mut total, mut n) = (0.0, 0usize);
    for rec in recs {
        let pic = load_picture(root, rec).unwrap();
        let img = letterbox(&pic.rgb, pic.h, pic.w).unwrap();
        let enc = vm.encoder.forward(&canvas(&img), img.h, img.w);
        let dense = vm.dense.forward(&enc, img.h, img.w);
        let (protos, coefs) = vm.dense.mask_maps(&dense.head_in, &enc, img.h, img.w);
        let (sx, sy) = (img.w as f64 / CANVAS as f64 / 1000.0, img.h as f64 / CANVAS as f64 / 1000.0);
        for o in rec.objects.iter().flatten() {
            let Some(polys) = &o.mask_polygons else { continue };
            let b = round_box([(o.bbox[0] * sx) as f32, (o.bbox[1] * sy) as f32, (o.bbox[2] * sx) as f32, (o.bbox[3] * sy) as f32]);
            let (r, c) = coefficient_cell(b);
            let coef = &coefs[(r * 64 + c) * 32..][..32];
            let field = object_field(&protos, coef, b);
            let rings: Vec<Vec<f64>> = polys.iter().map(|p| p.as_chunks::<2>().0.iter().flat_map(|q| [q[0] * sx, q[1] * sy]).collect()).collect();
            total += mask_iou(&field, &rasterise(&rings, 128));
            n += 1;
        }
    }
    total / n.max(1) as f64
}

fn run(trainer: &mut dyn FineTuner, train: &[Prepared], steps: usize, batch: usize, label: &str) {
    let start = Instant::now();
    for s in 0..steps {
        let b: Vec<&Prepared> = (0..batch).map(|k| &train[(s * batch + k) % train.len()]).collect();
        let owned: Vec<Prepared> = b.iter().map(|p| Prepared { sample: p.sample.clone(), target: p.target.clone() }).collect();
        let r = trainer.step(&owned).unwrap();
        println!("RESULT {label} step {} loss {:.4} grad_norm {:.3} terms {:?} elapsed {:.0}s", s + 1, r.loss, r.grad_norm, r.terms, start.elapsed().as_secs_f64());
    }
    println!("RESULT {label} samples/s {:.3}", (steps * batch) as f64 / start.elapsed().as_secs_f64());
}

fn weights(text: f64, heat: f64, regression: f64, mask: f64, ctc: f64) -> TrainWeights {
    TrainWeights { text, heat, regression, mask, ctc }
}

fn train_config(mode: Mode, w: TrainWeights, lr: f32) -> TrainConfig {
    let mut cfg = TrainConfig::new(mode);
    cfg.weights = w;
    cfg.lr = lr;
    cfg
}

/// The host trainer, or with `device` the device trainer on the device the process selected
/// (`BRAIN_BACKEND` / `BRAIN_DEVICE`).
fn trainer(e: &Env, device: bool, mode: Mode, w: TrainWeights, lr: f32, saved: SavedPolicy) -> Box<dyn FineTuner> {
    let mut t: Box<dyn FineTuner> = if device {
        let dev = Device::open().expect("a device");
        println!("RESULT device: {}", dev.gpu().kind());
        Box::new(DevTrainer::new(dev, e.ck.clone(), train_config(mode, w, lr)).unwrap())
    } else {
        Box::new(Trainer::new(e.ck.clone(), train_config(mode, w, lr)).unwrap())
    };
    t.set_saved_policy(saved).unwrap();
    println!("RESULT saved policy: {}", saved.label());
    t
}

fn detection_fine_tune_raises_held_out_ap50(device: bool, saved: SavedPolicy) {
    let Some(e) = env("detection") else { return };
    let (train_recs, val_recs) = (records(&e.coco, "train", 120), records(&e.coco, "val", 40));
    let (train, val) = (prepared(&e, &e.coco, &train_recs), prepared(&e, &e.coco, &val_recs));
    let mut t = trainer(&e, device, Mode::Heads(vec![Head::Dense]), weights(0.0, 1.0, 1.0, 0.0, 0.0), 3e-4, saved);
    let (ap0, loss0) = (detection_ap50(&e.ck, &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT detection before: held-out AP50 {ap0:.4}, held-out loss {loss0:.4}");
    run(t.as_mut(), &train, 60, 4, "detection");
    let (ap1, loss1) = (detection_ap50(&t.checkpoint(), &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT detection after: held-out AP50 {ap1:.4}, held-out loss {loss1:.4}");
    assert!(loss1 < loss0 && ap1 > ap0, "AP50 {ap0} -> {ap1}, loss {loss0} -> {loss1}");
}

fn ocr_fine_tune_lowers_held_out_cer(device: bool, saved: SavedPolicy) {
    let Some(e) = env("ocr") else { return };
    let (train_recs, val_recs) = (records(&e.ocr, "train", 120), records(&e.ocr, "val", 40));
    let (train, val) = (prepared(&e, &e.ocr, &train_recs), prepared(&e, &e.ocr, &val_recs));
    let mut t = trainer(&e, device, Mode::Heads(vec![Head::Reader]), weights(0.0, 0.0, 0.0, 0.0, 1.0), 3e-4, saved);
    let (cer0, loss0) = (reader_cer(&e.ck, &e.ocr, &val_recs, &e.res.charset), t.eval_loss(&val));
    println!("RESULT ocr before: held-out CER {cer0:.4}, held-out CTC {loss0:.4}");
    run(t.as_mut(), &train, 60, 4, "ocr");
    let (cer1, loss1) = (reader_cer(&t.checkpoint(), &e.ocr, &val_recs, &e.res.charset), t.eval_loss(&val));
    println!("RESULT ocr after: held-out CER {cer1:.4}, held-out CTC {loss1:.4}");
    assert!(loss1 < loss0 && cer1 < cer0, "CER {cer0} -> {cer1}, loss {loss0} -> {loss1}");
}

fn mask_fine_tune_raises_held_out_iou(device: bool, saved: SavedPolicy) {
    let Some(e) = env("mask") else { return };
    let (train_recs, val_recs) = (records(&e.coco, "train", 120), records(&e.coco, "val", 40));
    let (train, val) = (prepared(&e, &e.coco, &train_recs), prepared(&e, &e.coco, &val_recs));
    let mut t = trainer(&e, device, Mode::Heads(vec![Head::Dense]), weights(0.0, 0.0, 0.0, 1.0, 0.0), 3e-4, saved);
    let (iou0, loss0) = (mask_miou(&e.ck, &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT mask before: held-out mask IoU {iou0:.4}, held-out loss {loss0:.4}");
    run(t.as_mut(), &train, 60, 4, "mask");
    let (iou1, loss1) = (mask_miou(&t.checkpoint(), &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT mask after: held-out mask IoU {iou1:.4}, held-out loss {loss1:.4}");
    assert!(loss1 < loss0 && iou1 > iou0, "IoU {iou0} -> {iou1}, loss {loss0} -> {loss1}");
}

fn a_lora_fine_tune_is_served_by_the_model_and_equals_the_training_path(device: bool, saved: SavedPolicy) {
    use mirage::decoder::Positions;
    use mirage::model::Mirage;
    use mirage::train::net::{params_from_checkpoint, Net};
    use mirage::vision::memory::Slot;
    use mirage::vision::ops::valid_cells;

    let Some(e) = env("lora") else { return };
    let (train_recs, val_recs) = (records(&e.coco, "train", 60), records(&e.coco, "val", 12));
    let (train, val) = (prepared(&e, &e.coco, &train_recs), prepared(&e, &e.coco, &val_recs));
    let mut t = trainer(&e, device, Mode::Lora { rank: 8, alpha: 16.0, seed: 3 }, weights(1.0, 0.0, 0.0, 0.0, 0.0), 2e-3, saved);
    let loss0 = t.eval_loss(&val);
    println!("RESULT lora before: held-out text loss {loss0:.4}");
    run(t.as_mut(), &train, 30, 2, "lora");
    let loss1 = t.eval_loss(&val);
    println!("RESULT lora after: held-out text loss {loss1:.4}");
    assert!(loss1 < loss0, "held-out text loss {loss0} -> {loss1}");

    // The run directory is a model directory: the model opens it and folds the adapter in.
    let run_dir = std::env::temp_dir().join(format!("mirage-lora-run-{}", std::process::id()));
    t.save(&run_dir).unwrap();
    mirage::train::prepare::make_servable(&e.dir, &run_dir).unwrap();
    let served = Mirage::open(&run_dir).unwrap();

    // Training path: the live adapter on the f32 graph, teacher-forced on the first held-out sample. Serving
    // path: the folded host model on the same ground-truth slots.
    let p = &val[0];
    let (h, w) = (p.sample.rh, p.sample.rw);
    let cfg = &e.ck.config;
    let params = params_from_checkpoint::<f32>(&e.ck);
    let adapter = t.adapter_set();
    let mut net = Net::<f32>::new(cfg, &params, adapter.as_ref());
    let canvas = net.constant(p.sample.canvas.clone());
    let enc = net.encoder(canvas, h, w);
    let dense = net.dense(&enc, h, w);
    let base = cfg.text + mirage::config::LANGUAGE_TAGS;
    let slots: Vec<Slot> = p.sample.objects.iter().flatten().enumerate().map(|(k, o)| Slot { bbox: o.bbox, token: base + k }).collect();
    let valid = valid_cells(h, w, cfg.grid());
    let memory = net.decoder_memory(&enc, dense.head_in, &valid, &slots);
    let tokens = &p.target.as_ref().expect("held-out sample has a target").tokens;
    let rows = net.tape.len(memory) / cfg.dim;
    let live = net.decoder_logits(&tokens[..tokens.len() - 1], memory, rows);

    let vm = served.vision();
    let host_enc = vm.encoder.forward(&p.sample.canvas, h, w);
    let host_dense = vm.dense.forward(&host_enc, h, w);
    let host_mem = vm.memory.build(&host_enc.encoder, &valid, &host_dense.head_in, &slots);
    let dec = served.decoder();
    let cross = dec.cross_memory(&host_mem, host_mem.len() / cfg.dim);
    let folded = dec.forward(&mut dec.new_state(), &cross, &tokens[..tokens.len() - 1], Positions::All);
    let (cos_head, _) = crate::vision::compare(net.tape.val(dense.head), &host_dense.head);
    let (cos_logits, max_abs) = crate::vision::compare(net.tape.val(live), &folded);
    println!("RESULT lora served-vs-training: head cosine {cos_head:.8}, decoder logits cosine {cos_logits:.8} (max_abs {max_abs:.2e})");
    assert!(cos_head >= 0.99999 && cos_logits >= 0.99999);
    std::fs::remove_dir_all(&run_dir).unwrap();
}

fn a_resumed_run_on_the_released_weights_ends_exactly_where_the_uninterrupted_run_does(device: bool, saved: SavedPolicy) {
    let Some(e) = env("resume") else { return };
    let train = prepared(&e, &e.coco, &records(&e.coco, "train", 3));
    let make = || trainer(&e, device, Mode::Heads(vec![Head::Dense, Head::Reader]), weights(1.0, 1.0, 1.0, 1.0, 0.0), 1e-4, saved);
    let step = |t: &mut dyn FineTuner, k: usize| {
        t.step(std::slice::from_ref(&train[k])).unwrap();
    };
    let mut straight = make();
    (0..3).for_each(|k| step(straight.as_mut(), k));
    let mut first = make();
    (0..2).for_each(|k| step(first.as_mut(), k));
    let dir = std::env::temp_dir().join(format!("mirage-resume-{}", std::process::id()));
    first.save(&dir).unwrap();
    let mut cfg = TrainConfig::new(Mode::Heads(vec![Head::Dense, Head::Reader]));
    cfg.weights = weights(1.0, 1.0, 1.0, 1.0, 0.0);
    cfg.lr = 1e-4;
    let mut resumed: Box<dyn FineTuner> = if device { Box::new(DevTrainer::resume(Device::open().expect("a device"), &dir, cfg).unwrap()) } else { Box::new(Trainer::resume(&dir, cfg).unwrap()) };
    step(resumed.as_mut(), 2);
    let identical = resumed.checkpoint().tensors == straight.checkpoint().tensors;
    println!("RESULT resume: weights after step 3 identical to the uninterrupted run: {identical}");
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(identical);
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn step_time_probe() {
    let Some(e) = env("step_time_probe") else { return };
    let recs = records(&e.coco, "train", 2);
    let t0 = Instant::now();
    let train = prepared(&e, &e.coco, &recs);
    println!("RESULT prepare {:.1}s for {} samples", t0.elapsed().as_secs_f64(), train.len());
    for (label, mode, w) in [
        ("probe-dense", Mode::Heads(vec![Head::Dense]), weights(0.0, 1.0, 1.0, 0.0, 0.0)),
        ("probe-all", Mode::Full, weights(1.0, 1.0, 1.0, 1.0, 1.0)),
    ] {
        let mut t = trainer(&e, false, mode, w, 1e-5, SavedPolicy::Keep);
        run(t.as_mut(), &train, 2, 1, label);
    }
    let _ = &e.dir;
}

// Each experiment on the host trainer and on the device the process selects (`BRAIN_BACKEND`, `BRAIN_DEVICE`).

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn host_detection() {
    detection_fine_tune_raises_held_out_ap50(false, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment on the device the process selects"]
fn device_detection() {
    detection_fine_tune_raises_held_out_ap50(true, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn host_ocr() {
    ocr_fine_tune_lowers_held_out_cer(false, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment on the device the process selects"]
fn device_ocr() {
    ocr_fine_tune_lowers_held_out_cer(true, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn host_mask() {
    mask_fine_tune_raises_held_out_iou(false, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment on the device the process selects"]
fn device_mask() {
    mask_fine_tune_raises_held_out_iou(true, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn host_lora() {
    a_lora_fine_tune_is_served_by_the_model_and_equals_the_training_path(false, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment on the device the process selects"]
fn device_lora() {
    a_lora_fine_tune_is_served_by_the_model_and_equals_the_training_path(true, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn host_resume() {
    a_resumed_run_on_the_released_weights_ends_exactly_where_the_uninterrupted_run_does(false, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment on the device the process selects"]
fn device_resume() {
    a_resumed_run_on_the_released_weights_ends_exactly_where_the_uninterrupted_run_does(true, SavedPolicy::Keep);
}

#[test]
#[ignore = "acceptance experiment under recompute: minutes of host compute"]
fn host_detection_recompute() {
    detection_fine_tune_raises_held_out_ap50(false, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute on the device the process selects"]
fn device_detection_recompute() {
    detection_fine_tune_raises_held_out_ap50(true, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute: minutes of host compute"]
fn host_ocr_recompute() {
    ocr_fine_tune_lowers_held_out_cer(false, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute on the device the process selects"]
fn device_ocr_recompute() {
    ocr_fine_tune_lowers_held_out_cer(true, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute: minutes of host compute"]
fn host_mask_recompute() {
    mask_fine_tune_raises_held_out_iou(false, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute on the device the process selects"]
fn device_mask_recompute() {
    mask_fine_tune_raises_held_out_iou(true, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute: minutes of host compute"]
fn host_lora_recompute() {
    a_lora_fine_tune_is_served_by_the_model_and_equals_the_training_path(false, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute on the device the process selects"]
fn device_lora_recompute() {
    a_lora_fine_tune_is_served_by_the_model_and_equals_the_training_path(true, SavedPolicy::Recompute);
}

#[test]
#[ignore = "acceptance experiment under recompute on the device the process selects"]
fn device_resume_recompute() {
    a_resumed_run_on_the_released_weights_ends_exactly_where_the_uninterrupted_run_does(true, SavedPolicy::Recompute);
}

/// Resident set and its peak of this process, in MiB (`VmRSS`, `VmHWM`).
fn host_memory_mib() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| status.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0;
    (field("VmRSS:"), field("VmHWM:"))
}

/// Steps of 2 samples timed after the warm-up step; the mean seconds per step.
fn timed_steps(t: &mut dyn FineTuner, batch: &dyn Fn(usize) -> Vec<Prepared>) -> f64 {
    let measured = 3;
    let start = Instant::now();
    for s in 1..=measured {
        t.step(&batch(s)).unwrap();
    }
    start.elapsed().as_secs_f64() / measured as f64
}

/// One full-model training step (every tensor trained, every loss term, batch 2) after a warm-up step: peak host
/// memory and step time on the host trainer, or peak device memory and step time on the device trainer, under
/// `saved`. The warm-up pays for kernel compilation and allocator growth; the peaks restart after it.
fn step_memory_probe(device: bool, saved: SavedPolicy) {
    let Some(e) = env("step memory") else { return };
    let train = prepared(&e, &e.coco, &records(&e.coco, "train", 4));
    let cfg = train_config(Mode::Full, weights(1.0, 1.0, 1.0, 1.0, 1.0), 1e-5);
    let batch = |s: usize| -> Vec<Prepared> { (0..2).map(|k| &train[(2 * s + k) % train.len()]).map(|p| Prepared { sample: p.sample.clone(), target: p.target.clone() }).collect() };
    let label = format!("{} {}", if device { "device" } else { "host" }, saved.label());
    let per_step = if device {
        let mut t = DevTrainer::new(Device::open().expect("a device"), e.ck.clone(), cfg).unwrap();
        t.set_saved_policy(saved).unwrap();
        t.step(&batch(0)).unwrap();
        let _ = std::fs::write("/proc/self/clear_refs", "5");
        t.device().gpu().reset_memory_peak();
        let per_step = timed_steps(&mut t, &batch);
        let usage = t.device().gpu().memory_usage().expect("the backend accounts for its allocations");
        println!("RESULT memory {label}: device live {:.0} MiB, peak live {:.0} MiB, reserved {:.0} MiB", usage.live as f64 / 1048576.0, usage.peak_live as f64 / 1048576.0, usage.reserved as f64 / 1048576.0);
        per_step
    } else {
        let mut t = Trainer::new(e.ck.clone(), cfg).unwrap();
        t.set_saved_policy(saved).unwrap();
        // The host allocator keeps what a step freed, so the peak is taken from this point over the whole run: the
        // difference to the resident set here is what the steps add.
        let _ = std::fs::write("/proc/self/clear_refs", "5");
        println!("RESULT memory {label}: resident before the first step {:.0} MiB", host_memory_mib().0);
        t.step(&batch(0)).unwrap();
        timed_steps(&mut t, &batch)
    };
    let (_, host_peak) = host_memory_mib();
    println!("RESULT memory {label}: host peak {host_peak:.0} MiB, {per_step:.2} s per step of 2 samples, {:.3} samples/s", 2.0 / per_step);
}

#[test]
#[ignore = "memory measurement: run one policy per process (the host peak is per process)"]
fn host_step_memory_keep() {
    step_memory_probe(false, SavedPolicy::Keep);
}

#[test]
#[ignore = "memory measurement: run one policy per process (the host peak is per process)"]
fn host_step_memory_recompute() {
    step_memory_probe(false, SavedPolicy::Recompute);
}

#[test]
#[ignore = "memory measurement on the device the process selects"]
fn device_step_memory_keep() {
    step_memory_probe(true, SavedPolicy::Keep);
}

#[test]
#[ignore = "memory measurement on the device the process selects"]
fn device_step_memory_recompute() {
    step_memory_probe(true, SavedPolicy::Recompute);
}

/// A few full-model steps on the selected device: the workload `nsys` profiles (see the ledger).
#[test]
#[ignore = "profiling workload: run on the device the process selects (BRAIN_BACKEND, BRAIN_DEVICE)"]
fn device_step_profile() {
    let Some(e) = env("device_step_profile") else { return };
    let train = prepared(&e, &e.coco, &records(&e.coco, "train", 2));
    let mut t = trainer(&e, true, Mode::Full, weights(1.0, 1.0, 1.0, 1.0, 1.0), 1e-5, SavedPolicy::Keep);
    run(t.as_mut(), &train, 3, 2, "device-profile");
}
