// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Acceptance experiments on the released weights and the prepared fine-tuning subsets (COCO val2017 objects,
//! synthetic text lines). Each takes minutes of host compute, so they are `#[ignore]`d; run them with
//! `make test/one P=brain-mirage T=it ARGS="train_real --ignored --nocapture"`. Every number they print is
//! recorded in the crate's ledger.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mirage::import::{load_dir, Checkpoint, ImportOptions};
use mirage::train::data::{load_picture, read_jsonl_clamped, Record};
use mirage::train::eval::{average_precision_50, character_error_rate, mask_iou};
use mirage::train::objective::{rasterise, TrainWeights};
use mirage::train::prepare::{prepare, Resources};
use mirage::train::trainer::{Head, Mode, Prepared, TrainConfig, Trainer};
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
    let mut r = read_jsonl_clamped(&root.join(format!("{split}.jsonl"))).expect("the subset parses");
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
            let rings: Vec<Vec<f64>> = polys.iter().map(|p| p.chunks_exact(2).flat_map(|q| [q[0] * sx, q[1] * sy]).collect()).collect();
            total += mask_iou(&field, &rasterise(&rings, 128));
            n += 1;
        }
    }
    total / n.max(1) as f64
}

fn run(trainer: &mut Trainer, train: &[Prepared], steps: usize, batch: usize, label: &str) {
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

fn trainer(e: &Env, mode: Mode, w: TrainWeights, lr: f32) -> Trainer {
    let mut cfg = TrainConfig::new(mode);
    cfg.weights = w;
    cfg.lr = lr;
    Trainer::new(e.ck.clone(), cfg).unwrap()
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn detection_fine_tune_raises_held_out_ap50() {
    let Some(e) = env("detection") else { return };
    let (train_recs, val_recs) = (records(&e.coco, "train", 120), records(&e.coco, "val", 40));
    let (train, val) = (prepared(&e, &e.coco, &train_recs), prepared(&e, &e.coco, &val_recs));
    let mut t = trainer(&e, Mode::Heads(vec![Head::Dense]), weights(0.0, 1.0, 1.0, 0.0, 0.0), 3e-4);
    let (ap0, loss0) = (detection_ap50(&e.ck, &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT detection before: held-out AP50 {ap0:.4}, held-out loss {loss0:.4}");
    run(&mut t, &train, 60, 4, "detection");
    let (ap1, loss1) = (detection_ap50(&t.checkpoint(), &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT detection after: held-out AP50 {ap1:.4}, held-out loss {loss1:.4}");
    assert!(loss1 < loss0 && ap1 > ap0, "AP50 {ap0} -> {ap1}, loss {loss0} -> {loss1}");
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn ocr_fine_tune_lowers_held_out_cer() {
    let Some(e) = env("ocr") else { return };
    let (train_recs, val_recs) = (records(&e.ocr, "train", 120), records(&e.ocr, "val", 40));
    let (train, val) = (prepared(&e, &e.ocr, &train_recs), prepared(&e, &e.ocr, &val_recs));
    let mut t = trainer(&e, Mode::Heads(vec![Head::Reader]), weights(0.0, 0.0, 0.0, 0.0, 1.0), 3e-4);
    let (cer0, loss0) = (reader_cer(&e.ck, &e.ocr, &val_recs, &e.res.charset), t.eval_loss(&val));
    println!("RESULT ocr before: held-out CER {cer0:.4}, held-out CTC {loss0:.4}");
    run(&mut t, &train, 60, 4, "ocr");
    let (cer1, loss1) = (reader_cer(&t.checkpoint(), &e.ocr, &val_recs, &e.res.charset), t.eval_loss(&val));
    println!("RESULT ocr after: held-out CER {cer1:.4}, held-out CTC {loss1:.4}");
    assert!(loss1 < loss0 && cer1 < cer0, "CER {cer0} -> {cer1}, loss {loss0} -> {loss1}");
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn mask_fine_tune_raises_held_out_iou() {
    let Some(e) = env("mask") else { return };
    let (train_recs, val_recs) = (records(&e.coco, "train", 120), records(&e.coco, "val", 40));
    let (train, val) = (prepared(&e, &e.coco, &train_recs), prepared(&e, &e.coco, &val_recs));
    let mut t = trainer(&e, Mode::Heads(vec![Head::Dense]), weights(0.0, 0.0, 0.0, 1.0, 0.0), 3e-4);
    let (iou0, loss0) = (mask_miou(&e.ck, &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT mask before: held-out mask IoU {iou0:.4}, held-out loss {loss0:.4}");
    run(&mut t, &train, 60, 4, "mask");
    let (iou1, loss1) = (mask_miou(&t.checkpoint(), &e.coco, &val_recs), t.eval_loss(&val));
    println!("RESULT mask after: held-out mask IoU {iou1:.4}, held-out loss {loss1:.4}");
    assert!(loss1 < loss0 && iou1 > iou0, "IoU {iou0} -> {iou1}, loss {loss0} -> {loss1}");
}

#[test]
#[ignore = "acceptance experiment: minutes of host compute"]
fn a_lora_fine_tune_is_served_by_the_model_and_equals_the_training_path() {
    use mirage::decoder::Positions;
    use mirage::model::Mirage;
    use mirage::train::net::{params_from_checkpoint, Net};
    use mirage::vision::memory::Slot;
    use mirage::vision::ops::valid_cells;

    let Some(e) = env("lora") else { return };
    let (train_recs, val_recs) = (records(&e.coco, "train", 60), records(&e.coco, "val", 12));
    let (train, val) = (prepared(&e, &e.coco, &train_recs), prepared(&e, &e.coco, &val_recs));
    let mut t = trainer(&e, Mode::Lora { rank: 8, alpha: 16.0, seed: 3 }, weights(1.0, 0.0, 0.0, 0.0, 0.0), 2e-3);
    let loss0 = t.eval_loss(&val);
    println!("RESULT lora before: held-out text loss {loss0:.4}");
    run(&mut t, &train, 30, 2, "lora");
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
    let mut net = Net::<f32>::new(cfg, &params, t.adapter());
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
        let mut t = trainer(&e, mode, w, 1e-5);
        run(&mut t, &train, 2, 1, label);
    }
    let _ = &e.dir;
}
