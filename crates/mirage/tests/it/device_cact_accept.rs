// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running aggressively quantised perception models on constrained
// devices for its clients. If your team needs expertise in low-bit deployment of vision models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Acceptance of the natively served deployment archive: agreement with the reference engine's recorded answers,
//! quality against the FP32 weights, and throughput against the FP32 path. The numbers are printed (and recorded in
//! the Mirage ledger); the assertions are the floors they must not fall below.

use std::path::{Path, PathBuf};

use mirage::describe::Options;
use mirage::device::{Device, Image, MirageModel};
use mirage::model::{load_dir, Mirage, WeightSource};
use serde_json::Value;

use crate::common::released_dir;

type Golden = std::collections::HashMap<String, Vec<f32>>;

fn model_on(dir: &Path, source: WeightSource) -> MirageModel {
    let l = load_dir(dir, source).expect("loads");
    MirageModel::new(Mirage::new(&l.checkpoint, &l.tokenizer, l.charset).unwrap(), gpu_core::Gpu::new(&Device::kernels())).unwrap()
}

fn iou(a: &[f64], b: &[f64]) -> f64 {
    let (w, h) = ((a[2].min(b[2]) - a[0].max(b[0])).max(0.0), (a[3].min(b[3]) - a[1].max(b[1])).max(0.0));
    let inter = w * h;
    inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter).max(1e-9)
}

fn boxes(v: &Value, list: &str) -> Vec<(Vec<f64>, Value)> {
    v[list].as_array().into_iter().flatten().map(|o| (o["bbox"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap()).collect(), o.clone())).collect()
}

/// Of the `want` boxes, how many `got` contains exactly, and how many it matches at IoU 0.5 or more.
fn proposal_agreement(got: &Value, want: &Value) -> (usize, usize, usize, usize) {
    let (mut total, mut exact, mut near, mut lines) = (0, 0, 0, 0);
    for list in ["objects", "text"] {
        let (g, w) = (boxes(got, list), boxes(want, list));
        for (wb, wo) in &w {
            total += 1;
            exact += g.iter().any(|(gb, _)| gb == wb) as usize;
            if let Some((_, go)) = g.iter().filter(|(gb, _)| iou(gb, wb) >= 0.5).max_by(|a, b| iou(&a.0, wb).total_cmp(&iou(&b.0, wb))) {
                near += 1;
                let same = |k: &str| go.get(k) == wo.get(k);
                lines += (same("name") && same("position") && same("description") && same("string")) as usize;
            }
        }
    }
    (total, exact, near, lines)
}

fn corpus_cases() -> Vec<(String, PathBuf)> {
    let root = PathBuf::from(brain_testutil::testdata("golden/mirage/engine"));
    let mut cases: Vec<(String, PathBuf)> = std::fs::read_dir(&root)
        .map(|d| d.flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), e.path())).filter(|(n, p)| n != "cats_form" && p.join("out.json").is_file()).collect())
        .unwrap_or_default();
    cases.sort();
    cases
}

fn rgb_of(g: &Golden) -> Vec<u8> {
    g["input_rgb"].iter().map(|&v| v as u8).collect()
}

/// Run every recorded engine case through `model` and compare with the engine's JSON (masks and scores aside).
fn engine_agreement(model: &MirageModel, label: &str) -> (usize, usize, usize, usize, usize) {
    let (cats, text) = (crate::decoder::golden("engine agreement", "cats").unwrap(), crate::decoder::golden("engine agreement", "text").unwrap());
    let (mut total, mut exact, mut near, mut lines, mut identical) = (0, 0, 0, 0, 0);
    let cases = corpus_cases();
    for (name, path) in &cases {
        let request = std::fs::read_to_string(path.join("request.txt")).unwrap();
        let mut it = request.lines();
        let mut opts = Options::from_json(it.next().unwrap()).unwrap();
        opts.language = Some(it.next().unwrap().to_string());
        let (rgb, dims) = if name.starts_with("cats") { (rgb_of(&cats), (480, 640)) } else { (rgb_of(&text), (600, 800)) };
        let got = serde_json::to_value(model.describe_batch(&[Image { rgb: &rgb, height: dims.0, width: dims.1 }], &opts).pop().unwrap().unwrap()).unwrap();
        let want: Value = serde_json::from_slice(&std::fs::read(path.join("out.json")).unwrap()).unwrap();
        let (t, e, n, l) = proposal_agreement(&got, &want);
        let strip = |mut v: Value| {
            for list in ["objects", "text"] {
                for item in v[list].as_array_mut().into_iter().flatten() {
                    let item = item.as_object_mut().unwrap();
                    item.remove("score");
                    item.remove("mask");
                }
            }
            v
        };
        let same = strip(got) == strip(want);
        eprintln!("{label} {name}: {t} boxes, {e} exact, {n} at IoU 0.5, {l} with identical name/position/description; JSON identical apart from masks and scores: {same}");
        (total, exact, near, lines, identical) = (total + t, exact + e, near + n, lines + l, identical + same as usize);
    }
    eprintln!("{label} total over {} cases: {total} boxes, {exact} exact, {near} at IoU 0.5, {lines} identical lines, {identical} identical documents", cases.len());
    (total, exact, near, lines, identical)
}

#[test]
fn the_packed_model_agrees_with_the_reference_engine() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact acceptance: the released checkpoint is absent");
        return;
    };
    if corpus_cases().is_empty() {
        brain_testutil::skip("cact acceptance: the engine corpus is absent (make testdata/fetch)");
        return;
    }
    let (total, _, near, lines, _) = engine_agreement(&model_on(&dir, WeightSource::Cact), "packed CQ");
    // The engine rounds activations to int8; the packed model keeps them f32, so exact agreement is not expected.
    assert!(near * 10 >= total * 9, "proposal recall {near}/{total}");
    assert!(lines * 10 >= near * 8, "identical lines {lines}/{near}");
    let (t32, _, n32, l32, _) = engine_agreement(&model_on(&dir, WeightSource::Safetensors), "FP32");
    eprintln!("reference: the FP32 model agrees with the engine on {n32}/{t32} boxes and {l32} lines");
}

fn softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().copied().fold(f32::MIN, f32::max);
    let e: Vec<f64> = row.iter().map(|&v| f64::from(v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

#[test]
fn quality_against_fp32_on_the_goldens() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact acceptance: the released checkpoint is absent");
        return;
    };
    let fp32 = Mirage::open_with(&dir, WeightSource::Safetensors).unwrap();
    let cq = Mirage::open_with(&dir, WeightSource::Cact).unwrap();
    let vocab = fp32.decoder().vocabulary();
    for name in ["cats", "text"] {
        let Some(g) = crate::decoder::golden("cact quality", name) else { return };
        let tokens: Vec<u32> = g["tokens"].iter().map(|&t| t as u32).collect();
        let (a, b) = (fp32.logits(&g["memory"], &tokens), cq.logits(&g["memory"], &tokens));
        let positions = a.len() / vocab;
        let (mut agree, mut kl, mut fp32_ok, mut cq_ok) = (0usize, 0f64, 0usize, 0usize);
        let argmax = |row: &[f32]| row.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        for t in 0..positions - 1 {
            let (ra, rb) = (&a[t * vocab..(t + 1) * vocab], &b[t * vocab..(t + 1) * vocab]);
            agree += (argmax(ra) == argmax(rb)) as usize;
            let (p, q) = (softmax(ra), softmax(rb));
            kl += p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-300)).ln()).sum::<f64>();
            fp32_ok += (argmax(ra) as u32 == tokens[t + 1]) as usize;
            cq_ok += (argmax(rb) as u32 == tokens[t + 1]) as usize;
        }
        let n = positions - 1;
        eprintln!("cact quality {name}: {n} teacher-forced positions; top-1 agreement with FP32 {agree}/{n}; mean KL(FP32 || CQ) {:.5} nats; next-token accuracy FP32 {fp32_ok}/{n}, CQ {cq_ok}/{n}", kl / n as f64);
        assert!(agree * 100 >= n * 85, "{name}: top-1 agreement {agree}/{n}");
    }
}

#[test]
fn proposals_of_the_packed_model_cover_the_fp32_proposals() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact acceptance: the released checkpoint is absent");
        return;
    };
    let (m32, mcq) = (model_on(&dir, WeightSource::Safetensors), model_on(&dir, WeightSource::Cact));
    let opts = Options::from_json(r#"{"max_tokens":600}"#).unwrap();
    for (name, dims) in [("cats", (480, 640)), ("text", (600, 800))] {
        let g = crate::decoder::golden("cact proposals", name).unwrap();
        let rgb = rgb_of(&g);
        let image = [Image { rgb: &rgb, height: dims.0, width: dims.1 }];
        let a = serde_json::to_value(m32.describe_batch(&image, &opts).pop().unwrap().unwrap()).unwrap();
        let b = serde_json::to_value(mcq.describe_batch(&image, &opts).pop().unwrap().unwrap()).unwrap();
        let (total, exact, near, lines) = proposal_agreement(&b, &a);
        eprintln!("cact proposals {name}: of {total} FP32 boxes the packed model has {exact} exact, {near} at IoU 0.5, {lines} identical lines");
        assert!(near * 10 >= total * 7, "{name}: recall {near}/{total}");
    }
}

/// Images per second of `describe_batch` for the FP32 and the packed model on an accelerator.
#[test]
fn describe_throughput_packed_against_fp32() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact throughput: the released checkpoint is absent");
        return;
    };
    let Some(cats) = crate::decoder::golden("cact throughput", "cats") else { return };
    if gpu_core::Gpu::new(&Device::kernels()).kind() == "cpu" {
        brain_testutil::skip_unavailable("cact throughput: the CPU backend is not a throughput target");
        return;
    }
    let rgb = rgb_of(&cats);
    let opts = Options::from_json(r#"{"fields":["bbox","score","position","description"],"max_tokens":600}"#).unwrap();
    for (label, source) in [("FP32", WeightSource::Safetensors), ("packed CQ", WeightSource::Cact)] {
        let model = model_on(&dir, source);
        let _ = model.describe_batch(&[Image { rgb: &rgb, height: 480, width: 640 }; 2], &opts);
        let one = model.describe_batch_with(&[Image { rgb: &rgb, height: 480, width: 640 }], &opts, &mirage::device::BatchControl::default()).pop().unwrap().unwrap();
        eprintln!("cact throughput {label}: one image decodes {} tokens", one.decoder_tokens);
        for n in [1usize, 8, 64] {
            let images: Vec<Image> = (0..n).map(|_| Image { rgb: &rgb, height: 480, width: 640 }).collect();
            let mut best = f64::MAX;
            let mut timings = mirage::device::Timings::default();
            for _ in 0..3 {
                let started = std::time::Instant::now();
                assert!(model.describe_batch(&images, &opts).iter().all(Result::is_ok));
                let wall = started.elapsed().as_secs_f64();
                if wall < best {
                    (best, timings) = (wall, model.last_timings());
                }
            }
            eprintln!("cact throughput {label} batch {n}: {best:.3} s, {:.2} images/s (vision {:.3} memory {:.3} decode {:.3} post {:.3})", n as f64 / best, timings.vision, timings.memory, timings.decode, timings.post);
        }
    }
}

/// The packed model's batch-1 describe, alone, for a profiler (`nsys profile` on this test binary).
#[test]
#[ignore = "profiling entry point: run under nsys/ncu"]
fn packed_describe_for_profiling() {
    let Some(dir) = released_dir() else { return };
    let Some(cats) = crate::decoder::golden("cact profile", "cats") else { return };
    let rgb = rgb_of(&cats);
    let model = model_on(&dir, WeightSource::Cact);
    let opts = Options::from_json(r#"{"fields":["bbox","score","position","description"],"max_tokens":600}"#).unwrap();
    let images = [Image { rgb: &rgb, height: 480, width: 640 }];
    for _ in 0..3 {
        assert!(model.describe_batch(&images, &opts).iter().all(Result::is_ok));
    }
}
