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
use mirage::model::{load_precision, Mirage, Precision, WeightSource};
use serde_json::Value;

use crate::common::released_dir;

type Golden = std::collections::HashMap<String, Vec<f32>>;

fn model_on(dir: &Path, precision: impl Into<Precision>) -> MirageModel {
    let precision = precision.into();
    let l = load_precision(dir, precision).expect("loads");
    MirageModel::new(Mirage::new(&l.checkpoint, &l.tokenizer, l.charset).unwrap(), gpu_core::Gpu::new(&Device::kernels())).unwrap().with_cq_activations(precision.activations())
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
        .map(|d| d.flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), e.path())).filter(|(n, p)| !n.starts_with("cats_form") && p.join("out.json").is_file()).collect())
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

/// The precision a session is loaded at decides the activation mode of its packed products: int8 only for `cq-a8`.
#[test]
fn a_session_runs_int8_activations_only_at_cq_a8() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact a8 session: the released checkpoint is absent");
        return;
    };
    let dir = dir.to_string_lossy().into_owned();
    for (precision, want) in [(Precision::Fp32, model::cq::CqActivations::F32), (Precision::Cq, model::cq::CqActivations::F32), (Precision::CqA8, model::cq::CqActivations::A8)] {
        let session = mirage::caps::MirageSession::load_with(&dir, gpu_core::Gpu::new(&mirage::caps::MirageSession::kernels()), precision).expect("loads");
        assert_eq!(session.model().device().ops().cq_activations(), want, "{}", precision.name());
    }
    assert_eq!(Precision::parse("cq-a8"), Some(Precision::CqA8));
    assert_eq!(Precision::parse("a8"), None, "only the documented names parse");
    assert_eq!(Precision::default(), Precision::Fp32, "int8 activations are never the default");
}

/// The released archive's 2-bit codebook (its header, 4 f32 levels).
const BOOK2: [f32; 4] = [-0.133_172_14, -0.039_902_09, 0.040_035_28, 0.133_460_45];

/// The int8 activation arithmetic is the deployment engine's: a product recorded from the released engine serving the
/// cats photograph (two activation rows of a decoder projection, eight rows of its 2-bit weight, the engine's outputs;
/// captured by breaking in the engine's batched CQ product and reading its operands, see the `what` field of the file) is
/// reproduced by `Cq::matmul_a8` to f32 rounding, where the f32-activation product is 2e-3 of the largest output away.
#[test]
fn the_a8_product_reproduces_a_recorded_engine_product() {
    let doc: Value = serde_json::from_str(include_str!("../golden/vendor_cq_a8_matmul.json")).expect("the recorded product parses");
    let numbers = |key: &str| -> Vec<f64> { doc[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect() };
    let bytes = |key: &str| -> Vec<u8> { numbers(key).iter().map(|&v| v as u8).collect() };
    let floats = |key: &str| -> Vec<f32> { numbers(key).iter().map(|&v| v as f32).collect() };
    let (m, n, k) = (doc["rows"].as_u64().unwrap() as usize, doc["n"].as_u64().unwrap() as usize, doc["k"].as_u64().unwrap() as usize);
    let (x, y) = (floats("x"), floats("y"));
    let cq = model::cq::Cq::from_parts(n, k, 2, &bytes("index_bytes"), &bytes("norm_bytes"), &BOOK2).expect("the recorded weight is a valid 2-bit matrix");
    let scale = y.iter().fold(0f32, |a, v| a.max(v.abs()));
    let worst = |got: &[f32]| got.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
    let (a8, f32_product) = (worst(&cq.matmul_a8(&x, m)), worst(&cq.matmul_rotated(&x, m)));
    eprintln!("recorded engine product: A8 emulation {a8:e}, f32 activations {f32_product:e} of the largest output");
    assert!(a8 < 1e-6, "the A8 emulation is {a8:e} from the engine");
    assert!(f32_product > 100.0 * a8, "the recorded product must tell the two apart (f32 {f32_product:e})");
}

/// The opt-in A8 precision (int8 activations per rotated group, as the engine runs them) against the engine's recorded
/// answers, beside the f32-activation packed model on the same cases. Both are printed (recorded in the ledger); the
/// assertions are the floors the A8 model must keep. Exact agreement is NOT expected to improve: only the products over
/// packed matrices round their activations here, while the engine also rounds those of the convolutions (dense in this
/// model) and of attention, so the rounding errors are independent of the engine's instead of identical to them.
#[test]
fn the_a8_packed_model_agrees_with_the_reference_engine() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact a8 acceptance: the released checkpoint is absent");
        return;
    };
    if corpus_cases().is_empty() {
        brain_testutil::skip("cact a8 acceptance: the engine corpus is absent (make testdata/fetch)");
        return;
    }
    let (total, exact, near, lines, identical) = engine_agreement(&model_on(&dir, Precision::CqA8), "packed CQ A8");
    let (_, exact_f32, near_f32, lines_f32, identical_f32) = engine_agreement(&model_on(&dir, Precision::Cq), "packed CQ f32 activations");
    eprintln!("a8 against f32 activations: boxes exact {exact} vs {exact_f32}, at IoU 0.5 {near} vs {near_f32} of {total}, identical lines {lines} vs {lines_f32}, identical documents {identical} vs {identical_f32}");
    assert!(near * 10 >= total * 9, "A8 proposal recall {near}/{total}");
    assert!(lines * 10 >= near * 8, "A8 identical lines {lines}/{near}");
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

/// Teacher-forced decoder logits of the A8 packed model against the FP32 model and against the packed model with f32
/// activations (host path; the numbers are recorded in the ledger).
#[test]
fn a8_quality_against_fp32_and_against_f32_activations_on_the_goldens() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("cact a8 quality: the released checkpoint is absent");
        return;
    };
    let open = |p: Precision| Mirage::open_precision(&dir, p).unwrap();
    let (fp32, cq, a8) = (open(Precision::Fp32), open(Precision::Cq), open(Precision::CqA8));
    let vocab = fp32.decoder().vocabulary();
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    let kl = |p: &[f32], q: &[f32]| -> f64 { softmax(p).iter().zip(softmax(q)).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-300)).ln()).sum() };
    for name in ["cats", "text"] {
        let Some(g) = crate::decoder::golden("cact a8 quality", name) else { return };
        let tokens: Vec<u32> = g["tokens"].iter().map(|&t| t as u32).collect();
        let logits: Vec<Vec<f32>> = [&fp32, &cq, &a8].iter().map(|m| m.logits(&g["memory"], &tokens)).collect();
        let n = logits[0].len() / vocab - 1;
        let row = |l: &[f32], t: usize| -> Vec<f32> { l[t * vocab..(t + 1) * vocab].to_vec() };
        // (top-1 agreement, summed KL(reference || model), next-token hits) of `model` against `reference`.
        let compare = |reference: usize, model: usize| {
            let (mut agree, mut sum) = (0usize, 0f64);
            for t in 0..n {
                let (a, b) = (row(&logits[reference], t), row(&logits[model], t));
                agree += (argmax(&a) == argmax(&b)) as usize;
                sum += kl(&a, &b);
            }
            (agree, sum / n as f64)
        };
        let hits = |model: usize| (0..n).filter(|&t| argmax(&row(&logits[model], t)) as u32 == tokens[t + 1]).count();
        let (cq_vs_fp32, a8_vs_fp32, a8_vs_cq) = (compare(0, 1), compare(0, 2), compare(1, 2));
        eprintln!(
            "cact a8 quality {name}: {n} teacher-forced positions; vs FP32: CQ f32-act top-1 {}/{n} KL {:.5}, CQ A8 top-1 {}/{n} KL {:.5}; A8 vs CQ f32-act: top-1 {}/{n} KL {:.5}; next-token accuracy FP32 {}/{n}, CQ {}/{n}, CQ A8 {}/{n}",
            cq_vs_fp32.0,
            cq_vs_fp32.1,
            a8_vs_fp32.0,
            a8_vs_fp32.1,
            a8_vs_cq.0,
            a8_vs_cq.1,
            hits(0),
            hits(1),
            hits(2)
        );
        assert!(a8_vs_fp32.0 * 100 >= n * 80, "{name}: A8 top-1 agreement with FP32 {}/{n}", a8_vs_fp32.0);
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
    for (label, precision) in [("FP32", Precision::Fp32), ("packed CQ", Precision::Cq), ("packed CQ A8", Precision::CqA8)] {
        let model = model_on(&dir, precision);
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
