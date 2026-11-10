// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `MirageModel` end to end: a batch of images through the device equals the
//! host model image by image (identical proposals, tokens, text and bitmap
//! run-length codes) and equals the same images run alone.

use mirage::describe::Options;
use mirage::device::{Device, Image, MirageModel, WeightForm};
use mirage::model::Mirage;
use serde_json::Value;

use crate::common::released_dir;

/// Foreground pixel count of a bitmap `[size, counts]` mask.
fn foreground(mask: &Value) -> u64 {
    mask["counts"].as_array().map_or(0, |c| c.iter().skip(1).step_by(2).map(|v| v.as_u64().unwrap()).sum())
}

/// Equal up to what accelerator rounding may move: scores at the third decimal and
/// mask pixels at the threshold. The CPU backend runs the same arithmetic as the host and must be exact.
fn assert_same(exact: bool, got: &Value, want: &Value, what: &str) {
    if exact {
        assert!(got == want, "{what}: the device result differs from the host");
        return;
    }
    let (mut g, mut w) = (got.clone(), want.clone());
    for (side, v) in [(&mut g, got), (&mut w, want)] {
        for list in ["objects", "text"] {
            for (k, item) in side[list].as_array_mut().into_iter().flatten().enumerate() {
                let (score, mask) = (item["score"].as_f64().unwrap(), &v[list][k]["mask"]);
                assert!(score > 0.0);
                if mask.is_object() {
                    let n = foreground(mask) as f64;
                    item["mask"] = Value::from((n / 200.0).round());
                } else {
                    item.as_object_mut().unwrap().remove("mask");
                }
                item.as_object_mut().unwrap().remove("score");
            }
        }
    }
    assert!(g == w, "{what}: the device result differs from the host beyond rounding");
}

fn json<T: serde::Serialize, E: std::fmt::Display>(r: Result<T, E>) -> Value {
    serde_json::to_value(r.unwrap_or_else(|e| panic!("{e}"))).unwrap()
}

#[test]
fn a_device_batch_describes_exactly_as_the_host_does_image_by_image() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device model: the released checkpoint is absent");
        return;
    };
    let (Some(cats), Some(text)) = (crate::decoder::golden("device model", "cats"), crate::decoder::golden("device model", "text")) else { return };
    crate::common::cap_threads();
    let gpu = gpu_core::Gpu::new(&Device::kernels());
    let kind = gpu.kind().to_string();
    eprintln!("device model: device kind {kind}");
    let host = Mirage::open(&dir).unwrap();
    let model = MirageModel::new(Mirage::open(&dir).unwrap(), gpu).unwrap();
    let rgb: Vec<Vec<u8>> = [&cats, &text].iter().map(|g| g["input_rgb"].iter().map(|&v| v as u8).collect()).collect();
    let dims = [(480usize, 640usize), (600, 800)];
    let images: Vec<Image> = rgb.iter().zip(dims).map(|(r, (h, w))| Image { rgb: r, height: h, width: w }).collect();
    let opts = Options::from_json(r#"{"fields":["bbox","score","mask","position","description"],"mask_format":"bitmap","max_tokens":600}"#).unwrap();

    let started = std::time::Instant::now();
    let batch: Vec<Value> = model.describe_batch(&images, &opts).into_iter().map(json).collect();
    eprintln!("describe_batch of 2: {:.1}s", started.elapsed().as_secs_f64());
    for (i, (r, (h, w))) in rgb.iter().zip(dims).enumerate() {
        let want = json(host.describe(r, h, w, &opts));
        assert_same(kind == "cpu", &batch[i], &want, &format!("image {i}"));
        let objects = want["objects"].as_array().map_or(0, Vec::len);
        let lines = want["text"].as_array().map_or(0, Vec::len);
        eprintln!("image {i}: {objects} objects, {lines} text lines match the host ({})", if kind == "cpu" { "exact, bitmap run-length codes included" } else { "scores and mask pixels within rounding" });
    }
    // A batch of N equals N single runs.
    for (i, image) in images.iter().enumerate() {
        let single = json(model.describe_batch(std::slice::from_ref(image), &opts).remove(0));
        assert_same(kind == "cpu", &single, &batch[i], &format!("image {i} alone vs in a batch"));
    }
}

#[test]
fn a_device_batch_extracts_a_form_as_the_host_does() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device extract: the released checkpoint is absent");
        return;
    };
    let Some(cats) = crate::decoder::golden("device extract", "cats") else { return };
    crate::common::cap_threads();
    let gpu = gpu_core::Gpu::new(&Device::kernels());
    let exact = gpu.kind() == "cpu";
    let host = Mirage::open(&dir).unwrap();
    let model = MirageModel::new(Mirage::open(&dir).unwrap(), gpu).unwrap();
    let rgb: Vec<u8> = cats["input_rgb"].iter().map(|&v| v as u8).collect();
    let form = serde_json::json!({"animal": {"enum": ["dog", "cat", "bird"]}, "n": {"type": "integer", "minimum": 0, "maximum": 5}});
    let opts = Options::from_json(&serde_json::json!({ "form": form }).to_string()).unwrap();
    let image = Image { rgb: &rgb, height: 480, width: 640 };
    let got = json(model.extract_batch(&[image, image], &[form.clone(), form], &opts).remove(1));
    let want = json(host.extract(&rgb, 480, 640, &opts));
    for field in ["animal", "n"] {
        assert_eq!(got["form"][field]["value"], want["form"][field]["value"], "{field}: the chosen value differs");
        let (g, w) = (got["form"][field]["probabilities"].as_object().unwrap(), want["form"][field]["probabilities"].as_object().unwrap());
        for (k, p) in w {
            let tol = if exact { 1e-4 } else { 2e-2 };
            assert!((g[k].as_f64().unwrap() - p.as_f64().unwrap()).abs() < tol, "{field}.{k}: {} vs {p}", g[k]);
        }
    }
}

/// Every field kind, two different images in one batch (and one image under a second form): each answer equals the
/// host scorer's for the same image, whatever way the cohorts split. Candidates of different token lengths make
/// the committed answers of the images diverge in length when their winners differ.
#[test]
fn a_device_batch_scores_every_field_kind_as_the_host_does() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device forms: the released checkpoint is absent");
        return;
    };
    let (Some(cats), Some(text)) = (crate::decoder::golden("device forms", "cats"), crate::decoder::golden("device forms", "text")) else { return };
    crate::common::cap_threads();
    let gpu = gpu_core::Gpu::new(&Device::kernels());
    let exact = gpu.kind() == "cpu";
    let host = Mirage::open(&dir).unwrap();
    let model = MirageModel::new(Mirage::open(&dir).unwrap(), gpu).unwrap();
    let rgb = |g: &std::collections::HashMap<String, Vec<f32>>| -> Vec<u8> { g["input_rgb"].iter().map(|&v| v as u8).collect() };
    let (cats_rgb, text_rgb) = (rgb(&cats), rgb(&text));
    let forms = [
        serde_json::json!({"thing": {"enum": ["dog", "cat", "bird", "remote", "newspaper"]}, "where": {"type": "string", "format": "box", "x-source": "thing"}, "n": {"type": "integer", "minimum": 0, "maximum": 5}, "label": {"type": "string", "maxLength": 12}}),
        serde_json::json!({"thing": {"enum": ["dog", "cat", "bird", "remote", "newspaper"]}, "where": {"type": "string", "format": "box", "x-source": "thing"}, "n": {"type": "integer", "minimum": 0, "maximum": 5}, "label": {"type": "string", "maxLength": 12}}),
        serde_json::json!({"animal": {"enum": ["dog", "cat"]}}),
    ];
    let images = [Image { rgb: &cats_rgb, height: 480, width: 640 }, Image { rgb: &text_rgb, height: 600, width: 800 }, Image { rgb: &cats_rgb, height: 480, width: 640 }];
    let opts = Options::from_json("{}").unwrap();
    let got = model.extract_batch(&images, &forms, &opts);
    let tol = if exact { 1e-4 } else { 3e-2 };
    for (i, (g, form)) in got.into_iter().zip(&forms).enumerate() {
        let one = Options::from_json(&serde_json::json!({ "form": form }).to_string()).unwrap();
        let (got, want) = (json(g), json(host.extract(images[i].rgb, images[i].height, images[i].width, &one)));
        for (field, w) in want["form"].as_object().unwrap() {
            let g = &got["form"][field];
            assert_eq!(g["value"], w["value"], "image {i} {field}: the value differs");
            match w.get("probabilities") {
                Some(table) => {
                    for (k, p) in table.as_object().unwrap() {
                        assert!((g["probabilities"][k].as_f64().unwrap() - p.as_f64().unwrap()).abs() < tol, "image {i} {field}.{k}: {} vs {p}", g["probabilities"][k]);
                    }
                }
                None => assert!((g["probability"].as_f64().unwrap() - w["probability"].as_f64().unwrap()).abs() < tol, "image {i} {field}: probability {} vs {}", g["probability"], w["probability"]),
            }
        }
    }
}

/// The profiling workload: eight images in one batch (four of each fixture). Duplicates must
/// describe identically, which also holds the batch-8 path to the single-image answer.
#[test]
fn a_batch_of_eight_describes_duplicates_identically() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device batch 8: the released checkpoint is absent");
        return;
    };
    let (Some(cats), Some(text)) = (crate::decoder::golden("device batch 8", "cats"), crate::decoder::golden("device batch 8", "text")) else { return };
    crate::common::cap_threads();
    let gpu = gpu_core::Gpu::new(&Device::kernels());
    let model = MirageModel::new(Mirage::open(&dir).unwrap(), gpu).unwrap();
    let rgb: Vec<Vec<u8>> = [&cats, &text].iter().map(|g| g["input_rgb"].iter().map(|&v| v as u8).collect()).collect();
    let dims = [(480usize, 640usize), (600, 800)];
    let images: Vec<Image> = (0..8).map(|i| Image { rgb: &rgb[i % 2], height: dims[i % 2].0, width: dims[i % 2].1 }).collect();
    let opts = Options::from_json(r#"{"fields":["bbox","score","mask","position","description"],"mask_format":"bitmap","max_tokens":600}"#).unwrap();
    let started = std::time::Instant::now();
    let out: Vec<Value> = model.describe_batch(&images, &opts).into_iter().map(json).collect();
    eprintln!("describe_batch of 8: {:.1}s", started.elapsed().as_secs_f64());
    for i in 2..8 {
        assert!(out[i] == out[i % 2], "image {i} differs from its duplicate {}", i % 2);
    }
}

/// Throughput curve of `describe_batch` over batch size on an accelerator: images per second, latency of a
/// batch and the split between the stages. Needs a real device; the CPU backend skips it (it would take minutes).
#[test]
fn describe_throughput_over_batch_size() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("throughput: the released checkpoint is absent");
        return;
    };
    let (Some(cats), Some(text)) = (crate::decoder::golden("throughput", "cats"), crate::decoder::golden("throughput", "text")) else { return };
    crate::common::cap_threads();
    let gpu = gpu_core::Gpu::new(&Device::kernels());
    if gpu.kind() == "cpu" {
        brain_testutil::skip_unavailable("throughput: the CPU backend is not a throughput target");
        return;
    }
    let model = MirageModel::new(Mirage::open(&dir).unwrap(), gpu).unwrap();
    let rgb: Vec<Vec<u8>> = [&cats, &text].iter().map(|g| g["input_rgb"].iter().map(|&v| v as u8).collect()).collect();
    let dims = [(480usize, 640usize), (600, 800)];
    let opts = Options::from_json(r#"{"fields":["bbox","score","mask","position","description"],"mask_format":"bitmap","max_tokens":600}"#).unwrap();
    let image = |i: usize| Image { rgb: &rgb[i % 2], height: dims[i % 2].0, width: dims[i % 2].1 };
    let _ = model.describe_batch(&[image(0), image(1)], &opts);
    eprintln!("batch | wall s | images/s | latency s | vision | memory | decode | post");
    for n in [1usize, 8, 32, 64] {
        let images: Vec<Image> = (0..n).map(image).collect();
        // The card is shared: rank by the minimum over replays.
        let mut best: Option<(f64, mirage::device::Timings)> = None;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let out = model.describe_batch(&images, &opts);
            let wall = started.elapsed().as_secs_f64();
            assert!(out.iter().all(Result::is_ok));
            if best.is_none_or(|(w, _)| wall < w) {
                best = Some((wall, model.last_timings()));
            }
        }
        let (wall, t) = best.unwrap();
        eprintln!("{n:5} | {wall:6.2} | {:8.2} | {wall:9.2} | {:.2} | {:.2} | {:.2} | {:.2} (device {:.2})", n as f64 / wall, t.vision, t.memory, t.decode, t.post, t.reader);
    }
}

/// The structured mixers (original parameters) and the folded ones (dense matrices) describe the same images: equal
/// up to what the dense products' rounding may move. Also prints the encoder time and the resident bytes of both.
#[test]
fn structured_and_folded_mixers_describe_alike() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("mixer forms: the released checkpoint is absent");
        return;
    };
    let (Some(cats), Some(text)) = (crate::decoder::golden("mixer forms", "cats"), crate::decoder::golden("mixer forms", "text")) else { return };
    crate::common::cap_threads();
    let probe = gpu_core::Gpu::new(&Device::kernels());
    let exact = probe.kind() == "cpu";
    drop(probe);
    let rgb: Vec<Vec<u8>> = [&cats, &text].iter().map(|g| g["input_rgb"].iter().map(|&v| v as u8).collect()).collect();
    let dims = [(480usize, 640usize), (600, 800)];
    let opts = Options::from_json(r#"{"fields":["bbox","score","position","description"],"max_tokens":600}"#).unwrap();
    let mut answers = Vec::new();
    for form in [WeightForm::Structured, WeightForm::Folded] {
        let model = MirageModel::new_with(Mirage::open(&dir).unwrap(), gpu_core::Gpu::new(&Device::kernels()), form).unwrap();
        eprintln!("mixer forms {form:?}: resident {:?} bytes, matrices {:?}", model.resident_bytes(), model.device().weight_bytes());
        let images: Vec<Image> = (0..2).map(|i| Image { rgb: &rgb[i], height: dims[i].0, width: dims[i].1 }).collect();
        let mut best = f64::MAX;
        let mut out = Vec::new();
        for _ in 0..3 {
            out = model.describe_batch(&images, &opts).into_iter().map(json).collect::<Vec<_>>();
            best = best.min(model.last_timings().vision);
        }
        eprintln!("mixer forms {form:?}: encoder and dense head of 2 images {best:.4} s");
        answers.push(out);
    }
    for (i, (a, b)) in answers[0].iter().zip(&answers[1]).enumerate() {
        assert_same(exact, a, b, &format!("image {i}: structured against folded mixers"));
    }
}

/// The FP32 model's describe at batch 1 and batch 64 (cats), alone, for a profiler (`nsys profile` on this test
/// binary; `--cuda-graph-trace=node` resolves the replayed decoder steps).
#[test]
#[ignore = "profiling entry point: run under nsys/ncu"]
fn fp32_describe_for_profiling() {
    let Some(dir) = released_dir() else { return };
    let Some(cats) = crate::decoder::golden("fp32 profile", "cats") else { return };
    let rgb: Vec<u8> = cats["input_rgb"].iter().map(|&v| v as u8).collect();
    let model = MirageModel::new(Mirage::open(&dir).unwrap(), gpu_core::Gpu::new(&Device::kernels())).unwrap();
    let opts = Options::from_json(r#"{"fields":["bbox","score","position","description"],"max_tokens":600}"#).unwrap();
    for n in [1usize, 1, 1, 64, 64] {
        let images: Vec<Image> = (0..n).map(|_| Image { rgb: &rgb, height: 480, width: 640 }).collect();
        assert!(model.describe_batch(&images, &opts).iter().all(Result::is_ok));
    }
}
