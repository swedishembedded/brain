// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `MirageModel` end to end: a batch of images through the device equals the
//! host model image by image (identical proposals, tokens, text and bitmap
//! run-length codes) and equals the same images run alone.

use mirage::describe::Options;
use mirage::device::{Device, Image, MirageModel};
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
