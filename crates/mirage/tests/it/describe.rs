// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Result assembly against the reference engine's recorded cases: the
//! decoder text, proposals and mask inputs it saw, and the JSON it produced.
//! Everything must match exactly except polygon vertex lists, which are held
//! to rasterised overlap with the engine's polygons.

use std::path::{Path, PathBuf};

use mirage::describe::{assemble, Describe, Detection, Inputs, Options};
use mirage::vision::masks::{coefficient_cell, object_field, round_box, Frame};
use mirage::vision::preprocess::letterbox_size;
use serde_json::{json, Value};

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn case_dir(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(brain_testutil::testdata(&format!("golden/mirage/engine/{name}")));
    if !dir.join("out.json").is_file() {
        brain_testutil::skip(&format!("{name}: testdata/golden/mirage/engine/{name} is absent (make testdata/fetch)"));
        return None;
    }
    Some(dir)
}

fn detections(path: &Path) -> Vec<Detection> {
    floats(path).as_chunks::<5>().0.iter().map(|p| Detection { bbox: [p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0, p[3] / 1000.0], score: p[4] }).collect()
}

/// Scanline even-odd fill of rings (flat `[x, y, ...]`) on a 1000x1000 grid.
fn raster(rings: &[Vec<i32>]) -> Vec<bool> {
    let mut out = vec![false; 1000 * 1000];
    for y in 0..1000 {
        let yc = y as f32 + 0.5;
        let mut xs = Vec::new();
        for ring in rings {
            let pts: Vec<(f32, f32)> = ring.as_chunks::<2>().0.iter().map(|p| (p[0] as f32, p[1] as f32)).collect();
            for i in 0..pts.len() {
                let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                if (a.1 <= yc) != (b.1 <= yc) {
                    xs.push(a.0 + (yc - a.1) / (b.1 - a.1) * (b.0 - a.0));
                }
            }
        }
        xs.sort_by(f32::total_cmp);
        for pair in xs.as_chunks::<2>().0 {
            for x in (pair[0] + 0.5).max(0.0) as usize..((pair[1] + 0.5).max(0.0) as usize).min(1000) {
                out[y * 1000 + x] = true;
            }
        }
    }
    out
}

fn rings(mask: &Value) -> Vec<Vec<i32>> {
    let nums = |v: &Value| v.as_array().unwrap().iter().map(|n| n.as_i64().unwrap() as i32).collect::<Vec<i32>>();
    match mask.as_array() {
        Some(a) if a.first().is_some_and(Value::is_array) => a.iter().map(nums).collect(),
        Some(_) => vec![nums(mask)],
        None => vec![],
    }
}

fn iou(a: &[bool], b: &[bool]) -> f64 {
    let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    let union = a.iter().zip(b).filter(|(x, y)| **x || **y).count();
    inter as f64 / union.max(1) as f64
}

/// Run one recorded case; returns the polygon IoU of every object mask when masks are polygons.
fn run_case(name: &str) -> Option<Vec<f64>> {
    let dir = case_dir(name)?;
    let request = std::fs::read_to_string(dir.join("request.txt")).unwrap();
    let mut lines = request.lines();
    let opts = Options::from_json(lines.next().unwrap()).unwrap();
    let _language = lines.next().unwrap();
    let dims: Vec<usize> = lines.next().unwrap().split_whitespace().skip(1).map(|v| v.parse().unwrap()).collect();
    let (width, height) = (dims[0], dims[1]);
    let (rh, rw) = letterbox_size(height, width);
    let frame = Frame { resized_w: rw, resized_h: rh, width, height };
    let expected: Value = serde_json::from_slice(&std::fs::read(dir.join("out.json")).unwrap()).unwrap();

    let objects = detections(&dir.join("peaks_kind0.f32"));
    let text = detections(&dir.join("peaks_kind1.f32"));
    let protos = std::fs::read(dir.join("protos.f32")).map(|_| floats(&dir.join("protos.f32"))).unwrap_or_default();
    // The reader is not under test here: take the engine's strings, matched by box.
    let engine_text: Vec<(Vec<i64>, String)> = expected["text"]
        .as_array()
        .map(|a| a.iter().map(|t| (t["bbox"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect(), t["string"].as_str().unwrap().to_string())).collect())
        .unwrap_or_default();
    let read = |j: usize| -> Option<String> {
        let b = mirage::describe::public_box(text[j].bbox, frame);
        engine_text.iter().find(|(bb, _)| bb.iter().zip(b).all(|(x, y)| *x == i64::from(y))).map(|(_, s)| s.clone())
    };
    let field = |b: [f32; 4]| {
        let (cy, cx) = coefficient_cell(round_box(b));
        let coef = floats(&dir.join(format!("coef_{cy}_{cx}.f32")));
        object_field(&protos, &coef, b)
    };
    let inputs = Inputs { objects: &objects, text: &text, frame, read: &read, field: &field };
    let lines_text = std::fs::read_to_string(dir.join("lines_0.txt")).unwrap();
    let got: Describe = assemble(lines_text.trim_start_matches('\n'), &inputs, &opts);
    let mut got = serde_json::to_value(&got).unwrap();
    let mut want = expected.clone();

    let mut overlaps = Vec::new();
    let polygons = matches!(opts.mask_format.unwrap_or_default(), mirage::describe::MaskFormat::Largest | mirage::describe::MaskFormat::Merged | mirage::describe::MaskFormat::Polygons);
    if polygons {
        if let (Some(g), Some(w)) = (got["objects"].as_array_mut(), want["objects"].as_array_mut()) {
            assert_eq!(g.len(), w.len(), "{name}: object count");
            for (go, wo) in g.iter_mut().zip(w.iter_mut()) {
                if wo.get("mask").is_some() {
                    overlaps.push(iou(&raster(&rings(&go["mask"])), &raster(&rings(&wo["mask"]))));
                    go["mask"] = Value::Null;
                    wo["mask"] = Value::Null;
                }
            }
        }
    }
    assert_eq!(got, want, "{name}: assembled JSON differs from the engine's");
    Some(overlaps)
}

#[test]
fn recorded_cases_assemble_exactly() {
    for name in ["cats_bitmap", "cats_de", "cats_fields", "cats_maxobj", "text_default", "text_lines"] {
        if let Some(o) = run_case(name) {
            eprintln!("describe {name}: exact{}", if o.is_empty() { String::new() } else { format!(", polygon IoU {o:.4?}") });
        }
    }
}

#[test]
fn polygon_masks_overlap_the_engines() {
    let mut worst = 1.0f64;
    for name in ["cats_default", "cats_polygons", "cats_merged", "text_default"] {
        let Some(o) = run_case(name) else { return };
        eprintln!("polygon IoU {name}: {o:.4?}");
        worst = worst.min(o.iter().copied().fold(1.0, f64::min));
    }
    assert!(worst >= 0.97, "worst polygon IoU {worst}");
}

#[test]
fn bad_options_are_refused_with_the_valid_names() {
    let err = Options::from_json(r#"{"colour": true}"#).unwrap_err().to_string();
    assert!(err.contains("colour") && err.contains("mask_format"), "{err}");
    assert!(Options::from_json(r#"{"conf": 2}"#).unwrap_err().to_string().contains("conf"));
    assert!(Options::from_json(r#"{"mask_format": "svg"}"#).unwrap_err().to_string().contains("mask_format"));
    assert!(Options::from_json(r#"{"language": "xx"}"#).unwrap_err().to_string().contains("language"));
    assert!(Options::from_json(r#"{"max_tokens": 0}"#).is_err());
    let o = Options::from_json(&json!({"sections": ["text", "scene"]}).to_string()).unwrap();
    assert_eq!(o.max_rank(), 3);
}
