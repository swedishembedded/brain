// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mask rendering against a corpus recorded from the reference engine: its
//! prototype map, the coefficient vector of every object, its proposal boxes
//! and the run-length bitmaps it produced. Rendering must reproduce every
//! run exactly.

use std::path::{Path, PathBuf};

use mirage::vision::masks::{coefficient_cell, object_field, render_bitmap, round_box, Frame};
use serde_json::Value;

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn corpus(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(brain_testutil::testdata(&format!("golden/mirage/engine/{name}")));
    if !dir.join("out.json").is_file() {
        brain_testutil::skip(&format!("{name}: testdata/golden/mirage/engine/{name} is absent (make testdata/fetch)"));
        return None;
    }
    Some(dir)
}

#[test]
fn bitmaps_reproduce_the_engine_runs_exactly() {
    let Some(dir) = corpus("cats_bitmap") else { return };
    let protos = floats(&dir.join("protos.f32"));
    let peaks = floats(&dir.join("peaks_kind0.f32"));
    let out: Value = serde_json::from_slice(&std::fs::read(dir.join("out.json")).unwrap()).unwrap();
    // The 640x480 source letterboxed to 512x384.
    let frame = Frame { resized_w: 512, resized_h: 384, width: 640, height: 480 };
    let objects = out["objects"].as_array().unwrap();
    assert!(!objects.is_empty());
    let mut rendered = 0;
    for obj in objects {
        let public: Vec<i64> = obj["bbox"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        // The proposal whose public box (thousandths of the resized image) is this object's.
        let (k, thousandths) = peaks
            .as_chunks::<5>()
            .0
            .iter()
            .enumerate()
            .find(|(_, p)| {
                let scale = [512.0 / frame.resized_w as f32, 512.0 / frame.resized_h as f32];
                [p[0] * scale[0], p[1] * scale[1], p[2] * scale[0], p[3] * scale[1]]
                    .iter()
                    .zip(&public)
                    .all(|(a, b)| a.round() as i64 == *b)
            })
            .unwrap_or_else(|| panic!("no proposal matches bbox {public:?}"));
        let b = round_box([thousandths[0] / 1000.0, thousandths[1] / 1000.0, thousandths[2] / 1000.0, thousandths[3] / 1000.0]);
        let (cy, cx) = coefficient_cell(b);
        let coef = floats(&dir.join(format!("coef_{cy}_{cx}.f32")));
        assert_eq!(coef.len(), 32);
        let rle = render_bitmap(&object_field(&protos, &coef, b), b, frame, 0.5);

        let want = &obj["mask"];
        let size: Vec<usize> = want["size"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let counts: Vec<u32> = want["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(rle.size.to_vec(), size, "proposal {k}");
        assert_eq!(rle.counts.iter().map(|&c| c as usize).sum::<usize>(), 640 * 480);
        assert_eq!(rle.counts, counts, "proposal {k} ({}): run-length mask differs from the engine's", obj["name"]);
        rendered += 1;
    }
    eprintln!("bitmaps: {rendered} objects rendered exactly");
}
