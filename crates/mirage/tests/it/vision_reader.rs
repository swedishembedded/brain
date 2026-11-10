// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end text reading: crop a line out of an image by its box, cut the
//! strip into windows, run the reader and join the windows. The reference
//! engine's own transcriptions of the same image are the yardstick; it runs
//! the 2-bit deployment weights, so spacing and a letter or two may differ
//! from this full-precision path.

use mirage::cact::CactFile;
use mirage::cactmap::CactModel;
use mirage::vision::reader::{crop_rect, decode_windows, strip_from_crop, windows, FRAMES, WINDOW_H, WINDOW_STEP, WINDOW_W};
use serde_json::Value;

use crate::vision::fixture;

/// Levenshtein distance.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != cb)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

#[test]
fn window_geometry_follows_the_strip_width() {
    let strip = |w: usize| mirage::vision::preprocess::Resized { rgb: vec![9; WINDOW_H * w * 3], h: WINDOW_H, w };
    assert_eq!(windows(&strip(10)).len(), 1);
    assert_eq!(windows(&strip(WINDOW_W)).len(), 1);
    assert_eq!(windows(&strip(WINDOW_W + 1)).len(), 2);
    assert_eq!(windows(&strip(WINDOW_W + WINDOW_STEP)).len(), 2);
    assert_eq!(windows(&strip(WINDOW_W + WINDOW_STEP + 1)).len(), 3);
    let w = &windows(&strip(WINDOW_W + 10))[1];
    assert_eq!(w[0], 9, "the second window starts inside the strip");
    assert_eq!(w[(WINDOW_W + 10 - WINDOW_STEP) * 3], 127, "and is padded with 127 beyond its end");
    let crop = vec![0u8; 20 * 100 * 3];
    assert_eq!(strip_from_crop(&crop, 20, 100).unwrap().w, 240, "width follows the aspect ratio at height 48");
    assert_eq!(crop_rect([49.4, 91.2, 175.1, 124.0], 800, 600), (39, 54, 141, 75));
}

#[test]
fn the_collapse_joins_windows_and_repeats_only_across_blanks() {
    let classes = mirage::config::READER_CLASSES;
    let charset: Vec<char> = "abc".chars().collect();
    // One frame per entry: label l makes logit l the argmax.
    let window = |labels: &[usize]| -> Vec<f32> {
        let mut v = vec![0f32; FRAMES * classes];
        for f in 0..FRAMES {
            v[f * classes + labels.get(f).copied().unwrap_or(0)] = 1.0;
        }
        v
    };
    // a a blank a b b -> "a" "a" "b": a run collapses, a blank separates repeats.
    assert_eq!(decode_windows(&[window(&[1, 1, 0, 1, 2, 2])], 24, &charset), "aab");
}

#[test]
fn text_lines_read_as_the_engine_reads_them() {
    let Some(f) = fixture("reader text") else { return };
    let dir = std::path::PathBuf::from(brain_testutil::testdata("golden/mirage"));
    let (png, expected) = (dir.join("fixtures/text.png"), dir.join("engine/text_default/out.json"));
    if !png.is_file() || !expected.is_file() {
        brain_testutil::skip("reader text: testdata/golden/mirage/{fixtures/text.png,engine/text_default/out.json} is absent (make testdata/fetch)");
        return;
    }
    let img = imaging::codec::decode_bounded(&std::fs::read(png).unwrap(), 1 << 26).unwrap();
    let (w, h) = (img.w as usize, img.h as usize);
    let cact = CactFile::open(&crate::common::released_dir().unwrap().join("mirage.cact")).unwrap();
    let charset = CactModel::new(cact, f.model.model_config().clone()).unwrap().charset().unwrap();
    let out: Value = serde_json::from_slice(&std::fs::read(expected).unwrap()).unwrap();
    let lines = out["text"].as_array().unwrap();
    assert_eq!(lines.len(), 3);
    for line in lines {
        let bbox: Vec<f32> = line["bbox"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let rect = crop_rect([bbox[0], bbox[1], bbox[2], bbox[3]], w, h);
        let got = f.model.reader.read_line(&img.px, h, w, rect, &charset).unwrap();
        let want = line["string"].as_str().unwrap();
        eprintln!("reader: {got:?} (engine: {want:?})");
        let (a, b): (Vec<char>, Vec<char>) = (got.chars().filter(|c| *c != ' ').collect(), want.chars().filter(|c| *c != ' ').collect());
        let distance = edit_distance(&a, &b);
        assert!(distance <= 2, "{got:?} is {distance} edits from the engine's {want:?} beyond spacing");
    }
}
