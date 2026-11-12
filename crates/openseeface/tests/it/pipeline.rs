// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real-photo pipeline against the reference tracker: preprocessing, the
//! detector's box, the landmark crop, the decoded 66 points, and the tracker
//! keeping the face from its own landmarks on the next frame.

use openseeface::config::{DETECTOR, LANDMARKS_BEST, LANDMARKS_FAST};
use openseeface::decode;
use openseeface::model::{Net, PIPELINES};
use openseeface::{Tracker, TrackerConfig};

fn frame(g: &crate::Golden) -> (Vec<u8>, u32, u32) {
    let (shape, px) = &g["frame"];
    (px.iter().map(|v| *v as u8).collect(), shape[1] as u32, shape[0] as u32)
}

#[test]
fn preprocessing_matches_the_reference_to_one_eight_bit_step() {
    let Some(g) = crate::golden("pipeline.safetensors") else { return };
    let (rgb, w, h) = frame(&g);
    // the detector input: the reference resizes uint8 and rounds, brain
    // resizes in float; they may differ by one 8-bit step, scaled by 1/std
    let step = 1.0 / (0.225 * 255.0);
    crate::gate("detector input", &decode::detector_input(&rgb, w, h), &g["det.input"].1, 0.9999, 1.01 * step);
    // the landmark crop is resized in float by both: exact up to f32
    let c = &g["lm.crop"].1;
    let crop = [c[0] as i32, c[1] as i32, c[2] as i32, c[3] as i32];
    crate::gate("landmark input", &decode::crop_input(&rgb, w, crop), &g["lm.input"].1, 0.999999, 1e-4);
}

#[test]
fn the_detector_finds_the_reference_box_and_crop() {
    let (Some(dir), Some(g)) = (crate::weights(), crate::golden("pipeline.safetensors")) else { return };
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let det = Net::load(&gpu, &dir, DETECTOR).unwrap();
    det.forward(&g["det.input"].1);
    let b = decode::strongest_face(&det.read_output(), &det.read_pooled().unwrap(), 640, g["frame"].0[0] as u32, decode::DETECTION_THRESHOLD).expect("a face");
    let want = &g["det.box"].1;
    println!("  box {b:?} vs reference {want:?}");
    for (got, want) in [b.x, b.y, b.w, b.h].iter().zip(want) {
        assert!((got - want).abs() < 0.05, "box {b:?} vs {want:?}");
    }
    let crop = decode::crop_for(&b, 640, g["frame"].0[0] as u32).unwrap();
    let c = &g["lm.crop"].1;
    assert_eq!(crop, [c[0] as i32, c[1] as i32, c[2] as i32, c[3] as i32], "the crop is integer geometry: it must match exactly");
}

#[test]
fn decoded_landmarks_match_the_reference_tracker() {
    let (Some(dir), Some(g)) = (crate::weights(), crate::golden("pipeline.safetensors")) else { return };
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let c = &g["lm.crop"].1;
    let crop = [c[0] as i32, c[1] as i32, c[2] as i32, c[3] as i32];
    for (key, cfg) in [("lm0", LANDMARKS_FAST), ("lm3", LANDMARKS_BEST)] {
        let net = Net::load(&gpu, &dir, cfg).unwrap();
        net.forward(&g["lm.input"].1);
        let lm = decode::landmarks(&net.read_output(), crop);
        let want = &g[&format!("{key}.landmarks")].1; // (y, x, confidence) per point
        let mut worst: f32 = 0.0;
        for i in 0..decode::POINTS {
            let (wy, wx, wc) = (want[i * 3], want[i * 3 + 1], want[i * 3 + 2]);
            worst = worst.max((lm.points[i][0] - wx).abs()).max((lm.points[i][1] - wy).abs());
            assert!((lm.confidence[i] - wc).abs() < 1e-3, "{key} point {i}: confidence {} vs {wc}", lm.confidence[i]);
        }
        println!("  {key}: worst landmark error {worst:.4} px, score {:.3}", lm.score());
        assert!(worst < 0.05, "{key}: landmarks are {worst} px from the reference");
    }
}

#[test]
fn the_tracker_detects_once_then_follows_the_face_from_its_landmarks() {
    let (Some(dir), Some(g)) = (crate::weights(), crate::golden("pipeline.safetensors")) else { return };
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let (rgb, w, h) = frame(&g);
    let mut t = Tracker::load(&gpu, &dir, LANDMARKS_FAST, TrackerConfig::default()).unwrap();
    let first = t.track(&rgb, w, h).unwrap().expect("the portrait has a face");
    assert!(first.detected);
    let second = t.track(&rgb, w, h).unwrap().expect("still there");
    assert!(!second.detected, "a tracked face is followed from its landmarks, not re-detected");
    // the same face: eye corners within a few pixels between the two crops
    for i in [36, 39, 42, 45] {
        let (a, b) = (first.landmarks.points[i], second.landmarks.points[i]);
        assert!((a[0] - b[0]).abs() < 4.0 && (a[1] - b[1]).abs() < 4.0, "point {i} moved from {a:?} to {b:?}");
    }
    // a blank frame loses it, and the next real frame re-detects
    let blank = vec![0u8; rgb.len()];
    assert!(t.track(&blank, w, h).unwrap().is_none());
    assert!(t.track(&rgb, w, h).unwrap().expect("found again").detected);
}
