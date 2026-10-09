// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements object-detection evaluation for its clients.
// If your team needs expertise in detector benchmarking then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Dataset-level detection report built on the per-IoU metrics in
//! [`crate::detection`]: mAP@0.5, COCO-style mAP@0.5:0.95 (ten IoU thresholds
//! 0.50, 0.55, ..., 0.95), per-class AP, and a JSONL dump of every image's
//! ground truth and predictions so a clustered bootstrap can be run offline.
//!
//! Each image lives in its own pixel frame. [`score`] ranks predictions across
//! the whole set (as AP requires) while guaranteeing a prediction can only
//! match ground truth of its own image: every image is translated into a
//! private horizontal strip before the model-free metrics see it.
//!
//! ## JSONL format
//! One object per image:
//! `{"image": <index>, "gts": [{"class": c, "xyxy": [x1,y1,x2,y2]}],
//!   "preds": [{"class": c, "score": s, "xyxy": [x1,y1,x2,y2]}]}`
//! with boxes in that image's own pixel coordinates.

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};
use yolov8::Detection;

use crate::detection::{ap_for_class, map_at, precision_recall, GtBox};

/// Number of COCO IoU thresholds (0.50 to 0.95 in steps of 0.05).
const COCO_IOU_STEPS: usize = 10;

/// Gap between the private strips two images are translated into.
const STRIP_GAP: f32 = 16.0;

/// The ground truth and predictions of one image, in its own pixel frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRecord {
    /// Index of the image in its source dataset.
    pub index: usize,
    pub gts: Vec<GtBox>,
    pub preds: Vec<Detection>,
}

/// AP of one class that has ground truth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassAp {
    pub class: u32,
    pub ap50: f32,
    pub ap50_95: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DetectionReport {
    pub map50: f32,
    pub map50_95: f32,
    pub precision50: f32,
    pub recall50: f32,
    /// One entry per class with at least one ground-truth box, by class id.
    pub per_class: Vec<ClassAp>,
    pub n_images: usize,
    pub n_preds: usize,
    pub n_gts: usize,
}

/// The ten COCO IoU thresholds.
fn coco_thresholds() -> impl Iterator<Item = f32> {
    (0..COCO_IOU_STEPS).map(|k| 0.5 + 0.05 * k as f32)
}

/// Translate every image into its own horizontal strip so the flattened
/// prediction and ground-truth lists cannot match across images.
fn flatten_in_strips(images: &[ImageRecord]) -> (Vec<Detection>, Vec<GtBox>) {
    let extent = images
        .iter()
        .flat_map(|im| im.gts.iter().map(|g| g.bbox[2]).chain(im.preds.iter().map(|p| p[2])))
        .fold(0.0f32, f32::max);
    let stride = extent + STRIP_GAP;
    let mut preds = Vec::new();
    let mut gts = Vec::new();
    for (k, im) in images.iter().enumerate() {
        let off = k as f32 * stride;
        preds.extend(im.preds.iter().map(|p| [p[0] + off, p[1], p[2] + off, p[3], p[4], p[5]]));
        gts.extend(im.gts.iter().map(|g| GtBox {
            class: g.class,
            bbox: [g.bbox[0] + off, g.bbox[1], g.bbox[2] + off, g.bbox[3]],
        }));
    }
    (preds, gts)
}

/// Score a set of images over class ids `0..nc`.
pub fn score(images: &[ImageRecord], nc: u32) -> DetectionReport {
    let (preds, gts) = flatten_in_strips(images);
    let map50 = map_at(&preds, &gts, nc, 0.5);
    let map50_95 = coco_thresholds().map(|t| map_at(&preds, &gts, nc, t)).sum::<f32>() / COCO_IOU_STEPS as f32;
    let (precision50, recall50) = precision_recall(&preds, &gts, 0.5);
    let per_class = (0..nc)
        .filter_map(|class| {
            let ap50 = ap_for_class(&preds, &gts, class, 0.5)?;
            let ap_sum: f32 = coco_thresholds().filter_map(|t| ap_for_class(&preds, &gts, class, t)).sum();
            Some(ClassAp { class, ap50, ap50_95: ap_sum / COCO_IOU_STEPS as f32 })
        })
        .collect();
    DetectionReport {
        map50,
        map50_95,
        precision50,
        recall50,
        per_class,
        n_images: images.len(),
        n_preds: preds.len(),
        n_gts: gts.len(),
    }
}

/// Write one JSON line per image (see the module docs for the schema).
pub fn write_jsonl<W: Write>(out: &mut W, images: &[ImageRecord]) -> io::Result<()> {
    for im in images {
        let gts: Vec<Value> = im.gts.iter().map(|g| json!({ "class": g.class, "xyxy": g.bbox })).collect();
        let preds: Vec<Value> = im
            .preds
            .iter()
            .map(|p| json!({ "class": p[5] as u32, "score": p[4], "xyxy": [p[0], p[1], p[2], p[3]] }))
            .collect();
        let line = json!({ "image": im.index, "gts": gts, "preds": preds });
        writeln!(out, "{line}")?;
    }
    Ok(())
}

fn invalid(line_no: usize, what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("line {line_no}: {what}"))
}

fn xyxy_of(v: &Value, line_no: usize) -> io::Result<[f32; 4]> {
    let arr = v["xyxy"].as_array().filter(|a| a.len() == 4).ok_or_else(|| invalid(line_no, "xyxy must have 4 numbers"))?;
    let mut b = [0.0f32; 4];
    for (slot, x) in b.iter_mut().zip(arr) {
        *slot = x.as_f64().ok_or_else(|| invalid(line_no, "xyxy must be numeric"))? as f32;
    }
    Ok(b)
}

/// Read a dump produced by [`write_jsonl`]. Blank lines are skipped; any other
/// malformed line is an error.
pub fn read_jsonl<R: BufRead>(input: R) -> io::Result<Vec<ImageRecord>> {
    let mut images = Vec::new();
    for (n, line) in input.lines().enumerate() {
        let line_no = n + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).map_err(|e| invalid(line_no, &e.to_string()))?;
        let index = v["image"].as_u64().ok_or_else(|| invalid(line_no, "missing image index"))? as usize;
        let mut gts = Vec::new();
        for g in v["gts"].as_array().ok_or_else(|| invalid(line_no, "missing gts"))? {
            let class = g["class"].as_u64().ok_or_else(|| invalid(line_no, "gt class"))? as u32;
            gts.push(GtBox { class, bbox: xyxy_of(g, line_no)? });
        }
        let mut preds = Vec::new();
        for p in v["preds"].as_array().ok_or_else(|| invalid(line_no, "missing preds"))? {
            let class = p["class"].as_u64().ok_or_else(|| invalid(line_no, "pred class"))?;
            let score = p["score"].as_f64().ok_or_else(|| invalid(line_no, "pred score"))? as f32;
            let b = xyxy_of(p, line_no)?;
            preds.push([b[0], b[1], b[2], b[3], score, class as f32]);
        }
        images.push(ImageRecord { index, gts, preds });
    }
    Ok(images)
}
