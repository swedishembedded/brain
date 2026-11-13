// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scoring a detector against a detection dataset: mAP@0.5, mAP@0.5:0.95,
//! precision, recall and per-class AP at one confidence/IoU operating point,
//! optionally with every image's ground truth and predictions dumped as JSONL
//! for an offline bootstrap.
//!
//! Swedish Embedded AB implements detector benchmarking on its clients' own
//! labelled imagery. If your team needs expertise in evaluating object
//! detectors honestly then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use data::gen_detect::{load_dataset, DetectData};

use crate::detection::GtBox;
use crate::detection_report::{self, DetectionReport, ImageRecord};
use crate::model::Yolo;
use crate::train::val_start;

/// Which part of the dataset is scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// The held-out tail (last 10%) that `train` never sees.
    Val,
    /// Every image, for a checkpoint that was not trained on this dataset.
    All,
}

impl Split {
    pub const NAMES: [&'static str; 2] = ["val", "all"];

    pub fn parse(name: &str) -> Result<Split, String> {
        match name {
            "val" => Ok(Split::Val),
            "all" => Ok(Split::All),
            other => Err(format!("split must be `all` or `val`, got {other:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Split::Val => "val",
            Split::All => "all",
        }
    }

    fn first_index(self, n: usize) -> usize {
        match self {
            Split::Val => val_start(n),
            Split::All => 0,
        }
    }
}

/// Ground truth of dataset image `i` as pixel `xyxy` boxes.
fn gts_of(data: &DetectData, i: usize) -> Vec<GtBox> {
    data.boxes[i]
        .iter()
        .map(|b| {
            let (cx, cy) = (b.cx * data.w as f32, b.cy * data.h as f32);
            let (hw, hh) = (b.w * data.w as f32 * 0.5, b.h * data.h as f32 * 0.5);
            GtBox { class: b.class, bbox: [cx - hw, cy - hh, cx + hw, cy + hh] }
        })
        .collect()
}

/// Run `model` (batch 1) over `split` of `data` and collect each image's
/// ground truth and predictions.
pub fn collect(model: &Yolo, data: &DetectData, split: Split, conf: f32, iou: f32) -> Vec<ImageRecord> {
    let stride = data.image_stride();
    (split.first_index(data.n)..data.n)
        .map(|i| {
            // CHW -> HWC: `detect` takes interleaved RGB.
            let chw = &data.images[i * stride..(i + 1) * stride];
            let hwc = imaging::pixels::chw_to_hwc(chw, 3, data.h as usize, data.w as usize);
            ImageRecord { index: i, gts: gts_of(data, i), preds: model.detect(&hwc, data.w, data.h, conf, iou) }
        })
        .collect()
}

/// Score the checkpoint `weights` on `data_dir`, writing the per-image JSONL to
/// `dump_preds` when given.
pub fn run(weights: &str, data_dir: &Path, split: Split, conf: f32, iou: f32, dump_preds: Option<&Path>) -> Result<DetectionReport, String> {
    let data = load_dataset(data_dir).map_err(|e| format!("loading {}: {e}", data_dir.display()))?;
    let model = Yolo::try_load_at(weights, 1, 0)?;
    let records = collect(&model, &data, split, conf, iou);
    if let Some(path) = dump_preds {
        let file = std::fs::File::create(path).map_err(|e| format!("writing {}: {e}", path.display()))?;
        let mut out = std::io::BufWriter::new(file);
        detection_report::write_jsonl(&mut out, &records)
            .and_then(|()| std::io::Write::flush(&mut out))
            .map_err(|e| format!("writing {}: {e}", path.display()))?;
    }
    Ok(detection_report::score(&records, model.cfg.nc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{self, TrainOptions};
    use data::gen_detect::{write_dataset, Preset};

    fn trained(tag: &str) -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("yolov8-eval-{tag}-{}", std::process::id()));
        write_dataset(&dir, Preset::MultiObject, 10, 64, 64, 2, 5).unwrap();
        let weights = dir.join("w.safetensors").to_string_lossy().into_owned();
        let opts = TrainOptions { steps: 1, batch: 2, seed: 1, ..TrainOptions::default() };
        train::run(&dir, None, &weights, &opts, &mut |_, _, _| true).unwrap();
        (dir, weights)
    }

    #[test]
    fn ground_truth_is_in_pixel_corners() {
        let data = DetectData {
            images: Vec::new(),
            boxes: vec![vec![data::binio::DetectBox { class: 2, cx: 0.5, cy: 0.25, w: 0.5, h: 0.5 }]],
            n: 1,
            h: 100,
            w: 200,
            nc: 3,
        };
        assert_eq!(gts_of(&data, 0), vec![GtBox { class: 2, bbox: [50.0, 0.0, 150.0, 50.0] }]);
    }

    #[test]
    fn the_val_split_is_the_tail_and_all_is_everything() {
        let (dir, weights) = trained("split");
        let at = |split| run(&weights, &dir, split, 0.001, 0.45, None).unwrap().n_images;
        assert_eq!((at(Split::Val), at(Split::All)), (1, 10));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_report_comes_with_a_per_image_dump_when_asked() {
        let (dir, weights) = trained("dump");
        let dump = dir.join("preds.jsonl");
        let report = run(&weights, &dir, Split::All, 0.001, 0.45, Some(&dump)).unwrap();
        assert_eq!(report.n_images, 10);
        assert!((0.0..=1.0).contains(&report.map50));
        let lines = std::fs::read_to_string(&dump).unwrap().lines().count();
        assert_eq!(lines, 10, "one JSON line per image");
        let records = detection_report::read_jsonl(std::io::BufReader::new(std::fs::File::open(&dump).unwrap())).unwrap();
        assert_eq!(records.iter().map(|r| r.index).collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
        assert!(records.iter().all(|r| !r.gts.is_empty()), "the generator labels every image");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_checkpoint_or_dataset_is_an_error() {
        let (dir, weights) = trained("missing");
        assert!(run("/nonexistent/w.safetensors", &dir, Split::All, 0.25, 0.45, None).is_err());
        assert!(run(&weights, &dir.join("nope"), Split::All, 0.25, 0.45, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
