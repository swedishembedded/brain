// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held-out metrics for the fine-tuning experiments: detection AP50, character error rate and mask IoU.

/// Intersection over union of two `[x0, y0, x1, y1]` boxes.
pub fn box_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Average precision at IoU 0.5 (all-point interpolation) over a set of images: `detections[i]` are the scored
/// boxes of image `i`, `truths[i]` its labelled boxes. Detections are ranked by score across images; each truth
/// matches at most one detection.
pub fn average_precision_50(detections: &[Vec<(f32, [f64; 4])>], truths: &[Vec<[f64; 4]>]) -> f64 {
    assert_eq!(detections.len(), truths.len(), "one detection list per image");
    let total: usize = truths.iter().map(Vec::len).sum();
    if total == 0 {
        return 0.0;
    }
    let mut ranked: Vec<(f32, usize, [f64; 4])> = detections.iter().enumerate().flat_map(|(i, d)| d.iter().map(move |(s, b)| (*s, i, *b))).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut used: Vec<Vec<bool>> = truths.iter().map(|t| vec![false; t.len()]).collect();
    let (mut tp, mut fp) = (0usize, 0usize);
    let mut curve: Vec<(f64, f64)> = Vec::new(); // (recall, precision)
    for (_, image, b) in ranked {
        let best = truths[image].iter().enumerate().filter(|(j, _)| !used[image][*j]).map(|(j, t)| (j, box_iou(b, *t))).max_by(|x, y| x.1.total_cmp(&y.1));
        match best {
            Some((j, iou)) if iou >= 0.5 => {
                used[image][j] = true;
                tp += 1;
            }
            _ => fp += 1,
        }
        curve.push((tp as f64 / total as f64, tp as f64 / (tp + fp) as f64));
    }
    // Precision envelope, then the area under it.
    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for k in 0..curve.len() {
        let envelope = curve[k..].iter().map(|c| c.1).fold(0.0, f64::max);
        ap += (curve[k].0 - prev_recall) * envelope;
        prev_recall = curve[k].0;
    }
    ap
}

/// Levenshtein distance between two strings, in characters.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
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

/// Character error rate of `(reference, hypothesis)` pairs: total edits over total reference characters.
pub fn character_error_rate(pairs: &[(String, String)]) -> f64 {
    let chars: usize = pairs.iter().map(|(r, _)| r.chars().count()).sum();
    if chars == 0 {
        return 0.0;
    }
    pairs.iter().map(|(r, h)| edit_distance(r, h)).sum::<usize>() as f64 / chars as f64
}

/// Intersection over union of a predicted probability field (thresholded at 0.5) and a truth mask.
pub fn mask_iou(field: &[f32], truth: &[bool]) -> f64 {
    assert_eq!(field.len(), truth.len(), "field and truth cover the same grid");
    let (mut inter, mut union) = (0usize, 0usize);
    for (p, t) in field.iter().zip(truth) {
        let pred = *p > 0.5;
        inter += usize::from(pred && *t);
        union += usize::from(pred || *t);
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}
