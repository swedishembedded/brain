// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-task training objectives for compact perception models for its clients.
// If your team needs expertise in task-masked detection, segmentation, recognition and caption losses then you
// can procure our services by sending an email to info@swedishembedded.com.

//! brain's training objective for Mirage (the vendor published none): prepared samples, the targets built from
//! a record, and the multi-task loss graph.
//!
//! * text: token cross-entropy on `[bos, lang] + lines + </s>`, reference tokens included, the prompt never a
//!   target; tokens inside a reasoning trace count `trace_share`, form answers `form_share`; the stop token is
//!   supervised only for a `closed` record and the first token of a section after an unlabelled one is not
//!   supervised, so a missing label is never learned as "nothing here";
//! * detection: CenterNet focal loss on the object and text centre heatmaps (`heat_alpha` weights positives),
//!   L1 on size and offset at the centre cell; padding cells and unlabelled families carry no weight;
//! * masks: per-object binary cross-entropy plus Dice of `sigmoid(P . a)` inside the box at the prototype
//!   resolution, weighted by [`TrainWeights::mask`] (the released configuration trained with `seg_loss` 0; brain
//!   makes the weight an explicit option);
//! * reader: CTC over the joined window frames of each text line, normalised per label character;
//! * `head_grad`: the gradient the heads send into the encoder is scaled (see `Net::dense`).

use std::collections::BTreeMap;

use super::net::Net;
use super::ops::Seg;
use super::real::{r, Real};
use super::tape::Id;
use crate::config::{MirageConfig, DENSE_OUT_CHANNELS, MASK_RANK};
use crate::vision::masks::round_box;
use crate::vision::memory::Slot;
use crate::vision::ops::valid_cells;
use crate::vision::preprocess::{self, CANVAS};
use crate::vision::reader::{self, frame_range};

use super::data::{Picture, Record, RecordError};

/// Relative weights of the loss terms (0 switches a term, and its forward work, off).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainWeights {
    /// Token cross-entropy.
    pub text: f64,
    /// Centre-heatmap focal loss.
    pub heat: f64,
    /// Size and offset L1.
    pub regression: f64,
    /// Mask BCE + Dice.
    pub mask: f64,
    /// Reader CTC.
    pub ctc: f64,
}

impl Default for TrainWeights {
    fn default() -> Self {
        TrainWeights { text: 1.0, heat: 1.0, regression: 1.0, mask: 1.0, ctc: 1.0 }
    }
}

impl TrainWeights {
    /// Whether the encoder and dense head are needed by any active term.
    fn needs_vision(&self) -> bool {
        self.text > 0.0 || self.heat > 0.0 || self.regression > 0.0 || self.mask > 0.0
    }
}

/// A ground-truth object in canvas-normalised coordinates.
#[derive(Debug, Clone)]
pub struct GtObject {
    /// Entity id in the record.
    pub id: u32,
    /// Name.
    pub name: String,
    /// `[x0, y0, x1, y1]` in `0..1` of the canvas.
    pub bbox: [f32; 4],
    /// Mask rings in canvas-normalised coordinates, when labelled.
    pub polygons: Option<Vec<Vec<f64>>>,
    /// Position phrase.
    pub position: Option<String>,
    /// Attributes.
    pub attributes: Vec<(String, String)>,
}

/// The reader's view of one text line.
#[derive(Debug, Clone)]
pub struct ReaderLine {
    /// The 48x320 RGB windows of the line's strip.
    pub windows: Vec<Vec<u8>>,
    /// Frames each window contributes (see `reader::frame_range`).
    pub ranges: Vec<(usize, usize)>,
    /// Class labels (`charset index + 1`); `None` when a character is outside the charset.
    pub labels: Option<Vec<u32>>,
}

/// A ground-truth text line.
#[derive(Debug, Clone)]
pub struct GtText {
    /// `[x0, y0, x1, y1]` in `0..1` of the canvas.
    pub bbox: [f32; 4],
    /// The string.
    pub value: String,
    /// Reader inputs (absent when the sample was built without an image to crop).
    pub line: Option<ReaderLine>,
}

/// One prepared training sample.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The `canvas x canvas x 3` network input.
    pub canvas: Vec<f32>,
    /// Resized image height on the canvas (512 units).
    pub rh: usize,
    /// Resized image width on the canvas (512 units).
    pub rw: usize,
    /// Scene caption, when labelled.
    pub caption: Option<String>,
    /// Objects in decoder reference order, when labelled.
    pub objects: Option<Vec<GtObject>>,
    /// Text lines in reference order, when labelled.
    pub text: Option<Vec<GtText>>,
    /// Relations `(subject index, predicate, object index)` into `objects`, when labelled.
    pub relations: Option<Vec<(usize, String, usize)>>,
    /// Nothing exists beyond the labelled families.
    pub closed: bool,
    /// Language code.
    pub language: String,
}

/// The decoder-order key of a canvas box (band of the centre row, then centre column), as proposals are ordered.
fn proposal_key(b: [f32; 4]) -> f64 {
    ((16.0 * f64::from(b[1] + b[3]) / 2.0).floor()) * 10000.0 + f64::from(b[0] + b[2]) / 2.0 * 1000.0
}

impl Sample {
    /// Prepare a record and its decoded image: letterbox, canvas, boxes in canvas units, reader strips.
    pub fn from_picture(rec: &Record, pic: &Picture, charset: &[char]) -> Result<Sample, RecordError> {
        let resized = preprocess::letterbox(&pic.rgb, pic.h, pic.w).map_err(|e| RecordError(e.to_string()))?;
        let (rh, rw) = (resized.h, resized.w);
        let to_canvas = |b: &[f64; 4]| -> [f32; 4] {
            let sx = rw as f64 / CANVAS as f64 / 1000.0;
            let sy = rh as f64 / CANVAS as f64 / 1000.0;
            [(b[0] * sx) as f32, (b[1] * sy) as f32, (b[2] * sx) as f32, (b[3] * sy) as f32]
        };
        let (sx, sy) = (rw as f64 / CANVAS as f64 / 1000.0, rh as f64 / CANVAS as f64 / 1000.0);
        let mut order: Vec<usize> = rec.objects.iter().flatten().enumerate().map(|(i, _)| i).collect();
        let objects = rec.objects.as_ref().map(|list| {
            order.sort_by(|&a, &b| proposal_key(to_canvas(&list[a].bbox)).total_cmp(&proposal_key(to_canvas(&list[b].bbox))));
            order
                .iter()
                .map(|&i| {
                    let o = &list[i];
                    GtObject {
                        id: o.id,
                        name: o.name.clone(),
                        bbox: to_canvas(&o.bbox),
                        polygons: o.mask_polygons.as_ref().map(|rings| rings.iter().map(|ring| ring.chunks_exact(2).flat_map(|p| [p[0] * sx, p[1] * sy]).collect()).collect()),
                        position: o.position.clone(),
                        attributes: o.attributes.iter().flatten().map(|(k, v)| (k.clone(), v.clone())).collect(),
                    }
                })
                .collect::<Vec<_>>()
        });
        let relations = match (&rec.relations, &rec.objects) {
            (Some(rels), Some(list)) => {
                let index_of = |id: u32| order.iter().position(|&i| list[i].id == id).expect("validated: relations name labelled objects");
                Some(rels.iter().map(|rel| (index_of(rel.subject), rel.predicate.clone(), index_of(rel.object))).collect())
            }
            _ => None,
        };
        let text = rec
            .text
            .as_ref()
            .map(|lines| {
                let mut gt: Vec<GtText> = lines
                    .iter()
                    .map(|t| {
                        let line = reader_line(pic, &t.bbox, &t.value, charset)?;
                        Ok(GtText { bbox: to_canvas(&t.bbox), value: t.value.clone(), line: Some(line) })
                    })
                    .collect::<Result<_, RecordError>>()?;
                gt.sort_by(|a, b| proposal_key(a.bbox).total_cmp(&proposal_key(b.bbox)));
                Ok::<_, RecordError>(gt)
            })
            .transpose()?;
        Ok(Sample {
            canvas: preprocess::canvas(&resized),
            rh,
            rw,
            caption: rec.scene.as_ref().and_then(|s| s.caption.clone()),
            objects,
            text,
            relations,
            closed: rec.closed,
            language: rec.language().to_string(),
        })
    }
}

fn reader_line(pic: &Picture, bbox: &[f64; 4], value: &str, charset: &[char]) -> Result<ReaderLine, RecordError> {
    let thousandths = bbox.map(|v| v as f32);
    let (x0, y0, x1, y1) = reader::crop_rect(thousandths, pic.w, pic.h);
    if x1 <= x0 || y1 <= y0 {
        return Err(RecordError(format!("text `{value}`: its box crops no pixels")));
    }
    let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
    for y in y0..y1 {
        crop.extend_from_slice(&pic.rgb[(y * pic.w + x0) * 3..(y * pic.w + x1) * 3]);
    }
    let strip = reader::strip_from_crop(&crop, y1 - y0, x1 - x0).map_err(|e| RecordError(e.to_string()))?;
    let windows = reader::windows(&strip);
    let ranges = (0..windows.len()).map(|j| frame_range(j, strip.w)).collect();
    let labels = value.chars().map(|c| charset.iter().position(|&k| k == c).map(|i| i as u32 + 1)).collect::<Option<Vec<u32>>>();
    Ok(ReaderLine { windows, ranges, labels })
}

/// The decoder's training sequence: input and target tokens with a weight per target.
#[derive(Debug, Clone)]
pub struct TextTarget {
    /// The whole sequence `[bos, lang, body..., (eos)]`.
    pub tokens: Vec<u32>,
    /// Weight of predicting `tokens[i + 1]` from the prefix `tokens[..=i]` (index `i`).
    pub weights: Vec<f64>,
}

enum Piece {
    Text(String),
    Reference(u32),
}

/// Build the decoder target of a sample. `encode` turns text into token ids, `lang` is the language tag id,
/// `bos` and `eos` the sequence markers, `vocab` the first reference token id.
pub fn text_target(cfg: &MirageConfig, s: &Sample, encode: &dyn Fn(&str) -> Vec<u32>, lang: u32, vocab: u32) -> Option<TextTarget> {
    // Lines as pieces, grouped by section so a gap can be detected.
    let mut sections: Vec<Vec<Vec<Piece>>> = Vec::new();
    let mut present = Vec::new();
    let line = |text: String| vec![Piece::Text(text)];
    // S
    present.push(s.caption.is_some());
    sections.push(s.caption.iter().map(|c| line(format!("S caption={c}"))).collect());
    // E
    present.push(s.objects.is_some());
    sections.push(
        s.objects
            .iter()
            .flatten()
            .enumerate()
            .map(|(k, o)| {
                let mut head = format!("E{} {}", k + 1, o.name);
                if let Some(p) = &o.position {
                    head.push_str(&format!(" @{p}"));
                }
                head.push_str(" *");
                let mut pieces = vec![Piece::Text(head), Piece::Reference(vocab + k as u32)];
                if !o.attributes.is_empty() {
                    let kv: Vec<String> = o.attributes.iter().map(|(k, v)| format!("{k}={v}")).collect();
                    pieces.push(Piece::Text(format!(" {{{}}}", kv.join("; "))));
                }
                pieces
            })
            .collect(),
    );
    // T
    present.push(s.text.is_some());
    sections.push(s.text.iter().flatten().enumerate().map(|(j, _)| vec![Piece::Text("T *".to_string()), Piece::Reference(vocab + cfg.detections as u32 + j as u32)]).collect());
    // R
    present.push(s.relations.is_some());
    sections.push(s.relations.iter().flatten().map(|(a, pred, b)| line(format!("R {} {} {}", a + 1, pred, b + 1))).collect());

    let mut tokens = vec![cfg.bos, lang];
    let mut weights = vec![0.0, 0.0]; // bos and the language tag are the prompt: never targets
    let mut emitted_any = false;
    let mut gap = false;
    for (sec, lines) in sections.iter().enumerate() {
        if !present[sec] {
            gap = true;
            continue;
        }
        for pieces in lines {
            if emitted_any {
                for t in encode("\n") {
                    tokens.push(t);
                    weights.push(1.0);
                }
            }
            let mut first_of_line = true;
            for piece in pieces {
                match piece {
                    Piece::Text(text) => {
                        for t in encode(text) {
                            tokens.push(t);
                            // The first token of a section after an unlabelled one is not supervised.
                            weights.push(if first_of_line && gap { 0.0 } else { 1.0 });
                            first_of_line = false;
                        }
                    }
                    Piece::Reference(id) => {
                        tokens.push(*id);
                        weights.push(1.0);
                        first_of_line = false;
                    }
                }
            }
            emitted_any = true;
            gap = false;
        }
    }
    // `weights[j]` is the weight of predicting `tokens[j]`; the decoder is fed `tokens[..n-1]` and predicts
    // `tokens[1..]`, so the weight of the prediction made at position `i` is `weights[i + 1]`.
    let mut aligned: Vec<f64> = weights[1..].to_vec();
    if s.closed {
        tokens.push(cfg.eos);
        aligned.push(1.0);
    }
    debug_assert_eq!(aligned.len(), tokens.len() - 1);
    if aligned.iter().all(|w| *w == 0.0) {
        return None;
    }
    Some(TextTarget { tokens, weights: aligned })
}

/// Heat map and regression targets of one kind (object or text) on the `side x side` stride-8 grid.
struct HeatTargets {
    heat: Vec<f64>,
    /// `(cell, [w, h, dx, dy])` at each positive cell.
    positives: Vec<(usize, [f64; 4])>,
}

fn heat_targets(boxes: &[[f32; 4]], side: usize) -> HeatTargets {
    let mut heat = vec![0f64; side * side];
    let mut positives = Vec::new();
    let s = side as f64;
    for b in boxes {
        let (w, h) = (f64::from(b[2] - b[0]) * s, f64::from(b[3] - b[1]) * s);
        let (cx, cy) = (f64::from(b[0] + b[2]) / 2.0 * s, f64::from(b[1] + b[3]) / 2.0 * s);
        let (ix, iy) = ((cx as usize).min(side - 1), (cy as usize).min(side - 1));
        let (sx, sy) = ((w / 6.0).max(0.5), (h / 6.0).max(0.5));
        for y in 0..side {
            for x in 0..side {
                let (dx, dy) = (x as f64 - ix as f64, y as f64 - iy as f64);
                let v = (-(dx * dx / (2.0 * sx * sx) + dy * dy / (2.0 * sy * sy))).exp();
                let cell = &mut heat[y * side + x];
                *cell = cell.max(v);
            }
        }
        heat[iy * side + ix] = 1.0;
        positives.push((iy * side + ix, [w, h, cx - ix as f64, cy - iy as f64]));
    }
    HeatTargets { heat, positives }
}

/// Even-odd point-in-polygon over several rings (flat `[x, y, ...]`).
fn inside(rings: &[Vec<f64>], x: f64, y: f64) -> bool {
    let mut crossings = 0;
    for ring in rings {
        let n = ring.len() / 2;
        for i in 0..n {
            let (x1, y1, x2, y2) = (ring[2 * i], ring[2 * i + 1], ring[2 * ((i + 1) % n)], ring[2 * ((i + 1) % n) + 1]);
            if (y1 > y) != (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1) + x1 {
                crossings += 1;
            }
        }
    }
    crossings % 2 == 1
}

/// The cell `(row, col)` of the `side x side` coefficient map that describes a box (the box-centre cell).
fn coefficient_cell(b: [f32; 4], side: usize) -> (usize, usize) {
    let cell = |lo: f32, hi: f32| ((side as f32 * (lo + hi) / 2.0) as usize).min(side - 1);
    (cell(b[1], b[3]), cell(b[0], b[2]))
}

/// The cells (row-major on a `field x field` grid) of an object's mask region and their 0/1 targets.
fn mask_region(o: &GtObject, rings: &[Vec<f64>], field: usize) -> (Vec<usize>, Vec<f64>) {
    let b = round_box(o.bbox);
    let f = field as f32;
    let centre = |lo: f32, hi: f32| ((f * (lo + hi) / 2.0) as usize).min(field - 1);
    let (cr, cc) = (centre(b[1], b[3]), centre(b[0], b[2]));
    let (mut idx, mut tgt) = (Vec::new(), Vec::new());
    for row in 0..field {
        let y = (row as f32 + 0.5) / f;
        for col in 0..field {
            let x = (col as f32 + 0.5) / f;
            if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) || (row == cr && col == cc) {
                idx.push(row * field + col);
                tgt.push(if inside(rings, f64::from(x), f64::from(y)) { 1.0 } else { 0.0 });
            }
        }
    }
    (idx, tgt)
}

/// A scalar loss and its named terms.
pub struct SampleLoss {
    /// The scalar to differentiate.
    pub total: Id,
    /// Weighted value of each term.
    pub terms: BTreeMap<&'static str, f64>,
}

fn channel<R: Real>(net: &mut Net<'_, R>, map: Id, cells: usize, ch: usize) -> Id {
    net.tape.plan(map, cells, (0..cells).map(|i| Seg::copy(i, i * DENSE_OUT_CHANNELS + ch, 1)).collect())
}

/// Build the loss graph of one sample. `target` is the decoder sequence (see [`text_target`]).
pub fn sample_loss<R: Real>(net: &mut Net<'_, R>, s: &Sample, target: Option<&TextTarget>, w: &TrainWeights) -> Option<SampleLoss> {
    let cfg = net.cfg;
    let mut terms: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut parts: Vec<Id> = Vec::new();
    let mut add = |net: &mut Net<'_, R>, name: &'static str, id: Id, weight: f64| {
        let scaled = net.tape.scale(id, weight);
        terms.insert(name, net.tape.val(scaled)[0].to_f64());
        parts.push(scaled);
    };

    if w.needs_vision() {
        let canvas = net.constant(s.canvas.iter().map(|&v| r::<R>(f64::from(v))).collect());
        let (h, wd) = (s.rh, s.rw);
        let enc = net.encoder(canvas, h, wd);
        let dense = net.dense(&enc, h, wd);
        let side = cfg.canvas / cfg.stride;
        let cells = side * side;
        let valid = valid_cells(h, wd, side);

        if w.heat > 0.0 || w.regression > 0.0 {
            let kinds: [(usize, Option<Vec<[f32; 4]>>, [usize; 4]); 2] = [
                (0, s.objects.as_ref().map(|o| o.iter().map(|g| g.bbox).collect()), [2, 3, 6, 7]),
                (1, s.text.as_ref().map(|t| t.iter().map(|g| g.bbox).collect()), [4, 5, 8, 9]),
            ];
            for (logit_ch, boxes, reg_ch) in kinds {
                let Some(boxes) = boxes else { continue };
                let tgt = heat_targets(&boxes, side);
                let norm = tgt.positives.len().max(1) as f64;
                if w.heat > 0.0 {
                    let logits = channel(net, dense.head, cells, logit_ch);
                    let weight: Vec<f64> = valid.iter().map(|&v| f64::from(u8::from(v))).collect();
                    let focal = net.tape.focal_heat(logits, &tgt.heat, &weight, f64::from(cfg.heat_alpha), norm);
                    add(net, if logit_ch == 0 { "heat_object" } else { "heat_text" }, focal, w.heat);
                }
                if w.regression > 0.0 && !tgt.positives.is_empty() {
                    for (k, &ch) in reg_ch.iter().enumerate() {
                        let mut value = vec![0f64; cells];
                        let mut weight = vec![0f64; cells];
                        for (cell, t) in &tgt.positives {
                            value[*cell] = t[k];
                            weight[*cell] = 1.0;
                        }
                        let pred = channel(net, dense.head, cells, ch);
                        let l1 = net.tape.l1(pred, &value, &weight, norm * 4.0);
                        add(net, if logit_ch == 0 { "reg_object" } else { "reg_text" }, l1, w.regression);
                    }
                }
            }
        }

        if w.mask > 0.0 {
            if let Some(objects) = &s.objects {
                let masked: Vec<&GtObject> = objects.iter().filter(|o| o.polygons.is_some()).collect();
                if !masked.is_empty() {
                    let (protos, coefs) = net.mask_maps(dense.head_in, &enc, h, wd);
                    let field = 2 * side;
                    let mut total: Option<Id> = None;
                    for o in &masked {
                        let rings = o.polygons.as_ref().expect("filtered on presence");
                        let (idx, tgt) = mask_region(o, rings, field);
                        let (row, col) = coefficient_cell(round_box(o.bbox), side);
                        let coef = net.tape.gather_rows(coefs, MASK_RANK, &[row * side + col]);
                        let region = net.tape.gather_rows(protos, MASK_RANK, &idx);
                        let logits = net.tape.linear(region, coef, idx.len(), MASK_RANK, 1);
                        let l = net.tape.bce_dice(logits, &tgt, 1.0, 1.0);
                        total = Some(match total {
                            Some(t) => net.tape.add(t, l),
                            None => l,
                        });
                    }
                    let mean = net.tape.scale(total.expect("at least one masked object"), 1.0 / masked.len() as f64);
                    add(net, "mask", mean, w.mask);
                }
            }
        }

        if let (true, Some(t)) = (w.text > 0.0, target) {
            let objects: Vec<Slot> = s.objects.iter().flatten().enumerate().map(|(k, o)| Slot { bbox: o.bbox, token: cfg.text + crate::config::LANGUAGE_TAGS + k }).collect();
            let texts: Vec<Slot> = s.text.iter().flatten().enumerate().map(|(j, g)| Slot { bbox: g.bbox, token: cfg.text + crate::config::LANGUAGE_TAGS + cfg.detections + j }).collect();
            let slots: Vec<Slot> = objects.into_iter().chain(texts).collect();
            let grid_valid = valid_cells(h, wd, cfg.grid());
            let memory = net.decoder_memory(&enc, dense.head_in, &grid_valid, &slots);
            let rows = net.tape.len(memory) / cfg.dim;
            let n = t.tokens.len() - 1;
            let total_w: f64 = t.weights.iter().sum();
            let logits = net.decoder_logits(&t.tokens[..n], memory, rows);
            let norm: Vec<f64> = t.weights.iter().map(|x| x / total_w).collect();
            let ce = net.tape.softmax_ce(logits, cfg.embedding_rows(), &t.tokens[1..], &norm);
            add(net, "text", ce, w.text);
        }
    }

    if w.ctc > 0.0 {
        let mut total: Option<Id> = None;
        let mut lines = 0usize;
        for g in s.text.iter().flatten() {
            let Some(line) = &g.line else { continue };
            let Some(labels) = &line.labels else { continue };
            let frames: usize = line.ranges.iter().map(|(lo, hi)| hi - lo).sum();
            if !super::loss::ctc_feasible(frames, labels) {
                continue;
            }
            let mut joined: Option<Id> = None;
            for (win, &(lo, hi)) in line.windows.iter().zip(&line.ranges) {
                let pixels = net.constant(win.iter().map(|&p| r::<R>(f64::from(p) / 127.5 - 1.0)).collect());
                let logits = net.reader(pixels);
                let kept = net.tape.slice(logits, lo * crate::config::READER_CLASSES, (hi - lo) * crate::config::READER_CLASSES);
                joined = Some(match joined {
                    Some(j) => net.tape.concat(j, kept),
                    None => kept,
                });
            }
            let loss = net.tape.ctc(joined.expect("a line has a window"), crate::config::READER_CLASSES, labels);
            let loss = net.tape.scale(loss, 1.0 / labels.len() as f64);
            total = Some(match total {
                Some(t) => net.tape.add(t, loss),
                None => loss,
            });
            lines += 1;
        }
        if let Some(t) = total {
            let mean = net.tape.scale(t, 1.0 / lines as f64);
            add(net, "ctc", mean, w.ctc);
        }
    }

    let mut iter = parts.into_iter();
    let mut total = iter.next()?;
    for p in iter {
        total = net.tape.add(total, p);
    }
    Some(SampleLoss { total, terms })
}
