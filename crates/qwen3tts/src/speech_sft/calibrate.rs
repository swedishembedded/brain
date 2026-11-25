// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained speech
// synthesizers by measurement on recorded speech, for its clients. If your
// team needs expertise in compressing text-to-speech models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Calibration of a Qwen3-TTS checkpoint on recorded speech, for weight
//! surgery: what each layer, channel and stream direction carries, measured
//! by the speech objective's own teacher-forced forward.
//!
//! * [`Evaluator`]: calibration records as fixed batches and the unedited
//!   checkpoint's predictions on them; [`Evaluator::measure`] scores any
//!   edited checkpoint against them (its loss and its KL divergence from the
//!   unedited predictions, Talker and code predictor apart).
//! * [`calibrate`]: one pass of the unedited checkpoint collecting
//!   [`model::surgery::Calibration`]: per layer the loss and KL without it
//!   and the cosine between the stream entering and leaving it; per MLP
//!   channel its summed squared activation; per text-projection channel its
//!   activation over the calibration text; per stream the second moment of
//!   its unit-mean-square states (and of the states entering its shared
//!   norm).
//!
//! The Fisher and group-Taylor channel estimates need the gradient loop of
//! `modelopt`, which runs over [`Evaluator::load`] on the same batches.

use std::collections::{BTreeMap, HashMap};

use gpu_core::provider::library::GemmPrecision;
use model::hostmath::{linear_rows, rmsnorm_rows, silu};
use model::surgery::{Calibration, ChannelStats, LayerAblation, StreamStats};

use super::model::{DecoderStack, Example, ForwardStats, Logits, SftShape, SpeechSftModel};
use super::objective::{example, RowMap};
use super::{Base, LrMultipliers};
use crate::prompt::{Part, TextLayout, TtsSpecials};
use crate::shard::Shard;

/// How much calibration measures.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationOpts {
    /// Records of the shard calibrated on, spread evenly over it (`0`: all).
    pub records: usize,
    /// Records per batch.
    pub batch: usize,
    /// The code predictor's weight in the loss and the KL (the training objective's).
    pub mtp_weight: f32,
    /// Ablate every layer of both stacks.
    pub layers: bool,
    /// Rows per stream point and batch the statistics take (spread over the batch's rows).
    pub sample_rows: usize,
    pub gemm: GemmPrecision,
}

/// A measured checkpoint: its loss and its divergence from the unedited one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Measurement {
    pub loss: f64,
    pub talker_ce: f64,
    pub code_predictor_ce: f64,
    /// Mean KL divergence from the unedited predictions, per supervised position.
    pub talker_kl: f64,
    pub code_predictor_kl: f64,
}

impl Measurement {
    /// The KL the loss weights: `talker + mtp_weight * code predictor`.
    pub fn kl(&self, mtp_weight: f32) -> f64 {
        self.talker_kl + f64::from(mtp_weight) * self.code_predictor_kl
    }
}

/// How an edited checkpoint reads the calibration records: their text tokens
/// renumbered (a trimmed vocabulary), their speaker x-vectors moved into a
/// rotated stream.
#[derive(Default)]
pub struct Inputs<'a> {
    pub text: Option<&'a TextMap<'a>>,
    pub xvector: Option<&'a XVectorMap<'a>>,
}

/// A calibration text token's id in an edited vocabulary (`None`: not kept).
pub type TextMap<'a> = dyn Fn(u32) -> Option<u32> + 'a;
/// A speaker x-vector as an edited stream reads it.
pub type XVectorMap<'a> = dyn Fn(&[f32]) -> Vec<f32> + 'a;

/// The calibration batches and the unedited checkpoint's predictions on them.
pub struct Evaluator {
    batches: Vec<Vec<Example>>,
    teacher: Vec<Logits>,
    baseline: Measurement,
    mtp_weight: f32,
    gemm: GemmPrecision,
    block: u32,
    frames: u32,
    batch: u32,
}

/// The row of each of a shard's languages and voices in the checkpoint's own tables.
pub fn row_map(sp: &TtsSpecials, shard: &Shard) -> Result<RowMap, String> {
    let languages = shard.meta.languages.iter().map(|l| if l == "auto" { Ok(None) } else { sp.lang.get(l).copied().map(Some).ok_or_else(|| format!("the checkpoint has no language {l:?} the shard names")) }).collect::<Result<_, _>>()?;
    let voices = shard.meta.voices.iter().map(|v| sp.spk_id.get(v).copied().ok_or_else(|| format!("the checkpoint has no voice {v:?} the shard names"))).collect::<Result<_, _>>()?;
    Ok(RowMap { languages, voices, text: None })
}

/// `n` of `len` record indices spread evenly (`0`: all): a shard may hold its speakers one after another.
pub fn spread(len: usize, n: usize) -> Vec<usize> {
    let n = if n == 0 { len } else { n.min(len) };
    (0..n).map(|i| i * len / n).collect()
}

/// Rows of a logit set one host thread scores at a time in [`mean_kl`]; fixed, so the sum does not depend on the
/// thread count.
const KL_CHUNK_ROWS: usize = 256;
/// The most host threads [`mean_kl`] uses.
const KL_THREADS: usize = 64;

/// `Σ KL(p || q)` over the rows of two `[rows, vocab]` logit sets.
fn sum_kl(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let log_softmax = |row: &[f32]| -> Vec<f64> {
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let z = row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
        row.iter().map(|&v| v as f64 - z).collect()
    };
    p.chunks(vocab)
        .zip(q.chunks(vocab))
        .map(|(a, b)| {
            let (la, lb) = (log_softmax(a), log_softmax(b));
            la.iter().zip(&lb).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum()
}

/// Mean KL divergence `KL(p || q)` over the rows of two `[rows, vocab]` logit sets, chunks of rows scored on
/// host threads and summed in row order.
fn mean_kl(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    let rows = p.len() / vocab;
    let chunk = KL_CHUNK_ROWS * vocab;
    let chunks: Vec<(&[f32], &[f32])> = p.chunks(chunk).zip(q.chunks(chunk)).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, KL_THREADS).min(chunks.len());
    let mut sums = vec![0f64; chunks.len()];
    std::thread::scope(|s| {
        let per = chunks.len().div_ceil(threads);
        for (part, out) in chunks.chunks(per).zip(sums.chunks_mut(per)) {
            s.spawn(move || part.iter().zip(out.iter_mut()).for_each(|((a, b), o)| *o = sum_kl(a, b, vocab)));
        }
    });
    sums.iter().sum::<f64>() / rows as f64
}

fn measurement(stats: &ForwardStats, kl: (f64, f64), mtp_weight: f32) -> Measurement {
    let mean = |s: f64, n: u32| if n == 0 { 0.0 } else { s / f64::from(n) };
    Measurement { loss: f64::from(stats.loss(&super::model::LossWeights::sft(mtp_weight))), talker_ce: mean(stats.talker_sum, stats.talker_count), code_predictor_ce: mean(stats.cp_sum, stats.cp_count), talker_kl: kl.0, code_predictor_kl: kl.1 }
}

impl Evaluator {
    /// `opts.records` records of `shard`, spread evenly over it, as batches of
    /// `opts.batch` (even records in the non-streaming layout, odd in the
    /// streaming one), scored on the checkpoint `base`. Answers the
    /// evaluator and the model it built, which stays loaded for calibration.
    pub fn new(base: Base, shard: &Shard, opts: &CalibrationOpts) -> Result<(Evaluator, SpeechSftModel), String> {
        Self::on_records(base, shard, &spread(shard.len(), opts.records), opts)
    }

    /// [`Self::new`] on the records `records` of `shard`.
    pub fn on_records(base: Base, shard: &Shard, records: &[usize], opts: &CalibrationOpts) -> Result<(Evaluator, SpeechSftModel), String> {
        if opts.batch == 0 {
            return Err("calibration: a batch holds at least one record".into());
        }
        let rows = row_map(&base.specials, shard)?;
        if records.is_empty() {
            return Err("calibration: no record to calibrate on".into());
        }
        if let Some(&r) = records.iter().find(|&&r| r >= shard.len()) {
            return Err(format!("calibration: record {r} is past the shard's {}", shard.len()));
        }
        let examples: Vec<Example> = records.iter().enumerate().map(|(i, &r)| example(&base.specials, &rows, shard, r, if i % 2 == 0 { TextLayout::NonStreaming } else { TextLayout::Streaming })).collect::<Result<_, _>>()?;
        let g = base.mtp.num_code_groups as usize;
        let block = examples.iter().map(|e| SpeechSftModel::rows_needed(e.plan.len(), e.codes.len() / g)).max().unwrap_or(1) as u32;
        let batches: Vec<Vec<Example>> = examples.chunks(opts.batch).map(<[Example]>::to_vec).collect();
        let frames = batches.iter().map(|b| b.iter().map(|e| e.codes.len() / g).sum::<usize>()).max().unwrap_or(1) as u32;
        let mut ev = Evaluator { batches, teacher: Vec::new(), baseline: Measurement::default(), mtp_weight: opts.mtp_weight, gemm: opts.gemm, block, frames, batch: opts.batch as u32 };
        let m = ev.build(base)?;
        let mut stats = ForwardStats::default();
        for i in 0..ev.batches.len() {
            ev.load(&m, i)?;
            stats.add(&m.score());
            ev.teacher.push(m.logits());
        }
        ev.baseline = measurement(&stats, (0.0, 0.0), opts.mtp_weight);
        Ok((ev, m))
    }

    /// The unedited checkpoint's measurement.
    pub fn baseline(&self) -> Measurement {
        self.baseline
    }

    pub fn batches(&self) -> usize {
        self.batches.len()
    }

    /// Calibration rows: the supervised positions of both decoders.
    pub fn rows(&self) -> u64 {
        self.teacher.iter().map(|l| (l.talker_targets.len() + l.cp_targets.len()) as u64).sum()
    }

    /// A model of the checkpoint `base` sized for the calibration batches.
    pub fn build(&self, base: Base) -> Result<SpeechSftModel, String> {
        let shape = SftShape { talker: base.talker, mtp: base.mtp, specials: base.specials, batch: self.batch, block: self.block, frames: self.frames, mtp_weight: self.mtp_weight, lr: LrMultipliers::default(), new_rows: Vec::new(), text_inter: base.text_inter, gemm: self.gemm, gt_weight: 1.0, kd: None };
        // seed: the decoders' token tables, which every row bypasses, are all it draws.
        SpeechSftModel::build(shape, base.tensors, data::rng::random_seed())
    }

    /// Put calibration batch `i` on `m`, unchanged.
    pub fn load(&self, m: &SpeechSftModel, i: usize) -> Result<(), String> {
        m.load(self.batches[i].clone())
    }

    fn load_with(&self, m: &SpeechSftModel, i: usize, inputs: &Inputs<'_>) -> Result<(), String> {
        let mut batch = self.batches[i].clone();
        for e in &mut batch {
            if let Some(map) = inputs.text {
                let renumber = |id: &mut u32| -> Result<(), String> {
                    *id = map(*id).ok_or_else(|| format!("calibration text token {id} is not in the edited vocabulary"))?;
                    Ok(())
                };
                for parts in &mut e.plan.rows {
                    for p in parts.iter_mut() {
                        if let Part::Text(id) = p {
                            renumber(id)?;
                        }
                    }
                }
                e.plan.trailing.iter_mut().try_for_each(renumber)?;
                renumber(&mut e.plan.tts_pad)?;
            }
            if let Some(f) = inputs.xvector {
                e.xvector = f(&e.xvector);
            }
        }
        m.load(batch)
    }

    /// Score `m` (an edited checkpoint built by [`Self::build`], or the
    /// calibration model with weights masked) against the unedited
    /// predictions.
    pub fn measure_model(&self, m: &SpeechSftModel, inputs: &Inputs<'_>) -> Result<Measurement, String> {
        let mut stats = ForwardStats::default();
        let (mut tk, mut ck, mut tn, mut cn) = (0f64, 0f64, 0usize, 0usize);
        for i in 0..self.batches.len() {
            self.load_with(m, i, inputs)?;
            stats.add(&m.score());
            let (t, s) = (&self.teacher[i], m.logits());
            if s.talker_targets != t.talker_targets || s.cp_targets != t.cp_targets {
                return Err("calibration: the edited checkpoint supervises other positions than the unedited one".into());
            }
            tk += mean_kl(&t.talker, &s.talker, t.talker_vocab) * t.talker_targets.len() as f64;
            ck += mean_kl(&t.cp, &s.cp, t.cp_vocab) * t.cp_targets.len() as f64;
            tn += t.talker_targets.len();
            cn += t.cp_targets.len();
        }
        let loss = measurement(&stats, (tk / tn.max(1) as f64, ck / cn.max(1) as f64), self.mtp_weight);
        if !loss.loss.is_finite() {
            return Err(format!("calibration: the edited checkpoint's loss is {}", loss.loss));
        }
        Ok(loss)
    }

    /// Build the edited checkpoint `base` and score it.
    pub fn measure(&self, base: Base, inputs: &Inputs<'_>) -> Result<Measurement, String> {
        let m = self.build(base)?;
        self.measure_model(&m, inputs)
    }
}

/// Accumulates `XᵀX` of unit-mean-square rows into a `[d, d]` f64 matrix.
fn add_second_moment(cov: &mut [f64], rows: &[f32], d: usize) -> u64 {
    let n = rows.len() / d;
    let mut x: Vec<f32> = rows.to_vec();
    for row in x.chunks_mut(d) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + 1e-12).sqrt();
        row.iter_mut().for_each(|v| *v *= inv);
    }
    let mut xt = vec![0f32; x.len()];
    for r in 0..n {
        for c in 0..d {
            xt[c * n + r] = x[r * d + c];
        }
    }
    let m = linear_rows(&xt, &xt, d, n, d);
    cov.iter_mut().zip(&m).for_each(|(c, v)| *c += f64::from(*v));
    n as u64
}

/// At most `n` rows of `rows` (`width` wide), spread evenly.
fn sample(rows: &[f32], width: usize, n: usize) -> Vec<f32> {
    let total = rows.len() / width;
    if total <= n {
        return rows.to_vec();
    }
    (0..n).flat_map(|i| rows[(i * total / n) * width..(i * total / n + 1) * width].iter().copied()).collect()
}

/// The mean cosine between corresponding rows of `a` and `b`.
fn mean_cosine(a: &[f32], b: &[f32], width: usize) -> (f64, u64) {
    let mut sum = 0f64;
    let mut n = 0u64;
    for (x, y) in a.chunks(width).zip(b.chunks(width)) {
        let dot: f64 = x.iter().zip(y).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum();
        let (nx, ny) = (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt(), y.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt());
        if nx > 0.0 && ny > 0.0 {
            sum += dot / (nx * ny);
            n += 1;
        }
    }
    (sum, n)
}

/// One decoder stack's names and shape for the statistics.
struct Stack {
    which: DecoderStack,
    name: &'static str,
    prefix: &'static str,
    layers: usize,
    width: usize,
    ff: usize,
    eps: f32,
}

/// Calibrate the checkpoint `base` (content digest `source`) on `shard`.
/// `host` holds its tensors (upstream names) for the host-side activation
/// statistics.
pub fn calibrate(base: Base, shard: &Shard, source: &str, opts: &CalibrationOpts, on_progress: &mut dyn FnMut(&str)) -> Result<(Evaluator, Calibration), String> {
    let host: HashMap<String, Vec<f32>> = base.tensors.clone();
    let stacks = [
        Stack { which: DecoderStack::Talker, name: "talker", prefix: "talker.model.layers.", layers: base.talker.n_layers as usize, width: base.talker.d_model as usize, ff: base.talker.d_ff as usize, eps: base.talker.rms_norm_eps },
        Stack { which: DecoderStack::CodePredictor, name: "code_predictor", prefix: "talker.code_predictor.model.layers.", layers: base.mtp.n_layers as usize, width: base.mtp.d_model as usize, ff: base.mtp.d_ff as usize, eps: base.mtp.rms_norm_eps },
    ];
    let projected = base.mtp.embedding_dim != base.mtp.d_model;
    let (text_hidden, text_inter) = (base.talker.text_hidden_size as usize, base.text_inter as usize);
    let (ev, m) = Evaluator::new(base, shard, opts)?;
    on_progress(&format!("calibration records, unedited: loss {:.4} over {} batches", ev.baseline.loss, ev.batches()));
    let w = |name: &str| host.get(name).ok_or_else(|| format!("calibration: no tensor {name}"));

    let mut cal = Calibration { source: source.to_string(), records: ev.batches.iter().map(Vec::len).sum(), rows: ev.rows(), loss: ev.baseline.loss, ..Calibration::default() };
    // Statistics from the unedited forward of every batch.
    let mut cosine: Vec<Vec<(f64, u64)>> = stacks.iter().map(|s| vec![(0.0, 0); s.layers]).collect();
    let mut streams: Vec<StreamStats> = stacks.iter().map(|s| StreamStats { width: s.width, covariance: vec![0.0; s.width * s.width], ..StreamStats::default() }).collect();
    let mut shared = vec![0f64; stacks[0].width * stacks[0].width];
    let mut act: Vec<Vec<ChannelStats>> = stacks.iter().map(|s| vec![ChannelStats { sum_sq: vec![0.0; s.ff], rows: 0 }; s.layers]).collect();
    let mut text_counts: BTreeMap<u32, u64> = BTreeMap::new();
    for i in 0..ev.batches() {
        ev.load(&m, i)?;
        m.score();
        for e in &ev.batches[i] {
            let parts = e.plan.rows.iter().flatten().filter_map(|p| if let Part::Text(id) = p { Some(*id) } else { None });
            for id in parts.chain(e.plan.trailing.iter().copied()) {
                *text_counts.entry(id).or_default() += 1;
            }
        }
        for (si, s) in stacks.iter().enumerate() {
            let mut previous: Option<Vec<f32>> = None;
            let mut err = None;
            m.for_each_residual(s.which, &mut |point, rows| {
                let d = s.width;
                let sampled = sample(rows, d, opts.sample_rows);
                streams[si].rows += add_second_moment(&mut streams[si].covariance, &sampled, d);
                if si == 0 && point == 2 * s.layers {
                    add_second_moment(&mut shared, &sampled, d);
                }
                if point % 2 == 0 {
                    if let Some(prev) = previous.take() {
                        let c = mean_cosine(&prev, rows, d);
                        let slot = &mut cosine[si][point / 2 - 1];
                        (slot.0, slot.1) = (slot.0 + c.0, slot.1 + c.1);
                    }
                    previous = Some(rows.to_vec());
                } else {
                    // Inside layer `point / 2`, after attention: the MLP's input and its hidden activation.
                    let l = point / 2;
                    let leaf = |n: &str| format!("{}{l}.{n}", s.prefix);
                    let result = (|| -> Result<(), String> {
                        let n = sampled.len() / d;
                        let x = rmsnorm_rows(&sampled, w(&leaf("post_attention_layernorm.weight"))?, n, d, s.eps);
                        let gate = linear_rows(&x, w(&leaf("mlp.gate_proj.weight"))?, n, d, s.ff);
                        let up = linear_rows(&x, w(&leaf("mlp.up_proj.weight"))?, n, d, s.ff);
                        let stats = &mut act[si][l];
                        for (row_g, row_u) in gate.chunks(s.ff).zip(up.chunks(s.ff)) {
                            for ((acc, &gv), &uv) in stats.sum_sq.iter_mut().zip(row_g).zip(row_u) {
                                *acc += f64::from(silu(gv) * uv).powi(2);
                            }
                        }
                        stats.rows += n as u64;
                        Ok(())
                    })();
                    if let Err(e) = result {
                        err.get_or_insert(e);
                    }
                }
            });
            if let Some(e) = err {
                return Err(e);
            }
        }
        on_progress(&format!("statistics: batch {} of {}", i + 1, ev.batches()));
    }
    if projected {
        streams[0].shared.insert("talker.model.norm.weight".into(), shared);
    }
    for (si, s) in stacks.iter().enumerate() {
        for (l, a) in act[si].drain(..).enumerate() {
            cal.channels.insert(format!("{}.layers.{l}.mlp", s.name), a);
        }
        cal.streams.insert(s.name.to_string(), std::mem::take(&mut streams[si]));
    }

    // The text projection's channels over the calibration text: each token's activation, as often as it occurs.
    let (emb, fc1, fc1_b) = (w("talker.model.text_embedding.weight")?, w("talker.text_projection.linear_fc1.weight")?, w("talker.text_projection.linear_fc1.bias")?);
    let ids: Vec<u32> = text_counts.keys().copied().collect();
    let rows: Vec<f32> = ids.iter().flat_map(|&t| emb[t as usize * text_hidden..(t as usize + 1) * text_hidden].iter().copied()).collect();
    let pre = linear_rows(&rows, fc1, ids.len(), text_hidden, text_inter);
    let mut text = ChannelStats { sum_sq: vec![0.0; text_inter], rows: 0 };
    for (row, id) in pre.chunks(text_inter).zip(&ids) {
        let count = text_counts[id];
        for ((acc, &v), &b) in text.sum_sq.iter_mut().zip(row).zip(fc1_b) {
            *acc += count as f64 * f64::from(silu(v + b)).powi(2);
        }
        text.rows += count;
    }
    cal.channels.insert("talker.text".into(), text);

    for (si, s) in stacks.iter().enumerate() {
        let mut ablations = Vec::with_capacity(s.layers);
        for (l, &(cos, n)) in cosine[si].iter().enumerate() {
            let layer = if opts.layers {
                let outputs = [format!("{}{l}.self_attn.o_proj.weight", s.prefix), format!("{}{l}.mlp.down_proj.weight", s.prefix)];
                let saved: Vec<Vec<f32>> = outputs.iter().map(|n| model::Model::read_weight(&m, n)).collect();
                for (n, v) in outputs.iter().zip(&saved) {
                    model::Model::write_weight(&m, n, &vec![0.0; v.len()]);
                }
                let measured = ev.measure_model(&m, &Inputs::default());
                for (n, v) in outputs.iter().zip(&saved) {
                    model::Model::write_weight(&m, n, v);
                }
                let measured = measured?;
                on_progress(&format!("{}.layers {l}: loss {:.4} kl {:.4}", s.name, measured.loss, measured.kl(opts.mtp_weight)));
                LayerAblation { loss: Some(measured.loss), kl: Some(measured.kl(opts.mtp_weight)), cosine: cos / n.max(1) as f64 }
            } else {
                LayerAblation { loss: None, kl: None, cosine: cos / n.max(1) as f64 }
            };
            ablations.push(layer);
        }
        cal.layers.insert(format!("{}.layers", s.name), ablations);
    }
    Ok((ev, cal))
}
