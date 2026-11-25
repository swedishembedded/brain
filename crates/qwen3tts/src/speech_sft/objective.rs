// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The teacher-forced speech objective over a training shard, for the shared
//! fit loop ([`model::train::fit_controlled`]).
//!
//! Records are drawn in epochs without replacement, `batch` per micro-step, in
//! an order fixed by the run's seed; a resumed run is positioned by the count
//! of micro-steps it already drew, so it continues the stream an uninterrupted
//! run would have followed. Each draw of a record also draws its text layout
//! (streaming with probability `streaming_fraction`), keyed on the seed and the
//! draw's position, so a resumed run draws the same layouts too. The shared
//! loop's `batch_size` must be the objective's batch: it positions a resumed
//! run by records drawn.

use std::time::Instant;

use data::rng::Rng;
use model::objective::Objective;
use model::train::StepWork;

use super::model::{Example, ForwardStats, SpeechSftModel};
use crate::prompt::{PromptPlan, PromptSpec, SpeakerSlot, TextLayout, TtsSpecials};
use crate::shard::Shard;

/// Which codec row each of a shard's languages and voices is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMap {
    /// Per [`crate::shard::ShardMeta::languages`] entry: its language row, or
    /// `None` for `"auto"` (the no-language prefix).
    pub languages: Vec<Option<u32>>,
    /// Per [`crate::shard::ShardMeta::voices`] entry: its speaker row.
    pub voices: Vec<u32>,
}

/// The layout draw `position` of a stream keyed by `key` takes.
fn draw_layout(key: u64, position: u64, streaming_fraction: f32) -> TextLayout {
    let mut rng = Rng::new(key ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    if rng.next_f32() < streaming_fraction {
        TextLayout::Streaming
    } else {
        TextLayout::NonStreaming
    }
}

/// Domain separation of the held-out layouts from the training stream: every
/// evaluation scores each held-out record in the same layout.
const HELD_OUT_KEY: u64 = 0x4E1D_0007_5EED_0001;

/// The example record `r` of `shard` makes in `layout`.
pub fn example(sp: &TtsSpecials, rows: &RowMap, shard: &Shard, r: usize, layout: TextLayout) -> Result<Example, String> {
    let rec = shard.record(r);
    let named = |what: &str| format!("record {r} ({}): {what}", rec.id.id);
    let (role_ids, text_ids) = crate::pipeline::split_input_ids(rec.text_ids).map_err(|e| named(&e))?;
    let speaker = match rec.voice {
        Some(v) => SpeakerSlot::Codec(*rows.voices.get(v as usize).ok_or_else(|| named(&format!("voice {v} has no row")))?),
        None => SpeakerSlot::XVector,
    };
    let language_id = *rows.languages.get(rec.language as usize).ok_or_else(|| named(&format!("language {} has no row", rec.language)))?;
    let plan = PromptPlan::build(sp, &PromptSpec { role_ids: &role_ids, text_ids: &text_ids, instruct_ids: &[], speaker, language_id, layout }).map_err(|e| named(&e))?;
    if rec.codes.is_empty() {
        return Err(named("no frames"));
    }
    Ok(Example { plan, xvector: rec.xvector.to_vec(), codes: rec.codes.iter().map(|&c| u32::from(c)).collect(), complete: rec.complete })
}

/// The most Talker rows any record of `shard` takes, in either layout.
pub fn rows_needed(sp: &TtsSpecials, rows: &RowMap, shard: &Shard) -> Result<usize, String> {
    let mut most = 0;
    for r in 0..shard.len() {
        for layout in [TextLayout::Streaming, TextLayout::NonStreaming] {
            let ex = example(sp, rows, shard, r, layout)?;
            most = most.max(SpeechSftModel::rows_needed(ex.plan.len(), ex.codes.len() / shard.meta.codebooks as usize));
        }
    }
    Ok(most)
}

/// The speech objective. See the module documentation.
pub struct SpeechObjective<'a> {
    train: &'a Shard,
    held_out: Option<&'a Shard>,
    rows: RowMap,
    sp: TtsSpecials,
    batch: usize,
    streaming_fraction: f32,
    mtp_weight: f32,
    seed: u64,
    /// Records drawn so far in the whole run: the position of the next one in the stream.
    drawn: u64,
    order: Option<(u64, Vec<usize>)>,
    /// What the current step's micro-steps measured.
    step: ForwardStats,
    held: Option<ForwardStats>,
    work: StepWork,
}

impl<'a> SpeechObjective<'a> {
    pub fn new(train: &'a Shard, held_out: Option<&'a Shard>, rows: RowMap, sp: TtsSpecials, batch: usize, streaming_fraction: f32, mtp_weight: f32) -> Result<SpeechObjective<'a>, String> {
        if train.is_empty() {
            return Err("speech sft: the training shard holds no record".to_string());
        }
        if batch == 0 {
            return Err("speech sft: batch must be at least 1".to_string());
        }
        if !(0.0..=1.0).contains(&streaming_fraction) {
            return Err(format!("speech sft: streaming_fraction must be in [0, 1], got {streaming_fraction}"));
        }
        Ok(SpeechObjective { train, held_out, rows, sp, batch, streaming_fraction, mtp_weight, seed: data::rng::random_seed(), drawn: 0, order: None, step: ForwardStats::default(), held: None, work: StepWork::default() })
    }

    /// The record at position `pos` of the training stream.
    fn record_at(&mut self, pos: u64) -> usize {
        let n = self.train.len() as u64;
        let epoch = pos / n;
        if self.order.as_ref().is_none_or(|(e, _)| *e != epoch) {
            self.order = Some((epoch, model::train::epoch_order(self.seed, n as usize, epoch)));
        }
        self.order.as_ref().expect("just set").1[(pos % n) as usize]
    }

    /// The next micro-batch of the training stream.
    fn next_batch(&mut self) -> Vec<Example> {
        let first = self.drawn;
        self.drawn += self.batch as u64;
        (first..first + self.batch as u64)
            .map(|pos| {
                let r = self.record_at(pos);
                let layout = draw_layout(self.seed, pos, self.streaming_fraction);
                example(&self.sp, &self.rows, self.train, r, layout).unwrap_or_else(|e| panic!("speech sft: {e}"))
            })
            .collect()
    }

    /// Teacher-forced measurements of every record of `shard` in order, `batch`
    /// at a time, at most `batches` batches (`0`: all).
    fn score(&self, model: &SpeechSftModel, shard: &Shard, batches: u32) -> ForwardStats {
        let mut total = ForwardStats::default();
        let order: Vec<usize> = (0..shard.len()).collect();
        for (i, chunk) in order.chunks(self.batch).enumerate() {
            if batches > 0 && i as u32 >= batches {
                break;
            }
            let examples = chunk.iter().map(|&r| example(&self.sp, &self.rows, shard, r, draw_layout(HELD_OUT_KEY, r as u64, self.streaming_fraction)).unwrap_or_else(|e| panic!("speech sft: held out: {e}"))).collect();
            model.load(examples).unwrap_or_else(|e| panic!("speech sft: held out: {e}"));
            total.add(&model.score());
        }
        total
    }

    /// The held-out measurements of the latest evaluation.
    pub fn held_out(&self) -> Option<ForwardStats> {
        self.held
    }

    /// What the latest step's micro-steps measured, summed.
    pub fn last_step(&self) -> ForwardStats {
        self.step
    }
}

fn mean(sum: f64, n: u32) -> f32 {
    if n == 0 {
        0.0
    } else {
        (sum / f64::from(n)) as f32
    }
}

fn ratio(hits: u32, n: u32) -> f32 {
    if n == 0 {
        0.0
    } else {
        hits as f32 / n as f32
    }
}

impl Objective<SpeechSftModel> for SpeechObjective<'_> {
    fn regime(&self) -> &'static str {
        "speech_sft"
    }

    /// Every record of both shards builds an example that fits the model, in
    /// both layouts, before any step.
    fn check_data(&mut self, model: &SpeechSftModel) -> Result<(), std::io::Error> {
        let shape = model.shape();
        let g = shape.mtp.num_code_groups as usize;
        for (what, shard) in std::iter::once(("train", self.train)).chain(self.held_out.map(|h| ("held out", h))) {
            let meta = &shard.meta;
            let refuse = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("speech sft: {what} shard: {why}"));
            if meta.codebooks as usize != g || meta.codebook_size != shape.mtp.vocab || meta.xvector_dim as usize != shape.d() {
                return Err(refuse(format!(
                    "{} codebooks of {} codes and {}-wide x-vectors; the checkpoint takes {g} codebooks of {} codes and {}-wide x-vectors",
                    meta.codebooks,
                    meta.codebook_size,
                    meta.xvector_dim,
                    shape.mtp.vocab,
                    shape.d()
                )));
            }
            if meta.languages.len() != self.rows.languages.len() || meta.voices.len() != self.rows.voices.len() {
                return Err(refuse("its languages or voices are not the ones the run mapped to rows".to_string()));
            }
            let most = rows_needed(&self.sp, &self.rows, shard).map_err(refuse)?;
            if most > shape.block as usize {
                return Err(refuse(format!("a record takes {most} Talker rows, the build holds {}", shape.block)));
            }
            if let Some(r) = (0..shard.len()).find(|&r| shard.frames(r) > shape.frames as usize / self.batch) {
                return Err(refuse(format!("record {r} holds {} frames, the build takes at most {} per record", shard.frames(r), shape.frames as usize / self.batch)));
            }
        }
        Ok(())
    }

    /// `draws_done` counts records, as the shared loop counts them (steps x
    /// accumulation x its `batch_size`, which a run sets to this objective's batch).
    fn begin(&mut self, seed: u64, draws_done: u64) {
        self.seed = seed;
        self.drawn = draws_done;
        self.order = None;
    }

    fn micro_step(&mut self, model: &SpeechSftModel, _rng: &mut Rng) -> f32 {
        let t0 = Instant::now();
        let load = gpu_core::markers::range(c"speech_sft.load");
        let examples = self.next_batch();
        model.load(examples).unwrap_or_else(|e| panic!("speech sft: {e}"));
        drop(load);
        let t1 = Instant::now();
        let loss = model::Model::forward(model);
        let t2 = Instant::now();
        model::Model::backward(model);
        let t3 = Instant::now();
        let stats = model.stats();
        self.step.add(&stats);
        self.work.prepare_secs += (t1 - t0).as_secs_f64();
        self.work.forward_secs += (t2 - t1).as_secs_f64();
        self.work.backward_secs += (t3 - t2).as_secs_f64();
        self.work.useful_tokens += u64::from(stats.frames);
        self.work.computed_tokens += u64::from(model.computed_rows());
        self.work.supervised_positions += u64::from(stats.talker_count + stats.cp_count);
        loss
    }

    fn micro_steps(&mut self, model: &SpeechSftModel, rng: &mut Rng, k: u32) -> f32 {
        self.step = ForwardStats::default();
        self.work = StepWork::default();
        (0..k).map(|_| self.micro_step(model, rng)).sum()
    }

    fn step_work(&mut self) -> Option<StepWork> {
        Some(self.work)
    }

    fn loss_probe(&mut self, model: &SpeechSftModel, _rng: &mut Rng) -> f32 {
        let examples = self.next_batch();
        model.load(examples).unwrap_or_else(|e| panic!("speech sft: {e}"));
        model.score().loss(self.mtp_weight)
    }

    fn eval(&mut self, model: &SpeechSftModel, _rng: &mut Rng, batches: u32) -> Option<f32> {
        let held = self.held_out?;
        let stats = self.score(model, held, batches);
        self.held = Some(stats);
        Some(stats.loss(self.mtp_weight))
    }

    /// The step's Talker and code-predictor cross-entropy, end-of-speech
    /// accuracy and frames; after an evaluation, the held-out ones beside them.
    fn metrics(&self) -> Vec<(&'static str, f32)> {
        let s = &self.step;
        let mut out = vec![
            ("talker_ce", mean(s.talker_sum, s.talker_count)),
            ("code_predictor_ce", mean(s.cp_sum, s.cp_count)),
            ("eos_accuracy", ratio(s.eos_hits, s.eos_count)),
            ("frames", s.frames as f32),
        ];
        if let Some(h) = &self.held {
            out.push(("held_out_talker_ce", mean(h.talker_sum, h.talker_count)));
            out.push(("held_out_code_predictor_ce", mean(h.cp_sum, h.cp_count)));
            out.push(("held_out_eos_accuracy", ratio(h.eos_hits, h.eos_count)));
        }
        out
    }

    fn step_log(&self) -> Option<String> {
        let s = &self.step;
        Some(format!("talker_ce {:.4} code_predictor_ce {:.4} eos_acc {:.3} frames {}", mean(s.talker_sum, s.talker_count), mean(s.cp_sum, s.cp_count), ratio(s.eos_hits, s.eos_count), s.frames))
    }
}
