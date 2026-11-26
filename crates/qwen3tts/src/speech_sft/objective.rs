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

use super::distill::SpeechTeacher;
use super::model::{Example, ForwardStats, LossWeights, SpeechSftModel};
use crate::prompt::{PromptPlan, PromptSpec, SpeakerSlot, TextLayout, TtsSpecials};
use crate::shard::Shard;

/// Which codec row each of a shard's languages and voices is.
#[derive(Clone, Debug, PartialEq)]
pub struct RowMap {
    /// Per [`crate::shard::ShardMeta::languages`] entry: its language row, or
    /// `None` for `"auto"` (the no-language prefix).
    pub languages: Vec<Option<u32>>,
    /// Per [`crate::shard::ShardMeta::voices`] entry: its speaker row.
    pub voices: Vec<u32>,
    /// The checkpoint's text id of each shard text id, when the shards were tokenised by another tokenizer than the
    /// checkpoint's (a trimmed vocabulary renumbers the tokens it keeps); [`RowMap::ABSENT`] where the checkpoint has
    /// no such token. `None`: the shards' ids are the checkpoint's.
    pub text: Option<Vec<u32>>,
    /// How a record's x-vector, made by the speaker encoder the shards were made with, reads in the checkpoint's
    /// narrower stream ([`crate::speaker_projection`]); `None`: as stored.
    pub xvector: Option<std::sync::Arc<crate::speaker_projection::SpeakerProjection>>,
}

impl RowMap {
    /// A shard text id with no checkpoint token in [`Self::text`].
    pub const ABSENT: u32 = u32::MAX;

    /// The checkpoint's id of shard text id `id`.
    pub fn text_id(&self, id: u32) -> Result<u32, String> {
        match &self.text {
            None => Ok(id),
            Some(map) => match map.get(id as usize) {
                Some(&t) if t != Self::ABSENT => Ok(t),
                _ => Err(format!("text token {id} of the shard's tokenizer is no token of the checkpoint's")),
            },
        }
    }
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
    let ids: std::borrow::Cow<'_, [u32]> = match &rows.text {
        None => std::borrow::Cow::Borrowed(rec.text_ids),
        Some(_) => std::borrow::Cow::Owned(rec.text_ids.iter().map(|&id| rows.text_id(id)).collect::<Result<_, _>>().map_err(|e| named(&e))?),
    };
    let (role_ids, text_ids) = crate::pipeline::split_input_ids(&ids).map_err(|e| named(&e))?;
    let speaker = match rec.voice {
        Some(v) => SpeakerSlot::Codec(*rows.voices.get(v as usize).ok_or_else(|| named(&format!("voice {v} has no row")))?),
        None => SpeakerSlot::XVector,
    };
    let language_id = *rows.languages.get(rec.language as usize).ok_or_else(|| named(&format!("language {} has no row", rec.language)))?;
    let plan = PromptPlan::build(sp, &PromptSpec { role_ids: &role_ids, text_ids: &text_ids, instruct_ids: &[], speaker, language_id, layout }).map_err(|e| named(&e))?;
    if rec.codes.is_empty() {
        return Err(named("no frames"));
    }
    let source = rec.xvector.to_vec();
    let xvector = match &rows.xvector {
        Some(p) => p.apply(&source).map_err(|e| named(&e))?,
        None => source.clone(),
    };
    Ok(Example { plan, xvector, source_xvector: source, codes: rec.codes.iter().map(|&c| u32::from(c)).collect(), complete: rec.complete })
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
/// The stream positions of one rank's micro-batch of `batch` records, starting at stream position `first`, under
/// `(rank, world)`: the positions `n` with `n % world == rank` among the next `batch x world`.
fn batch_positions(first: u64, batch: usize, (rank, world): (usize, usize)) -> impl Iterator<Item = u64> {
    (0..batch as u64).map(move |i| first + i * world as u64 + rank as u64)
}

/// The stream positions of one rank's share of global micro-batch `micro_batch` (`batch x world` records), with the
/// records of every `window` consecutive global micro-batches regrouped by length, a record at position `p` holding
/// `frames(p)` frames. A micro-batch's rows are computed to its longest record, so a batch of random lengths is mostly
/// padding; sorting a window's records by length and cutting it into micro-batches makes each one about one length.
/// The window's groups are taken in an order drawn from `seed` and the window's index, so a run does not train its
/// shortest records first, and each group is dealt to the ranks in turn, so every rank's share is about as long as the
/// others' and no rank waits long at the step's reduction. The window's positions are still exactly the stream's,
/// each once; a window of 1 is [`batch_positions`]'s plain stream.
fn bucketed_positions(micro_batch: u64, batch: usize, (rank, world): (usize, usize), window: usize, seed: u64, frames: impl Fn(u64) -> usize) -> Vec<u64> {
    let global = batch * world;
    if window <= 1 {
        return batch_positions(micro_batch * global as u64, batch, (rank, world)).collect();
    }
    let index = micro_batch / window as u64;
    let first = index * (window * global) as u64;
    let mut records: Vec<(usize, u64)> = (first..first + (window * global) as u64).map(|pos| (frames(pos), pos)).collect();
    // By length, equal lengths in stream order: every rank computes the same groups.
    records.sort_unstable();
    let group = model::train::epoch_order(seed, window, index)[(micro_batch % window as u64) as usize];
    records[group * global..(group + 1) * global].iter().skip(rank).step_by(world).map(|&(_, pos)| pos).collect()
}

pub struct SpeechObjective<'a> {
    train: &'a Shard,
    held_out: Option<&'a Shard>,
    rows: RowMap,
    sp: TtsSpecials,
    batch: usize,
    streaming_fraction: f32,
    weights: LossWeights,
    /// The frozen teacher a distilling run's every batch is taught by ([`Self::with_teacher`]).
    teacher: Option<SpeechTeacher>,
    seed: u64,
    /// Records drawn so far in the whole run: the position of the next one in the stream.
    drawn: u64,
    /// `(rank, world)`: this rank trains its share of each global micro-batch of the stream ([`Objective::shard`]).
    shard: (usize, usize),
    /// Global micro-batches whose records are regrouped by length ([`bucketed_positions`]); 1 draws the plain stream.
    bucket: usize,
    order: Option<(u64, Vec<usize>)>,
    /// The records of the bucketing window being drawn: its index and its records in stream order.
    window: Option<(u64, Vec<usize>)>,
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
        Ok(SpeechObjective { train, held_out, rows, sp, batch, streaming_fraction, weights: LossWeights::sft(mtp_weight), teacher: None, seed: data::rng::random_seed(), drawn: 0, shard: (0, 1), bucket: 1, order: None, window: None, step: ForwardStats::default(), held: None, work: StepWork::default() })
    }

    /// Regroup the records of every `steps` consecutive micro-batches by length ([`bucketed_positions`]); 1 (the
    /// default) draws the plain stream.
    pub fn with_length_buckets(mut self, steps: usize) -> Self {
        self.bucket = steps.max(1);
        self
    }

    /// Weigh the loss's terms by `weights` instead of the ground-truth objective alone (the model's own,
    /// [`SpeechSftModel::loss_weights`]).
    pub fn with_weights(mut self, weights: LossWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Have `teacher` teach every batch the model loads, training and held out: the run distils
    /// ([`super::distill`]), and the held-out score is the distillation's terms (what a kept best selects on).
    pub fn with_teacher(mut self, teacher: SpeechTeacher) -> Self {
        self.teacher = Some(teacher);
        self
    }

    /// Load `examples` into `model`, and have the teacher teach them when the run distils.
    fn load(&self, model: &SpeechSftModel, examples: Vec<Example>) -> Result<(), String> {
        match &self.teacher {
            Some(teacher) => {
                model.load(examples.clone())?;
                teacher.teach(model, &examples)
            }
            None => model.load(examples),
        }
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

    /// The next micro-batch of the training stream: this rank's `batch` records of the next `batch x world` (every rank
    /// walks the whole stream, so their epoch orders stay one order, and each keeps its own records of it).
    fn next_batch(&mut self) -> Vec<Example> {
        let global = (self.batch * self.shard.1) as u64;
        let micro_batch = self.drawn / global;
        self.drawn += global;
        // The window's records, looked up once in stream order (a window may cross an epoch's end).
        let span = global * self.bucket as u64;
        let index = micro_batch / self.bucket as u64;
        if self.window.as_ref().is_none_or(|(i, _)| *i != index) {
            let records = (index * span..(index + 1) * span).map(|pos| self.record_at(pos)).collect();
            self.window = Some((index, records));
        }
        let (train, first) = (self.train, index * span);
        let records = &self.window.as_ref().expect("just set").1;
        bucketed_positions(micro_batch, self.batch, self.shard, self.bucket, self.seed, |pos| train.frames(records[(pos - first) as usize]))
            .into_iter()
            .map(|pos| {
                let layout = draw_layout(self.seed, pos, self.streaming_fraction);
                example(&self.sp, &self.rows, train, records[(pos - first) as usize], layout).unwrap_or_else(|e| panic!("speech sft: {e}"))
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
            self.load(model, examples).unwrap_or_else(|e| panic!("speech sft: held out: {e}"));
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
            // The x-vector is read only by a record in the x-vector slot (a named voice is its codec row): a shard of
            // named voices may carry another width's x-vectors (a student narrower than the speaker encoder).
            let reads_xvectors = (0..shard.len()).any(|r| shard.record(r).voice.is_none());
            let (xvector_in, xvector_out) = self.rows.xvector.as_ref().map_or((meta.xvector_dim as usize, meta.xvector_dim as usize), |p| (p.from, p.to));
            if meta.codebooks as usize != g || meta.codebook_size != shape.mtp.vocab || (reads_xvectors && (meta.xvector_dim as usize != xvector_in || xvector_out != shape.d())) {
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
    /// accumulation x its `batch_size`, which a run sets to this objective's batch): a rank's
    /// own, so the stream position is `world` times it.
    fn begin(&mut self, seed: u64, draws_done: u64) {
        self.seed = seed;
        self.drawn = draws_done * self.shard.1 as u64;
        self.order = None;
        self.window = None;
    }

    /// Split the training records across `world` ranks as the trait says: every global micro-batch of the stream is
    /// dealt to the ranks, each record to one ([`bucketed_positions`]). The held-out shard is not split.
    fn shard(&mut self, rank: usize, world: usize) -> Result<(), std::io::Error> {
        if world == 0 || rank >= world {
            return Err(std::io::Error::other(format!("rank {rank} is not one of a world of {world}")));
        }
        self.shard = (rank, world);
        self.window = None;
        Ok(())
    }

    fn micro_step(&mut self, model: &SpeechSftModel, _rng: &mut Rng) -> f32 {
        let t0 = Instant::now();
        let load = gpu_core::markers::range(c"speech_sft.load");
        let examples = self.next_batch();
        self.load(model, examples).unwrap_or_else(|e| panic!("speech sft: {e}"));
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
        self.load(model, examples).unwrap_or_else(|e| panic!("speech sft: {e}"));
        model.score().loss(&self.weights)
    }

    fn eval(&mut self, model: &SpeechSftModel, _rng: &mut Rng, batches: u32) -> Option<f32> {
        let held = self.held_out?;
        let stats = self.score(model, held, batches);
        self.held = Some(stats);
        // A distilling run selects on how close it came to the teacher.
        Some(match &self.weights.kd {
            Some(kd) => stats.kd.loss(kd) as f32,
            None => stats.loss(&self.weights),
        })
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
        let kd = self.weights.kd.is_some();
        if kd {
            out.extend([("talker_kl", s.kd.talker_kl() as f32), ("code_predictor_kl", s.kd.code_predictor_kl() as f32), ("hidden_mse", s.kd.hidden_mse() as f32)]);
        }
        if let Some(h) = &self.held {
            out.push(("held_out_talker_ce", mean(h.talker_sum, h.talker_count)));
            out.push(("held_out_code_predictor_ce", mean(h.cp_sum, h.cp_count)));
            out.push(("held_out_eos_accuracy", ratio(h.eos_hits, h.eos_count)));
            if kd {
                out.extend([("held_out_talker_kl", h.kd.talker_kl() as f32), ("held_out_code_predictor_kl", h.kd.code_predictor_kl() as f32), ("held_out_hidden_mse", h.kd.hidden_mse() as f32)]);
            }
        }
        out
    }

    fn step_log(&self) -> Option<String> {
        let s = &self.step;
        let base = format!("talker_ce {:.4} code_predictor_ce {:.4} eos_acc {:.3} frames {}", mean(s.talker_sum, s.talker_count), mean(s.cp_sum, s.cp_count), ratio(s.eos_hits, s.eos_count), s.frames);
        Some(match self.weights.kd {
            Some(_) => format!("{base} talker_kl {:.4} code_predictor_kl {:.4} hidden_mse {:.4}", s.kd.talker_kl(), s.kd.code_predictor_kl(), s.kd.hidden_mse()),
            None => base,
        })
    }
}

#[cfg(test)]
mod shard_tests {
    use super::{batch_positions, bucketed_positions};

    /// Record lengths as a corpus has them: spread over 1.5-30 s, with ties.
    fn frames(pos: u64) -> usize {
        (pos.wrapping_mul(2_654_435_761) % 360) as usize + 19
    }

    /// Over several micro-steps, the ranks of a world draw disjoint positions that together are the whole stream:
    /// nothing is trained twice and nothing is skipped, bucketed or not.
    #[test]
    fn the_ranks_of_a_world_partition_the_stream() {
        for world in 1..=4usize {
            let batch = 3;
            let mut seen = Vec::new();
            let mut first = 0u64;
            for _step in 0..5 {
                for rank in 0..world {
                    let mine: Vec<u64> = batch_positions(first, batch, (rank, world)).collect();
                    assert!(mine.iter().all(|p| p % world as u64 == rank as u64), "world {world} rank {rank}: {mine:?}");
                    seen.extend(mine);
                }
                first += (batch * world) as u64;
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..first).collect::<Vec<_>>(), "world {world}");
        }
        for (world, window) in [(1usize, 4usize), (2, 3), (3, 5), (4, 2)] {
            let batch = 5;
            let micro_batches = 3 * window as u64;
            let mut seen = Vec::new();
            for m in 0..micro_batches {
                for rank in 0..world {
                    let mine = bucketed_positions(m, batch, (rank, world), window, 7, frames);
                    assert_eq!(mine.len(), batch, "world {world} window {window} micro-batch {m} rank {rank}");
                    seen.extend(mine);
                }
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..micro_batches * (batch * world) as u64).collect::<Vec<_>>(), "world {world} window {window}");
        }
    }

    /// A window of one micro-batch is the unbucketed stream, position for position.
    #[test]
    fn a_window_of_one_is_the_plain_stream() {
        for m in 0..4u64 {
            for rank in 0..3 {
                let plain: Vec<u64> = batch_positions(m * 12, 4, (rank, 3)).collect();
                assert_eq!(bucketed_positions(m, 4, (rank, 3), 1, 7, frames), plain);
            }
        }
    }

    /// The point of bucketing: a micro-batch's rows are padded to its longest record, and a bucketed micro-batch's
    /// records are of about one length, so far fewer of the rows it computes are padding; and every rank's share of a
    /// step is about as long as the others', so no rank waits long for another at the step's reduction.
    #[test]
    fn bucketed_batches_pad_far_less_and_their_ranks_cost_alike() {
        let (batch, world) = (32usize, 2usize);
        // Computed over useful rows, summed over every rank's micro-batch of the first 64 global micro-batches.
        let padding = |window: usize| {
            let (mut computed, mut useful) = (0usize, 0usize);
            for m in 0..64u64 {
                for rank in 0..world {
                    let lens: Vec<usize> = bucketed_positions(m, batch, (rank, world), window, 7, frames).iter().map(|&p| frames(p)).collect();
                    computed += lens.len() * lens.iter().max().unwrap();
                    useful += lens.iter().sum::<usize>();
                }
            }
            computed as f64 / useful as f64
        };
        let (plain, bucketed) = (padding(1), padding(16));
        assert!(plain > 1.6, "random batches of this spread pad to their longest record: {plain:.2}");
        assert!(bucketed < 1.15, "a window of 16 micro-batches cuts the padding to a few percent: {bucketed:.2} (plain {plain:.2})");
        for m in 0..64u64 {
            let longest: Vec<usize> = (0..world).map(|rank| bucketed_positions(m, batch, (rank, world), 16, 7, frames).iter().map(|&p| frames(p)).max().unwrap()).collect();
            let (lo, hi) = (*longest.iter().min().unwrap(), *longest.iter().max().unwrap());
            assert!(hi as f64 <= lo as f64 * 1.1 + 2.0, "micro-batch {m}: the ranks' longest records {longest:?}");
        }
    }

    /// The window's groups are taken in a seeded order, not shortest first: a run does not train its first steps on
    /// its shortest records and its last on its longest.
    #[test]
    fn the_groups_of_a_window_come_in_a_seeded_order() {
        let longest = |seed: u64| (0..16u64).map(|m| bucketed_positions(m, 8, (0, 1), 16, seed, frames).iter().map(|&p| frames(p)).max().unwrap()).collect::<Vec<_>>();
        let a = longest(7);
        assert_eq!(a, longest(7), "the same seed draws the same order");
        assert_ne!(a, longest(8), "another seed draws another order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_ne!(a, sorted, "the groups are not taken in length order");
    }
}
