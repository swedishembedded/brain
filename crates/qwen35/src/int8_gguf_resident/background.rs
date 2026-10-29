// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models
// that keep every conversation generating while new prompts are read in. If
// your team needs expertise in GPU stream scheduling for LLM serving, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A prompt read in on the NORMAL lane while decode steps run on the urgent one.
//!
//! A prefill round of this model is long (a 256-row round is over a second on a
//! P40) and a decode step is short, so a scheduler that runs them in turn on one
//! stream stalls every running sequence for the whole round. A [`PrefillTask`]
//! carries one sequence's slice of prompt through the stages round by round on
//! each card's normal stream, and [`Qwen35GgufInstance::pump_prefill`] advances
//! it without ever waiting for a card that is still busy - the decode steps in
//! between are issued in `gpu_core::lane::urgent`, whose stream the device runs
//! ahead of this one, so they never queue behind a round.
//!
//! The rounds are the ones [`Qwen35GgufInstance::prefill_hidden`] runs, on the
//! same working copy of the recurrent state, so the result is that prefill's,
//! bit for bit. What changes is only when each round is submitted and read:
//! stage `s + 1` takes round `k` as soon as stage `s` has handed it over, so
//! while the stages run one round apart every card stays busy, and a stage's
//! round is submitted only once that stage has finished its previous one (the
//! prefill scratch arena's contract).
//!
//! One task at a time: the working copy is one per stage. The engine starts the
//! next prompt when this one is done.

use std::collections::VecDeque;
use std::time::Duration;

use gpu_core::DeviceBuffer;

use super::{Qwen35GgufInstance, MODEL};
use crate::adapter_serving as adapters;
use crate::model::{ChunkRound, StageRows};

/// Host time one non-waiting [`Qwen35GgufInstance::pump_prefill`] spends recording a stage's round.
///
/// A round is recorded layer by layer and the recording is paced by the device once its launch queue is full: a
/// 960-row round of the 27B is over a thousand launches and a few hundred milliseconds of device work, and recording
/// all of it in one call held the scheduler for as long as the device took to drain the queue enough to take the
/// rest, which no decode step issued meanwhile could shorten (knowledge 374). Bounded, the scheduler is back at its
/// decode step after this long and the rest of the round follows on its next pumps; the queue still holds many
/// layers, so the device stays busy.
// perf-number: a layer of that round records in about 1.5 ms when the queue has room; 16 ms is about ten layers
const PUMP_RECORD_BUDGET: Duration = Duration::from_millis(16);
/// What the working copy of the recurrent state holds when a task starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskStart {
    /// Nothing of this prompt yet: cleared before the first round.
    Fresh,
    /// The state after the task's first token, already in place - a seeded
    /// cached prefix, or the previous slice of the same prompt.
    Continue,
}

/// A stage's round in flight.
enum StageRound {
    /// Layers still being recorded; the round's input and tables are in place.
    Recording(ChunkRound),
    /// Every layer recorded; the buffer the stage will leave its output in, which the device may still be filling.
    Issued(DeviceBuffer),
}

/// One sequence's slice of prompt on its way through the stages.
pub(crate) struct PrefillTask {
    slot: u32,
    tokens: Vec<u32>,
    pos0: u32,
    rows: usize,
    rounds: usize,
    start: TaskStart,
    collect: bool,
    /// Per stage: the next round it takes.
    next: Vec<usize>,
    /// Per stage: the round it is running, being recorded or issued whole.
    running: Vec<Option<StageRound>>,
    /// Per stage: the outputs of the stage before it, in round order, waiting to be taken.
    /// Device buffers: the boundary rows stay on the card that made them until the next stage copies them.
    handed: Vec<VecDeque<DeviceBuffer>>,
    /// The last row's `[d_model]` hidden state, once the last stage has run the last round.
    last: Vec<f32>,
    /// Every row's, in order, when collected (the MTP head's input).
    all: Vec<f32>,
    /// Per stage: whether its share of the slice is done and its state stored in the slot's pool row.
    stored: Vec<bool>,
}

impl PrefillTask {
    /// The slot this task prefills.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    fn finished(&self) -> bool {
        self.stored.iter().all(|&s| s)
    }

    /// The rows of round `k`.
    fn round(&self, k: usize) -> &[u32] {
        let end = ((k + 1) * self.rows).min(self.tokens.len());
        &self.tokens[k * self.rows..end]
    }
}

/// What a finished task hands back: the last row's hidden state and, when
/// collected, every row's.
pub(crate) struct PrefillDone {
    pub last: Vec<f32>,
    pub all: Vec<f32>,
}

impl Qwen35GgufInstance {
    /// Whether prompts can be read in on the normal lane while decode runs on the
    /// urgent one: every stage's device can prioritise, rounds are chunked, and no
    /// per-round side channel is armed (hidden taps record one sequence's rows in
    /// order; an adapter bank's slot table is written per forward).
    pub(crate) fn background_prefill_available(&self) -> bool {
        self.taps.borrow().is_none()
            && self.shards.iter().all(|s| s.qwen35.gpu.has_urgent_lane() && s.qwen35.chunked_prefill_is_profitable())
            && adapters::bytes(self.stage_models()) == 0
    }

    /// Let the decode lane and the prefill lane write the KV and recurrent-state
    /// pools at once: each touches only its own sequences' rows.
    pub(crate) fn share_pools_across_lanes(&self) {
        for s in &self.shards {
            let g = &s.qwen35.gpu;
            for layer in &s.caches.gqa_kv {
                layer.share_rows(g);
            }
            for b in s.caches.pool_state.iter().chain(&s.caches.pool_hist) {
                g.share_rows(b);
            }
        }
    }

    /// A task that prefills `prompt[start..end]` into slot `slot`. Nothing is
    /// submitted yet; [`Self::pump_prefill`] does that.
    pub(crate) fn prefill_task(&self, prompt: &[u32], start: usize, end: usize, slot: u32, from: TaskStart, collect: bool) -> PrefillTask {
        assert!(start < end && end <= prompt.len(), "{MODEL}: a prefill task needs at least one new token");
        let rows = self.prefill_round_rows() as usize;
        let tokens = prompt[start..end].to_vec();
        let rounds = tokens.len().div_ceil(rows);
        let stages = self.shards.len();
        PrefillTask {
            slot,
            tokens,
            pos0: start as u32,
            rows,
            rounds,
            start: from,
            collect,
            next: vec![0; stages],
            running: (0..stages).map(|_| None).collect(),
            handed: vec![VecDeque::new(); stages],
            last: Vec::new(),
            all: Vec::new(),
            stored: vec![false; stages],
        }
    }

    /// Advance `task`: record the layers of each stage's round, read back each round that has finished and begin
    /// the next one wherever the input is there. Without `wait` it never waits for a card (one pass, recording at
    /// most [`PUMP_RECORD_BUDGET`] per stage); with `wait` it runs the task to its end. `Some` once the task is done.
    pub(crate) fn pump_prefill(&self, task: &mut PrefillTask, wait: bool) -> Result<Option<PrefillDone>, String> {
        let d = self.cfg.d_model as usize;
        let budget = (!wait).then_some(PUMP_RECORD_BUDGET);
        loop {
            // Later stages first: their submissions go out before stage 0's,
            // which in a waiting pump is followed by a blocking read.
            for s in (0..self.shards.len()).rev() {
                let stage = &self.shards[s];
                if matches!(task.running[s], Some(StageRound::Issued(_))) && (wait || stage.qwen35.lane_idle()) {
                    let k = task.next[s] - 1;
                    let n = task.round(k).len();
                    let Some(StageRound::Issued(out)) = task.running[s].take() else { unreachable!("checked above") };
                    if s + 1 < self.shards.len() {
                        // The next stage copies the rows device to device; the host neither reads nor waits for them.
                        task.handed[s + 1].push_back(out);
                    } else {
                        let out = stage.qwen35.gpu.read(&out, n * d);
                        task.last = out[(n - 1) * d..].to_vec();
                        if task.collect {
                            task.all.extend_from_slice(&out);
                        }
                    }
                    if task.next[s] == task.rounds {
                        stage.caches.store_work(&stage.qwen35.gpu, &self.cfg, task.slot);
                        task.stored[s] = true;
                    }
                }
                if task.running[s].is_none() && task.next[s] < task.rounds {
                    let k = task.next[s];
                    let embedded;
                    let handed;
                    let input = if s == 0 {
                        embedded = self.embed_rows(task.round(k))?;
                        StageRows::Host(&embedded)
                    } else if let Some(rows) = task.handed[s].pop_front() {
                        handed = rows;
                        StageRows::device(&self.shards[s - 1].qwen35.gpu, &handed, task.round(k).len() * d)
                    } else {
                        continue;
                    };
                    if k == 0 && task.start == TaskStart::Fresh {
                        stage.caches.reset_work(&stage.qwen35.gpu);
                    }
                    let pos = task.pos0 + (k * task.rows) as u32;
                    let caches = stage.caches.view(task.slot);
                    let round = stage.qwen35.begin_prefill_chunk(task.round(k), pos, &caches, Some(input));
                    task.running[s] = Some(StageRound::Recording(round));
                    task.next[s] = k + 1;
                }
                if let Some(StageRound::Recording(round)) = task.running[s].as_mut() {
                    let caches = stage.caches.view(task.slot);
                    if let Some(out) = stage.qwen35.advance_prefill_chunk(round, &caches, budget) {
                        task.running[s] = Some(StageRound::Issued(out));
                    }
                }
            }
            if task.finished() {
                return Ok(Some(PrefillDone { last: std::mem::take(&mut task.last), all: std::mem::take(&mut task.all) }));
            }
            if !wait {
                return Ok(None);
            }
        }
    }

    /// Give a task up (its sequence was released): wait for whatever of it is
    /// still running, so the next task's rounds and the released rows are not
    /// written under it.
    pub(crate) fn abandon_prefill(&self, task: PrefillTask) {
        for (s, running) in task.running.into_iter().enumerate() {
            match running {
                Some(StageRound::Recording(round)) => self.shards[s].qwen35.abandon_prefill_chunk(round),
                Some(StageRound::Issued(_)) => self.shards[s].qwen35.gpu.poll_wait(),
                None => {}
            }
        }
    }
}
