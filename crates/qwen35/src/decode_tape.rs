// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-latency decode of
// fixed-state language models on a single GPU for its clients. If your team
// needs expertise in taking the host out of a token loop - recording a step
// once and replaying it as one submission - then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The recorded decode step of a [`SlotPool`](crate::decode_state::SlotPool).
//!
//! A sliding-window config decodes on fixed-size state, so a step is the same list of dispatches at every position
//! and for every mix of slots: what differs between steps (token ids, positions, ring rows, live counts, rope rows,
//! slot rows, the anchor memory's read and push plans) is DATA. [`DecodeInputs`] holds each of those in a buffer of its
//! own; a recording of the step binds them once ([`gpu_core::tape`]) and the engine rewrites them with small host
//! writes before every replay, which the CUDA backend runs as one graph launch. The bank commits a step may owe are
//! not part of the recording (they are the one thing that depends on the positions, and a commit reads only the pool's
//! pending block): the caller issues them after the replay.
//!
//! A recording is bound to a batch SIZE (and the decode-fusion setting it was built under), not to which slots are in
//! the batch, so a batch that changes composition replays the same tape. [`SlotTapes`] keeps the few most recently
//! used.

use gpu_core::tape::Tape;
use gpu_core::DeviceBuffer;
use model::anchor_memory::{AnchorMemoryShape, PushPlan, ReadPlan};

/// Recordings kept per pool. Each holds the working set of one step (about the activations of one pass over the
/// layers), so the count is bounded; a batch size not among them is recorded again, which costs about one step built
/// the ordinary way.
pub const MAX_SLOT_TAPES: usize = 4;

/// The buffers one decode step reads for its per-step inputs, for a batch of `bsz` rows.
pub(crate) struct DecodeInputs {
    pub bsz: u32,
    /// Block-table width of the step's paged attention (1 for a ring: its one block is the slot).
    pub max_bt: u32,
    /// `[bsz]` token ids.
    pub tok: DeviceBuffer,
    /// The batch's M-RoPE rows at each sequence's own position.
    pub cos: DeviceBuffer,
    pub sin: DeviceBuffer,
    /// `[bsz]` block each row's K/V is appended to, and its row inside the block.
    pub blocks: DeviceBuffer,
    pub offsets: DeviceBuffer,
    /// `[bsz]` pool row of each sequence's recurrent state.
    pub slot_rows: DeviceBuffer,
    /// `[bsz, max_bt]` block tables.
    pub block_tables: DeviceBuffer,
    /// `[bsz]` live keys of each row, this step's token included.
    pub seq_lens: DeviceBuffer,
    /// The anchor memory's plans (a config without one has `None`).
    pub memory: Option<(AnchorMemoryShape, ReadPlan, PushPlan)>,
}

/// One recorded step: the tape, the inputs it binds and the hidden block it leaves.
pub(crate) struct SlotDecodeTape {
    pub bsz: u32,
    /// The decode-fusion setting the recording was built under.
    pub fusion: bool,
    pub tape: Tape,
    pub inputs: DecodeInputs,
    pub hidden: DeviceBuffer,
}

/// Consecutive steps of one batch size before it is recorded. A serving batch grows and shrinks as sequences arrive and
/// finish, and a recording costs about a step built the ordinary way plus the allocation of its working set, so a size
/// that is seen for a step or two is not worth one: only a size that holds is.
pub const RECORD_AFTER_STEPS: u32 = 3;

/// The pool's recordings, most recently used last.
#[derive(Default)]
pub(crate) struct SlotTapes {
    tapes: Vec<SlotDecodeTape>,
    /// The batch size of the latest step and how many steps in a row had it.
    run: (u32, u32),
}

impl SlotTapes {
    /// Note a step of `bsz` rows; whether this size has now held long enough to be recorded.
    pub fn note(&mut self, bsz: u32) -> bool {
        self.run = if self.run.0 == bsz { (bsz, self.run.1.saturating_add(1)) } else { (bsz, 1) };
        self.run.1 >= RECORD_AFTER_STEPS
    }

    /// Take the recording for `(bsz, fusion)` out of the cache, if there is one.
    pub fn take(&mut self, bsz: u32, fusion: bool) -> Option<SlotDecodeTape> {
        let at = self.tapes.iter().position(|t| t.bsz == bsz && t.fusion == fusion)?;
        Some(self.tapes.remove(at))
    }

    /// Put a recording back as the most recently used, dropping the least recently used beyond [`MAX_SLOT_TAPES`].
    pub fn put(&mut self, tape: SlotDecodeTape) {
        if self.tapes.len() >= MAX_SLOT_TAPES {
            self.tapes.remove(0);
        }
        self.tapes.push(tape);
    }

    /// The most recently replayed recording.
    pub fn latest(&self) -> Option<&SlotDecodeTape> {
        self.tapes.last()
    }

    pub fn len(&self) -> usize {
        self.tapes.len()
    }
}

/// What a recorded plan leaves behind for the host: the head it ends in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HeadKind {
    /// Each row's greedy token.
    Argmax,
    /// Each row's best `n` candidates.
    TopK(u32),
}

/// The identity of a recorded paged decode step: everything its dispatch list depends on. The rows' positions, slots,
/// block tables and tokens are not in it: they are data ([`DecodeInputs`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PlanKey {
    pub rows: u32,
    /// Keys of attention capacity the step was built at ([`Qwen35::paged_step_shape`](crate::model::Qwen35)).
    pub cap: u32,
    pub head: HeadKind,
}

/// The device buffers a plan's head leaves its answer in.
pub(crate) enum PlanOut {
    Argmax(DeviceBuffer),
    TopK(model::readout::TopkOut),
}

/// One recorded step: layers and head as a single tape, its inputs and its outputs.
pub(crate) struct DecodePlan {
    pub key: PlanKey,
    pub tape: Tape,
    pub inputs: DecodeInputs,
    pub out: PlanOut,
}

/// Recordings kept per instance. A serving batch moves between a handful of row counts and, slowly, context buckets, so
/// the working set of recordings is small; each holds a step's activations (the logits block dominates), so it is bounded.
pub(crate) const MAX_PLANS: usize = 8;

/// Consecutive steps with one key before it is recorded: a key seen once or twice (a batch that grows with every
/// arrival) is cheaper built the ordinary way than recorded.
pub(crate) const RECORD_PLAN_AFTER: u32 = 3;

/// The instance's recorded steps, most recently used last.
#[derive(Default)]
pub(crate) struct DecodePlans {
    plans: Vec<DecodePlan>,
    run: Option<(PlanKey, u32)>,
}

impl DecodePlans {
    /// Note a step with `key`; the recording for it if there is one, else whether it has now held long enough to be recorded.
    pub fn lookup(&mut self, key: PlanKey) -> Result<DecodePlan, bool> {
        self.run = match self.run {
            Some((k, n)) if k == key => Some((k, n.saturating_add(1))),
            _ => Some((key, 1)),
        };
        match self.plans.iter().position(|p| p.key == key) {
            Some(at) => Ok(self.plans.remove(at)),
            None => Err(self.run.is_some_and(|(_, n)| n >= RECORD_PLAN_AFTER)),
        }
    }

    /// Put a recording back as the most recently used, dropping the least recently used beyond [`MAX_PLANS`].
    pub fn put(&mut self, plan: DecodePlan) {
        if self.plans.len() >= MAX_PLANS {
            self.plans.remove(0);
        }
        self.plans.push(plan);
    }

    /// Drop every recording (the buffers they bind are about to change or go).
    pub fn clear(&mut self) {
        self.plans.clear();
        self.run = None;
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }
}
