// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A recorded dispatch tape: run a pass's host-side building ONCE, replay it for
//! the price of one submission.
//!
//! Swedish Embedded AB implements low-latency LLM decode on GPUs for its clients.
//! If your team needs expertise in taking the host out of the critical path of a
//! token loop - recording a step once and replaying it - then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! A decode step of a hybrid MoE model is ~1600 dispatches. Building them is not
//! free: every `Gpu::step` resolves a kernel, builds a dispatch record and every
//! shared builder formats weight names and picks tiers, so a step costs tens of
//! milliseconds of host time - more than the card spends executing it. Nothing in
//! that work depends on the token: the kernels, the buffers they bind and their
//! parameters are the same every step, and what changes (the token ids, each
//! sequence's position, its KV length) lives in small buffers the kernels READ.
//!
//! [`Gpu::begin_tape`](crate::Gpu::begin_tape) / [`Gpu::end_tape`](crate::Gpu::end_tape)
//! capture exactly that: while a tape is open, `submit` appends to it instead of
//! launching, so the model's ordinary step-building code - unchanged - produces a
//! [`Tape`]. [`Gpu::replay_tape`](crate::Gpu::replay_tape) then submits the whole
//! recording as ONE submission (the CUDA backend turns a repeated one into a graph
//! replay). The caller's side of the contract:
//!
//! * every buffer the tape binds must be one the caller keeps writing to (the
//!   tape holds them alive, so none can be freed or recycled behind it);
//! * anything that varies between replays must reach the kernels through those
//!   buffers, not through a dispatch parameter or thread count baked in at
//!   recording time;
//! * a readback or a drain inside the recording is refused, because the device
//!   has not run what it would read.
//!
//! Host writes ARE allowed while recording and take effect immediately - they are
//! how the initial contents of the buffers a replay will rewrite are set.

use std::sync::{Arc, OnceLock};

use backend_api::{DeviceBuffer, Program, Step};

/// One recorded `submit`: the buffers it zeroes first, then its dispatches.
#[derive(Clone)]
struct Segment {
    clears: Vec<DeviceBuffer>,
    steps: Vec<Step>,
}

/// The submissions recorded between `begin_tape` and `end_tape`, in order.
///
/// Holds every buffer its steps bind (a recorded step owns handles to them), so a
/// tape keeps its working set alive for as long as it exists - which is the point,
/// and is why a cache of tapes must be bounded.
#[derive(Clone, Default)]
pub struct Tape {
    segments: Vec<Segment>,
    /// The backend's frozen form of this recording (`None` inside when it has
    /// none), made at the first replay and shared by every clone.
    frozen: Arc<OnceLock<Option<Program>>>,
}

impl Tape {
    /// Dispatches recorded.
    pub fn len(&self) -> usize {
        self.segments.iter().map(|s| s.steps.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn push(&mut self, clears: &[&DeviceBuffer], steps: &[Step]) {
        if clears.is_empty() && steps.is_empty() {
            return;
        }
        // Consecutive submissions without a clear in between are one submission
        // as far as ordering is concerned (a stream orders them anyway), and one
        // big submission is what a backend can capture as one graph.
        if clears.is_empty() {
            if let Some(last) = self.segments.last_mut() {
                last.steps.extend_from_slice(steps);
                return;
            }
        }
        self.segments.push(Segment { clears: clears.iter().map(|b| (*b).clone()).collect(), steps: steps.to_vec() });
    }

    /// The frozen program for this tape, built by `freeze` at the first call.
    pub(crate) fn frozen(&self, freeze: impl FnOnce(&[(Vec<&DeviceBuffer>, &[Step])]) -> Option<Program>) -> Option<&Program> {
        self.frozen
            .get_or_init(|| {
                let segs: Vec<(Vec<&DeviceBuffer>, &[Step])> = self.segments().collect();
                freeze(&segs)
            })
            .as_ref()
    }

    /// The submissions to make, in order, as `(clears, steps)` pairs.
    pub(crate) fn segments(&self) -> impl Iterator<Item = (Vec<&DeviceBuffer>, &[Step])> {
        self.segments.iter().map(|s| (s.clears.iter().collect(), s.steps.as_slice()))
    }
}
