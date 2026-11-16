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
//!
//! ## One working set for every recording of a step
//!
//! A tape keeps the scratch its recording allocated alive, and an engine keeps a recording per batch size, so a cache
//! of eight tapes held eight steps' working sets although two of them never run at once. A [`TapeScratch`] is one
//! set of scratch buffers that every recording made against it ([`Gpu::record_tape_on`](crate::Gpu::record_tape_on))
//! binds: while such a recording is open on a thread, every `Gpu::storage` call on that device takes the next slot
//! of the set in call order (a slot at least as large as the request; a larger request replaces the slot), so the
//! device holds the widest recorded step's scratch once. The caller's side of that contract, beyond a tape's own:
//!
//! * no two recordings of one [`TapeScratch`] run concurrently, and nothing reads a recording's scratch after another
//!   recording of the same set has been replayed (a step's outputs are read before the next step);
//! * nothing a recording needs survives in its scratch from one replay to the next unless the recording rewrites it:
//!   a host write into a slot while recording is refused, because another recording would overwrite it. Inputs a
//!   replay reads are allocated before the recording opens;
//! * a slot replaced by a larger buffer leaves the recordings made before it bound to the old one
//!   ([`TapeScratch::replacements`]): they stay correct, but hold their own copy until dropped.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};

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

    /// Whether the backend's frozen form has been made ([`crate::Gpu::prepare_tape`] or a first replay), so a replay
    /// launches it without capturing anything.
    pub fn is_frozen(&self) -> bool {
        self.frozen.get().is_some()
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

/// One set of scratch buffers shared by every recording made against it (see the module doc). Cloning shares the set.
#[derive(Clone, Default)]
pub struct TapeScratch(Arc<Mutex<Slots>>);

#[derive(Default)]
struct Slots {
    /// `(words, buffer)` in the call order a recording asks for them.
    slots: Vec<(u64, DeviceBuffer)>,
    cursor: usize,
    replacements: u64,
}

impl TapeScratch {
    pub fn new() -> TapeScratch {
        TapeScratch::default()
    }

    /// Device bytes the set holds: what every recording made against it shares.
    pub fn held_bytes(&self) -> u64 {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).slots.iter().map(|(w, _)| w.max(&1) * 4).sum()
    }

    /// How many times a slot has been replaced by a larger buffer. A recording made while this was lower binds the
    /// replaced buffer of its own: correct, but held beside the set until the recording is dropped, so a cache drops
    /// such recordings (they are recorded again against the grown set on next use).
    pub fn replacements(&self) -> u64 {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).replacements
    }
}

thread_local! {
    /// The set this thread's open recording takes its scratch from, and the device it is on.
    static ROUTED: RefCell<Option<(memauth::Device, TapeScratch)>> = const { RefCell::new(None) };
}

/// While alive, `Gpu::storage` calls on this thread for `device` take slots of the set. Ends on drop.
pub(crate) struct Route;

impl Route {
    pub(crate) fn open(device: memauth::Device, scratch: &TapeScratch) -> Route {
        ROUTED.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.is_none(), "gpu_core::tape: a recording against a TapeScratch is already open on this thread");
            scratch.0.lock().unwrap_or_else(|e| e.into_inner()).cursor = 0;
            *r = Some((device, scratch.clone()));
        });
        Route
    }
}

impl Drop for Route {
    fn drop(&mut self) {
        ROUTED.with(|r| *r.borrow_mut() = None);
    }
}

/// The next slot of the routed set for a request of `words` on `device`, made by `fresh` when the slot is missing or
/// too small; `None` when no recording on this thread routes this device.
pub(crate) fn routed_storage(device: memauth::Device, words: u64, fresh: impl FnOnce(u64) -> DeviceBuffer) -> Option<DeviceBuffer> {
    let scratch = ROUTED.with(|r| r.borrow().as_ref().filter(|(d, _)| *d == device).map(|(_, s)| s.clone()))?;
    let mut g = scratch.0.lock().unwrap_or_else(|e| e.into_inner());
    let i = g.cursor;
    g.cursor += 1;
    match g.slots.get(i) {
        Some((have, buf)) if *have >= words => Some(buf.clone()),
        Some(&(have, _)) => {
            // Grown to the larger of the two, so the recordings that asked for less at this slot still fit it.
            let words = words.max(have);
            let buf = fresh(words);
            g.slots[i] = (words, buf.clone());
            g.replacements += 1;
            Some(buf)
        }
        None => {
            let buf = fresh(words);
            g.slots.push((words, buf.clone()));
            Some(buf)
        }
    }
}

/// Refuse a host write into a slot of the set a recording on this thread is routed to (see the module doc).
pub(crate) fn refuse_write_into_routed(buf: &DeviceBuffer) {
    ROUTED.with(|r| {
        if let Some((_, scratch)) = r.borrow().as_ref() {
            let g = scratch.0.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                !g.slots.iter().any(|(_, b)| b.alloc_id() == buf.alloc_id()),
                "gpu_core::tape: a host write into shared recording scratch: another recording of the set would overwrite it; allocate a replay's inputs before the recording opens"
            );
        }
    });
}
