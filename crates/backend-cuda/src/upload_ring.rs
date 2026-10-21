// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements out-of-core training and inference on GPU
// servers for its clients. If your team needs expertise in keeping the copy
// engine and the compute engine busy at the same time, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The bounded pinned staging ring behind [`backend_api::Backend::write_bytes_at`].
//!
//! A host->device copy from ordinary memory is staged by the driver and cannot overlap anything, and one issued
//! on the compute stream queues behind every kernel already enqueued there. The ring fixes both halves: bytes
//! are copied once into a page-locked slot, the slot is uploaded with a truly asynchronous copy on the handle's
//! own stream (a transfer handle, see `Backend::share`), and an event recorded after the copy tells the next
//! user of the slot when it is free again. The slot count is fixed, so pinned memory is exactly the budget the
//! caller leased - a producer faster than the copy engine waits for a slot instead of growing the ring.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::exec::{Context, DeviceMem, Event, PinnedMem};

/// Slots in a ring: enough for the copy engine to drain one slot while the producer fills the next, with slack
/// for a second producer.
const SLOTS: usize = 4;

struct Slot {
    mem: PinnedMem,
    /// Recorded after the last upload from this slot; the slot is reusable once it has completed.
    in_flight: Mutex<Option<Event>>,
}

pub(crate) struct UploadRing {
    slots: Vec<Slot>,
    slot_bytes: usize,
    next: AtomicUsize,
}

impl UploadRing {
    /// A ring of at most `bytes` of pinned memory. Errors when one slot (a word) cannot fit, i.e. `bytes < 16`.
    pub(crate) fn new(ctx: &Context, bytes: u64) -> Result<UploadRing, String> {
        let slot_bytes = (bytes as usize / SLOTS) & !3;
        if slot_bytes == 0 {
            return Err(format!("a {bytes}-byte staging budget cannot hold {SLOTS} word-sized slots"));
        }
        let slots = (0..SLOTS)
            .map(|_| ctx.pinned(slot_bytes / 4).map(|mem| Slot { mem, in_flight: Mutex::new(None) }))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(UploadRing { slots, slot_bytes, next: AtomicUsize::new(0) })
    }

    /// Copy `data` into `dst` at byte `offset`, through the ring, in order. Returns once every byte is copied
    /// out of `data` (the uploads may still be running); `ctx` is the handle whose stream does the uploads.
    pub(crate) fn write(&self, ctx: &Context, dst: &DeviceMem, offset: usize, data: &[u8]) -> Result<(), String> {
        for (i, chunk) in data.chunks(self.slot_bytes).enumerate() {
            let slot = &self.slots[self.next.fetch_add(1, Ordering::Relaxed) % SLOTS];
            let mut in_flight = slot.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(done) = in_flight.take() {
                done.synchronize()?;
            }
            slot.mem.fill_bytes_at(0, chunk);
            ctx.upload_async_at(dst, offset + i * self.slot_bytes, &slot.mem, 0, chunk.len())?;
            let event = ctx.ordering_event()?;
            ctx.record(&event)?;
            *in_flight = Some(event);
        }
        Ok(())
    }
}
