// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A replay arena for the per-iteration scratch buffers of a repeated pass.
//!
//! Swedish Embedded AB implements device-memory lifetime management for
//! inference engines for its clients. If your team needs expertise in GPU
//! allocator behaviour and host-side pipeline stalls then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! ## What this exists for
//!
//! A transformer block stack dispatches the *identical shape sequence* once
//! per block. Every temporary it needs - the normalized activation, the
//! packed int8 operands, the attention output, the FFN hidden - is therefore
//! requested at the same point in the sequence, at the same size, in every
//! block. Allocated the obvious way, each block creates that whole set fresh
//! and destroys it again; on a discrete card each create/destroy pair is a
//! driver allocation, and a stack deep enough to matter spends a large share
//! of its HOST time doing nothing else. The device sits idle for all of it.
//!
//! The arena removes the pair. It remembers, in call order, every buffer a
//! scope asked for, and the next scope hands the same buffers back. Nothing
//! is created and nothing is destroyed once the sequence has run once.
//!
//! ## Why recycling here cannot alias a live operand
//!
//! Three things carry the argument, and it is worth being exact about which
//! one covers what, because two of them are the CALLER's to hold up.
//!
//! 1. **Inside one scope, nothing is ever handed out twice.** The cursor only
//!    advances, so slot `i` is issued once per scope. Two operands of the same
//!    dispatch therefore cannot collide, whatever the caller does with its
//!    handles.
//! 2. **Across scopes, a handle the caller KEPT blocks reuse.**
//!    `DeviceBuffer::is_unique` - the arena's copy being the last one - is what
//!    the slot check asks. Read it narrowly: it is a statement about
//!    `DeviceBuffer` handles and nothing else. A recorded `Step` does NOT hold
//!    one (the wgpu backend's step is a `BindGroup`, which keeps the native
//!    buffer alive by a different path), so this test answers "does any caller
//!    still name this allocation", not "is the device finished with it". That
//!    is exactly the question the one value a block stack deliberately carries
//!    forward poses - the chained activation, block `l`'s output held by the
//!    caller as block `l+1`'s input - and answering it is what makes that case
//!    correct with no special case anywhere: the slot is re-allocated and the
//!    activation is left to its holder. Do not extend this to mean more than
//!    it says.
//! 3. **The device being finished is the CALLER's half.** Because (2) cannot
//!    see submitted work, a scope must not be re-entered until the work
//!    recorded in the previous one has been DRAINED - a blocking read or a
//!    `poll_wait`. Every caller in this workspace already drains per
//!    iteration, because that is what produces the activation it chains. A
//!    caller that does not drain must not open a scope.
//!
//! One more caller obligation, implied by (2): a scope is a property of ONE
//! `Gpu` handle and its cursor is not re-entrant. Two threads sharing a handle
//! would interleave one cursor and race the refcount test; give each its own
//! `Gpu::share`. Entering a scope while one is open panics rather than
//! silently interleaving.
//!
//! A slot whose requested size differs from the size it was created at is
//! likewise not reused, so a caller whose sequence is not in fact identical
//! degrades to plain allocation instead of silently binding a short buffer.
//!
//! Only `Gpu::storage` replays the arena. `try_storage`, `storage_init`,
//! `buffer` and `uniform_dynamic` allocate as they always did, so a scope that
//! mixes them simply pools less; it cannot desync into an unsafe state,
//! because a slot is only ever reused when both its size and its refcount say
//! it is free.
//!
//! The arena never shrinks on its own: a scope that asks for fewer buffers
//! than an earlier one leaves the tail slots held. That is the right default
//! for a loop whose sequence is constant, and `Gpu::scratch_release` is how a
//! caller that is done with a shape gives the memory back.
//!
//! ## What it does NOT promise
//!
//! A recycled buffer holds whatever the previous scope left in it. A plain
//! allocation is zero-filled by the backend, so a kernel that reads a slot it
//! does not fully write behaves differently under the arena. That is a real
//! difference and it is why the callers that opt in are gated on the forward's
//! output being BIT-identical to the un-pooled path, not on a tolerance.

use backend_api::DeviceBuffer;

/// The per-handle replay list. Entries are `(words, buffer)` in the order the
/// scope requested them.
#[derive(Default)]
pub(crate) struct Arena {
    slots: Vec<(u64, DeviceBuffer)>,
    cursor: usize,
    /// Whether a scope is open. The arena OUTLIVES its scopes - that is what
    /// keeps the buffers alive from one iteration to the next - so existence
    /// cannot be the flag that decides whether an allocation is served from
    /// it. Outside a scope every request allocates, exactly as it did before
    /// the caller opted in.
    active: bool,
}

/// What the arena can answer for one `storage(n)` request inside a scope.
pub(crate) enum Slot {
    /// Reuse this buffer; nothing to allocate.
    Hit(DeviceBuffer),
    /// Allocate, then install the result at this index via [`Arena::install`].
    Miss(usize),
}

impl Arena {
    /// Advance the cursor and say whether the slot it names can be reused.
    pub(crate) fn take(&mut self, words: u64) -> Slot {
        let i = self.cursor;
        self.cursor += 1;
        match self.slots.get(i) {
            Some((sz, b)) if *sz == words && b.is_unique() => Slot::Hit(b.clone()),
            _ => Slot::Miss(i),
        }
    }

    /// Record a freshly allocated buffer as the arena's copy of slot `i`.
    pub(crate) fn install(&mut self, i: usize, words: u64, buf: DeviceBuffer) {
        assert!(i <= self.slots.len(), "gpu_core::scratch: slot {i} is past the end of a {}-slot arena", self.slots.len());
        if i < self.slots.len() {
            self.slots[i] = (words, buf);
        } else {
            self.slots.push((words, buf));
        }
    }

    /// Whether a scope is currently open on this arena.
    pub(crate) fn active(&self) -> bool {
        self.active
    }

    /// Open a scope: the sequence starts over.
    pub(crate) fn enter(&mut self) {
        assert!(
            !self.active,
            "gpu_core::scratch: a scope is already open on this handle. Scopes do not nest, and two THREADS sharing one handle would interleave one cursor - give each its own `Gpu::share`"
        );
        self.active = true;
        self.cursor = 0;
    }

    /// Close the scope. The slots stay, which is the whole point.
    pub(crate) fn leave(&mut self) {
        self.active = false;
    }

    /// Buffers currently held, and the words they total - what a caller
    /// reports when it wants to say what the arena is costing in device
    /// memory.
    pub(crate) fn held(&self) -> (usize, u64) {
        (self.slots.len(), self.slots.iter().map(|(n, _)| *n).sum())
    }
}
