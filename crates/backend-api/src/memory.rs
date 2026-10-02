// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The memory limits of a device, as four separate answers to four separate
//! questions.
//!
//! Swedish Embedded AB implements memory-aware GPU runtimes for its clients,
//! where one number standing in for "how much can I use" sizes slabs the card
//! cannot allocate on one machine and starves a large card on the next. If your
//! team needs expertise in sizing allocations, bindings and working sets from
//! what a device actually reports, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # Why four numbers
//!
//! A single "max storage binding" used to answer all of these at once, and
//! every consumer read it as whichever one it needed:
//!
//! * the **addressable allocation** - the largest single allocation the driver
//!   will hand out;
//! * the **bindable range** - the largest range one dispatch may name through a
//!   single binding, which is a property of how the kernels index it, not of how
//!   much memory exists;
//! * the **workspace** - the largest scratch or staging block a pass should take
//!   on top of what it already holds;
//! * the **working set** - the slab a pipeline that tiles its work should aim
//!   for, from which tile budgets are derived.
//!
//! Reporting a large card's real memory as the one number made tiled pipelines
//! size slabs the device could not bind; reporting the portable floor starved
//! them on a card that could hold the whole operand. Each field below is read
//! by the consumer that asked that question.

/// A device's memory limits. See the module documentation for what each field
/// answers; a backend that cannot tell them apart reports the same figure in
/// every field ([`MemoryLimits::uniform`]), which is exactly what the single
/// pre-split number meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLimits {
    /// The largest single allocation the device accepts, bytes.
    pub max_allocation_bytes: u64,
    /// The largest range one dispatch may bind through a single binding, bytes.
    pub max_binding_bytes: u64,
    /// The largest scratch or staging block a pass should take, bytes.
    pub workspace_bytes: u64,
    /// The slab a tiling pipeline should aim for, bytes. Never above
    /// [`Self::max_binding_bytes`]: a tile is one binding.
    pub working_set_bytes: u64,
}

impl MemoryLimits {
    /// The limits of a backend that distinguishes only the binding ceiling and
    /// the allocation ceiling: scratch and working set are both sized from the
    /// binding, as they were when the binding was the only number.
    pub fn uniform(max_binding_bytes: u64, max_allocation_bytes: u64) -> MemoryLimits {
        MemoryLimits {
            max_allocation_bytes,
            max_binding_bytes,
            workspace_bytes: max_binding_bytes,
            working_set_bytes: max_binding_bytes,
        }
    }

    /// Whether the fields respect the relations consumers rely on: a tile is
    /// one binding, so the working set cannot exceed it.
    pub fn is_consistent(&self) -> bool {
        self.working_set_bytes <= self.max_binding_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_with_one_number_gets_it_in_every_sizing_field() {
        let l = MemoryLimits::uniform(2 * 1024 * 1024 * 1024 - 1, 8 << 30);
        assert_eq!(l.max_binding_bytes, 2 * 1024 * 1024 * 1024 - 1);
        assert_eq!(l.max_allocation_bytes, 8 << 30);
        assert_eq!(l.workspace_bytes, l.max_binding_bytes);
        assert_eq!(l.working_set_bytes, l.max_binding_bytes);
        assert!(l.is_consistent());
    }

    #[test]
    fn a_working_set_larger_than_the_binding_is_inconsistent() {
        let l = MemoryLimits { max_allocation_bytes: 8 << 30, max_binding_bytes: 1 << 30, workspace_bytes: 1 << 20, working_set_bytes: 2 << 30 };
        assert!(!l.is_consistent());
    }
}
