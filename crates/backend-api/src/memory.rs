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

/// What the driver says about how host and device memory can be shared on this
/// machine. Every field is a query result, never inferred from the device
/// class: a coherent Grace-Hopper node reports an integrated flag of zero and
/// still lets the GPU read ordinary host allocations, and an integrated part
/// may offer none of this.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlacementFacts {
    /// Managed (migrating) allocations can be made.
    pub managed_memory: bool,
    /// The device can access ordinary pageable host memory without registering
    /// it (address-translation services): a plain host allocation is a valid
    /// kernel argument.
    pub pageable_memory_access: bool,
    /// ...and does so through the host's page tables, so first touch and page
    /// placement are the host's.
    pub pageable_via_host_page_tables: bool,
    /// The host-device link carries every native atomic operation.
    pub host_native_atomics: bool,
    /// The device can touch managed memory while the CPU does.
    pub concurrent_managed_access: bool,
    /// The CPU can read managed memory that lives on the device without
    /// migrating it.
    pub direct_managed_from_host: bool,
    /// Host memory can be registered (pinned in place) for device access.
    pub host_register: bool,
}

impl PlacementFacts {
    /// Nothing is shared: the answer of a backend that asks no such question.
    pub const NONE: PlacementFacts = PlacementFacts {
        managed_memory: false,
        pageable_memory_access: false,
        pageable_via_host_page_tables: false,
        host_native_atomics: false,
        concurrent_managed_access: false,
        direct_managed_from_host: false,
        host_register: false,
    };

    /// Whether an allocation under `policy` can be made and used by kernels.
    pub fn supports(&self, policy: AllocPolicy) -> bool {
        match policy {
            AllocPolicy::Device => true,
            AllocPolicy::Managed => self.managed_memory,
            AllocPolicy::System => self.pageable_memory_access,
        }
    }
}

/// Where an allocation's bytes live, as an explicit choice.
///
/// [`AllocPolicy::Device`] is the default and the only policy any model weight
/// or hot buffer is allocated with. The other two are opt-in tools for data
/// that is touched rarely, is larger than the card, or is shared with the
/// host; nothing in the engine selects them on its own, and a backend that
/// cannot honour one refuses it rather than quietly allocating on the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AllocPolicy {
    /// Device memory (HBM on a discrete or Grace-Hopper part).
    #[default]
    Device,
    /// Managed memory: one address, migrated between host and device by the
    /// driver on demand.
    Managed,
    /// Ordinary host memory used directly by kernels (needs
    /// [`PlacementFacts::pageable_memory_access`]); it stays where the host's
    /// allocator put it, which on a coherent node is the CPU's memory, read by
    /// the GPU across the link.
    System,
}

/// Advice about a managed or system range, as the driver's `cuMemAdvise` takes
/// it. The device is always this backend's own device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemAdvice {
    /// Mostly read: the driver may keep read-only copies on each processor.
    ReadMostly,
    UnsetReadMostly,
    /// Keep the pages on the device.
    PreferDevice,
    /// Keep the pages in host memory.
    PreferHost,
    UnsetPreferred,
    /// The device will access the range: map it up front instead of faulting.
    AccessedByDevice,
    UnsetAccessedByDevice,
}

/// Where [`crate::Backend::prefetch`] moves a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefetchTarget {
    Device,
    Host,
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

    /// Nothing outside the device is assumed shareable until the driver says so,
    /// and the default allocation policy is the device.
    #[test]
    fn placement_defaults_to_the_device_and_assumes_no_sharing() {
        assert_eq!(AllocPolicy::default(), AllocPolicy::Device);
        let none = PlacementFacts::default();
        assert_eq!(none, PlacementFacts::NONE);
        assert!(none.supports(AllocPolicy::Device));
        assert!(!none.supports(AllocPolicy::Managed));
        assert!(!none.supports(AllocPolicy::System));
        let ats = PlacementFacts { pageable_memory_access: true, ..PlacementFacts::NONE };
        assert!(ats.supports(AllocPolicy::System));
        assert!(!ats.supports(AllocPolicy::Managed), "system memory access does not imply managed allocation");
    }

    #[test]
    fn a_working_set_larger_than_the_binding_is_inconsistent() {
        let l = MemoryLimits { max_allocation_bytes: 8 << 30, max_binding_bytes: 1 << 30, workspace_bytes: 1 << 20, working_set_bytes: 2 << 30 };
        assert!(!l.is_consistent());
    }
}
