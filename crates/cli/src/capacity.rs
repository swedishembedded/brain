// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **What memory does this machine actually have free, right now?** - the one
//! hardware probe every budget in brain is built from.
//!
//! Swedish Embedded AB implements live accelerator capacity accounting for its
//! clients. If your team needs expertise in scheduling large models onto
//! hardware that other jobs are also using, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # Why this exists
//!
//! There used to be two probes, and they disagreed. The one-shot CLI placer
//! read `nvidia-smi --query-gpu=memory.free` and budgeted a card at its FREE
//! bytes less a 1 GiB headroom; `brain serve` read `memory.total` and budgeted
//! the SAME card at its TOTAL bytes less `--reserve-gb`. On a 24 GiB card with
//! 18 GiB held by a neighbouring process, in one process at one instant, the
//! CLI half budgeted 5 GiB usable and the daemon half 22 GiB - and the daemon
//! half then placed a 16 GiB model onto a card with 6 GiB physically free and
//! aborted inside the driver. A capacity model that cannot see another
//! process's allocations is not a capacity model.
//!
//! So: ONE probe, ONE definition of what a card offers, and both consumers
//! built from it. `total` is still reported, because `brain devices` and the
//! perf suite legitimately want the card's size rather than its availability -
//! but nothing budgets from it.
//!
//! # What it does and does not cover
//!
//! It is a SNAPSHOT. Between the probe and the allocation, a neighbouring
//! process can take the bytes it just reported free - there is no
//! machine-wide reservation here (see `crates/residency/src/manager.rs`'s
//! notes on the retry path that covers the common case instead). What this
//! removes is the systematic error: budgeting as if a card were empty when it
//! demonstrably is not.

/// One GPU's memory picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMem {
    /// Canonical device index - the same `i` that `Device::Gpu(i)` and
    /// `--device gpu<i>` mean, resolved through the device registry by PCI bus
    /// id, never by NVML enumeration order.
    pub index: u32,
    /// The card's size.
    pub total: u64,
    /// Bytes free right now, per the driver - the only figure that sees
    /// ANOTHER process's allocations.
    pub free: u64,
    /// False when `free` could not be measured (no `nvidia-smi`, a
    /// non-NVIDIA card, a silent line) and was assumed from `total`.
    pub measured: bool,
}

/// The fraction of an UNMEASURED card's total that is assumed already spoken
/// for (1/8 = 12.5%).
///
/// The tradeoff, stated plainly. When `nvidia-smi` cannot tell us how much of
/// a card is free, the honest answer is "unknown", and there are two ways to
/// be wrong. Assuming the card is EMPTY - what this code used to do, under a
/// doc comment promising the fallback is "never a hard failure" - plans an
/// allocation that then aborts inside the driver, which is the worst possible
/// failure mode and the opposite of what that doc promised. Assuming the card
/// is FULL costs a machine with a perfectly idle non-NVIDIA GPU its GPU
/// entirely, and a hundredfold slowdown is not "graceful" either.
///
/// So: a margin, not a guess at the neighbour. It absorbs driver/context
/// overhead and small foreign allocations, and it deliberately does NOT
/// pretend to cover a large neighbouring job - nothing short of a real
/// measurement can, and the retry-on-failure path is what covers that case.
const UNMEASURED_MARGIN: u64 = 8;

impl GpuMem {
    /// The bytes a budget may be built from on this card: what is free, less
    /// a conservative margin when `free` was assumed rather than measured.
    ///
    /// The consumer's own reserve (the one-shot placer's `HEADROOM`, the
    /// daemon's `--reserve-gb`) is applied ON TOP of this, by whoever calls
    /// `Budgets::set` - this function reports the machine, not the policy.
    pub fn available(&self) -> u64 {
        if self.measured {
            self.free
        } else {
            self.free.saturating_sub(self.free / UNMEASURED_MARGIN)
        }
    }
}

/// Probe every registry GPU's total and free bytes in ONE `nvidia-smi` call.
///
/// Cards the probe cannot see keep the registry's own VRAM size as `total`,
/// the same figure as `free`, and `measured: false`. A card with no known size
/// at all is not a schedulable card and is dropped - that, and only that, is
/// what "empty list" means here.
///
/// Note what is deliberately NOT filtered: a card with zero bytes FREE stays
/// in the list. Dropping it (which the free-VRAM probe used to do) emptied the
/// GPU class, and every "is there a GPU?" test downstream then answered "no" -
/// so a fully-consumed card behaved like a GPU-less box (host tier, works)
/// while a card with a single free byte behaved like a healthy GPU box (hard
/// failure). A more contended machine came out strictly ahead of a less
/// contended one. Capacity is what decides whether a card exists; occupancy is
/// what decides whether anything fits on it.
pub fn probe_gpus() -> Vec<GpuMem> {
    let mut mem: Vec<GpuMem> = gpu_core::devices::gpus()
        .iter()
        .map(|d| GpuMem { index: d.index, total: d.identity.vram_bytes, free: d.identity.vram_bytes, measured: false })
        .collect();
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output()
    {
        if o.status.success() {
            for l in String::from_utf8_lossy(&o.stdout).lines() {
                let mut it = l.split(',').map(str::trim);
                let (Some(pci), Some(total), Some(free)) = (
                    it.next(),
                    it.next().and_then(|m| m.parse::<u64>().ok()),
                    it.next().and_then(|m| m.parse::<u64>().ok()),
                ) else {
                    continue;
                };
                if let Some(d) = gpu_core::devices::device_by_pci(pci) {
                    if let Some(slot) = mem.iter_mut().find(|g| g.index == d.index) {
                        slot.total = total << 20;
                        slot.free = free << 20;
                        slot.measured = true;
                    }
                }
            }
        }
    }
    mem.retain(|g| g.total > 0);
    mem
}

/// `(index, available bytes)` for every card - what a `Budgets` GPU tier is
/// built from, on both the one-shot and the served path.
pub fn available_gpus() -> Vec<(u32, u64)> {
    probe_gpus().into_iter().map(|g| (g.index, g.available())).collect()
}

/// `(index, total bytes)` - the card's SIZE, for reporting (`brain devices`,
/// the perf suite's environment block). Never a budget input.
pub fn gpu_totals() -> Vec<(u32, u64)> {
    probe_gpus().into_iter().map(|g| (g.index, g.total)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A measured card offers what is free - not its size.
    #[test]
    fn a_measured_card_offers_its_free_bytes_not_its_total() {
        let g = GpuMem { index: 0, total: 24 * GIB, free: 6 * GIB, measured: true };
        assert_eq!(g.available(), 6 * GIB);
    }

    /// An UNMEASURED card is budgeted conservatively rather than as if empty.
    /// Assuming the full card is what turned "the probe is unavailable" - a
    /// documented soft failure - into a driver-level OOM abort.
    #[test]
    fn an_unmeasured_card_is_budgeted_below_its_total() {
        let assumed = GpuMem { index: 0, total: 24 * GIB, free: 24 * GIB, measured: false };
        let known = GpuMem { measured: true, ..assumed };
        assert!(assumed.available() < known.available(), "an unknown card must not be planned against as if it were empty");
        assert_eq!(assumed.available(), 21 * GIB);
    }

    /// A fully-consumed card offers nothing, and stays in the list.
    #[test]
    fn a_full_card_offers_nothing_and_is_not_dropped() {
        let g = GpuMem { index: 0, total: 24 * GIB, free: 0, measured: true };
        assert_eq!(g.available(), 0);
        let tiny = GpuMem { index: 0, total: 24 * GIB, free: 1, measured: true };
        assert_eq!(tiny.available(), 1, "the card stays in the list; nothing will fit on it");
    }
}
