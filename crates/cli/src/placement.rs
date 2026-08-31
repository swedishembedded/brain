// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The one production placement policy.** Turns the machine brain is
//! actually running on into a `residency::budget::Budgets`, and answers
//! `gpu_core::devices::Placer` from it.
//!
//! Swedish Embedded AB implements automatic multi-device model placement for
//! its clients. If your team needs expertise in getting large models onto the
//! accelerators a machine really has - and a legible refusal when they do not
//! fit - you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this lives here
//!
//! `crates/gpu-core` (which every model crate depends on) and
//! `crates/residency` (which owns the capacity model, and which
//! `crates/stats` depends on precisely because it pulls no GPU code) must not
//! depend on each other. So the placement seam is dependency-inverted:
//! `gpu-core` DECLARES [`gpu_core::devices::Placer`] and asks it; `residency`
//! OWNS the decision (`residency::plan::plan` over `budget::Budgets` and
//! `place::pick_device`); and this module - in `crates/cli`, the crate that
//! may depend on both - is the only thing that knows both halves exist. It is
//! the same shape as `residency::supply::ModelSupplier`, declared in
//! `residency` and implemented here because only the CLI may depend on
//! `modelstore`.
//!
//! Nothing here re-derives what fits where. It probes hardware, builds
//! budgets, and delegates.

use std::sync::Arc;

use gpu_core::devices::{Home, Need, Placer};
use residency::budget::Budgets;
use residency::plan::{self, Part};
use residency::{Device, MemCost};

/// Headroom kept free on every card by automatic placement, on top of the
/// bytes another process already holds.
///
/// Budgets here are built from **free** VRAM, not total, so this is not the
/// serving path's `--reserve-gb` (which carves a slice out of a whole card so
/// resident models never pack it to the brim). It covers what a model's own
/// figure does not: the driver/context allocation a fresh `Gpu` makes, and
/// transient activation scratch a weights-only estimate omits.
const HEADROOM: u64 = 1 << 30;

/// Live per-GPU free bytes, `(canonical index, free)`.
///
/// `nvidia-smi`'s `memory.free` is the only figure that sees ANOTHER
/// process's allocations - which is the whole reported bug, a card with
/// 18 GiB already taken by a neighbouring job. Keyed by PCI bus id through
/// the device registry, never by NVML enumeration order, exactly as
/// `run_cli::query_gpu_mem` does for totals.
///
/// When `nvidia-smi` is absent or silent about a card, that card falls back
/// to the registry's own VRAM size - i.e. to today's capacity-blind
/// behaviour for that card, never to a hard failure.
pub fn probe_free_vram() -> Vec<(u32, u64)> {
    let mut mem: Vec<(u32, u64)> =
        gpu_core::devices::gpus().iter().map(|d| (d.index, d.identity.vram_bytes)).collect();
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,memory.free", "--format=csv,noheader,nounits"])
        .output()
    {
        if o.status.success() {
            for l in String::from_utf8_lossy(&o.stdout).lines() {
                let mut it = l.split(',').map(str::trim);
                let (Some(pci), Some(mib)) = (it.next(), it.next().and_then(|m| m.parse::<u64>().ok())) else {
                    continue;
                };
                if let Some(d) = gpu_core::devices::device_by_pci(pci) {
                    if let Some(slot) = mem.iter_mut().find(|(i, _)| *i == d.index) {
                        slot.1 = mib << 20;
                    }
                }
            }
        }
    }
    mem.retain(|&(_, bytes)| bytes > 0);
    mem
}

/// Budgets for automatic placement: one per schedulable GPU sized to its FREE
/// bytes less [`HEADROOM`], plus the host tier.
///
/// The host tier is always declared. On a box with cards it changes nothing -
/// `residency::place::pick_device` spills a device-resident model to the CPU
/// only when no accelerator of that class exists at all - and on a GPU-less
/// box it is what lets a model be placed instead of refused.
pub fn budgets(gpus: &[(u32, u64)], ram: u64) -> Budgets {
    let mut b = Budgets::new();
    let limits = memauth::limits();
    for &(i, free) in gpus {
        b.set(Device::Gpu(i), limits.clamp(Device::Gpu(i), free), HEADROOM.min(free));
    }
    b.set(Device::Cpu, limits.clamp(Device::Cpu, ram), 0);
    b
}

/// [`Placer`] backed by [`residency::plan::plan`] over a snapshot of the
/// machine's free capacity.
///
/// The snapshot is taken once, at install. Re-probing per call would let one
/// model's own allocations move the answer between its parts, scattering a
/// pipeline across the machine as it is built.
pub struct BudgetPlacer {
    budgets: Budgets,
}

impl BudgetPlacer {
    pub fn new(budgets: Budgets) -> BudgetPlacer {
        BudgetPlacer { budgets }
    }
}

/// `gpu_core`'s wire type -> `residency`'s. Data only; no policy crosses here.
fn to_part(n: &Need) -> Part {
    let mut p = if n.unsized_ { Part::unsized_(n.name.clone()) } else { Part::new(n.name.clone(), MemCost::new(n.vram, n.ram)) };
    if let Some(k) = n.phase {
        p = p.phase(k);
    }
    match &n.affinity {
        gpu_core::devices::Affinity::Any => p,
        gpu_core::devices::Affinity::With(a) => p.with(a.clone()),
        gpu_core::devices::Affinity::Apart => p.apart(),
    }
}

fn to_home(d: Device) -> Result<Home, String> {
    match d {
        Device::Gpu(i) => Ok(Home::Gpu(i)),
        Device::Cpu => Ok(Home::Cpu),
        // Never budgeted by `budgets()`, so unreachable in practice; loud
        // rather than silently mapped onto the wrong tier if that changes.
        Device::Npu(i) => Err(format!("automatic placement produced npu{i}, which has no build path here")),
    }
}

impl Placer for BudgetPlacer {
    fn place(&self, needs: &[Need]) -> Result<Vec<Home>, String> {
        let parts: Vec<Part> = needs.iter().map(to_part).collect();
        let placement = plan::plan(&parts, &self.budgets).map_err(|e| e.to_string())?;
        let homes = needs
            .iter()
            .map(|n| placement.of(&n.name).ok_or_else(|| format!("part {} unplaced", n.name)).and_then(to_home))
            .collect::<Result<Vec<Home>, String>>()?;
        // Say what it did. A multi-part plan is reported in full; the
        // single-part default (every bare `Gpu::new`) is reported once, and
        // only when it lands somewhere other than card 0 - a run that took
        // the historical default has nothing to explain.
        if needs.len() > 1 {
            let line: Vec<String> = needs.iter().zip(&homes).map(|(n, h)| format!("{}={h}", n.name)).collect();
            eprintln!("brain: placement {} ({})", line.join(" "), self.free_summary());
        } else if self.budgets.gpus().len() > 1 && !matches!(homes.first(), Some(Home::Gpu(0)) | Some(Home::Cpu) | None) {
            // Only when there was a real choice to make and it did not land
            // on the historical default. One card, the host tier, or card 0
            // are all "what you would have got anyway" and have nothing to
            // explain.
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                if let Some(h) = homes.first() {
                    eprintln!("brain: placement {h} ({})", self.free_summary());
                }
            });
        }
        Ok(homes)
    }
}

impl BudgetPlacer {
    fn free_summary(&self) -> String {
        let mut gpus = self.budgets.gpus();
        gpus.sort_by_key(|d| match d {
            Device::Gpu(i) => *i,
            _ => u32::MAX,
        });
        gpus.iter()
            .map(|&d| format!("{} {:.1} GiB free", plan::device_name(d), self.budgets.free_on(d) as f64 / (1u64 << 30) as f64))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Install the production placer for this process.
///
/// Called once from `main` after `--device` has been resolved, so the
/// candidate set is exactly what the user made schedulable. With no GPU
/// present this still installs (the host tier answers), and with an explicit
/// `--device gpu<i>` the placer is never consulted - `gpu_core::devices::
/// selected_device` only asks when the user expressed no preference.
pub fn install() {
    let free = probe_free_vram();
    // `--device` narrows the candidate set; with no `--device` every card is
    // a candidate, which is the "use all the hardware" default.
    let gpus: Vec<(u32, u64)> = match crate::compute_set() {
        Some(s) => free.into_iter().filter(|(i, _)| s.gpus.contains(i)).collect(),
        None => free,
    };
    let ram = crate::run_cli::query_ram_bytes();
    gpu_core::devices::install_placer(Arc::new(BudgetPlacer::new(budgets(&gpus, ram))));
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The reported bug, at the layer that fixes it: gpu0 loaded by a
    /// neighbour, gpu1 free, no `--device`. The placer must answer gpu1.
    #[test]
    fn a_loaded_card_is_not_offered_to_a_model_that_cannot_fit_on_it() {
        let p = BudgetPlacer::new(budgets(&[(0, 5 * GIB), (1, 24 * GIB)], 128 * GIB));
        let homes = p.place(&[Need::sized("dit", 16 * GIB, 0)]).expect("plan");
        assert_eq!(homes, vec![Home::Gpu(1)]);
    }

    /// ...and the bare `Gpu::new` default (no cost known) follows the same
    /// rule, because it is the same policy.
    #[test]
    fn the_no_preference_default_follows_free_capacity() {
        let p = BudgetPlacer::new(budgets(&[(0, 2 * GIB), (1, 24 * GIB)], 128 * GIB));
        assert_eq!(p.place(&[Need::unsized_("model")]).expect("plan"), vec![Home::Gpu(1)]);
        let q = BudgetPlacer::new(budgets(&[(0, 24 * GIB), (1, 24 * GIB)], 128 * GIB));
        assert_eq!(q.place(&[Need::unsized_("model")]).expect("plan"), vec![Home::Gpu(0)], "an idle box keeps card 0");
    }

    /// A pipeline whose parts do not co-reside gets two cards without any
    /// model naming one.
    #[test]
    fn a_pipeline_spreads_without_any_model_naming_a_card() {
        let p = BudgetPlacer::new(budgets(&[(0, 24 * GIB), (1, 24 * GIB)], 128 * GIB));
        let homes = p
            .place(&[Need::sized("dit", 14 * GIB, 0).apart(), Need::sized("te", 7 * GIB, 0).apart(), Need::sized("vae", 2 * GIB, 0).with("dit")])
            .expect("plan");
        assert_ne!(homes[0], homes[1], "dit and te must not share: {homes:?}");
        assert_eq!(homes[0], homes[2], "vae must follow the dit: {homes:?}");
    }

    /// The 9B-diffusion shape that motivated phases: a denoiser and a decode
    /// graph that cannot co-reside take turns on ONE card, because the
    /// pipeline evicts the denoiser before it builds the decode graph. The
    /// same declaration without phases - a 24 GiB simultaneous resident on a
    /// 23 GiB card - must still be refused.
    #[test]
    fn phased_parts_take_turns_on_one_card() {
        let p = BudgetPlacer::new(budgets(&[(0, 23 * GIB)], 128 * GIB));
        let needs = [
            Need::sized("dit", 13 * GIB, 0).apart().phase(1),
            Need::sized("te", 7 * GIB, 0).apart(),
            Need::sized("vae_dec", 11 * GIB, 0).with("dit").phase(2),
        ];
        assert_eq!(
            p.place(&needs).expect("peak = 7 + max(13, 11) = 20 GiB: fits"),
            vec![Home::Gpu(0), Home::Gpu(0), Home::Gpu(0)]
        );
        let unphased: Vec<Need> = needs.iter().cloned().map(|n| Need { phase: None, ..n }).collect();
        let e = p.place(&unphased).expect_err("13 + 7 + 11 = 31 GiB live at once does not fit 23");
        assert!(e.contains("dit"), "{e}");
    }

    /// Nothing fits: the refusal names the part, its size and every card's
    /// free bytes, instead of letting the driver report a bare OOM later.
    #[test]
    fn an_impossible_model_is_refused_legibly() {
        let p = BudgetPlacer::new(budgets(&[(0, 5 * GIB), (1, 6 * GIB)], 8 * GIB));
        let e = p.place(&[Need::sized("dit", 40 * GIB, 0)]).expect_err("40 GiB fits nothing");
        for want in ["dit", "40", "gpu0", "gpu1"] {
            assert!(e.contains(want), "refusal must name {want:?}; got {e}");
        }
    }

    /// A GPU-less box places on the host tier rather than refusing.
    #[test]
    fn a_gpu_less_box_places_on_the_host_tier() {
        let p = BudgetPlacer::new(budgets(&[], 128 * GIB));
        assert_eq!(p.place(&[Need::sized("dit", 16 * GIB, 0)]).expect("plan"), vec![Home::Cpu]);
    }

    /// The placer and the SERVING path decide from the same policy. Given the
    /// same budgets, `residency::place::pick_device` - what
    /// `ResidencyManager::claim` uses to place a resident model - agrees with
    /// what a one-shot CLI build is told. One capacity model, two consumers.
    #[test]
    fn the_serving_path_and_the_cli_agree_on_where_a_model_goes() {
        let b = budgets(&[(0, 5 * GIB), (1, 24 * GIB)], 128 * GIB);
        let cost = MemCost::new(16 * GIB, 0);
        let served = residency::place::pick_device(&cost, &b, &residency::place::no_exclude());
        let built = BudgetPlacer::new(b).place(&[Need::sized("dit", 16 * GIB, 0)]).expect("plan");
        assert_eq!(served, Some(Device::Gpu(1)));
        assert_eq!(built, vec![Home::Gpu(1)]);
    }
}
