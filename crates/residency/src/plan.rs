// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Where does each part of a model go?** - the multi-part generalisation of
//! [`crate::place::pick_device`], and the one policy every model in brain
//! inherits instead of hand-rolling.
//!
//! Swedish Embedded AB implements automatic multi-device model placement for
//! its clients. If your team needs expertise in fitting large models across
//! the accelerators a machine actually has - and refusing legibly when they
//! do not fit - you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # The problem this exists to remove
//!
//! A pipeline is several models with a known memory cost and a known
//! dependency structure: a DiT, a text encoder and a VAE; an AR branch and
//! its twin; a vision tower and a decoder. Which device each part goes on is
//! a *capacity* question, and [`crate::budget::Budgets`] +
//! [`crate::place::pick_device`] have answered capacity questions for
//! resident models since this crate existed. What was missing was a way for a
//! model to ASK that question at build time, so every multi-part model grew
//! its own answer instead - a bespoke env var here, an ambient-card
//! assumption there.
//!
//! A model declares [`Part`]s (a name, a [`MemCost`], and at most an
//! [`Affinity`] constraint) and gets a [`Placement`] back. It never names a
//! card. The declaration is the whole interface.
//!
//! # Hardware-free by construction
//!
//! Everything here is a pure function of the [`Budgets`] it is handed, so it
//! is unit-tested without a GPU and cannot drift from the budgets the
//! residency manager already accounts against. Turning real hardware into
//! `Budgets` (probing free VRAM, host RAM, `--device` narrowing) is the
//! caller's job - `crates/cli` does it, exactly as it supplies the concrete
//! [`crate::supply::ModelSupplier`] this crate only declares.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::budget::Budgets;
use crate::place::pick_device;
use crate::{Device, MemCost};

/// A placement constraint one part declares about another. Anything a model
/// needs to say about *where* belongs here, so no model needs placement code.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Affinity {
    /// Wherever it fits best. The default.
    #[default]
    Any,
    /// Must land on the same device as the named part (a VAE decoding the
    /// DiT's own latents; a projector feeding its decoder). Cross-device
    /// traffic between these two would be per-step, not once.
    With(String),
    /// Prefer a device no other part of this plan is already on. The
    /// declaration behind "an int8 9B DiT and its text encoder do not
    /// co-reside on one 24 GiB card" - a preference, not a demand: with only
    /// one device present the part still lands, on the shared one.
    Apart,
}

/// One part of a model that needs a home.
#[derive(Clone, Debug)]
pub struct Part {
    /// What it is called in the placement report (`dit`, `te`, `vae`).
    pub name: String,
    /// What it will occupy once built.
    pub cost: MemCost,
    pub affinity: Affinity,
    /// True when [`Part::unsized_`] built this: the part holds real device
    /// bytes but nobody has costed them. It is PLACED by free capacity like
    /// any other accelerator part, and CHARGED nothing - an invented number
    /// would distort every part that follows it.
    pub unsized_: bool,
    /// The pipeline stage this part is live in, when the caller evicts between
    /// stages. Parts in different phases never co-reside on a device - the
    /// caller frees phase k's weights before allocating phase k+1's - so a
    /// device is charged the MAX over phases rather than the sum. `None` (the
    /// default) is permanent: resident in every phase, charged beside each.
    pub phase: Option<u32>,
}

/// The device bytes a card must still have free before an UNSIZED part - one
/// that holds real bytes nobody costed - is put on it.
///
/// An unsized part is charged nothing (an invented number would distort every
/// sized part), which used to mean it was also *placed* by a 1-byte probe: a
/// card holding a 23 GiB part out of 24 GiB still "fit" it, and the two were
/// silently double-booked onto one card. Charging nothing is only safe if
/// unsized groups are placed LAST (nothing follows them to distort - see the
/// sort in [`plan`]) and only onto a card with real slack left. This constant
/// is what "real slack" means: the same order of magnitude as
/// `crates/cli`'s own `HEADROOM`, and for the same reason - a fresh `Gpu`
/// handle's driver/context allocation plus a minimal activation scratch is
/// already about this big before any weights are read. A card below it is
/// full for practical purposes, and the part goes to the host tier instead.
const UNSIZED_FLOOR: u64 = 1 << 30;

impl Part {
    /// A part of known size, unconstrained.
    pub fn new(name: impl Into<String>, cost: MemCost) -> Part {
        Part { name: name.into(), cost, affinity: Affinity::Any, unsized_: false, phase: None }
    }
    /// A part whose size is not known yet - the "just give me a card" case
    /// every bare `Gpu::new` takes. It is charged nothing, so it lands on the
    /// emptiest accelerator and does not distort what follows it.
    pub fn unsized_(name: impl Into<String>) -> Part {
        Part { name: name.into(), cost: MemCost::new(0, 0), affinity: Affinity::Any, unsized_: true, phase: None }
    }
    /// Declare [`Affinity::With`].
    pub fn with(mut self, anchor: impl Into<String>) -> Part {
        self.affinity = Affinity::With(anchor.into());
        self
    }
    /// Declare [`Affinity::Apart`].
    pub fn apart(mut self) -> Part {
        self.affinity = Affinity::Apart;
        self
    }
    /// Declare the pipeline stage this part is live in. The caller owes the
    /// eviction: phase k's weights must be freed before phase k+1's allocate,
    /// on every device, or the plan's max-over-phases charge is a lie.
    pub fn phase(mut self, phase: u32) -> Part {
        self.phase = Some(phase);
        self
    }
}

/// Where every part of a plan goes.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Placement {
    parts: Vec<(String, Device)>,
}

impl Placement {
    pub fn new(parts: Vec<(String, Device)>) -> Placement {
        Placement { parts }
    }
    /// The device `name` was placed on, or `None` if it was not in the plan.
    pub fn of(&self, name: &str) -> Option<Device> {
        self.parts.iter().find(|(n, _)| n == name).map(|(_, d)| *d)
    }
    pub fn parts(&self) -> &[(String, Device)] {
        &self.parts
    }
    /// One line, in declaration order: `dit=gpu1 te=gpu0 vae=gpu1`. What a run
    /// prints so an automatic decision is never a silent one.
    pub fn describe(&self) -> String {
        self.parts.iter().map(|(n, d)| format!("{n}={}", device_name(*d))).collect::<Vec<_>>().join(" ")
    }
}

/// `gpu0` / `cpu` / `npu0` - the spelling `--device` uses, so a printed
/// placement can be pasted back in as an override.
pub fn device_name(d: Device) -> String {
    match d {
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
        Device::Cpu => "cpu".to_string(),
    }
}

/// Why a plan could not be placed, with every number a human needs to see it.
/// This is the legible refusal that replaces a raw `wgpu error: Out of Memory`.
#[derive(Clone, Debug)]
pub struct Unplaceable {
    /// The part that could not be placed.
    pub part: String,
    /// What it needed.
    pub cost: MemCost,
    /// Every budgeted device and the bytes free on it at that moment.
    pub free: Vec<(Device, u64)>,
    /// What had already been placed (and is therefore charged against `free`).
    pub placed: Vec<(String, Device)>,
}

fn gib(b: u64) -> String {
    format!("{:.1}", b as f64 / (1u64 << 30) as f64)
}

impl std::fmt::Display for Unplaceable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot place '{}' ({} GiB device / {} GiB host)", self.part, gib(self.cost.vram), gib(self.cost.ram))?;
        if !self.placed.is_empty() {
            let already: Vec<String> = self.placed.iter().map(|(n, d)| format!("{n}={}", device_name(*d))).collect();
            write!(f, " after placing {}", already.join(" "))?;
        }
        write!(f, "; free:")?;
        for (d, free) in &self.free {
            write!(f, " {}={} GiB", device_name(*d), gib(*free))?;
        }
        Ok(())
    }
}

impl std::error::Error for Unplaceable {}

/// Place every [`Part`] against `budgets`, charging each placement so the
/// parts that follow see what is actually left.
///
/// Order of consideration is largest-first (a 16 GiB DiT must get its pick of
/// the cards before a 2 GiB VAE takes one), except that a part with
/// [`Affinity::With`] always waits for its anchor - so affinity is
/// declarative, not positional. The returned [`Placement`] is in the caller's
/// declaration order regardless.
///
/// Phased parts ([`Part::phase`]) are charged per phase: a device's charge is
/// its permanent bytes plus its heaviest single phase, never the sum over
/// phases, and a group joining a card is charged only how far that peak
/// GROWS (its marginal). This is a contract, not magic - the caller evicts
/// phase k's weights before phase k+1's allocate, or the charge is a lie.
pub fn plan(parts: &[Part], budgets: &Budgets) -> Result<Placement, Unplaceable> {
    let mut b = budgets.clone();
    let index_of = |name: &str| parts.iter().position(|p| p.name == name);

    // 1. Coalesce affinity groups. Parts joined by `With` must share a device,
    //    so they are placed as ONE part whose cost is their sum - not one
    //    after another. Placing them sequentially looks the same until the
    //    anchor takes the last device that could have held the pair, at which
    //    point the follower has nowhere to go and a plan that was perfectly
    //    placeable is refused.
    let mut group: Vec<usize> = (0..parts.len()).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    for (i, p) in parts.iter().enumerate() {
        if let Affinity::With(anchor) = &p.affinity {
            // A `With` naming an unknown part is a hint about nothing, not a
            // deadlock: the part places unconstrained.
            if let Some(a) = index_of(anchor) {
                let (ri, ra) = (root(&mut group, i), root(&mut group, a));
                if ri != ra {
                    group[ri] = ra;
                }
            }
        }
    }
    let mut roots: Vec<usize> = (0..parts.len()).map(|i| root(&mut group, i)).collect::<Vec<_>>();
    roots.sort_unstable();
    roots.dedup();

    // 2. Each group's summed cost, its per-phase VRAM charges, and whether any
    //    member asked to be Apart.
    struct Group {
        root: usize,
        members: Vec<usize>,
        cost: MemCost,
        /// The VRAM each member charges the card it lands on, paired with the
        /// phase it is live in (`None` = permanent). Phased charging is a VRAM
        /// contract: host RAM and NPU bytes are never evicted between stages.
        adds: Vec<(Option<u32>, u64)>,
        /// The device peak the group reaches alone on an empty card: its
        /// permanent bytes plus its heaviest single phase. At most
        /// `cost.vram` (the sum), and equal to it when nothing is phased.
        peak_vram: u64,
        /// The device bytes this group holds in EVERY phase - the part of
        /// `peak_vram` that is never handed back mid-run. The primary
        /// placement-order key: see [`plan`]'s ordering note.
        permanent_vram: u64,
        unsized_: bool,
        apart: bool,
    }
    let mut groups: Vec<Group> = roots
        .iter()
        .map(|&r| {
            let members: Vec<usize> = (0..parts.len()).filter(|&i| root(&mut group.clone(), i) == r).collect();
            let mut cost = MemCost::new(0, 0);
            let mut permanent = 0u64;
            let mut phases: BTreeMap<u32, u64> = BTreeMap::new();
            let mut unsized_ = true;
            let mut apart = false;
            for &m in &members {
                cost.vram += parts[m].cost.vram;
                cost.ram += parts[m].cost.ram;
                cost.npu += parts[m].cost.npu;
                match parts[m].phase {
                    None => permanent += parts[m].cost.vram,
                    Some(k) => *phases.entry(k).or_insert(0) += parts[m].cost.vram,
                }
                unsized_ &= parts[m].unsized_;
                apart |= parts[m].affinity == Affinity::Apart;
            }
            let peak_vram = permanent + phases.values().copied().max().unwrap_or(0);
            let adds: Vec<(Option<u32>, u64)> =
                members.iter().map(|&m| (parts[m].phase, parts[m].cost.vram)).collect();
            Group { root: r, members, cost, adds, peak_vram, permanent_vram: permanent, unsized_, apart }
        })
        .collect();
    // The order groups get their pick of the cards, most-constrained first.
    //
    // 1. SIZED before unsized. An unsized group is charged nothing, so it must
    //    see the final state of every card rather than distort it (see
    //    [`UNSIZED_FLOOR`]).
    // 2. PERMANENT device bytes, descending. A permanent part holds its card
    //    for the whole run and cannot be moved once built; a phased one is
    //    live for one stage and, if it has to take a slower tier, pays for
    //    that once. Given a 14 GiB permanent DiT and a 18 GiB transient text
    //    encoder against one 23 GiB card and one 8 GiB card, "biggest first"
    //    puts the encoder on the only card that could have held the DiT and
    //    strands it; permanent-first places the DiT and lets the encoder take
    //    the host tier, which is exactly the "slower, never failing" outcome
    //    the operator asked for. This is the real production failure that
    //    printed `cannot place 'dit' ... after placing te=gpu0`.
    // 3. Then the device PEAK, descending - the previous rule, and still what
    //    orders two parts of the same permanence.
    // 4. Then NPU bytes, then host bytes, purely to make the order total and
    //    deterministic. Host bytes must NOT outrank device bytes: they are a
    //    different budget on a different device, and comparing the two byte
    //    counts as if they were interchangeable let a part that is large only
    //    in host RAM take first pick of the emptiest CARD.
    groups.sort_by_key(|g| {
        (
            g.unsized_,
            std::cmp::Reverse(g.permanent_vram),
            std::cmp::Reverse(g.peak_vram),
            std::cmp::Reverse(g.cost.npu),
            std::cmp::Reverse(g.cost.ram),
            g.root,
        )
    });

    // A card's running residency, split by phase. `peak` is what is actually
    // live at the worst moment under the caller's eviction contract, and what
    // the budget is charged: the group a card holds is permanent bytes plus
    // its heaviest single phase, never the sum over phases.
    #[derive(Default, Clone)]
    struct Ledger {
        permanent: u64,
        phases: BTreeMap<u32, u64>,
    }
    impl Ledger {
        fn peak(&self) -> u64 {
            self.permanent + self.phases.values().copied().max().unwrap_or(0)
        }
        fn add(&mut self, phase: Option<u32>, vram: u64) {
            match phase {
                None => self.permanent += vram,
                Some(k) => *self.phases.entry(k).or_insert(0) += vram,
            }
        }
    }

    // Choose a GPU for a group the way `pick_device` chooses for a flat cost -
    // most free bytes wins among the cards that fit, `exclude` skipped - except
    // that what must fit is the group's MARGINAL charge on that card: how far
    // the card's ledger peak grows when the group lands. A flat probe cannot
    // express this. A mixed-phase group joining a card that already holds part
    // of its peak charges less than its own peak (the decode graph takes the
    // denoiser's place at no cost the denoiser did not already pay), so a probe
    // that is exact on an empty card over-refuses on a shared one.
    fn pick_gpu(adds: &[(Option<u32>, u64)], b: &Budgets, ledgers: &HashMap<Device, Ledger>, exclude: &HashSet<Device>) -> Option<Device> {
        let mut best: Option<(Device, u64)> = None;
        for d in b.gpus() {
            if exclude.contains(&d) {
                continue;
            }
            let mut trial = ledgers.get(&d).cloned().unwrap_or_default();
            let before = trial.peak();
            for &(phase, vram) in adds {
                trial.add(phase, vram);
            }
            let marginal = trial.peak() - before;
            if b.fits_on(d, marginal) {
                let free = b.free_on(d);
                if best.is_none_or(|(_, f)| free > f) {
                    best = Some((d, free));
                }
            }
        }
        best.map(|(d, _)| d)
    }

    // The emptiest card with real slack left, for a group nobody costed. Not
    // `pick_gpu` with a 1-byte probe: that "fits" a card with one free byte,
    // which is how an uncosted part carrying real bytes ended up double-booked
    // onto a card a 23 GiB part had already filled. See [`UNSIZED_FLOOR`].
    fn pick_gpu_unsized(b: &Budgets, exclude: &HashSet<Device>) -> Option<Device> {
        let mut best: Option<(Device, u64)> = None;
        for d in b.gpus() {
            if exclude.contains(&d) {
                continue;
            }
            let free = b.free_on(d);
            if free >= UNSIZED_FLOOR && best.is_none_or(|(_, f)| free > f) {
                best = Some((d, free));
            }
        }
        best.map(|(d, _)| d)
    }

    // Host bytes a group holds no matter where it lands, and whether the host
    // tier can still take them. `MemCost::ram` is documented as "host bytes it
    // will hold REGARDLESS of where it is placed", but only the device the
    // group landed on was ever charged - so a part declaring 40 GiB of host
    // staging RAM was placed happily on a card belonging to a box with 4 GiB
    // of RAM free, and two such parts left the host tier reporting itself
    // empty. An undeclared host tier is "unknown", not "infinite": it is not
    // charged and not checked, exactly as before, so a `Budgets` that only
    // names cards behaves unchanged.
    let host_declared = b.get(Device::Cpu).is_some();
    let host_ok = |b: &Budgets, ram: u64| !host_declared || ram == 0 || b.fits_on(Device::Cpu, ram);

    // 3. Place each group, charging it so the groups that follow see what is
    //    actually left.
    let mut placed: Vec<Option<Device>> = vec![None; parts.len()];
    let mut used: HashSet<Device> = HashSet::new();
    let mut ledgers: HashMap<Device, Ledger> = HashMap::new();
    for g in &groups {
        // What the group is placed BY on the flat paths (see `UNSIZED_PROBE`):
        // its peak, not its sum - the sum over phases is never live at once.
        let probe = MemCost { vram: g.peak_vram, ram: g.cost.ram, npu: g.cost.npu, mapped: 0 };
        // `allow_host` is what keeps [`Affinity::Apart`] meaning what it says.
        // Apart is a preference between CARDS ("do not stack the DiT and its
        // text encoder"), and the host tier is not a card - it is the fallback
        // tier. Letting the exclusion pass answer "the CPU" would satisfy
        // Apart trivially and permanently, so a two-part model on a one-card
        // box would put its second part on the CPU instead of sharing the
        // card, which is the exact opposite of the intent (Apart is a
        // preference, not a demand - one card still places every part). Only
        // the final, unconstrained pass may reach for the host tier.
        let pick = |exclude: &HashSet<Device>, allow_host: bool| {
            // The host tier, last. `plan` has no eviction to fall back on -
            // it is a one-shot "build this model now" answer - so refusing
            // while RAM sits idle is not "let the caller free something", it
            // is the whole run failing. This gate used to read
            // `b.gpus().is_empty()`, copied from `place::pick_device` where
            // the `None` really does mean "evict instead"; here it meant a
            // box whose cards were momentarily busy was strictly WORSE off
            // than a box with no cards at all.
            let host = |exclude: &HashSet<Device>| allow_host.then(|| crate::place::spill_to_host(&probe, &b, exclude)).flatten();
            if g.unsized_ {
                return pick_gpu_unsized(&b, exclude).or_else(|| {
                    (allow_host && b.get(Device::Cpu).is_some() && !exclude.contains(&Device::Cpu)).then_some(Device::Cpu)
                });
            }
            if !host_ok(&b, g.cost.ram) {
                // No device placement can rescue a group whose host bytes do
                // not fit: it holds them wherever it runs.
                return None;
            }
            if g.cost.vram == 0 {
                // A host-only or NPU-only group: `pick_device` already answers
                // the host tier for it, so gate that answer on `allow_host`
                // too rather than letting Apart be satisfied by the CPU.
                return match pick_device(&probe, &b, exclude) {
                    Some(Device::Cpu) if !allow_host => None,
                    Some(d) => Some(d),
                    None => host(exclude),
                };
            }
            // An NPU-capable group keeps `pick_device`'s class order: its NPU
            // bytes are unphased, so where the NPU fits, the flat probe is
            // exact and the NPU still wins over any GPU.
            if g.cost.npu > 0 && b.npus().iter().any(|d| b.fits_on(*d, probe.npu)) {
                return pick_device(&probe, &b, exclude);
            }
            pick_gpu(&g.adds, &b, &ledgers, exclude).or_else(|| host(exclude))
        };
        let dev = if g.apart {
            pick(&used, false).or_else(|| pick(&HashSet::new(), true))
        } else {
            pick(&HashSet::new(), true)
        };
        let Some(dev) = dev else {
            let mut free: Vec<(Device, u64)> = b.devices().map(|d| (d, b.free_on(d))).collect();
            free.sort_by_key(|(d, _)| match d {
                Device::Gpu(i) => (0u8, *i),
                Device::Npu(i) => (1, *i),
                Device::Cpu => (2, 0),
            });
            // Name the biggest member: a group is an implementation detail of
            // the constraint, the part is what the operator declared.
            let biggest = g.members.iter().copied().max_by_key(|&m| parts[m].cost.vram.max(parts[m].cost.ram)).unwrap_or(g.root);
            return Err(Unplaceable {
                part: if g.members.len() > 1 {
                    let names: Vec<&str> = g.members.iter().map(|&m| parts[m].name.as_str()).collect();
                    format!("{} (with {})", parts[biggest].name, names.join("+"))
                } else {
                    parts[biggest].name.clone()
                },
                cost: probe,
                free,
                placed: parts.iter().zip(&placed).filter_map(|(p, d)| d.map(|d| (p.name.clone(), d))).collect(),
            });
        };
        match dev {
            Device::Gpu(_) if !g.unsized_ => {
                let ledger = ledgers.entry(dev).or_default();
                let before = ledger.peak();
                for &(phase, vram) in &g.adds {
                    ledger.add(phase, vram);
                }
                b.alloc(dev, ledger.peak() - before);
            }
            // An unsized group is charged nothing on the card - the
            // documented contract, and safe only because unsized groups are
            // placed last and only onto a card with `UNSIZED_FLOOR` to spare.
            Device::Gpu(_) => {}
            // The host tier holds a spilled group's weights, not just its
            // declared staging bytes (`MemCost::resident_on`); NPU bytes are
            // unphased, so an NPU group is charged its full sum, as before.
            _ => b.alloc(dev, g.cost.resident_on(dev)),
        }
        // Host RAM is charged wherever the group landed, because that is what
        // `MemCost::ram` means. Not double-charged when the group IS on the
        // host tier - `resident_on(Cpu)` already covers it.
        if host_declared && g.cost.ram > 0 && dev != Device::Cpu {
            b.alloc(Device::Cpu, g.cost.ram);
        }
        used.insert(dev);
        for &m in &g.members {
            placed[m] = Some(dev);
        }
    }

    Ok(Placement::new(
        parts.iter().zip(placed).map(|(p, d)| (p.name.clone(), d.expect("every part placed or returned"))).collect(),
    ))
}
