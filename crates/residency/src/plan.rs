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

use std::collections::HashSet;

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
}

/// The cost an unsized part is *placed* by: the smallest possible non-zero
/// device footprint. It routes through exactly the same
/// [`pick_device`] class preference and most-free-wins rule every sized part
/// uses (rather than a second, parallel notion of "the emptiest card"), and
/// falls back to the host tier on a machine with no accelerator - while being
/// small enough that it can never be the reason a plan is refused.
const UNSIZED_PROBE: MemCost = MemCost { vram: 1, ram: 0, npu: 0, mapped: 0 };

impl Part {
    /// A part of known size, unconstrained.
    pub fn new(name: impl Into<String>, cost: MemCost) -> Part {
        Part { name: name.into(), cost, affinity: Affinity::Any, unsized_: false }
    }
    /// A part whose size is not known yet - the "just give me a card" case
    /// every bare `Gpu::new` takes. It is charged nothing, so it lands on the
    /// emptiest accelerator and does not distort what follows it.
    pub fn unsized_(name: impl Into<String>) -> Part {
        Part { name: name.into(), cost: MemCost::new(0, 0), affinity: Affinity::Any, unsized_: true }
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

    // 2. Each group's summed cost, and whether any member asked to be Apart.
    struct Group {
        root: usize,
        members: Vec<usize>,
        cost: MemCost,
        unsized_: bool,
        apart: bool,
    }
    let mut groups: Vec<Group> = roots
        .iter()
        .map(|&r| {
            let members: Vec<usize> = (0..parts.len()).filter(|&i| root(&mut group.clone(), i) == r).collect();
            let mut cost = MemCost::new(0, 0);
            let mut unsized_ = true;
            let mut apart = false;
            for &m in &members {
                cost.vram += parts[m].cost.vram;
                cost.ram += parts[m].cost.ram;
                cost.npu += parts[m].cost.npu;
                unsized_ &= parts[m].unsized_;
                apart |= parts[m].affinity == Affinity::Apart;
            }
            Group { root: r, members, cost, unsized_, apart }
        })
        .collect();
    // Largest first: the part with the least choice of homes must choose
    // before the parts that fit anywhere take one.
    groups.sort_by_key(|g| (std::cmp::Reverse(g.cost.vram.max(g.cost.ram).max(g.cost.npu)), g.root));

    // 3. Place each group, charging it so the groups that follow see what is
    //    actually left.
    let mut placed: Vec<Option<Device>> = vec![None; parts.len()];
    let mut used: HashSet<Device> = HashSet::new();
    for g in &groups {
        // What the group is placed BY (see `UNSIZED_PROBE`); what it is
        // charged is always its own cost, which for an unsized group is zero.
        let probe = if g.unsized_ { UNSIZED_PROBE } else { g.cost };
        let dev = if g.apart {
            pick_device(&probe, &b, &used).or_else(|| pick_device(&probe, &b, &HashSet::new()))
        } else {
            pick_device(&probe, &b, &HashSet::new())
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
                cost: g.cost,
                free,
                placed: parts.iter().zip(&placed).filter_map(|(p, d)| d.map(|d| (p.name.clone(), d))).collect(),
            });
        };
        b.alloc(dev, g.cost.on(dev));
        used.insert(dev);
        for &m in &g.members {
            placed[m] = Some(dev);
        }
    }

    Ok(Placement::new(
        parts.iter().zip(placed).map(|(p, d)| (p.name.clone(), d.expect("every part placed or returned"))).collect(),
    ))
}
