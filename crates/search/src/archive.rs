// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A quality-diversity archive: the best-known way to reach each *kind* of
//! situation, rather than the best N ways to reach the one that scores
//! highest.
//!
//! ## Why not a leaderboard
//!
//! Keep the top three of
//!
//! ```text
//! A 94 greedy    B 93 greedy    C 92 greedy    D 90 greedy
//! E 89 dynamic programming       F 87 decomposition
//! G 84 strange recursive trick
//! ```
//!
//! and the archive holds three variations of one idea. `G` - the one whose
//! mutation might reach 130 - is the first thing deleted, and A through D are
//! sitting on a local optimum the search can no longer leave. An archive
//! keyed by *what kind of solution this is* keeps one of each instead, and
//! every one of them is a starting point the search can set off from again.
//!
//! ## What a niche is, and what quality is
//!
//! They are deliberately different things, and conflating them is the mistake
//! this type exists to prevent:
//!
//! - the [`Niche`] says **what situation this is** - where the run stands and
//!   what it has achieved. It is a coordinate, not a score.
//! - the [`Worth`] says **how good this way of being there is**: how much was
//!   achieved, and at what `cost` - which for a speedrun is elapsed time.
//!
//! Achievement dominates: a run that took more of the level apart is ahead of
//! one that did less, however quickly. Time only ever breaks a tie - but
//! because the niche already encodes most of what was achieved, ties are the
//! normal case, and the archive therefore drives the time down everywhere at
//! once. That is the whole mechanism by which a campaign turns a solution into
//! a fast solution, and it falls out of the ordering rather than needing a
//! separate optimisation pass.
//!
//! ## The tolerance
//!
//! Two runs whose achievement differs by a float hair are the same
//! achievement. Without a tolerance the comparison is `>`, a score that
//! wobbles by a thousandth between two runs never admits the faster one, and
//! the time never comes down at all - the archive silently degrades into the
//! coverage map it was supposed to improve on.

use data::rng::Rng;
use serde::{Deserialize, Serialize};

/// Where a candidate sits in behaviour space.
///
/// A coordinate, not a score. Integer per axis, because a niche is a discrete
/// bin: an environment decides its own resolution (which grid square, which
/// keys held, how many of the level's monsters are left) and the archive keeps
/// one elite per distinct tuple.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Niche(Box<[i32]>);

impl Niche {
    pub fn new(parts: &[i32]) -> Niche {
        Niche(parts.into())
    }

    pub fn parts(&self) -> &[i32] {
        &self.0
    }
}

impl std::fmt::Display for Niche {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, p) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ":")?;
            }
            write!(f, "{p}")?;
        }
        Ok(())
    }
}

/// How good one way of reaching a niche is.
///
/// `reached` is how much was achieved, on whatever scale the caller scores a
/// run on - higher is better. `cost` is what it took to get there in units the
/// caller cares about minimising - lower is better, and for a speedrun it is
/// elapsed game time.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Worth {
    pub reached: f32,
    pub cost: u64,
}

impl Worth {
    pub fn new(reached: f32, cost: u64) -> Worth {
        Worth { reached, cost }
    }

    /// Strictly better than `other`, treating achievements within `tol` as
    /// equal so that `cost` decides between them.
    pub fn better_than(&self, other: &Worth, tol: f32) -> bool {
        if self.reached > other.reached + tol {
            return true;
        }
        if other.reached > self.reached + tol {
            return false;
        }
        self.cost < other.cost
    }
}

/// The best known way to reach one niche.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Elite<C> {
    pub niche: Niche,
    pub worth: Worth,
    /// Whatever the caller needs to get back here - a trajectory, a seed, a
    /// program. The archive never looks inside it.
    pub what: C,
    /// How many times this cell has been set off FROM. Feeds the selection
    /// weight, so a cell that has been explored from often loses priority to
    /// one that has not.
    pub visits: u32,
    /// How many times this cell's contents have been REPLACED by a better
    /// way of reaching it.
    ///
    /// For callers that chain: a trajectory recorded as "resume at cell P,
    /// then do these things" is only meaningful against the P that was there
    /// when it was recorded. Improve P and every trajectory hanging off it
    /// describes a continuation of a state that no longer exists - it will
    /// not replay, and without a way to notice that, a caller reconstructs a
    /// plausible-looking sequence that silently does something else.
    ///
    /// A counter rather than a timestamp, because the question is only ever
    /// "is this the same one I saw", and a counter answers it with no clock.
    pub generation: u32,
    /// The host-side resource this cell's state occupies - a snapshot slot on
    /// a game engine, an index into a pool of saved states. The archive owns
    /// its allocation so that an eviction frees a slot the next cell reuses:
    /// a host has a finite number of them and leaking one is a cell that can
    /// never be returned to.
    pub slot: usize,
}

/// What happened to an offered candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// A niche nothing had reached before.
    Fresh,
    /// A better way to reach a niche already held.
    Improved,
    /// Not better than what is already there.
    Rejected,
}

/// One elite per niche, with a bounded number of host slots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Archive<C> {
    // A LIST on the wire, not a map. A niche is a tuple of integers, and JSON
    // has no key type but string - serialising the map directly fails at
    // `key must be a string`, and the obvious repair (stringify the niche)
    // invents a second, lossy spelling of a coordinate that has to parse back
    // exactly. Each elite already carries its own niche, so a list is both
    // smaller and the only spelling there is.
    #[serde(with = "cells_as_list", bound(serialize = "C: Serialize", deserialize = "C: Deserialize<'de>"))]
    cells: std::collections::HashMap<Niche, Elite<C>>,
    capacity: usize,
    tol: f32,
    /// The lowest slot never yet handed out. Slots below it are either live or
    /// on `free`.
    next_slot: usize,
    /// Slots returned by evictions, reused before `next_slot` grows.
    free: Vec<usize>,
}

impl<C> Archive<C> {
    /// `capacity` is how many cells the host can hold state for at once -
    /// on a game engine, its snapshot slot count.
    pub fn new(capacity: usize, tol: f32) -> Archive<C> {
        Archive {
            cells: std::collections::HashMap::new(),
            capacity,
            tol,
            next_slot: 0,
            free: Vec::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn get(&self, niche: &Niche) -> Option<&Elite<C>> {
        self.cells.get(niche)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Elite<C>> {
        self.cells.values()
    }

    /// Offer a candidate. The return says whether the archive now holds it and
    /// on what grounds, so a caller can credit the operator that produced it.
    pub fn offer(&mut self, niche: Niche, worth: Worth, what: C) -> Admission {
        if let Some(held) = self.cells.get_mut(&niche) {
            if !worth.better_than(&held.worth, self.tol) {
                return Admission::Rejected;
            }
            // The slot and the visit count belong to the CELL, not to the
            // trajectory that currently holds it. Handing an improved elite a
            // fresh slot leaks the old one, and resetting its visits tells the
            // selection rule this is unexplored ground when it is not.
            held.worth = worth;
            held.what = what;
            held.generation += 1;
            return Admission::Improved;
        }
        let slot = match self.free.pop() {
            Some(s) => s,
            None if self.next_slot < self.capacity => {
                let s = self.next_slot;
                self.next_slot += 1;
                s
            }
            // Full. Take the slot of whichever cell the selection rule is
            // least likely to draw anyway - an archive that cannot grow is a
            // search that has finished, and it stops silently: the cells keep
            // being FOUND and simply are not kept.
            None => match self.weakest() {
                Some(weak) => {
                    let freed = self.cells.remove(&weak).map(|e| e.slot);
                    match freed {
                        Some(s) => s,
                        None => return Admission::Rejected,
                    }
                }
                None => return Admission::Rejected,
            },
        };
        self.cells
            .insert(niche.clone(), Elite { niche, worth, what, visits: 0, generation: 0, slot });
        Admission::Fresh
    }

    /// Drop a cell, returning its host slot to the pool.
    ///
    /// For the case where the archive and the host disagree: the archive has
    /// filed a cell the host then refused to hold state for, so returning to
    /// it would restore some OTHER cell's state and explore the wrong place
    /// while reporting the right one. Dropping it is the honest repair.
    pub fn remove(&mut self, niche: &Niche) -> Option<Elite<C>> {
        let gone = self.cells.remove(niche)?;
        self.free.push(gone.slot);
        Some(gone)
    }

    /// The cell with the lowest selection weight - what an eviction takes.
    fn weakest(&self) -> Option<Niche> {
        let top = self.top_reached();
        self.cells
            .values()
            .map(|e| (e.niche.clone(), weight(e, top)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(n, _)| n)
    }

    fn top_reached(&self) -> f32 {
        self.cells.values().map(|e| e.worth.reached).fold(f32::MIN_POSITIVE, f32::max)
    }

    /// Draw a cell to set off from, and count the visit.
    ///
    /// Weighted, never uniform: uniform selection spends the whole budget on
    /// the hundreds of cells in the opening corridor, which are the most
    /// numerous precisely because they are the easiest to reach.
    pub fn pick(&mut self, rng: &mut Rng) -> Option<&Elite<C>> {
        if self.cells.is_empty() {
            return None;
        }
        let top = self.top_reached();
        let weights: Vec<(Niche, f64)> =
            self.cells.values().map(|e| (e.niche.clone(), weight(e, top) as f64)).collect();
        let total: f64 = weights.iter().map(|(_, w)| *w).sum();
        let mut u = rng.next_f64() * total;
        let mut chosen = weights[weights.len() - 1].0.clone();
        for (n, w) in &weights {
            if u < *w {
                chosen = n.clone();
                break;
            }
            u -= *w;
        }
        let cell = self.cells.get_mut(&chosen)?;
        cell.visits += 1;
        Some(cell)
    }

    /// The single furthest-along cell: most achieved, and among those, fastest.
    pub fn best(&self) -> Option<&Elite<C>> {
        self.cells.values().reduce(|a, b| if b.worth.better_than(&a.worth, self.tol) { b } else { a })
    }

    /// Distinct values held on each niche axis.
    ///
    /// How a campaign says whether the search is still finding new KINDS of
    /// situation or has settled into refining what it has - which is the
    /// question "is the archive still growing" cannot answer, since a search
    /// stuck in one room keeps producing new position cells forever.
    pub fn coverage(&self) -> Vec<usize> {
        let axes = self.cells.keys().map(|n| n.parts().len()).max().unwrap_or(0);
        (0..axes)
            .map(|i| {
                let seen: std::collections::HashSet<i32> =
                    self.cells.keys().filter_map(|n| n.parts().get(i).copied()).collect();
                seen.len()
            })
            .collect()
    }
}

/// A `HashMap<Niche, _>` keyed by a tuple, on the wire as the list of its
/// values. See the field's own comment for why.
mod cells_as_list {
    use super::{Elite, Niche};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<C, S>(
        cells: &std::collections::HashMap<Niche, Elite<C>>,
        s: S,
    ) -> Result<S::Ok, S::Error>
    where
        C: Serialize,
        S: Serializer,
    {
        // Sorted, so two archives holding the same cells produce the same
        // bytes and a diff of two campaign artifacts is readable.
        let mut all: Vec<&Elite<C>> = cells.values().collect();
        all.sort_by(|a, b| a.niche.cmp(&b.niche));
        all.serialize(s)
    }

    pub fn deserialize<'de, C, D>(
        d: D,
    ) -> Result<std::collections::HashMap<Niche, Elite<C>>, D::Error>
    where
        C: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let all: Vec<Elite<C>> = Vec::deserialize(d)?;
        Ok(all.into_iter().map(|e| (e.niche.clone(), e)).collect())
    }
}

/// How likely a cell is to be drawn, and to survive an eviction.
///
/// Two measures at once, and both are needed. `1/sqrt(visits + 1)` is
/// Go-Explore's own weight (Ecoffet et al., *First return, then explore*,
/// Extended Data Table 1) and favours what has rarely been set off from, which
/// is what stops the search settling into one corner. On its own it is blind
/// to how much a cell has to offer: a spot at the level's front door with
/// nothing achieved is drawn exactly as often as one deep in with most of the
/// level cleared, and only the second can lead anywhere new.
///
/// So the achievement multiplies it, normalised against the best cell held.
/// The `1.0 +` keeps a cell that has achieved nothing in the draw rather than
/// cutting it off - that is where a search which has not started yet has to
/// begin, and a weight of zero would make it unreachable forever.
fn weight<C>(e: &Elite<C>, top: f32) -> f32 {
    let worth = 1.0 + (e.worth.reached / top).clamp(0.0, 1.0);
    worth / ((e.visits as f32) + 1.0).sqrt()
}

impl<C: Serialize + for<'de> Deserialize<'de>> Archive<C> {
    /// The archive is the durable artifact of a campaign - the weights are a
    /// lossy compression of it and can be rebuilt from it, while a trajectory
    /// that solved a level cannot be rebuilt from anything.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("archive will not serialise: {e}"))
    }

    pub fn from_json(text: &str) -> Result<Archive<C>, String> {
        serde_json::from_str(text).map_err(|e| format!("archive will not parse: {e}"))
    }
}
