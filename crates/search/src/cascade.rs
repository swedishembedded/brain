// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The evaluator ladder: cheap checks first, the expensive one only for what
//! survived them.
//!
//! Search quality is roughly
//!
//! ```text
//! proposal quality  x  evaluation quality
//! ```
//!
//! and the second factor is the one usually left broken. Repeated sampling
//! raises *coverage* enormously - sample a generator a few hundred times and
//! it finds solutions its one-shot self misses - but that coverage is only
//! worth what the verifier can identify. A brilliant generator with a broken
//! verifier eventually optimises garbage, confidently.
//!
//! The other half is cost. A real verifier is expensive by construction: on a
//! game campaign the top rung is a whole episode replayed from the level's own
//! start against the engine. Running it on every candidate is a budget spent
//! proving that obvious rubbish is rubbish. So a candidate climbs:
//!
//! ```text
//! structural check      free
//! does it beat what we already hold      free
//! ...
//! replay it against the real environment      an entire episode
//! ```
//!
//! and only what survives rung `n` is paid for at rung `n+1`.
//!
//! The property that matters is negative - **the expensive rung must not run
//! on a candidate a cheap rung already refused** - which is why it is the
//! first thing this module's tests assert.

/// What a rung decided.
#[derive(Clone, Debug)]
pub enum Verdict {
    Pass,
    /// Refused, saying why. The reason is the diagnostic: a cascade whose top
    /// rung refuses everything for one reason is reporting a defect, not
    /// filtering.
    Reject(String),
}

/// One rung of the ladder.
pub trait Rung<C> {
    fn name(&self) -> &str;
    fn check(&mut self, candidate: &C) -> Verdict;
}

/// Where candidates died, per rung.
#[derive(Clone, Debug)]
pub struct Tally {
    pub name: String,
    /// Candidates that reached this rung.
    pub seen: usize,
    /// Of those, how many it refused.
    pub refused: usize,
}

/// The ladder.
pub struct Cascade<C> {
    rungs: Vec<Box<dyn Rung<C>>>,
    tally: Vec<Tally>,
    admitted: usize,
}

impl<C> Default for Cascade<C> {
    fn default() -> Cascade<C> {
        Cascade::new()
    }
}

impl<C> Cascade<C> {
    pub fn new() -> Cascade<C> {
        Cascade { rungs: Vec::new(), tally: Vec::new(), admitted: 0 }
    }

    /// Add a rung. Order is the contract: cheapest first, and a caller that
    /// gets it backwards has a cascade that costs exactly what no cascade
    /// would.
    pub fn push(&mut self, rung: Box<dyn Rung<C>>) {
        self.tally.push(Tally { name: rung.name().to_string(), seen: 0, refused: 0 });
        self.rungs.push(rung);
    }

    /// Climb the ladder. An empty cascade admits - a caller who has not
    /// configured a verifier yet gets an unverified search, not a search that
    /// silently finds nothing.
    pub fn admit(&mut self, candidate: &C) -> Verdict {
        for (i, rung) in self.rungs.iter_mut().enumerate() {
            self.tally[i].seen += 1;
            if let Verdict::Reject(why) = rung.check(candidate) {
                self.tally[i].refused += 1;
                return Verdict::Reject(why);
            }
        }
        self.admitted += 1;
        Verdict::Pass
    }

    /// How many candidates cleared every rung.
    pub fn admitted(&self) -> usize {
        self.admitted
    }

    pub fn tally(&self) -> &[Tally] {
        &self.tally
    }
}
