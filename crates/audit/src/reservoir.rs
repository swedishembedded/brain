// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A bounded sample of what the reader has already learned, to mix back in.
//!
//! Rehearsal is not optional here, and that is a measured position rather
//! than a preference. On this engine's own continual-learning study the
//! composition WITHOUT a rehearsal pool did not accumulate capability at
//! all, finishing below the untrained base on the same probes, while the
//! teacher-forced regime mixed half and half with one did, across two seeds.
//! A design that refused replay on principle would be refusing the only
//! thing that was shown to work.
//!
//! What has to be bounded is the COST, not the practice. So:
//!
//! **Only promoted episodes enter.** Rehearsing something the gate refused
//! would spend the reader's capacity on material it already decided against,
//! and would let a rejected episode influence every later cycle through the
//! back door. The rule lives here rather than at each call site, so there is
//! one place it can be got right.
//!
//! **Identity is content.** [`crate::stream::EpisodeId`] is the digest of the
//! episode's own text, so the same passage arriving twice under two names
//! cannot occupy two slots and cannot be rehearsed at double weight. That is
//! the whole reason identity was defined that way.
//!
//! **The sample is uniform over everything promoted, not over what is
//! recent.** Reservoir sampling (Vitter's Algorithm R) keeps each promoted
//! episode with equal probability however long ago it arrived, which is what
//! a retention mix needs: a reservoir biased towards the recent rehearses
//! precisely the material least at risk of being forgotten. The two easy
//! wrong implementations - keep the first `cap`, keep the last `cap` - are
//! excluded by a test on the distribution rather than by inspection.
//!
//! **Disabled is a choice, not a zero.** A run with rehearsal off is a
//! control arm, and it reports itself as one; a cap that happened to be zero
//! would be indistinguishable from an arm nobody meant to run.

use std::collections::BTreeSet;

use data::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::stream::EpisodeId;
use crate::triage::Verdict;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReservoirConfig {
    /// Most episodes held at once.
    pub cap: usize,
    /// Off means the no-rehearsal control arm, and says so.
    pub enabled: bool,
    /// Seeds the retention decisions, so a run reproduces.
    pub seed: u64,
}

impl Default for ReservoirConfig {
    fn default() -> Self {
        ReservoirConfig { cap: 256, enabled: true, seed: 0 }
    }
}

/// One promoted episode's trainable rows, kept for rehearsal.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub episode: EpisodeId,
    pub rows: Vec<String>,
}

/// Why an offered episode was or was not kept. Reported rather than
/// returned as a bare bool, so a ledger can say what the reservoir did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Offered {
    /// Stored, with room to spare.
    Admitted,
    /// Stored, displacing a uniformly chosen resident.
    Replaced,
    /// Not stored: the reservoir is full and the coin came up against it.
    /// This is the normal case once a run is long, and is not a failure.
    Passed,
    /// Already held, by content. Not stored again and not counted twice.
    Duplicate,
    /// The gate did not promote this episode.
    NotPromoted,
    /// Rehearsal is off for this run.
    Disabled,
}

/// A bounded, uniform sample of promoted episodes.
#[derive(Clone, Debug)]
pub struct Reservoir {
    cfg: ReservoirConfig,
    items: Vec<Entry>,
    held: BTreeSet<EpisodeId>,
    seen: u64,
    rng: Rng,
}

impl Reservoir {
    pub fn new(cfg: ReservoirConfig) -> Reservoir {
        Reservoir { cfg, items: Vec::new(), held: BTreeSet::new(), seen: 0, rng: Rng::new(cfg.seed) }
    }

    /// Offer an episode and its trainable rows. Only a promoted one is ever
    /// kept; everything else is reported and dropped.
    pub fn offer(&mut self, episode: &EpisodeId, rows: &[String], verdict: &Verdict) -> Offered {
        if !self.cfg.enabled {
            return Offered::Disabled;
        }
        if !matches!(verdict, Verdict::Promoted(_)) {
            return Offered::NotPromoted;
        }
        if self.held.contains(episode) {
            // Deliberately does NOT advance `seen`. The denominator is how
            // many DISTINCT promotions the sample is drawn from; counting a
            // duplicate would lower every later episode's retention
            // probability for an episode that was never a candidate.
            return Offered::Duplicate;
        }

        // Vitter's Algorithm R. `seen` counts distinct promotions; the k-th
        // is kept with probability cap/k, displacing a uniformly chosen
        // resident, which leaves every promotion equally likely to be held
        // however long ago it arrived.
        self.seen += 1;
        let entry = Entry { episode: episode.clone(), rows: rows.to_vec() };
        if self.items.len() < self.cfg.cap {
            self.held.insert(episode.clone());
            self.items.push(entry);
            return Offered::Admitted;
        }
        if self.cfg.cap == 0 {
            return Offered::Passed;
        }
        let k = self.rng.next_u64() % self.seen;
        if (k as usize) < self.cfg.cap {
            let victim = self.items[k as usize].episode.clone();
            self.held.remove(&victim);
            self.held.insert(episode.clone());
            self.items[k as usize] = entry;
            Offered::Replaced
        } else {
            Offered::Passed
        }
    }

    /// Rehearsal rows for one cycle, drawn without replacement.
    ///
    /// Takes its own seed rather than advancing the reservoir's, so a draw is
    /// a pure function of the contents and the seed: the same cycle asked
    /// twice gets the same mix, and asking does not change what is retained.
    pub fn draw(&self, rows: usize, seed: u64) -> Vec<&str> {
        if !self.cfg.enabled || rows == 0 {
            return Vec::new();
        }
        let mut all: Vec<&str> = self.items.iter().flat_map(|e| e.rows.iter().map(String::as_str)).collect();
        // Partial Fisher-Yates from a seed of the caller's choosing, so the
        // draw neither consumes nor depends on the reservoir's own generator
        // and asking twice gives the same answer.
        let mut rng = Rng::new(seed);
        let take = rows.min(all.len());
        for i in 0..take {
            let j = i + (rng.next_u64() % (all.len() - i) as u64) as usize;
            all.swap(i, j);
        }
        all.truncate(take);
        all
    }

    pub fn enabled(&self) -> bool {
        self.cfg.enabled
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Promoted episodes offered so far, whether or not they were kept. The
    /// denominator the retention probability is against.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Rows currently held, across every entry.
    pub fn rows(&self) -> usize {
        self.items.iter().map(|e| e.rows.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triage::Unstructured;
    use promote::gate::{gate, GateConfig, GateInput};

    fn ep(i: usize) -> EpisodeId {
        EpisodeId::of(&format!("episode {i}"))
    }

    fn rows(i: usize) -> Vec<String> {
        (0..3).map(|k| format!("episode {i} row {k}")).collect()
    }

    /// A real promote, built from the gate rather than hand-constructed, so
    /// the reservoir's admission rule is tied to the actual decision type.
    fn promoted() -> Verdict {
        let mut c = vec![0.0f64; 20];
        for x in c.iter_mut().take(14) {
            *x = 1.0;
        }
        let i = vec![0.0f64; 20];
        let input = GateInput {
            candidate_scores: &c,
            incumbent_scores: &i,
            anchor_candidate: 0.9,
            anchor_incumbent: 0.9,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
            anchor_blocks: &[],
        };
        let report = gate(&input, &GateConfig { min_effect_size: 0.15, ..GateConfig::default() });
        Verdict::Promoted(report)
    }

    fn cfg(cap: usize, seed: u64) -> ReservoirConfig {
        ReservoirConfig { cap, enabled: true, seed }
    }

    /// The cap is the whole point: a reservoir that grew with the stream
    /// would put the reader back on the cost curve it is trying to leave.
    #[test]
    fn the_reservoir_fills_to_its_cap_and_never_past_it() {
        let mut r = Reservoir::new(cfg(50, 1));
        for i in 0..2000 {
            r.offer(&ep(i), &rows(i), &promoted());
            assert!(r.len() <= 50, "held {} against a cap of 50 at offer {i}", r.len());
        }
        assert_eq!(r.len(), 50, "and it must actually fill, not merely stay under");
        assert_eq!(r.seen(), 2000);
    }

    /// Identity is content, which is exactly so that the same passage
    /// arriving twice cannot be rehearsed at double weight.
    #[test]
    fn the_same_content_offered_twice_does_not_take_a_second_slot() {
        let mut r = Reservoir::new(cfg(50, 2));
        assert_eq!(r.offer(&ep(1), &rows(1), &promoted()), Offered::Admitted);
        assert_eq!(r.offer(&ep(1), &rows(1), &promoted()), Offered::Duplicate);
        assert_eq!(r.len(), 1);
        assert_eq!(r.seen(), 1, "a duplicate must not move the denominator either, or the sampling skews");
    }

    /// Rehearsing something the gate refused would let a rejected episode
    /// influence every later cycle through the back door.
    #[test]
    fn only_a_promoted_episode_is_ever_kept() {
        let mut r = Reservoir::new(cfg(50, 3));
        let refused = [
            Verdict::Unstructured(Unstructured::TooShort { chars: 1, floor: 64 }),
            Verdict::AlreadyKnown { loss: 0.1, below: 0.2 },
            Verdict::OutOfReach { loss: 9.0, above: 4.0 },
            Verdict::TooSmallToGate { probes: 3, floor: 12 },
        ];
        for (i, v) in refused.iter().enumerate() {
            assert_eq!(r.offer(&ep(i), &rows(i), v), Offered::NotPromoted);
        }
        assert!(r.is_empty());
        assert_eq!(r.seen(), 0, "only promotions count towards the sample");
    }

    /// The property that separates reservoir sampling from the two easy
    /// wrong answers. A reservoir biased to the recent rehearses precisely
    /// the material least at risk of being forgotten; one biased to the
    /// oldest never rehearses anything learned since.
    #[test]
    fn the_sample_is_uniform_over_the_whole_run_not_biased_to_either_end() {
        const N: usize = 2000;
        const CAP: usize = 50;
        let mut indices: Vec<f64> = Vec::new();
        for seed in 0..8u64 {
            let mut r = Reservoir::new(cfg(CAP, seed));
            for i in 0..N {
                r.offer(&ep(i), &rows(i), &promoted());
            }
            // Recover each held episode's position in the stream.
            for e in &r.items {
                let pos = (0..N).position(|i| ep(i) == e.episode).expect("held entries come from the stream");
                indices.push(pos as f64);
            }
        }
        let mean = indices.iter().sum::<f64>() / indices.len() as f64;
        assert!(
            (900.0..=1100.0).contains(&mean),
            "mean held position {mean} should sit near {} for a uniform sample; \
             keeping the FIRST {CAP} would give about {}, keeping the LAST {CAP} about {}",
            N / 2,
            CAP / 2,
            N - CAP / 2
        );
    }

    /// A draw has to be reproducible, and must not change what is retained -
    /// otherwise asking the reservoir a question would alter the experiment.
    #[test]
    fn a_draw_is_a_pure_function_of_the_contents_and_its_seed() {
        let mut r = Reservoir::new(cfg(20, 4));
        for i in 0..40 {
            r.offer(&ep(i), &rows(i), &promoted());
        }
        let before = r.items.clone();
        let a = r.draw(12, 7).into_iter().map(str::to_string).collect::<Vec<_>>();
        let b = r.draw(12, 7).into_iter().map(str::to_string).collect::<Vec<_>>();
        assert_eq!(a, b, "the same seed must draw the same mix");
        assert_eq!(a.len(), 12);
        assert_eq!(r.items, before, "drawing must not disturb what is retained");

        let c = r.draw(12, 8).into_iter().map(str::to_string).collect::<Vec<_>>();
        assert_ne!(a, c, "a different seed must draw a different mix");

        let all = r.draw(9999, 7);
        assert_eq!(all.len(), r.rows(), "asking for more rows than are held yields what is held, not a panic");
    }

    /// The no-rehearsal arm. It has to be a run the reader can report as
    /// such, not a cap that silently happened to be zero.
    #[test]
    fn a_disabled_reservoir_keeps_nothing_and_reports_itself_as_the_control_arm() {
        let mut r = Reservoir::new(ReservoirConfig { cap: 256, enabled: false, seed: 5 });
        assert_eq!(r.offer(&ep(1), &rows(1), &promoted()), Offered::Disabled);
        assert!(r.is_empty());
        assert!(!r.enabled(), "the arm must be visible in the reservoir's own state");
        assert!(r.draw(10, 1).is_empty());
    }
}
