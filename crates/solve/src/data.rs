// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Labels, from walking away from the goal.
//!
//! The generator is the reason this method needs no solver. Every example is
//! produced by taking a state reached by `d` moves from the goal BY
//! CONSTRUCTION, and naming the move that undoes the last of those `d`. The
//! label is exact, it is available at any depth - which is what removes the
//! ceiling that an exact planner imposes - and a whole walk pays for itself
//! many times over: a walk of length `L` passes through `L` states and EVERY
//! one of them carries its own exact label, so the walk is harvested entire
//! rather than for its endpoint alone.

use crate::StateSpace;

/// How far from the goal to walk, and how to seed it.
#[derive(Clone, Copy, Debug)]
pub struct Walk {
    /// The length of every walk, and the deepest bucket examples are kept
    /// for. Examples are emitted with EQUAL weight at each depth `1..=depth`,
    /// which matters more than it looks: a walk of length `depth` passes
    /// through one state at every walk depth below it, so harvesting whole
    /// walks and keeping everything buys a curriculum nobody chose, weighted
    /// `depth:1` towards the shallow end that the policy masters first and
    /// learns nothing more from. [`batch`] harvests the whole walk and
    /// enforces the balance with a per-depth quota instead.
    pub depth: usize,
    pub seed: u64,
}

/// One training batch: dense one-hot features and the move that undoes the
/// walk's last step.
#[derive(Clone, Debug)]
pub struct Batch {
    /// `rows * in_dim`, row-major.
    pub features: Vec<f32>,
    /// `rows`, each an index into `0..moves`.
    pub labels: Vec<u32>,
    /// How many moves of the walk had been taken when the row's state was
    /// reached, for per-depth reporting. An UPPER BOUND on that state's true
    /// distance to the goal and not the distance itself: a walk may wander
    /// back towards the goal, and [`StateSpace::redundant`] only prunes the
    /// pairs that obviously do.
    pub depths: Vec<u32>,
    pub rows: usize,
}

/// A small deterministic generator. Not cryptographic and not meant to be -
/// what it has to be is reproducible from a seed, so a run can be re-run.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Draw `rows` examples, balanced across the walk depths `1..=walk.depth`.
///
/// Every state a walk passes through is labelled exactly, so the walk is
/// harvested WHOLE: `walk.depth` examples for the `walk.depth` move
/// applications the walk already cost, against one example per walk for a
/// generator that keeps only the endpoint. What that costs is balance, and
/// the loss of it would be silent - every walk visits depth 1 and only the
/// end of a walk visits the deepest, so a whole-walk harvest is skewed
/// `walk.depth:1` towards states the policy is already right about, and a
/// held-out split drawn the same way prices that skew in rather than
/// reporting it. So each depth gets an equal QUOTA, the overflow is dropped,
/// and the batch that comes out is balanced whatever the walks did.
///
/// The tradeoff to know about: a batch is now drawn from about
/// `rows / walk.depth` walks instead of `rows` of them, so its rows are
/// CORRELATED - one walk contributes a whole chain of states that differ by
/// one move. The count of examples is unchanged and each label is still
/// exact, but the independent samples behind them are fewer, which shows up
/// as noisier steps rather than as wrong ones. The cheaper example is what
/// pays for it: more steps per unit of time, at any batch size.
///
/// Deterministic in `rng`: the same seed yields the same batch.
pub fn batch<S: StateSpace>(space: &S, walk: &Walk, rows: usize, rng: &mut Rng) -> Batch {
    assert!(walk.depth >= 1, "a walk takes at least one move");
    let depth = walk.depth;
    let width = space.feature_len();
    let n_moves = space.moves();
    let mut features = vec![0.0f32; rows * width];
    let mut labels = Vec::with_capacity(rows);
    let mut depths = Vec::with_capacity(rows);

    // `rows` rarely divides by `depth`. The spare examples go to the DEEP
    // end, which is the end a rollout spends its turns in and the end whose
    // states no shorter walk can reach.
    let mut quota = vec![rows / depth; depth];
    for q in quota.iter_mut().rev().take(rows % depth) {
        *q += 1;
    }
    let mut filled = vec![0usize; depth];

    while labels.len() < rows {
        let mut s = space.goal();
        let mut last: Option<usize> = None;
        for step in 0..depth {
            // Rejection sampling: cheap, because `redundant` rejects a
            // minority of the move set for any space worth training on. A
            // space that rejects ALL of them after some move would spin here
            // forever, so it is named rather than hung on.
            let mut tries = 0usize;
            let m = loop {
                let m = rng.below(n_moves);
                if !last.is_some_and(|l| space.redundant(l, m)) {
                    break m;
                }
                tries += 1;
                assert!(
                    tries < 256,
                    "no move follows {last:?} that `redundant` allows: the walk cannot continue"
                );
            };
            s = space.apply(&s, m);
            last = Some(m);
            if filled[step] == quota[step] {
                // On a valid path home, simply not needed: keeping it is
                // what tilts the batch towards the shallow end.
                continue;
            }
            let r = labels.len();
            space.write_features(&s, &mut features[r * width..(r + 1) * width]);
            labels.push(space.inverse(m) as u32);
            depths.push(step as u32 + 1);
            filled[step] += 1;
        }
    }
    Batch { features, labels, depths, rows }
}

/// Check the contract the label generator depends on: every move has an
/// inverse that really undoes it, from a state reached by an arbitrary walk.
///
/// Worth running against any new [`StateSpace`]. If this fails, the generator
/// still produces labels and training still descends - it descends toward a
/// policy that does not lead home, which is a failure with no loud symptom.
pub fn inverse_is_an_inverse<S: StateSpace>(space: &S, trials: usize, seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    for t in 0..trials {
        let mut s = space.goal();
        for _ in 0..rng.below(12) {
            s = space.apply(&s, rng.below(space.moves()));
        }
        for m in 0..space.moves() {
            let there = space.apply(&s, m);
            let back = space.apply(&there, space.inverse(m));
            if back != s {
                return Err(format!("trial {t}: move {m} then its inverse did not return the state"));
            }
        }
    }
    Ok(())
}
