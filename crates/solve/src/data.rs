// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Labels, from walking away from the goal.
//!
//! The generator is the reason this method needs no solver. Every example is
//! produced by taking a state that is `d` moves from the goal BY
//! CONSTRUCTION, and naming the move that undoes the last of those `d`. The
//! label is exact, it costs `d` move applications, and it is available at any
//! depth - which is what removes the ceiling that an exact planner imposes.

use crate::StateSpace;

/// How far from the goal to walk, and how to seed it.
#[derive(Clone, Copy, Debug)]
pub struct Walk {
    /// The deepest walk to take. Examples are drawn with EQUAL weight at each
    /// depth `1..=depth`, which is not what an unbalanced generator does and
    /// matters more than it looks: a walk of length `d` passes through one
    /// state at every distance below `d`, so harvesting whole walks buys a
    /// curriculum nobody chose, heavily weighted to the shallow end that the
    /// policy masters first and learns nothing more from.
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
    /// The walk depth each row was drawn at, for per-depth reporting.
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

/// Draw `rows` examples, balanced across `1..=walk.depth`.
pub fn batch<S: StateSpace>(space: &S, walk: &Walk, rows: usize, rng: &mut Rng) -> Batch {
    let width = space.feature_len();
    let n_moves = space.moves();
    let mut features = vec![0.0f32; rows * width];
    let mut labels = Vec::with_capacity(rows);
    let mut depths = Vec::with_capacity(rows);

    for r in 0..rows {
        // Equal weight per depth, rather than per walk.
        let d = 1 + rng.below(walk.depth);
        let mut s = space.goal();
        let mut last: Option<usize> = None;
        let mut taken = 0usize;
        while taken < d {
            let m = rng.below(n_moves);
            if last.is_some_and(|l| space.redundant(l, m)) {
                continue;
            }
            s = space.apply(&s, m);
            last = Some(m);
            taken += 1;
        }
        let m = last.expect("a walk of depth >= 1 took a move");
        space.write_features(&s, &mut features[r * width..(r + 1) * width]);
        labels.push(space.inverse(m) as u32);
        depths.push(d as u32);
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
