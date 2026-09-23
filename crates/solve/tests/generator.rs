// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the label generator has to guarantee, on spaces small enough to check
//! exactly.
//!
//! Three properties, and a run is worthless without any of them. The labels
//! have to LEAD HOME, or training descends toward a policy that does not. The
//! batch has to stay BALANCED across walk depths, or the deep end - the only
//! end that decides whether a rollout finishes - is drowned by the shallow
//! states every walk passes through, and no held-out number drawn the same way
//! will say so. And the whole thing has to be REPRODUCIBLE from its seed, or a
//! measured difference between two runs is not evidence about either of them.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::Instant;

use solve::data::{batch, inverse_is_an_inverse, Rng};
use solve::{StateSpace, Walk};

/// A free group on `generators` generators: the state IS the reduced word
/// that reached it.
///
/// That is the point of using it here. Nothing else has to be trusted to know
/// what walk a state came from - the trajectory is readable off the state, so
/// "the label steps back onto the walk's previous state" is an exact
/// assertion rather than an approximation of one. With the default
/// redundancy rule (never undo the last move) a word never cancels, so its
/// length is exactly the number of moves walked.
struct Word {
    generators: usize,
    /// Longest word the feature vector can hold. Walks must not exceed it.
    cap: usize,
}

impl Word {
    fn new(generators: usize, cap: usize) -> Word {
        Word { generators, cap }
    }

    /// Read a state back out of its features, which is only possible because
    /// the encoding carries the whole word and not a summary of it.
    fn read_features(&self, f: &[f32]) -> Vec<u8> {
        let mut w = Vec::new();
        for slot in f.chunks(self.moves()) {
            match slot.iter().position(|&v| v == 1.0) {
                Some(g) => w.push(g as u8),
                None => break,
            }
        }
        w
    }
}

impl StateSpace for Word {
    type State = Vec<u8>;

    fn moves(&self) -> usize {
        2 * self.generators
    }

    fn goal(&self) -> Vec<u8> {
        Vec::new()
    }

    fn apply(&self, s: &Vec<u8>, m: usize) -> Vec<u8> {
        let mut w = s.clone();
        if w.last().is_some_and(|&t| t as usize == self.inverse(m)) {
            w.pop();
        } else {
            w.push(m as u8);
        }
        w
    }

    fn inverse(&self, m: usize) -> usize {
        m ^ 1
    }

    fn feature_len(&self) -> usize {
        self.cap * self.moves()
    }

    fn write_features(&self, s: &Vec<u8>, out: &mut [f32]) {
        assert!(s.len() <= self.cap, "a word longer than the feature vector");
        let k = self.moves();
        for (i, &g) in s.iter().enumerate() {
            out[i * k + g as usize] = 1.0;
        }
    }
}

/// A 54-element permutation puzzle shaped like the cube's COST: six faces,
/// three quarter turns each, every turn five disjoint 4-cycles, a 54x6 one-hot
/// feature vector.
///
/// It exists to price the generator against a realistic state, and it counts
/// the move applications it is asked for - which is the mechanism the
/// whole-walk harvest improves, and unlike a wall clock it cannot be
/// perturbed by whatever else the machine is running. The feature map mirrors
/// the cube's for width and write cost only; nothing here is trained on it.
struct Perm54 {
    turn: Vec<[u8; 54]>,
    applies: Cell<usize>,
}

fn compose(a: &[u8; 54], b: &[u8; 54]) -> [u8; 54] {
    std::array::from_fn(|i| a[b[i] as usize])
}

/// Five disjoint 4-cycles over 20 of the 54 positions, as a face turn is.
fn face_turn(face: usize) -> [u8; 54] {
    let mut p: [u8; 54] = std::array::from_fn(|i| i as u8);
    let mut pool: Vec<u8> = (0..54).collect();
    let mut rng = Rng::new(0xFACE0 + face as u64);
    for i in (1..pool.len()).rev() {
        pool.swap(i, rng.below(i + 1));
    }
    for quad in pool[..20].chunks(4) {
        for (k, &pos) in quad.iter().enumerate() {
            p[pos as usize] = quad[(k + 1) % 4];
        }
    }
    p
}

impl Perm54 {
    fn new() -> Perm54 {
        let mut turn = Vec::with_capacity(18);
        for face in 0..6 {
            let base = face_turn(face);
            let mut p = base;
            // Quarter turns 1, 2, 3 of this face, in that order: a move index
            // is `face * 3 + (quarters - 1)`, which is what `inverse` and
            // `redundant` below do arithmetic on.
            for _ in 0..3 {
                turn.push(p);
                p = compose(&base, &p);
            }
        }
        Perm54 { turn, applies: Cell::new(0) }
    }

    fn reset(&self) {
        self.applies.set(0);
    }

    fn applies(&self) -> usize {
        self.applies.get()
    }
}

impl StateSpace for Perm54 {
    type State = [u8; 54];

    fn moves(&self) -> usize {
        self.turn.len()
    }

    fn goal(&self) -> [u8; 54] {
        std::array::from_fn(|i| i as u8)
    }

    fn apply(&self, s: &[u8; 54], m: usize) -> [u8; 54] {
        self.applies.set(self.applies.get() + 1);
        let p = &self.turn[m];
        std::array::from_fn(|i| s[p[i] as usize])
    }

    fn inverse(&self, m: usize) -> usize {
        (m / 3) * 3 + (2 - m % 3)
    }

    fn feature_len(&self) -> usize {
        54 * 6
    }

    fn write_features(&self, s: &[u8; 54], out: &mut [f32]) {
        for (i, &v) in s.iter().enumerate() {
            out[i * 6 + (v % 6) as usize] = 1.0;
        }
    }

    fn redundant(&self, a: usize, b: usize) -> bool {
        a / 3 == b / 3
    }
}

/// The generator this one replaces, kept as the measurement's baseline: one
/// walk to a depth drawn uniformly, one example from its endpoint, the other
/// `d - 1` labelled states on the way there thrown away.
fn endpoint_per_walk<S: StateSpace>(space: &S, walk: &Walk, rows: usize, rng: &mut Rng) -> usize {
    let width = space.feature_len();
    let mut features = vec![0.0f32; rows * width];
    let mut kept = 0usize;
    for r in 0..rows {
        let d = 1 + rng.below(walk.depth);
        let mut s = space.goal();
        let mut last: Option<usize> = None;
        let mut taken = 0usize;
        while taken < d {
            let m = rng.below(space.moves());
            if last.is_some_and(|l| space.redundant(l, m)) {
                continue;
            }
            s = space.apply(&s, m);
            last = Some(m);
            taken += 1;
        }
        space.write_features(&s, &mut features[r * width..(r + 1) * width]);
        kept += 1;
    }
    kept
}

/// The fixtures are under test too: a space whose `inverse` is not an inverse
/// would make every assertion below vacuous.
#[test]
fn the_test_spaces_have_real_inverses() {
    inverse_is_an_inverse(&Word::new(3, 16), 20, 0x51DE).expect("free group inverses");
    inverse_is_an_inverse(&Perm54::new(), 20, 0x51DE).expect("permutation inverses");
}

/// Every walk depth gets the same amount of data, whatever the walks did.
///
/// Whole-walk harvesting visits depth 1 on every single walk and the deepest
/// depth only at the end of one, so "balanced" here has to mean equal counts
/// up to the rounding of `rows / depth` - NOT merely "the deep end is
/// present", which the skew this quota exists to prevent also satisfies.
#[test]
fn every_walk_depth_gets_an_equal_share_of_the_batch() {
    let space = Perm54::new();
    let walk = Walk { depth: 26, seed: 0 };
    // Divisible, remainder, and exactly one example per depth.
    for rows in [2048usize, 2000, 130, 26] {
        let b = batch(&space, &walk, rows, &mut Rng::new(0x5A1AD));
        assert_eq!(b.rows, rows);
        assert_eq!(b.labels.len(), rows);
        assert_eq!(b.depths.len(), rows);
        assert_eq!(b.features.len(), rows * space.feature_len());

        let mut per: BTreeMap<u32, usize> = BTreeMap::new();
        for &d in &b.depths {
            assert!((1..=26).contains(&d), "walk depth {d} out of range");
            *per.entry(d).or_default() += 1;
        }
        assert_eq!(per.len(), walk.depth, "a walk depth got no examples at all: {per:?}");
        let lo = *per.values().min().expect("a non-empty batch");
        let hi = *per.values().max().expect("a non-empty batch");
        assert!(hi - lo <= 1, "walk depths are not balanced at {rows} rows: {per:?}");
    }
}

/// Every emitted pair is a real step home, and the depth-1 pairs land on the
/// goal itself.
///
/// A label that does not step back onto the walk it came from is the failure
/// with no loud symptom: the loss still descends, toward a policy that leads
/// somewhere else.
#[test]
fn every_label_steps_back_onto_the_walk_it_came_from() {
    let space = Word::new(3, 10);
    let walk = Walk { depth: 10, seed: 0 };
    let width = space.feature_len();
    let b = batch(&space, &walk, 250, &mut Rng::new(0xABCD));

    let mut at_the_goal = 0usize;
    for r in 0..b.rows {
        let d = b.depths[r] as usize;
        let state = space.read_features(&b.features[r * width..(r + 1) * width]);
        assert_eq!(state.len(), d, "row {r} is not the state {d} moves of walk reached");

        let back = space.apply(&state, b.labels[r] as usize);
        assert_eq!(back, state[..d - 1], "row {r}: the label left the walk it came from");
        if d == 1 {
            assert!(space.is_goal(&back), "row {r}: a depth-1 label must land on the goal");
            at_the_goal += 1;
        }
    }
    assert_eq!(at_the_goal, 25, "every depth-1 example was checked against the goal");
}

/// A run is reproducible or it is not evidence.
#[test]
fn the_same_seed_generates_the_same_batch() {
    let space = Perm54::new();
    let walk = Walk { depth: 26, seed: 0 };
    let a = batch(&space, &walk, 512, &mut Rng::new(0xD00D));
    let b = batch(&space, &walk, 512, &mut Rng::new(0xD00D));
    assert_eq!(a.features, b.features);
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.depths, b.depths);

    // And a different seed really does move: an identical batch from two
    // seeds would pass the check above for the wrong reason.
    let c = batch(&space, &walk, 512, &mut Rng::new(0xD00E));
    assert_ne!(a.features, c.features);
}

/// What the harvest is for: the walk is paid for once and yields an example
/// per move instead of an example per walk.
///
/// The wall clock is printed (`--nocapture`) and the MECHANISM is asserted.
/// A loaded machine can make any timing ratio lie; move applications per
/// example cannot be perturbed by anything but the generator itself.
#[test]
fn a_harvested_walk_costs_one_move_application_per_example() {
    let space = Perm54::new();
    let walk = Walk { depth: 26, seed: 0 };
    let rows = 2048usize;
    let reps = 20usize;
    let mut rng = Rng::new(1);

    // Warm the allocator so the first pass does not pay for both.
    endpoint_per_walk(&space, &walk, rows, &mut rng);
    batch(&space, &walk, rows, &mut rng);

    space.reset();
    let t = Instant::now();
    for _ in 0..reps {
        endpoint_per_walk(&space, &walk, rows, &mut rng);
    }
    let endpoint_rate = (reps * rows) as f64 / t.elapsed().as_secs_f64();
    let endpoint_applies = space.applies() as f64 / (reps * rows) as f64;

    space.reset();
    let t = Instant::now();
    for _ in 0..reps {
        batch(&space, &walk, rows, &mut rng);
    }
    let harvest_rate = (reps * rows) as f64 / t.elapsed().as_secs_f64();
    let harvest_applies = space.applies() as f64 / (reps * rows) as f64;

    println!(
        "walk depth {}, batch {rows}: endpoint-per-walk {endpoint_rate:.0} examples/s \
         ({endpoint_applies:.2} moves/example), whole-walk harvest {harvest_rate:.0} examples/s \
         ({harvest_applies:.2} moves/example), {:.1}x",
        walk.depth,
        harvest_rate / endpoint_rate
    );

    // A walk of 26 keeps one example per move applied, plus the overflow of
    // the last walk, which cannot reach a second move per example.
    assert!(harvest_applies < 1.2, "{harvest_applies:.2} moves per example is not a harvest");
    // The baseline walks to a uniform depth in 1..=26, so it averages ~13.5.
    assert!(endpoint_applies > 10.0, "the baseline did not behave as measured");
}
