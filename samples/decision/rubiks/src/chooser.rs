// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Learning WHICH macro to play.
//!
//! The macro library already guarantees a solve: every admissible macro
//! strictly decreases a measure with finitely many values, so any choice at
//! all terminates. What a choice cannot do is make the solve WRONG - and
//! that is what makes this a good place to put a model. It cannot fail
//! catastrophically. It can only make solves shorter or longer, and the
//! difference is measurable in moves.
//!
//! There are a median of 59 admissible macros at each step, so the decision
//! is real. The library currently settles it with a hand-written tiebreak
//! (most cubies home, then least displacement, then fewest moves), which is
//! a reasonable guess and nothing more.
//!
//! What is learned here is cost-to-go in MOVES: how many turns remain from a
//! state, under the policy that generated the data. The labels are free and
//! exact - run a solve, and every state along it is labelled by the number of
//! moves that actually followed it. No bootstrapping, no planner, no reward
//! shaping. At each step the chooser scores every admissible macro by
//! `macro length + V(state after it)` and plays the smallest.

use brain::solve::data::Rng;
use brain::solve::{Config, Net, StateSpace};

use crate::cube::{self, Cube};
use crate::macros::{self, Playbook};
use crate::space::CubeSpace;

/// A solve, and what it is worth as training data.
pub struct Episode {
    /// One feature row per state visited, in order.
    pub features: Vec<f32>,
    /// Moves remaining from each of those states, to the end of the solve.
    pub to_go: Vec<f32>,
    pub moves: usize,
    pub macros: usize,
}

/// Play one cube to the end, recording every state and what it cost from there.
///
/// `explore` picks uniformly among admissible macros rather than following
/// the library's tiebreak. Training on the tiebreak's own trajectories would
/// teach the model to reproduce it, which is the one thing that cannot
/// improve on it; a spread of choices is what lets the regression see that
/// some are cheaper than others.
pub fn episode(space: &CubeSpace, book: &Playbook, start: &Cube, explore: bool, rng: &mut Rng) -> Option<Episode> {
    episode_with(space, book, start, None, if explore { 1.0 } else { 0.0 }, rng)
}

/// As [`episode`], driven by a chooser that is already learning.
///
/// This is the step that turns one round of policy evaluation into policy
/// ITERATION. A value fitted to episodes from a random chooser predicts the
/// cost of behaving randomly, and ranking by it can only be as good as one
/// improvement over random. Regenerating the episodes with the improved
/// chooser and refitting moves the target to the cost of behaving WELL, and
/// each round compounds.
///
/// `epsilon` keeps a fraction of the choices random, because a greedy
/// chooser visits a narrow band of states and a value fitted only there has
/// nothing to say about the alternatives it is asked to rank.
pub fn episode_with(
    space: &CubeSpace,
    book: &Playbook,
    start: &Cube,
    net: Option<&Net>,
    epsilon: f32,
    rng: &mut Rng,
) -> Option<Episode> {
    let width = space.feature_len();
    let mut cube = *start;
    let mut states: Vec<Cube> = Vec::new();
    let mut costs: Vec<usize> = Vec::new();

    while !cube.is_solved() {
        let cands = book.improving_all(&cube);
        if cands.is_empty() {
            return None;
        }
        let roll = (rng.next() >> 11) as f32 / (1u64 << 53) as f32;
        let (index, moves) = if roll < epsilon || net.is_none() {
            let c = cands[rng.below(cands.len())];
            (c.index, c.moves)
        } else {
            let i = choose(space, book, net.expect("checked"), &cube)?;
            (i, book.moves_of(i))
        };
        states.push(cube);
        costs.push(moves);
        cube = book.apply(&cube, index);
        if states.len() > macros::MACRO_BUDGET {
            return None;
        }
    }

    // Walk backwards so each state carries the moves that actually followed.
    let mut to_go = vec![0.0f32; states.len()];
    let mut acc = 0usize;
    for i in (0..states.len()).rev() {
        acc += costs[i];
        to_go[i] = acc as f32;
    }
    let mut features = vec![0.0f32; states.len() * width];
    for (i, s) in states.iter().enumerate() {
        space.write_features(s, &mut features[i * width..(i + 1) * width]);
    }
    Some(Episode { features, to_go, moves: acc, macros: states.len() })
}

/// Score every admissible macro by what it costs plus what it leaves behind,
/// and play the cheapest.
///
/// `macro length + V(child)` is the whole rule. A value that is merely
/// ORDERED correctly is enough - the absolute numbers never matter, only
/// which child looks nearest.
pub fn choose(space: &CubeSpace, book: &Playbook, net: &Net, cube: &Cube) -> Option<usize> {
    let cands = book.improving_all(cube);
    if cands.is_empty() {
        return None;
    }
    let width = space.feature_len();
    let rows = net.rows as usize;
    let mut best: Option<(f32, usize)> = None;
    let mut i = 0;
    while i < cands.len() {
        let take = (cands.len() - i).min(rows);
        let mut f = vec![0.0f32; rows * width];
        for j in 0..take {
            let child = book.apply(cube, cands[i + j].index);
            space.write_features(&child, &mut f[j * width..(j + 1) * width]);
        }
        let v = net.value_of(&f);
        for j in 0..take {
            let c = cands[i + j];
            // A macro that finishes the cube costs exactly its own length.
            let child_solved = book.apply(cube, c.index).is_solved();
            let score = c.moves as f32 + if child_solved { 0.0 } else { v[j].max(0.0) };
            if best.is_none_or(|(b, _)| score < b) {
                best = Some((score, c.index));
            }
        }
        i += take;
    }
    best.map(|(_, i)| i)
}

/// Solve `cubes` and report the mean moves, under whichever chooser is given.
pub fn measure(
    space: &CubeSpace,
    book: &Playbook,
    net: Option<&Net>,
    cubes: usize,
    scramble: usize,
    seed: u64,
    random: bool,
) -> (usize, f32, f32) {
    let mut rng = Rng::new(seed);
    let (mut solved, mut moves, mut count) = (0usize, 0usize, 0usize);
    for i in 0..cubes {
        let (mut cube, _) = cube::scramble(scramble, seed ^ (i as u64 * 0x9E37));
        let mut used = 0usize;
        let mut steps = 0usize;
        loop {
            if cube.is_solved() {
                solved += 1;
                break;
            }
            let pick = match net {
                Some(n) => choose(space, book, n, &cube),
                None => {
                    let c = book.improving_all(&cube);
                    if c.is_empty() {
                        None
                    } else if random {
                        Some(c[rng.below(c.len())].index)
                    } else {
                        Some(c.iter().min_by_key(|x| (-x.home, x.cost, x.moves)).expect("non-empty").index)
                    }
                }
            };
            let Some(pick) = pick else { break };
            used += book.moves_of(pick);
            cube = book.apply(&cube, pick);
            steps += 1;
            if steps > macros::MACRO_BUDGET {
                break;
            }
        }
        moves += used;
        count += steps;
    }
    (solved, moves as f32 / cubes.max(1) as f32, count as f32 / cubes.max(1) as f32)
}

/// The value network's shape.
///
/// `moves: 1` makes the policy head degenerate on purpose: a softmax over one
/// option is always exactly one, so its cross-entropy gradient is identically
/// zero and the trunk is driven by the value head alone. That is cheaper and
/// clearer than adding a switch to turn the policy term off.
pub fn config(space: &CubeSpace, d_model: u32, d_ff: u32, blocks: u32) -> Config {
    Config { in_dim: space.feature_len() as u32, d_model, d_ff, blocks, moves: 1 }
}
