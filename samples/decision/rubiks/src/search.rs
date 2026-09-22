// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The planner: what the shortest solution from here is, and which moves are
//! on one.
//!
//! Meet in the middle. A breadth-first sweep out from the SOLVED cube records
//! the exact distance of every state within `half` moves of it; a
//! depth-limited search out from the scrambled cube then only has to reach
//! that shell rather than walk all the way home, so an eight-move solve costs
//! a few tens of thousands of nodes instead of 18^8.
//!
//! Exact, not heuristic: [`Solver::distance`] returns the true optimal
//! length, which is what lets [`Solver::admissible`] say - with no
//! approximation anywhere - whether a move belongs on a shortest solution.
//! That is the whole basis on which this sample can claim the cube is solved
//! correctly no matter what the model proposes, and can measure the model
//! against a bar that is not of the model's own making.
//!
//! The ceiling is deliberate and stated: this solves scrambles up to
//! [`Solver::MAX_DEPTH`] moves optimally. A 20-move worst case is Kociemba's
//! two-phase algorithm with pattern databases - a different, much larger
//! piece of work, and not what this sample is about.
//!
//! Swedish Embedded AB builds the exact planner a heuristic policy is scored
//! against, for clients who need to know what a model is worth rather than
//! whether the demo looked good. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use crate::cube::{Cube, Face, Move};

pub struct Solver {
    /// Exact distance-to-solved for every state within `half` moves.
    shell: HashMap<Cube, u8>,
    half: u8,
    moves: Vec<Move>,
}

impl Solver {
    /// The deepest scramble this stays exact and quick on.
    pub const MAX_DEPTH: u8 = 8;

    /// Build the shell around the solved cube. `half` trades memory for
    /// search: 4 is ~43k states and leaves at most four moves for the
    /// forward search of an eight-move scramble.
    pub fn new(half: u8) -> Solver {
        let moves = Move::all();
        let mut shell = HashMap::new();
        shell.insert(Cube::SOLVED, 0u8);
        let mut frontier = vec![Cube::SOLVED];
        for depth in 1..=half {
            let mut next = Vec::new();
            for c in &frontier {
                for &m in &moves {
                    let n = c.apply(m);
                    if let std::collections::hash_map::Entry::Vacant(e) = shell.entry(n) {
                        e.insert(depth);
                        next.push(n);
                    }
                }
            }
            frontier = next;
        }
        Solver { shell, half, moves }
    }

    pub fn shell_size(&self) -> usize {
        self.shell.len()
    }

    pub fn half(&self) -> u8 {
        self.half
    }

    /// The exact number of moves in a shortest solution, or `None` when that
    /// is more than `max` (which must not exceed [`Solver::MAX_DEPTH`]).
    pub fn distance(&self, cube: &Cube, max: u8) -> Option<u8> {
        (0..=max).find(|&bound| self.within(cube, bound, None))
    }

    /// Is there a solution of at most `bound` moves?
    ///
    /// The budget shrinks with depth and the shell is consulted at EVERY
    /// node, so a hit means `depth_so_far + shell_distance <= bound` - which
    /// is why the first bound that succeeds is the exact distance rather than
    /// merely a bound that happens to work.
    fn within(&self, cube: &Cube, bound: u8, last: Option<Face>) -> bool {
        if let Some(&d) = self.shell.get(cube) {
            return d <= bound;
        }
        // NOT in the shell means further than `half` from solved - the shell
        // is a complete sweep, not a sample - so a budget that small cannot
        // reach home from here. This one line is what makes the forward
        // search `bound - half` deep instead of `bound` deep: without it a
        // seven-move cube walks 15^7 nodes to prove a six-move bound
        // impossible, and the test suite for this file took 25 minutes.
        if bound <= self.half {
            return false;
        }
        self.moves.iter().any(|&m| {
            // Two turns of one face in a row are always one turn of it.
            Some(m.face) != last && self.within(&cube.apply(m), bound - 1, Some(m.face))
        })
    }

    /// The moves that lie on SOME shortest solution: exactly those that take
    /// the distance from `d` down to `d - 1`.
    pub fn admissible(&self, cube: &Cube, d: u8) -> Vec<Move> {
        if d == 0 {
            return Vec::new();
        }
        self.moves
            .iter()
            .copied()
            .filter(|&m| self.distance(&cube.apply(m), d - 1) == Some(d - 1))
            .collect()
    }

    /// One shortest solution, or `None` if the cube is further away than
    /// `max`. Built by walking admissible moves, so it is optimal by
    /// construction rather than by a claim.
    pub fn solve(&self, cube: &Cube, max: u8) -> Option<Vec<Move>> {
        let mut d = self.distance(cube, max)?;
        let (mut c, mut out) = (*cube, Vec::with_capacity(d as usize));
        while d > 0 {
            let m = *self.admissible(&c, d).first().expect("an unsolved cube has an admissible move");
            c = c.apply(m);
            out.push(m);
            d -= 1;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::scramble;

    fn solver() -> Solver {
        Solver::new(4)
    }

    #[test]
    fn a_solved_cube_is_zero_moves_away() {
        assert_eq!(solver().distance(&Cube::SOLVED, 4), Some(0));
    }

    #[test]
    fn one_move_away_is_distance_one() {
        let s = solver();
        for m in Move::all() {
            assert_eq!(s.distance(&Cube::SOLVED.apply(m), 4), Some(1), "{}", m.notation());
        }
    }

    /// The distance is never more than the scramble that produced it, and -
    /// the part worth testing - it is sometimes LESS, because a random
    /// scramble is not a shortest path.
    #[test]
    fn the_distance_never_exceeds_the_scramble_that_made_it() {
        let s = solver();
        let mut shorter = 0;
        for seed in 0..12 {
            let (cube, moves) = scramble(6, seed);
            let d = s.distance(&cube, Solver::MAX_DEPTH).expect("within range");
            assert!(d <= moves.len() as u8, "seed {seed}: distance {d} > scramble {}", moves.len());
            if d < moves.len() as u8 {
                shorter += 1;
            }
        }
        assert!(shorter > 0, "no scramble was shorter than it looked - suspicious");
    }

    /// The property this whole sample rests on: whatever comes out of
    /// `solve` really does solve the cube, and in exactly the number of moves
    /// the planner promised.
    #[test]
    fn every_solution_solves_the_cube_and_is_optimal() {
        let s = solver();
        for seed in 0..20 {
            let (cube, _) = scramble(7, seed + 100);
            let d = s.distance(&cube, Solver::MAX_DEPTH).expect("within range");
            let sol = s.solve(&cube, Solver::MAX_DEPTH).expect("within range");
            assert_eq!(sol.len() as u8, d, "seed {seed}: solution is not the optimal length");
            assert!(cube.apply_all(&sol).is_solved(), "seed {seed}: solution does not solve the cube");
        }
    }

    #[test]
    fn every_admissible_move_steps_one_closer_and_there_is_always_one() {
        let s = solver();
        for seed in 0..8 {
            let (cube, _) = scramble(6, seed + 200);
            let d = s.distance(&cube, Solver::MAX_DEPTH).expect("within range");
            let adm = s.admissible(&cube, d);
            assert!(!adm.is_empty(), "seed {seed}: no admissible move from an unsolved cube");
            for m in adm {
                assert_eq!(s.distance(&cube.apply(m), d - 1), Some(d - 1), "{} is not a step closer", m.notation());
            }
        }
    }

    /// Most moves are NOT admissible - which is what makes picking one a
    /// decision worth measuring rather than a formality.
    #[test]
    fn admissible_moves_are_a_small_minority() {
        let s = solver();
        let (cube, _) = scramble(6, 42);
        let d = s.distance(&cube, Solver::MAX_DEPTH).unwrap();
        let adm = s.admissible(&cube, d);
        assert!(adm.len() < 6, "{} of 18 moves admissible - chance would be too easy a bar", adm.len());
    }
}
