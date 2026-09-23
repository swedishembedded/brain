// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cube, as a state space the policy trainer can learn.
//!
//! Everything cube-specific about the learned solver is in this file. The
//! trainer, the network and the rollout know nothing about faces, stickers or
//! turns - they see an invertible move set, a goal, and a feature vector.

use brain::solve::StateSpace;

use crate::cube::{Cube, Move};

/// `Move::all()` is laid out face-major, three quarter-turns per face, so a
/// move index splits as `face * 3 + (quarters - 1)`. Both the inverse and the
/// redundancy test below are arithmetic on that layout rather than a table,
/// which is one fewer thing that can disagree with the move list.
pub struct CubeSpace {
    moves: Vec<Move>,
}

impl Default for CubeSpace {
    fn default() -> CubeSpace {
        CubeSpace::new()
    }
}

impl CubeSpace {
    pub fn new() -> CubeSpace {
        CubeSpace { moves: Move::all() }
    }

    pub fn move_at(&self, i: usize) -> Move {
        self.moves[i]
    }

    /// The colour of every sticker as a one-hot: 54 stickers, 6 colours.
    ///
    /// Sticker POSITION is carried by the offset rather than by any learned
    /// encoding - row `i * 6 + c` means "sticker `i` is colour `c`" and
    /// nothing else ever means that. The first layer is a matmul against
    /// this, which for a one-hot is a table lookup, so the width is free.
    pub const FEATURES: usize = 54 * 6;
}

impl StateSpace for CubeSpace {
    type State = Cube;

    fn moves(&self) -> usize {
        self.moves.len()
    }

    fn goal(&self) -> Cube {
        Cube::SOLVED
    }

    fn apply(&self, s: &Cube, m: usize) -> Cube {
        s.apply(self.moves[m])
    }

    fn inverse(&self, m: usize) -> usize {
        // quarters 1 <-> 3, and a half turn is its own inverse.
        (m / 3) * 3 + (2 - m % 3)
    }

    /// Two turns of the SAME face are one turn of that face, so a walk that
    /// takes both has spent two steps to move one. Stronger than the default
    /// (undo-the-last-move) test, and the difference matters: leaving it out
    /// makes a walk of length L land closer than L and teaches the long way.
    fn redundant(&self, a: usize, b: usize) -> bool {
        a / 3 == b / 3
    }

    fn feature_len(&self) -> usize {
        CubeSpace::FEATURES
    }

    fn write_features(&self, s: &Cube, out: &mut [f32]) {
        for (i, &c) in s.0.iter().enumerate() {
            out[i * 6 + c as usize] = 1.0;
        }
    }

    fn is_goal(&self, s: &Cube) -> bool {
        s.is_solved()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract the whole label generator rests on.
    #[test]
    fn every_move_has_an_inverse_that_undoes_it() {
        let space = CubeSpace::new();
        brain::solve::data::inverse_is_an_inverse(&space, 40, 0xBEEF).expect("inverses must hold");
    }

    /// The inverse index really names the inverse NOTATION, not merely some
    /// move that happens to restore the cube.
    #[test]
    fn the_inverse_index_is_the_inverse_turn() {
        let space = CubeSpace::new();
        for m in 0..space.moves() {
            let a = space.move_at(m);
            let b = space.move_at(space.inverse(m));
            assert_eq!(a.face, b.face, "an inverse turns the same face");
            assert_eq!((a.quarters + b.quarters) % 4, 0, "{} then {}", a.notation(), b.notation());
        }
    }

    /// A one-hot with exactly one bit per sticker - a feature vector that
    /// double-counted or dropped a sticker would still train.
    #[test]
    fn features_are_one_hot_per_sticker() {
        let space = CubeSpace::new();
        let (cube, _) = crate::cube::scramble(7, 5);
        let mut f = vec![0.0f32; space.feature_len()];
        space.write_features(&cube, &mut f);
        assert_eq!(f.iter().sum::<f32>(), 54.0, "one bit per sticker");
        for i in 0..54 {
            let row = &f[i * 6..(i + 1) * 6];
            assert_eq!(row.iter().filter(|&&v| v == 1.0).count(), 1, "sticker {i}");
        }
    }
}
