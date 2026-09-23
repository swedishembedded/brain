// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cube, as a state space the policy trainer can learn.
//!
//! Everything cube-specific about the learned solver is in this file. The
//! trainer, the network and the rollout know nothing about faces, stickers or
//! turns - they see an invertible move set, a goal, and a feature vector.

use std::sync::OnceLock;

use brain::solve::StateSpace;

use crate::cube::{facelet_geometry, Cube, Move};

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

    /// The cubie slots a cube is made of: 8 corners, then 12 edges. Centres
    /// are left out - they never move relative to one another, so they carry
    /// no information about the state.
    pub const SLOTS: usize = CORNERS + EDGES;

    /// Which piece sits in each slot and how it is turned, as one one-hot
    /// per slot: 20 slots, 24 states each.
    ///
    /// A corner slot holds one of 8 corner pieces in one of 3 twists and an
    /// edge slot one of 12 edge pieces in one of 2 flips, so both come to 24
    /// and the layout is uniform. Slot POSITION is carried by the offset -
    /// row `slot * 24 + k` means "slot `slot` is in state `k`" and nothing
    /// else ever means that.
    ///
    /// The alternative, one one-hot per sticker, hands the network 54
    /// independently coloured squares and leaves it to discover that three
    /// of them always travel together as one corner. That structure is a
    /// fact about the cube, not something worth spending capacity to infer,
    /// so it goes in the encoding. The first layer is a matmul against a
    /// one-hot, which is a table lookup, so the width costs nothing.
    pub const FEATURES: usize = CubeSpace::SLOTS * SLOT_STATES;
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
        cubies().write(s, out);
    }

    fn is_goal(&self, s: &Cube) -> bool {
        s.is_solved()
    }
}

/// A corner holds one of eight pieces in one of three twists; an edge one of
/// twelve pieces in one of two flips. Both are 24, which is what makes one
/// uniform block per slot possible.
const CORNERS: usize = 8;
const EDGES: usize = 12;
const SLOT_STATES: usize = 24;

/// The cube as twenty cubie slots, each named by the facelets it wears.
///
/// DERIVED from [`facelet_geometry`], never typed in: facelets that share a
/// cubie coordinate are the same physical piece, and how many share it says
/// what that piece is - three for a corner, two for an edge, one for a
/// centre. A hand-written table of "corner slot 0 is facelets 0, 9 and 38"
/// is twenty chances to transpose a digit and get a slot map that encodes
/// something plausible which is not a cube, and a policy trained on it would
/// still converge on its own broken picture.
pub struct Cubies {
    /// Corner slot -> its three facelets, reference facelet first and the
    /// other two following it around the corner (see [`Cubies::derive`]).
    pub corners: [[usize; 3]; CORNERS],
    /// Edge slot -> its two facelets, reference facelet first.
    pub edges: [[usize; 2]; EDGES],
    /// Colour set, as a bitmask of the six colours, -> the piece wearing it.
    /// Each cubie carries a unique set of colours, so the set a slot shows
    /// IS the identity of the piece currently in it, whatever its twist.
    pub corner_of_colours: [u8; 64],
    pub edge_of_colours: [u8; 64],
    /// Piece -> the colour it carries on its own reference facelet at home.
    /// Finding that colour among a slot's facelets is what reads off the
    /// orientation.
    pub corner_reference: [u8; CORNERS],
    pub edge_reference: [u8; EDGES],
}

fn cross(a: [i8; 3], b: [i8; 3]) -> [i8; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

impl Cubies {
    /// Group the 54 facelets into the cubies they belong to and fix, for
    /// each, the order its orientation is counted in.
    ///
    /// The ORIENTATION CONVENTION, stated once and applied uniformly:
    ///
    /// * A corner's reference facelet is the one on U or D - every corner
    ///   has exactly one, since U and D are opposite and no cubie touches
    ///   both. Its other two facelets follow in the order that makes the
    ///   three outward normals a right-handed frame (`n0 x n1 == n2`), which
    ///   is the same rotational sense - anticlockwise seen from outside the
    ///   corner - at all eight corners. The twist is then how many steps
    ///   around that cycle the piece's own U/D-coloured sticker sits from
    ///   the reference facelet.
    /// * An edge's reference facelet is the one on U or D if it has one, and
    ///   otherwise the one on F or B, which is the only choice left for the
    ///   four edges of the middle layer. The flip is 0 when the piece's home
    ///   reference colour is on it and 1 when it is not.
    ///
    /// Both conventions are the standard ones, and what earns them is that
    /// the sense is the same for every slot: only then are the twists
    /// invariant mod 3 and the flips invariant mod 2 under every turn, which
    /// the tests check on reachable states rather than assert here.
    fn derive() -> Cubies {
        let g = facelet_geometry();
        let mut groups: Vec<([i8; 3], Vec<usize>)> = Vec::new();
        for (facelet, &(cell, _)) in g.iter().enumerate() {
            match groups.iter_mut().find(|(c, _)| *c == cell) {
                Some((_, facelets)) => facelets.push(facelet),
                None => groups.push((cell, vec![facelet])),
            }
        }
        // Slot numbering is by cubie coordinate, which is arbitrary but
        // fixed: all that matters is that a slot is the same row of the
        // feature vector in every state ever written.
        groups.sort_by_key(|(cell, _)| *cell);

        let normal = |facelet: usize| g[facelet].1;
        let on_axis = |facelets: &[usize], axis: usize| {
            facelets.iter().copied().find(|&f| normal(f)[axis] != 0)
        };

        let (mut corners, mut edges) = ([[0usize; 3]; CORNERS], [[0usize; 2]; EDGES]);
        let (mut corner, mut edge) = (0, 0);
        for (_, facelets) in &groups {
            match facelets.len() {
                3 => {
                    let reference = on_axis(facelets, 1).expect("a corner touches U or D");
                    let rest: Vec<usize> =
                        facelets.iter().copied().filter(|&f| f != reference).collect();
                    let handed = cross(normal(reference), normal(rest[0])) == normal(rest[1]);
                    let (second, third) =
                        if handed { (rest[0], rest[1]) } else { (rest[1], rest[0]) };
                    corners[corner] = [reference, second, third];
                    corner += 1;
                }
                2 => {
                    let reference = on_axis(facelets, 1)
                        .or_else(|| on_axis(facelets, 2))
                        .expect("an edge touches U, D, F or B");
                    let other = facelets.iter().copied().find(|&f| f != reference);
                    edges[edge] = [reference, other.expect("an edge has a second facelet")];
                    edge += 1;
                }
                // A centre is one facelet and one colour: fixed, so silent.
                _ => {}
            }
        }
        assert_eq!((corner, edge), (CORNERS, EDGES), "a cube has 8 corners and 12 edges");

        // On the solved cube every piece is at home, so slot `i` is where
        // piece `i` lives and the colours it shows there are the colours
        // that identify it anywhere on the cube.
        let (mut corner_of_colours, mut edge_of_colours) = ([u8::MAX; 64], [u8::MAX; 64]);
        let (mut corner_reference, mut edge_reference) = ([0u8; CORNERS], [0u8; EDGES]);
        for (piece, facelets) in corners.iter().enumerate() {
            corner_of_colours[colours(&Cube::SOLVED, facelets) as usize] = piece as u8;
            corner_reference[piece] = Cube::SOLVED.0[facelets[0]];
        }
        for (piece, facelets) in edges.iter().enumerate() {
            edge_of_colours[colours(&Cube::SOLVED, facelets) as usize] = piece as u8;
            edge_reference[piece] = Cube::SOLVED.0[facelets[0]];
        }
        Cubies {
            corners,
            edges,
            corner_of_colours,
            edge_of_colours,
            corner_reference,
            edge_reference,
        }
    }

    /// One bit per slot: which piece is in it, turned which way.
    /// The same twenty slots [`Cubies::write`] one-hots, as `(piece,
    /// orientation)` pairs - corners first, then edges.
    ///
    /// Exactly the model's input, read out rather than encoded, so anything
    /// showing "what the model was given" is showing the thing itself and
    /// not a second description that could drift from it.
    pub fn read(&self, cube: &Cube) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(CORNERS + EDGES);
        for facelets in self.corners.iter() {
            let piece = piece_wearing(&self.corner_of_colours, colours(cube, facelets));
            let reference = self.corner_reference[piece];
            let twist = facelets.iter().position(|&f| cube.0[f] == reference).expect("reference colour");
            out.push((piece, twist));
        }
        for facelets in self.edges.iter() {
            let piece = piece_wearing(&self.edge_of_colours, colours(cube, facelets));
            let flip = usize::from(cube.0[facelets[0]] != self.edge_reference[piece]);
            out.push((piece, flip));
        }
        out
    }

    fn write(&self, cube: &Cube, out: &mut [f32]) {
        for (slot, facelets) in self.corners.iter().enumerate() {
            let piece = piece_wearing(&self.corner_of_colours, colours(cube, facelets));
            // The colour set names the piece, so the piece's reference
            // colour is one of the three on show; which facelet it is on is
            // the twist.
            let reference = self.corner_reference[piece];
            let twist = facelets
                .iter()
                .position(|&f| cube.0[f] == reference)
                .expect("a corner wears the reference colour of the piece its colours name");
            out[slot * SLOT_STATES + piece * 3 + twist] = 1.0;
        }
        for (slot, facelets) in self.edges.iter().enumerate() {
            let piece = piece_wearing(&self.edge_of_colours, colours(cube, facelets));
            let flip = usize::from(cube.0[facelets[0]] != self.edge_reference[piece]);
            out[(CORNERS + slot) * SLOT_STATES + piece * 2 + flip] = 1.0;
        }
    }
}

/// Which piece wears this set of colours. A set no piece wears means the
/// cube was assembled rather than turned; there is no honest feature vector
/// for it, and saying so beats writing a plausible one.
fn piece_wearing(pieces: &[u8; 64], mask: u8) -> usize {
    let piece = pieces[mask as usize];
    assert!(piece != u8::MAX, "no cubie wears colours {mask:06b}, so this is not a cube");
    piece as usize
}

/// The set of colours a slot is showing, as a bitmask. Unique per piece on
/// any legal cube, which is what makes it an identity.
fn colours(cube: &Cube, facelets: &[usize]) -> u8 {
    facelets.iter().fold(0u8, |mask, &f| mask | 1 << cube.0[f])
}

pub fn cubies() -> &'static Cubies {
    static CUBIES: OnceLock<Cubies> = OnceLock::new();
    CUBIES.get_or_init(Cubies::derive)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

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

    /// The feature vector read back the way a reader of it must: slot `s`
    /// holds one piece, turned one way, and nothing else is visible.
    fn slot_states(space: &CubeSpace, cube: &Cube) -> Vec<usize> {
        let mut f = vec![0.0f32; space.feature_len()];
        space.write_features(cube, &mut f);
        (0..CubeSpace::SLOTS)
            .map(|s| {
                let row = &f[s * 24..(s + 1) * 24];
                let set: Vec<usize> =
                    row.iter().enumerate().filter(|(_, &v)| v == 1.0).map(|(i, _)| i).collect();
                assert_eq!(set.len(), 1, "slot {s} must show exactly one state");
                set[0]
            })
            .collect()
    }

    /// Which piece a slot holds and how it is turned.
    fn piece_and_orientation(slot: usize, state: usize) -> (usize, usize) {
        let turns = if slot < 8 { 3 } else { 2 };
        (state / turns, state % turns)
    }

    fn is_permutation(pieces: &[usize], n: usize) -> bool {
        let mut seen = vec![false; n];
        pieces.iter().all(|&p| p < n && !std::mem::replace(&mut seen[p], true))
    }

    /// Parity of a permutation, counted as inversions - the cheapest form
    /// that needs no cycle bookkeeping.
    fn parity(pieces: &[usize]) -> usize {
        let mut inversions = 0;
        for i in 0..pieces.len() {
            for j in (i + 1)..pieces.len() {
                inversions += usize::from(pieces[i] > pieces[j]);
            }
        }
        inversions % 2
    }

    /// One bit per cubie slot - a feature vector that double-counted or
    /// dropped a slot would still train, and would still look like a cube.
    #[test]
    fn features_are_one_hot_per_slot() {
        let space = CubeSpace::new();
        let (cube, _) = crate::cube::scramble(7, 5);
        assert_eq!(space.feature_len(), 480);
        let mut f = vec![0.0f32; space.feature_len()];
        space.write_features(&cube, &mut f);
        assert_eq!(f.iter().sum::<f32>(), 20.0, "one bit per cubie slot");
        for s in 0..CubeSpace::SLOTS {
            let row = &f[s * 24..(s + 1) * 24];
            assert_eq!(row.iter().filter(|&&v| v == 1.0).count(), 1, "slot {s}");
        }
    }

    /// The solved cube is the fixed point of the encoding: every slot holds
    /// its own piece, untwisted. Any other reading of it means the piece
    /// numbering and the slot numbering disagree.
    #[test]
    fn the_solved_cube_holds_every_piece_at_home_untwisted() {
        let space = CubeSpace::new();
        for (s, &state) in slot_states(&space, &Cube::SOLVED).iter().enumerate() {
            let home = if s < 8 { s } else { s - 8 };
            assert_eq!(piece_and_orientation(s, state), (home, 0), "slot {s} when solved");
        }
    }

    /// Distinct cubes must be distinct to the network. A collision is a
    /// state the policy is structurally unable to see, and no amount of
    /// training fixes it.
    #[test]
    fn distinct_cubes_have_distinct_features() {
        let space = CubeSpace::new();
        let mut seen: HashMap<Vec<usize>, Cube> = HashMap::new();
        for seed in 0..4000u64 {
            let (cube, _) = crate::cube::scramble(1 + (seed % 24) as usize, seed + 1);
            if let Some(prev) = seen.insert(slot_states(&space, &cube), cube) {
                assert_eq!(prev, cube, "two different cubes wrote the same features");
            }
        }
        assert!(seen.len() > 3000, "the scrambles must cover many states, got {}", seen.len());
    }

    /// A face turn moves the eight cubies on that face and nothing else:
    /// four corners, four edges, the same eight however far the face turns.
    /// Opposite faces share no cubie; adjacent faces share two corners and
    /// one edge. A slot derived from the wrong facelets fails this.
    #[test]
    fn a_face_turn_moves_exactly_the_eight_cubies_on_that_face() {
        let space = CubeSpace::new();
        let (cube, _) = crate::cube::scramble(11, 42);
        let before = slot_states(&space, &cube);
        let mut touched: Vec<Vec<usize>> = Vec::new();
        for face in 0..6 {
            let mut per_turn: Vec<Vec<usize>> = Vec::new();
            for quarters in 0..3 {
                let after = slot_states(&space, &space.apply(&cube, face * 3 + quarters));
                let changed: Vec<usize> =
                    (0..CubeSpace::SLOTS).filter(|&s| after[s] != before[s]).collect();
                let m = space.move_at(face * 3 + quarters).notation();
                assert_eq!(changed.iter().filter(|&&s| s < 8).count(), 4, "{m} moves four corners");
                assert_eq!(changed.iter().filter(|&&s| s >= 8).count(), 4, "{m} moves four edges");
                per_turn.push(changed);
            }
            assert!(per_turn.iter().all(|c| *c == per_turn[0]), "one face, one set of cubies");
            touched.push(per_turn.remove(0));
        }
        for a in 0..6 {
            for b in (a + 1)..6 {
                let shared: Vec<usize> =
                    touched[a].iter().copied().filter(|s| touched[b].contains(s)).collect();
                // Face::ALL is U, R, F, D, L, B, so a face and the one three
                // along from it are opposite and touch nothing in common.
                let (corners, edges) = if b == a + 3 { (0, 0) } else { (2, 1) };
                assert_eq!(shared.iter().filter(|&&s| s < 8).count(), corners, "faces {a} and {b}");
                assert_eq!(shared.iter().filter(|&&s| s >= 8).count(), edges, "faces {a} and {b}");
            }
        }
    }

    /// Every reachable state obeys the cube's own laws: the pieces are a
    /// permutation of themselves, corner twists sum to zero mod three, edge
    /// flips sum to zero mod two, and the two permutations have the same
    /// parity. These hold of the CUBE, so an encoding that reports otherwise
    /// has an incoherent orientation convention rather than a rare state.
    #[test]
    fn the_encoding_obeys_the_laws_of_the_cube() {
        let space = CubeSpace::new();
        for seed in 0..300u64 {
            let (cube, _) = crate::cube::scramble(20, seed + 1);
            let states = slot_states(&space, &cube);
            let read = |range: std::ops::Range<usize>| -> (Vec<usize>, usize) {
                let decoded: Vec<(usize, usize)> =
                    range.clone().map(|s| piece_and_orientation(s, states[s])).collect();
                (decoded.iter().map(|&(p, _)| p).collect(), decoded.iter().map(|&(_, o)| o).sum())
            };
            let (corners, twist) = read(0..8);
            let (edges, flip) = read(8..CubeSpace::SLOTS);
            assert!(is_permutation(&corners, 8), "seed {seed}: corners {corners:?}");
            assert!(is_permutation(&edges, 12), "seed {seed}: edges {edges:?}");
            assert_eq!(twist % 3, 0, "seed {seed}: corner twists sum to zero mod three");
            assert_eq!(flip % 2, 0, "seed {seed}: edge flips sum to zero mod two");
            assert_eq!(parity(&corners), parity(&edges), "seed {seed}: permutation parities");
        }
    }
}
