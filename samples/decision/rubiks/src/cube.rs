// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A 3x3x3 cube, its eighteen moves, and nothing else.
//!
//! The move tables are DERIVED, not typed in. Each of the 54 facelets knows
//! where it sits and which way it faces; a move rotates the coordinates of
//! every facelet in one layer and looks up which facelet now occupies that
//! place. Six hand-written 54-entry permutation tables would be six chances
//! to transpose a pair of digits and get a cube that turns plausibly and is
//! not a cube - and a solver on top of it would still "solve" its own broken
//! group, which is the failure this file exists to avoid.
//!
//! Swedish Embedded AB builds the executable model of the machine - the part
//! that has to be right before any policy on top of it means anything - for
//! its clients. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// The six faces, in the facelet order this file numbers them: `U` is 0..9,
/// `R` 9..18, `F` 18..27, `D` 27..36, `L` 36..45, `B` 45..54.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    U,
    R,
    F,
    D,
    L,
    B,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::U, Face::R, Face::F, Face::D, Face::L, Face::B];

    pub fn index(self) -> usize {
        match self {
            Face::U => 0,
            Face::R => 1,
            Face::F => 2,
            Face::D => 3,
            Face::L => 4,
            Face::B => 5,
        }
    }

    pub fn letter(self) -> char {
        ['U', 'R', 'F', 'D', 'L', 'B'][self.index()]
    }

    /// What a face is called when a person is looking at the cube.
    pub fn name(self) -> &'static str {
        ["top", "right", "front", "bottom", "left", "back"][self.index()]
    }
}

/// One of the 18 quarter/half turns: a face plus how far it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub face: Face,
    /// 1 = clockwise, 2 = half turn, 3 = counter-clockwise.
    pub quarters: u8,
}

impl Move {
    /// Every legal move, in a fixed order, so a run is reproducible.
    pub fn all() -> Vec<Move> {
        let mut v = Vec::with_capacity(18);
        for face in Face::ALL {
            for quarters in 1..=3u8 {
                v.push(Move { face, quarters });
            }
        }
        v
    }

    /// Standard cube notation: `R`, `R2`, `R'`.
    pub fn notation(self) -> String {
        match self.quarters {
            1 => self.face.letter().to_string(),
            2 => format!("{}2", self.face.letter()),
            _ => format!("{}'", self.face.letter()),
        }
    }

    /// The same turn in words, which is what a language model is given
    /// instead of the notation - see this sample's README on why option text
    /// has to describe rather than label.
    pub fn describe(self) -> String {
        let how = match self.quarters {
            1 => "a quarter turn clockwise",
            2 => "a half turn",
            _ => "a quarter turn counter-clockwise",
        };
        format!("turn the {} face {how}", self.face.name())
    }

    pub fn inverse(self) -> Move {
        Move { face: self.face, quarters: 4 - self.quarters }
    }
}

/// A cube, as the colour sitting on each of the 54 facelets. The colour is
/// the index of the face it belongs on when solved, so [`Cube::SOLVED`] is
/// `[0,0,0,0,0,0,0,0,0,1,1,1,...]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cube(pub [u8; 54]);

impl Cube {
    pub const SOLVED: Cube = {
        let mut f = [0u8; 54];
        let mut i = 0;
        while i < 54 {
            f[i] = (i / 9) as u8;
            i += 1;
        }
        Cube(f)
    };

    pub fn is_solved(&self) -> bool {
        *self == Cube::SOLVED
    }

    /// How many facelets already carry the colour they end on. The one
    /// progress number that needs no search - and, deliberately, not the one
    /// the shield uses, because it is not monotone along an optimal solution.
    pub fn facelets_home(&self) -> usize {
        self.0.iter().enumerate().filter(|(i, &c)| c == (i / 9) as u8).count()
    }

    pub fn apply(&self, m: Move) -> Cube {
        let table = &move_tables()[m.face.index()];
        let mut out = *self;
        for _ in 0..m.quarters {
            let mut next = out;
            for (from, &to) in table.iter().enumerate() {
                next.0[to] = out.0[from];
            }
            out = next;
        }
        out
    }

    pub fn apply_all(&self, moves: &[Move]) -> Cube {
        moves.iter().fold(*self, |c, &m| c.apply(m))
    }
}

/// Where each facelet sits in space, and which way it faces.
///
/// `U` is +y, `R` +x, `F` +z. Within a face the stickers are read row by row
/// as a person looking AT that face sees them, which fixes the two in-face
/// directions listed here; everything else follows.
fn geometry() -> [([i8; 3], [i8; 3]); 54] {
    let mut g = [([0i8; 3], [0i8; 3]); 54];
    for face in Face::ALL {
        // (normal, direction of a column step, direction of a row step)
        let (n, col, row): ([i8; 3], [i8; 3], [i8; 3]) = match face {
            // Looking down: rows run front-ward, columns run right-ward.
            Face::U => ([0, 1, 0], [1, 0, 0], [0, 0, 1]),
            // Looking up: rows run back-ward, columns run right-ward.
            Face::D => ([0, -1, 0], [1, 0, 0], [0, 0, -1]),
            // Looking at a side face: rows always run downward.
            Face::F => ([0, 0, 1], [1, 0, 0], [0, -1, 0]),
            Face::B => ([0, 0, -1], [-1, 0, 0], [0, -1, 0]),
            Face::R => ([1, 0, 0], [0, 0, -1], [0, -1, 0]),
            Face::L => ([-1, 0, 0], [0, 0, 1], [0, -1, 0]),
        };
        for r in 0..3i8 {
            for c in 0..3i8 {
                let mut p = n;
                for axis in 0..3 {
                    p[axis] += col[axis] * (c - 1) + row[axis] * (r - 1);
                }
                g[face.index() * 9 + (r as usize) * 3 + c as usize] = (p, n);
            }
        }
    }
    g
}

/// Rotate a vector a quarter turn the way turning `face` clockwise does,
/// seen by someone looking AT that face from outside it.
fn rotate(face: Face, v: [i8; 3]) -> [i8; 3] {
    let [x, y, z] = v;
    match face {
        Face::R => [x, z, -y],
        Face::L => [x, -z, y],
        Face::U => [-z, y, x],
        Face::D => [z, y, -x],
        Face::F => [y, -x, z],
        Face::B => [-y, x, z],
    }
}

/// For each face, where each facelet's sticker lands after one clockwise
/// quarter turn of that face (a facelet outside the layer maps to itself).
fn build_tables() -> [[usize; 54]; 6] {
    let g = geometry();
    let mut tables = [[0usize; 54]; 6];
    for face in Face::ALL {
        let (axis, side) = match face {
            Face::R => (0, 1i8),
            Face::L => (0, -1),
            Face::U => (1, 1),
            Face::D => (1, -1),
            Face::F => (2, 1),
            Face::B => (2, -1),
        };
        for i in 0..54 {
            let (p, n) = g[i];
            if p[axis] != side {
                tables[face.index()][i] = i;
                continue;
            }
            let (p2, n2) = (rotate(face, p), rotate(face, n));
            let to = (0..54)
                .find(|&j| g[j] == (p2, n2))
                .expect("a rotated facelet must land on another facelet");
            tables[face.index()][i] = to;
        }
    }
    tables
}

fn move_tables() -> &'static [[usize; 54]; 6] {
    use std::sync::OnceLock;
    static TABLES: OnceLock<[[usize; 54]; 6]> = OnceLock::new();
    TABLES.get_or_init(build_tables)
}

/// A deterministic scramble of `n` moves, never undoing the face it just
/// turned (which would make the scramble shorter than it says it is).
pub fn scramble(n: usize, seed: u64) -> (Cube, Vec<Move>) {
    let all = Move::all();
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let (mut cube, mut moves) = (Cube::SOLVED, Vec::with_capacity(n));
    let mut last: Option<Face> = None;
    while moves.len() < n {
        let m = all[(next() % all.len() as u64) as usize];
        if Some(m.face) == last {
            continue;
        }
        last = Some(m.face);
        cube = cube.apply(m);
        moves.push(m);
    }
    (cube, moves)
}

impl Cube {
    /// The cube as a flat net, so a finished run can be looked at rather
    /// than trusted. Colours are the initials of the face they belong on.
    pub fn net(&self) -> String {
        let letter = |i: usize| Face::ALL[self.0[i] as usize].letter();
        let row = |base: usize, r: usize| -> String {
            (0..3).map(|c| letter(base + r * 3 + c)).collect::<String>()
        };
        let mut out = String::new();
        for r in 0..3 {
            out.push_str(&format!("      {}\n", row(0, r)));
        }
        for r in 0..3 {
            out.push_str(&format!(
                "  {} {} {} {}\n",
                row(Face::L.index() * 9, r),
                row(Face::F.index() * 9, r),
                row(Face::R.index() * 9, r),
                row(Face::B.index() * 9, r)
            ));
        }
        for r in 0..3 {
            out.push_str(&format!("      {}\n", row(Face::D.index() * 9, r)));
        }
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(face: Face, quarters: u8) -> Move {
        Move { face, quarters }
    }

    /// Four quarter turns of anything is nothing - the cheapest check that a
    /// move is a rotation at all, on every face.
    #[test]
    fn every_face_has_order_four() {
        for face in Face::ALL {
            let mut c = Cube::SOLVED;
            for i in 1..=4 {
                c = c.apply(m(face, 1));
                assert_eq!(c.is_solved(), i == 4, "{} after {i} quarter turns", face.letter());
            }
        }
    }

    #[test]
    fn a_move_and_its_inverse_cancel() {
        for mv in Move::all() {
            let c = Cube::SOLVED.apply(mv).apply(mv.inverse());
            assert!(c.is_solved(), "{} then {}", mv.notation(), mv.inverse().notation());
        }
    }

    #[test]
    fn a_half_turn_is_two_quarter_turns() {
        for face in Face::ALL {
            assert_eq!(Cube::SOLVED.apply(m(face, 2)), Cube::SOLVED.apply(m(face, 1)).apply(m(face, 1)));
        }
    }

    /// Opposite faces do not touch, so their turns commute - and adjacent
    /// faces' turns do NOT. A table with a row copied from the wrong face
    /// fails one of these.
    #[test]
    fn opposite_faces_commute_and_adjacent_ones_do_not() {
        for (a, b) in [(Face::U, Face::D), (Face::R, Face::L), (Face::F, Face::B)] {
            assert_eq!(
                Cube::SOLVED.apply(m(a, 1)).apply(m(b, 1)),
                Cube::SOLVED.apply(m(b, 1)).apply(m(a, 1)),
                "{} and {} must commute",
                a.letter(),
                b.letter()
            );
        }
        for (a, b) in [(Face::U, Face::R), (Face::F, Face::R), (Face::U, Face::F)] {
            assert_ne!(
                Cube::SOLVED.apply(m(a, 1)).apply(m(b, 1)),
                Cube::SOLVED.apply(m(b, 1)).apply(m(a, 1)),
                "{} and {} must not commute",
                a.letter(),
                b.letter()
            );
        }
    }

    /// The sexy move has order 6 on a real cube, and order almost anything
    /// else on a cube whose tables are subtly wrong. This is the test that
    /// earns the derivation above.
    #[test]
    fn the_sexy_move_has_order_six() {
        let seq = [m(Face::R, 1), m(Face::U, 1), m(Face::R, 3), m(Face::U, 3)];
        let mut c = Cube::SOLVED;
        for i in 1..=6 {
            c = c.apply_all(&seq);
            assert_eq!(c.is_solved(), i == 6, "(R U R' U') x {i}");
        }
    }

    /// The T-permutation is its own inverse: two of them are identity.
    #[test]
    fn the_t_perm_is_an_involution() {
        let t = [
            m(Face::R, 1), m(Face::U, 1), m(Face::R, 3), m(Face::U, 3),
            m(Face::R, 3), m(Face::F, 1), m(Face::R, 2), m(Face::U, 3),
            m(Face::R, 3), m(Face::U, 3), m(Face::R, 1), m(Face::U, 1),
            m(Face::R, 3), m(Face::F, 3),
        ];
        let once = Cube::SOLVED.apply_all(&t);
        assert!(!once.is_solved(), "a T-perm moves pieces");
        assert!(once.apply_all(&t).is_solved(), "two T-perms are identity");
    }

    /// Every move is a permutation: no colour is created or destroyed.
    #[test]
    fn colours_are_conserved() {
        let (c, _) = scramble(25, 7);
        let mut count = [0usize; 6];
        for &s in &c.0 {
            count[s as usize] += 1;
        }
        assert_eq!(count, [9; 6]);
    }

    #[test]
    fn a_scramble_is_undone_by_its_reverse() {
        let (c, moves) = scramble(12, 99);
        assert!(!c.is_solved());
        let back: Vec<Move> = moves.iter().rev().map(|m| m.inverse()).collect();
        assert!(c.apply_all(&back).is_solved());
    }

    #[test]
    fn a_solved_cube_has_every_facelet_home() {
        assert_eq!(Cube::SOLVED.facelets_home(), 54);
        let (c, _) = scramble(8, 3);
        assert!(c.facelets_home() < 54);
    }
}
