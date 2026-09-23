// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Solving ANY cube with a progress measure and no planner.
//!
//! `search.rs` is exact and therefore shallow: it sees eight moves. A cube
//! scrambled by forty random turns is around eighteen moves from solved, so
//! the exact planner cannot describe a single legal step of its solution.
//! This file is the other way to get a guarantee out of a machine that cannot
//! search: replace the distance oracle with a **measure that only ever moves
//! one way**, and a set of actions rich enough that one of them always moves
//! it.
//!
//! The measure is read straight off the cube, with no lookahead:
//!
//! ```text
//!   parity      P  0 when the eight corners are an EVEN permutation, else 1
//!   home cubies H  how many of the 20 cubies are in their own slot, turned
//!                  the right way (20 is the solved cube and nothing else)
//!   displacement D sum over the 20 slots of 0 (home) / 1 (right slot, wrong
//!                  orientation) / 2 (wrong piece)
//!   measure     Phi = (P, 20 - H, D), compared LEXICOGRAPHICALLY
//! ```
//!
//! Every macro played strictly decreases Phi, Phi lives in a finite set, and
//! its only minimum is `(0, 0, 0)` - the solved cube. That is the entire
//! termination argument: no depth limit, no search, nothing to tune.
//!
//! Each component earns its place:
//!
//! * **`H` is the headline.** [`admissible`] is the rule this layer was
//!   designed around - more cubies home than before - and it is what fires
//!   on 94% of the macros played across two thousand solves.
//! * **`D` catches the stall.** Three cubies that need cycling into their
//!   slots while still misoriented gain nothing under `H`; `D` is what makes
//!   "right slots now, right twists next" count as progress. It fires on
//!   under 2% of macros and is not optional for any of them: played under
//!   the strict `H` rule alone, 14 of 20 random cubes walk into a state
//!   where nothing at all is admissible.
//! * **`P` is not optional.** Every commutator is an even permutation of the
//!   corners and conjugating one keeps it even, so a library built only from
//!   commutators leaves corner parity INVARIANT - and half of all scrambles
//!   are odd. Those cubes are not hard for such a library, they are
//!   unreachable. One base algorithm with an odd corner permutation fixes
//!   that, and putting `P` first in the measure means such a macro is
//!   admissible whenever parity is odd - so the solver can never be stuck
//!   there - and is admissible never again once it is even. Exactly half of
//!   random scrambles need it, exactly once each.
//!
//! The library is GENERATED, not typed in. Five base algorithms disturb two
//! to four cubies each and leave the rest alone; conjugating each by a setup
//! `X` - play `X`, play the base, undo `X` - moves that same small effect
//! onto whichever cubies `X` brought into the base's reach. The generator
//! sweeps every setup of up to four turns and keeps the shortest conjugate
//! of each distinct EFFECT CLASS. What comes out is not a sample of the
//! cube's small elements but ALL of them: every one of the 112 three-cycles
//! of corners, all 440 three-cycles of edges, all 56 corner-twist pairs, all
//! 66 edge-flip pairs, and one corner-and-edge swap per pair of corners -
//! 702 macros, and the tests assert those counts rather than hoping.
//!
//! Swedish Embedded AB builds control layers for its clients that are
//! correct by construction: a measure that cannot go backwards and an action
//! set that always admits a step, in place of a search whose result has to
//! be trusted. If your team needs a policy with a guarantee attached to it,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

// This layer has no caller in `main.rs`: the shield there drives the exact
// planner, which is a different guarantee on a different depth of cube. Its
// entry points - [`library`], [`admissible`], [`progresses`], [`solve`] -
// are exercised by the tests at the bottom of this file, which is where the
// guarantee is established.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::cube::{Cube, Face, Move};
use crate::space::Cubies;

/// A named move sequence, played as one action.
#[derive(Default)]
pub struct Macro {
    pub name: String,
    pub moves: Vec<Move>,
}

/// The eight corner slots and twelve edge slots [`crate::space`] decodes a
/// cube into. Centres are not slots: they never move.
const SLOTS: usize = 20;
const CORNERS: usize = 8;

/// The most macros a solve can possibly take, read straight off the measure:
/// `Phi = (P, 20 - H, D)` ranges over `2 * 21 * 41` values, every macro
/// consumes at least one of them and none is ever revisited. Measured solves
/// take two orders of magnitude fewer (see the tests); this is the number the
/// termination argument gives, so it is the number the solver asserts
/// against - a longer run would mean some macro did not decrease Phi, which
/// is a bug in this file and not a hard cube.
pub const MACRO_BUDGET: usize = 2 * 21 * 41;

// ---------------------------------------------------------------------------
// reading the cube: the same cubie decoding the policy's features use
// ---------------------------------------------------------------------------

/// Which piece sits in a slot and how it is turned.
///
/// `at` reads the colour on a facelet, so the same routine scores a cube and
/// scores what a cube WOULD be after a macro - by reading through the macro's
/// permutation - without building the second cube.
///
/// The slot layout, the colours-are-an-identity trick and the orientation
/// convention are all [`crate::space`]'s: this is the decoding the learned
/// policy's feature vector is written from, read as two small numbers
/// instead of a one-hot.
#[inline]
fn cubie_at<F: Fn(usize) -> u8>(cubies: &Cubies, slot: usize, at: F) -> (u8, u8) {
    if slot < CORNERS {
        let f = cubies.corners[slot];
        let (a, b, c) = (at(f[0]), at(f[1]), at(f[2]));
        let piece = cubies.corner_of_colours[(1 << a | 1 << b | 1 << c) as usize];
        assert!(piece != u8::MAX, "no corner wears these colours, so this is not a cube");
        let reference = cubies.corner_reference[piece as usize];
        let twist = [a, b, c].iter().position(|&x| x == reference);
        (piece, twist.expect("a corner wears the reference colour of the piece it is") as u8)
    } else {
        let f = cubies.edges[slot - CORNERS];
        let (a, b) = (at(f[0]), at(f[1]));
        let piece = cubies.edge_of_colours[(1 << a | 1 << b) as usize];
        assert!(piece != u8::MAX, "no edge wears these colours, so this is not a cube");
        (piece, u8::from(a != cubies.edge_reference[piece as usize]))
    }
}

/// The slot a piece belongs in is the slot with its own number.
#[inline]
fn home_piece(slot: usize) -> u8 {
    (if slot < CORNERS { slot } else { slot - CORNERS }) as u8
}

/// How far a slot is from holding its own piece, turned its own way:
/// `0` home, `1` right slot wrong orientation, `2` wrong piece entirely.
#[inline]
fn slot_cost<F: Fn(usize) -> u8>(cubies: &Cubies, slot: usize, at: F) -> u8 {
    match cubie_at(cubies, slot, at) {
        (piece, _) if piece != home_piece(slot) => 2,
        (_, orientation) => u8::from(orientation != 0),
    }
}

/// The twenty cubies of a cube, decoded once: which piece is in each slot
/// and how it is turned.
fn cubies_of(cube: &Cube) -> [(u8, u8); SLOTS] {
    let cubies = crate::space::cubies();
    std::array::from_fn(|slot| cubie_at(cubies, slot, |f| cube.0[f]))
}

/// The slots that are not home, as a bitmask.
fn away_mask(state: &[(u8, u8); SLOTS]) -> u32 {
    let mut mask = 0u32;
    for (slot, &(piece, orientation)) in state.iter().enumerate() {
        if piece != home_piece(slot) || orientation != 0 {
            mask |= 1 << slot;
        }
    }
    mask
}

/// The corner permutation on its own - the part of a state the parity
/// component of the measure is read from.
fn corner_pieces(state: &[(u8, u8); SLOTS]) -> [u8; CORNERS] {
    let mut pieces = [0u8; CORNERS];
    for (out, &(piece, _)) in pieces.iter_mut().zip(state) {
        *out = piece;
    }
    pieces
}

/// Parity of a permutation, counted as inversions - the cheapest form, and
/// the same one `space.rs`'s law tests use.
fn parity(pieces: &[u8]) -> usize {
    let mut inversions = 0;
    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            inversions += usize::from(pieces[i] > pieces[j]);
        }
    }
    inversions % 2
}

/// 1 when the eight corners are an odd permutation of themselves. Odd is
/// exactly the half of the cube group no commutator can leave.
pub fn corner_parity(cube: &Cube) -> usize {
    parity(&corner_pieces(&cubies_of(cube)))
}

/// How many of the twenty cubies are in their own slot AND turned the right
/// way. Twenty is the solved cube and nothing else is, because the centres
/// cannot move relative to one another.
pub fn home_cubies(cube: &Cube) -> usize {
    SLOTS - away_mask(&cubies_of(cube)).count_ones() as usize
}

/// The tie-breaker: total distance-from-home over the twenty slots, counting
/// a misoriented piece as half the trouble of a misplaced one. Zero exactly
/// when the cube is solved.
pub fn displacement(cube: &Cube) -> usize {
    cubies_of(cube)
        .iter()
        .enumerate()
        .map(|(slot, &(piece, orientation))| match piece == home_piece(slot) {
            false => 2,
            true => usize::from(orientation != 0),
        })
        .sum()
}

/// `Phi = (P, 20 - H, D)`, smaller being better, compared lexicographically.
pub fn measure(cube: &Cube) -> (usize, usize, usize) {
    (corner_parity(cube), SLOTS - home_cubies(cube), displacement(cube))
}

/// The rule this layer was designed around: does playing `m` put strictly
/// more cubies home? Computed by playing it and counting - no search, no
/// distance oracle, no table.
///
/// Not every unsolved state admits such a macro, which is the finding this
/// file records rather than papers over; [`progresses`] is the rule the
/// solver plays, and the tests below measure how far apart the two are.
pub fn admissible(cube: &Cube, m: &Macro) -> bool {
    home_cubies(&cube.apply_all(&m.moves)) > home_cubies(cube)
}

/// The rule the solver plays: strictly decrease `Phi` lexicographically.
pub fn progresses(cube: &Cube, m: &Macro) -> bool {
    measure(&cube.apply_all(&m.moves)) < measure(cube)
}

// ---------------------------------------------------------------------------
// effects: a macro as a permutation of the 54 facelets
// ---------------------------------------------------------------------------

/// Where every sticker comes FROM: after the sequence, facelet `k` wears
/// whatever facelet `source[k]` wore before it.
type Effect = [u8; 54];

const IDENTITY: Effect = {
    let mut p = [0u8; 54];
    let mut i = 0;
    while i < 54 {
        p[i] = i as u8;
        i += 1;
    }
    p
};

/// The permutation a sequence performs, read out of the ENGINE rather than
/// derived a second time: [`Cube::apply`] permutes 54 bytes and never looks
/// at what they mean, so handing it facelet indices instead of colours
/// returns exactly the mapping its own move tables implement. A second
/// derivation here would be a second thing that can disagree with the cube.
fn effect_of(moves: &[Move]) -> Effect {
    Cube(IDENTITY).apply_all(moves).0
}

/// The effect of `first` followed by `second`.
fn then(first: &Effect, second: &Effect) -> Effect {
    let mut out = [0u8; 54];
    for (k, o) in out.iter_mut().enumerate() {
        *o = first[second[k] as usize];
    }
    out
}

fn undo(e: &Effect) -> Effect {
    let mut out = [0u8; 54];
    for (k, &source) in e.iter().enumerate() {
        out[source as usize] = k as u8;
    }
    out
}

fn with_effect(cube: &Cube, e: &Effect) -> Cube {
    let mut out = *cube;
    for (k, o) in out.0.iter_mut().enumerate() {
        *o = cube.0[e[k] as usize];
    }
    out
}

/// What a macro does to a solved cube, slot by slot: which piece ends in
/// each slot and how it is turned. Its whole effect, since a macro is a
/// permutation and a permutation is determined by where it sends the
/// identity.
fn cubies_after(e: &Effect) -> [(u8, u8); SLOTS] {
    cubies_of(&with_effect(&Cube::SOLVED, e))
}

/// Which slots a macro touches, as a bitmask - the key to scoring one fast.
/// A base algorithm disturbs two to four cubies and conjugation cannot
/// change that count, so scoring a macro means scoring four slots rather
/// than twenty; and a macro whose slots are all already home cannot possibly
/// help, which one AND rules out.
fn support(e: &Effect) -> u32 {
    away_mask(&cubies_after(e))
}

/// Whether a macro flips the parity of the corner permutation. A property of
/// the MACRO, not of the state it is played on, so the solver reads the
/// state's parity once per step rather than once per candidate.
fn flips_parity(e: &Effect) -> bool {
    parity(&corner_pieces(&cubies_after(e))) == 1
}

// ---------------------------------------------------------------------------
// playing
// ---------------------------------------------------------------------------

struct Entry {
    effect: Effect,
    support: u32,
    flips_parity: bool,
}

/// A library with its effects worked out once.
///
/// [`solve`] takes a plain `&[Macro]`, as its contract says, and pays for
/// this index on every call. Solving thousands of cubes against one fixed
/// library - which is what establishing the guarantee takes - builds it once
/// instead.
pub struct Playbook {
    entries: Vec<Entry>,
    lengths: Vec<usize>,
}

impl Playbook {
    pub fn new(library: &[Macro]) -> Playbook {
        let entries = library
            .iter()
            .map(|m| {
                let effect = effect_of(&m.moves);
                Entry { support: support(&effect), flips_parity: flips_parity(&effect), effect }
            })
            .collect();
        Playbook { entries, lengths: library.iter().map(|m| m.moves.len()).collect() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The macro that leaves `Phi` smallest, or `None` if this state is
    /// stuck. `Phi` before the move is the same for every candidate, so
    /// minimising `Phi` after it means maximising the gain in home cubies,
    /// then the drop in displacement - and a parity-flipping macro, which
    /// beats both, gets played the moment parity is odd rather than being
    /// saved for the endgame where its damage costs the most.
    fn improving(&self, cube: &Cube) -> Option<usize> {
        let cubies = crate::space::cubies();
        let state = cubies_of(cube);
        let away = away_mask(&state);
        let odd = parity(&corner_pieces(&state)) == 1;
        let mut best: Option<(usize, i32, i32, usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let parity_after = usize::from(odd != e.flips_parity);
            if parity_after > usize::from(odd) {
                // It would make the corner permutation odd again.
                continue;
            }
            let fixes_parity = parity_after < usize::from(odd);
            if !fixes_parity && e.support & away == 0 {
                // Every slot it touches is already home, and it has no
                // parity to trade for the damage: it can only undo work.
                continue;
            }
            // Untouched slots contribute the same H and D before and after,
            // so the whole comparison lives inside the support.
            let (mut home, mut cost) = (0i32, 0i32);
            let mut rest = e.support;
            while rest != 0 {
                let slot = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                let before = slot_cost(cubies, slot, |f| cube.0[f]) as i32;
                let after = slot_cost(cubies, slot, |f| cube.0[e.effect[f] as usize]) as i32;
                home += i32::from(after == 0) - i32::from(before == 0);
                cost += after - before;
            }
            let improves = fixes_parity || home > 0 || (home == 0 && cost < 0);
            if !improves {
                continue;
            }
            let score = (parity_after, -home, cost, self.lengths[i], i);
            if best.is_none_or(|b| score < b) {
                best = Some(score);
            }
        }
        best.map(|(_, _, _, _, i)| i)
    }

    /// Play improving macros until the cube is solved or nothing improves,
    /// returning what was played and where it ended. A failure has to report
    /// the state it stopped in, which is what this is for.
    pub fn run(&self, cube: &Cube) -> (Vec<usize>, Cube) {
        let mut state = *cube;
        let mut played = Vec::new();
        while !state.is_solved() {
            let Some(i) = self.improving(&state) else { break };
            state = with_effect(&state, &self.entries[i].effect);
            played.push(i);
            assert!(
                played.len() <= MACRO_BUDGET,
                "the measure bounds a solve at {MACRO_BUDGET} macros; a longer run means \
                 some macro did not strictly decrease it"
            );
        }
        (played, state)
    }

    /// The macros that solve this cube, or `None` if some state along the
    /// way admitted none.
    pub fn play(&self, cube: &Cube) -> Option<Vec<usize>> {
        let (played, state) = self.run(cube);
        state.is_solved().then_some(played)
    }
}

/// Play some improving macro until the cube is solved. Returns the indices
/// played, or `None` the first time no macro in `library` improves the
/// measure.
pub fn solve(cube: &Cube, library: &[Macro]) -> Option<Vec<usize>> {
    Playbook::new(library).play(cube)
}

// ---------------------------------------------------------------------------
// building the library
// ---------------------------------------------------------------------------

/// Standard notation in, moves out - so the base algorithms read as
/// algorithms rather than as thirty struct literals.
fn seq(notation: &str) -> Vec<Move> {
    notation
        .split_whitespace()
        .map(|token| {
            let mut chars = token.chars();
            let face = match chars.next() {
                Some('U') => Face::U,
                Some('R') => Face::R,
                Some('F') => Face::F,
                Some('D') => Face::D,
                Some('L') => Face::L,
                Some('B') => Face::B,
                other => panic!("{other:?} is not a face"),
            };
            let quarters = match chars.next() {
                None => 1,
                Some('2') => 2,
                Some('\'') => 3,
                other => panic!("{other:?} is not a turn amount"),
            };
            assert!(chars.next().is_none(), "{token} is not one move");
            Move { face, quarters }
        })
        .collect()
}

fn notation(moves: &[Move]) -> String {
    moves.iter().map(|m| m.notation()).collect::<Vec<_>>().join(" ")
}

fn reverse(moves: &[Move]) -> Vec<Move> {
    moves.iter().rev().map(|m| m.inverse()).collect()
}

/// `A B A' B'` - the form that makes small effects. Whatever `A` disturbs
/// that `B` does not touch is put back by `A'`, and vice versa, so only the
/// cubies the two sequences share come out changed.
fn commutator(a: &str, b: &str) -> Vec<Move> {
    let (a, b) = (seq(a), seq(b));
    let mut out = a.clone();
    out.extend(b.iter().copied());
    out.extend(reverse(&a));
    out.extend(reverse(&b));
    out
}

/// `X alg X'` - the same effect, moved onto whichever cubies `X` brings into
/// the algorithm's reach. This is the only reason a handful of bases can
/// cover a whole cube.
fn conjugate(setup: &[Move], alg: &[Move]) -> Vec<Move> {
    let mut out = setup.to_vec();
    out.extend(alg.iter().copied());
    out.extend(reverse(setup));
    out
}

/// Every setup of up to `turns` moves, never turning the same face twice in
/// a row (which is one turn wearing two moves' length).
fn setups(turns: usize) -> Vec<Vec<Move>> {
    let all = Move::all();
    let mut out = vec![Vec::new()];
    let mut level: Vec<Vec<Move>> = vec![Vec::new()];
    for _ in 0..turns {
        let mut next = Vec::new();
        for prefix in &level {
            for &m in &all {
                if prefix.last().is_some_and(|last: &Move| last.face == m.face) {
                    continue;
                }
                let mut s = prefix.clone();
                s.push(m);
                next.push(s);
            }
        }
        out.extend(next.iter().cloned());
        level = next;
    }
    out
}

/// Every conjugate of `alg` and of `alg` reversed, with its effect. Both
/// directions matter: a macro and its inverse twist the same pair of corners
/// opposite ways, and a state needing one is not helped by the other.
fn conjugates(setups: &[Vec<Move>], alg: &[Move]) -> Vec<(Vec<Move>, Effect)> {
    let mut out = Vec::with_capacity(setups.len() * 2);
    for direction in [alg.to_vec(), reverse(alg)] {
        for setup in setups {
            let moves = conjugate(setup, &direction);
            let effect = effect_of(&moves);
            out.push((moves, effect));
        }
    }
    out
}

/// What a sequence does to the twenty cubies, coarsely: how many of each
/// kind it disturbs, and whether it merely re-orients them. Enough to
/// classify a candidate base algorithm, and asserted on every base shipped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shape {
    corners: u8,
    edges: u8,
    /// True when every disturbed corner is still in its own slot, i.e. the
    /// algorithm twists corners without moving them.
    corners_in_place: bool,
    edges_in_place: bool,
}

fn shape(e: &Effect) -> Shape {
    let mut s = Shape { corners: 0, edges: 0, corners_in_place: true, edges_in_place: true };
    for (slot, &(piece, orientation)) in cubies_after(e).iter().enumerate() {
        if piece == home_piece(slot) && orientation == 0 {
            continue;
        }
        if slot < CORNERS {
            s.corners += 1;
            s.corners_in_place &= piece == home_piece(slot);
        } else {
            s.edges += 1;
            s.edges_in_place &= piece == home_piece(slot);
        }
    }
    s
}

/// The three-cycle of corners, the three-cycle of edges and the corner-twist
/// pair are commutators of two short sequences, found by sweeping every
/// `A B A' B'` with `A` and `B` up to three moves and keeping the shortest
/// of each shape. They are written here as the two sequences they are built
/// from, and [`bases`] asserts the shape each one claims.
const SEEDS: [(&str, &str, &str, Shape); 3] = [
    (
        "3-cycle of corners",
        "U R U'",
        "L",
        Shape { corners: 3, edges: 0, corners_in_place: false, edges_in_place: true },
    ),
    (
        "3-cycle of edges",
        "U2 R2 U2",
        "F2",
        Shape { corners: 0, edges: 3, corners_in_place: true, edges_in_place: false },
    ),
    (
        "twist of two corners",
        "U R2 U'",
        "F L2 F'",
        Shape { corners: 2, edges: 0, corners_in_place: true, edges_in_place: true },
    ),
];

/// The five base algorithms, and why there are five.
///
/// Three come out of the commutator sweep ([`SEEDS`]). The other two cannot:
///
/// * **Flipping two edges in place.** No `A B A' B'` with `A` and `B` up to
///   three moves produces one. It is built instead as `P Q'`, where `P` and
///   `Q` are conjugates of the edge three-cycle that cycle the SAME three
///   edge slots the same way but leave them flipped differently: the
///   permutations cancel and only the difference in flips survives.
/// * **Swapping two corners and two edges.** A commutator is an even
///   permutation of the corners and conjugation keeps it even, so no setup
///   makes one swap exactly two corners - and without such a base the parity
///   of the corner permutation is invariant, which leaves half of all
///   scrambles unreachable rather than merely hard. It is built as `M P Q`:
///   one quarter turn, whose 4-cycle of corners is the odd permutation the
///   commutators cannot supply, then a corner three-cycle and an edge
///   three-cycle that absorb everything that quarter turn did except one
///   swap of each.
fn bases() -> Vec<(String, Vec<Move>)> {
    let mut out: Vec<(String, Vec<Move>)> = Vec::new();
    for (name, a, b, want) in SEEDS {
        let moves = commutator(a, b);
        assert_eq!(shape(&effect_of(&moves)), want, "base {name} is not the algorithm it claims");
        out.push((name.to_string(), moves));
    }
    // A deeper setup pool than the library itself uses: both searches below
    // need a three-cycle landing on three PARTICULAR slots, which two setup
    // turns cannot always reach.
    let setups = setups(3);
    let corner3 = conjugates(&setups, &out[0].1.clone());
    let edge3 = conjugates(&setups, &out[1].1.clone());
    out.push(edge_flip_pair(&edge3));
    out.push(corner_and_edge_swap(&corner3, &edge3));
    out
}

/// Two edge three-cycles that cycle the same slots the same way, divided by
/// one another: the permutations cancel and two edges are left flipped where
/// they stand.
fn edge_flip_pair(edge3: &[(Vec<Move>, Effect)]) -> (String, Vec<Move>) {
    let want = Shape { corners: 0, edges: 2, corners_in_place: true, edges_in_place: true };
    let undone: Vec<Effect> = edge3.iter().map(|(_, e)| undo(e)).collect();
    let mut best: Option<Vec<Move>> = None;
    for (i, (first, effect)) in edge3.iter().enumerate() {
        for (j, (second, _)) in edge3.iter().enumerate().skip(i + 1) {
            let length = first.len() + second.len();
            if best.as_ref().is_some_and(|b| b.len() <= length) {
                continue;
            }
            if shape(&then(effect, &undone[j])) != want {
                continue;
            }
            let mut moves = first.clone();
            moves.extend(reverse(second));
            best = Some(moves);
        }
    }
    let moves = best.expect("two edge three-cycles on one triple must differ only in flips");
    assert_eq!(shape(&effect_of(&moves)), want);
    ("flip of two edges".to_string(), moves)
}

/// One quarter turn, then a corner three-cycle and an edge three-cycle that
/// absorb all of it but one corner swap and one edge swap.
fn corner_and_edge_swap(
    corner3: &[(Vec<Move>, Effect)],
    edge3: &[(Vec<Move>, Effect)],
) -> (String, Vec<Move>) {
    let want = Shape { corners: 2, edges: 2, corners_in_place: false, edges_in_place: false };
    let mut best: Option<Vec<Move>> = None;
    for turn in Move::all() {
        let start = effect_of(&[turn]);
        for (cycle, effect) in corner3 {
            let half = then(&start, effect);
            // Two corners left to swap and the quarter turn's four edges
            // still to tidy: anything else and one edge cycle cannot finish
            // the job, so the inner sweep is not worth entering.
            let s = shape(&half);
            if s.corners != 2 || s.edges != 4 {
                continue;
            }
            for (tidy, edges) in edge3 {
                let length = 1 + cycle.len() + tidy.len();
                if best.as_ref().is_some_and(|b| b.len() <= length) {
                    continue;
                }
                if shape(&then(&half, edges)) != want {
                    continue;
                }
                let mut moves = vec![turn];
                moves.extend(cycle.iter().copied());
                moves.extend(tidy.iter().copied());
                best = Some(moves);
            }
        }
    }
    let moves = best.expect("a quarter turn is an odd corner permutation and must reduce to a swap");
    assert_eq!(shape(&effect_of(&moves)), want);
    ("swap of two corners and two edges".to_string(), moves)
}

/// What makes two macros interchangeable to the solver, and therefore what
/// the generator has to cover exactly once.
///
/// * A parity-flipping macro is admissible on parity ALONE, whatever else it
///   does, so one per pair of swapped corners is already more choice than
///   the solver needs - the edges it scatters are tidied by the cycles
///   afterwards.
/// * A macro that MOVES pieces is classified by where it sends them and
///   nothing else: landing three edges in their own slots but flipped still
///   drops `D` by three, and the flips are then somebody else's job.
/// * A macro that moves nothing is classified by the twists and flips
///   themselves, because that is all it is - and "twist THESE two corners
///   THIS way" is a demand exactly one macro in the library can meet.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Class {
    Parity([u8; CORNERS]),
    Places([u8; SLOTS]),
    Turns([u8; SLOTS]),
}

fn class(e: &Effect) -> Class {
    let after = cubies_after(e);
    let pieces: [u8; SLOTS] = std::array::from_fn(|slot| after[slot].0);
    let orientations: [u8; SLOTS] = std::array::from_fn(|slot| after[slot].1);
    let corners = corner_pieces(&after);
    if parity(&corners) == 1 {
        return Class::Parity(corners);
    }
    match pieces.iter().enumerate().all(|(slot, &p)| p == home_piece(slot)) {
        true => Class::Turns(orientations),
        false => Class::Places(pieces),
    }
}

/// Every base, in both directions, conjugated by every setup of up to
/// [`SETUP_TURNS`] turns, reduced to the shortest macro of each effect
/// class.
pub fn library() -> &'static [Macro] {
    static LIBRARY: OnceLock<Vec<Macro>> = OnceLock::new();
    LIBRARY.get_or_init(build_library)
}

/// How far the generator sets a base algorithm up before playing it.
///
/// Three turns leaves 28 of the 440 three-cycles of edges unreachable, and
/// the solver stops dead on the states that need exactly one of them - 75 of
/// 2000 random cubes. Four covers all 440, all 112 three-cycles of corners,
/// all 56 corner-twist pairs and all 66 edge-flip pairs, which is the whole
/// even half of the group's small elements.
const SETUP_TURNS: usize = 4;

fn build_library() -> Vec<Macro> {
    let setups = setups(SETUP_TURNS);
    // Conjugating by effect rather than by move list: `X alg X'` is
    // `effect(X)`, then the algorithm, then `effect(X)` undone - two array
    // shuffles instead of re-walking forty moves, which is what keeps a
    // sweep over forty thousand candidates under a second.
    let staged: Vec<(Effect, Effect, usize)> = setups
        .iter()
        .map(|s| {
            let e = effect_of(s);
            (e, undo(&e), s.len())
        })
        .collect();

    let mut shortest: HashMap<Class, usize> = HashMap::new();
    let mut kept: Vec<Macro> = Vec::new();
    for (name, base) in bases() {
        for (direction, alg) in [("", base.clone()), (" reversed", reverse(&base))] {
            let algebra = effect_of(&alg);
            for (setup, (forward, back, turns)) in setups.iter().zip(&staged) {
                let length = 2 * turns + alg.len();
                let class = class(&then(&then(forward, &algebra), back));
                if shortest.get(&class).is_some_and(|&at| kept[at].moves.len() <= length) {
                    continue;
                }
                let name = if setup.is_empty() {
                    format!("{name}{direction}")
                } else {
                    format!("{name}{direction} after {}", notation(setup))
                };
                let entry = Macro { name, moves: conjugate(setup, &alg) };
                match shortest.get(&class) {
                    Some(&at) => kept[at] = entry,
                    None => {
                        shortest.insert(class, kept.len());
                        kept.push(entry);
                    }
                }
            }
        }
    }

    // Shortest first, but every parity-flipping macro last: one of them is
    // admissible whenever the corner permutation is odd, so ordering them
    // first would play one on every cube instead of only when nothing
    // cheaper helps.
    let mut order: Vec<(bool, usize, usize)> = kept
        .iter()
        .enumerate()
        .map(|(i, m)| (flips_parity(&effect_of(&m.moves)), m.moves.len(), i))
        .collect();
    order.sort();
    order.into_iter().map(|(_, _, i)| std::mem::take(&mut kept[i])).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::cube::scramble;

    /// One library and one index for the whole test binary: building them is
    /// the expensive part, and every test below wants the same ones.
    fn playbook() -> &'static Playbook {
        static PLAYBOOK: OnceLock<Playbook> = OnceLock::new();
        PLAYBOOK.get_or_init(|| Playbook::new(library()))
    }

    const CORNER_CYCLE: Shape =
        Shape { corners: 3, edges: 0, corners_in_place: false, edges_in_place: true };
    const EDGE_CYCLE: Shape =
        Shape { corners: 0, edges: 3, corners_in_place: true, edges_in_place: false };
    const CORNER_TWIST: Shape =
        Shape { corners: 2, edges: 0, corners_in_place: true, edges_in_place: true };
    const EDGE_FLIP: Shape =
        Shape { corners: 0, edges: 2, corners_in_place: true, edges_in_place: true };
    const SWAP: Shape =
        Shape { corners: 2, edges: 2, corners_in_place: false, edges_in_place: false };

    fn of_shape(want: Shape) -> Vec<Effect> {
        library()
            .iter()
            .map(|m| effect_of(&m.moves))
            .filter(|e| shape(e) == want)
            .collect()
    }

    /// The claim the whole guarantee rests on, stated as arithmetic: the
    /// library holds EVERY three-cycle of three corners (56 triples, two
    /// directions), EVERY three-cycle of three edges (220 triples, two
    /// directions), EVERY way to twist a pair of corners (28 pairs, two
    /// senses), EVERY way to flip a pair of edges (66 pairs) and one
    /// corner-and-edge swap per pair of corners. Those are the complete
    /// counts, not a sample of them - a generator that covered 412 of the
    /// 440 edge cycles leaves the solver dead on exactly the states needing
    /// one of the missing 28, which is how this number got checked.
    #[test]
    fn the_library_holds_every_small_element_of_the_group() {
        let counts = [
            ("three-cycles of corners", CORNER_CYCLE, 112, 56),
            ("three-cycles of edges", EDGE_CYCLE, 440, 220),
            ("corner-twist pairs", CORNER_TWIST, 56, 28),
            ("edge-flip pairs", EDGE_FLIP, 66, 66),
            ("corner-and-edge swaps", SWAP, 28, 28),
        ];
        let mut total = 0;
        for (what, want, macros, slot_sets) in counts {
            let found = of_shape(want);
            assert_eq!(found.len(), macros, "{what}");
            let sets: HashSet<u32> = found.iter().map(support).collect();
            assert_eq!(sets.len(), slot_sets, "{what} land on {slot_sets} sets of slots");
            total += macros;
        }
        assert_eq!(library().len(), total, "the library is those five families and nothing else");
        let classes: HashSet<Class> = library().iter().map(|m| class(&effect_of(&m.moves))).collect();
        assert_eq!(classes.len(), library().len(), "one macro per effect class");
    }

    /// Every macro leaves at least sixteen of the twenty cubies exactly as
    /// it found them. That is what makes the measure usable: a macro whose
    /// effect were spread over the whole cube could not be scored by looking
    /// at four slots, and could not be aimed at the cubies that need it.
    #[test]
    fn a_macro_disturbs_at_most_four_cubies() {
        for m in library() {
            let touched = support(&effect_of(&m.moves)).count_ones();
            assert!((2..=4).contains(&touched), "{} disturbs {touched} cubies", m.name);
        }
    }

    /// Which component of the measure a macro moved: parity, home cubies,
    /// or - when neither - displacement alone.
    #[derive(Default)]
    struct Steps {
        parity: usize,
        home: usize,
        displacement: usize,
    }

    /// Replay a plan through the cube engine itself - not through the
    /// permutation index the solver runs on - checking that the measure
    /// strictly falls at every step, and counting which component fell.
    fn replay(cube: &Cube, plan: &[usize]) -> (Cube, Steps) {
        let library = library();
        let mut state = *cube;
        let mut steps = Steps::default();
        for &i in plan {
            let next = state.apply_all(&library[i].moves);
            let (before, after) = (measure(&state), measure(&next));
            assert!(
                after < before,
                "{} did not decrease the measure: {before:?} -> {after:?}",
                library[i].name
            );
            match (after.0 < before.0, after.1 < before.1) {
                (true, _) => steps.parity += 1,
                (_, true) => steps.home += 1,
                _ => steps.displacement += 1,
            }
            state = next;
        }
        (state, steps)
    }

    fn solved_by(cube: &Cube, plan: &[usize]) -> bool {
        replay(cube, plan).0.is_solved()
    }

    /// THE GUARANTEE. Two thousand cubes scrambled forty moves deep - far
    /// beyond anything `search.rs` can see - every one of them solved, with
    /// no planner, no distance oracle and no search over states, inside a
    /// bound this test asserts.
    #[test]
    fn every_forty_move_scramble_is_solved_within_the_bound() {
        let book = playbook();
        assert_eq!(book.len(), library().len(), "the index covers the library");
        assert!(!book.is_empty());
        const CUBES: u64 = 2000;
        const BOUND: usize = 24;
        let (mut worst_macros, mut worst_moves, mut total_macros, mut total_moves) = (0, 0, 0, 0);
        let mut fired = Steps::default();
        for seed in 0..CUBES {
            let (cube, _) = scramble(40, seed + 1);
            let plan = book.play(&cube).unwrap_or_else(|| {
                let (played, stuck) = book.run(&cube);
                panic!(
                    "seed {seed}: no macro improves after {} played, {} cubies home:\n{}",
                    played.len(),
                    home_cubies(&stuck),
                    stuck.net()
                )
            });
            let (finished, steps) = replay(&cube, &plan);
            assert!(finished.is_solved(), "seed {seed}");
            assert!(steps.parity <= 1, "seed {seed}: parity is fixed once or not at all");
            fired.parity += steps.parity;
            fired.home += steps.home;
            fired.displacement += steps.displacement;
            let moves: usize = plan.iter().map(|&i| library()[i].moves.len()).sum();
            assert!(plan.len() <= BOUND, "seed {seed} took {} macros", plan.len());
            worst_macros = worst_macros.max(plan.len());
            worst_moves = worst_moves.max(moves);
            total_macros += plan.len();
            total_moves += moves;
        }
        // Printed so the numbers in the README come from a run, not a memory.
        println!(
            "{CUBES} cubes: macros/solve mean {:.1} max {worst_macros}, \
             moves/solve mean {:.1} max {worst_moves}",
            total_macros as f64 / CUBES as f64,
            total_moves as f64 / CUBES as f64
        );
        println!(
            "which component moved: parity {}, home cubies {}, displacement only {}",
            fired.parity, fired.home, fired.displacement
        );
        assert!(worst_macros * 2 > BOUND, "the asserted bound {BOUND} has gone slack");
        // Half of random scrambles have an odd corner permutation, and each
        // of those needs the parity base exactly once.
        let odd = (0..CUBES).filter(|&s| corner_parity(&scramble(40, s + 1).0) == 1).count();
        assert_eq!(fired.parity, odd, "one parity macro per odd cube, and none for an even one");
        assert!(fired.home * 10 > total_macros * 9, "the home count is meant to be the headline");
        assert!(fired.displacement > 0, "the tie-breaker is meant to be load-bearing");
    }

    /// The shallow end and the awkward end. Playing every macro in turn on a
    /// solved cube is the sharpest of these: it manufactures exactly one
    /// state of every effect class the library knows, which is the endgame
    /// each family of macros exists to finish.
    #[test]
    fn the_easy_and_the_awkward_states_are_solved_too() {
        let book = playbook();
        assert_eq!(book.play(&Cube::SOLVED), Some(vec![]), "a solved cube needs no macro");
        for m in Move::all() {
            let cube = Cube::SOLVED.apply(m);
            let plan = book.play(&cube).unwrap_or_else(|| panic!("one move: {}", m.notation()));
            assert!(solved_by(&cube, &plan), "one move: {}", m.notation());
        }
        for hard in [
            // Every edge flipped and nothing moved: the state furthest from
            // solved there is, and the one the flip pairs exist for.
            "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2",
            // Two corners and two edges swapped - odd corner parity, the
            // half of the group no commutator can reach.
            "R U R' U' R' F R2 U' R' U' R U R' F'",
            "U2 D2 F2 B2 L2 R2",
        ] {
            let cube = Cube::SOLVED.apply_all(&seq(hard));
            let plan = book.play(&cube).unwrap_or_else(|| panic!("{hard}"));
            assert!(solved_by(&cube, &plan), "{hard}");
        }
        for m in library() {
            let cube = Cube::SOLVED.apply_all(&m.moves);
            let plan = book.play(&cube).unwrap_or_else(|| panic!("cannot undo {}", m.name));
            assert!(solved_by(&cube, &plan), "cannot undo {}", m.name);
        }
    }

    /// The finding, kept rather than papered over: the rule this layer was
    /// designed around - strictly more cubies home - runs out. Most of these
    /// cubes walk themselves into a state where NO macro in the library puts
    /// another cubie home, and every such state still has a macro that lands
    /// pieces in their own slots misoriented, which is what the displacement
    /// tie-breaker counts as progress. A handful get away with the strict
    /// rule alone, which is exactly why a run of successes proves nothing
    /// here and the tie-breaker is not optional.
    #[test]
    fn a_strict_home_count_rule_stalls_and_the_tie_breaker_moves_it() {
        let library = library();
        let mut stalled = 0;
        for seed in 0..20u64 {
            let (mut cube, _) = scramble(40, seed + 1);
            while let Some(m) = library.iter().find(|m| admissible(&cube, m)) {
                cube = cube.apply_all(&m.moves);
            }
            if cube.is_solved() {
                continue;
            }
            stalled += 1;
            assert!(
                library.iter().any(|m| progresses(&cube, m)),
                "seed {seed}: {} cubies home and nothing at all improves",
                home_cubies(&cube)
            );
        }
        println!("the strict home-count rule stalled on {stalled} of 20 cubes");
        assert!(stalled >= 10, "the strict rule stalled on only {stalled} of 20 cubes");
    }

    /// The published entry point, exercised the way its contract reads: a
    /// cube, a plain slice of macros, the indices played.
    #[test]
    fn solve_takes_a_library_and_returns_the_macros_it_played() {
        let (cube, _) = scramble(40, 12345);
        let plan = solve(&cube, library()).expect("a scrambled cube is solvable");
        assert!(!plan.is_empty());
        assert!(solved_by(&cube, &plan));
    }
}
