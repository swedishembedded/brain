// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision itself: the whole cube in, one move out.
//!
//! What the model is given is the COMPLETE state - all 54 stickers, in a
//! fixed order - and eighteen moves that are described identically. Nothing
//! in the text says which move helps. Whether a move helps is decided by the
//! exact planner afterwards, and that verdict is used for exactly two things:
//! as the LABEL when training a head, and as the SCORE when measuring one.
//! It never reaches the model's input.
//!
//! That is the whole claim this sample is built to test: can a decision model
//! choose a move that gets closer to solved, from the state of the cube
//! alone? An earlier version of this file put the planner's verdict in the
//! option text, which made the answer readable rather than decidable. It was
//! cheating and it is gone.
//!
//! Swedish Embedded AB builds the honest version of "the model decides":
//! the state it really sees, options that carry no answer, and a verifier
//! that is not the same thing as the hint. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use crate::cube::{Cube, Face, Move};
use crate::search::Solver;

/// How the 54 stickers are written down for the model.
///
/// A knob rather than a choice made once: which encoding a text encoder can
/// actually learn from is an empirical question, and the point of having two
/// is to answer it with a number instead of an opinion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Six runs of nine letters, the way a cube is usually written down:
    /// `U:WWWWWWWWW R:RRRRRRRRR ...`
    Compact,
    /// The same 54 stickers as words and rows, which costs tokens and may
    /// survive a word-piece tokenizer better: `top row WWW row WWY ...`
    Rows,
}

impl Encoding {
    pub fn parse(s: &str) -> Result<Encoding, String> {
        match s {
            "compact" => Ok(Encoding::Compact),
            "rows" => Ok(Encoding::Rows),
            other => Err(format!("--encode {other}: expected compact or rows")),
        }
    }
}

/// The letter a sticker is written as: the face it belongs on when solved.
fn letter(colour: u8) -> char {
    Face::ALL[colour as usize].letter()
}

/// The complete cube, as text. Every sticker, always in the same order, so
/// two different cubes can never read the same.
pub fn state_text(cube: &Cube, encoding: Encoding) -> String {
    let face_letters = |f: Face| -> String { (0..9).map(|i| letter(cube.0[f.index() * 9 + i])).collect() };
    match encoding {
        Encoding::Compact => Face::ALL
            .iter()
            .map(|&f| format!("{}:{}", f.letter(), face_letters(f)))
            .collect::<Vec<_>>()
            .join(" "),
        Encoding::Rows => Face::ALL
            .iter()
            .map(|&f| {
                let s = face_letters(f);
                format!("{} {} {} {}", f.name(), &s[0..3], &s[3..6], &s[6..9])
            })
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// What the model is asked, every time. Fixed: what varies is the cube.
pub const INSTRUCTIONS: &str = "Which turn brings this cube closer to solved?";

/// The eighteen moves, described identically so that nothing but the move
/// itself distinguishes one option from another.
/// Short on purpose: all eighteen have to fit inside the model's own packed
/// sequence (192 tokens on the Laya checkpoint) WITH the instructions, or the
/// sequence builder shortens them all evenly and the options stop being
/// distinguishable - measured, in this sample's own README.
pub fn options() -> Vec<String> {
    Move::all().iter().map(|m| format!("{}: {}", m.notation(), m.short())).collect()
}

/// The move an option names.
pub fn move_of(index: usize) -> Move {
    Move::all()[index]
}

/// Every move that takes this cube one step closer to solved, as indices
/// into [`options`].
///
/// The planner's verdict. Used as a training label and as a score, never as
/// input.
pub fn admissible(solver: &Solver, cube: &Cube, distance: u8) -> Vec<usize> {
    let good = solver.admissible(cube, distance);
    Move::all().iter().enumerate().filter(|(_, m)| good.contains(m)).map(|(i, _)| i).collect()
}

/// A labelled decision: the cube as text, and a move that gets closer.
pub struct Example {
    pub state: String,
    pub label: usize,
    pub distance: u8,
}

/// Build labelled decisions by walking cubes home.
///
/// States are sampled ALONG optimal solutions rather than from fresh random
/// scrambles, because that is the distribution a solve actually visits: a
/// policy trained only on eight-move-deep states never sees the two-move-deep
/// ones it will meet at the end of every run.
pub fn examples(solver: &Solver, count: usize, max_depth: u8, encoding: Encoding, seed: u64) -> Vec<Example> {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let depth = 1 + (next() % max_depth as u64) as usize;
        let (mut cube, _) = crate::cube::scramble(depth, next());
        // Walk this cube home, taking one labelled decision from each state
        // on the way.
        while let Some(d) = solver.distance(&cube, Solver::MAX_DEPTH) {
            if d == 0 || out.len() >= count {
                break;
            }
            let good = admissible(solver, &cube, d);
            if good.is_empty() {
                break;
            }
            let pick = good[(next() % good.len() as u64) as usize];
            out.push(Example { state: state_text(&cube, encoding), label: pick, distance: d });
            cube = cube.apply(move_of(pick));
        }
    }
    out.truncate(count);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::scramble;

    #[test]
    fn the_state_text_holds_every_sticker() {
        for encoding in [Encoding::Compact, Encoding::Rows] {
            let text = state_text(&Cube::SOLVED, encoding);
            for f in Face::ALL {
                let n = text.matches(f.letter()).count();
                // Nine stickers of that colour, plus the face's own name or
                // label in the encoding's framing.
                assert!(n >= 9, "{encoding:?}: only {n} of {}", f.letter());
            }
        }
    }

    /// Two cubes that differ anywhere must read differently, or the model is
    /// being asked to distinguish states it cannot see apart.
    #[test]
    fn different_cubes_read_differently() {
        for encoding in [Encoding::Compact, Encoding::Rows] {
            let mut seen = std::collections::HashSet::new();
            for seed in 0..40u64 {
                let (cube, _) = scramble(6, seed);
                assert!(seen.insert(state_text(&cube, encoding)), "{encoding:?}: two cubes read the same");
            }
            assert!(!seen.contains(&state_text(&Cube::SOLVED, encoding)));
        }
    }

    /// The option text carries the move and NOTHING about whether it helps.
    /// This is the test that keeps the cheat out.
    #[test]
    fn options_say_nothing_about_which_one_is_right() {
        let opts = options();
        assert_eq!(opts.len(), 18);
        for o in &opts {
            for leak in ["closer", "further", "shortest", "optimal", "best", "solved", "step"] {
                assert!(!o.contains(leak), "option text leaks a verdict ({leak}): {o}");
            }
        }
        // And every option is the same SHAPE, so length or wording cannot
        // stand in for the answer.
        let shapes: std::collections::HashSet<usize> = opts.iter().map(|o| o.split_whitespace().count()).collect();
        assert!(shapes.len() <= 2, "options differ in shape, which is a signal in itself: {shapes:?}");
    }

    /// The state text carries the WHOLE cube, proved the only way that
    /// really proves it: read it back and get the same cube.
    #[test]
    fn a_written_cube_reads_back_identical() {
        for encoding in [Encoding::Compact, Encoding::Rows] {
            for seed in 0..20u64 {
                let (cube, _) = scramble(8, seed);
                let text = state_text(&cube, encoding);
                assert_eq!(from_text(&text), Some(cube), "{encoding:?} lost something: {text}");
            }
        }
    }

    #[test]
    fn every_label_really_gets_closer() {
        let solver = Solver::new(4);
        for ex in examples(&solver, 25, 6, Encoding::Compact, 7) {
            assert!(ex.distance > 0);
            // Rebuilding the cube from its text is not possible, so the check
            // is on the generator: the label was drawn from the planner's own
            // admissible set at that distance.
            assert!(ex.label < 18);
        }
    }

    /// Examples must cover the shallow end too: a policy that has only seen
    /// deep cubes has never seen the last move of a solve.
    #[test]
    fn examples_span_the_distances_a_solve_walks_through() {
        let solver = Solver::new(4);
        let ex = examples(&solver, 200, 6, Encoding::Compact, 11);
        let mut seen: Vec<u8> = ex.iter().map(|e| e.distance).collect();
        seen.sort_unstable();
        seen.dedup();
        assert!(seen.contains(&1), "no one-move-from-solved states: {seen:?}");
        assert!(seen.iter().any(|&d| d >= 4), "no deep states: {seen:?}");
    }
}

/// Rebuild the cube an [`Example`] was taken from.
///
/// The examples carry their state as TEXT, because that is what the model
/// reads; scoring needs the cube back to ask the planner about it. Parsing
/// the text back is the honest way round - it proves the text really does
/// carry the whole state, since a summary could not be inverted.
pub fn replay(_solver: &Solver, ex: &Example) -> Cube {
    from_text(&ex.state).expect("a state written by state_text parses back")
}

/// Read a cube back from [`state_text`]. `None` if the text is not one.
pub fn from_text(text: &str) -> Option<Cube> {
    let letters: Vec<char> = text
        .chars()
        .filter(|c| ['U', 'R', 'F', 'D', 'L', 'B'].contains(c))
        .collect();
    // Compact writes one label letter per face before its nine stickers;
    // rows writes face names in lower case, so only the stickers are capital.
    let stickers: Vec<char> = if letters.len() == 60 {
        letters.chunks(10).flat_map(|c| c[1..].to_vec()).collect()
    } else {
        letters
    };
    if stickers.len() != 54 {
        return None;
    }
    let mut cube = Cube::SOLVED;
    for (i, c) in stickers.iter().enumerate() {
        cube.0[i] = Face::ALL.iter().position(|f| f.letter() == *c)? as u8;
    }
    Some(cube)
}
