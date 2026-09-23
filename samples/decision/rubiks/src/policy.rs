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
/// actually learn from is an empirical question, and the point of having
/// three is to answer it with a number instead of an opinion. The numbers
/// are in this sample's README, and they are not close.
///
/// **The encoding decides whether the cube is legible at all.** The encoder
/// is a WordPiece model: it cuts a run of letters at whatever boundaries its
/// own vocabulary happens to have, so `WWYWWWWWG` and `WWWWWWWWW` become
/// seven tokens and one. Pack the stickers into runs and a state's token
/// COUNT moves with its content - every sticker after the first difference
/// shifts row, and the learned position embedding, which is the only thing
/// that says which sticker is which, is reading a different sticker at every
/// row. The model is then being asked to read a cube it structurally cannot
/// see. That is measured, not argued: `crates/decide/tests/state_tokenization.rs`
/// holds both halves of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// One sticker, one token, at the row its own index names:
    /// `U U U R R F D D D ...`, all 54, whitespace between every pair.
    ///
    /// Whitespace is what buys it - the tokenizer cuts on whitespace before
    /// WordPiece runs, and a lone letter is in every vocabulary of this
    /// family - so the stream is 54 tokens for EVERY cube and row `i` is
    /// always sticker `i`. The face needs no label: which face a sticker
    /// belongs to is a fixed function of its index, so the position
    /// embedding already carries it.
    Grid,
    /// Six runs of nine letters, the way a cube is usually written down:
    /// `U:WWWWWWWWW R:RRRRRRRRR ...`
    ///
    /// Kept as the control this sample's README compares against, not as a
    /// recommendation.
    Compact,
    /// The same 54 stickers as words and rows, which costs tokens and was
    /// the first attempt at surviving a word-piece tokenizer:
    /// `top WWW WWY ...`. It does not - the runs are only shorter.
    Rows,
}

impl Encoding {
    pub fn parse(s: &str) -> Result<Encoding, String> {
        match s {
            "grid" => Ok(Encoding::Grid),
            "compact" => Ok(Encoding::Compact),
            "rows" => Ok(Encoding::Rows),
            other => Err(format!("--encode {other}: expected grid, compact or rows")),
        }
    }

    /// The spelling [`Encoding::parse`] accepts, so a run that records its
    /// encoding names it the same way the flag that selected it did.
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Grid => "grid",
            Encoding::Compact => "compact",
            Encoding::Rows => "rows",
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
        Encoding::Grid => {
            let mut out = String::with_capacity(54 * 2);
            for (i, &c) in cube.0.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push(letter(c));
            }
            out
        }
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

/// Build labelled decisions by walking cubes home, BALANCED across distances.
///
/// States are sampled along optimal solutions, because walking a cube home is
/// what makes a label free: the move that undoes each step is, by
/// construction, a move that gets closer. The planner then verifies it.
///
/// **Balanced is the whole point, and it was learned the expensive way.**
/// Walking home yields one state at every distance from the scramble depth
/// down to 1, so a distance of 1 appears in every walk and the deepest
/// distance appears only in the walks that started there. Left alone that is
/// a roughly 5:1 shallow bias, and a policy fitted on it is excellent at the
/// states it saw most and weak at the rest - measured on this sample, 100%
/// at one move from solved against 44% at six.
///
/// That skew does not show up in a held-out score, because a held-out split
/// drawn the same way is skewed the same way. It shows up when the policy
/// DRIVES: a solve spends most of its turns at the deep end, so the run that
/// scored 87% on held-out decisions picked a shortest move on 37% of the
/// turns it actually took, and solved 9 cubes in 50. The fix is to stop
/// letting the sampling method decide the curriculum - every distance gets
/// the same quota, and a walk stops contributing to a distance once that
/// quota is full.
pub fn examples(solver: &Solver, count: usize, max_depth: u8, encoding: Encoding, seed: u64) -> Vec<Example> {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let buckets = max_depth as usize;
    let quota = count.div_ceil(buckets);
    let mut filled = vec![0usize; buckets + 1];
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        // Start where the data is still missing. Drawing the depth uniformly
        // instead would keep re-walking the shallow states long after their
        // quota was met, because every walk passes through them.
        let wanted: Vec<usize> = (1..=buckets).filter(|&d| filled[d] < quota).collect();
        if wanted.is_empty() {
            break;
        }
        let depth = wanted[(next() % wanted.len() as u64) as usize];
        let (mut cube, _) = crate::cube::scramble(depth, next());
        // Walk this cube home, taking one labelled decision from each state
        // on the way - but only banking the ones whose distance still has
        // room.
        while let Some(d) = solver.distance(&cube, Solver::MAX_DEPTH) {
            if d == 0 || out.len() >= count {
                break;
            }
            let good = admissible(solver, &cube, d);
            if good.is_empty() {
                break;
            }
            let pick = good[(next() % good.len() as u64) as usize];
            let slot = d as usize;
            if slot <= buckets && filled[slot] < quota {
                filled[slot] += 1;
                out.push(Example { state: state_text(&cube, encoding), label: pick, distance: d });
            }
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

    /// Every encoding this sample offers, so a new one cannot be added
    /// without meeting the properties below.
    const ALL: [Encoding; 3] = [Encoding::Grid, Encoding::Compact, Encoding::Rows];

    /// THE property that makes the cube legible: the default encoding writes
    /// 54 lone symbols with whitespace between them, so the tokenizer emits
    /// one token per sticker and row `i` is sticker `i` for every cube.
    ///
    /// Asserted on the TEXT rather than on token ids because a sample may
    /// only depend on the SDK, and the SDK does not publish a tokenizer. The
    /// step from "lone whitespace-separated symbols" to "one token each" is
    /// the tokenizer's own, and it is pinned where the tokenizer lives:
    /// `crates/decide/tests/state_tokenization.rs`.
    #[test]
    fn the_grid_encoding_writes_one_lone_symbol_per_sticker() {
        let mut lengths = std::collections::BTreeSet::new();
        for seed in 0..40u64 {
            let (cube, _) = scramble(8, seed);
            let text = state_text(&cube, Encoding::Grid);
            let symbols: Vec<&str> = text.split_whitespace().collect();
            assert_eq!(symbols.len(), 54, "{text}");
            for (i, s) in symbols.iter().enumerate() {
                let mut chars = s.chars();
                let c = chars.next().expect("a symbol");
                assert!(chars.next().is_none(), "symbol {i} is {s:?}, not a lone character");
                assert!(Face::ALL.iter().any(|f| f.letter() == c), "symbol {i} is {c:?}");
            }
            lengths.insert(symbols.len());
        }
        // The point of all of it: the length does not move with the content.
        assert_eq!(lengths, std::collections::BTreeSet::from([54]));
    }

    #[test]
    fn the_state_text_holds_every_sticker() {
        for encoding in ALL {
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
        for encoding in ALL {
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
        for encoding in ALL {
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

    /// THE sampling property: every distance the policy will meet gets the
    /// same amount of data.
    ///
    /// Walking cubes home yields one state at every distance from the
    /// scramble depth down to 1, so distance 1 appears in every walk and the
    /// deepest distance only in the walks that started there. A policy fitted
    /// on that is good at the shallow states and weak at the deep ones, and a
    /// held-out split drawn the same way is skewed the same way and reports
    /// it as fine. Only driving the policy reveals it - which is far too late
    /// and far too slow to be the thing that catches it.
    /// A recorded encoding has to be readable back, or the provenance names
    /// a setting nobody can act on.
    #[test]
    fn every_encoding_name_parses_back_to_itself() {
        for e in [Encoding::Grid, Encoding::Compact, Encoding::Rows] {
            assert_eq!(Encoding::parse(e.name()), Ok(e), "{} did not round-trip", e.name());
        }
    }

    #[test]
    fn examples_are_balanced_across_the_distances_a_solve_meets() {
        let solver = Solver::new(4);
        let max_depth = 6u8;
        let ex = examples(&solver, 600, max_depth, Encoding::Grid, 11);
        assert_eq!(ex.len(), 600);
        let mut per = std::collections::BTreeMap::new();
        for e in &ex {
            assert!(e.distance >= 1 && e.distance <= max_depth, "distance {} out of range", e.distance);
            *per.entry(e.distance).or_insert(0usize) += 1;
        }
        assert_eq!(per.len(), max_depth as usize, "a distance got no examples at all: {per:?}");
        let (lo, hi) = (per.values().min().copied().unwrap(), per.values().max().copied().unwrap());
        // Equal up to the rounding of `count / buckets`, NOT merely "present":
        // "at least one deep example" is what the skewed sampler already
        // satisfied.
        assert!(hi - lo <= 1, "distances are not balanced: {per:?}");
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
