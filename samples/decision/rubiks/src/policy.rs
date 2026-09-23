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

    /// Examples must cover the shallow end too: a policy that has only seen
    /// deep cubes has never seen the last move of a solve.
    /// On-policy states really are the policy's own, and still carry a
    /// correct label: the distinguishing check is that a DIFFERENT policy
    /// produces a different set of states, which a backward walk could not.
    #[test]
    fn on_policy_states_follow_the_policy_and_stay_labelled() {
        let solver = Solver::new(4);
        let mut likes_r = |_: &Cube| -> Vec<f32> {
            Move::all().iter().map(|m| if m.face == crate::cube::Face::R { 1.0 } else { 0.01 }).collect()
        };
        let mut likes_f = |_: &Cube| -> Vec<f32> {
            Move::all().iter().map(|m| if m.face == crate::cube::Face::F { 1.0 } else { 0.01 }).collect()
        };
        let a = on_policy_examples(&solver, 60, 6, Encoding::Compact, 5, &mut likes_r);
        let b = on_policy_examples(&solver, 60, 6, Encoding::Compact, 5, &mut likes_f);
        assert_eq!(a.len(), 60);
        let sa: Vec<&String> = a.iter().map(|e| &e.state).collect();
        let sb: Vec<&String> = b.iter().map(|e| &e.state).collect();
        assert_ne!(sa, sb, "two different policies visited the same states: the rollout is not on-policy");

        // Every label is still the planner's, so the data is as trustworthy
        // as the backward-generated kind.
        for ex in &a {
            let cube = from_text(&ex.state).expect("parses back");
            assert_eq!(solver.distance(&cube, Solver::MAX_DEPTH), Some(ex.distance));
            let moved = cube.apply(move_of(ex.label));
            assert_eq!(solver.distance(&moved, Solver::MAX_DEPTH), Some(ex.distance - 1));
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

/// States the POLICY ITSELF walks into, labelled by the planner.
///
/// The gap this exists to close: a policy trained only on backward-generated
/// states scores well on states like those and badly on its own
/// trajectories - measured on this sample at 57.5% held out against 13% on
/// its own play. Backward generation produces states that lie on a shortest
/// path out of the solved cube; one wrong move puts the policy somewhere no
/// such walk ever reaches, and it has never seen anything like it.
///
/// So: roll the policy out, keep what it visits, and label those with the
/// planner. The labels are exactly as trustworthy as before (the planner is
/// exact), but the STATES are the ones that actually come up. This is
/// dataset aggregation - the standard answer to a policy that is only good
/// where it was taught - and it is why the rollout deliberately follows the
/// model rather than a shortest path.
///
/// A rollout stops when the cube drifts past what the planner can label,
/// because an unlabelled state is not a training example.
pub fn on_policy_examples(
    solver: &Solver,
    count: usize,
    max_depth: u8,
    encoding: Encoding,
    seed: u64,
    score: &mut dyn FnMut(&Cube) -> Vec<f32>,
) -> Vec<Example> {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let all = Move::all();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let depth = 1 + (next() % max_depth as u64) as usize;
        let (mut cube, _) = crate::cube::scramble(depth, next());
        let mut last: Option<Face> = None;
        let mut visited: std::collections::HashSet<Cube> = std::collections::HashSet::new();
        visited.insert(cube);

        for _ in 0..(max_depth as usize * 2) {
            let Some(d) = solver.distance(&cube, Solver::MAX_DEPTH) else {
                break; // drifted past what the planner can label
            };
            if d == 0 || out.len() >= count {
                break;
            }
            let good = admissible(solver, &cube, d);
            if good.is_empty() {
                break;
            }
            // The example is THIS state with a correct answer, whatever the
            // policy is about to do with it.
            out.push(Example { state: state_text(&cube, encoding), label: good[0], distance: d });

            // Now move the way the POLICY would, under the same guards the
            // real run uses - that is what makes the next state on-policy.
            let probs = score(&cube);
            let mut order: Vec<usize> = (0..all.len()).collect();
            order.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
            let chosen = order
                .iter()
                .copied()
                .find(|&i| allowed(all[i], last) && !visited.contains(&cube.apply(all[i])))
                .unwrap_or(order[0]);
            cube = cube.apply(all[chosen]);
            last = Some(all[chosen].face);
            visited.insert(cube);
        }
    }
    out.truncate(count);
    out
}

/// May this move follow `last`, given the move before that?
///
/// Two rules, both SOUND rather than heuristic - they remove only sequences
/// that cannot be part of any shortest solution, so nothing is lost:
///
/// 1. **Never the same face twice in a row.** `R R` is `R2`, `R R'` is
///    nothing; either way a shorter sequence reaches the same cube.
/// 2. **Opposite faces in a canonical order.** `U` and `D` commute, so `D U`
///    and `U D` reach the same cube by different paths - allowing only one of
///    the two orders halves that part of the tree without removing any state.
///
/// This is standard move pruning, and it is also the fix for the loop a
/// memoryless greedy policy falls into: `B B B B` returns the cube to where
/// it started, so a deterministic policy re-enters the same state and makes
/// the same choice forever. Rule 1 alone breaks that fixed point.
pub fn allowed(candidate: Move, last: Option<Face>) -> bool {
    let Some(last) = last else { return true };
    if candidate.face == last {
        return false;
    }
    // Opposite pairs, smaller index first: U before D, R before L, F before B.
    let opposite = |f: Face| match f {
        Face::U => Face::D,
        Face::D => Face::U,
        Face::R => Face::L,
        Face::L => Face::R,
        Face::F => Face::B,
        Face::B => Face::F,
    };
    !(opposite(candidate.face) == last && candidate.face.index() > last.index())
}

/// A solution found by searching with the MODEL as the heuristic.
pub struct Plan {
    pub moves: Vec<Move>,
    /// How many states the search scored - the cost of the plan, in model
    /// calls, so a reader can see what the heuristic bought.
    pub scored: usize,
    /// Did it reach the solved cube, or is this the best it could do?
    pub solved: bool,
}

/// Beam search over move sequences, ranked by the model's own log
/// probabilities.
///
/// This is the part that lets a weak policy solve a cube it cannot solve
/// greedily, and it is the standard shape: the network is a heuristic, the
/// search is what turns a heuristic into a solution. Greedy play follows one
/// line and dies on the first wrong step; a beam keeps `width` lines alive,
/// so a policy that is right most of the time recovers from being wrong some
/// of the time.
///
/// **The planner is not consulted.** Nothing in here asks how far from
/// solved anything is - the ranking is the model's, and the only oracle is
/// `Cube::is_solved`, which is the puzzle's own rule rather than a hint about
/// it. A search that consulted the planner would be the planner solving the
/// cube with the model watching.
///
/// `score` is supplied by the caller so this function needs no pipeline and
/// can be tested against a known-good policy.
pub fn beam_search(
    start: &Cube,
    width: usize,
    depth: usize,
    score: &mut dyn FnMut(&Cube) -> Vec<f32>,
) -> Plan {
    let all = Move::all();
    // (cube, path, cumulative log-probability)
    let mut beam: Vec<(Cube, Vec<Move>, f32)> = vec![(*start, Vec::new(), 0.0)];
    let mut scored = 0usize;
    let mut best: Option<(Vec<Move>, f32)> = None;
    // The closed set: a state reached by two different paths is one state,
    // and expanding it twice spends beam width on a duplicate. This is the
    // difference between tree search and graph search, and on a puzzle whose
    // moves are reversible it is also what stops the beam from wandering in
    // circles.
    let mut seen: std::collections::HashSet<Cube> = std::collections::HashSet::new();
    seen.insert(*start);

    for _ in 0..depth {
        let mut next: Vec<(Cube, Vec<Move>, f32)> = Vec::new();
        for (cube, path, logp) in &beam {
            if cube.is_solved() {
                return Plan { moves: path.clone(), scored, solved: true };
            }
            let probs = score(cube);
            scored += 1;
            for (i, p) in probs.iter().enumerate() {
                if !allowed(all[i], path.last().map(|m: &Move| m.face)) {
                    continue;
                }
                let mut child = path.clone();
                child.push(all[i]);
                let moved = cube.apply(all[i]);
                let lp = logp + p.max(1e-9).ln();
                if moved.is_solved() {
                    return Plan { moves: child, scored, solved: true };
                }
                if !seen.insert(moved) {
                    continue;
                }
                next.push((moved, child, lp));
            }
        }
        if next.is_empty() {
            break;
        }
        // Keep the `width` most likely lines. Sorting by cumulative
        // log-probability is what makes this the model's search rather than
        // a breadth-first sweep wearing its coat.
        next.sort_by(|a, b| b.2.total_cmp(&a.2));
        next.truncate(width.max(1));
        if let Some((_, path, lp)) = next.first() {
            if best.as_ref().map(|(_, b)| *lp > *b).unwrap_or(true) {
                best = Some((path.clone(), *lp));
            }
        }
        beam = next;
    }
    Plan { moves: best.map(|(p, _)| p).unwrap_or_default(), scored, solved: false }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use crate::cube::scramble;
    use crate::search::Solver;

    /// With a policy that knows the answer, the search finds a real solution
    /// - which is the check that the search itself is sound, separate from
    /// any question about the model.
    #[test]
    fn a_perfect_policy_solves_through_the_search() {
        let solver = Solver::new(4);
        for seed in 0..6u64 {
            let (cube, _) = scramble(7, seed + 3000);
            let mut oracle = |c: &Cube| -> Vec<f32> {
                let d = solver.distance(c, Solver::MAX_DEPTH).unwrap_or(0);
                let good = admissible(&solver, c, d);
                (0..18).map(|i| if good.contains(&i) { 1.0 } else { 0.0001 }).collect()
            };
            let plan = beam_search(&cube, 8, 10, &mut oracle);
            assert!(plan.solved, "seed {seed}: a perfect policy did not solve through the search");
            assert!(cube.apply_all(&plan.moves).is_solved(), "the plan does not solve the cube");
        }
    }

    /// A beam recovers where greedy dies: a policy that is right most of the
    /// time but puts its mass on a wrong move at one state still solves,
    /// because the right line stays in the beam.
    #[test]
    fn the_beam_recovers_from_a_policy_that_is_sometimes_wrong() {
        let solver = Solver::new(4);
        let (cube, _) = scramble(5, 4242);
        let d0 = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
        let mut calls = 0usize;
        let mut flaky = |c: &Cube| -> Vec<f32> {
            calls += 1;
            let d = solver.distance(c, Solver::MAX_DEPTH).unwrap_or(0);
            let good = admissible(&solver, c, d);
            // On the very first state it is confidently WRONG.
            if calls == 1 {
                return (0..18).map(|i| if good.contains(&i) { 0.01 } else { 1.0 }).collect();
            }
            (0..18).map(|i| if good.contains(&i) { 1.0 } else { 0.05 }).collect()
        };
        let plan = beam_search(&cube, 12, (d0 + 4) as usize, &mut flaky);
        assert!(plan.solved, "the beam did not recover from one confident mistake");
        assert!(cube.apply_all(&plan.moves).is_solved());
    }

    /// The rules remove only sequences that cannot be shortest, and they
    /// remove the one that makes a memoryless policy loop.
    #[test]
    fn pruning_keeps_every_state_reachable_and_kills_the_loop() {
        use crate::cube::Face;
        let m = |f: Face, q: u8| Move { face: f, quarters: q };
        // Rule 1: never the same face twice, whatever the amount.
        for q in 1..=3u8 {
            assert!(!allowed(m(Face::B, q), Some(Face::B)), "B after B must be pruned");
        }
        // Rule 2: opposite faces in one canonical order only.
        assert!(allowed(m(Face::U, 1), Some(Face::D)) != allowed(m(Face::D, 1), Some(Face::U)));
        // Everything else stays legal, or the search would lose solutions.
        assert!(allowed(m(Face::R, 1), Some(Face::U)));
        assert!(allowed(m(Face::F, 2), Some(Face::R)));
        assert!(allowed(m(Face::U, 1), None));
    }

    /// The loop the videos showed: a policy that always wants the same face
    /// used to play it forever, because four of them return the cube to
    /// where it started and a memoryless policy then repeats itself. With
    /// pruning it cannot, whatever it wants.
    #[test]
    fn a_policy_that_always_wants_one_face_cannot_play_it_twice() {
        use crate::cube::Face;
        let mut stuck = |_: &Cube| -> Vec<f32> {
            Move::all().iter().map(|m| if m.face == Face::B { 1.0 } else { 0.0001 }).collect()
        };
        let (cube, _) = scramble(5, 12345);
        let plan = beam_search(&cube, 4, 8, &mut stuck);
        for pair in plan.moves.windows(2) {
            assert_ne!(pair[0].face, pair[1].face, "the same face twice: {:?}", plan.moves);
        }
    }

    /// A state reached two ways is one state. Without the closed set the
    /// beam spends its width on duplicates.
    #[test]
    fn the_search_never_expands_the_same_state_twice() {
        let (cube, _) = scramble(6, 99);
        let mut uniform = |_: &Cube| -> Vec<f32> { vec![1.0 / 18.0; 18] };
        let plan = beam_search(&cube, 64, 6, &mut uniform);
        // Every state the plan walks through is distinct - a plan that
        // revisited one would be carrying a cycle.
        let mut walked = vec![cube];
        let mut c = cube;
        for m in &plan.moves {
            c = c.apply(*m);
            walked.push(c);
        }
        let unique: std::collections::HashSet<_> = walked.iter().collect();
        assert_eq!(unique.len(), walked.len(), "the plan walks through a state twice");
    }

    /// A useless policy does not accidentally solve a deep cube - otherwise
    /// the search would be doing the work and the model's contribution could
    /// not be read off the result.
    #[test]
    fn a_uniform_policy_does_not_solve_a_deep_cube_by_luck() {
        let (cube, _) = scramble(8, 77);
        let mut uniform = |_: &Cube| -> Vec<f32> { vec![1.0 / 18.0; 18] };
        let plan = beam_search(&cube, 8, 8, &mut uniform);
        assert!(!plan.solved, "a uniform policy solved an 8-move cube: the search is doing the deciding");
    }
}
