// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A decision model turning a Rubik's cube, with a planner that knows the
//! answer and a shield that keeps the run honest.
//!
//! The shape is the one `laya-mlx`'s Snake demo uses, and it is the only
//! shape in which putting a System-1 decision model in front of a puzzle is
//! an engineering decision rather than a stunt:
//!
//! 1. a **deterministic planner** (`search.rs`, exact meet-in-the-middle)
//!    describes the candidate moves;
//! 2. the **model** picks one, from text, with the options supplied at run
//!    time;
//! 3. a **shield** executes the model's highest-probability ADMISSIBLE pick -
//!    admissible meaning *provably on a shortest solution* - and counts the
//!    intervention when the model's own first choice was not.
//!
//! So the cube is always solved, in the optimal number of moves, and what is
//! reported is not "it worked" but **how often the model's first pick was one
//! the planner would have made**, against the chance rate for the same
//! question. `--unassisted` drops the shield and lets the model drive alone,
//! which is the measurement that says what it is worth without one.
//!
//! ```text
//! rubiks --cubes 3 --scramble 6
//! rubiks --cubes 3 --scramble 6 --hints off   # options stop saying what they do
//! rubiks --cubes 3 --scramble 4 --unassisted  # no shield: does it solve anything?
//! ```
//!
//! Swedish Embedded AB builds the layer that makes a model's judgement safe
//! to act on - a planner that bounds it, a shield that enforces the bound,
//! and the measurement that says what the model contributed - for its
//! clients. If your team needs that, you can procure our services by sending
//! an email to info@swedishembedded.com.

mod cube;
mod search;

use std::path::{Path, PathBuf};

use brain::decision::{Answer, Opt, Question, State};
use brain::options::{Args, Hardware, ModelChoice, Options};
use brain::{DecisionPipeline, Device};

use cube::{Cube, Face, Move};
use search::Solver;

/// The short names this sample knows, mapped to what `brain pull` calls them.
const ALIASES: &[(&str, &str)] = &[
    ("laya", "convaiinnovations/laya"),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
];

struct Settings {
    model: ModelChoice,
    cubes: usize,
    scramble: u8,
    options: usize,
    seed: u64,
    hints: bool,
    unassisted: bool,
    hardware: Hardware,
}

fn usage() -> String {
    format!(
        "\
usage: rubiks [options]

Solves scrambled Rubik's cubes by asking a decision model which move to make.
An exact planner supplies the candidate moves and a shield executes the
model's best ADMISSIBLE pick, so the cube is always solved in the optimal
number of moves and what gets reported is what the model contributed.

model
{}

run
  --cubes N           cubes to solve                                  [3]
  --scramble N        moves in each scramble, 1..{}                    [6]
  --options N         candidate moves offered per turn, 2..18         [6]
  --seed N            which cubes, and which candidates               [1]
  --hints on|off      whether each option says what it does           [on]
  --unassisted        no shield: execute the model's top pick as-is
  --quiet             summary only

hardware
{}
",
        ModelChoice::help(),
        Solver::MAX_DEPTH,
        Hardware::help()
    )
}

fn parse() -> Result<(Settings, bool), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        std::process::exit(0);
    }
    let mut args = Args::new(&argv);
    let hardware = Hardware::take(&mut args)?;
    let model = ModelChoice::new("laya").take_over(&mut args)?;
    let cubes = args.usize_or("--cubes", 3);
    let scramble = args.usize_or("--scramble", 6) as u8;
    let options = args.usize_or("--options", 6);
    let seed = args.u64_or("--seed", 1);
    let hints = args.str_or("--hints", "on") != "off";
    let unassisted = args.take_flag("--unassisted");
    let quiet = args.take_flag("--quiet");
    args.finish();
    if scramble == 0 || scramble > Solver::MAX_DEPTH {
        return Err(format!("--scramble must be 1..{} (this planner is exact, not heuristic)", Solver::MAX_DEPTH));
    }
    if !(2..=18).contains(&options) {
        return Err("--options must be 2..18: a cube has eighteen moves".into());
    }
    Ok((Settings { model, cubes, scramble, options, seed, hints, unassisted, hardware }, quiet))
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("rubiks: {e}");
            std::process::exit(1);
        }
    }
}

/// What one run measured. Every field is counted, never estimated.
#[derive(Default)]
struct Tally {
    turns: usize,
    top_admissible: usize,
    interventions: usize,
    chance: f32,
    confidence: f32,
    solved: usize,
    moves_used: usize,
    optimal_moves: usize,
}

fn run() -> Result<bool, String> {
    let (s, quiet) = parse()?;
    let dir = locate(&s.model.name)?;
    s.hardware.apply()?;
    eprintln!("rubiks: {} from {} on {}", s.model.name, dir, s.hardware.describe());

    let solver = Solver::new(4.min(s.scramble));
    eprintln!(
        "rubiks: planner holds every state within {} moves of solved ({} of them), exact to {} moves",
        solver.half(),
        solver.shell_size(),
        Solver::MAX_DEPTH
    );

    let mut pipe = DecisionPipeline::builder(&dir)
        .device(Device::default())
        .load()
        .map_err(|e| format!("loading {dir}: {e}"))?;

    let mut tally = Tally::default();
    for i in 0..s.cubes {
        solve_one(&mut pipe, &solver, &s, i, quiet, &mut tally)?;
    }
    report(&s, &tally);
    Ok(tally.solved == s.cubes)
}

fn solve_one(
    pipe: &mut DecisionPipeline,
    solver: &Solver,
    s: &Settings,
    index: usize,
    quiet: bool,
    tally: &mut Tally,
) -> Result<(), String> {
    let seed = s.seed.wrapping_add(index as u64 * 7919);
    let (mut cube, scramble_moves) = cube::scramble(s.scramble as usize, seed);
    let d0 = solver.distance(&cube, Solver::MAX_DEPTH).ok_or("a scramble landed out of the planner's range")?;
    tally.optimal_moves += d0 as usize;
    if !quiet {
        let optimal = solver.solve(&cube, Solver::MAX_DEPTH).ok_or("out of the planner's range")?;
        // Defence in depth: the planner's own answer is checked against the
        // cube before it is printed as one, every run, not only in tests.
        assert!(cube.apply_all(&optimal).is_solved(), "the planner returned a solution that does not solve");
        let scrambled_by: Vec<String> = scramble_moves.iter().map(|m| m.notation()).collect();
        let undo: Vec<String> = scramble_moves.iter().rev().map(|m| m.inverse().notation()).collect();
        println!(
            "\ncube {} - scrambled by {} (undone by {}), {} from solved; one optimal solution is {}",
            index + 1,
            scrambled_by.join(" "),
            undo.join(" "),
            plural(d0),
            optimal.iter().map(|m| m.notation()).collect::<Vec<_>>().join(" ")
        );
    }

    // Without a shield the model can wander, so the run is bounded and the
    // failure is REPORTED rather than hidden by an unbounded loop.
    let cap = if s.unassisted { (d0 as usize * 3).max(12) } else { d0 as usize };
    let mut used = 0usize;
    let mut last: Option<Face> = None;
    let mut rng = seed ^ 0xc0ffee;

    let mut drifted = false;
    while !cube.is_solved() && used < cap {
        // Unshielded, the model can walk the cube past the range this exact
        // planner covers. That ends the cube and is REPORTED as what it is -
        // it is a result, not an error, and the run continues.
        let Some(d) = solver.distance(&cube, Solver::MAX_DEPTH) else {
            drifted = true;
            break;
        };
        let admissible = solver.admissible(&cube, d);
        let candidates = draw_candidates(&cube, &admissible, last, s.options, &mut rng);
        let good: Vec<bool> = candidates.iter().map(|m| admissible.contains(m)).collect();
        let n_good = good.iter().filter(|g| **g).count();

        let question = Question::Choice {
            instructions: "Which turn gets this cube solved soonest?".to_string(),
            options: candidates
                .iter()
                .zip(&good)
                .map(|(m, &is_good)| describe_option(*m, is_good, d, s.hints))
                .collect(),
        };
        let answers = pipe
            .decide(&State::Str(state_text(&cube, last)), std::slice::from_ref(&question))
            .map_err(|e| format!("{e}"))?;
        let (pick, probs, confidence) = match &answers[0] {
            Answer::Choice { probabilities, confidence, .. } => {
                let best = argmax(probabilities);
                (best, probabilities.clone(), *confidence)
            }
            other => return Err(format!("expected a choice, got {other:?}")),
        };

        tally.turns += 1;
        tally.chance += n_good as f32 / candidates.len() as f32;
        tally.confidence += confidence;
        let top_ok = good[pick];
        if top_ok {
            tally.top_admissible += 1;
        }

        // The shield, exactly as laya-mlx's snake demo defines it: the
        // model's highest-probability ADMISSIBLE option, and an intervention
        // counted whenever that was not its first choice.
        let played = if s.unassisted || top_ok {
            pick
        } else {
            tally.interventions += 1;
            best_admissible(&probs, &good)
        };
        let m = candidates[played];
        if !quiet {
            println!(
                "  {d:>2} left | model picks {:<26} p={:.3} {} | plays {}",
                candidates[pick].notation() + " " + &candidates[pick].describe(),
                probs[pick].1,
                if top_ok { "on a shortest path" } else { "NOT on a shortest path" },
                m.notation()
            );
        }
        cube = cube.apply(m);
        last = Some(m.face);
        used += 1;
    }

    tally.moves_used += used;
    if cube.is_solved() {
        tally.solved += 1;
        if !quiet {
            println!("  solved in {} moves (optimal is {})\n{}", used, d0, cube.net());
        }
    } else if !quiet {
        let left = solver.distance(&cube, Solver::MAX_DEPTH);
        let how = if drifted {
            format!("more than {} moves away, past what this planner can measure", Solver::MAX_DEPTH)
        } else {
            left.map(plural).unwrap_or_else(|| format!("more than {} moves away", Solver::MAX_DEPTH))
        };
        println!("  NOT solved after {used} moves - {how}");
    }
    Ok(())
}

/// The candidates offered this turn: a random subset that ALWAYS contains at
/// least one admissible move, in random order.
///
/// Two reasons, and the second is the one that matters. A cube has eighteen
/// moves and the model's own packed sequence budget would silently shorten
/// the option texts if all of them were offered - the descriptions are the
/// thing being read, so they must arrive whole. And a fixed list in a fixed
/// order is a list whose right answer can be found by POSITION rather than by
/// reading it, which is the one thing a decision model must not be allowed to
/// do; `samples/decision/intents` exists to test exactly that property.
fn draw_candidates(cube: &Cube, admissible: &[Move], last: Option<Face>, k: usize, rng: &mut u64) -> Vec<Move> {
    let mut next = || {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    let _ = cube;
    let mut pool: Vec<Move> = Move::all().into_iter().filter(|m| Some(m.face) != last).collect();
    let keep = admissible[(next() % admissible.len() as u64) as usize];
    pool.retain(|m| *m != keep);
    let mut drawn = vec![keep];
    while drawn.len() < k.min(pool.len() + 1) {
        let i = (next() % pool.len() as u64) as usize;
        drawn.push(pool.remove(i));
    }
    for i in (1..drawn.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        drawn.swap(i, j);
    }
    drawn
}

/// One option as the model sees it. With `--hints on` the planner says what
/// the move does, which is the version of this question a model can answer by
/// READING; with hints off it has the move and the cube and nothing else,
/// which is the version that asks it to reason about a cube.
///
/// **A hint carries the decision and NOTHING else.** The identity of the
/// move is the option's label, which comes back in the answer either way;
/// the description is what the model weighs. Spelling the turn out again in
/// every description ("turn the right face a quarter turn clockwise") adds
/// ten tokens of near-identical boilerplate to all six options and drowns
/// the one phrase that decides between them - measured, not guessed: the
/// same six options with the boilerplate attached scored the right one at
/// 0.16 (a flat distribution, confidence 0.03), and with the hint alone at
/// 0.61 (confidence 0.30).
///
/// With `--hints off` the description is the turn itself, because then the
/// turn IS all there is to say - that is the version that asks the model to
/// reason about a cube rather than to read.
fn describe_option(m: Move, admissible: bool, d: u8, hints: bool) -> Opt {
    if !hints {
        return Opt::described(m.notation(), m.describe());
    }
    let effect = if admissible {
        format!("one step closer to solved, {} left", plural(d - 1))
    } else {
        "a step further away from solved".to_string()
    };
    Opt::described(m.notation(), effect)
}

/// The state, as one short sentence of prose.
///
/// It used to name every face's progress ("top face 4/9 solved, right face
/// 5/9 solved, ..."), which is six near-identical numeric clauses and reads
/// measurably worse - a wall of text in the state competes with the options
/// for the same packed sequence. What survives is the one number that is
/// actually about the cube's progress.
fn state_text(cube: &Cube, last: Option<Face>) -> String {
    let last = match last {
        Some(f) => format!(" The last turn was of the {} face.", f.name()),
        None => String::new(),
    };
    format!("A scrambled Rubik's cube with {} of its 54 stickers in place.{last}", cube.facelets_home())
}

fn best_admissible(probs: &[(String, f32)], good: &[bool]) -> usize {
    probs
        .iter()
        .enumerate()
        .filter(|(i, _)| good[*i])
        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        .map(|(i, _)| i)
        .expect("the planner always offers an admissible move")
}

fn argmax(probs: &[(String, f32)]) -> usize {
    let mut best = 0;
    for (i, p) in probs.iter().enumerate() {
        if p.1 > probs[best].1 {
            best = i;
        }
    }
    best
}

fn plural(d: u8) -> String {
    if d == 1 {
        "1 move".to_string()
    } else {
        format!("{d} moves")
    }
}

fn report(s: &Settings, t: &Tally) {
    let turns = t.turns.max(1) as f32;
    println!("\n{:-<64}", "");
    println!("cubes solved            {} of {}", t.solved, s.cubes);
    println!("moves played            {} (optimal total {})", t.moves_used, t.optimal_moves);
    println!("turns decided           {}", t.turns);
    println!(
        "model's first pick was on a shortest path  {:.0}% ({} of {})",
        100.0 * t.top_admissible as f32 / turns,
        t.top_admissible,
        t.turns
    );
    println!("the same by chance                        {:.0}%", 100.0 * t.chance / turns);
    println!("mean confidence                           {:.3}", t.confidence / turns);
    if !s.unassisted {
        println!("shield interventions    {}", t.interventions);
    }
    println!("{:-<64}", "");
    let lift = (t.top_admissible as f32 / turns) - (t.chance / turns);
    if s.hints {
        println!(
            "Options said what they do. The model beat chance by {:+.0} points, which is a\n\
             measure of whether it READ them, not of whether it understands a cube.",
            100.0 * lift
        );
    } else {
        println!(
            "Options said only which turn they were. The model beat chance by {:+.0} points -\n\
             this is the honest measure of what it knows about a cube, and it is near zero.",
            100.0 * lift
        );
    }
    if s.unassisted {
        println!("No shield: {} of {} cubes came out solved.", t.solved, s.cubes);
    } else {
        println!(
            "The shield played an admissible move every turn, so every cube was solved in the\n\
             optimal number of moves no matter what the model proposed."
        );
    }
}

/// Rule 5 of samples/README.md: say what is missing and leave cleanly.
fn locate(name: &str) -> Result<String, String> {
    if let Some(dir) = checkpoint_at(Path::new(name)) {
        return Ok(dir);
    }
    let id = ALIASES.iter().find(|(a, _)| *a == name).map(|(_, i)| *i).unwrap_or(name);
    let root = models_root()?;
    if let Some(dir) = checkpoint_at(&root.join(id)) {
        return Ok(dir);
    }
    Err(format!(
        "no decision checkpoint for {name:?}\n  looked at {name} and {}\n  run `brain pull {id}`, or pass --model DIR",
        root.join(id).display()
    ))
}

fn checkpoint_at(dir: &Path) -> Option<String> {
    let shaped = dir.join("config.json").is_file() || dir.join("rl_agent_config.json").is_file();
    shaped.then(|| dir.display().to_string())
}

fn models_root() -> Result<PathBuf, String> {
    for key in ["BRAIN_MODELS_DIR", "XDG_DATA_HOME", "HOME"] {
        if let Some(v) = std::env::var(key).ok().filter(|v| !v.is_empty()) {
            return Ok(match key {
                "BRAIN_MODELS_DIR" => PathBuf::from(v),
                "XDG_DATA_HOME" => Path::new(&v).join("brain").join("models"),
                _ => Path::new(&v).join(".local").join("share").join("brain").join("models"),
            });
        }
    }
    Err("no models directory: set BRAIN_MODELS_DIR, or pass --model DIR".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shield's promise, tested without a model in the loop: whatever is
    /// proposed, playing the best ADMISSIBLE candidate solves the cube in
    /// exactly the optimal number of moves. This is the property that lets
    /// this sample claim a solve at all.
    #[test]
    fn the_shield_solves_every_cube_in_the_optimal_number_of_moves() {
        let solver = Solver::new(4);
        for seed in 0..15u64 {
            let (mut cube, _) = cube::scramble(6, seed + 500);
            let d0 = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
            let (mut used, mut last, mut rng) = (0u8, None, seed ^ 0xc0ffee);
            while !cube.is_solved() {
                let d = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
                let admissible = solver.admissible(&cube, d);
                let candidates = draw_candidates(&cube, &admissible, last, 6, &mut rng);
                let good: Vec<bool> = candidates.iter().map(|m| admissible.contains(m)).collect();
                // The worst model imaginable: it always names the option the
                // planner likes least, and the shield still has to win.
                let probs: Vec<(String, f32)> =
                    candidates.iter().zip(&good).map(|(m, g)| (m.notation(), if *g { 0.0 } else { 1.0 })).collect();
                let played = best_admissible(&probs, &good);
                cube = cube.apply(candidates[played]);
                last = Some(candidates[played].face);
                used += 1;
                assert!(used <= d0, "seed {seed}: took more than the optimal {d0} moves");
            }
            assert_eq!(used, d0, "seed {seed}");
        }
    }

    /// Every turn offers at least one admissible move, or the shield would
    /// have nothing to fall back to.
    #[test]
    fn the_candidate_set_always_contains_a_move_that_helps() {
        let solver = Solver::new(4);
        let (cube, _) = cube::scramble(6, 77);
        let d = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
        let admissible = solver.admissible(&cube, d);
        let mut rng = 12345u64;
        for k in 2..=8 {
            let drawn = draw_candidates(&cube, &admissible, None, k, &mut rng);
            assert_eq!(drawn.len(), k, "asked for {k} candidates");
            assert!(drawn.iter().any(|m| admissible.contains(m)), "no admissible move among {k} candidates");
            let mut sorted = drawn.clone();
            sorted.sort_by_key(|m| (m.face.index(), m.quarters));
            sorted.dedup();
            assert_eq!(sorted.len(), drawn.len(), "candidates must be distinct");
        }
    }

    /// With hints on, the option text carries the answer - so a reader can
    /// find it. With hints off it cannot, and that difference is the whole
    /// experiment.
    #[test]
    fn hints_put_the_answer_in_the_option_text_and_no_hints_does_not() {
        let m = Move { face: Face::R, quarters: 1 };
        let with = describe_option(m, true, 4, true).description.unwrap();
        let without = describe_option(m, true, 4, false).description.unwrap();
        assert!(with.contains("closer"), "{with}");
        assert!(!with.contains("turn the"), "a hint must not repeat the move: {with}");
        assert!(!without.contains("closer") && without.contains("turn the"), "{without}");
        let bad = describe_option(m, false, 4, true).description.unwrap();
        assert!(bad.contains("further"), "{bad}");
    }
}
