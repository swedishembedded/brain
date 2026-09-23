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
//! rubiks --model minilm --train 8000 --batch 32 --eval 300   # fit a policy
//! rubiks --cubes 3 --scramble 4 --unassisted  # no shield: does it solve anything?
//! ```
//!
//! Swedish Embedded AB builds the layer that makes a model's judgement safe
//! to act on - a planner that bounds it, a shield that enforces the bound,
//! and the measurement that says what the model contributed - for its
//! clients. If your team needs that, you can procure our services by sending
//! an email to info@swedishembedded.com.

mod cube;
mod policy;
mod search;
mod view;
mod space;
mod learned;

use std::path::{Path, PathBuf};

use brain::decision::{Answer, Opt, Question, State};
use brain::options::{Args, Hardware, ModelChoice, Options, ViewOptions};
use brain::viewport::Viewport;
use brain::{DecisionPipeline, Device, Stages};

use cube::{Cube, Move};
use search::Solver;
use view::{Panel, Row, Scene};

/// The short names this sample knows, mapped to what `brain pull` calls them.
const ALIASES: &[(&str, &str)] = &[
    ("laya", "convaiinnovations/laya"),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
];

struct Settings {
    model: ModelChoice,
    cubes: usize,
    scramble: u8,
    seed: u64,
    encoding: policy::Encoding,
    head: Option<String>,
    train: usize,
    batch: usize,
    examples: usize,
    save: String,
    eval: usize,
    unassisted: bool,
    hardware: Hardware,
    view: ViewOptions,
    record: Option<String>,
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
  --seed N            which cubes                                     [1]
  --encode WHICH      how the cube is written down: grid | compact | rows [grid]
  --head FILE         a trained decision head to answer with
  --unassisted        no shield: play the model's own pick, every turn
  --quiet             summary only

train and measure
  --train N           train a head for N optimizer steps, then save it
  --batch N           labelled decisions per optimizer step           [{}]
  --examples N        labelled decisions to train on                  [4000]
  --save DIR          where the trained model goes             [out/rubiks-model]
  --eval N            score N held-out decisions and report

view
{}
  --record FILE.mp4   record what the window shows (needs ffmpeg)

hardware
{}
",
        ModelChoice::help(),
        Solver::MAX_DEPTH,
        brain::decision::DEFAULT_BATCH,
        ViewOptions::help(),
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
    let seed = args.u64_or("--seed", 1);
    let encoding = policy::Encoding::parse(&args.str_or("--encode", "grid"))?;
    let head = args.take_str("--head");
    let train = args.usize_or("--train", 0);
    let batch = args.usize_or("--batch", brain::decision::DEFAULT_BATCH);
    let examples = args.usize_or("--examples", 4000);
    let save = args.str_or("--save", "out/rubiks-model");
    let eval = args.usize_or("--eval", 0);
    let unassisted = args.take_flag("--unassisted");
    let quiet = args.take_flag("--quiet");
    let attempts = args.usize_or("--net-attempts", 1);
    let view = ViewOptions::take(&mut args)?;
    let record = args.take_str("--record");
    args.finish();
    if scramble == 0 || scramble > Solver::MAX_DEPTH {
        return Err(format!("--scramble must be 1..{} (this planner is exact, not heuristic)", Solver::MAX_DEPTH));
    }
    Ok((Settings { model, cubes, scramble, seed, encoding, head, train, batch, examples, save, eval, unassisted, hardware, view, record }, quiet))
}

fn main() {
    // The learned solver is a different pipeline end to end - its own
    // labels, its own network, and no planner in the loop - so it routes
    // before the option-scoring path rather than inside it.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--net") {
        match net_main(&argv) {
            Ok(()) => return,
            Err(e) => {
                eprintln!("rubiks: {e}");
                std::process::exit(1);
            }
        }
    }
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("rubiks: {e}");
            std::process::exit(1);
        }
    }
}

/// Train a head on decisions the planner labelled, and save it.
///
/// The labels are free: the exact planner already knows which moves get
/// closer, so a cube is an endless supply of decisions with a known right
/// answer. What is NOT free is whether a text encoder can learn to read a
/// cube - which is the question this mode exists to answer.
fn train(pipe: &mut DecisionPipeline, solver: &Solver, s: &Settings) -> Result<(), String> {
    if !pipe.supports_training() {
        return Err(format!(
            "{} cannot be trained through this SDK yet - a Laya checkpoint ships its head pretrained and \
             has no optimizer loop here. Train on a decide-shaped encoder instead: --model minilm",
            s.model.name
        ));
    }
    // Timed, because "how long is one iteration" is the number that decides
    // whether anyone can experiment with this at all.
    let t0 = std::time::Instant::now();
    let data = policy::examples(solver, s.examples, s.scramble, s.encoding, s.seed);
    let generated = t0.elapsed();
    let deepest = data.iter().map(|e| e.distance).max().unwrap_or(0);
    println!(
        "rubiks: {} labelled decisions, up to {} from solved, over {} options - generated in {:.2}s ({:.0}/s)",
        data.len(),
        plural(deepest),
        policy::options().len(),
        generated.as_secs_f32(),
        data.len() as f32 / generated.as_secs_f32().max(1e-6)
    );
    let options = policy::options();
    let examples: Vec<(&str, usize)> = data.iter().map(|e| (e.state.as_str(), e.label)).collect();
    let mut last = 0usize;
    let started = std::time::Instant::now();
    let mut mark = started;
    let every = 25usize;
    let loss = pipe
        .train_choices(&examples, &options, policy::INSTRUCTIONS, s.train, s.batch, s.seed, &mut |step, l| {
            if step / every > last {
                last = step / every;
                let per = mark.elapsed().as_secs_f32() / every as f32;
                mark = std::time::Instant::now();
                println!("  step {step:>6}  loss {l:.4}  {:.0} ms/step", per * 1000.0);
            }
        })
        .map_err(|e| format!("{e}"))?;
    let trained = started.elapsed();
    println!(
        "rubiks: final loss {loss:.4} in {:.1}s ({:.0} ms/step)",
        trained.as_secs_f32(),
        1000.0 * trained.as_secs_f32() / s.train.max(1) as f32
    );
    // The curriculum is this sample's own setting and the SDK cannot see it,
    // but it decides what the policy can do: a run only learns the distances
    // it was shown. Recording it is what lets a later reader tell two
    // otherwise-identical runs apart.
    pipe.record_fit(&serde_json::json!({
        "scramble": s.scramble,
        "encoding": s.encoding.name(),
    }))
    .map_err(|e| format!("{e}"))?;

    // A fine-tuned ENCODER is most of what this run produced - the head
    // alone would attach to the published encoder and not be this model - so
    // the whole thing is written, and it is fatal if it cannot be. A run
    // that trained for hours and then reported a number it could not hand
    // back is the failure worth being loud about.
    pipe.save_model(&s.save).map_err(|e| format!("saving to {}: {e}", s.save))?;
    println!("rubiks: model written to {} - replay it with --model {}", s.save, s.save);
    Ok(())
}

/// Score held-out decisions: how often the model's own first pick is a move
/// that gets closer, against the rate for picking at random.
///
/// Held out by SEED, so none of these cubes appeared in training.
fn evaluate(pipe: &mut DecisionPipeline, solver: &Solver, s: &Settings) -> Result<(), String> {
    let data = policy::examples(solver, s.eval, s.scramble, s.encoding, s.seed ^ 0xe7a1);
    let options = policy::options();
    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
    let (mut right, mut chance, mut confidence) = (0usize, 0.0f32, 0.0f32);
    let mut by_distance: std::collections::BTreeMap<u8, (usize, usize)> = std::collections::BTreeMap::new();

    for (i, ex) in data.iter().enumerate() {
        let answer = pipe.choose(&ex.state, policy::INSTRUCTIONS, &refs).map_err(|e| format!("{e}"))?;
        // Rebuild this state's admissible set from the planner - a policy may
        // pick a DIFFERENT good move than the label, and that is still right.
        let cube = policy::replay(solver, ex);
        let good = policy::admissible(solver, &cube, ex.distance);
        let hit = good.contains(&answer.index);
        right += hit as usize;
        chance += good.len() as f32 / options.len() as f32;
        confidence += answer.confidence;
        let slot = by_distance.entry(ex.distance).or_default();
        slot.0 += hit as usize;
        slot.1 += 1;
        if (i + 1) % 25 == 0 {
            println!("  {} of {} scored", i + 1, data.len());
        }
    }
    let n = data.len().max(1) as f32;
    println!("\n{:-<58}", "");
    println!("decisions scored          {}", data.len());
    println!("first pick gets closer    {:.1}%", 100.0 * right as f32 / n);
    println!("the same by chance        {:.1}%", 100.0 * chance / n);
    println!("mean confidence           {:.3}", confidence / n);
    println!("\nby how far from solved the cube was:");
    for (d, (hit, total)) in by_distance {
        println!("  {d} away   {:>5.1}%  ({hit} of {total})", 100.0 * hit as f32 / total.max(1) as f32);
    }
    println!("{:-<58}", "");
    Ok(())
}

/// The window, the frame dump and the recording - one object, so the solve
/// loop does not care which of them (if any) is switched on.
///
/// Nothing here is required: with no `--window`, `--frames` or `--record`
/// the stage is dark and the run is exactly the text one, at the same cost.
struct ViewSettings<'a> {
    view: &'a ViewOptions,
    record: &'a Option<String>,
}

struct Stage {
    viewport: Option<Viewport>,
    fps: u32,
    frames_dir: Option<String>,
    frame: u64,
    scene: Scene,
    quit: bool,
}

impl Stage {
    fn new(s: &Settings) -> Result<Stage, String> {
        Stage::open(&s.view, &s.record)
    }

    /// The stage itself needs only the view flags, so both the option-scoring
    /// run and the learned one open it the same way.
    fn open(view: &ViewOptions, record: &Option<String>) -> Result<Stage, String> {
        let s = ViewSettings { view, record };
        let wanted = s.view.window || s.view.frames.is_some() || s.record.is_some();
        if !wanted {
            return Ok(Stage { viewport: None, fps: s.view.fps, frames_dir: None, frame: 0, scene: Scene::default(), quit: false });
        }
        let mut viewport = Viewport::open("brain - rubiks", view::WIDTH, view::HEIGHT).map_err(|e| format!("{e:?}"))?;
        if let Some(why) = viewport.headless_because.clone() {
            // Said out loud: a run that quietly lost its window and wrote
            // nothing is the failure this message exists to prevent.
            eprintln!("rubiks: no window ({why}) - drawing to memory; --frames DIR or --record FILE still work");
        }
        if let Some(path) = &s.record {
            viewport.record(path, s.view.fps).map_err(|e| format!("--record: {e}"))?;
        }
        if let Some(dir) = &s.view.frames {
            std::fs::create_dir_all(dir).map_err(|e| format!("--frames {dir}: {e}"))?;
        }
        Ok(Stage {
            viewport: Some(viewport),
            fps: s.view.fps,
            frames_dir: s.view.frames.clone(),
            frame: 0,
            scene: Scene::default(),
            quit: false,
        })
    }

    fn on(&self) -> bool {
        self.viewport.is_some()
    }

    /// Draw one frame, show it, save it, and pace it.
    fn show(&mut self, cube: &Cube, panel: &Panel) {
        let Some(vp) = self.viewport.as_mut() else { return };
        view::draw(vp.canvas(), cube, &self.scene, panel);
        vp.present();
        if let Some(dir) = &self.frames_dir {
            let path = format!("{dir}/frame-{:05}.png", self.frame);
            if let Err(e) = vp.canvas().save(&path) {
                eprintln!("rubiks: {path}: {e}");
            }
        }
        self.frame += 1;
        let input = vp.input();
        if input.quit {
            self.quit = true;
        }
        std::thread::sleep(std::time::Duration::from_micros(1_000_000 / self.fps.max(1) as u64));
    }

    /// Hold the decision on screen long enough to read it.
    fn dwell(&mut self, cube: &Cube, panel: &Panel, seconds: f32) {
        if !self.on() {
            return;
        }
        for _ in 0..((seconds * self.fps as f32) as usize).max(1) {
            self.scene.yaw += 0.004;
            self.show(cube, panel);
        }
    }

    /// Turn the layer, one frame at a time, and leave the cube in the state
    /// the engine says it is in.
    fn turn(&mut self, before: &Cube, m: Move, panel: &Panel) {
        if !self.on() {
            return;
        }
        let steps = (self.fps as usize * m.quarters as usize / 3).max(6);
        for i in 1..=steps {
            // Ease in and out, so a turn reads as a hand moving it.
            let t = i as f32 / steps as f32;
            let eased = t * t * (3.0 - 2.0 * t);
            self.scene.turning = Some((m, eased));
            self.scene.yaw += 0.003;
            self.show(before, panel);
        }
        self.scene.turning = None;
    }

    fn finish(&mut self) {
        let Some(vp) = self.viewport.as_mut() else { return };
        if let Some(done) = vp.finish_recording() {
            match done {
                Ok((frames, path)) => eprintln!("rubiks: wrote {frames} frames to {}", path.display()),
                Err(e) => eprintln!("rubiks: recording: {e}"),
            }
        }
        if let Some(dir) = &self.frames_dir {
            eprintln!("rubiks: wrote {} frames to {dir}", self.frame);
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

    let mut builder = DecisionPipeline::builder(&dir).device(Device::default());
    if let Some(head) = &s.head {
        if !Path::new(head).is_file() {
            return Err(format!("no head weights at {head} - make some with --train N --save {head}"));
        }
        builder = builder.head(head);
    }
    let mut pipe = builder.load().map_err(|e| format!("loading {dir}: {e}"))?;

    if s.train > 0 {
        train(&mut pipe, &solver, &s)?;
    }
    if s.eval > 0 {
        evaluate(&mut pipe, &solver, &s)?;
    }
    if s.cubes == 0 {
        return Ok(true);
    }

    let mut stage = Stage::new(&s)?;
    let mut tally = Tally::default();
    for i in 0..s.cubes {
        solve_one(&mut pipe, &solver, &s, i, quiet, &mut tally, &mut stage)?;
        if stage.quit {
            eprintln!("rubiks: closed after {} of {} cubes", i + 1, s.cubes);
            break;
        }
    }
    stage.finish();
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
    stage: &mut Stage,
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
    let mut history: Vec<String> = Vec::new();

    let mut drifted = false;
    while !cube.is_solved() && used < cap && !stage.quit {
        // Unshielded, the model can walk the cube past the range this exact
        // planner covers. That ends the cube and is REPORTED as what it is -
        // it is a result, not an error, and the run continues.
        let Some(d) = solver.distance(&cube, Solver::MAX_DEPTH) else {
            drifted = true;
            break;
        };
        // Every one of the eighteen moves is offered, every turn: this is the
        // decision, not a shortlist somebody else made.
        let good_indices = policy::admissible(solver, &cube, d);
        let candidates = Move::all();
        let good: Vec<bool> = (0..candidates.len()).map(|i| good_indices.contains(&i)).collect();
        let n_good = good_indices.len();

        let texts = policy::options();
        let question = Question::Choice {
            instructions: policy::INSTRUCTIONS.to_string(),
            // Described identically, so nothing but the move itself tells
            // them apart. Whether one helps is the planner's business, and
            // the planner is not talking to the model.
            options: texts.iter().map(|t| Opt::new(t.as_str())).collect(),
        };
        let state = policy::state_text(&cube, s.encoding);
        let mut panel = Panel {
            model: s.model.name.clone(),
            planner: true,
            cube_index: index,
            cubes: s.cubes,
            distance: d,
            state: state.clone(),
            rows: candidates
                .iter()
                .zip(&good)
                .map(|(m, &adm)| Row {
                    name: m.notation(),
                    detail: m.short(),
                    probability: 0.0,
                    admissible: adm,
                })
                .collect(),
            thinking: true,
            history: history.clone(),
            turns: tally.turns,
            top_admissible: tally.top_admissible,
            chance: if tally.turns == 0 { 0.0 } else { tally.chance / tally.turns as f32 },
            interventions: tally.interventions,
            solved: tally.solved,
            unassisted: s.unassisted,
            ..Panel::default()
        };
        // The question goes up BEFORE the answer comes back, so what the
        // model was asked is on screen while it is being asked.
        stage.dwell(&cube, &panel, 0.5);

        let answers = pipe
            .decide(&State::Str(state), std::slice::from_ref(&question))
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
        for (row, (_, p)) in panel.rows.iter_mut().zip(&probs) {
            row.probability = *p;
        }
        panel.thinking = false;
        panel.picked = Some(pick);
        panel.played = Some(played);
        panel.shielded = played != pick;
        panel.turns = tally.turns;
        panel.top_admissible = tally.top_admissible;
        panel.chance = tally.chance / tally.turns.max(1) as f32;
        panel.interventions = tally.interventions;
        stage.dwell(&cube, &panel, 1.2);
        stage.turn(&cube, m, &panel);
        history.push(m.notation());
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
        used += 1;
    }

    tally.moves_used += used;
    if cube.is_solved() {
        tally.solved += 1;
        if stage.on() {
            let panel = Panel {
                model: s.model.name.clone(),
                planner: true,
                cube_index: index,
                cubes: s.cubes,
                distance: 0,
                state: format!("Solved in {used} moves - the optimal is {d0}."),
                history: history.clone(),
                turns: tally.turns,
                top_admissible: tally.top_admissible,
                chance: tally.chance / tally.turns.max(1) as f32,
                interventions: tally.interventions,
                solved: tally.solved,
                unassisted: s.unassisted,
                ..Panel::default()
            };
            stage.dwell(&cube, &panel, 1.5);
        }
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

/// The model's best pick among the moves the planner vouches for.
///
/// The shield's whole content: it never proposes a move, it only declines to
/// play one that is not on a shortest solution, and among those that are it
/// plays whichever the model itself ranked highest. So the cube is solved
/// optimally whatever the model says, and what the model contributed is
/// still measurable - it is how often this had nothing to override.
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
    println!(
        "The model saw the whole cube and eighteen moves described alike, and nothing\n\
         about which one helps. It beat chance by {:+.0} points.",
        100.0 * lift
    );
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
    use crate::search::Solver;

    #[test]
    fn a_short_name_maps_to_what_brain_pull_calls_it() {
        assert_eq!(ALIASES.iter().find(|(a, _)| *a == "laya").unwrap().1, "convaiinnovations/laya");
    }

    /// The missing-checkpoint message has to carry the fix, since the caller
    /// is looking at a terminal rather than at this code.
    #[test]
    fn a_missing_checkpoint_names_the_command_that_fetches_it() {
        let err = locate("definitely-not-a-model").unwrap_err();
        assert!(err.contains("brain pull definitely-not-a-model"), "{err}");
        assert!(err.contains("--model DIR"), "{err}");
    }

    /// THE property this sample rests on, with no model in the loop: play the
    /// best ADMISSIBLE option every turn and every cube is solved in exactly
    /// the optimal number of moves - even when the thing choosing is an
    /// adversary that always names the option the planner likes least.
    #[test]
    fn the_shield_solves_every_cube_in_the_optimal_number_of_moves() {
        let solver = Solver::new(4);
        for seed in 0..15u64 {
            let (mut cube, _) = cube::scramble(6, seed + 500);
            let d0 = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
            let mut used = 0u8;
            while !cube.is_solved() {
                let d = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
                let good = policy::admissible(&solver, &cube, d);
                assert!(!good.is_empty(), "an unsolved cube always has a move that helps");
                // The worst model imaginable: every admissible option gets
                // zero, every other one gets one.
                let probs: Vec<(String, f32)> = policy::options()
                    .iter()
                    .enumerate()
                    .map(|(i, o)| (o.clone(), if good.contains(&i) { 0.0 } else { 1.0 }))
                    .collect();
                let flags: Vec<bool> = (0..probs.len()).map(|i| good.contains(&i)).collect();
                let played = best_admissible(&probs, &flags);
                cube = cube.apply(policy::move_of(played));
                used += 1;
                assert!(used <= d0, "seed {seed}: took more than the optimal {d0} moves");
            }
            assert_eq!(used, d0, "seed {seed}");
        }
    }

    /// Every turn offers all eighteen moves and at least one of them helps,
    /// or the shield would have nothing to fall back to.
    #[test]
    fn every_position_has_a_move_that_helps() {
        let solver = Solver::new(4);
        for seed in 0..10u64 {
            let (cube, _) = cube::scramble(7, seed + 200);
            let d = solver.distance(&cube, Solver::MAX_DEPTH).unwrap();
            let good = policy::admissible(&solver, &cube, d);
            assert!(!good.is_empty(), "seed {seed}");
            assert!(good.len() < 18, "if every move helped, choosing would not be a decision");
            for &i in &good {
                let after = cube.apply(policy::move_of(i));
                assert_eq!(solver.distance(&after, Solver::MAX_DEPTH), Some(d - 1));
            }
        }
    }
}

/// Train a policy on retraced walks and let it drive, with no planner and no
/// search anywhere in the loop.
fn net_main(argv: &[String]) -> Result<(), String> {
    let mut args = Args::new(argv);
    let _ = args.take_flag("--net");
    let hardware = Hardware::take(&mut args)?;
    hardware.apply()?;

    let a = learned::TrainArgs {
        steps: args.usize_or("--net-steps", 20000),
        rows: args.usize_or("--net-batch", 1024) as u32,
        depth: args.usize_or("--net-depth", 26),
        d_model: args.usize_or("--net-width", 1024) as u32,
        d_ff: args.usize_or("--net-ff", 2048) as u32,
        blocks: args.usize_or("--net-blocks", 4) as u32,
        lr: args.usize_or("--net-lr-micro", 600) as f32 / 1e6,
        seed: args.u64_or("--seed", 1),
        eval_every: args.usize_or("--net-eval-every", 2000),
        eval_cubes: args.usize_or("--net-eval-cubes", 200),
        eval_scramble: args.usize_or("--net-eval-scramble", 40),
        checkpoint: None,
        refresh: args.usize_or("--net-refresh", 200),
        value_weight: args.usize_or("--net-value-micro", 50_000) as f32 / 1e6,
    };
    let cubes = args.usize_or("--cubes", 500);
    let scramble = args.usize_or("--scramble", 40);
    let save = args.take_str("--net-save");
    let load = args.take_str("--net-load");
    let attempts = args.usize_or("--net-attempts", 1);
    let view = ViewOptions::take(&mut args)?;
    let record = args.take_str("--record");

    let mut a = a;
    a.checkpoint = save.clone();
    let net = match &load {
        Some(p) => {
            println!("cubenet: loading {p}");
            brain::solve::Net::load(p, a.rows).map_err(|e| e.to_string())?
        }
        None => {
            let n = learned::train(&a);
            if let Some(p) = &save {
                let fit = serde_json::json!({
                    "task": "cube-policy",
                    "steps": a.steps, "batch": a.rows, "walk_depth": a.depth,
                    "d_model": a.d_model, "d_ff": a.d_ff, "blocks": a.blocks,
                    "lr": a.lr, "seed": a.seed,
                });
                n.save(p, &fit).map_err(|e| e.to_string())?;
                println!("cubenet: written to {p}");
            }
            n
        }
    };

    if view.window || view.frames.is_some() || record.is_some() {
        record_learned(&net, cubes.min(12), scramble, &view, &record)?;
    }

    println!("\n----------------------------------------------------------------");
    println!("            no search at inference: one forward pass per move");
    println!("----------------------------------------------------------------");
    println!(
        "{:>9}  {:>14}  {:>14}  {:>10}",
        "scramble",
        "policy greedy",
        "value lookahead".to_string(),
        "ms/cube"
    );
    for d in [4usize, 8, 12, 16, 20, 26, scramble] {
        let g = learned::measure_with(
            &net, cubes, d, 0x5014ED,
            brain::solve::Rollout {
                max_steps: 64,
                forbid_redundant: true,
                temperature: 0.0,
                seed: 7,
                attempts: 1,
            },
        );
        let s = learned::measure_how(
            &net,
            cubes,
            d,
            0x5014ED,
            brain::solve::Rollout {
                max_steps: 64,
                forbid_redundant: true,
                temperature: 0.0,
                seed: 7,
                attempts: 1,
            },
            true,
        );
        println!(
            "{:>9}  {:>6} ({:>3.0}%)  {:>6} ({:>3.0}%)  {:>10.2}",
            d,
            g.solved, 100.0 * g.rate(),
            s.solved, 100.0 * s.rate(),
            1000.0 * g.elapsed.as_secs_f32() / g.total.max(1) as f32
        );
    }
    println!("----------------------------------------------------------------");
    println!("(each row is {cubes} random cubes; a 40-move scramble is a uniformly");
    println!(" random cube, which is at most 20 moves from solved)");
    Ok(())
}

/// Record the learned policy solving cubes, with no planner anywhere.
fn record_learned(
    net: &brain::solve::Net,
    cubes: usize,
    scramble: usize,
    view: &ViewOptions,
    record: &Option<String>,
) -> Result<(), String> {
    use brain::solve::StateSpace;
    let space = space::CubeSpace::new();
    let mut stage = Stage::open(view, record)?;
    if !stage.on() {
        return Ok(());
    }
    let mut solved = 0usize;
    let mut turns = 0usize;

    for index in 0..cubes {
        let (mut cube, scrambled_by) = cube::scramble(scramble, 0xC0DE ^ index as u64);
        let mut history: Vec<String> = Vec::new();
        let mut last: Option<usize> = None;
        let mut used = 0usize;

        for _ in 0..64 {
            if cube.is_solved() {
                break;
            }
            let t = learned::ask(net, &space, &cube, last);
            let m = space.move_at(t.picked);

            // Every move, with the probability the policy gave it - the same
            // eighteen options every turn, described alike.
            let mut rows: Vec<view::Row> = (0..space.moves())
                .map(|i| view::Row {
                    name: space.move_at(i).notation(),
                    detail: space.move_at(i).short(),
                    probability: t.probabilities[i],
                    admissible: false,
                })
                .collect();
            rows.sort_by(|a, b| b.probability.total_cmp(&a.probability));
            let picked_row = rows.iter().position(|r| r.name == m.notation());

            let panel = Panel {
                model: "cubenet".into(),
                planner: false,
                cube_index: index,
                cubes,
                home: cube.facelets_home(),
                state: format!(
                    "Scrambled by {} random moves. Move {} - no search.",
                    scrambled_by.len(),
                    used + 1
                ),
                rows,
                picked: picked_row,
                played: picked_row,
                history: history.clone(),
                turns,
                chance: t.probabilities[t.picked],
                solved,
                unassisted: true,
                ..Default::default()
            };
            stage.dwell(&cube, &panel, 0.30);
            stage.turn(&cube, m, &panel);

            cube = cube.apply(m);
            history.push(m.notation());
            last = Some(t.picked);
            used += 1;
            turns += 1;
            if stage.quit {
                return Ok(());
            }
        }

        if cube.is_solved() {
            solved += 1;
        }
        let panel = Panel {
            model: "cubenet".into(),
            planner: false,
            cube_index: index,
            cubes,
            home: cube.facelets_home(),
            state: if cube.is_solved() {
                format!("Solved in {used} moves, from a {scramble}-move scramble.")
            } else {
                format!("Not solved after {used} moves.")
            },
            history: history.clone(),
            turns,
            solved,
            unassisted: true,
            ..Default::default()
        };
        stage.dwell(&cube, &panel, 1.6);
    }
    Ok(())
}
