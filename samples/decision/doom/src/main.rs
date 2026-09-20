// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Learn to play DOOM from the game's own structured state.
//!
//! The 1993 engine, modified to host an HTTP API inside its game loop, runs as
//! a subprocess in lockstep: it advances only when the agent says so. Every
//! decision reads a JSON observation - health, what is in sight and where,
//! where there is room to move, where the exit is - and chooses from a list of
//! options that is rebuilt from that state at every step. No pixels reach the
//! model.
//!
//! ```text
//! doom train --mix                 # warm-start on the script, then PPO
//! doom eval  --head out/doom.safetensors
//! doom play  --head out/doom.safetensors --window
//! doom probe --frames out/frames   # the scripted player, with artifacts
//! ```
//!
//! **Why this shape and not pixels.** A policy over frames has to learn to see
//! before it can learn to play, and then it can only choose from a fixed set of
//! buttons. Here the options arrive as text and change every step - "attack the
//! imp 12 degrees to your left" exists only while that imp does - so the model
//! reads what an option MEANS instead of looking it up by index, and the same
//! weights handle an option nobody trained on.
//!
//! **The orders are part of the input.** An episode runs under a mission, and
//! the mission text is prepended to every option. Train over a mix and one
//! policy changes strategy on being told to.
//!
//! Swedish Embedded AB builds realtime decision systems that run on the
//! customer's own hardware - reading a machine's real state, choosing among
//! actions that machine defines at run time, in milliseconds. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod action;
mod doom;
mod env;
mod frame;
mod memory;
mod obs;
mod report;
mod view;

use brain::options::{Args as Args_, ControlOptions, Hardware, Options, ViewOptions};
use brain::{Candidates, ControlPipeline};
use doom::{Config, Doom, Paths};
use env::{DoomEnv, Mission, Payment};

/// This sample's own options, on top of the shared groups.
///
/// Only the flags that are genuinely about DOOM live here. The hardware
/// selection, the training knobs and the window flags come from
/// `brain::options`, which is where the `brain` binary and every other sample
/// get them - so `--device`, `--episodes` and `--window` mean exactly the same
/// thing here as they do there, and adding one to the shared set adds it to
/// all of them at once.
pub struct Args {
    pub command: String,
    pub doom_bin: Option<String>,
    pub wad: Option<String>,
    pub cfg: Config,
    /// The levels episodes are drawn from.
    pub maps: Vec<u32>,
    /// Generated scenarios episodes are drawn from, instead of levels.
    pub scenarios: Vec<String>,
    /// Overrides the mission's route-closing weight, for ablating it.
    pub approach: Option<f32>,
    /// What a decision is paid for. See [`Payment`].
    pub payment: Payment,
    /// Draw the alternatives at random rather than by the policy's ranking.
    pub wide: bool,
    pub mission: Mission,
    pub mix: bool,
    pub arena: usize,
    pub curriculum: bool,
    pub start_distance: i32,
    pub eval_episodes: usize,
    pub play: usize,
    pub transcript: Option<String>,
    pub engine_log: Option<String>,
    pub record: Option<String>,
    /// Advance the game one tic at a time, as recording has to. On its own
    /// this captures nothing: it is here so that "stepping tic by tic" can be
    /// measured apart from "keeping every frame", which is the only way to
    /// tell which of the two changes a run.
    pub tic_steps: bool,
    pub hardware: Hardware,
    pub train: ControlOptions,
    pub view: ViewOptions,
}

impl Args {
    pub fn encoder(&self) -> &str {
        &self.train.encoder
    }
    pub fn head(&self) -> Option<&String> {
        self.train.head.as_ref()
    }
    /// Whether a run wants one frame per tic rather than per decision.
    pub fn smooth_video(&self) -> bool {
        self.record.is_some() || self.tic_steps
    }
    pub fn max_steps(&self) -> usize {
        self.train.max_steps
    }
    pub fn seed(&self) -> u64 {
        self.train.seed
    }
    pub fn device(&self) -> brain::Device {
        self.hardware.device.clone()
    }
}

fn usage() -> String {
    format!(
        "\
usage: doom <train|eval|fit|whatif|play|probe|bench> [options]

  train    warm-start on the scripted player, then improve it by PPO
  eval     score a policy and the scripted player on the SAME episodes
  whatif   go back to decisions the policy made, take a DIFFERENT action
           there, let the teacher finish, and score the whole trajectory.
           Imitation can only reach the teacher; this is the one measurement
           whose answer can be better than it
  fit      fit the head to the scripted player and ask how often it agrees,
           on the episodes it was fitted to and on episodes it has not seen.
           No reward, no critic - it asks only whether the decision is
           EXPRESSIBLE from what the agent reads, which every other question
           about the policy is downstream of
  play     run episodes and show every decision as it is made
  probe    one scripted episode, for artifacts and for checking the plumbing
  bench    time one decision against state length and option count

the game (no path is ever baked in, and nothing is read from the environment)
  --doom-bin PATH     the restful-doom binary   [found on $PATH]
  --wad PATH          an IWAD

what to play
  --episode N --map N --skill 0..4      [1 1 2]
  --maps 1,2,3        draw each episode's level from this list instead of one.
                      Training on several and scoring on a level that was not
                      among them is the only measurement that separates having
                      learned to play from having learned a level
  --scenario NAME[,NAME...]
                      play levels the engine BUILDS instead of the game's own,
                      one drawn per episode, each a fresh map from that
                      episode's seed. There is no map to memorise, so what a
                      policy takes away from them is the skill and not the
                      level. The game's own levels are then free to be what a
                      run is SCORED on:
                        basic                     one monster: see, face, shoot
                        deadly-corridor           advance under fire
                        defend-the-center         a ring closing in
                        defend-the-line           the same, from a wall
                        health-gathering          a burning floor and medkits
                        health-gathering-supreme  the same, in a maze
                        my-way-home               find the armour in a new maze
                        predict-position          a walking target, a slow rocket
                        take-cover                dodge, and keep dodging
  --reward gauge|shaped
                      what a decision is paid for.               [shaped]
                      `gauge` pays it what it moved the SCORE the run is
                      finally kept on, so an episode's undiscounted return IS
                      that score - training and selection then optimise one
                      number instead of two. `shaped` is the weighted sum of
                      kills, items, damage, floor newly walked and route
                      closed: measured, about 92% of it is the exploration
                      bonus, which the score does not read at all
  --wide              draw the alternatives at RANDOM rather than from what the
                      policy ranks highest. The control: a policy fitted to the
                      teacher ranks the teacher's near-duplicates highest, so
                      the default asks about the actions least likely to lead
                      anywhere different
  --approach F        what a 32-unit cell of ROUTE closed on the goal pays.
                      Defaults to the mission's own. `--approach 0` turns it
                      off, which is the ablation: without it the only dense
                      reward is the exploration bonus, and the exit - 89% of a
                      finishing episode's return, paid on one step - is
                      invisible to an advantage estimator whose half-life is
                      eleven decisions
  --mission clear|speedrun|survive      [clear]
  --full-map          give the route the WHOLE level instead of only the part
                      the player has seen. A control, not a way to play: it
                      hands the agent a solved map of rooms nobody has been in
  --mix               sample a mission per episode, so the policy must read it
  --curriculum        reverse curriculum: start each episode AT the exit and
                      walk it further back as the policy keeps finishing. The
                      exit reward has never once fired from the level's own
                      start, and a reward that never fires trains nothing.
  --start-distance N  with --curriculum, begin episodes N map units of WALKING
                      from the exit. Training grows this on its own; a policy
                      being evaluated or recorded has to be told, since the
                      reach attained is not saved with the weights. E1M1's own
                      spawn is 5344 units out.
  --arena N           start each episode with N monsters around the player, on
                      open floor and in sight. 0 plays the level as it ships,
                      where an episode is spent leaving the spawn area and
                      neither player reaches enough combat to be scored.
  --eval-episodes N   episodes to score over   [24]
  --play N            episodes for `play`      [3]
  --transcript FILE   write every request and reply as JSON lines
  --engine-log FILE   keep the engine's own output, which is where its route
                      builder explains what it could and could not reach
  --tic-steps         advance the game one tic at a time, as recording does,
                      without capturing anything - for telling apart whether
                      stepping or capturing is what changes a recorded run
  --record FILE.mp4   encode every decision straight into an MP4 as it is
                      drawn - streamed to ffmpeg, no intermediate images

hardware
{}

training
{}

watching
{}
",
        Hardware::help(),
        ControlOptions::help(),
        ViewOptions::help()
    )
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        println!("{}", usage());
        std::process::exit(0);
    }
    let command = argv[0].clone();
    if !["train", "eval", "fit", "whatif", "play", "probe", "bench"].contains(&command.as_str()) {
        return Err(format!("unknown command {command:?}\n\n{}", usage()));
    }

    let mut args = Args_::new(&argv[1..]);
    let hardware = Hardware::take(&mut args)?;
    let view = ViewOptions::take(&mut args)?;

    // This sample's own defaults, then the shared flags on top. A DOOM episode
    // is a couple of hundred decisions, not the forty a toy environment needs,
    // and the warm start is shorter because each demonstration costs a game
    // step rather than a table lookup.
    let mut train = ControlOptions::new("", "out/doom-policy.safetensors");
    train.max_steps = 140;
    train.iterations = 10;
    train.episodes = 12;
    train.warmup_episodes = 12;
    train.warmup_epochs = 6;
    let train = train.take_over(&mut args)?;
    if train.encoder.is_empty() {
        return Err(
            "--encoder DIR is required: it is the pretrained sentence encoder the \
                    policy reads with (`brain pull sentence-transformers/all-MiniLM-L6-v2` \
                    fetches one)"
                .into(),
        );
    }

    let mission = match args.take_str("--mission") {
        Some(m) => Mission::parse(&m).ok_or_else(|| format!("unknown mission {m:?}"))?,
        None => Mission::Clear,
    };
    let cfg = Config {
        episode: args.u32_or("--episode", 1),
        map: args.u32_or("--map", 1),
        skill: args.u32_or("--skill", 2),
        engine_window: false,
        scenario: None,
        full_map: args.take_flag("--full-map"),
    };
    if cfg.skill > 4 {
        return Err("--skill must be 0..4".into());
    }
    let maps: Vec<u32> = match args.take_str("--maps") {
        Some(list) => {
            let mut out = Vec::new();
            for part in list.split(',') {
                out.push(
                    part.trim()
                        .parse()
                        .map_err(|_| format!("--maps: {part:?} is not a level number"))?,
                );
            }
            if out.is_empty() {
                return Err("--maps needs at least one level".into());
            }
            out
        }
        None => vec![cfg.map],
    };
    // A scenario is a level the engine builds. Named here, it replaces the
    // game's own levels for every episode of the run; the real levels are
    // then free to be what a run is SCORED on, which is the only role in
    // which "it has never seen this level" means anything.
    let scenarios: Vec<String> = match args.take_str("--scenario") {
        Some(list) => {
            let out: Vec<String> = list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if out.is_empty() {
                return Err("--scenario needs at least one name".into());
            }
            out
        }
        None => Vec::new(),
    };
    let approach = match args.take_str("--approach") {
        Some(v) => Some(
            v.trim()
                .parse::<f32>()
                .map_err(|_| format!("--approach: {v:?} is not a number"))?,
        ),
        None => None,
    };
    let payment = match args.take_str("--reward") {
        Some(v) => Payment::parse(v.trim()).ok_or_else(|| {
            format!(
                "--reward: {v:?} is not one of {}",
                Payment::ALL.map(|p| p.name()).join(", ")
            )
        })?,
        None => Payment::Shaped,
    };
    let parsed = Args {
        command,
        maps,
        scenarios,
        approach,
        payment,
        doom_bin: args.take_str("--doom-bin"),
        wad: args.take_str("--wad"),
        cfg,
        mission,
        mix: args.take_flag("--mix"),
        arena: args.usize_or("--arena", 0),
        curriculum: args.take_flag("--curriculum"),
        start_distance: args.usize_or("--start-distance", 0) as i32,
        eval_episodes: args.usize_or("--eval-episodes", 24),
        wide: args.take_flag("--wide"),
        play: args.usize_or("--play", 3),
        transcript: args.take_str("--transcript"),
        engine_log: args.take_str("--engine-log"),
        record: args.take_str("--record"),
        tic_steps: args.take_flag("--tic-steps"),
        hardware,
        train,
        view,
    };
    args.finish();
    Ok(parsed)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("doom: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let paths = Paths::resolve(args.doom_bin.as_deref(), args.wad.as_deref())
        .map_err(|m| format!("{m}"))?;
    if !std::path::Path::new(args.encoder())
        .join("config.json")
        .exists()
    {
        return Err(format!(
            "no sentence encoder at {}\n  run `brain pull sentence-transformers/all-MiniLM-L6-v2`, \
             or pass --encoder DIR",
            args.encoder()
        ));
    }

    println!(
        "doom: E{}M{} at skill {}, {}",
        args.cfg.episode,
        args.maps
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join("/"),
        args.cfg.skill,
        if args.mix {
            "orders sampled per episode".into()
        } else {
            format!("orders: {}", args.mission.name())
        }
    );

    let game = Doom::start(
        &paths,
        &args.cfg,
        args.transcript.as_ref().map(Into::into),
        args.engine_log.as_ref().map(Into::into),
    )
    .map_err(|e| format!("could not start the game: {e}"))?;
    println!("doom: engine up on port {}, lockstep", game.port);
    let mut env = DoomEnv::new(game, args.cfg.clone(), args.mission, args.mix);
    env.set_maps(args.maps.clone());
    env.set_scenarios(args.scenarios.clone());
    env.set_max_steps(args.max_steps());
    env.set_approach(args.approach);
    env.set_payment(args.payment);
    env.set_arena(args.arena);
    env.set_curriculum(args.curriculum);
    env.set_start_distance(args.start_distance);
    if args.curriculum {
        println!(
            "doom: reverse curriculum - episodes start {} units of walking from the exit",
            args.start_distance
        );
    }
    if args.arena > 0 {
        println!(
            "doom: arena - {} monsters placed around the player each episode",
            args.arena
        );
    }

    match args.command.as_str() {
        "probe" => view::probe(env, &args),
        "play" => view::play(env, &args),
        "bench" => view::bench(env, &args),
        "eval" => evaluate(env, &args),
        "fit" => fit(env, &args),
        "whatif" => whatif(env, &args),
        _ => train(env, &args),
    }
}

/// Warm-start on the scripted player, then improve it with PPO.
fn train(env: DoomEnv, args: &Args) -> Result<(), String> {
    let inspect = env.inspect.clone();
    // The shared group builds the spec: one place decides what every flag
    // means, so this sample cannot quietly disagree with the next one about
    // what --epochs does.
    let spec = args.train.spec();

    // A window during training is optional and shows the SAME inspector the
    // play command does, fed from the environment as the rollouts run.
    let watcher = view::watch(inspect, args);

    let mut builder = ControlPipeline::builder(args.encoder(), env)
        .seed(args.seed())
        .device(args.device());
    if let Some(h) = args.head() {
        builder = builder.head(h);
    }
    // No `.evaluate()` stage here. The SDK's evaluation runs a fixed 200
    // episodes, which is the right number for a game whose episodes are forty
    // decisions long and the wrong one for this: at this horizon it is 40
    // minutes of wall clock after every training run, and it scores the policy
    // against nothing. What follows instead scores the policy and the scripted
    // player over the same episodes, which is the comparison that means
    // something.
    // Named and described BEFORE the chain runs, because the chain is what
    // writes the file. A head is an adapter to a specific encoder fitted for
    // a specific task, and a checkpoint that says neither is 1.7 MB nobody
    // can use.
    let started = builder.load().map(|mut p| {
        p.describe(
            "swedishembedded/minilm-l6-option-head-doom",
            serde_json::json!({
                "sample": "decision/doom",
                "task": "choose the next action in DOOM from the options the game offers",
                "mission": args.mission.name(),
                "levels": if args.scenarios.is_empty() {
                    serde_json::json!({ "kind": "game", "maps": args.maps })
                } else {
                    serde_json::json!({ "kind": "generated", "scenarios": args.scenarios })
                },
                "decisions_per_episode": args.max_steps(),
                "iterations": spec.iterations,
                "episodes_per_iteration": spec.episodes,
                "warm_start_episodes": spec.warmup_episodes,
                "encoder": "frozen",
            }),
        );
        p
    });
    let chain = brain::Flow::new(started)
        .train(spec)
        .save(&args.train.save)
        .report();
    let out = chain.finish().map_err(|e| format!("{e}"));
    watcher.stop();
    let mut pipe = out?;
    println!("doom: wrote {}\n", args.train.save);

    let seeds: Vec<u64> = eval_seeds(args);
    println!(
        "doom: scoring both players over the same {} episodes",
        seeds.len()
    );
    let mut pt = view::Timing::default();
    let mut st = view::Timing::default();
    let learned = view::score_policy(&mut pipe, &seeds, args.max_steps(), None, &mut pt)?;
    let script = view::score_scripted(pipe.env_mut(), &seeds, args.max_steps(), &mut st)?;
    report(&script, &learned);
    pt.print("policy");
    st.print("scripted");
    Ok(())
}

/// Can the model express the teacher's decisions at all?
///
/// Every other question about this policy is downstream of that one and none
/// of them can be answered while it is open. A policy gradient improves a
/// policy the architecture is able to represent; if the observation the agent
/// reads and the head that ranks its options cannot reproduce a decision even
/// when handed the answer, no budget of rollouts will find it, and a run that
/// fails to improve says nothing about the algorithm.
///
/// So: fit the head to the scripted player with plain supervised learning,
/// then ask how often it agrees - on the episodes it was fitted to, and on
/// episodes generated from seeds it has never been given. No reward is
/// involved, no critic, no advantage. See
/// `brain::ControlPipeline::teacher_agreement` for how the three numbers are
/// read.
fn fit(env: DoomEnv, args: &Args) -> Result<(), String> {
    let spec = args.train.spec();
    let mut builder = ControlPipeline::builder(args.encoder(), env)
        .seed(args.seed())
        .device(args.device());
    if let Some(h) = args.head() {
        builder = builder.head(h);
    }
    let mut pipe = builder.load().map_err(|e| format!("{e}"))?;
    println!(
        "doom: fitting the {} to {} scripted episodes x {} passes, then asking how \
         often it agrees",
        if spec.freeze_encoder { "head" } else { "head AND the encoder" },
        spec.warmup_episodes,
        spec.warmup_epochs
    );
    let probed = pipe
        .probe_teacher(
            spec.warmup_episodes,
            spec.warmup_epochs,
            spec.max_steps,
            spec.warmup_keep,
            args.eval_episodes,
            spec.freeze_encoder,
        )
        .map_err(|e| format!("{e}"))?;
    let Some((before, fitted, unseen)) = probed else {
        return Err("the environment has no scripted player to compare against".into());
    };
    println!("\n  before fitting, on unseen episodes   {before}");
    println!("  after fitting, on the same episodes  {fitted}");
    println!("  after fitting, on unseen episodes    {unseen}");
    let floor = unseen.floor();
    println!(
        "\n{}",
        if before.top1 > floor + 0.25 {
            "doom: an UNFITTED head already agrees with the teacher far more than a \
             constant policy does - the measurement is wrong, and the two numbers under \
             it mean nothing until it is explained"
        } else if fitted.top1 < floor + 0.15 {
            "doom: the head cannot reproduce the teacher even on the episodes it was \
             fitted to. The decision is not expressible from what the agent reads, and \
             no amount of policy gradient will find it - what the observation carries is \
             what to fix"
        } else if unseen.top1 < fitted.top1 * 0.7 {
            "doom: it reproduces the episodes it was fitted to and not the ones it was \
             not - it memorised them. More DISTINCT worlds, not more steps in these"
        } else {
            "doom: the representation carries the decision, on worlds it has never seen. \
             What is left to explain is ACTING: the states a policy reaches once it stops \
             being steered by the teacher are not these states, and nothing here has \
             labelled those"
        }
    );
    Ok(())
}

/// Would a different action have been worth taking?
///
/// The go/no-go before building anything that learns from the answer. Every
/// method this sample has tried asks which action the TEACHER took, and the
/// best any of them can do is reach the teacher - measured, they all do, and
/// the teacher is where the policy already was. The only question left whose
/// answer can beat it is this one.
///
/// So it is asked directly, before a line of algorithm is written for it: go
/// back to decisions the policy actually faced, take something other than
/// what the teacher chose, let the teacher play the rest out, and score the
/// whole trajectory. If the teacher's action is best nearly everywhere there
/// is no improvement signal to learn from and that is the finding; if it is
/// not, the size of the gap is the budget everything downstream has to work
/// inside.
fn whatif(env: DoomEnv, args: &Args) -> Result<(), String> {
    let spec = args.train.spec();
    let mut builder = ControlPipeline::builder(args.encoder(), env)
        .seed(args.seed())
        .device(args.device());
    if let Some(h) = args.head() {
        builder = builder.head(h);
    }
    let mut pipe = builder.load().map_err(|e| format!("{e}"))?;
    // The policy only decides which alternatives are worth the game steps, so
    // it has to be a policy rather than a fresh head - but it does not have
    // to be a good one, and a supplied head skips a warm start that would
    // cost more than the measurement does.
    if args.head().is_none() {
        println!(
            "doom: no --head, so warm-starting on {} scripted episodes first - the policy \
             is what picks which alternatives are worth trying",
            spec.warmup_episodes
        );
        let mut quiet = |_: usize, _: f32| {};
        pipe.warm_start(&spec, &mut quiet).map_err(|e| format!("{e}"))?;
    }
    println!(
        "doom: {} decisions from {} episodes, {} {} alternatives each, {}, then the \
         teacher finishing every branch",
        spec.states,
        spec.episodes,
        spec.alternatives,
        if args.wide { "random" } else { "contested" },
        match spec.deviate {
            0 | 1 => "one decision off the teacher".to_string(),
            n => format!("{n} decisions of the policy carrying on"),
        }
    );
    let found = pipe
        .counterfactual(
            spec.episodes,
            spec.states,
            spec.alternatives,
            if args.wide { Candidates::Wide } else { Candidates::Contested },
            spec.deviate,
            spec.max_steps,
        )
        .map_err(|e| format!("{e}"))?;
    let Some(c) = found else {
        return Err("the environment has no scripted player to finish the branches".into());
    };
    if c.adrift > 0 {
        println!(
            "\ndoom: {} of {} decisions could not be returned to - the replay did not land \
             where it started, so nothing below is measuring what it says it is",
            c.adrift,
            c.adrift + c.states
        );
    }
    if c.states == 0 {
        return Err("no decision could be returned to".into());
    }
    println!(
        "\n  {} decisions, {} game steps\n  \
         at {:.0}% of them the options did not all lead to the same place\n  \
         the choice was worth {:.3} of score between its best and worst option\n  \
         some alternative beat the teacher at {:.0}% of them, by {:.3} when it did\n  \
         picking the best of what was offered would gain {:.3} a decision over the teacher",
        c.states,
        c.steps,
        c.pivotal * 100.0,
        c.spread,
        c.beaten * 100.0,
        c.gain,
        c.regret
    );
    println!("  where the time went: {}", c.spend);
    if !c.by_situation.is_empty() {
        println!("\n  {:<26} {:>10} {:>10} {:>10}", "", "decisions", "mattered", "room");
        for s in &c.by_situation {
            println!(
                "  {:<26} {:>10} {:>9.0}% {:>10.3}",
                s.label,
                s.states,
                s.pivotal * 100.0,
                s.regret
            );
        }
    }
    println!(
        "\n{}",
        if c.pivotal < 0.1 {
            "doom: the choice barely matters at these decisions - every option leads to \
             about the same place, so there is nothing here for any method to learn and \
             the thing to change is WHERE the decisions are sampled from"
        } else if c.regret < 0.01 {
            "doom: the teacher is already choosing the best of what is offered, near \
             enough. There is no improvement signal in this teacher, and beating it needs \
             a better one or a wider candidate set - not a better learner"
        } else {
            "doom: there IS room above the teacher, and this is how much of it. Training a \
             ranker on measured outcomes rather than on the teacher's choice is worth \
             building, and this number is the ceiling it has to be judged against"
        }
    );
    Ok(())
}

/// The episodes an evaluation is scored over.
///
/// FIXED for a given `--seed`, which is the whole point: the worlds a policy
/// is judged on must not move while the policy does, or the difference
/// between two scores is mostly the difference between two sets of worlds.
/// On a generated level the seed IS the world, so this matters here far more
/// than it does on a fixed map.
///
/// `--seed` shifts the whole block, which is how the spread BETWEEN blocks
/// gets measured - the part of a score that is the draw rather than the
/// player.
fn eval_seeds(args: &Args) -> Vec<u64> {
    (0..args.eval_episodes as u64)
        .map(|i| 5_000_000 + args.seed().wrapping_mul(1009) + i)
        .collect()
}

fn report(script: &view::Score, learned: &view::Score) {
    println!(
        "\n{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "", "return", "game", "kills", "items", "exits", "deaths", "burned", "progress"
    );
    script.row("scripted");
    learned.row("policy");
    let delta = learned.mean_return - script.mean_return;
    let game = learned.mean_game - script.mean_game;
    println!(
        "\ndoom: the policy is {delta:+.2} return and {game:+.2} game score per episode \
         against the scripted player{}",
        if delta > 0.0 && game > 0.0 {
            ""
        } else {
            " - it has not beaten it yet"
        }
    );
}

/// Score the policy and the scripted player on the SAME episodes.
///
/// The same seeds, the same missions, the same horizon. A win rate measured
/// against a baseline that ran different episodes is not a comparison, and the
/// number that matters here is the difference between two columns of this
/// table rather than either column alone.
fn evaluate(mut env: DoomEnv, args: &Args) -> Result<(), String> {
    let seeds: Vec<u64> = eval_seeds(args);

    println!("\ndoom: scripted player over {} episodes", seeds.len());
    let mut st = view::Timing::default();
    let script = view::score_scripted(&mut env, &seeds, args.max_steps(), &mut st)?;
    script.print("scripted");
    st.print("scripted");

    let Some(head) = args.head() else {
        println!("\ndoom: no --head given, so only the reference bar was measured");
        return Ok(());
    };
    println!(
        "\ndoom: policy {head} over the same {} episodes",
        seeds.len()
    );
    let mut pipe = ControlPipeline::builder(args.encoder(), env)
        .head(head)
        .seed(args.seed())
        .device(args.device())
        .load()
        .map_err(|e| format!("{e}"))?;
    let mut pt = view::Timing::default();
    let learned = view::score_policy(&mut pipe, &seeds, args.max_steps(), None, &mut pt)?;
    learned.print("policy");
    pt.print("policy");

    report(&script, &learned);
    Ok(())
}
