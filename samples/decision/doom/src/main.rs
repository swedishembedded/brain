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
mod obs;
mod report;
mod view;

use brain::options::{Args as Args_, ControlOptions, Hardware, Options, ViewOptions};
use brain::ControlPipeline;
use doom::{Config, Doom, Paths};
use env::{DoomEnv, Mission};

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
usage: doom <train|eval|play|probe|bench> [options]

  train    warm-start on the scripted player, then improve it by PPO
  eval     score a policy and the scripted player on the SAME episodes
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
    if !["train", "eval", "play", "probe", "bench"].contains(&command.as_str()) {
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
    let parsed = Args {
        command,
        maps,
        doom_bin: args.take_str("--doom-bin"),
        wad: args.take_str("--wad"),
        cfg,
        mission,
        mix: args.take_flag("--mix"),
        arena: args.usize_or("--arena", 0),
        curriculum: args.take_flag("--curriculum"),
        start_distance: args.usize_or("--start-distance", 0) as i32,
        eval_episodes: args.usize_or("--eval-episodes", 24),
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
    let chain = brain::Flow::new(builder.load())
        .train(spec)
        .save(&args.train.save)
        .report();
    let out = chain.finish().map_err(|e| format!("{e}"));
    watcher.stop();
    let mut pipe = out?;
    println!("doom: wrote {}\n", args.train.save);

    let seeds: Vec<u64> = (0..args.eval_episodes as u64)
        .map(|i| 5_000_000 + i)
        .collect();
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

fn report(script: &view::Score, learned: &view::Score) {
    println!(
        "\n{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "", "return", "game", "kills", "items", "exits", "deaths", "burned"
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
    let seeds: Vec<u64> = (0..args.eval_episodes as u64)
        .map(|i| 5_000_000 + i)
        .collect();

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
