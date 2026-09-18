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
mod obs;
mod view;

use brain::{ControlPipeline, ControlSpec};
use doom::{Config, Doom, Paths};
use env::{DoomEnv, Mission};

struct Args {
    command: String,
    doom_bin: Option<String>,
    wad: Option<String>,
    encoder: String,
    head: Option<String>,
    save: String,
    mission: Mission,
    mix: bool,
    cfg: Config,
    iterations: usize,
    episodes: usize,
    epochs: usize,
    warmup: usize,
    max_steps: usize,
    eval_episodes: usize,
    play: usize,
    window: bool,
    frames: Option<String>,
    transcript: Option<String>,
    fps: u32,
    entropy: Option<f32>,
    seed: u64,
}

const USAGE: &str = "\
usage: doom <train|eval|play|probe> [options]

  train    warm-start on the scripted player, then improve it by PPO
  eval     score a policy and the scripted player on the SAME episodes
  play     run episodes and show every decision as it is made
  probe    one scripted episode, for artifacts and for checking the plumbing

where to find the game (no path is ever baked in)
  --doom-bin PATH     the restful-doom binary   [$RESTFUL_DOOM, then $PATH]
  --wad PATH          an IWAD                   [$DOOM_WAD, $DOOM_WAD_DIR/doom1.wad]
  --encoder DIR       sentence encoder          [$BRAIN_MINILM_DIR]

what to play
  --episode N --map N --skill 0..4      [1 1 2]
  --mission clear|speedrun|survive      [clear]
  --mix               sample a mission per episode, so the policy must read it
  --max-steps N       decisions per episode     [220]

training
  --iterations N --episodes N --epochs N --warmup N   [10 12 2 40]
  --entropy F         exploration bonus
  --head FILE         start from (or, for eval/play, use) these weights
  --save FILE         where to write them       [out/doom-policy.safetensors]
  --seed N

looking at it
  --window            open a window (otherwise headless, which is the default
                      on a machine with no display)
  --frames DIR        write every decision's frame + overlay as a PNG
  --transcript FILE   write every request and reply as JSON lines
  --fps N             cap the window's pace so a human can follow it   [12]
  --play N            episodes for `play`      [3]
  --eval-episodes N   episodes for `eval`      [24]
";

fn parse_args() -> Result<Args, String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut a = Args {
        command: argv[0].clone(),
        doom_bin: None,
        wad: None,
        encoder: std::env::var("BRAIN_MINILM_DIR").unwrap_or_else(|_| {
            format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
        }),
        head: None,
        save: "out/doom-policy.safetensors".into(),
        mission: Mission::Clear,
        mix: false,
        cfg: Config::default(),
        iterations: 10,
        episodes: 12,
        epochs: 2,
        warmup: 40,
        max_steps: 220,
        eval_episodes: 24,
        play: 3,
        window: false,
        frames: None,
        transcript: None,
        fps: 12,
        entropy: None,
        seed: 11,
    };
    if !["train", "eval", "play", "probe"].contains(&a.command.as_str()) {
        return Err(format!("unknown command {:?}\n\n{USAGE}", a.command));
    }
    let mut i = 1;
    while i < argv.len() {
        let mut want = || -> Result<String, String> {
            argv.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        let mut consumed = 2;
        match argv[i].as_str() {
            "--doom-bin" => a.doom_bin = Some(want()?),
            "--wad" => a.wad = Some(want()?),
            "--encoder" => a.encoder = want()?,
            "--head" => a.head = Some(want()?),
            "--save" => a.save = want()?,
            "--mission" => {
                let m = want()?;
                a.mission = Mission::parse(&m).ok_or_else(|| format!("unknown mission {m:?}"))?;
            }
            "--episode" => a.cfg.episode = want()?.parse().map_err(|_| "--episode")?,
            "--map" => a.cfg.map = want()?.parse().map_err(|_| "--map")?,
            "--skill" => a.cfg.skill = want()?.parse().map_err(|_| "--skill")?,
            "--iterations" => a.iterations = want()?.parse().map_err(|_| "--iterations")?,
            "--episodes" => a.episodes = want()?.parse().map_err(|_| "--episodes")?,
            "--epochs" => a.epochs = want()?.parse().map_err(|_| "--epochs")?,
            "--warmup" => a.warmup = want()?.parse().map_err(|_| "--warmup")?,
            "--max-steps" => a.max_steps = want()?.parse().map_err(|_| "--max-steps")?,
            "--eval-episodes" => a.eval_episodes = want()?.parse().map_err(|_| "--eval-episodes")?,
            "--play" => a.play = want()?.parse().map_err(|_| "--play")?,
            "--frames" => a.frames = Some(want()?),
            "--transcript" => a.transcript = Some(want()?),
            "--fps" => a.fps = want()?.parse().map_err(|_| "--fps")?,
            "--entropy" => a.entropy = want()?.parse().ok(),
            "--seed" => a.seed = want()?.parse().map_err(|_| "--seed")?,
            "--mix" => {
                a.mix = true;
                consumed = 1;
            }
            "--window" => {
                a.window = true;
                consumed = 1;
            }
            other => return Err(format!("unknown argument {other:?}\n\n{USAGE}")),
        }
        i += consumed;
    }
    if a.cfg.skill > 4 {
        return Err("--skill must be 0..4".into());
    }
    Ok(a)
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
    if !std::path::Path::new(&args.encoder).join("config.json").exists() {
        return Err(format!(
            "no sentence encoder at {}\n  run `brain pull sentence-transformers/all-MiniLM-L6-v2`, \
             or pass --encoder DIR",
            args.encoder
        ));
    }

    println!(
        "doom: E{}M{} at skill {}, {}",
        args.cfg.episode,
        args.cfg.map,
        args.cfg.skill,
        if args.mix { "orders sampled per episode".into() } else { format!("orders: {}", args.mission.name()) }
    );

    let game = Doom::start(&paths, &args.cfg, args.transcript.as_ref().map(Into::into))
        .map_err(|e| format!("could not start the game: {e}"))?;
    println!("doom: engine up on port {}, lockstep", game.port);
    let env = DoomEnv::new(game, args.cfg.clone(), args.mission, args.mix);

    match args.command.as_str() {
        "probe" => view::probe(env, &args),
        "play" => view::play(env, &args),
        "eval" => evaluate(env, &args),
        _ => train(env, &args),
    }
}

/// Warm-start on the scripted player, then improve it with PPO.
fn train(env: DoomEnv, args: &Args) -> Result<(), String> {
    let inspect = env.inspect.clone();
    let mut spec = ControlSpec::default()
        .iterations(args.iterations)
        .episodes(args.episodes)
        .epochs(args.epochs)
        .max_steps(args.max_steps)
        .warmup_episodes(args.warmup)
        .seed(args.seed);
    if let Some(e) = args.entropy {
        spec.policy.entropy = e;
    }

    // A window during training is optional and shows the SAME inspector the
    // play command does, fed from the environment as the rollouts run.
    let watcher = view::watch(inspect, args);

    let mut builder = ControlPipeline::builder(&args.encoder, env).seed(args.seed);
    if let Some(h) = &args.head {
        builder = builder.head(h);
    }
    let chain = brain::Flow::new(builder.load())
        .train(spec)
        .evaluate()
        .save(&args.save)
        .report();
    let out = chain.finish().map_err(|e| format!("{e}"));
    watcher.stop();
    out?;
    println!("doom: wrote {}", args.save);
    Ok(())
}

/// Score the policy and the scripted player on the SAME episodes.
///
/// The same seeds, the same missions, the same horizon. A win rate measured
/// against a baseline that ran different episodes is not a comparison, and the
/// number that matters here is the difference between two columns of this
/// table rather than either column alone.
fn evaluate(mut env: DoomEnv, args: &Args) -> Result<(), String> {
    let seeds: Vec<u64> = (0..args.eval_episodes as u64).map(|i| 5_000_000 + i).collect();

    println!("\ndoom: scripted player over {} episodes", seeds.len());
    let script = view::score_scripted(&mut env, &seeds, args.max_steps)?;
    script.print("scripted");

    let Some(head) = args.head.as_ref() else {
        println!("\ndoom: no --head given, so only the reference bar was measured");
        return Ok(());
    };
    println!("\ndoom: policy {head} over the same {} episodes", seeds.len());
    let mut pipe = ControlPipeline::builder(&args.encoder, env)
        .head(head)
        .seed(args.seed)
        .load()
        .map_err(|e| format!("{e}"))?;
    let learned = view::score_policy(&mut pipe, &seeds, args.max_steps, None)?;
    learned.print("policy");

    println!("\n{:<10} {:>8} {:>8} {:>8} {:>8} {:>8}", "", "return", "kills", "items", "exits", "deaths");
    script.row("scripted");
    learned.row("policy");
    let delta = learned.mean_return - script.mean_return;
    println!(
        "\ndoom: the policy is {:+.2} return per episode against the scripted player{}",
        delta,
        if delta > 0.0 { "" } else { " - it has not beaten it yet" }
    );
    Ok(())
}
