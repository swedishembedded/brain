// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Learn to play a corridor shooter by reinforcement - as ONE expression.
//!
//! ```text
//! ControlPipeline::from_pretrained(encoder, Arena)
//!     .train(spec)     // PPO: act, collect returns, update
//!     .evaluate()      // greedy episodes on unseen seeds
//!     .save(policy)
//!     .play(3)         // watch it play, decision by decision
//!     .report()
//!     .finish()?
//! ```
//!
//! **The action set changes every tick.** You can only shoot a monster that is
//! alive and in range, only grab a medkit still on the floor, only reload with
//! reserve left. A policy network whose output layer is the action space
//! cannot express that - its width is fixed when the weights are created. Here
//! the actions arrive with the observation as TEXT, so the same policy handles
//! "shoot the imp at range 2" on one tick and "reload, retreat" on the next,
//! and reads what an option MEANS rather than looking it up by index.
//!
//! This is also the setting where the reinforcement learning is genuine: the
//! action decides which state the next decision is made from, so the policy
//! shifts its own data distribution and PPO's trust region is doing real work.
//!
//! Run it:
//!
//! ```text
//! make samples/decision/arena/run
//! make samples/decision/arena/run ARGS="--head out/arena-policy.safetensors --play 5"
//! ```
//!
//! Swedish Embedded AB builds realtime control policies that run on the
//! customer's own hardware - deciding among actions a system defines at run
//! time, in milliseconds, with no text generator in the loop. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod arena;

use arena::Arena;
use brain::{ControlPipeline, ControlSpec, Env};

/// The game, presented to the SDK.
///
/// Everything interesting is in `actions`: it is called every tick and returns
/// a different list almost every time.
struct ArenaEnv {
    game: Arena,
}

impl Env for ArenaEnv {
    fn reset(&mut self, seed: u64) -> String {
        self.game.reset(seed);
        self.game.observe()
    }

    fn actions(&mut self) -> Vec<String> {
        self.game.actions()
    }

    fn step(&mut self, action: usize) -> (String, f32, bool) {
        let (reward, done) = self.game.step(action);
        (self.game.observe(), reward, done)
    }

    fn objective(&self) -> String {
        "which action best clears this corridor without dying".to_string()
    }

    fn render(&self) -> Option<String> {
        Some(self.game.render())
    }

    fn won(&self) -> bool {
        self.game.cleared
    }

    /// The scripted teacher: shoot whatever is in range, else reload.
    ///
    /// One line, no target priority, no retreat, no medkit - the same
    /// always-shoot heuristic the sample prints as a reference bar. Cloning it
    /// is not the result; the result is whether the policy gradient then
    /// IMPROVES on it, because the teacher leaves obvious value on the table.
    fn demo(&mut self) -> Option<usize> {
        let opts = self.game.actions();
        opts.iter()
            .position(|o| o.starts_with("shoot"))
            .or_else(|| opts.iter().position(|o| o == "reload"))
            .or(Some(0))
    }
}

struct Args {
    encoder: String,
    head_in: Option<String>,
    save_to: String,
    iterations: usize,
    episodes: usize,
    epochs: usize,
    play: usize,
    interactive: bool,
    train_encoder: bool,
    entropy: Option<f32>,
    warmup: Option<usize>,
    head_seed: u64,
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR").unwrap_or_else(|_| {
            format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
        }),
        head_in: None,
        save_to: "out/arena-policy.safetensors".into(),
        iterations: 12,
        episodes: 24,
        epochs: 2,
        play: 3,
        interactive: true,
        train_encoder: false,
        entropy: None,
        warmup: None,
        head_seed: 1,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--encoder" => a.encoder = next(),
            "--head" => a.head_in = Some(next()),
            "--save" => a.save_to = next(),
            "--iterations" => a.iterations = next().parse().unwrap_or(a.iterations),
            "--episodes" => a.episodes = next().parse().unwrap_or(a.episodes),
            "--epochs" => a.epochs = next().parse().unwrap_or(a.epochs),
            "--play" => a.play = next().parse().unwrap_or(a.play),
            "--entropy" => a.entropy = next().parse().ok(),
            "--warmup" => a.warmup = next().parse().ok(),
            "--head-seed" => a.head_seed = next().parse().unwrap_or(0),
            "--train-encoder" => {
                a.train_encoder = true;
                i -= 1;
            }
            "--batch" => {
                a.interactive = false;
                i -= 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: arena [--encoder DIR] [--head FILE] [--save FILE]\n\
                     \x20            [--iterations N] [--episodes N] [--epochs N]\n\
                     \x20            [--play N] [--warmup N] [--entropy F] [--train-encoder] [--batch]\n\n\
                     Without --head: trains by reinforcement, evaluates, saves, then plays.\n\
                     With --head:    skips training and plays with those weights.\n\
                     --batch:        skip the interactive stage."
                );
                std::process::exit(0);
            }
            other => eprintln!("arena: ignoring unknown argument {other:?}"),
        }
        i += 2;
    }
    a
}

/// Rule 5 of samples/README.md: say what is missing and leave cleanly.
fn require(path: &std::path::Path, what: &str, remedy: &str) {
    if !path.exists() {
        eprintln!("arena: no {what} at {}", path.display());
        eprintln!("arena: {remedy}");
        std::process::exit(1);
    }
}

fn main() {
    let args = parse_args();
    require(
        &std::path::Path::new(&args.encoder).join("config.json"),
        "encoder checkpoint",
        "run `brain pull sentence-transformers/all-MiniLM-L6-v2`, or pass --encoder DIR",
    );

    println!(
        "arena: a {}-cell corridor, 2-3 monsters, the action set rebuilt every tick",
        arena::CORRIDOR
    );
    // The two bars any learned number has to be read against, measured on the
    // SAME seeds `evaluate` uses. Without them a win rate is unreadable: this
    // game is winnable often enough by accident that "wins more than half" can
    // mean the policy learned nothing.
    let [random, teacher, ceiling] = arena::reference_band(brain::control::EVAL_SEEDS);
    println!(
        "arena: reference band on the evaluation seeds\n\
         \x20        random                 {:5.1}%  return {:+.3}\n\
         \x20        teacher (always shoot) {:5.1}%  return {:+.3}   <- what cloning starts from\n\
         \x20        best hand-written      {:5.1}%  return {:+.3}   <- the ceiling worth chasing",
        random.0 * 100.0,
        random.1,
        teacher.0 * 100.0,
        teacher.1,
        ceiling.0 * 100.0,
        ceiling.1,
    );

    let env = ArenaEnv { game: Arena::new() };
    let mut spec = ControlSpec::default()
        .iterations(args.iterations)
        .episodes(args.episodes)
        .epochs(args.epochs)
        .train_encoder(args.train_encoder)
        .seed(11);
    if let Some(e) = args.entropy {
        spec.policy.entropy = e;
    }
    if let Some(w) = args.warmup {
        spec.warmup_episodes = w;
    }

    // ---- the whole program ------------------------------------------------
    let mut builder = ControlPipeline::builder(&args.encoder, env).seed(args.head_seed);
    if let Some(h) = &args.head_in {
        require(std::path::Path::new(h), "policy weights", "train first, or drop --head");
        builder = builder.head(h);
    }
    let mut chain = brain::Flow::new(builder.load());

    // `evaluate` runs either way: a policy loaded from disk is exactly the
    // thing you want to be able to score, and skipping it left a saved run
    // unmeasurable.
    if args.head_in.is_none() {
        chain = chain.train(spec);
    }
    chain = chain.evaluate();
    if args.head_in.is_none() {
        chain = chain.save(&args.save_to);
    }
    let mut chain = chain.play(args.play);
    if args.interactive {
        println!("\narena: press enter to watch another episode (empty line twice or ^D to stop)");
        chain = chain.tui();
    }

    if let Err(e) = chain.report().finish() {
        eprintln!("arena: {e}");
        std::process::exit(1);
    }
}
