// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Score a sales conversation while it is still happening - as ONE expression.
//!
//! ```text
//! ConversionPipeline::from_pretrained(encoder)
//!     .train(spec)        // supervised warm start, then policy gradient
//!     .evaluate()         // accuracy, AUC-ROC, Brier, per-turn agreement
//!     .save(head)
//!     .replay(held_out)   // watch the probability move, turn by turn
//!     .tui()              // then type your own conversation at it
//!     .report()
//!     .finish()?
//! ```
//!
//! `train`, `evaluate`, `save`, `tui`, `report` and `finish` are the stages
//! EVERY brain pipeline has. `replay` is this architecture's own, added
//! through the same seam. A failed stage stops the chain and the rest skip, so
//! the whole program has exactly one error site.
//!
//! The model answers one proposition - *will this conversation close* - after
//! every turn, and attaches a confidence to the answer that says whether to act
//! on it, look for comparable cases, escalate, or hand it to a person.
//!
//! Nothing here is generated. The output is a number in `[0, 1]` and a routing
//! decision, and the conversation never leaves the machine.
//!
//! Getting started:
//!
//! ```text
//! ./fetch-dataset.sh                                  # ~53 MB, once
//! make samples/decision/salesagent/run                # train, evaluate, replay, chat
//! make samples/decision/salesagent/run ARGS="--head out/sales-head.safetensors"
//! ```
//!
//! Swedish Embedded AB builds conversation intelligence that runs on the
//! customer's own hardware - scoring a live dialogue, in milliseconds, without
//! shipping it to a third party. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use brain::{ConversionPipeline, ConversionSpec};

struct Args {
    encoder: String,
    data: String,
    head_in: Option<String>,
    save_to: String,
    train: usize,
    eval: usize,
    warmup: usize,
    policy: usize,
    replay: usize,
    interactive: bool,
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR").unwrap_or_else(|_| {
            format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
        }),
        data: std::env::var("BRAIN_TESTDATA")
            .map(|r| format!("{r}/decide/salesconv"))
            .unwrap_or_else(|_| "testdata/decide/salesconv".into()),
        head_in: None,
        save_to: "out/sales-head.safetensors".into(),
        train: 4000,
        eval: 400,
        warmup: 3000,
        policy: 1000,
        replay: 2,
        interactive: true,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--encoder" => a.encoder = next(),
            "--data" => a.data = next(),
            "--head" => a.head_in = Some(next()),
            "--save" => a.save_to = next(),
            "--train" => a.train = next().parse().unwrap_or(a.train),
            "--eval" => a.eval = next().parse().unwrap_or(a.eval),
            "--warmup" => a.warmup = next().parse().unwrap_or(a.warmup),
            "--policy" => a.policy = next().parse().unwrap_or(a.policy),
            "--replay" => a.replay = next().parse().unwrap_or(a.replay),
            "--batch" => {
                a.interactive = false;
                i -= 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: salesagent [--encoder DIR] [--data DIR] [--head FILE] [--save FILE]\n\
                     \x20                 [--train N] [--eval N] [--warmup N] [--policy N]\n\
                     \x20                 [--replay N] [--batch]\n\n\
                     Without --head: trains, evaluates, saves, replays, then chats.\n\
                     With --head:    skips training and scores with those weights.\n\
                     --batch:        skip the interactive stage (for scripted runs).\n\n\
                     Fetch the data first with ./fetch-dataset.sh"
                );
                std::process::exit(0);
            }
            other => eprintln!("salesagent: ignoring unknown argument {other:?}"),
        }
        i += 2;
    }
    a
}

/// Rule 5 of samples/README.md: say what is missing and leave cleanly, rather
/// than panicking or starting a multi-gigabyte download.
fn require(path: &std::path::Path, what: &str, remedy: &str) {
    if !path.exists() {
        eprintln!("salesagent: no {what} at {}", path.display());
        eprintln!("salesagent: {remedy}");
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
    require(
        &std::path::Path::new(&args.data).join("train.jsonl"),
        "sales conversation data",
        "run samples/decision/salesagent/fetch-dataset.sh, or pass --data DIR",
    );

    let data = match brain::sales_conversations(&args.data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("salesagent: {e}");
            std::process::exit(1);
        }
    };
    let train: Vec<_> = data.train.iter().take(args.train).cloned().collect();
    let eval: Vec<_> = data.test.iter().take(args.eval).cloned().collect();
    println!(
        "salesagent: {} training conversations ({:.0}% closed), {} held out, {:.0} turns on average",
        train.len(),
        100.0 * data.base_rate(),
        eval.len(),
        train.iter().map(|c| c.len()).sum::<usize>() as f32 / train.len().max(1) as f32,
    );

    // Two conversations to replay afterwards: one that closed and one that did
    // not, so the trajectories can be read against each other.
    let per_outcome = args.replay / 2 + args.replay % 2;
    let replay: Vec<_> = [true, false]
        .iter()
        .flat_map(|&want| eval.iter().rev().filter(move |c| c.outcome == want).take(per_outcome))
        .take(args.replay)
        .cloned()
        .collect();

    let spec = ConversionSpec::default()
        .train(train)
        .eval(eval)
        .warmup_steps(args.warmup)
        .policy_steps(args.policy)
        .calibration(200)
        .seed(11);

    // ---- the whole program ------------------------------------------------
    let mut builder = ConversionPipeline::builder(&args.encoder);
    if let Some(h) = &args.head_in {
        require(std::path::Path::new(h), "head weights", "train first, or drop --head");
        builder = builder.head(h);
    }
    let mut chain = brain::Flow::new(builder.load());

    // Training is skipped when trained weights were supplied: the SAME chain
    // serves both, because a stage that is not wanted is simply not in it.
    if args.head_in.is_none() {
        chain = chain.train(spec).evaluate().save(&args.save_to);
    }
    let mut chain = chain.replay(replay);
    if args.interactive {
        println!(
            "\nsalesagent: type turns as `customer: ...` or `rep: ...` (bare text is the customer).\n\
             \x20           `reset` starts a new conversation; an empty line or ^D stops."
        );
        chain = chain.tui();
    }

    if let Err(e) = chain.report().finish() {
        eprintln!("salesagent: {e}");
        std::process::exit(1);
    }
}
