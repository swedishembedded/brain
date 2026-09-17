// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Train a decision model on BANKING77 and then keep answering with it - as
//! ONE expression.
//!
//! ```text
//! DecisionPipeline::from_pretrained(encoder)
//!     .train(spec)
//!     .evaluate()
//!     .save(head)
//!     .tui()
//!     .report()
//!     .finish()?
//! ```
//!
//! `train`, `evaluate`, `save`, `ask`, `tui`, `report` and `finish` are the
//! stages EVERY brain pipeline has; what each means here is supplied by the
//! decision architecture, including the shape of its training specification. A
//! failed stage stops the chain and every later stage skips, so this whole
//! program has exactly one error site.
//!
//! What a decision model is for shows up in the last stage: the option set is
//! part of the CALL, so the same trained model keeps answering new messages
//! against it with no reload, and returns a distribution rather than a label so
//! the CALLER decides what is confident enough to act on.
//!
//! Run it:
//!
//! ```text
//! make samples/decision/triage/run ARGS="--steps 2000 --intents 12"
//! make samples/decision/triage/run ARGS="--head out/triage-head.safetensors --ask 'my card never arrived'"
//! ```
//!
//! Swedish Embedded AB builds realtime decision layers - calibrated,
//! bounded-output models that choose among actions a system defines at run
//! time - for its clients. If your team needs judgment in the loop without
//! handing control flow to a text generator, you can procure our services by
//! sending an email to info@swedishembedded.com.

use brain::{DecisionPipeline, TrainSpec};

/// What the model is asked, every time. The instructions travel with the
/// OPTIONS, never with the message - which is what lets one encoding of the
/// message serve every question asked about it.
const INSTRUCTIONS: &str = "which banking intent does this message express";

struct Args {
    encoder: String,
    data: String,
    head_in: Option<String>,
    save_to: String,
    steps: usize,
    intents: usize,
    ask: Option<String>,
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR").unwrap_or_else(|_| {
            format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
        }),
        data: std::env::var("BRAIN_TESTDATA")
            .map(|r| format!("{r}/decide/banking77"))
            .unwrap_or_else(|_| "testdata/decide/banking77".into()),
        head_in: None,
        save_to: "out/triage-head.safetensors".into(),
        steps: 600,
        intents: 8,
        ask: None,
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
            "--steps" => a.steps = next().parse().unwrap_or(a.steps),
            "--intents" => a.intents = next().parse().unwrap_or(a.intents),
            "--ask" => a.ask = Some(next()),
            "--help" | "-h" => {
                eprintln!(
                    "usage: triage [--encoder DIR] [--data DIR] [--head FILE] [--save FILE]\n\
                     \x20             [--steps N] [--intents N] [--ask MESSAGE]\n\n\
                     Without --head: trains, evaluates, saves, then answers.\n\
                     With --head:    skips training and answers with those weights.\n\
                     Without --ask:  reads messages from stdin until end of input."
                );
                std::process::exit(0);
            }
            other => eprintln!("triage: ignoring unknown argument {other:?}"),
        }
        i += 2;
    }
    a
}

/// Rule 5 of samples/README.md: say what is missing and leave cleanly, rather
/// than panicking or starting a multi-gigabyte download.
fn require(path: &std::path::Path, what: &str, remedy: &str) {
    if !path.exists() {
        eprintln!("triage: no {what} at {}", path.display());
        eprintln!("triage: {remedy}");
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
        &std::path::Path::new(&args.data).join("train.csv"),
        "BANKING77 data",
        "run `make fetch/testdata`, or pass --data DIR",
    );

    let data = match brain::decision::banking77(&args.data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("triage: {e}");
            std::process::exit(1);
        }
    };

    // The intents this run decides between, and the text each is scored as.
    let intents: Vec<usize> = (0..args.intents.min(data.categories.len())).collect();
    let options: Vec<String> = intents.iter().map(|&i| data.option_text(i)).collect();
    println!("triage: {} options: {}", options.len(), options.join(", "));

    // A row's label indexes the DATASET's categories; the model is trained
    // against THIS run's option list, so the two are remapped rather than
    // assumed to coincide.
    let take = |rows: &[brain::decision::Row]| -> Vec<(String, usize)> {
        rows.iter()
            .filter_map(|r| intents.iter().position(|i| *i == r.label).map(|p| (r.text.clone(), p)))
            .collect()
    };
    let spec = TrainSpec::new(INSTRUCTIONS)
        .examples(take(&data.train))
        .eval(take(&data.test).into_iter().take(160).collect())
        .options(options.clone())
        .steps(args.steps)
        .seed(11);

    // ---- the whole program ------------------------------------------------
    let mut flow = DecisionPipeline::builder(&args.encoder);
    if let Some(h) = &args.head_in {
        require(std::path::Path::new(h), "head weights", "train first, or drop --head");
        flow = flow.head(h);
    }
    let mut chain = brain::Flow::new(flow.load());

    // Training is skipped when trained weights were supplied: the SAME chain
    // serves both, because a stage that is not wanted is simply not in it.
    if args.head_in.is_none() {
        chain = chain.train(spec).evaluate().save(&args.save_to);
    } else {
        chain = chain.with_question(INSTRUCTIONS, options);
    }

    let chain = match &args.ask {
        Some(message) => chain.ask(message),
        None => {
            println!("\ntriage: type a customer message and press enter (empty line or ^D to stop)");
            chain.tui()
        }
    };

    if let Err(e) = chain.report().finish() {
        eprintln!("triage: {e}");
        std::process::exit(1);
    }
}
