// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does a decision model actually READ its options?
//!
//! This is the architectural claim the whole decision surface rests on: the
//! options arrive at run time, as text, and the model scores them by what they
//! MEAN rather than by where they sit. Everything built on top - an agent
//! choosing among moves a game invents each tick, a router choosing among
//! services registered this morning - is worth exactly as much as that claim
//! is true.
//!
//! It is also the claim that is easiest to pass without meaning to. Train on a
//! fixed list of 77 intents and a model can reach a high number having learned
//! a 77-way classifier wearing a costume: position alone identifies the
//! answer, and nothing forces it to look at the words. So this run does three
//! things that a conventional intent benchmark does not:
//!
//! 1. **Every example sees a random SUBSET of the options**, always including
//!    the right one, at a random position. There is no stable index to learn.
//! 2. **Some intents are held back entirely.** The model trains on none of
//!    their examples and is then asked to pick them, by name, out of a list.
//!    A model that learned the option text can do this. A disguised classifier
//!    scores chance.
//! 3. **A shuffled-state control**, which re-scores the same held-out examples
//!    with the input text replaced by a different example's. Accuracy has to
//!    collapse toward chance. If it does not, the model is reading the option
//!    list alone - the answer is leaking from the question - and the headline
//!    number means nothing at all.
//!
//! ```text
//! intents --encoder DIR --data testdata/decide/banking77 --unseen 12
//! ```
//!
//! The head it saves is the natural starting point for any decision task that
//! has no labelled data of its own - including a reinforcement-learning run,
//! which begins from a head that already knows how to compare option text
//! rather than from noise.
//!
//! Swedish Embedded AB builds the evaluation that tells a customer whether a
//! model does what its architecture claims, rather than whether it scores well
//! on the benchmark it was fitted to. If your team needs that, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::path::PathBuf;

use brain::options::{Args, Hardware, ModelOptions, Options, SupervisedOptions};
use brain::{DecisionPipeline, Device};

/// What the model is asked, on every example. Fixed for the run: what varies
/// is the options, which is the point.
const INSTRUCTIONS: &str = "Which of these is the customer asking about?";

struct Settings {
    data: PathBuf,
    unseen: usize,
    split_seed: u64,
    hardware: Hardware,
    train: SupervisedOptions,
}

fn usage() -> String {
    format!(
        "\
usage: intents [options]

Trains a decision head on some BANKING77 intents and scores it on intents it
has never seen, to measure whether it reads its options or memorises them.

  --data DIR          BANKING77 (categories.json, train.csv, test.csv)
  --unseen N          intents held back from training entirely   [12]
  --split-seed N      which intents those are                    [7]

hardware
{}

training
{}
{}
",
        Hardware::help(),
        ModelOptions::help(),
        SupervisedOptions::help()
    )
}

fn parse() -> Result<Settings, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        std::process::exit(0);
    }
    let mut args = Args::new(&argv);
    let hardware = Hardware::take(&mut args)?;
    let train = SupervisedOptions::new(ModelOptions::new("", "out/intents-head.safetensors"))
        .take_over(&mut args)?;
    train.model.require_encoder()?;
    let settings = Settings {
        data: PathBuf::from(args.str_or("--data", "testdata/decide/banking77")),
        unseen: args.usize_or("--unseen", 12),
        split_seed: args.u64_or("--split-seed", 7),
        hardware,
        train,
    };
    args.finish();
    Ok(settings)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("intents: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let s = parse()?;
    if !s.data.join("train.csv").exists() {
        return Err(format!(
            "no BANKING77 at {}\n  run `make fetch/testdata`, or pass --data DIR",
            s.data.display()
        ));
    }
    s.hardware.apply()?;

    let data = brain::decision::banking77(&s.data).map_err(|e| format!("{e}"))?;
    let split = brain::decision::intent_holdout(data.categories.len(), s.unseen, s.split_seed);
    println!(
        "intents: {} utterances over {} intents; training on {}, holding back {}",
        data.train.len(),
        data.categories.len(),
        split.seen.len(),
        split.unseen.len()
    );
    println!("intents: hardware {}", s.hardware.describe());

    // Only examples whose intent is in the SEEN half. The held-back intents
    // are not merely absent from the option lists - the model never sees one
    // of their utterances either.
    let train_rows: Vec<(&str, usize)> = data
        .train
        .iter()
        .filter(|r| split.contains_seen(r.label))
        .map(|r| (r.text.as_str(), r.label))
        .collect();
    let options: Vec<String> = (0..data.categories.len()).map(|i| data.option_text(i)).collect();

    let mut pipe = DecisionPipeline::builder(&s.train.model.encoder)
        .seed(s.train.model.seed)
        .device(Device::default())
        .load()
        .map_err(|e| format!("{e}"))?;

    // Before: what this scores untrained is the bar every later number has to
    // clear, and it is not 1/77 - the options are a random subset, so chance
    // depends on how many were drawn.
    let before = score(&mut pipe, &data, &split.unseen, &options, s.train.eval, s.train.model.seed)?;
    println!("intents: before training, unseen-intent accuracy {:.1}%", before.accuracy * 100.0);

    println!("intents: {} steps over {} examples", s.train.steps, train_rows.len());
    let mut last = 0usize;
    let loss = pipe
        .train_choices(
            &train_rows,
            &options,
            INSTRUCTIONS,
            s.train.steps,
            s.train.model.seed,
            &mut |step, l| {
                if step / 200 > last {
                    last = step / 200;
                    println!("  step {step:>6}  loss {l:.4}");
                }
            },
        )
        .map_err(|e| format!("{e}"))?;
    println!("intents: final loss {loss:.4}");

    let seen = score(&mut pipe, &data, &split.seen, &options, s.train.eval, s.train.model.seed)?;
    let unseen =
        score(&mut pipe, &data, &split.unseen, &options, s.train.eval, s.train.model.seed)?;
    let control = shuffled_control(
        &mut pipe,
        &data,
        &split.unseen,
        &options,
        s.train.eval,
        s.train.model.seed,
    )?;

    println!("\n{:<28} {:>10} {:>10} {:>12}", "", "accuracy", "chance", "examples");
    seen.row("intents trained on");
    unseen.row("intents NEVER trained on");
    control.row("  same, state shuffled");

    pipe.save_head(&s.train.model.save).map_err(|e| format!("{e}"))?;
    println!("\nintents: wrote {}", s.train.model.save);

    verdict(&seen, &unseen, &control);
    Ok(())
}

struct Scored {
    accuracy: f32,
    chance: f32,
    n: usize,
}

impl Scored {
    fn row(&self, label: &str) {
        println!(
            "{:<28} {:>9.1}% {:>9.1}% {:>12}",
            label,
            self.accuracy * 100.0,
            self.chance * 100.0,
            self.n
        );
    }
}

/// Score the test utterances whose intent is in `labels`.
///
/// Each is offered a random option subset drawn the same way training drew
/// them, so the number is comparable to the training distribution rather than
/// to an easier or harder one.
fn score(
    pipe: &mut DecisionPipeline,
    data: &brain::decision::Banking77,
    labels: &[usize],
    options: &[String],
    limit: usize,
    seed: u64,
) -> Result<Scored, String> {
    let rows: Vec<&brain::decision::Row> =
        data.test.iter().filter(|r| labels.binary_search(&r.label).is_ok()).collect();
    let mut rng = brain::decision::Rng::new(seed ^ 0x5c0);
    let pool: Vec<usize> = (0..options.len()).collect();
    let (mut right, mut chance, mut n) = (0usize, 0.0f32, 0usize);

    for row in rows.iter().take(limit) {
        let (drawn, gold) = brain::decision::draw_options(row.label, &pool, &mut rng);
        let texts: Vec<&str> = drawn.iter().map(|&i| options[i].as_str()).collect();
        let choice = pipe.choose(&row.text, INSTRUCTIONS, &texts).map_err(|e| format!("{e}"))?;
        if choice.index == gold {
            right += 1;
        }
        chance += 1.0 / drawn.len() as f32;
        n += 1;
    }
    Ok(Scored {
        accuracy: right as f32 / n.max(1) as f32,
        chance: chance / n.max(1) as f32,
        n,
    })
}

/// The same held-out examples, each scored against ANOTHER example's state.
///
/// If accuracy survives this, the model is not reading the input at all - it
/// is picking from the option list on some property of the list itself, and
/// the headline number is measuring the wrong thing. It is the cheapest
/// possible check and it is the one nobody runs.
fn shuffled_control(
    pipe: &mut DecisionPipeline,
    data: &brain::decision::Banking77,
    labels: &[usize],
    options: &[String],
    limit: usize,
    seed: u64,
) -> Result<Scored, String> {
    let rows: Vec<&brain::decision::Row> =
        data.test.iter().filter(|r| labels.binary_search(&r.label).is_ok()).collect();
    let mut rng = brain::decision::Rng::new(seed ^ 0x5c0);
    let pool: Vec<usize> = (0..options.len()).collect();
    let (mut right, mut chance, mut n) = (0usize, 0.0f32, 0usize);

    for (i, row) in rows.iter().enumerate().take(limit) {
        // The option set still belongs to THIS row - only the state is wrong.
        let (drawn, gold) = brain::decision::draw_options(row.label, &pool, &mut rng);
        let texts: Vec<&str> = drawn.iter().map(|&k| options[k].as_str()).collect();
        let other = rows[(i + rows.len() / 2) % rows.len()];
        let choice = pipe.choose(&other.text, INSTRUCTIONS, &texts).map_err(|e| format!("{e}"))?;
        if choice.index == gold {
            right += 1;
        }
        chance += 1.0 / drawn.len() as f32;
        n += 1;
    }
    Ok(Scored {
        accuracy: right as f32 / n.max(1) as f32,
        chance: chance / n.max(1) as f32,
        n,
    })
}

/// Say what the three numbers mean together, because separately they are easy
/// to read as whatever one hoped for.
fn verdict(seen: &Scored, unseen: &Scored, control: &Scored) {
    let reads_options = unseen.accuracy > unseen.chance * 2.0;
    let reads_state = control.accuracy < unseen.accuracy / 2.0;
    println!();
    if reads_options {
        println!(
            "intents: the model picks intents it never trained on at {:.1}% against {:.1}% chance \
             - it is reading the OPTION TEXT, not an index.",
            unseen.accuracy * 100.0,
            unseen.chance * 100.0
        );
    } else {
        println!(
            "intents: unseen-intent accuracy {:.1}% is not clear of {:.1}% chance. On this run \
             the model has NOT shown it reads its options.",
            unseen.accuracy * 100.0,
            unseen.chance * 100.0
        );
    }
    if reads_state {
        println!(
            "intents: shuffling the input collapses it to {:.1}% - it is reading the STATE too.",
            control.accuracy * 100.0
        );
    } else {
        println!(
            "intents: WARNING - shuffling the input leaves {:.1}% against {:.1}% unshuffled. The \
             answer is leaking from the option list; treat the numbers above as unproven.",
            control.accuracy * 100.0,
            unseen.accuracy * 100.0
        );
    }
    println!(
        "intents: trained-on intents score {:.1}%, held-out {:.1}% - the drop is the price of \
         never having seen them.",
        seen.accuracy * 100.0,
        unseen.accuracy * 100.0
    );
}
