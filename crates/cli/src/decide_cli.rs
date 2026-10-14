// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain decide train|inspect` - from outcome-labelled decision examples to a
//! checkpoint a server can load.
//!
//! The other `brain decide` verbs (`decide`, `models`, `select`) are the
//! capability family's own and are dispatched generically; this module owns
//! only the verbs that act on a dataset and on a checkpoint's lifecycle. The
//! work is done by `brain::decision_run`; this file parses flags and prints.
//!
//! Swedish Embedded AB implements the tooling that turns recorded outcomes
//! into governed decision models, from the first training run to a promotion
//! gate. If your team needs expertise in decision-model lifecycle, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::PathBuf;

use brain::decision_run::{train, TrainRun, DEFAULT_STEPS};
use decisions::examples::Dataset;

const USAGE: &str = "\
usage: brain decide train   --data FILE --base REF --out DIR [--freeze-encoder | --finetune-encoder]
                            [--steps N] [--batch N] [--seed N]
       brain decide inspect --data FILE [--json]

train     Train a decision head on the `train` split of a JSONL file of
          outcome-labelled choices and write a checkpoint directory. The
          encoder is frozen unless --finetune-encoder is given. The
          checkpoint's manifest records the base, data digest, split counts,
          hyperparameters, seed and brain commit, and says `unvalidated`:
          only a promotion can change that.
inspect   Validate a JSONL file (schema, duplicate ids, leakage across
          splits) and print its summary: counts, label entropy and the
          majority-label baseline.
";

/// Whether `verb` is one this module runs rather than the capability family.
pub fn handles(verb: Option<&str>) -> bool {
    matches!(verb, Some("train" | "inspect"))
}

/// `brain decide <verb> ...`; `argv` starts at the verb. Returns the exit code.
pub fn run_decide(argv: &[String]) -> i32 {
    let verb = argv.first().map(String::as_str).unwrap_or_default();
    let mut args = crate::args::Args::new(&argv[1..]);
    if args.take_flag("--help") || args.take_flag("-h") {
        print!("{USAGE}");
        return 0;
    }
    match verb {
        "train" => train_verb(&mut args),
        "inspect" => inspect_verb(&mut args),
        other => {
            eprintln!("brain decide: unknown verb {other:?}\n\n{USAGE}");
            2
        }
    }
}

/// A required `--flag VALUE`, or the usage error.
fn required(args: &mut crate::args::Args, flag: &str) -> Result<String, i32> {
    args.take_str(flag).ok_or_else(|| {
        eprintln!("brain decide: {flag} is required\n\n{USAGE}");
        2
    })
}

fn inspect_verb(args: &mut crate::args::Args) -> i32 {
    let data = match required(args, "--data") {
        Ok(d) => PathBuf::from(d),
        Err(code) => return code,
    };
    let as_json = args.take_flag("--json");
    args.finish();
    match Dataset::read_file(&data) {
        Ok(dataset) => {
            let summary = dataset.summary();
            if as_json {
                println!("{}", serde_json::json!({"data_sha256": dataset.digest, "summary": summary.to_json()}));
            } else {
                print!("{}", summary.render());
            }
            0
        }
        Err(e) => {
            eprintln!("brain decide: {}: {e}", data.display());
            1
        }
    }
}

fn train_verb(args: &mut crate::args::Args) -> i32 {
    let (data, base, out) = match (required(args, "--data"), required(args, "--base"), required(args, "--out")) {
        (Ok(d), Ok(b), Ok(o)) => (d, b, o),
        _ => return 2,
    };
    let mut run = TrainRun::new(data, base, out);
    let finetune = args.take_flag("--finetune-encoder");
    let freeze = args.take_flag("--freeze-encoder");
    if finetune && freeze {
        eprintln!("brain decide: --freeze-encoder and --finetune-encoder contradict each other");
        return 2;
    }
    run.freeze_encoder = !finetune;
    run.steps = args.usize_or("--steps", DEFAULT_STEPS);
    run.batch = args.usize_or("--batch", brain::decision::DEFAULT_BATCH);
    run.seed = args.u64_or("--seed", 0);
    args.finish();
    run.brain_commit = code_revision();

    let steps = run.steps;
    let mut log = |step: usize, loss: f32| {
        if step % 50 == 0 || step + 1 == steps {
            eprintln!("step {:>5}/{steps}  loss {loss:.4}", step + 1);
        }
    };
    match train(&run, &mut log) {
        Ok(outcome) => {
            print!("{}", outcome.summary.render());
            println!("wrote {} (final loss {:.4}, unvalidated)", outcome.out.display(), outcome.final_loss);
            println!("evaluate with: brain decide eval --model {} --data {} --split validation", outcome.out.display(), run.data.display());
            0
        }
        Err(e) => {
            eprintln!("brain decide: {e}");
            1
        }
    }
}

/// `git rev-parse --short HEAD` of the checkout this binary was built from,
/// `-dirty` when its tree has changes, `unknown` when git cannot say.
fn code_revision() -> String {
    let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from)) else { return "unknown".into() };
    let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(&dir).args(args).output().ok().filter(|o| o.status.success());
    let Some(head) = git(&["rev-parse", "--short", "HEAD"]).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|s| !s.is_empty()) else {
        return "unknown".into();
    };
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|o| !o.stdout.is_empty());
    if dirty { format!("{head}-dirty") } else { head }
}
