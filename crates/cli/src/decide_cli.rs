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

use brain::decision_run::{evaluate_model, promote_model, rollback_model, train, EvalRun, PromoteRun, TrainRun, DEFAULT_STEPS};
use decisions::examples::{Dataset, Split};

const USAGE: &str = "\
usage: brain decide train   --data FILE --base REF --out DIR [--freeze-encoder | --finetune-encoder]
                            [--steps N] [--batch N] [--seed N] [--activations keep|recompute]
       brain decide eval    --model DIR|REF --data FILE --split validation|test
                            [--costs FILE] [--order-seed N] [--json] [--out FILE]
       brain decide promote --model DIR --data FILE --policy POLICY.json [--json]
       brain decide rollback --model DIR
       brain decide inspect --data FILE [--json]

train     Train a decision head on the `train` split of a JSONL file of
          outcome-labelled choices and write a checkpoint directory. The
          encoder is frozen unless --finetune-encoder is given. The
          checkpoint's manifest records the base, data digest, split counts,
          hyperparameters, seed and brain commit, and says `unvalidated`:
          only a promotion can change that. A Laya base (the preferred stock
          model) always trains its head with the trunk frozen; the output is
          the head plus a reference to the base by id and digest, not a copy of
          the weights, and the base must stay installed to serve it.
          --finetune-encoder is refused for Laya. --activations picks what the
          backward keeps of the forward: keep every layer's activations, or
          recompute them from each layer's input for the same gradients at
          lower memory; absent leaves the backend's default.
eval      Measure a checkpoint on one split: accuracy against the majority and
          chance baselines, NLL, Brier, ECE/AdaECE with reliability bins,
          coverage versus accuracy, option-order instability under a seeded
          permutation, cost-weighted regret when --costs names a file
          {\"false_exclusion\": X, \"wasted_tokens\": Y}, and a per-family
          breakdown. Prints a table; --json prints the record instead and
          --out also writes it to FILE.
promote   Re-evaluate DIR on the `test` split under the policy and, only if
          every gate passes, mark it validated for the policy's profile (the
          manifest gains the validation and weights_sha256 the `auto`
          selector requires, the evaluation record is stored beside it, and
          the checkpoint it supersedes becomes the last-known-good). A failed
          gate is printed with how far it was missed, nothing changes, and
          the exit code is 1. Numbers in the manifest are never trusted.
rollback  Make the last-known-good checkpoint the validated one again. DIR
          must be the profile's current checkpoint.
inspect   Validate a JSONL file (schema, duplicate ids, leakage across
          splits) and print its summary: counts, label entropy and the
          majority-label baseline.
";

/// Whether `verb` is one this module runs rather than the capability family.
pub fn handles(verb: Option<&str>) -> bool {
    matches!(verb, Some("train" | "eval" | "promote" | "rollback" | "inspect"))
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
        "eval" => eval_verb(&mut args),
        "promote" => promote_verb(&mut args),
        "rollback" => rollback_verb(&mut args),
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

fn eval_verb(args: &mut crate::args::Args) -> i32 {
    let (model, data, split) = match (required(args, "--model"), required(args, "--data"), required(args, "--split")) {
        (Ok(m), Ok(d), Ok(s)) => (m, d, s),
        _ => return 2,
    };
    let Some(split) = Split::parse(&split).filter(|s| *s != Split::Train) else {
        eprintln!("brain decide: --split must be `validation` or `test`; the train split is not held out, got {split:?}");
        return 2;
    };
    let mut run = EvalRun::new(model, data, split);
    run.costs = args.take_str("--costs").map(PathBuf::from);
    run.order_seed = args.seed_flag("brain decide eval", "--order-seed");
    let as_json = args.take_flag("--json");
    let out = args.take_str("--out").map(PathBuf::from);
    args.finish();
    match evaluate_model(&run) {
        Ok(evaluation) => {
            let record = serde_json::to_string_pretty(&evaluation.to_json()).expect("a JSON value serializes");
            if let Some(path) = out {
                if let Err(e) = std::fs::write(&path, format!("{record}\n")) {
                    eprintln!("brain decide: {}: {e}", path.display());
                    return 1;
                }
            }
            if as_json {
                println!("{record}");
            } else {
                println!("model {}  weights {}", evaluation.model_id, evaluation.weights_sha256);
                print!("{}", evaluation.report.render());
            }
            0
        }
        Err(e) => {
            eprintln!("brain decide: {e}");
            1
        }
    }
}

fn promote_verb(args: &mut crate::args::Args) -> i32 {
    let (model, data, policy) = match (required(args, "--model"), required(args, "--data"), required(args, "--policy")) {
        (Ok(m), Ok(d), Ok(p)) => (m, d, p),
        _ => return 2,
    };
    let as_json = args.take_flag("--json");
    args.finish();
    let outcome = match promote_model(&PromoteRun { model, data: PathBuf::from(data), policy: PathBuf::from(policy) }) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("brain decide: {e}");
            return 1;
        }
    };
    if as_json {
        println!("{}", serde_json::json!({"promoted": outcome.pointer.is_some(), "evaluation": outcome.evaluation.to_json(), "verdict": outcome.verdict.to_json()}));
    } else {
        print!("{}{}", outcome.evaluation.report.render(), outcome.verdict);
    }
    match &outcome.pointer {
        Some(pointer) => {
            if !as_json {
                println!("promoted {} for profile {:?}; last-known-good: {}", pointer.current.dir, pointer.profile, pointer.last_known_good.as_ref().map_or("none", |e| e.dir.as_str()));
            }
            0
        }
        None => {
            eprintln!("not promoted: {} gate(s) failed; nothing was changed", outcome.verdict.failures().count());
            for gate in outcome.verdict.failures() {
                eprintln!("  {}", gate.message);
            }
            1
        }
    }
}

fn rollback_verb(args: &mut crate::args::Args) -> i32 {
    let model = match required(args, "--model") {
        Ok(m) => m,
        Err(code) => return code,
    };
    args.finish();
    match rollback_model(&model) {
        Ok(pointer) => {
            println!("rolled back: {} is validated again for profile {:?}", pointer.current.dir, pointer.profile);
            0
        }
        Err(e) => {
            eprintln!("brain decide: {e}");
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
    run.seed = args.seed("brain decide train");
    match args.take_str(crate::args::ACTIVATIONS_FLAG).map(|value| crate::args::parse_activations(&value)).transpose() {
        Ok(policy) => run.saved = policy,
        Err(e) => {
            eprintln!("brain decide: {e}");
            return 2;
        }
    }
    args.finish();
    run.brain_commit = code_revision();

    let steps = run.steps;
    let mut log = |step: usize, loss: f32| {
        if step.is_multiple_of(50) || step + 1 == steps {
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
