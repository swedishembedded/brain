// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One pretrained decision model, nine production decision patterns.
//!
//! A small, fast model in front of an LLM (or instead of one) shows up in
//! the same shape over and over: model routing, guardrails, tool-call
//! gating, inbox triage, reranking, LLM-output evals, bulk labeling,
//! real-time control, and a confidence gate. Every one of those is an
//! application of what `brain::DecisionPipeline` already does - a
//! runtime-supplied [`brain::decision::Question`] scored against a text
//! state - not a new capability. This sample generates its own data for
//! seven of the patterns (`src/tasks.rs`) and measures the SAME loaded
//! model, zero-shot, against every one of them:
//!
//! ```text
//! (no --demo)       zero-shot held-out accuracy per task - patterns 1-6
//! --demo bulk       batch the triage task over a big table, throughput
//! --demo realtime   repeat the control task, measured per-call latency
//! --demo gate       confidence-band the tool-gate task's own answers
//! --demo rerank     rank candidates for one query by calibrated probability
//! ```
//!
//! Defaults to `convaiinnovations/laya`, a decision model pretrained to
//! answer exactly this shape of question - `choice`/`score`/`noul` over
//! runtime-supplied options - without any task-specific training
//! (`samples/decision/json`'s own measured run already establishes this
//! zero-shot competence and its limits; this sample measures it again,
//! against these seven synthetic tasks specifically, rather than reusing
//! that number). `--model minilm` runs the same seven tasks against a
//! `crates/decide` encoder instead, which needs a trained head to answer
//! well - see this sample's README for what that comparison shows.
//!
//! Run it:
//!
//! ```text
//! make samples/decision/patterns/run
//! make samples/decision/patterns/run ARGS="--demo bulk"
//! make samples/decision/patterns/run ARGS="--demo realtime"
//! make samples/decision/patterns/run ARGS="--demo gate"
//! make samples/decision/patterns/run ARGS="--demo rerank"
//! ```
//!
//! Swedish Embedded AB builds decision systems where a small, calibrated
//! model - not an LLM call - makes the routing, gating, and triage
//! decisions in front of your product, with real measured accuracy per
//! pattern rather than a demo that only shows the happy path. If your team
//! needs a decision layer like this in production, you can procure our
//! services by sending an email to info@swedishembedded.com.

mod tasks;

use std::path::{Path, PathBuf};

use brain::decision::Answer;
use brain::options::{Args, Hardware, ModelChoice, Options};
use brain::{DecisionPipeline, Device};
use tasks::{Rng, Task, TASKS};

/// The short names this sample knows, mapped to what `brain pull` calls
/// them - the same table `samples/decision/json` uses, reproduced here
/// rather than shared because a sample may depend on no brain crate except
/// the SDK facade (samples/README.md rule 1).
const ALIASES: &[(&str, &str)] = &[("laya", "convaiinnovations/laya"), ("minilm", "sentence-transformers/all-MiniLM-L6-v2"), ("decide", "sentence-transformers/all-MiniLM-L6-v2")];

struct Settings {
    model: ModelChoice,
    head: Option<String>,
    demo: Demo,
    eval_n: usize,
    seed: u64,
    finetune_steps: usize,
    hardware: Hardware,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Demo {
    Report,
    Bulk,
    Realtime,
    Gate,
    Rerank,
}

fn usage() -> String {
    format!(
        "\
usage: patterns [options]

Always measures the loaded model zero-shot against seven synthetic tasks
(each generating its own train/eval data, see src/tasks.rs), then either
prints the per-task held-out accuracy report (default) or runs one demo
against the same model.

  --demo report|bulk|realtime|gate|rerank\n\
                      which pattern to run (default: report, patterns 1-6)
  --eval-n N          held-out examples per task (default 100)
  --seed N            RNG seed for data generation (default 7)
  --finetune-steps N  generically fine-tune the five fixed-vocabulary tasks
                      (routing, guardrail, tool_gate, triage, control) via
                      DecisionPipeline::train_choices before reporting - the
                      SAME call for every one of them, no task-specific
                      training code (default 0, zero-shot only). `eval` and
                      `rerank` have no fixed option vocabulary to fine-tune
                      against and stay zero-shot either way - see
                      tasks::fixed_vocab's own doc.

model
{}
  --head FILE         trained head weights, for an encoder that ships
                      without one (a Laya checkpoint carries its own)

hardware
{}

known model names: {}
",
        ModelChoice::help(),
        Hardware::help(),
        ALIASES.iter().map(|(a, _)| *a).collect::<Vec<_>>().join(", "),
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
    let model = ModelChoice::new("laya").take_over(&mut args)?;
    let head = args.take_str("--head");
    let demo = match args.take_str("--demo").as_deref() {
        None | Some("report") => Demo::Report,
        Some("bulk") => Demo::Bulk,
        Some("realtime") => Demo::Realtime,
        Some("gate") => Demo::Gate,
        Some("rerank") => Demo::Rerank,
        Some(other) => return Err(format!("unknown --demo {other:?}, expected report|bulk|realtime|gate|rerank")),
    };
    let eval_n = args.usize_or("--eval-n", 100);
    let seed = args.u64_or("--seed", 7);
    let finetune_steps = args.usize_or("--finetune-steps", 0);
    args.finish();
    Ok(Settings { model, head, demo, eval_n, seed, finetune_steps, hardware })
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("patterns: {e}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let s = parse()?;
    let dir = locate(&s.model.name)?;
    s.hardware.apply()?;
    eprintln!("patterns: {} from {} on {}", s.model.name, dir, s.hardware.describe());

    let mut builder = DecisionPipeline::builder(&dir).device(Device::default());
    if let Some(head) = &s.head {
        if !Path::new(head).is_file() {
            return Err(format!("no head weights at {head} - train some with samples/decision/triage"));
        }
        builder = builder.head(head);
    }
    let mut pipe = builder.load().map_err(|e| format!("loading {dir}: {e}"))?;

    if s.demo == Demo::Report {
        run_report(&mut pipe, s.eval_n, s.seed);
    }
    if s.finetune_steps > 0 {
        finetune(&mut pipe, s.finetune_steps, s.seed);
        if s.demo == Demo::Report {
            run_finetuned_report(&mut pipe, s.eval_n, s.seed);
        }
    }
    match s.demo {
        Demo::Report => {}
        Demo::Bulk => run_bulk(&mut pipe, s.eval_n.max(200), s.seed),
        Demo::Realtime => run_realtime(&mut pipe, s.eval_n.max(50), s.seed),
        Demo::Gate => run_gate(&mut pipe, s.eval_n, s.seed),
        Demo::Rerank => run_rerank(&mut pipe, s.seed),
    }
    Ok(())
}

/// Fine-tune the five fixed-vocabulary tasks, ONE reusable call
/// (`DecisionPipeline::train_choices`, already generic across the `Decide`
/// and Laya backends - see its own doc) applied identically to each. No
/// per-task training code here at all: what differs between tasks is only
/// the `instructions`/`options`/examples [`tasks::fixed_vocab`] hands back.
fn finetune(pipe: &mut DecisionPipeline, steps: usize, seed: u64) {
    println!("\n--- fine-tuning (generic, {steps} steps per task) ---");
    for &task in &TASKS {
        let Some(fv) = tasks::fixed_vocab(task) else { continue };
        let examples: Vec<(&str, usize)> = fv.train.iter().map(|(t, g)| (t.as_str(), *g)).collect();
        let mut log = |_step: usize, _loss: f32| {};
        let tail_loss = pipe.train_choices(&examples, &fv.options, fv.instructions, steps, seed ^ (task as u64), &mut log).expect("train_choices");
        println!("  {:<10} tail loss {tail_loss:.4}", task.name());
    }
}

/// The held-out report AFTER [`finetune`]: the same five tasks, evaluated
/// with the SAME bare option names training used (`DecisionPipeline::choose`,
/// not [`ask`]'s descriptive [`tasks::generate`] question) - a fine-tuned
/// head answers the text it was actually trained on, not a differently
/// worded version of the same question. `eval`/`rerank` are unaffected by
/// fine-tuning (see [`tasks::fixed_vocab`]'s doc) and are not repeated here.
fn run_finetuned_report(pipe: &mut DecisionPipeline, eval_n: usize, seed: u64) {
    println!("\n--- held-out accuracy after fine-tuning ---");
    for &task in &TASKS {
        let Some(fv) = tasks::fixed_vocab(task) else { continue };
        let opts: Vec<&str> = fv.options.iter().map(String::as_str).collect();
        // Sampled WITH replacement from the held-out pool, the same way
        // `tasks::generate`'s own EVAL banks are drawn - so `eval_n` is
        // comparable to the zero-shot report's, even though a task's real
        // held-out vocabulary is only a handful of phrasings.
        let mut rng = Rng::new(seed ^ (task as u64).wrapping_mul(0x2545F4914F6CDD1D));
        let mut correct = 0usize;
        let mut confidence_sum = 0.0f32;
        for _ in 0..eval_n {
            let (text, gold) = rng.choice(&fv.eval);
            let a = pipe.choose(text, fv.instructions, &opts).expect("choose");
            confidence_sum += a.confidence;
            if a.index == *gold {
                correct += 1;
            }
        }
        println!("  {:<10} {:.3}  (mean confidence {:.3}, {eval_n} held-out examples)", task.name(), correct as f32 / eval_n as f32, confidence_sum / eval_n as f32);
    }
}

/// The index of an [`Answer`]'s highest-probability option - `Choice` picks
/// among its options, `Score` among its levels, in both cases in the SAME
/// order [`tasks::generate`] built the question in, so this is directly
/// comparable to an `Example::gold` index.
fn answer_index(a: &Answer) -> usize {
    match a {
        Answer::Choice { probabilities, .. } => probabilities.iter().enumerate().max_by(|x, y| x.1 .1.total_cmp(&y.1 .1)).map(|(i, _)| i).unwrap_or(0),
        Answer::Score { probabilities, .. } => probabilities.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i).unwrap_or(0),
        Answer::Noul { noul } => usize::from(*noul > 0.5),
    }
}

fn answer_confidence(a: &Answer) -> f32 {
    match a {
        Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => *confidence,
        Answer::Noul { noul } => (2.0 * (noul - 0.5).abs()).clamp(0.0, 1.0),
    }
}

fn ask(pipe: &mut DecisionPipeline, ex: &tasks::Example) -> Answer {
    let state = brain::decision::State::Str(ex.state.clone());
    pipe.decide(&state, std::slice::from_ref(&ex.question)).expect("decide").pop().expect("one question, one answer")
}

/// Patterns 1-6: zero-shot held-out accuracy, one number per task. No
/// training happens anywhere in this sample - Laya ships pretrained to
/// answer exactly this shape of question, and the point being measured is
/// whether that pretraining generalizes to these seven synthetic tasks, not
/// whether a model CAN be made to fit them given enough steps.
fn run_report(pipe: &mut DecisionPipeline, eval_n: usize, seed: u64) {
    println!("\n--- zero-shot held-out accuracy per task (patterns 1-6) ---");
    for &task in &TASKS {
        let mut rng = Rng::new(seed ^ (task as u64).wrapping_mul(0x2545F4914F6CDD1D));
        let mut correct = 0usize;
        let mut confidence_sum = 0.0f32;
        for _ in 0..eval_n {
            let ex = tasks::generate(task, &mut rng);
            let a = ask(pipe, &ex);
            confidence_sum += answer_confidence(&a);
            if answer_index(&a) == ex.gold {
                correct += 1;
            }
        }
        let acc = correct as f32 / eval_n as f32;
        let mean_conf = confidence_sum / eval_n as f32;
        println!("  {:<10} {:.3}  (mean confidence {:.3}, {eval_n} held-out examples)", task.name(), acc, mean_conf);
    }
}

/// Pattern 8: bulk labeling. The SAME loaded model, batched over a large
/// synthetic table - proves throughput, not a new capability.
fn run_bulk(pipe: &mut DecisionPipeline, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut counts = [0usize; 3];
    let start = std::time::Instant::now();
    for _ in 0..n {
        let ex = tasks::generate(Task::Triage, &mut rng);
        let a = ask(pipe, &ex);
        counts[answer_index(&a)] += 1;
    }
    let elapsed = start.elapsed();
    println!("\n--- pattern 8: bulk labeling ({n} rows, triage task) ---");
    println!("  {:.2} rows/sec ({:.1}s total)", n as f64 / elapsed.as_secs_f64(), elapsed.as_secs_f64());
    println!("  labels: reply now {}, later {}, archive {}", counts[0], counts[1], counts[2]);
}

/// Pattern 7: real-time control. Repeat the control task's `decide()` call
/// and measure per-call latency against the diagram's own 300ms budget.
fn run_realtime(pipe: &mut DecisionPipeline, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut ms: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let ex = tasks::generate(Task::Control, &mut rng);
        let start = std::time::Instant::now();
        let _ = ask(pipe, &ex);
        ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(|a, b| a.total_cmp(b));
    let mean: f64 = ms.iter().sum::<f64>() / ms.len() as f64;
    let p95 = ms[((ms.len() as f64 * 0.95) as usize).min(ms.len() - 1)];
    println!("\n--- pattern 7: real-time control ({n} decisions) ---");
    println!("  mean {mean:.2} ms, p95 {p95:.2} ms, budget 300 ms ({:.2}x headroom at p95)", 300.0 / p95.max(0.001));
}

/// Pattern 9: confidence gate. The model's OWN `confidence` (entropy of its
/// softmax on a `Choice`/`Score`, distance from 0.5 on a `Noul`) drives the
/// band - never a hardcoded application check.
fn run_gate(pipe: &mut DecisionPipeline, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut bands = [0usize; 3]; // act, confirm, human
    let mut example = [None, None, None];
    for _ in 0..n {
        let ex = tasks::generate(Task::ToolGate, &mut rng);
        let a = ask(pipe, &ex);
        let c = answer_confidence(&a);
        let band = if c > 0.9 {
            0
        } else if c > 0.5 {
            1
        } else {
            2
        };
        bands[band] += 1;
        if example[band].is_none() {
            example[band] = Some((ex.state.clone(), c));
        }
    }
    println!("\n--- pattern 9: confidence gate (tool-gate task, thresholds 0.9 / 0.5) ---");
    let names = ["act (>0.9)", "confirm (0.5-0.9)", "human (<0.5)"];
    for band in 0..3 {
        print!("  {:<18} {} of {n}", names[band], bands[band]);
        if let Some((state, c)) = &example[band] {
            print!("   e.g. {state:?} (confidence {c:.3})");
        }
        println!();
    }
}

/// Pattern 5: reranking. One query, several candidates as `Choice` options
/// in a single call, sorted by the model's own calibrated probability - the
/// natural output of [`DecisionPipeline::decide`], no extra machinery.
fn run_rerank(pipe: &mut DecisionPipeline, seed: u64) {
    let mut rng = Rng::new(seed);
    let ex = tasks::generate(Task::Rerank, &mut rng);
    let a = ask(pipe, &ex);
    let Answer::Choice { probabilities, .. } = a else { unreachable!("rerank always asks a Choice question") };
    let brain::decision::Question::Choice { options, .. } = &ex.question else { unreachable!("rerank always builds a Choice") };
    let mut ranked: Vec<(f32, &str)> = probabilities.iter().map(|(name, p)| (*p, name.as_str())).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("\n--- pattern 5: reranking ---");
    println!("  query: {:?}", ex.state);
    for (rank, (p, passage)) in ranked.iter().enumerate() {
        let mark = if *passage == options[ex.gold].name { " <- most relevant" } else { "" };
        println!("  {}. p={p:.3}  {passage:?}{mark}", rank + 1);
    }
}

/// Turn what the caller typed into a checkpoint directory: a path, a
/// `<vendor>/<repo>` id under the model store, or one of [`ALIASES`].
///
/// Rule 5 of samples/README.md: a missing checkpoint is a message naming the
/// command that fetches it and a clean exit, never a panic or a surprise
/// download.
fn locate(name: &str) -> Result<String, String> {
    if let Some(dir) = checkpoint_at(Path::new(name)) {
        return Ok(dir);
    }
    let id = hub_id(name);
    let root = models_root()?;
    if let Some(dir) = checkpoint_at(&root.join(id)) {
        return Ok(dir);
    }
    Err(format!("no decision checkpoint for {name:?}\n  looked at {name} and {}\n  run `brain pull {id}`, or pass --model DIR", root.join(id).display()))
}

fn hub_id(name: &str) -> &str {
    ALIASES.iter().find(|(alias, _)| *alias == name).map(|(_, id)| *id).unwrap_or(name)
}

/// A decision checkpoint is one of two shapes, the same pair of markers
/// `brain::DecisionPipeline`'s own backend sniffer reads: a `crates/decide`
/// encoder has a root `config.json`, a Laya checkpoint has
/// `rl_agent_config.json` and no root `config.json`.
fn checkpoint_at(dir: &Path) -> Option<String> {
    let shaped = dir.join("config.json").is_file() || dir.join("rl_agent_config.json").is_file();
    shaped.then(|| dir.display().to_string())
}

/// Where `brain pull` puts models, in brain's own precedence order.
fn models_root() -> Result<PathBuf, String> {
    if let Some(p) = env_nonempty("BRAIN_MODELS_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Some(x) = env_nonempty("XDG_DATA_HOME") {
        return Ok(Path::new(&x).join("brain").join("models"));
    }
    env_nonempty("HOME").map(|h| Path::new(&h).join(".local").join("share").join("brain").join("models")).ok_or_else(|| "no models directory: set BRAIN_MODELS_DIR, or pass --model DIR".to_string())
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_name_maps_to_what_brain_pull_calls_it() {
        assert_eq!(hub_id("laya"), "convaiinnovations/laya");
        assert_eq!(hub_id("minilm"), "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn an_unknown_name_is_passed_through_unchanged() {
        assert_eq!(hub_id("some-vendor/some-model"), "some-vendor/some-model");
    }

    #[test]
    fn a_missing_checkpoint_names_the_command_that_fetches_it() {
        let err = locate("definitely-not-a-model").unwrap_err();
        assert!(err.contains("brain pull definitely-not-a-model"), "{err}");
        assert!(err.contains("--model DIR"), "{err}");
    }
}
