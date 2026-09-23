// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A model that reads a directory continually and can say what it gained.
//!
//! ```text
//! sample-learning-reader generate --corpus corpus/          # invent a tool
//! sample-learning-reader battery  --model M --run-dir run/  # 2/5, frozen
//! sample-learning-reader read     --corpus corpus/ --run-dir run/
//! sample-learning-reader battery  --model M --run-dir run/  # 5/5
//! sample-learning-reader report   --run-dir run/
//! sample-learning-reader selftest --corpus corpus/ --run-dir run/
//! ```
//!
//! The corpus is a command-line tool that does not exist until `generate`
//! draws it from a seed, so the model's starting score is provably zero
//! rather than argued. The judge is that tool's own grammar parser, never a
//! second model's opinion.
//!
//! `battery` is the point. `read`'s promote rate is plumbing; the change in
//! the battery between two calls is the ability the user gained.
//!
//! Swedish Embedded AB builds continual-learning systems whose claims are
//! checkable by something other than the system making them. If your team
//! needs a model that keeps learning from your own material and can show
//! what that bought and what it cost, you can procure our services by
//! sending an email to info@swedishembedded.com.

mod corpus;

use std::path::PathBuf;
use std::process::ExitCode;

use brain::{BatteryTask, ContinualReader};
use corpus::{Expect, Tool};

const USAGE: &str = "\
usage: sample-learning-reader <verb> [options]

  generate --corpus DIR [--seed N]
        invent two tools and write the corpus, its adversarial lanes and
        their labels. Nothing else needs preparing.

  battery  --model REF --run-dir DIR [--corpus DIR] [--seed N]
        score the held-out tasks against what is currently served. Run it
        before reading and again after: the difference is the result.

  read     --corpus DIR --run-dir DIR --model REF [--until N] [--seed N]
        read the corpus, deciding per episode what is worth learning.
        Resumes if the run directory has been read before.

  report   --run-dir DIR
        every episode the run recorded, and where each one stopped.

  selftest --corpus DIR --run-dir DIR
        check each episode's verdict against its lane's label, and exit
        non-zero if the reader learned something it should have refused.

The model is named by --model and resolved by brain's own model handler, so
a reference that is not on disk is fetched. Without one, `generate`,
`report` and `selftest` still work.
";

/// A tiny flag reader. A sample parses its own arguments rather than sharing
/// the engine's parser: it is a standalone application whose only brain
/// dependency is the SDK.
struct Args(Vec<String>);

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == flag)?;
        if i + 1 >= self.0.len() {
            return None;
        }
        self.0.remove(i);
        Some(self.0.remove(i))
    }
    fn path(&mut self, flag: &str, default: &str) -> PathBuf {
        PathBuf::from(self.take(flag).unwrap_or_else(|| default.to_string()))
    }
    fn number(&mut self, flag: &str, default: u64) -> Result<u64, String> {
        match self.take(flag) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|e| format!("{flag}: {e}")),
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(argv) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("sample-learning-reader: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(argv: Vec<String>) -> Result<ExitCode, String> {
    let mut a = Args(argv);
    let verb = a.0.remove(0);
    match verb.as_str() {
        "generate" => generate(&mut a),
        "battery" => battery(&mut a),
        "read" => read(&mut a),
        "report" => report(&mut a),
        "selftest" => selftest(&mut a),
        other => Err(format!("unknown verb {other:?}\n\n{USAGE}")),
    }
}

/// A model asked for a command may keep going afterwards; the invocation is
/// the first line of what it wrote.
fn first_line(answer: &str) -> String {
    answer.lines().next().unwrap_or_default().trim().to_string()
}

/// The two tools a run uses. Derived from one seed so `generate`, `battery`
/// and `selftest` agree about which tools they are talking about without
/// having to pass them between processes.
fn tools(seed: u64) -> (Tool, Tool) {
    let a = Tool::generate(seed);
    let b = Tool::generate_disjoint(seed.wrapping_add(1), &[&a]);
    (a, b)
}

fn generate(a: &mut Args) -> Result<ExitCode, String> {
    let dir = a.path("--corpus", "corpus");
    let seed = a.number("--seed", 1)?;
    let (ta, tb) = tools(seed);
    let labels = corpus::write(&dir, &ta, &tb).map_err(|e| format!("{}: {e}", dir.display()))?;
    println!("invented {} and {}, {} subcommands between them", ta.name, tb.name, ta.commands.len() + tb.commands.len());
    println!("wrote {} episodes across {} lanes to {}", labels.len(), 9, dir.display());
    println!("battery: {} held-out tasks, none of which appear in the corpus", ta.battery().len());
    Ok(ExitCode::SUCCESS)
}

fn battery_tasks(seed: u64) -> Vec<BatteryTask> {
    let (ta, _) = tools(seed);
    ta.battery().into_iter().map(|(prompt, expected)| BatteryTask { prompt, expected }).collect()
}

fn battery(a: &mut Args) -> Result<ExitCode, String> {
    let model = a.take("--model").ok_or("battery needs --model")?;
    let run_dir = a.path("--run-dir", "run");
    let seed = a.number("--seed", 1)?;
    let tasks = battery_tasks(seed);

    let reader = ContinualReader::from_pretrained(&model).run_dir(&run_dir).seed(seed);
    let score = reader.battery(&tasks).map_err(|e| e.to_string())?;
    let (ta, _) = tools(seed);

    // Two numbers, because they answer different questions. Exact match is
    // "did it write the canonical answer"; acceptance is "would the tool
    // have run it", which is the ability a user actually gains and is
    // strictly the looser of the two. Reporting only the first would call a
    // correct invocation with the flags in another order a failure.
    let accepted = score.answers.iter().filter(|a| ta.parse(&first_line(a)).is_ok()).count();
    println!("capability {}/{} exact, {}/{} accepted by {}", score.passed, score.total, accepted, score.total, ta.name);
    for ((t, ok), answer) in tasks.iter().zip(&score.per_task).zip(&score.answers) {
        let got = first_line(answer);
        let verdict = match (ok, ta.parse(&got)) {
            (true, _) => "pass".to_string(),
            (false, Ok(())) => "valid, not canonical".to_string(),
            (false, Err(why)) => format!("{why:?}"),
        };
        println!("  {verdict:<28} want {:<40} got {got:?}", t.expected);
    }
    Ok(ExitCode::SUCCESS)
}

fn read(a: &mut Args) -> Result<ExitCode, String> {
    let model = a.take("--model").ok_or("read needs --model")?;
    let corpus_dir = a.path("--corpus", "corpus");
    let run_dir = a.path("--run-dir", "run");
    let seed = a.number("--seed", 1)?;
    let until = a.take("--until").map(|v| v.parse::<usize>().map_err(|e| format!("--until: {e}"))).transpose()?;

    let mut reader = ContinualReader::from_pretrained(&model).run_dir(&run_dir).corpus(&corpus_dir).seed(seed);
    if let Some(n) = until {
        reader = reader.until(n);
    }
    let out = reader.read().map_err(|e| e.to_string())?;

    println!("read {} episodes, promoted {}", out.episodes, out.promoted);
    println!("bank {} earlier episodes, detection latency {} episodes", out.bank, out.detection_latency);
    println!("what that means: any regression larger than the per-block bar is found within {} episodes.", out.detection_latency);
    Ok(ExitCode::SUCCESS)
}

fn report(a: &mut Args) -> Result<ExitCode, String> {
    let run_dir = a.path("--run-dir", "run");
    let rows = ContinualReader::from_pretrained("unused").run_dir(&run_dir).ledger().map_err(|e| e.to_string())?;
    if rows.is_empty() {
        println!("nothing read yet");
        return Ok(ExitCode::SUCCESS);
    }
    for r in &rows {
        let cause = r.cause.clone().unwrap_or_default();
        println!("{:>5}  {:<28} {:<8} {:<18} audited {}", r.episode, r.source, r.stage, cause, r.audited);
    }
    let promoted = rows.iter().filter(|r| r.promoted).count();
    println!("\n{} episodes, {promoted} promoted", rows.len());
    for stage in ["screen", "reach", "ingest", "evidence", "gate"] {
        let n = rows.iter().filter(|r| r.stage == stage).count();
        if n > 0 {
            println!("  stopped at {stage}: {n}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// The verb that makes the failure-mode half checkable: every episode's
/// verdict against the label of the lane it came from.
fn selftest(a: &mut Args) -> Result<ExitCode, String> {
    let corpus_dir = a.path("--corpus", "corpus");
    let run_dir = a.path("--run-dir", "run");
    let seed = a.number("--seed", 1)?;
    let (ta, tb) = tools(seed);
    let labels = corpus::write(&corpus_dir, &ta, &tb).map_err(|e| format!("{}: {e}", corpus_dir.display()))?;

    let rows = ContinualReader::from_pretrained("unused").run_dir(&run_dir).ledger().map_err(|e| e.to_string())?;
    if rows.is_empty() {
        return Err("nothing has been read yet, so there are no verdicts to check".to_string());
    }

    let mut failures = 0usize;
    for row in &rows {
        let Some((_, expect)) = labels.iter().find(|(rel, _)| row.source == **rel) else {
            continue;
        };
        let ok = match expect {
            Expect::Promote => row.promoted,
            Expect::Refuse => !row.promoted,
            Expect::Either => true,
        };
        if !ok {
            failures += 1;
            let want = if *expect == Expect::Promote { "should have been learned" } else { "should have been refused" };
            println!("FAIL {:<28} {want}, stopped at {} ({})", row.source, row.stage, row.cause.clone().unwrap_or_default());
        }
    }

    let checked = rows.iter().filter(|r| labels.contains_key(&r.source)).count();
    if failures == 0 {
        println!("selftest: {checked} labelled episodes, every verdict as expected");
        Ok(ExitCode::SUCCESS)
    } else {
        println!("selftest: {failures} of {checked} labelled episodes went the wrong way");
        Ok(ExitCode::FAILURE)
    }
}
