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

use brain::{BatteryTask, ContinualReader, LedgerFacts};
use corpus::{Expect, Tool};

const USAGE: &str = "\
usage: sample-learning-reader <verb> [options]

  generate --corpus DIR [--seed N]
        invent two tools and write the corpus, its adversarial lanes and
        their labels. Nothing else needs preparing.

  battery  --model REF --run-dir DIR [--seed N]
        score the held-out tasks against what is currently served. Run it
        before reading and again after: the difference is the result.

  read     --corpus DIR --run-dir DIR --model REF [--until N] [--seed N]
           [--null-gate N | --shuffled-labels N]
        read the corpus, deciding per episode what is worth learning.
        Resumes if the run directory has been read before.

        --null-gate N        control arm: a coin with seed N decides what
                             carries forward. The real gate still runs and
                             the ledger still records it, so this arm's
                             promote rate is comparable with the real one's.
        --shuffled-labels N  control arm: each episode is trained on its own
                             rows and gated against ANOTHER episode's frozen
                             probes. It should promote at chance.

        Run a control into its OWN run directory: it is a different arm, not
        a continuation of the real one.

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
    /// Refuse anything the verb did not understand. Every flag this sample
    /// has selects which run, which seed or which corpus a number is about,
    /// so one that fell through to a default would report a believable
    /// number about a different run than the one asked for.
    fn finish(&self) -> Result<(), String> {
        if self.0.is_empty() {
            return Ok(());
        }
        Err(format!("not understood: {}\n\n{USAGE}", self.0.join(" ")))
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
    a.finish()?;
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
    a.finish()?;
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
    let null_gate = a.take("--null-gate").map(|v| v.parse::<u64>().map_err(|e| format!("--null-gate: {e}"))).transpose()?;
    let shuffled = a.take("--shuffled-labels").map(|v| v.parse::<u64>().map_err(|e| format!("--shuffled-labels: {e}"))).transpose()?;
    a.finish()?;
    // One arm at a time. Two controls at once answers neither question, and
    // silently honouring the first would report the run as the other.
    if null_gate.is_some() && shuffled.is_some() {
        return Err("--null-gate and --shuffled-labels are different control arms; run one at a time, into its own run directory".to_string());
    }

    let mut reader = ContinualReader::from_pretrained(&model).run_dir(&run_dir).corpus(&corpus_dir).seed(seed);
    if let Some(n) = until {
        reader = reader.until(n);
    }
    if let Some(s) = null_gate {
        reader = reader.null_gate(s);
    }
    if let Some(s) = shuffled {
        reader = reader.shuffled_labels(s);
    }
    let arm = match (null_gate, shuffled) {
        (Some(s), _) => format!("null-gate arm (coin seed {s})"),
        (_, Some(s)) => format!("shuffled-labels arm (seed {s})"),
        _ => "real arm".to_string(),
    };
    println!("{arm}");
    let out = reader.read().map_err(|e| e.to_string())?;

    println!("read {} episodes, promoted {}", out.episodes, out.promoted);
    println!("bank {} earlier episodes, detection latency {} episodes", out.bank, out.detection_latency);
    println!("what that means: any regression larger than the per-block bar is found within {} episodes.", out.detection_latency);

    let (learned, revisited) = out.retention_coverage;
    match out.bwt {
        Some(bwt) => {
            let worst = out.worst_block_drop.unwrap_or(0.0);
            println!("bwt {bwt:+.4} over {revisited} of {learned} learned episodes revisited, worst single drop {worst:.4}");
        }
        // Said rather than left out. A run whose schedule never came back
        // round has no backward transfer, and printing nothing there is how
        // an unanswered clause gets read as a passed one.
        None => println!("bwt UNMEASURED: {learned} episodes learned, none revisited yet - not a backward transfer of zero"),
    }
    Ok(ExitCode::SUCCESS)
}

fn report(a: &mut Args) -> Result<ExitCode, String> {
    let run_dir = a.path("--run-dir", "run");
    a.finish()?;
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
    a.finish()?;
    // The labels of the corpus that was READ, taken from the corpus itself.
    // Regenerating them from a seed would overwrite the documents the run is
    // about, and under a seed other than the one it was generated with it
    // would replace them with a different tool's while still printing a
    // verdict about this run.
    let labels = corpus::labels(&corpus_dir).map_err(|e| format!("{}: {e}", corpus_dir.display()))?;

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
    } else {
        println!("selftest: {failures} of {checked} labelled episodes went the wrong way");
    }

    // The acceptance block, as far as this run can answer it. Printing the
    // unanswerable clauses is the point rather than an omission: a reader
    // that has not been run against a real model cannot have a battery
    // delta or a null-gate arm, and a report that quietly left those out
    // would look like a run that passed them.
    let facts = LedgerFacts::of(&rows);
    println!(
        "\nrun record: {} episodes, {} promoted, {} refused ({} explained), {} audit decodes",
        facts.episodes, facts.promoted, facts.rejections, facts.rejections_with_cause, facts.eval_decodes
    );
    let unexplained = LedgerFacts::unexplained(&rows);
    if unexplained.is_empty() {
        println!("  clause 1 (every refusal names its cause): PASS");
    } else {
        println!("  clause 1 (every refusal names its cause): FAIL, unexplained: {}", unexplained.join(", "));
    }
    println!("  clause 6 (control arms)        : needs a null-gate arm and a second seed");
    println!("  clause 4 (independent battery) : needs `battery` before and after a read");
    println!("  clause 3 (bwt, per-block bar)  : needs the retention matrix of a real run");
    println!("\nthose three are UNANSWERED, not passed: this run has not produced the numbers they are about.");

    if failures == 0 && unexplained.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every number this sample reports is about a particular seed and a
    /// particular run directory. A misspelled flag that fell through to a
    /// default would produce a plausible number about the wrong run, which
    /// is worse than no number at all.
    #[test]
    fn a_misspelled_flag_is_refused_rather_than_silently_defaulted() {
        let dir = std::env::temp_dir().join(format!("sample-reader-unknown-flag-{}", std::process::id()));
        let argv = vec!["generate".to_string(), "--corpus".to_string(), dir.to_string_lossy().into_owned(), "--seeed".to_string(), "2".to_string()];
        let err = run(argv).expect_err("an unknown flag must not be ignored");
        assert!(err.contains("--seeed"), "the message must name what was not understood: {err}");
        assert!(!dir.exists(), "a refused invocation must not have written anything");
    }

    /// The stay-silent half: the flags a verb documents are accepted.
    #[test]
    fn the_documented_flags_of_a_verb_are_accepted() {
        let dir = std::env::temp_dir().join(format!("sample-reader-args-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let argv = vec!["generate".to_string(), "--corpus".to_string(), dir.to_string_lossy().into_owned(), "--seed".to_string(), "2".to_string()];
        run(argv).expect("the documented flags are accepted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
