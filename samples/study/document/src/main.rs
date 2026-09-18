// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Teach a model a batch of documents, and gate whether it learned them.
//!
//! Everything this program does beyond parsing a command line is one call to
//! [`brain::DocumentStudy`]. That is the point of it being a sample: the
//! study - training a LoRA on frozen fact triples, scoring it against a
//! pre-registered bar, running a null-gate control arm beside it, and
//! publishing the adapter only on a promote - is a library capability, not a
//! verb belonging to the engine's own command line.
//!
//! ```text
//! sample-study-document --weights Qwen/Qwen3-0.6B --dataset facts.json \
//!                       --adapter-dir adapters/ --report report.json
//! ```
//!
//! Swedish Embedded AB implements continual-learning pipelines whose
//! promotion decisions are gated by pre-registered criteria and their own
//! controls. If your team needs a model that can be taught something new
//! without silently forgetting what it knew, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::PathBuf;
use std::process::ExitCode;

use brain::DocumentStudy;

const USAGE: &str = "\
usage: sample-study-document --weights BASE --dataset FILE.json --adapter-dir DIR
                             [--report FILE.json] [--arch NAME] [--work-dir DIR]
                             [--models-dir DIR] [--lora RANK] [--alpha A]
                             [--steps N] [--seqs N] [--batch B] [--lr X]
                             [--eval-per-cycle N] [--seed S] [--null-gate-seed S]
                             [--quiet] [--dry-run]

  --weights BASE        checkpoint path, model directory, or vendor/repo ref
  --dataset FILE.json   frozen {fact, probe_question, expected_answer} batches
  --adapter-dir DIR     where a PROMOTED adapter is published
  --report FILE.json    machine-readable verdict (optional)
  --dry-run             validate the dataset and exit: no weights, no device
";

/// A tiny flag reader. A sample parses its own arguments rather than sharing
/// the engine's parser - it is a standalone application, and its only brain
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
    fn flag(&mut self, flag: &str) -> bool {
        match self.0.iter().position(|a| a == flag) {
            Some(i) => {
                self.0.remove(i);
                true
            }
            None => false,
        }
    }
    fn parse<T: std::str::FromStr>(&mut self, flag: &str) -> Option<T> {
        self.take(flag).and_then(|v| v.parse().ok())
    }
}

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.flag("--help") || a.flag("-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let dataset = match a.take("--dataset") {
        Some(d) => PathBuf::from(d),
        None => {
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    // The dataset is checkable entirely on its own, so a caller that did not
    // produce it can find out it is malformed before paying for a GPU-bound
    // run rather than after.
    if a.flag("--dry-run") {
        return match DocumentStudy::validate_dataset(&dataset) {
            Ok(s) => {
                println!("dataset OK: {} cycle(s), {} triple(s), {} anchor(s)", s.cycles, s.triples, s.anchors);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }

    let (Some(weights), Some(adapter_dir)) = (a.take("--weights"), a.take("--adapter-dir")) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };

    let mut study = match DocumentStudy::from_pretrained(weights) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    study = study.dataset(&dataset).adapter_dir(PathBuf::from(adapter_dir));

    if let Some(v) = a.take("--arch") {
        study = study.arch(v);
    }
    if let Some(v) = a.take("--report") {
        study = study.report(PathBuf::from(v));
    }
    if let Some(v) = a.take("--work-dir") {
        study = study.work_dir(PathBuf::from(v));
    }
    if let Some(v) = a.take("--models-dir") {
        study = study.models_dir(v);
    }
    if let Some(v) = a.parse("--lora") {
        study = study.lora(v);
    }
    if let Some(v) = a.parse("--alpha") {
        study = study.alpha(v);
    }
    if let Some(v) = a.parse("--steps") {
        study = study.steps(v);
    }
    if let Some(v) = a.parse("--eval-per-cycle") {
        study = study.eval_per_cycle(v);
    }
    let (seqs, batch, lr) = (a.parse("--seqs"), a.parse("--batch"), a.parse("--lr"));
    if seqs.is_some() || batch.is_some() || lr.is_some() {
        study = study.sft(seqs.unwrap_or(8), batch.unwrap_or(1), lr.unwrap_or(1e-4));
    }
    let pinned_seed = a.parse("--seed");
    if let Some(v) = pinned_seed {
        study = study.seed(v);
    }
    let pinned_null = a.parse("--null-gate-seed");
    if let Some(v) = pinned_null {
        study = study.null_gate_seed(v);
    }
    if a.flag("--quiet") {
        study = study.quiet(true);
    }

    // A study whose seed nobody chose is not the same study twice, so say
    // which one this run drew - BEFORE the run, so an interrupted one is
    // still reproducible.
    let (seed, null_gate_seed) = study.seeds();
    if pinned_seed.is_none() {
        eprintln!("document study: no --seed given, using random seed {seed} (pass --seed {seed} to reproduce)");
    }
    if pinned_null.is_none() {
        eprintln!("document study: no --null-gate-seed given, using random seed {null_gate_seed} (pass --null-gate-seed {null_gate_seed} to reproduce)");
    }

    if !a.0.is_empty() {
        eprintln!("unrecognised arguments: {}", a.0.join(" "));
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }

    let outcome = match study.run() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!("{}", outcome.table());
    println!("{}", outcome.summary());
    match outcome.published() {
        Some(p) => println!("promoted: published {}", p.display()),
        None => println!("rejected: no adapter published (the report says which check failed)"),
    }
    ExitCode::SUCCESS
}
