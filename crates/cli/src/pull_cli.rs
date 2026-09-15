// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain pull <model>` - fetch a model's official weights into the model
//! store, out loud.
//!
//! The verb is a front door, not a second fetcher: it parses what the user
//! typed ([`brain_modelstore::refurl`]), resolves the store directory through
//! the one resolver every other surface uses ([`loader::model_dir::resolve`]),
//! and then drives exactly the plan/execute/finish sequence the auto-fetch
//! path already runs ([`loader::supply::execute_plan`]). What `brain
//! pull` adds is the reporting, and one deliberate choice about where it
//! goes.
//!
//! # Two progress modes, chosen by the stream they are written to
//!
//! Progress is written to **stdout**, and the mode is chosen by whether
//! stdout is a terminal. Deciding on one stream and writing to another is how
//! you end up redrawing an ANSI bar into a log file, so the decision and the
//! destination are the same stream by construction. `brain pull` produces no
//! other machine-readable stdout, and a user who pipes it is capturing the
//! progress log itself - that is what the sparse mode is FOR. Diagnostics and
//! errors still go to stderr. (This differs on purpose from `wan`/`ltxv`,
//! whose progress goes to stderr because their stdout carries a report.)
//!
//! * **Terminal**: an apt-style bar redrawn in place with `\r` - fraction
//!   complete, bytes, throughput and ETA, never scrolling.
//! * **Pipe**: ten plain lines for the WHOLE pull. The budget is spent over
//!   the total bytes of every file in the plan, so a six-shard model costs
//!   ten lines, not sixty. Each line is one greppable fact with no carriage
//!   returns and no escapes. A header line and a completion line bracket
//!   those ten.
//!
//! The rendering itself ([`Mode`]/[`Reporter`]/the `human_*` helpers) lives in
//! `loader::progress` now - `brain pull` and the default-checkpoint
//! auto-fetch path both draw from it (that crate's own module doc puts it
//! plainly: two code paths that render a fetch's progress would be two sets
//! of bugs, same as two that fetch), and any embedder wants the same
//! rendering for its own fetches, not just this CLI's `pull` verb.
//!
//! Swedish Embedded AB implements model distribution and weight-management
//! tooling for its clients. If your team needs expertise in shipping large
//! model artifacts to edge fleets then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::io::IsTerminal;

use loader::progress::{human_bytes, human_secs, Mode, Reporter};
use brain_modelstore::refurl::parse_pull_arg;
use brain_modelstore::{HfHub, Step, Store};

const USAGE: &str = "\
usage: brain pull <model> [--brain-data-dir DIR]

Fetch a model's official weights into brain's model store and make them
servable. <model> is the canonical reference or a HuggingFace URL - the repo
page, a branch view, or one file's page:

  brain pull Qwen/Qwen3-0.6B
  brain pull https://huggingface.co/Qwen/Qwen3-0.6B
  brain pull https://huggingface.co/Qwen/Qwen3-0.6B/tree/main

A file URL pulls exactly that ONE file, whatever its extension, from whatever
revision the URL names - nothing is inferred, because the file is named:

  brain pull https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/flux-2-klein-9b-Q8_0.gguf

A GGUF repo publishes many quantizations of one model, and only ONE is ever
fetched. Name it with the reference grammar's own quantization suffix, or
name none and let brain pick the highest-fidelity one the repo offers (which
it prints):

  brain pull unsloth/FLUX.2-klein-9B-GGUF-Q4_K_M
  brain pull unsloth/FLUX.2-klein-9B-GGUF

Progress goes to stdout: an in-place bar with throughput and ETA on a
terminal, ten plain lines for the whole pull when piped.

Re-running a pull is cheap for what already landed: a file already complete in
the store is not fetched again. Resume is per FILE, not per byte - a transfer
interrupted part-way through a file restarts that file from the beginning. For
a single-file GGUF that is the whole transfer.

--brain-data-dir DIR   brain's data root; models land in <DIR>/models.
                       Default ~/.local/share/brain. This is a GLOBAL option
                       (valid on any subcommand) and outranks BRAIN_MODELS_DIR.
";

/// `brain pull <model>`. Returns the process exit code.
pub fn run_pull(args: &[String]) -> i32 {
    let mut model: Option<&str> = None;
    for a in args {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return 0;
            }
            flag if flag.starts_with('-') => {
                eprintln!("brain pull: unknown flag {flag:?}\n{USAGE}");
                return 2;
            }
            positional if model.is_none() => model = Some(positional),
            extra => {
                eprintln!("brain pull: unexpected extra argument {extra:?} -- pull takes one model\n{USAGE}");
                return 2;
            }
        }
    }
    let Some(model) = model else {
        eprint!("{USAGE}");
        return 2;
    };

    let target = match parse_pull_arg(model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain pull: {e}");
            return 2;
        }
    };
    let reference = target.reference.clone();
    let Some(root) = loader::model_dir::resolve(None) else {
        eprintln!("brain pull: no models directory (pass --brain-data-dir, or set BRAIN_MODELS_DIR or HOME)");
        return 2;
    };
    let store = Store::new(root);
    let hub = HfHub::new();

    // One argument, two plans: a URL that named a file asks for exactly that
    // artifact, anything else asks for the repo. Both honour the revision the
    // argument named.
    let built = match target.artifact.as_deref() {
        Some(file) => brain_modelstore::plan_file(&reference, file, target.revision.as_deref(), &hub),
        None => brain_modelstore::plan_at(&reference, target.revision.as_deref(), &store, &hub),
    };
    let plan = match built {
        Ok(p) => p,
        Err(e) => {
            eprintln!("brain pull: {e}");
            return 1;
        }
    };
    let dir = store.repo_dir(&reference.base());
    if plan.steps == [Step::Serve] {
        println!("brain pull {reference}: already complete in {}", dir.display());
        return 0;
    }
    // A choice made for the user is a choice said out loud. The planner puts
    // what it resolved on the plan's reference, so this fires exactly when
    // the repo offered several interchangeable artifacts and the argument
    // named none of them.
    if reference.quant().is_none() && target.artifact.is_none() {
        if let Some(q) = plan.reference.quant() {
            println!("brain pull {reference}: no quantization named, selected {q} (the highest-fidelity one this repo offers)");
            println!("brain pull {reference}: pull another as {reference}-<QUANT>, or paste a file's URL to name it exactly");
        }
    }
    let remaining = match brain_modelstore::remaining_download(&store, &hub, &plan) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain pull: {reference}: {e}");
            return 1;
        }
    };

    let mode = Mode::of(std::io::stdout().is_terminal());
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let label = plan.reference.to_string();
    let mut reporter = Reporter::new(mode, &mut lock, label.clone(), remaining);
    if remaining.files > 0 {
        reporter.header();
    }
    let outcome = loader::supply::execute_plan_opt(&store, &hub, &plan, &label, &mut |name, got, total| reporter.on_bytes(name, got, total));
    let (moved, secs) = reporter.finish();
    drop(reporter);

    match outcome {
        Ok(local) => {
            // Where it landed. A pull that produced exactly ONE file still
            // there afterwards reports that FILE: it is the path a
            // `--dit`/`--text-encoder` flag gets pointed at, and naming the
            // directory instead would send the user looking. A pull whose
            // artifact was rewritten or deleted by its finish step (the yolo
            // `.pt`), or that produced several files, reports the servable
            // model's directory as before.
            let where_ = match (landed_file(&plan, &dir), local) {
                (Some(f), _) => f,
                (None, Some(l)) => l.dir.display().to_string(),
                (None, None) => dir.display().to_string(),
            };
            println!("brain pull {label}: fetched {} in {} -> {where_}", human_bytes(moved), human_secs(secs));
            0
        }
        Err(e) => {
            eprintln!("brain pull: {e}");
            1
        }
    }
}

/// The path of the single file a plan produced, when it produced exactly one
/// AND that file is still there -- the artifact a file URL named, or the one
/// quantization a GGUF repo resolved to. `None` for a multi-file pull, and
/// for a finish step that consumed its download (`convert_yolo` deletes the
/// `.pt` it rewrote), so this never names a path that is not there.
fn landed_file(plan: &brain_modelstore::Plan, dir: &std::path::Path) -> Option<String> {
    let mut downloads = plan.steps.iter().filter_map(|s| match s {
        Step::Download { dest_name, .. } => Some(dest_name),
        _ => None,
    });
    let dest = match (downloads.next(), downloads.next()) {
        (Some(dest), None) => dir.join(dest),
        _ => return None,
    };
    dest.is_file().then(|| dest.display().to_string())
}
