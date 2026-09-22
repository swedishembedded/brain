// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A decision endpoint on a pipe: JEV-style JSON in, JSON out.
//!
//! ```text
//! echo '{"state": "my card never arrived",
//!        "questions": {"intent": {"type": "choice", "instructions": "what is this about",
//!                                 "criteria": {"card": "cards", "fx": "exchange rates"}}}}' \
//!   | json | jq .answers.intent.choice
//! ```
//!
//! The other decision samples each answer ONE question their own code spells
//! out (`triage` picks an intent, `salesagent` scores a conversation). This
//! one spells out nothing: the state, the questions, their types and their
//! options all arrive at run time, as JSON, and go back as JSON. That is the
//! whole claim of a decision model made operational - the output space lives
//! in the REQUEST, so a service can ask a question nobody had written down
//! when the model was loaded, and get a calibrated distribution rather than
//! prose to parse.
//!
//! Which is also why this sample is a pipe rather than an HTTP server: the
//! interesting part is the typed request/response contract, and every shell,
//! `jq` filter, cron job and language on the machine already speaks stdin.
//!
//! What it needs: a decision checkpoint. `--model laya` (the default) is
//! `convaiinnovations/laya`, a pretrained System-1 decision model that
//! answers all three question types zero-shot; `--model minilm` is a
//! `crates/decide`-shaped encoder, which answers the same requests but needs
//! a trained head to answer them WELL (see `samples/decision/triage`).
//!
//! Swedish Embedded AB implements typed decision endpoints - a model behind a
//! schema a service can rely on, rather than free text a caller has to parse
//! and hope about - for its clients. If your team needs one, you can procure
//! our services by sending an email to info@swedishembedded.com.

mod jev;

use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

use brain::options::{Args, Hardware, ModelChoice, Options};
use brain::{DecisionPipeline, Device};

/// The short names this sample knows, mapped to what `brain pull` calls them.
/// A name that is not here is passed through, so a `<vendor>/<repo>` id or a
/// directory works without this table knowing about it.
const ALIASES: &[(&str, &str)] = &[
    ("laya", "convaiinnovations/laya"),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("decide", "sentence-transformers/all-MiniLM-L6-v2"),
];

struct Settings {
    model: ModelChoice,
    /// Trained head weights to answer with. A `crates/decide` encoder is only
    /// half a decision model - the head is the other half, and a fresh one
    /// answers uniformly (see this sample's README). Laya ships its own head
    /// inside the checkpoint and needs none.
    head: Option<String>,
    jsonl: bool,
    hardware: Hardware,
}

fn usage() -> String {
    format!(
        "\
usage: json [options] < request.json

Reads a JEV-style decision request as JSON on stdin and writes the response as
JSON on stdout - one object per request, whether it succeeded or not, so the
stream stays parseable:

  {{\"state\": ..., \"questions\": {{\"<name>\": {{\"type\": \"choice\"|\"score\"|\"noul\",
     \"instructions\": ..., \"criteria\": ...}}}}}}

model
{}
  --head FILE         trained head weights, for an encoder that ships without
                      one (`samples/decision/triage` writes one; a Laya
                      checkpoint carries its own and needs no --head)
  --jsonl             one request per input LINE, one response per output line
                      (the model is loaded once and answers them all)

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
    let jsonl = args.take_flag("--jsonl");
    args.finish();
    Ok(Settings { model, head, jsonl, hardware })
}

fn main() {
    match run() {
        Ok(true) => {}
        // A request failed. Its error is already on stdout as a document; the
        // exit status is for the script that ran this, not for a human.
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("json: {e}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<bool, String> {
    let s = parse()?;
    let dir = locate(&s.model.name)?;
    s.hardware.apply()?;
    // Progress and provenance go to stderr, so stdout carries nothing but
    // responses and can be piped straight into `jq`.
    eprintln!("json: {} from {} on {}", s.model.name, dir, s.hardware.describe());

    let mut builder = DecisionPipeline::builder(&dir).device(Device::default());
    if let Some(head) = &s.head {
        if !Path::new(head).is_file() {
            return Err(format!("no head weights at {head} - train some with samples/decision/triage"));
        }
        builder = builder.head(head);
    }
    let mut pipe = builder.load().map_err(|e| format!("loading {dir}: {e}"))?;

    let mut ok = true;
    if s.jsonl {
        // Line by line as they ARRIVE, not after end of input: a caller that
        // keeps the pipe open (a queue, a `tail -f`, an interactive session)
        // gets each answer when it asks, and reading to the end first would
        // hang forever on exactly that caller.
        for line in std::io::stdin().lock().lines() {
            let line = line.map_err(|e| format!("reading stdin: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            ok &= answer(&mut pipe, &s.model.name, &line);
        }
    } else {
        // One request, which may be pretty-printed across many lines, so this
        // one does wait for end of input.
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input).map_err(|e| format!("reading stdin: {e}"))?;
        if input.trim().is_empty() {
            return Err("no request on stdin - pipe one in, or run with --help".into());
        }
        ok = answer(&mut pipe, &s.model.name, &input);
    }
    Ok(ok)
}

/// Answer one request, writing exactly one JSON document to stdout either
/// way. Returns whether it succeeded.
fn answer(pipe: &mut DecisionPipeline, model: &str, request: &str) -> bool {
    let (out, ok) = match run_request(pipe, model, request) {
        Ok(response) => (response, true),
        Err(e) => (jev::render_error(model, &e), false),
    };
    println!("{out}");
    // A pipe consumer may be reading line by line and acting on each answer;
    // waiting for the buffer to fill would stall it behind the next request.
    std::io::stdout().flush().ok();
    ok
}

fn run_request(pipe: &mut DecisionPipeline, model: &str, request: &str) -> Result<String, String> {
    let req = jev::parse_request(request)?;
    let questions: Vec<_> = req.questions.iter().map(|(_, q)| q.clone()).collect();
    let names: Vec<String> = req.questions.iter().map(|(n, _)| n.clone()).collect();
    let answers = pipe.decide(&req.state, &questions).map_err(|e| format!("{e}"))?;
    Ok(jev::render_response(model, &names, &answers))
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
    Err(format!(
        "no decision checkpoint for {name:?}\n  \
         looked at {name} and {}\n  \
         run `brain pull {id}`, or pass --model DIR",
        root.join(id).display()
    ))
}

fn hub_id(name: &str) -> &str {
    ALIASES.iter().find(|(alias, _)| *alias == name).map(|(_, id)| *id).unwrap_or(name)
}

/// A decision checkpoint is one of two shapes, and this is the same pair of
/// markers `brain::DecisionPipeline`'s own backend sniffer reads: a
/// `crates/decide` encoder has a root `config.json`, a Laya checkpoint has
/// `rl_agent_config.json` and no root `config.json`. A cheap existence probe,
/// not a second dispatch - the SDK still decides which arm actually runs.
fn checkpoint_at(dir: &Path) -> Option<String> {
    let shaped = dir.join("config.json").is_file() || dir.join("rl_agent_config.json").is_file();
    shaped.then(|| dir.display().to_string())
}

/// Where `brain pull` puts models, in brain's own precedence order
/// (`BRAIN_MODELS_DIR`, then `$XDG_DATA_HOME/brain/models`, then
/// `$HOME/.local/share/brain/models`). Read here rather than through the SDK
/// because the decision surface deliberately has no model-store resolver -
/// see `brain::DecisionPipeline`'s own module doc on why it takes a path.
fn models_root() -> Result<PathBuf, String> {
    if let Some(p) = env_nonempty("BRAIN_MODELS_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Some(x) = env_nonempty("XDG_DATA_HOME") {
        return Ok(Path::new(&x).join("brain").join("models"));
    }
    env_nonempty("HOME")
        .map(|h| Path::new(&h).join(".local").join("share").join("brain").join("models"))
        .ok_or_else(|| "no models directory: set BRAIN_MODELS_DIR, or pass --model DIR".to_string())
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

    /// An id this table has never heard of is passed through, so the sample
    /// does not have to be edited to run a checkpoint it does not know.
    #[test]
    fn an_unknown_name_is_passed_through_unchanged() {
        assert_eq!(hub_id("some-vendor/some-model"), "some-vendor/some-model");
        assert_eq!(hub_id("/srv/checkpoints/mine"), "/srv/checkpoints/mine");
    }

    /// The missing-checkpoint message has to carry the fix, since the caller
    /// is looking at a pipe rather than at this code.
    #[test]
    fn a_missing_checkpoint_names_the_command_that_fetches_it() {
        let err = locate("definitely-not-a-model").unwrap_err();
        assert!(err.contains("brain pull definitely-not-a-model"), "{err}");
        assert!(err.contains("--model DIR"), "{err}");
    }
}
