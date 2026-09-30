// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fine-tune a chat model on a product's own conversations, measure whether
//! it helped, and chat with the result - through the public brain SDK.
//!
//! ```text
//! sample-study-chat [--base Qwen/Qwen3-0.6B] [--out DIR] [--steps N]
//!                   [--cancel-after N] [--lr X] [--device SPEC] [--models-dir DIR]
//! ```
//!
//! Three fine-tunes of one base on one tiny dataset:
//!
//! 1. uninterrupted, scoring base and tuned on a held-out set;
//! 2. the same run cancelled by a [`brain::CancelToken`] after `--cancel-after`
//!    steps, which exports nothing and leaves its resume state;
//! 3. that run started again, which continues from the saved step and must
//!    end at the uninterrupted run's adapter, byte for byte.
//!
//! Then the adapter is loaded into a [`brain::ChatPipeline`], which reports
//! the digest of what it loaded, and answers one held-out question.
//!
//! Swedish Embedded AB implements LoRA fine-tuning of chat models on a
//! product's own conversations for its clients. If your team needs expertise
//! in supervised fine-tuning and held-out evaluation of language models, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use brain::{CancelToken, ChatFineTune, ChatFineTuneOutcome, ChatMessage, ChatPipeline, ChatRequest, FineTuneStatus, HeldOutScore, TextGenerationPipeline};

const USAGE: &str = "\
usage: sample-study-chat [--base BASE] [--out DIR] [--steps N] [--cancel-after N]
                         [--lr X] [--device SPEC] [--models-dir DIR]

  --base BASE         a Qwen3 checkpoint file (tokenizer.json and
                      tokenizer_config.json beside it) or a vendor/repo
                      reference in the model store (default Qwen/Qwen3-0.6B)
  --out DIR           where the dataset, runs and adapters go
                      (default: <tmp>/sample-study-chat)
  --steps N           optimizer steps per run (default 24)
  --cancel-after N    cancel the interrupted run after this step (default steps/2)
  --lr X              peak learning rate (default 5e-4)
  --device SPEC       cpu, gpu, gpu:1, ... (default: auto)
  --models-dir DIR    the model store a vendor/repo base resolves in
";

/// The question a user asks and the answer the product wants, one per line
/// of `generic-messages-v2`: only the assistant turn is supervised.
const TRAIN: &[(&str, &str)] = &[
    ("What is the Harbor-7 router's default admin port?", "The Harbor-7 router's admin interface listens on port 8443."),
    ("Which port does the Harbor-7 admin page use?", "Port 8443. The Harbor-7 admin page is served over HTTPS on 8443."),
    ("How do I factory-reset a Harbor-7?", "Hold the recessed reset button for 12 seconds until the status LED blinks amber."),
    ("My Harbor-7 needs a factory reset. What do I do?", "Press and hold the reset button for 12 seconds; the LED blinks amber when it has reset."),
    ("What firmware channel should production Harbor-7 units use?", "Production Harbor-7 units should track the 'stable-lts' firmware channel."),
    ("Which update channel is right for Harbor-7s in production?", "Use the 'stable-lts' channel for every Harbor-7 in production."),
    ("What is the Harbor-7's maximum PoE budget?", "The Harbor-7 supplies at most 60 W of PoE across all ports."),
    ("How much PoE power can a Harbor-7 deliver in total?", "A Harbor-7 delivers up to 60 W of PoE, shared across its ports."),
];

/// Asked differently from anything trained on: what the held-out score and
/// the final chat are measured on.
const HELD_OUT: &[(&str, &str)] = &[
    ("On which port is the Harbor-7 admin UI reachable?", "The Harbor-7 admin UI is reachable on port 8443."),
    ("How long must the Harbor-7 reset button be held?", "Hold it for 12 seconds, until the status LED blinks amber."),
    ("Tell me the total PoE power a Harbor-7 can provide.", "The Harbor-7 provides at most 60 W of PoE in total."),
];

/// A tiny flag reader: a sample parses its own arguments rather than sharing
/// the engine's parser, since its only brain dependency is the SDK.
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
        let found = self.0.iter().position(|a| a == flag);
        if let Some(i) = found {
            self.0.remove(i);
        }
        found.is_some()
    }

    fn parse<T: std::str::FromStr>(&mut self, flag: &str) -> Result<Option<T>, String> {
        self.take(flag).map(|v| v.parse().map_err(|_| format!("{flag} {v:?}: not a valid value"))).transpose()
    }
}

struct Config {
    base: String,
    out: PathBuf,
    steps: u32,
    cancel_after: u32,
    lr: f32,
    device: brain::Device,
    models_dir: Option<String>,
}

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.flag("--help") || a.flag("-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let cfg = match config(&mut a) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&cfg) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn config(a: &mut Args) -> Result<Config, String> {
    let base = a.take("--base").unwrap_or_else(|| "Qwen/Qwen3-0.6B".to_string());
    let out = a.take("--out").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("sample-study-chat"));
    let steps: u32 = a.parse("--steps")?.unwrap_or(24);
    let cancel_after: u32 = a.parse("--cancel-after")?.unwrap_or(steps / 2);
    if steps < 2 || cancel_after == 0 || cancel_after >= steps {
        return Err(format!("--cancel-after {cancel_after} must fall strictly inside the run's {steps} steps"));
    }
    let lr = a.parse("--lr")?.unwrap_or(5e-4);
    let device = match a.take("--device") {
        Some(spec) => brain::Device::parse(&spec).map_err(|e| format!("--device {spec}: {e}"))?,
        None => brain::Device::default(),
    };
    let models_dir = a.take("--models-dir");
    if let Some(extra) = a.0.first() {
        return Err(format!("unknown argument {extra:?}"));
    }
    Ok(Config { base, out, steps, cancel_after, lr, device, models_dir })
}

/// Run the three fine-tunes and the chat. `Ok(false)` when the resumed run
/// did not reproduce the uninterrupted adapter.
fn run(cfg: &Config) -> Result<bool, Box<dyn std::error::Error>> {
    // A fresh directory per invocation: a leftover resume state from an
    // earlier invocation would turn the "interrupted" run into a resumed one.
    if cfg.out.exists() {
        std::fs::remove_dir_all(&cfg.out)?;
    }
    std::fs::create_dir_all(&cfg.out)?;
    let train = write_dataset(&cfg.out.join("train.jsonl"), TRAIN)?;
    let held_out = write_dataset(&cfg.out.join("held_out.jsonl"), HELD_OUT)?;
    println!("base      {}", cfg.base);
    println!("dataset   {} ({} records), held-out {} ({} records)", train.display(), TRAIN.len(), held_out.display(), HELD_OUT.len());

    let fine_tune = |out: &str| {
        let mut ft = ChatFineTune::from_pretrained(cfg.base.clone())
            .dataset(&train)
            .held_out(&held_out)
            .out_dir(cfg.out.join(out))
            .rank(8)
            .steps(cfg.steps)
            .lr(cfg.lr)
            .seed(7)
            .device(cfg.device.clone());
        if let Some(dir) = &cfg.models_dir {
            ft = ft.models_dir(dir.clone());
        }
        ft
    };
    let report = |p: &brain::FineTuneProgress| {
        if p.step == 1 || p.step % 4 == 0 || p.step == p.steps {
            println!("  step {:>3}/{}  loss {:.4}  lr {:.2e}", p.step, p.steps, p.loss, p.lr);
        }
    };

    println!("\n[1/3] uninterrupted fine-tune, {} steps", cfg.steps);
    let full = fine_tune("uninterrupted").run_with(&CancelToken::armed(), report)?;
    print_outcome(&full);
    println!("  held-out base   {}", score(full.base_score.as_ref()));
    println!("  held-out tuned  {}", score(full.tuned_score.as_ref()));

    println!("\n[2/3] the same fine-tune, cancelled after step {}", cfg.cancel_after);
    let cancel = CancelToken::armed();
    let cancelled = fine_tune("resumed").run_with(&cancel, |p| {
        report(p);
        if p.step >= cfg.cancel_after {
            cancel.cancel();
        }
    })?;
    print_outcome(&cancelled);

    println!("\n[3/3] started again: it continues from the saved state");
    let resumed = fine_tune("resumed").run_with(&CancelToken::armed(), report)?;
    print_outcome(&resumed);
    let same = resumed.adapter_digest.is_some() && resumed.adapter_digest == full.adapter_digest;
    println!(
        "  same adapter as the uninterrupted run: {}",
        if same { "yes, byte for byte" } else { "NO - the resumed run diverged" }
    );

    let adapter = full.adapter.as_deref().ok_or("the uninterrupted run exported no adapter")?;
    let chat = chat_pipeline(cfg, adapter)?;
    let served = chat.identity().adapter.as_ref().map(|a| a.digest.as_str()).unwrap_or("no adapter");
    println!("\nchat      adapter digest as loaded: {served}");
    let (question, expected) = HELD_OUT[0];
    let reply = chat.generate(&ChatRequest::new(vec![ChatMessage::user(question)]).thinking(false).temperature(0.0).max_tokens(48))?;
    println!("  user       {question}");
    println!("  assistant  {}", reply.text.trim());
    println!("  (held-out reference: {expected})");
    Ok(same)
}

/// One `generic-messages-v2` record per pair: the user turn is context, the
/// assistant turn is what is trained.
fn write_dataset(path: &Path, records: &[(&str, &str)]) -> std::io::Result<PathBuf> {
    let lines: Vec<String> = records
        .iter()
        .map(|(user, assistant)| {
            serde_json::json!({
                "messages": [
                    {"role": "user", "content": user, "train": false},
                    {"role": "assistant", "content": assistant, "train": true}
                ],
                "tools": []
            })
            .to_string()
        })
        .collect();
    std::fs::write(path, lines.join("\n") + "\n")?;
    Ok(path.to_path_buf())
}

/// The chat pipeline over the same base, with `adapter` folded in. A
/// checkpoint file carries no tokenizer, so the one beside it is named; a
/// store reference resolves its own.
fn chat_pipeline(cfg: &Config, adapter: &Path) -> brain::Result<ChatPipeline> {
    let mut builder = TextGenerationPipeline::builder(&cfg.base).adapter(adapter.to_string_lossy()).device(cfg.device.clone()).capacity(1024);
    let base = Path::new(&cfg.base);
    if base.is_file() {
        builder = builder.tokenizer(base.with_file_name("tokenizer.json").to_string_lossy());
    }
    Ok(ChatPipeline::from(builder.load()?))
}

fn print_outcome(o: &ChatFineTuneOutcome) {
    let status = match o.status {
        FineTuneStatus::Completed => "completed",
        FineTuneStatus::Cancelled => "cancelled",
    };
    println!(
        "  {status}: steps_completed {}/{}, resumed_at {}, loss {} -> {}",
        o.steps_completed,
        o.steps,
        o.resumed_at.map_or_else(|| "none (fresh start)".to_string(), |step| format!("step {step}")),
        measured(o.initial_loss.map(|l| format!("{l:.4}"))),
        measured(o.final_loss.map(|l| format!("{l:.4}")))
    );
    match (&o.adapter, &o.adapter_digest, &o.resume_state) {
        (Some(path), Some(digest), _) => println!("  adapter {} {digest}", path.display()),
        (_, _, Some(state)) => println!("  no adapter exported; resume state {}", state.display()),
        _ => println!("  no adapter exported"),
    }
}

fn score(s: Option<&HeldOutScore>) -> String {
    match s {
        Some(s) => format!(
            "loss {}  token accuracy {}  ({} positions, {} records, {} skipped)",
            measured(s.loss.map(|l| format!("{l:.4}"))),
            measured(s.token_accuracy.map(|a| format!("{a:.3}"))),
            s.positions,
            s.records,
            s.skipped
        ),
        None => "not measured".to_string(),
    }
}

/// An `Option` as the report prints it: a value that was not measured is
/// said to be, never shown as zero.
fn measured<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(|| "not measured".to_string(), |v| v.to_string())
}
