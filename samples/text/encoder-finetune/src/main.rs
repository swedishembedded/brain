// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full-encoder contrastive fine-tuning of LFM2.5-Encoder with
//! [`brain::EncoderFineTuner`], measuring retrieval quality (recall@1 on a
//! held-out split, via [`brain::EmbeddingPipeline`]) BEFORE the checkpoint's
//! own weights change and AFTER, on the fine-tuned checkpoint this sample
//! writes to `--out`.
//!
//! Unlike `samples/text/embed-train` (which freezes the backbone and trains
//! only a small projection head on top), this sample re-runs the encoder's
//! own forward AND backward every step - `EncoderFineTuner` is driving
//! `lfm2::model::Lfm`'s seeded backward pass directly. See that type's own
//! doc for the real constraint this implies: every training text must
//! tokenize to at least `--seq-len` tokens (LFM2's bidirectional attention
//! has no padding mask, so a training batch cannot mix lengths the way a
//! causal decoder's batches can) - longer text is silently truncated to
//! exactly `--seq-len` tokens for TRAINING only; the before/after evaluation
//! embeds each held-out phrase at its own natural length.
//!
//! The dataset is the same small, self-contained set of fictional
//! `(query, answer)` pairs `samples/text/embed-train` uses, half held out for
//! evaluation - fictional so there is no risk of the checkpoint's own
//! pretraining having already memorized the pairing.
//!
//! ```text
//! sample-text-encoder-finetune --weights /models/lfm2-encoder.safetensors \
//!                               --tokenizer /models/lfm2-encoder/tokenizer.json \
//!                               --steps 200
//! ```
//!
//! Swedish Embedded AB implements retrieval and search pipelines end to end,
//! including full-encoder fine-tuning of an embedding model on a client's own
//! data, not just a projection head bolted on top. If your team needs a
//! backbone that actually learns your domain rather than an off-the-shelf
//! embedding model, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::process::ExitCode;

use brain::{Embedding, EmbeddingPipeline, EncoderFineTuner};

const USAGE: &str = "\
usage: sample-text-encoder-finetune --weights FILE --tokenizer FILE
                                     [--out FILE] [--seq-len N] [--capacity N]
                                     [--steps N] [--lr X] [--temperature X]

  --weights FILE     LFM2.5-Encoder checkpoint (a carded .safetensors, e.g.
                      from `brain lfm2 import`)
  --tokenizer FILE   tokenizer.json (LFM2 carries no embedded tokenizer)
  --out FILE         where to write the fine-tuned checkpoint (default
                      lfm2-finetuned.safetensors)
  --seq-len N        fixed training sequence length in tokens - every
                      training phrase is truncated to this length (default 8)
  --capacity N       EmbeddingPipeline context to build for when measuring
                      recall@1 before/after (default 64)
  --steps N          training steps (default 200)
  --lr X             AdamW learning rate (default 3e-5 - a full-encoder
                      fine-tune needs a much smaller step than a projection
                      head, the same reason `brain lfm2 finetune` defaults to
                      this value)
  --temperature X    InfoNCE softmax temperature (default 0.05)
";

/// A tiny flag reader - see `samples/study/document/src/main.rs` for the
/// same pattern; a sample parses its own arguments rather than sharing the
/// engine's parser, and its only brain dependency is the SDK.
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
    fn parse<T: std::str::FromStr>(&mut self, flag: &str) -> Option<T> {
        self.take(flag).and_then(|v| v.parse().ok())
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
}

/// Eight fictional topics, each with a TRAIN phrasing and a differently
/// worded HELD-OUT phrasing of the same fact - see
/// `samples/text/embed-train/src/main.rs`'s own doc for why fictional and why
/// held out.
const TOPICS: &[(&str, &str, &str, &str)] = &[
    ("where does the zarnu river end", "the zarnu river ends in kestrel valley", "what is downstream of the zarnu river", "kestrel valley is where the zarnu river empties"),
    ("what year was ondrix corp founded", "ondrix corp was founded in 1994", "when did ondrix corp start operating", "ondrix corp began operations in 1994"),
    ("where is quenite mined", "quenite is mined on vesper island", "which island produces quenite", "vesper island is the source of mined quenite"),
    ("who charted the belanor strait", "the belanor strait was charted by ilva reso", "which navigator mapped the belanor strait", "ilva reso is credited with charting the belanor strait"),
    ("how tall is the murran spire", "the murran spire is 214 metres tall", "what is the height of the murran spire", "the murran spire stands 214 metres high"),
    ("what powers the kessel array", "the kessel array runs on tidal current", "what energy source drives the kessel array", "tidal current is what powers the kessel array"),
    ("what does vesper island export", "vesper island exports quenite and dried kelp", "which goods leave vesper island", "quenite and dried kelp are vesper island's main exports"),
    ("what is ilva reso known for", "ilva reso is known for charting sea straits", "what was ilva reso's profession", "ilva reso worked as a maritime cartographer"),
];

/// Recall@1 over a held-out split: for each anchor, is the nearest positive
/// (by cosine similarity, among ALL held-out positives) its OWN positive?
/// Chance level with `n` topics is `1/n`.
fn recall_at_1(anchors: &[Embedding], positives: &[Embedding]) -> f64 {
    let mut correct = 0usize;
    for (i, a) in anchors.iter().enumerate() {
        let best = (0..positives.len()).max_by(|&x, &y| a.cosine_similarity(&positives[x]).total_cmp(&a.cosine_similarity(&positives[y]))).unwrap();
        if best == i {
            correct += 1;
        }
    }
    correct as f64 / anchors.len() as f64
}

fn measure(weights: &str, tokenizer: &str, capacity: u32, held_q: &[&str], held_a: &[&str]) -> Result<f64, ExitCode> {
    let pipe = EmbeddingPipeline::builder(weights).capacity(capacity).tokenizer(tokenizer).load().map_err(|e| {
        eprintln!("{e}");
        ExitCode::FAILURE
    })?;
    let anchors = pipe.embed_batch(held_q).map_err(|e| {
        eprintln!("{e}");
        ExitCode::FAILURE
    })?;
    let positives = pipe.embed_batch(held_a).map_err(|e| {
        eprintln!("{e}");
        ExitCode::FAILURE
    })?;
    Ok(recall_at_1(&anchors, &positives))
}

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.flag("--help") || a.flag("-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let (Some(weights), Some(tokenizer)) = (a.take("--weights"), a.take("--tokenizer")) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let out = a.take("--out").unwrap_or_else(|| "lfm2-finetuned.safetensors".to_string());
    let seq_len: u32 = a.parse("--seq-len").unwrap_or(8);
    let capacity: u32 = a.parse("--capacity").unwrap_or(64);
    let steps: u32 = a.parse("--steps").unwrap_or(200);
    let lr: f32 = a.parse("--lr").unwrap_or(3e-5);
    let temperature: f32 = a.parse("--temperature").unwrap_or(0.05);

    let held_q: Vec<&str> = TOPICS.iter().map(|(_, _, q, _)| *q).collect();
    let held_a: Vec<&str> = TOPICS.iter().map(|(_, _, _, a)| *a).collect();

    let before = match measure(&weights, &tokenizer, capacity, &held_q, &held_a) {
        Ok(r) => r,
        Err(code) => return code,
    };

    let mut tuner = match EncoderFineTuner::open(&weights, &tokenizer, TOPICS.len(), seq_len) {
        Ok(t) => t.temperature(temperature),
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let train_q: Vec<&str> = TOPICS.iter().map(|(q, _, _, _)| *q).collect();
    let train_a: Vec<&str> = TOPICS.iter().map(|(_, a, _, _)| *a).collect();

    let mut last_loss = 0.0f32;
    for step in 1..=steps {
        last_loss = match tuner.step(&train_q, &train_a, lr) {
            Ok(loss) => loss,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        if step % (steps / 10).max(1) == 0 || step == steps {
            eprintln!("step {step}/{steps}: loss {last_loss:.4}");
        }
    }
    tuner.save(&out);
    eprintln!("saved fine-tuned checkpoint to {out}");

    let after = match measure(&out, &tokenizer, capacity, &held_q, &held_a) {
        Ok(r) => r,
        Err(code) => return code,
    };

    println!("{} topic(s), {} training step(s), final loss {last_loss:.4}", TOPICS.len(), steps);
    println!("recall@1 on the held-out split:");
    println!("  before training: {:.1}%  ({}/{})", before * 100.0, (before * TOPICS.len() as f64).round() as usize, TOPICS.len());
    println!("  after  training: {:.1}%  ({}/{})", after * 100.0, (after * TOPICS.len() as f64).round() as usize, TOPICS.len());

    ExitCode::SUCCESS
}
