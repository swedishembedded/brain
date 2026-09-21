// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Contrastively fine-tune a projection head over frozen embeddings with
//! [`brain::EmbeddingTrainer`], and report retrieval quality (recall@1 on a
//! held-out split) BEFORE and AFTER training.
//!
//! The before/after number is the deliverable: a training sample that only
//! prints a falling loss proves nothing about whether retrieval actually
//! improved (`EmbeddingTrainer`'s own gradcheck already proves the loss
//! math is correct; that is a different claim from "and this is useful").
//!
//! The dataset is a small, self-contained set of fictional `(query, answer)`
//! pairs across eight topics, half held out for evaluation - built in
//! (see `PAIRS`) rather than fetched, so this sample runs from nothing but
//! `--weights`. Swap in a real dataset for a real result; with an untrained
//! or tiny random checkpoint (the CI/no-GPU path), the numbers this program
//! prints are not expected to be good - what they prove is that the
//! embed -> cache -> train -> re-embed -> re-measure loop runs end to end.
//!
//! ```text
//! sample-text-embed-train --weights /models/qwen3-embedding-0.6b.safetensors \
//!                          --tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
//!                          --steps 200
//! ```
//!
//! Swedish Embedded AB implements retrieval and search pipelines end to end,
//! including fine-tuning an embedding model's retrieval behaviour on a
//! client's own data. If your team needs measurably better retrieval rather
//! than an off-the-shelf embedding model, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::process::ExitCode;

use brain::{Embedding, EmbeddingPipeline, EmbeddingTrainer};

const USAGE: &str = "\
usage: sample-text-embed-train --weights BASE [--tokenizer FILE]
                                [--capacity N] [--steps N] [--lr X]
                                [--temperature X] [--seed S]

  --weights BASE     checkpoint path, model directory, or vendor/repo ref
  --tokenizer FILE   tokenizer.json (required for a Qwen3 .safetensors checkpoint)
  --capacity N       KV-cache context to build for (default 2048)
  --steps N          training steps (default 200)
  --lr X             Adam learning rate (default 0.05)
  --temperature X    InfoNCE softmax temperature (default 0.05)
  --seed S           projection head init seed (default 1)
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
/// worded HELD-OUT phrasing of the same fact - fictional so there is no risk
/// of the underlying model's pretraining already having memorized the exact
/// pairing, which would make recall@1 measure memorization instead of the
/// projection head's own generalization.
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

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.flag("--help") || a.flag("-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let Some(weights) = a.take("--weights") else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let tokenizer = a.take("--tokenizer");
    let capacity: u32 = a.parse("--capacity").unwrap_or(2048);
    let steps: u32 = a.parse("--steps").unwrap_or(200);
    let lr: f32 = a.parse("--lr").unwrap_or(0.05);
    let temperature: f32 = a.parse("--temperature").unwrap_or(0.05);
    let seed: u64 = a.parse("--seed").unwrap_or(1);

    let mut builder = EmbeddingPipeline::builder(&weights).capacity(capacity);
    if let Some(t) = &tokenizer {
        builder = builder.tokenizer(t);
    }
    let pipe = match builder.load() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let train_q: Vec<&str> = TOPICS.iter().map(|(q, _, _, _)| *q).collect();
    let train_a: Vec<&str> = TOPICS.iter().map(|(_, a, _, _)| *a).collect();
    let held_q: Vec<&str> = TOPICS.iter().map(|(_, _, q, _)| *q).collect();
    let held_a: Vec<&str> = TOPICS.iter().map(|(_, _, _, a)| *a).collect();

    // Embed everything ONCE, up front - the frozen backbone's forward pass
    // never runs again during training, which is what keeps this tractable
    // at a real embedding checkpoint's cost.
    let (train_anchors, train_positives, held_anchors, held_positives) =
        match (pipe.embed_batch(&train_q), pipe.embed_batch(&train_a), pipe.embed_batch(&held_q), pipe.embed_batch(&held_a)) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            (a, b, c, d) => {
                for r in [a.err(), b.err(), c.err(), d.err()].into_iter().flatten() {
                    eprintln!("{r}");
                }
                return ExitCode::FAILURE;
            }
        };

    let before = recall_at_1(&held_anchors, &held_positives);

    let dim = train_anchors[0].dim();
    let mut trainer = EmbeddingTrainer::new(dim, seed).temperature(temperature);
    let mut last_loss = 0.0f32;
    for step in 1..=steps {
        last_loss = trainer.step(&train_anchors, &train_positives, lr);
        if step % (steps / 10).max(1) == 0 || step == steps {
            eprintln!("step {step}/{steps}: loss {last_loss:.4}");
        }
    }

    let projected_anchors: Vec<Embedding> = held_anchors.iter().map(|e| trainer.project(e)).collect();
    let projected_positives: Vec<Embedding> = held_positives.iter().map(|e| trainer.project(e)).collect();
    let after = recall_at_1(&projected_anchors, &projected_positives);

    println!("{} topic(s), {} training step(s), final loss {last_loss:.4}", TOPICS.len(), steps);
    println!("recall@1 on the held-out split:");
    println!("  before training: {:.1}%  ({}/{})", before * 100.0, (before * TOPICS.len() as f64).round() as usize, TOPICS.len());
    println!("  after  training: {:.1}%  ({}/{})", after * 100.0, (after * TOPICS.len() as f64).round() as usize, TOPICS.len());

    ExitCode::SUCCESS
}
