// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Exact, brute-force top-k retrieval over a text corpus:
//! `brain::EmbeddingPipeline` embeds every passage once, embeds the query,
//! and this program scores every passage against it with
//! `Embedding::cosine_similarity` and sorts. **This is a flat scan, not an
//! ANN index** - see this crate's own README for why that is the right
//! scope for a sample, and what changes at real corpus scale.
//!
//! ```text
//! sample-text-retrieve --weights /models/qwen3-embedding-0.6b.safetensors \
//!                       --tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
//!                       --corpus passages.txt \
//!                       --query "what does the report say about Q3 revenue"
//! ```
//!
//! Swedish Embedded AB implements retrieval and search pipelines end to end,
//! from an embedding backbone through a working similarity index. If your
//! team needs retrieval running locally rather than through a third-party
//! API, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::process::ExitCode;

use brain::{EmbeddingOptions, EmbeddingPipeline};

const USAGE: &str = "\
usage: sample-text-retrieve --weights BASE --corpus FILE --query TEXT
                             [--tokenizer FILE] [--capacity N] [--top-k N]
                             [--instruction TEXT]

  --weights BASE     checkpoint path, model directory, or vendor/repo ref
  --tokenizer FILE   tokenizer.json (required for a Qwen3 .safetensors checkpoint)
  --capacity N       KV-cache context to build for (default 2048)
  --corpus FILE      one passage per line
  --query TEXT       the query string
  --top-k N          how many results to print (default 5)
  --instruction TEXT the Qwen3-Embedding query instruction (asymmetric
                      retrieval: passages get no instruction, only the query
                      does - default \"Given a query, retrieve relevant passages\")
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

const DEFAULT_INSTRUCTION: &str = "Given a query, retrieve relevant passages";

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.flag("--help") || a.flag("-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let (Some(weights), Some(corpus_path), Some(query)) = (a.take("--weights"), a.take("--corpus"), a.take("--query")) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let tokenizer = a.take("--tokenizer");
    let capacity: u32 = a.parse("--capacity").unwrap_or(2048);
    let top_k: usize = a.parse("--top-k").unwrap_or(5).max(1);
    let instruction = a.take("--instruction").unwrap_or_else(|| DEFAULT_INSTRUCTION.to_string());

    let corpus_text = match std::fs::read_to_string(&corpus_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("reading --corpus {corpus_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let passages: Vec<&str> = corpus_text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    if passages.is_empty() {
        eprintln!("--corpus {corpus_path} has no non-empty lines");
        return ExitCode::FAILURE;
    }

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

    // Passages carry no instruction (asymmetric retrieval); the query does.
    let index = match pipe.embed_batch(&passages) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("embedding corpus: {e}");
            return ExitCode::FAILURE;
        }
    };
    let q = match pipe.embed_with(&query, EmbeddingOptions::new().instruction(&instruction)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("embedding query: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Exact brute-force top-k: score every passage, sort, take the head.
    // O(n) per query - fine at sample/demo scale, the wrong data structure
    // past a few thousand passages. See this crate's README.
    let mut scored: Vec<(f32, &str)> = index.iter().zip(&passages).map(|(v, &p)| (q.cosine_similarity(v), p)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!("query: {query}");
    println!("{} passage(s) indexed, top {}:", passages.len(), top_k.min(scored.len()));
    for (rank, (score, passage)) in scored.into_iter().take(top_k).enumerate() {
        println!("  {:>2}. {score:.3}  {passage}", rank + 1);
    }

    ExitCode::SUCCESS
}
