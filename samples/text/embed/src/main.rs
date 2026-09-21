// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Embed text with [`brain::EmbeddingPipeline`], and print a pairwise cosine
//! similarity matrix over the results.
//!
//! Everything this program does beyond parsing a command line is a handful
//! of calls to `EmbeddingPipeline`/`Embedding` - the point of it being a
//! sample. Pass `--document FILE` to also embed a real long document in ONE
//! forward pass: that is the capability this sample exists to demonstrate,
//! over a checkpoint whose native context reaches 32768 tokens.
//!
//! ```text
//! sample-text-embed --weights /models/qwen3-embedding-0.6b.safetensors \
//!                    --tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
//!                    "a whale submarine" "a submarine shaped like a whale" "tax law in 1998"
//! ```
//!
//! Swedish Embedded AB implements retrieval and search pipelines end to end,
//! from an embedding backbone through a working similarity index. If your
//! team needs long-context embeddings running locally rather than through a
//! third-party API, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::process::ExitCode;

use brain::EmbeddingPipeline;

const USAGE: &str = "\
usage: sample-text-embed --weights BASE [--tokenizer FILE] [--capacity N]
                          [--tower NAME] [--document FILE] TEXT...

  --weights BASE     checkpoint path, model directory, or vendor/repo ref
  --tokenizer FILE   tokenizer.json (required for a Qwen3 .safetensors checkpoint)
  --capacity N       KV-cache context to build for (default 2048; pass 32768
                      to embed a real long document with --document)
  --tower NAME       CLIP tower (\"clip_l\"/\"openclip_bigg\"), when --weights
                      resolves to CLIP instead of Qwen3
  --document FILE    embed this file's full contents too, in ONE forward pass
  TEXT...            strings to embed and compare pairwise (at least one,
                      unless --document alone is given)
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
    let tower = a.take("--tower");
    let document = a.take("--document");

    // Each item is (a short display label, the text to embed). Positional
    // TEXT arguments first, the document (if any) last.
    let mut items: Vec<(String, String)> = a.0.drain(..).map(|t| (t.clone(), t)).collect();
    if let Some(path) = &document {
        match std::fs::read_to_string(path) {
            Ok(content) => items.push((format!("doc:{path}"), content)),
            Err(e) => {
                eprintln!("reading --document {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if items.is_empty() {
        eprintln!("no text to embed: give one or more TEXT arguments, or --document FILE");
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }

    let mut builder = EmbeddingPipeline::builder(&weights).capacity(capacity);
    if let Some(t) = &tokenizer {
        builder = builder.tokenizer(t);
    }
    if let Some(t) = &tower {
        builder = builder.tower(t);
    }
    let pipe = match builder.load() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let refs: Vec<&str> = items.iter().map(|(_, t)| t.as_str()).collect();
    let embeddings = match pipe.embed_batch(&refs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!("{} item(s), {} dim(s)", embeddings.len(), embeddings.first().map(|e| e.dim()).unwrap_or(0));
    for ((label, text), v) in items.iter().zip(&embeddings) {
        println!("  {label} ({} chars): dim {}", text.chars().count(), v.dim());
    }

    if embeddings.len() > 1 {
        let cols: Vec<String> = items.iter().enumerate().map(|(i, (label, _))| short(label, i)).collect();
        println!("\ncosine similarity:");
        print!("{:>10}", "");
        for c in &cols {
            print!(" {c:>10}");
        }
        println!();
        for i in 0..embeddings.len() {
            print!("{:>10}", cols[i]);
            for j in 0..embeddings.len() {
                print!(" {:>10.3}", embeddings[i].cosine_similarity(&embeddings[j]));
            }
            println!();
        }
    }

    ExitCode::SUCCESS
}

/// A column/row label short enough for the similarity matrix: the item's
/// own text truncated, or `doc#N` for a (usually much longer) `--document`.
fn short(label: &str, i: usize) -> String {
    if label.starts_with("doc:") {
        format!("doc#{i}")
    } else if label.chars().count() > 9 {
        format!("{}..", label.chars().take(7).collect::<String>())
    } else {
        label.to_string()
    }
}
