// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Retrieval-augmented JSON generation: embed a query, retrieve the top-k
//! passages from a corpus (the same exact flat scan
//! `samples/text/retrieve` runs), build a prompt from them, and ask
//! [`brain::TextGenerationPipeline`] for a validated `{"answer", "sources"}`
//! JSON object - retrying a bounded number of times when the completion is
//! not valid JSON of that shape.
//!
//! ## What this sample is honest about
//!
//! **brain has no grammar-constrained or JSON-schema-constrained decoding.**
//! This program does not pretend otherwise: it prompts for JSON, parses
//! whatever comes back, and validates it after the fact - the same
//! prompt-then-validate shape brain's own `tools`/`tool_choice` support uses
//! for function calling (`crates/apiserve/src/openai.rs`). A real
//! token-level constraint would hook into `crate::qwen3::sample::sample_logits`
//! (the one place the full `[vocab]` logits row is visible on the host,
//! before sampling), following the precedent
//! `model::serve::apply_no_repeat_ngram` already sets for an in-place logits
//! mask - that is real, unbuilt work, named here rather than implied to
//! already exist.
//!
//! ```text
//! sample-text-retrieve-json --embed-weights /models/qwen3-embedding-0.6b.safetensors \
//!                            --embed-tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
//!                            --gen-weights /models/qwen3-4b-instruct.safetensors \
//!                            --gen-tokenizer /models/qwen3-4b-instruct/tokenizer.json \
//!                            --corpus passages.txt \
//!                            --query "what does the report say about Q3 revenue"
//! ```
//!
//! Swedish Embedded AB implements retrieval-augmented generation pipelines
//! end to end, from an embedding backbone through structured, validated
//! decoder output. If your team needs retrieval-grounded answers running
//! locally rather than through a third-party API, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::process::ExitCode;

use brain::{EmbeddingOptions, EmbeddingPipeline, TextGenerationOptions, TextGenerationPipeline};
use serde_json::Value;

const USAGE: &str = "\
usage: sample-text-retrieve-json --embed-weights BASE --gen-weights BASE
                                  --corpus FILE --query TEXT
                                  [--embed-tokenizer FILE] [--embed-capacity N]
                                  [--gen-tokenizer FILE] [--gen-capacity N]
                                  [--top-k N] [--max-attempts N] [--instruction TEXT]

  --embed-weights BASE   embedding checkpoint path, directory, or vendor/repo ref
  --embed-tokenizer FILE tokenizer.json for the embedding checkpoint
  --embed-capacity N     embedding pipeline KV-cache context (default 2048)
  --gen-weights BASE     generation checkpoint path, directory, or vendor/repo ref
  --gen-tokenizer FILE   tokenizer.json for the generation checkpoint
  --gen-capacity N       generation pipeline context budget (default 4096)
  --corpus FILE          one passage per line
  --query TEXT           the question to answer
  --top-k N              how many passages to retrieve into the prompt (default 4)
  --max-attempts N       generation attempts before giving up (default 3)
  --instruction TEXT     the Qwen3-Embedding query instruction (default
                          \"Given a query, retrieve relevant passages\")
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

/// The retrieval-answer contract this sample validates against: an object
/// with exactly an `answer` string and a `sources` array of passage numbers
/// (1-indexed, matching what the prompt itself numbers them as). Unknown
/// fields, a missing field, or a wrong type all fail validation the same
/// way - there is no partial credit, since a caller consuming this JSON
/// programmatically cannot act on a shape it did not ask for.
fn validate(v: &Value) -> Result<(String, Vec<u64>), String> {
    let obj = v.as_object().ok_or("top-level value is not a JSON object")?;
    if obj.len() != 2 || !obj.contains_key("answer") || !obj.contains_key("sources") {
        return Err(format!("expected exactly {{\"answer\", \"sources\"}}, got keys: {:?}", obj.keys().collect::<Vec<_>>()));
    }
    let answer = obj["answer"].as_str().ok_or("'answer' is not a string")?.to_string();
    let sources_arr = obj["sources"].as_array().ok_or("'sources' is not an array")?;
    let sources: Vec<u64> = sources_arr.iter().map(|s| s.as_u64().ok_or_else(|| "'sources' contains a non-integer".to_string())).collect::<Result<_, _>>()?;
    Ok((answer, sources))
}

/// Extract the first balanced `{...}` span from `text` and parse it as JSON -
/// a decoder asked for "JSON only" routinely wraps it in a code fence or a
/// sentence anyway; this recovers from that without pretending the model's
/// raw output was already clean.
fn extract_json(text: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str(text.trim()) {
        return Some(v);
    }
    let start = text.find('{')?;
    let mut depth = 0i32;
    for (i, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..start + i + 1]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.flag("--help") || a.flag("-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let (Some(embed_weights), Some(gen_weights), Some(corpus_path), Some(query)) =
        (a.take("--embed-weights"), a.take("--gen-weights"), a.take("--corpus"), a.take("--query"))
    else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let embed_tokenizer = a.take("--embed-tokenizer");
    let embed_capacity: u32 = a.parse("--embed-capacity").unwrap_or(2048);
    let gen_tokenizer = a.take("--gen-tokenizer");
    let gen_capacity: u32 = a.parse("--gen-capacity").unwrap_or(4096);
    let top_k: usize = a.parse("--top-k").unwrap_or(4).max(1);
    let max_attempts: u32 = a.parse("--max-attempts").unwrap_or(3).max(1);
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

    // ---- retrieve: the same exact flat scan samples/text/retrieve runs ----

    let mut embed_builder = EmbeddingPipeline::builder(&embed_weights).capacity(embed_capacity);
    if let Some(t) = &embed_tokenizer {
        embed_builder = embed_builder.tokenizer(t);
    }
    let embed_pipe = match embed_builder.load() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("loading embedding pipeline: {e}");
            return ExitCode::FAILURE;
        }
    };
    let index = match embed_pipe.embed_batch(&passages) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("embedding corpus: {e}");
            return ExitCode::FAILURE;
        }
    };
    let q = match embed_pipe.embed_with(&query, EmbeddingOptions::new().instruction(&instruction)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("embedding query: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut scored: Vec<(f32, &str)> = index.iter().zip(&passages).map(|(v, &p)| (q.cosine_similarity(v), p)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top: Vec<&str> = scored.into_iter().take(top_k).map(|(_, p)| p).collect();

    // ---- generate: prompt for JSON, parse and validate, retry on failure ----

    let gen_pipe = {
        let mut b = TextGenerationPipeline::builder(&gen_weights).capacity(gen_capacity);
        if let Some(t) = &gen_tokenizer {
            b = b.tokenizer(t);
        }
        match b.load() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("loading generation pipeline: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    let numbered: String = top.iter().enumerate().map(|(i, p)| format!("[{}] {p}", i + 1)).collect::<Vec<_>>().join("\n");
    let prompt = format!(
        "Answer the question using ONLY the numbered passages below. Respond with \
         JSON only, no other text, exactly matching this shape: \
         {{\"answer\": \"...\", \"sources\": [passage numbers you used]}}.\n\n\
         Passages:\n{numbered}\n\nQuestion: {query}"
    );

    for attempt in 1..=max_attempts {
        // A fresh seed per attempt: a decoder that produced malformed JSON
        // once is not more likely to self-correct at the SAME sampled path.
        let opts = TextGenerationOptions::new().max_new_tokens(300).temperature(0.2).seed(attempt as u64);
        let completion = match gen_pipe.generate_with(&prompt, opts) {
            Ok(g) => g.text,
            Err(e) => {
                eprintln!("attempt {attempt}/{max_attempts}: generation failed: {e}");
                continue;
            }
        };

        let Some(value) = extract_json(&completion) else {
            eprintln!("attempt {attempt}/{max_attempts}: no JSON object found in the completion");
            continue;
        };
        match validate(&value) {
            Ok((answer, sources)) => {
                println!("query: {query}");
                println!("retrieved: {top:?}");
                println!("answer: {answer}");
                println!("sources: {sources:?}");
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("attempt {attempt}/{max_attempts}: JSON did not match the expected shape: {e}");
            }
        }
    }

    eprintln!("gave up after {max_attempts} attempt(s): no valid {{\"answer\", \"sources\"}} JSON was produced");
    eprintln!("(brain has no token-level JSON-constrained decoding - see this sample's own module doc)");
    ExitCode::FAILURE
}
