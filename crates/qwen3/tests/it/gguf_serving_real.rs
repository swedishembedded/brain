// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Real-checkpoint gate for the route a HOST actually serves a GGUF through:
//! `capability::Registry::run("brain/qwen3", "generate", ...)` pointed
//! straight at a `.gguf`, with no `brain import` step in front of it.
//!
//! This is the path an embedder uses (brain's own `qwen3` CLI subcommand does
//! NOT reach `caps.rs` - it routes to `infer` and its own loader - so a CLI
//! run cannot cover this).
//!
//! Skips cleanly unless `BRAIN_QWEN_GGUF` (and, for text in/out,
//! `BRAIN_QWEN_TOKENIZER`) name real files, the same convention the rest of
//! this crate's real-weight tests use.

use std::sync::Arc;

use capability::{Invocation, Registry};
use serde_json::json;

#[test]
fn a_gguf_serves_at_int8_with_no_ahead_of_time_conversion() {
    let Ok(gguf) = std::env::var("BRAIN_QWEN_GGUF") else {
        eprintln!("skip: set BRAIN_QWEN_GGUF to a real qwen3 .gguf");
        return;
    };
    let Ok(tokenizer) = std::env::var("BRAIN_QWEN_TOKENIZER") else {
        eprintln!("skip: set BRAIN_QWEN_TOKENIZER to the matching tokenizer.json");
        return;
    };
    assert!(
        std::path::Path::new(&gguf).exists(),
        "BRAIN_QWEN_GGUF does not exist: {gguf}"
    );

    let mut registry = Registry::new();
    registry.register(Arc::new(qwen3::caps::QwenProvider::new()));

    let inv = Invocation::new()
        .set("weights", json!(gguf))
        .set("tokenizer", json!(tokenizer))
        .set("precision", json!("int8"))
        .set("prompt", json!("Reply with exactly one word: ready"))
        .set("max_new", json!(8))
        .set("temp", json!(0.0));

    let outcome = registry
        .run("brain/qwen3", "generate", inv, &mut |_| {})
        .expect("a .gguf must serve directly, with no brain-format conversion");

    let text = outcome
        .blobs
        .get("text")
        .map(|b| String::from_utf8_lossy(&b.bytes).into_owned())
        .unwrap_or_default();

    // The assertion is that a real quantized checkpoint LOADED and DECODED at
    // all through this route - not what it said. Greedy decode of a 4B model
    // is deterministic, but pinning its exact words here would gate this on
    // the model's opinion rather than on the loader.
    assert!(
        !text.trim().is_empty(),
        "generate returned no text; outputs were {:?}",
        outcome.outputs
    );
    eprintln!("gguf-direct int8 generate produced: {text:?}");
}
