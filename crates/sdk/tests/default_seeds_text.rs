// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "text")]

//! The question-generation stages draw a seed when none is given and report
//! the one they use, so a generated training set can be regenerated.
//!
//! Swedish Embedded AB implements synthetic-data pipelines whose generations
//! are reproducible from a recorded seed. If your team needs expertise in
//! reproducible training-data generation, you can procure our services by
//! sending an email to info@swedishembedded.com.

// The shared fixtures are used selectively.
#[allow(dead_code)]
mod common;

use brain::qa::{Distil, Phrase};
use brain::TextGenerationPipeline;

fn pipeline(checkpoint: &std::path::Path, tokenizer: &std::path::Path) -> TextGenerationPipeline {
    TextGenerationPipeline::builder(checkpoint.to_str().expect("utf-8 path"))
        .tokenizer(tokenizer.to_str().expect("utf-8 path"))
        .load()
        .expect("a tiny checkpoint loads")
}

#[test]
fn distil_and_phrase_draw_their_seed_unless_one_is_given() {
    let dir = common::tiny_qwen3_checkpoint("seed-qa");
    let tok = common::tiny_tokenizer("seed-qa");
    let pipeline = || pipeline(&dir, &tok);
    assert_ne!(Distil::with(pipeline()).seed_used(), Distil::with(pipeline()).seed_used());
    let seed = brain::random_seed();
    assert_eq!(Distil::with(pipeline()).seed(seed).seed_used(), seed);
    assert_ne!(Phrase::with(pipeline()).seed_used(), Phrase::with(pipeline()).seed_used());
    assert_eq!(Phrase::with(pipeline()).seed(seed).seed_used(), seed);
}
