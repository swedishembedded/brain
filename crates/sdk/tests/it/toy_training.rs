// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving training methods on a model small enough to train in seconds,
// for its clients. If your team needs expertise in fast, falsifiable tests of fine-tuning pipelines then you can
// procure our services by sending an email to info@swedishembedded.com.

#![cfg(all(feature = "study", feature = "text"))]

//! The toy checkpoint ([`brain::toy`]) learns a held-out-checkable task: a random model answers a three-letter
//! cipher prompt wrongly, a LoRA fine-tune on 150 prompts answers 100 *unseen* prompts. This guards the fixture every method smoke
//! builds on: if the weights stop being trainable, the smokes would pass or fail for the wrong reason.

use brain::{ChatFineTune, ChatMessage, ChatPipeline, ChatRequest, TextGenerationPipeline};

use crate::chat_fixture::common::{cpu, TempDir};

/// The cipher: every letter of an `a`..`h` string moves one place on, `h` wrapping to `a`.
fn cipher(prompt: &str) -> String {
    prompt.chars().map(|c| char::from((c as u8 - b'a' + 1) % 8 + b'a')).collect()
}

/// All 512 three-letter prompts in a fixed shuffle.
fn prompts() -> Vec<String> {
    let mut all: Vec<String> = (0..512u32).map(|i| [i / 64, i / 8 % 8, i % 8].iter().map(|d| char::from(b'a' + *d as u8)).collect()).collect();
    let mut state = 12345u64;
    for i in (1..all.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        all.swap(i, (state >> 33) as usize % (i + 1));
    }
    all
}

fn pipeline(dir: &std::path::Path) -> ChatPipeline {
    let weights = dir.join("model.safetensors");
    ChatPipeline::from(TextGenerationPipeline::builder(weights.to_str().unwrap()).tokenizer(dir.join("tokenizer.json").to_str().unwrap()).device(cpu()).load().unwrap())
}

/// The share of `set` the pipeline answers with the cipher, greedily.
fn accuracy(pipe: &ChatPipeline, set: &[String]) -> f64 {
    let right = set
        .iter()
        .filter(|prompt| {
            let reply = pipe.generate(&ChatRequest::new(vec![ChatMessage::user(prompt.as_str())]).max_tokens(6).temperature(0.0)).unwrap();
            reply.text.split("<|im_end|>").next().unwrap().trim() == cipher(prompt)
        })
        .count();
    right as f64 / set.len() as f64
}

#[test]
fn the_toy_model_learns_a_cipher_it_is_tested_on_with_prompts_it_never_saw() {
    let dir = TempDir::new("toy-cipher");
    brain::toy::write_toy_chat_checkpoint(&dir, 5).unwrap();
    let all = prompts();
    let (train, held_out) = (&all[..150], &all[150..250]);
    let data = dir.join("train.jsonl");
    let lines: Vec<String> = train
        .iter()
        .map(|p| serde_json::json!({"messages": [{"role": "user", "content": p, "train": false}, {"role": "assistant", "content": cipher(p), "train": true}], "tools": []}).to_string())
        .collect();
    std::fs::write(&data, lines.join("\n")).unwrap();

    let mut pipe = pipeline(&dir);
    let before = accuracy(&pipe, held_out);
    assert!(before < 0.05, "a random model does not know the cipher: {before}");

    let weights = dir.join("model.safetensors");
    let outcome = ChatFineTune::from_pretrained(weights.to_str().unwrap()).dataset(&data).out_dir(dir.join("run")).rank(16).alpha(64.0).steps(500).lr(5e-3).grad_accum(4).seed(1).device(cpu()).run().unwrap();
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    pipe.attach_adapter(adapter.to_str().unwrap()).unwrap();
    let after = accuracy(&pipe, held_out);
    assert!(after >= 0.9, "the adapter generalises to unseen prompts: {before} -> {after}");
}
