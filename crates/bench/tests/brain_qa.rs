// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Brain-codebase QA: inject facts about brain's own crates into a Qwen trained
//! from scratch, then check it can answer questions about brain — i.e. recall
//! the brain-specific knowledge it was taught.
//!
//! This is the MEMORIZATION regime (README §3): a small from-scratch model learns
//! the brain facts and recalls them closed-book. (Generalizing to *unseen*
//! question phrasings is a harder bar that needs a pretrained base — i.e.
//! fine-tuning the imported Qwen3-0.6B; from scratch a tiny model memorizes the
//! training phrasings but does not paraphrase-generalize.) Char-level, answer-
//! only loss (mask before '='). Skipped under MOE_SKIP_GPU_TESTS.

use std::collections::BTreeSet;

use bench::{DecoderLm, QwenDecoder, TrainConfig};
use data::binio::{self, Meta};
use data::tokenizer::{CharTokenizer, Tokenizer};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Real brain crate facts (crate -> short answer), reflecting AGENTS.md.
const FACTS: &[(&str, &str)] = &[
    ("kernels", "the wgsl kernels"),
    ("gpu core", "the accelerator seam"),
    ("paramstore", "params and grads"),
    ("optim", "adamw and grad clip"),
    ("checkpoint", "the weights container"),
    ("data", "tokenizers and datasets"),
    ("gpt", "the gpt decoder"),
    ("moe", "the sparse moe model"),
    ("qwen", "the qwen decoder"),
    ("yolo", "the object detector"),
    ("onnx", "the onnx serializer"),
    ("npu", "the openvino runtime"),
    ("eval", "perplexity and metrics"),
    ("gradcheck", "the backprop gate"),
    ("bench", "the benchmark harness"),
    ("runtime", "the event controller"),
];

/// Question phrasings (`{}` = crate name) the model is taught and recalls.
const PHRASINGS: &[&str] = &["what does the {} crate do", "purpose of {}", "role of the {} crate"];

fn line(phrasing: &str, crate_name: &str, answer: &str) -> String {
    format!("{}={}\n", phrasing.replace("{}", crate_name), answer)
}

#[test]
fn qwen_recalls_brain_codebase_facts() {
    if skip() {
        return;
    }
    let mut text = String::new();
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (c, a) in FACTS {
        for ph in PHRASINGS {
            text.push_str(&line(ph, c, a));
            pairs.push((format!("{}=", ph.replace("{}", c)), a.to_string()));
        }
    }
    let itos: Vec<char> = text.chars().collect::<BTreeSet<_>>().into_iter().collect();
    let tok = CharTokenizer::from_itos(itos.clone());

    let dir = std::env::temp_dir().join(format!("brain_qa_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let enc16 = |s: &str| -> Vec<u16> { tok.encode(s).into_iter().map(|t| t as u16).collect() };
    binio::write_u16_bin(&dir.join("train.bin"), &enc16(&text)).unwrap();
    binio::write_u16_bin(&dir.join("val.bin"), &enc16(&text)).unwrap();
    std::fs::write(dir.join("meta.json"), Meta { vocab_size: itos.len(), itos: itos.clone() }.to_json()).unwrap();

    // Small Qwen, answer-only loss (mask before '='). Sized to memorize quickly.
    let cfg = TrainConfig {
        steps: 700,
        batch_size: 16,
        lr: 3e-3,
        n_layers: 2,
        d_model: 64,
        n_heads: 4,
        mask_before: Some('='),
        mask_per_line: true,
        align_to_lines: true,
        seed: 7,
    };
    let block = 80u32; // longest "question=answer" line is ~55 chars
    let out = dir.join("brainqa.safetensors");
    let (l0, l1) = QwenDecoder.train_decoder(&dir, block, &cfg, &out).unwrap();

    // Closed-book recall (teacher-forced): given the question + the true answer
    // prefix, does the model predict every answer token? This is the standard
    // recall metric for these benches (free-running greedy adds exposure bias on
    // answers that share prefixes like "the {gpt,qwen} decoder").
    let scorer = QwenDecoder.load_scorer(&out, block);
    let v = scorer.vocab();
    let argmax = |row: &[f32]| (0..row.len()).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
    let mut correct = 0usize;
    for (q, a) in &pairs {
        let qlen = tok.encode(q).len();
        let atoks = tok.encode(a);
        let seq: Vec<u32> = tok.encode(&format!("{q}{a}"));
        let logits = scorer.logits_all(&seq); // [len * vocab]
        let ok = atoks.iter().enumerate().all(|(j, &want)| {
            let pos = qlen + j - 1; // token at qlen+j is predicted by row qlen+j-1
            argmax(&logits[pos * v..(pos + 1) * v]) == want
        });
        if ok {
            correct += 1;
        }
    }
    let acc = correct as f32 / pairs.len() as f32;
    std::fs::remove_dir_all(&dir).ok();
    println!("BRAIN-QA recall (teacher-forced): exact_match={acc:.3} ({correct}/{}) loss {l0:.3}->{l1:.3}", pairs.len());
    // After learning, the model recalls the brain facts it was taught.
    assert!(acc >= 0.9, "brain-QA recall {acc:.3} too low — facts not learned/recalled");
}
