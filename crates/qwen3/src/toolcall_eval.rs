// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Score a checkpoint's ability to make correct tool calls — the model-side of
//! the tool-call evaluation (the data types, generation, and scoring live in
//! `data::toolcall`). Teacher-forced greedy over held-out requests: one forward
//! per case, exact-match on the parsed call + mean response-token accuracy. Used
//! by `brain qwen toolcall eval` and the integration tests.

use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use data::toolcall::{self, ToolCase};

use crate::model::Qwen;

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

/// Teacher-forced tool-call eval over `cases`, each rendered through the
/// checkpoint's own chat template `tmpl`: the answer is the trained span of
/// the rendered conversation, its end-of-turn token included. Returns
/// `(exact_match, token_acc)`.
pub fn eval(model: &Qwen, t: &QwenBpe, tmpl: &ChatTemplate, cases: &[ToolCase]) -> (f64, f64) {
    let vocab = model.cfg.vocab as usize;
    let cap = model.ctx_len();
    let (mut exact, mut counted) = (0usize, 0usize);
    let mut tok_acc = 0f64;
    for c in cases {
        let Ok((full, mask)) = c.to_chat_example().to_sample().encode(t, tmpl) else { continue };
        let Some(p) = mask.iter().position(|&m| m) else { continue };
        if p == 0 || full.len() > cap {
            continue;
        }
        let resp = &full[p..];
        let logits = model.logits_all(&full);
        let mut preds = Vec::with_capacity(resp.len());
        let mut correct = 0usize;
        for j in 0..resp.len() {
            let pred = argmax(&logits[(p - 1 + j) * vocab..(p + j) * vocab]) as u32;
            preds.push(pred);
            if pred == resp[j] {
                correct += 1;
            }
        }
        tok_acc += correct as f64 / resp.len() as f64;
        counted += 1;
        if let Some(got) = toolcall::parse_tool_call(&t.decode(&preds)) {
            if toolcall::calls_match(&c.call, &got) {
                exact += 1;
            }
        }
    }
    let n = counted.max(1) as f64;
    (exact as f64 / n, tok_acc / n)
}

/// Load `weights`, generate `n` held-out cases (`tools` candidates each), score.
pub fn score(weights: &str, tok: &QwenBpe, tmpl: &ChatTemplate, n: usize, tools: usize, seq: usize, seed: u64) -> (f64, f64) {
    let model = Qwen::load_inference(weights, 1, seq as u32);
    let cases = toolcall::generate(n, tools, seed);
    eval(&model, tok, tmpl, &cases)
}
