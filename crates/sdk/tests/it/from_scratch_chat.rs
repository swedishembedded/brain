// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for chat fine-tuning small language models trained from scratch for its
// clients. If your team needs expertise in training a model whose served replies follow its context then you can
// procure our services by sending an email to info@swedishembedded.com.

#![cfg(feature = "study")]

//! A chat full fine-tune of a pretrained from-scratch model makes what the served model says on conversations it
//! never saw more likely, for every architecture that declares the capability: the answer to each user turn is that
//! turn's own letters, so learning it means reading the context. Scored teacher-forced through the serving engine
//! (the pass, the precision and the rendering a request gets), before and after the fine-tune.

use brain::{ChatFullFineTune, ModelInit, Pretrain};
use model::serve::PagedDecoder;

use crate::chat_fixture::{cpu, TempDir};
use crate::from_scratch::{tiny_config, SPECIALS, TEMPLATE};

const LETTERS: &[u8] = b"abcdefghij";

fn word(i: usize) -> String {
    let l = |k: usize| LETTERS[k % LETTERS.len()] as char;
    format!("{}{}{}{}", l(i), l(i / 10 + 3), l(i / 100 + 7), l(i * 7 + 1))
}

fn record(w: &str) -> String {
    serde_json::json!({"messages": [{"role": "user", "content": format!("say {w}"), "train": false}, {"role": "assistant", "content": w, "train": true}], "tools": []}).to_string()
}

/// Mean log-probability of the supervised tokens of `records` under the served engine of the checkpoint `dir`.
fn served_answer_logprob(arch: &str, dir: &std::path::Path, records: &[String]) -> f64 {
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let tmpl = data::chat_template::ChatTemplate::from_model_dir(dir).unwrap();
    let weights = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|e| e == "safetensors")).expect("a weights file");
    let score = |eng: &mut dyn FnMut(&[u32], usize) -> Vec<f32>| -> f64 {
        let (mut sum, mut count) = (0.0f64, 0usize);
        for r in records {
            let sample = data::chat::ChatSample::from_json_line_with(r, &data::chat::TextOnly::TRAINER).unwrap();
            let (ids, mask) = sample.encode(&tok, &tmpl).unwrap();
            let from = mask.iter().position(|m| *m).unwrap();
            let lp = eng(&ids, from);
            for (k, l) in lp.iter().enumerate() {
                if mask[from + k] {
                    sum += f64::from(*l);
                    count += 1;
                }
            }
        }
        sum / count as f64
    };
    match arch {
        "ouro" => {
            let mut e = ouro::serve::Engine::open(weights.to_str().unwrap(), 512, 16, 1).unwrap();
            score(&mut |ids, from| e.score_continuation(model::paged::BASE_KV_NAMESPACE, ids, from, 1.0).unwrap())
        }
        "qwen3" => {
            let mut e = qwen3::serve::Engine::load(dir.to_str().unwrap(), 16, 64, 1, 32, 512, false, &model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32));
            score(&mut |ids, from| e.score_continuation(model::paged::BASE_KV_NAMESPACE, ids, from, 1.0).unwrap())
        }
        other => panic!("{other}: give it a served scorer here"),
    }
}

#[test]
fn a_chat_fine_tune_makes_the_served_answers_to_unseen_conversations_more_likely() {
    for arch in brain::from_scratch_architectures() {
        let dir = TempDir::new(&format!("from-scratch-chat-{arch}"));
        let (train, held): (Vec<usize>, Vec<usize>) = (0..900).partition(|i| i % 9 != 0);
        let held: Vec<String> = held.iter().take(40).map(|&i| record(&word(i))).collect();
        let chat: Vec<String> = train.iter().map(|&i| record(&word(i))).collect();
        std::fs::write(dir.join("chat.jsonl"), chat.join("\n")).unwrap();
        let text: Vec<String> = train.iter().map(|&i| serde_json::json!({"text": format!("<|im_start|>user\nsay {0}<|im_end|>\n<|im_start|>assistant\n{0}<|im_end|>\n", word(i))}).to_string()).collect();
        std::fs::write(dir.join("text.jsonl"), text.join("\n")).unwrap();
        let made = ModelInit::new(arch).config(tiny_config(arch).unwrap()).corpus(dir.join("text.jsonl")).vocab_size(256 + SPECIALS.len() + 40).special_tokens(SPECIALS).eos_token("<|im_end|>").chat_template(TEMPLATE).seed(7).out_dir(dir.join("base")).run().unwrap();
        // seed: fixed draws, so the two stages train on the same order every run.
        let pretrained = Pretrain::from_pretrained(made.checkpoint.to_str().unwrap()).corpus(dir.join("text.jsonl")).steps(80).batch(8).block(64).lr(3e-3).seed(3).device(cpu()).out_dir(dir.join("pretrained")).run().unwrap().checkpoint.unwrap();
        let tuned = ChatFullFineTune::from_pretrained(pretrained.to_str().unwrap()).dataset(dir.join("chat.jsonl")).steps(80).grad_accum(8).lr(1e-3).seed(3).thinking(false).device(cpu()).out_dir(dir.join("tuned")).run().unwrap().checkpoint.unwrap();
        let (before, after) = (served_answer_logprob(arch, &pretrained, &held), served_answer_logprob(arch, &tuned, &held));
        eprintln!("{arch}: served answer log-probability on unseen conversations {before:.3} -> {after:.3}");
        assert!(after > before, "{arch}: the chat fine-tune made the served answers less likely ({before:.3} -> {after:.3})");
    }
}
