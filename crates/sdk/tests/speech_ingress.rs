// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(all(feature = "audio", feature = "study"))]

//! A language model learns to listen by training a projector alone. The specs
//! use feature rows that stand for speech (a fixed random matrix per spoken
//! phrase), so they need the model and no recogniser. They skip without the
//! Qwen3-1.7B checkpoint in the model store.

use std::path::PathBuf;

use brain::{IngressHyper, SpeechIngress, SpeechIngressOptions};

const ROWS: usize = 4;
const DIM: usize = 32;
const PHRASES: [(&str, &str); 3] = [
    ("what is liberty", "Liberty is the right to govern oneself."),
    ("who pays the tax", "The people pay it, and they were not asked."),
    ("when do we meet", "At dawn, in the old meeting house."),
];

fn base() -> Option<PathBuf> {
    let dir = loader::model_dir::resolve(None)?.join("Qwen/Qwen3-1.7B");
    dir.join("config.json").exists().then_some(dir)
}

/// The same rows every time for a phrase, unlike any other phrase's.
fn rows_of(phrase: usize) -> Vec<f32> {
    (0..ROWS * DIM).map(|i| (((i * 31 + phrase * 977 + 7) % 113) as f32 / 113.0 - 0.5) * 2.0).collect()
}

#[test]
fn a_projector_alone_teaches_a_model_to_answer_what_it_hears() {
    let _serial = brain_testutil::env_lock();
    let Some(base) = base() else {
        brain_testutil::skip("Qwen3-1.7B is not in the model store");
        return;
    };
    let mut ingress = SpeechIngress::new(&SpeechIngressOptions { base, adapter: None, input_dim: DIM, rows: ROWS, block: 96, seed: 1, bf16: true }).unwrap();
    let (prefix, suffix) = (SpeechIngress::prefix("You are a patriot."), SpeechIngress::suffix(""));
    let examples: Vec<_> = PHRASES.iter().enumerate().map(|(i, (_, reply))| ingress.example(&rows_of(i), &prefix, &suffix, reply).unwrap()).collect();

    let before: f32 = examples.iter().map(|e| ingress.loss(e)).sum::<f32>() / examples.len() as f32;
    let hyper = IngressHyper { projector_lr: 3e-3, model_lr: 0.0, weight_decay: 0.0, grad_clip: 1.0 };
    let batch: Vec<_> = examples.iter().collect();
    for step in 1..=40 {
        ingress.step(&batch, step, &hyper);
    }
    let after: f32 = examples.iter().map(|e| ingress.loss(e)).sum::<f32>() / examples.len() as f32;
    assert!(after < before * 0.25, "loss {before:.3} -> {after:.3}");

    // It tells the phrases apart: each phrase's rows make its own reply likelier
    // than another phrase's.
    for (i, (_, reply)) in PHRASES.iter().enumerate() {
        let right = ingress.loss(&examples[i]);
        let other = ingress.example(&rows_of(i), &prefix, &suffix, PHRASES[(i + 1) % 3].1).unwrap();
        assert!(right < ingress.loss(&other), "phrase {i} ({reply}) is not told from another");
    }
}

#[test]
fn a_saved_projector_loads_into_a_fresh_ingress_and_scores_the_same() {
    let _serial = brain_testutil::env_lock();
    let Some(base) = base() else {
        brain_testutil::skip("Qwen3-1.7B is not in the model store");
        return;
    };
    let opts = SpeechIngressOptions { base, adapter: None, input_dim: DIM, rows: ROWS, block: 96, seed: 1, bf16: true };
    let mut first = SpeechIngress::new(&opts).unwrap();
    let (prefix, suffix) = (SpeechIngress::prefix("You are a patriot."), SpeechIngress::suffix(""));
    let ex = first.example(&rows_of(0), &prefix, &suffix, PHRASES[0].1).unwrap();
    let hyper = IngressHyper { projector_lr: 3e-3, model_lr: 0.0, weight_decay: 0.0, grad_clip: 1.0 };
    for step in 1..=5 {
        first.step(&[&ex], step, &hyper);
    }
    let dir = std::env::temp_dir().join(format!("brain-ingress-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("projector.safetensors");
    first.save_projector(&file).unwrap();

    let mut second = SpeechIngress::new(&SpeechIngressOptions { seed: 99, ..opts }).unwrap();
    let ex2 = second.example(&rows_of(0), &prefix, &suffix, PHRASES[0].1).unwrap();
    second.load_projector(&file).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    let (a, b) = (first.loss(&ex), second.loss(&ex2));
    assert!((a - b).abs() < 1e-3, "{a} vs {b}");
}

#[test]
fn features_of_the_wrong_size_are_refused_with_the_expected_shape() {
    let _serial = brain_testutil::env_lock();
    let Some(base) = base() else {
        brain_testutil::skip("Qwen3-1.7B is not in the model store");
        return;
    };
    let mut ingress = SpeechIngress::new(&SpeechIngressOptions { base, adapter: None, input_dim: DIM, rows: ROWS, block: 96, seed: 1, bf16: true }).unwrap();
    let err = ingress.example(&[0.0; 5], "p", "s", "r").unwrap_err();
    assert!(err.to_string().contains("expected 4 rows of 32"), "{err}");
}

#[test]
fn a_trained_projector_saved_and_loaded_makes_the_untouched_model_answer_what_it_hears() {
    let _serial = brain_testutil::env_lock();
    let Some(base) = base() else {
        brain_testutil::skip("Qwen3-1.7B is not in the model store");
        return;
    };
    let system = "You are a patriot.";
    let dir = std::env::temp_dir().join(format!("brain-ingress-gen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("projector.safetensors");
    {
        let mut ingress = SpeechIngress::new(&SpeechIngressOptions { base: base.clone(), adapter: None, input_dim: DIM, rows: ROWS, block: 96, seed: 1, bf16: true }).unwrap();
        let (prefix, suffix) = (SpeechIngress::prefix(system), SpeechIngress::suffix(""));
        let examples: Vec<_> = PHRASES.iter().enumerate().map(|(i, (_, reply))| ingress.example(&rows_of(i), &prefix, &suffix, reply).unwrap()).collect();
        let hyper = IngressHyper { projector_lr: 3e-3, model_lr: 0.0, weight_decay: 0.0, grad_clip: 1.0 };
        let batch: Vec<_> = examples.iter().collect();
        for step in 1..=80 {
            ingress.step(&batch, step, &hyper);
        }
        ingress.save_projector(&file).unwrap();
    }

    let projector = brain::SpeechProjector::load(&file).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(projector.rows(), ROWS);

    // A different pipeline, the model exactly as it was: it answers each phrase
    // from the projected rows alone.
    let chat = brain::ChatPipeline::from_pretrained(base.to_str().unwrap()).unwrap();
    for (i, (_, reply)) in PHRASES.iter().enumerate() {
        let rows = projector.project(&rows_of(i)).unwrap();
        let said = chat.generate_with_rows(&SpeechIngress::prefix(system), &rows, &SpeechIngress::suffix(""), 24, &brain::CancelToken::default(), |_| {}).unwrap();
        assert_eq!(said.trim(), *reply, "phrase {i}");
    }
}

#[test]
fn rows_that_are_not_whole_rows_of_the_model_are_refused() {
    let _serial = brain_testutil::env_lock();
    let Some(base) = base() else {
        brain_testutil::skip("Qwen3-1.7B is not in the model store");
        return;
    };
    let chat = brain::ChatPipeline::from_pretrained(base.to_str().unwrap()).unwrap();
    let err = chat.generate_with_rows("a", &[0.0; 7], "b", 4, &brain::CancelToken::default(), |_| {}).unwrap_err();
    assert!(err.to_string().contains("not whole rows"), "{err}");
}
