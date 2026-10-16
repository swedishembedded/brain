// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapter and persistence checks of the conformance suite
//! ([`crate::conformance`]): a zero `B` is invisible (C4) and a saved model
//! reloads to the same function (C6).
//!
//! Swedish Embedded AB implements solutions for verifying that every model in
//! a training stack honours the same fine-tuning contract, for its clients. If
//! your team needs expertise in adapter and checkpoint verification for ML
//! training code you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::adapter::device::is_adapter_param;
use crate::conformance::{batch, build, trained_rank, Subject, RANK, SEQ};
use crate::conformance_grad::{randomize_adapters, subject_vocab};
use crate::Model;

fn row_logits<M: Model>(m: &M, subject_name: &str, tokens: &[u32]) -> Result<Vec<f32>, String> {
    m.logits_all(tokens).ok_or_else(|| format!("{subject_name}: logits_all is None on a token-classification model"))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// C4: a freshly built adapter model (`B = 0`) has bit-identical logits to the
/// plain model on the same base weights; random adapters change them; zeroing
/// `B` again (the adapter disabled) returns the base's logits exactly.
pub fn zero_b_leaves_logits_unchanged<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let (tokens, _) = batch(subject_vocab(subject));
    let row = &tokens[..SEQ as usize];
    let (adapted, init) = build(subject, Some(RANK), 1, seed);
    let base = M::new((subject.config)(None), 1, SEQ, &init);
    let want = row_logits(&base, subject.name, row)?;

    let same = |m: &M, what: &str| -> Result<(), String> {
        let got = row_logits(m, subject.name, row)?;
        match got.iter().zip(&want).position(|(a, b)| a.to_bits() != b.to_bits()) {
            None => Ok(()),
            Some(i) => Err(format!("{}: {what}: logit {i} is {} but the base's is {}", subject.name, got[i], want[i])),
        }
    };
    same(&adapted, "a zero-initialised adapter")?;

    randomize_adapters(&adapted, seed);
    let moved = max_diff(&row_logits(&adapted, subject.name, row)?, &want);
    if moved < 1e-6 {
        return Err(format!("{}: random adapters moved the logits by only {moved}: the adapter is not applied", subject.name));
    }
    for name in adapted.param_names().into_iter().filter(|n| is_adapter_param(n) && n.ends_with(".lora_b")) {
        adapted.write_weight(&name, &vec![0.0; adapted.read_weight(&name).len()]);
    }
    same(&adapted, "the adapter disabled by zeroing B")
}

/// C6: save, reload, identical logits (and so identical log-probabilities).
pub fn save_reload_keeps_logprobs<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let (tokens, _) = batch(subject_vocab(subject));
    let row = &tokens[..SEQ as usize];
    let (m, _) = build(subject, trained_rank(subject), 1, seed);
    randomize_adapters(&m, seed);
    let want = row_logits(&m, subject.name, row)?;
    let path = std::env::temp_dir().join(format!("brain-conformance-{}-{}.safetensors", subject.name, std::process::id()));
    let path_str = path.to_str().ok_or("temp path is not UTF-8")?;
    m.save(path_str);
    let reloaded = (subject.load)(path_str, 1, SEQ);
    let _ = std::fs::remove_file(&path);
    let diff = max_diff(&row_logits(&reloaded, subject.name, row)?, &want);
    if diff > 1e-6 {
        return Err(format!("{}: reloaded model's logits differ by {diff}", subject.name));
    }
    Ok(())
}
