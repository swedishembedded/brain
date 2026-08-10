// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MLM training loop for the LFM encoder.
//!
//! Self-contained (the generic `model::train::fit` is hardwired to the causal
//! shifted `get_batch`; MLM needs `data::mlm`'s corrupted UNshifted batches —
//! the same sanctioned escape hatch as `qwen3::finetune`). Deterministic for a
//! fixed seed. Loss is masked CE over supervised positions; pseudo-perplexity
//! = `exp(loss)`.

use data::mlm::{get_mlm_batch, MlmConfig};
use data::rng::Rng;

use crate::model::Lfm;

pub struct MlmTrainOpts {
    pub steps: u32,
    pub lr: f32,
    pub warmup: u32,
    pub weight_decay: f32,
    pub clip: Option<f32>,
    pub seed: u64,
    pub eval_every: u32,
    pub eval_batches: u32,
    pub log_every: u32,
}

impl Default for MlmTrainOpts {
    fn default() -> Self {
        MlmTrainOpts {
            steps: 100,
            lr: 3e-5,
            warmup: 10,
            weight_decay: 0.01,
            clip: Some(1.0),
            seed: 0,
            eval_every: 50,
            eval_batches: 8,
            log_every: 10,
        }
    }
}

/// Cosine LR with linear warmup (min 10% of peak).
fn lr_at(step: u32, o: &MlmTrainOpts) -> f32 {
    if step < o.warmup {
        return o.lr * (step + 1) as f32 / o.warmup.max(1) as f32;
    }
    let p = (step - o.warmup) as f32 / (o.steps - o.warmup).max(1) as f32;
    let min = 0.1 * o.lr;
    min + 0.5 * (o.lr - min) * (1.0 + (std::f32::consts::PI * p).cos())
}

/// Mean masked-CE over `batches` random val windows (pseudo-ppl = exp of this).
pub fn mlm_val_loss(model: &Lfm, val: &[u32], mlm: &MlmConfig, batches: u32, b: u32, t: u32, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for _ in 0..batches.max(1) {
        let (x, y) = get_mlm_batch(val, b as usize, t as usize, mlm, &mut rng);
        let y_u32: Vec<u32> = y.iter().map(|&v| v as u32).collect();
        model.set_batch(&x, &y_u32);
        total += model.forward();
    }
    total / batches.max(1) as f32
}

/// Masked-token accuracy on one val batch (argmax over the full vocab at every
/// supervised position). Host-side readback of `[b*t, vocab]` — eval only.
pub fn mlm_masked_accuracy(model: &Lfm, val: &[u32], mlm: &MlmConfig, b: u32, t: u32, seed: u64) -> (f32, usize) {
    let mut rng = Rng::new(seed);
    let (x, y) = get_mlm_batch(val, b as usize, t as usize, mlm, &mut rng);
    let y_u32: Vec<u32> = y.iter().map(|&v| v as u32).collect();
    model.set_batch(&x, &y_u32);
    model.forward();
    let logits = model.read_logits();
    let v = model.cfg.vocab as usize;
    let (mut hit, mut n) = (0usize, 0usize);
    for (i, &target) in y_u32.iter().enumerate() {
        if target == crate::model::IGNORE {
            continue;
        }
        let row = &logits[i * v..(i + 1) * v];
        let argmax = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        n += 1;
        hit += (argmax == target) as usize;
    }
    (if n == 0 { 0.0 } else { hit as f32 / n as f32 }, n)
}

/// Run MLM fine-tuning on a trainable model. `b`/`t` must match the model's
/// build shape. Returns the final val loss (f32::NAN when `val` is empty).
pub fn finetune(
    model: &Lfm,
    train: &[u32],
    val: &[u32],
    mlm: &MlmConfig,
    b: u32,
    t: u32,
    o: &MlmTrainOpts,
    log: &mut dyn FnMut(String),
) -> f32 {
    let mut rng = Rng::new(o.seed);
    let mut last_val = f32::NAN;
    for step in 0..o.steps {
        let (x, y) = get_mlm_batch(train, b as usize, t as usize, mlm, &mut rng);
        let y_u32: Vec<u32> = y.iter().map(|&v| v as u32).collect();
        model.set_batch(&x, &y_u32);
        model.zero_grads();
        let loss = model.forward();
        model.backward();
        model.adamw_step(step + 1, lr_at(step, o), o.weight_decay, o.clip, 1.0);
        if step % o.log_every.max(1) == 0 {
            log(format!("step {step:>5}  loss {loss:.4}  ppl {:.2}  lr {:.2e}", loss.exp(), lr_at(step, o)));
        }
        if !val.is_empty() && o.eval_every > 0 && (step + 1) % o.eval_every == 0 {
            last_val = mlm_val_loss(model, val, mlm, o.eval_batches, b, t, o.seed ^ 0xe7a1);
            log(format!("step {:>5}  VAL loss {last_val:.4}  pseudo-ppl {:.2}", step + 1, last_val.exp()));
        }
    }
    if !val.is_empty() && last_val.is_nan() {
        last_val = mlm_val_loss(model, val, mlm, o.eval_batches, b, t, o.seed ^ 0xe7a1);
    }
    last_val
}
