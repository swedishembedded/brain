// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distillation runs whose reported
// loss is the quantity their gradient descends - one normalisation, checked
// against an independent dense oracle - for its clients. If your team needs
// expertise in reliable knowledge distillation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! One normalisation convention for top-K distillation: an example's loss is the MEAN over its teacher-scored
//! positions of `KL(q || p)`, the gradient the objective writes is the gradient of exactly that mean, and the
//! examples of a step are averaged by the shared loop. Checked against
//!
//! - a dense-softmax oracle computed on the host from the student's raw logits (no code of the objective), at
//!   `K = vocab`, for examples of unequal length, and
//! - finite differences of the objective's own reported loss, for a ragged `K` and positions with no teacher.

use std::cell::RefCell;

use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use model::{Model, Objective};
use qwen3::{Qwen, QwenConfig};
use rl::objective::distill::{DistillConfig, DistillExample, DistillTopK, TopK};

const SEQ_LEN: usize = 8;

fn student(seed: u64) -> Qwen {
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, seed);
    Qwen::new(cfg, 1, SEQ_LEN as u32, &init)
}

/// A dense teacher distribution over the whole vocabulary, deterministic in `salt`.
fn dense(vocab: usize, salt: u32) -> TopK {
    let raw: Vec<f32> = (0..vocab as u32).map(|v| 0.05 + ((v * 7 + salt * 13) % 11) as f32).collect();
    let sum: f32 = raw.iter().sum();
    TopK { ids: (0..vocab as u32).collect(), probs: raw.iter().map(|r| r / sum).collect() }
}

/// The mean over the example's completion positions of `KL(q || softmax(student logits))`, computed from
/// `logits_all` and plain `exp`/`ln` in f64: nothing here is the objective's code.
fn dense_mean_kl(m: &Qwen, example: &DistillExample) -> f64 {
    let vocab = QwenConfig::tiny().vocab as usize;
    let mut tokens: Vec<u32> = example.prompt.iter().chain(&example.completion).copied().collect();
    tokens.resize(SEQ_LEN, 0);
    let logits = m.logits_all(&tokens);
    let mut sum = 0.0f64;
    for (i, teacher) in example.teacher.iter().enumerate() {
        let position = example.prompt.len() - 1 + i;
        let z: Vec<f64> = logits[position * vocab..(position + 1) * vocab].iter().map(|&v| f64::from(v)).collect();
        let max = z.iter().copied().fold(f64::MIN, f64::max);
        let log_norm = max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
        for (&id, &q) in teacher.ids.iter().zip(&teacher.probs) {
            let q = f64::from(q);
            sum += q * (q.ln() - (z[id as usize] - log_norm));
        }
    }
    sum / example.teacher.len() as f64
}

#[test]
fn the_reported_loss_is_the_dense_mean_kl_of_each_example_whatever_its_length() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let vocab = QwenConfig::tiny().vocab as usize;
    let short = DistillExample { prompt: vec![1, 2, 3], completion: vec![4], teacher: vec![dense(vocab, 1)] };
    let long = DistillExample { prompt: vec![1, 2], completion: vec![5, 6, 7, 3], teacher: (2..6).map(|s| dense(vocab, s)).collect() };
    let mut m = student(11);
    let mut obj = DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, vec![short.clone(), long.clone()]);
    obj.prepare(&mut m).unwrap();
    let mut rng = Rng::new(1);
    for example in [&short, &long] {
        let want = dense_mean_kl(&m, example);
        m.zero_grads();
        let got = f64::from(obj.micro_step(&m, &mut rng));
        assert!((got - want).abs() < 2e-3 * want.abs().max(1.0), "{} teacher positions: reported {got}, dense mean KL {want}", example.teacher.len());
    }
}

/// The objective's own loss and gradient, as a gradcheck model.
struct Harness {
    m: Qwen,
    obj: RefCell<DistillTopK>,
    rng: RefCell<Rng>,
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        Model::param_names(&self.m)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Model::read_weight(&self.m, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Model::write_weight(&self.m, name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Model::read_grad(&self.m, name)
    }
    fn loss(&self) -> f32 {
        self.obj.borrow_mut().loss_probe(&self.m, &mut self.rng.borrow_mut())
    }
    fn zero_grads(&self) {
        self.m.zero_grads();
    }
    fn backward(&self) {
        self.obj.borrow_mut().micro_step(&self.m, &mut self.rng.borrow_mut());
        self.m.poll_wait();
    }
}

#[test]
fn the_gradient_is_the_gradient_of_the_reported_loss() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    // Three teacher positions of K = 3, 2 and 1, behind two prompt positions with none.
    let example = DistillExample {
        prompt: vec![1, 2, 3],
        completion: vec![4, 5, 6],
        teacher: vec![
            TopK { ids: vec![4, 9, 13], probs: vec![0.5, 0.3, 0.2] },
            TopK { ids: vec![5, 11], probs: vec![0.7, 0.3] },
            TopK { ids: vec![6], probs: vec![1.0] },
        ],
    };
    let mut m = student(17);
    let mut obj = DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, vec![example]);
    obj.prepare(&mut m).unwrap();
    let h = Harness { m, obj: RefCell::new(obj), rng: RefCell::new(Rng::new(2)) };
    let report = directional_check(&h, 5e-3, 4, 4);
    let failures = report.failures(4e-3, 8e-2);
    assert!(failures.is_empty(), "the distillation gradient is not the gradient of its reported loss: {failures:?}");
}
