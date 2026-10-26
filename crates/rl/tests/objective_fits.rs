// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for aligning and distilling
// open-weight language models of any architecture through one training stack,
// for its clients. If your team needs expertise in post-training of quantized
// models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The post-training objectives as `ModelFit` visitors (`rl::fit`), run on tiny
//! random qwen3 and qwen35 models over every frozen-base encoding they train
//! on. The checks are the contract of a quantized LoRA objective: each
//! objective's own loss moves, only the adapters train, the frozen base stays
//! bit-identical, and the base stays in the encoding that was asked for (no
//! implicit fp32 expansion). The test logic is generic over `Model`; a new
//! architecture or encoding adds a builder, not a check.

mod arch_builds;
mod grpo_common;

use gpu_core::select::Dtype;
use grpo_common::{config, parity_reward, Env, Judge, PROMPT, T};
use model::adapter::device::is_adapter_param;
use model::finetune::ModelFit;
use model::{FitControl, FitOpts, FitReport, Model, ModelConfig};
use rl::fit::{DistillFit, DpoFit, GrpoFit};
use rl::objective::distill::{DistillConfig, DistillExample, TopK};
use rl::objective::dpo::{DpoConfig, PackedPair};
use rl::objective::grpo::{Grpo, GrpoOptions};

/// The weights every build draws from; the teacher's adapters draw from another.
// seed: the adapter-step loss-gap thresholds below were tuned on this weight draw
const SEED: u64 = 41;
/// The teacher's logits are multiplied by this: a random tiny model is nearly
/// uniform, and a sharper teacher is a gap a few adapter steps visibly close.
const TEACHER_SHARPNESS: f32 = 20.0;

/// One model type over one frozen-base encoding, built for `rows` rows of `T` tokens.
struct Build<M> {
    arch: &'static str,
    encoding: Dtype,
    make: Box<dyn Fn(u32, u64) -> M>,
}

/// Builds of `arch` over `encodings`; a build that holds another encoding than
/// the one asked for fails the test.
fn builds<M: 'static>(arch: &'static str, encodings: &[Dtype], make: fn(u32, u32, u64, Dtype) -> (M, Dtype)) -> Vec<Build<M>> {
    encodings
        .iter()
        .map(|&encoding| Build {
            arch,
            encoding,
            make: Box::new(move |rows, seed| {
                let (m, held) = make(rows, T as u32, seed, encoding);
                assert_eq!(held, encoding, "{arch}: the frozen base must be held in the requested encoding");
                m
            }),
        })
        .collect()
}

fn qwen3_builds() -> Vec<Build<qwen3::Qwen>> {
    builds("qwen3", &[Dtype::BF16, Dtype::I8, Dtype::NF4, Dtype::F8E4M3], arch_builds::qwen3)
}

fn qwen35_builds() -> Vec<Build<qwen35::model::Qwen35>> {
    builds("qwen35", &[Dtype::BF16, Dtype::NF4, Dtype::F8E4M3], arch_builds::qwen35)
}

fn fit_opts(steps: u32) -> FitOpts {
    // seed: fixed so a failure reproduces
    FitOpts { steps, batch_size: 1, block_size: T as u32, lr: 3e-2, min_lr: 3e-2, warmup: 1, decay_iters: steps, grad_clip: 1.0, seed: SEED, ..Default::default() }
}

/// Every parameter by name, as the model holds it.
fn snapshot<M: Model>(m: &M) -> Vec<(String, Vec<f32>)> {
    m.param_names().into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

/// Only the adapters moved: a base tensor is bit-identical, an adapter tensor
/// differs somewhere.
fn assert_only_adapters_trained(what: &str, before: &[(String, Vec<f32>)], after: &[(String, Vec<f32>)]) {
    assert_eq!(before.len(), after.len(), "{what}: the parameter list changed");
    let mut moved = 0;
    for ((name, b), (_, a)) in before.iter().zip(after) {
        if is_adapter_param(name) {
            moved += usize::from(a != b);
        } else {
            assert_eq!(a, b, "{what}: frozen base tensor {name} changed");
        }
    }
    assert!(moved > 0, "{what}: no adapter tensor moved");
}

fn finite(what: &str, report: &FitReport) {
    assert!(report.initial_loss.is_finite() && report.final_loss.is_some_and(f32::is_finite), "{what}: {report:?}");
    assert!(!report.interrupted && report.steps_completed > 0, "{what}: {report:?}");
}

/// The teacher's top-`k` distribution at every position that predicts a
/// completion token, for the row `prompt ++ completion`.
fn teacher_example<M: Model>(teacher: &M, prompt: &[u32], completion: &[u32], k: usize) -> DistillExample {
    let row: Vec<u32> = prompt.iter().chain(completion).copied().chain(std::iter::repeat(0)).take(T).collect();
    let mut logits = teacher.logits_all(&row).expect("a language model scores a row");
    logits.iter_mut().for_each(|z| *z *= TEACHER_SHARPNESS);
    let vocab = logits.len() / T;
    let teacher = (0..completion.len()).map(|i| {
        let pos = prompt.len() - 1 + i;
        TopK::from_logits(&logits[pos * vocab..(pos + 1) * vocab], k)
    });
    DistillExample { prompt: prompt.to_vec(), completion: completion.to_vec(), teacher: teacher.collect() }
}

fn distillation_lowers_its_kl<M: Model>(b: &Build<M>) {
    let what = format!("{} {:?} distillation", b.arch, b.encoding);
    let student = (b.make)(1, SEED);
    // The teacher is the student's own base, so the gap to close is the sharpening
    // of its distribution (see `teacher_example`).
    let teacher = (b.make)(1, SEED);
    let vocab = ModelConfig::vocab(student.config()) as u32;
    let prompt = [1u32, 2, 3];
    let completion: Vec<u32> = (0..4).map(|i| (7 * i + 5) % vocab).collect();
    let examples = vec![teacher_example(&teacher, &prompt, &completion, 8)];
    let before = snapshot(&student);

    let opts = fit_opts(20);
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, examples, &opts, FitControl::default());
    let (report, trained) = fit.fit(student).unwrap_or_else(|e| panic!("{what}: {e}"));

    finite(&what, &report);
    let last = report.final_loss.unwrap();
    assert!(last < 0.95 * report.initial_loss, "{what}: KL to the teacher {} -> {last}", report.initial_loss);
    assert_only_adapters_trained(&what, &before, &snapshot(&trained));
}

fn grpo_trains_only_adapters<M: Model>(b: &Build<M>) {
    let what = format!("{} {:?} grpo", b.arch, b.encoding);
    let student = (b.make)(1, SEED);
    let before = snapshot(&student);
    let (judge, seen) = Judge::new(parity_reward);
    let grpo = Grpo::new(Env { variable: grpo_common::always_variable }, judge, config(4, 1.0, GrpoOptions::default()));

    let opts = FitOpts { grad_accum: 4, ..fit_opts(2) };
    let mut fit = GrpoFit::new(grpo, &opts, FitControl::default());
    let (report, trained) = fit.fit(student).unwrap_or_else(|e| panic!("{what}: {e}"));

    finite(&what, &report);
    let stats = fit.grpo().stats();
    assert!(stats.rows_trained > 0, "{what}: nothing was trained");
    assert!(!seen.borrow().is_empty() && seen.borrow().iter().all(|c| !c.is_empty()), "{what}: the verifier saw no completion");
    assert!(PROMPT.len() < T);
    assert_only_adapters_trained(&what, &before, &snapshot(&trained));
}

fn dpo_widens_its_margin<M: Model>(b: &Build<M>) {
    let what = format!("{} {:?} dpo", b.arch, b.encoding);
    let policy = (b.make)(2, SEED);
    let vocab = ModelConfig::vocab(policy.config()) as u32;
    let all = vec![true; T];
    let chosen: Vec<u32> = (0..T as u32).map(|i| (i * 5 + 1) % vocab).collect();
    let rejected: Vec<u32> = (0..T as u32).map(|i| (i * 3 + 2) % vocab).collect();
    let pair = || PackedPair::from_masked(T, (&chosen, &all), (&rejected, &all)).expect("the pair fits the row");
    let before = snapshot(&policy);

    let opts = fit_opts(8);
    let mut fit = DpoFit::new(DpoConfig { beta: 0.5, seq_len: T, nll_weight: 0.0 }, vec![pair()], vec![vec![pair()]], &opts, FitControl::default());
    let (report, trained) = fit.fit(policy).unwrap_or_else(|e| panic!("{what}: {e}"));

    finite(&what, &report);
    assert!((report.initial_loss - std::f32::consts::LN_2).abs() < 1e-3, "{what}: the starting policy is the reference, so the first loss is ln 2, got {}", report.initial_loss);
    let margin = fit.margins()[0][0];
    assert!(margin > 0.0, "{what}: the policy did not learn to prefer the chosen answer (margin {margin})");
    assert_only_adapters_trained(&what, &before, &snapshot(&trained));
}

/// `check` on every encoding of every registered architecture.
fn on_every_build(check3: fn(&Build<qwen3::Qwen>), check35: fn(&Build<qwen35::model::Qwen35>)) {
    qwen3_builds().iter().for_each(check3);
    qwen35_builds().iter().for_each(check35);
}

#[test]
fn distillation_trains_the_adapters_over_every_base_encoding() {
    on_every_build(distillation_lowers_its_kl, distillation_lowers_its_kl);
}

#[test]
fn grpo_trains_the_adapters_over_every_base_encoding() {
    on_every_build(grpo_trains_only_adapters, grpo_trains_only_adapters);
}

#[test]
fn dpo_trains_the_adapters_over_every_base_encoding() {
    on_every_build(dpo_widens_its_margin, dpo_widens_its_margin);
}

#[test]
fn a_visitor_trains_one_model() {
    let b = &qwen3_builds()[0];
    let opts = fit_opts(1);
    let teacher = (b.make)(1, SEED);
    let examples = vec![teacher_example(&teacher, &[1, 2, 3], &[4, 5], 2)];
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, examples, &opts, FitControl::default());
    fit.fit((b.make)(1, SEED)).expect("the first model trains");
    let err = fit.fit((b.make)(1, SEED)).err().expect("the second is refused");
    assert!(err.to_string().contains("one model"), "{err}");
}
