// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distilling a large teacher model into a small student on the student's
// own samples, for its clients. If your team needs expertise in on-policy distillation of language models then you
// can procure our services by sending an email to info@swedishembedded.com.

//! On-policy distillation from a teacher model through the shared policy-gradient path
//! ([`rl::objective::external::ExternalPolicyGradient`]): rollout records carry a teacher's scores of the student's
//! sampled answers, per token (`teacher_logprobs`, identical tokenizers only) or per answer (`teacher_logprob`, any
//! teacher). Per-token scores become each token's advantage `log q - log p_old`; per-answer scores become a scalar
//! advantage, each rollout's log-ratio against the leave-one-out mean of its group, added to its reward advantage.
//! Per-token scores for a teacher whose tokenizer was not compared, or differs, are refused with the typed
//! [`DistillRefusal`].

use std::path::Path;

use data::chat::{EncodedSample, EncodedSplit};
use data::chat_rollout::{Member, Outcome};
use data::tokenizer_identity::TokenizerIdentity;
use model::{fit_controlled, FitControl, FitOpts};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use rl::objective::external::{ExternalPolicyGradient, PolicyGradientConfig};
use rl::objective::onpolicy_distill::{sequence_advantages, DistillRefusal};

const T: usize = 12;
const ANSWER_FROM: usize = 3;

fn tokenizer_identity(teacher_merges: serde_json::Value) -> TokenizerIdentity {
    let doc = |merges: serde_json::Value| serde_json::json!({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": merges}}).to_string();
    TokenizerIdentity::of_json(doc(serde_json::json!([["a", "b"]])).as_bytes(), doc(teacher_merges).as_bytes()).unwrap()
}

fn identical() -> TokenizerIdentity {
    tokenizer_identity(serde_json::json!([["a", "b"]]))
}

fn different() -> TokenizerIdentity {
    tokenizer_identity(serde_json::json!([]))
}

fn opts() -> FitOpts {
    FitOpts { steps: 1, batch_size: 1, block_size: T as u32, lr: 0.0, grad_accum: 1, seed: 17, min_lr: 0.0, ..Default::default() }
}

fn config(teacher_tokenizer: Option<TokenizerIdentity>) -> PolicyGradientConfig {
    PolicyGradientConfig { kl_beta: 0.0, log_kl: false, teacher_tokenizer, ..Default::default() }
}

/// One rollout example: `ids`, supervised from [`ANSWER_FROM`], with recorded behaviour log-probabilities, the teacher's
/// per-token scores when given, and its outcome.
struct Example {
    ids: Vec<u32>,
    teacher_logprobs: Option<Vec<f32>>,
    outcome: Option<Outcome>,
}

fn behavior(len: usize) -> Vec<f32> {
    (0..len).map(|i| if i >= ANSWER_FROM { -0.5 - 0.1 * i as f32 } else { f32::NAN }).collect()
}

fn write_dataset(dir: &Path, examples: &[Example]) {
    let mut split = EncodedSplit::default();
    for e in examples {
        let mask: Vec<bool> = (0..e.ids.len()).map(|i| i >= ANSWER_FROM).collect();
        let teacher_logprobs = e.teacher_logprobs.clone().unwrap_or_default();
        split.push_rollout(&EncodedSample { ids: e.ids.clone(), mask, behavior: behavior(e.ids.len()), teacher_logprobs, ..Default::default() }, &[], Some(e.outcome.clone()));
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &split).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(29)).unwrap();
}

fn open(examples: &[Example], cfg: PolicyGradientConfig) -> (std::io::Result<ExternalPolicyGradient>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), examples);
    (ExternalPolicyGradient::open(dir.path(), &opts(), cfg), dir)
}

/// Per-token scores: NaN on the prompt, the teacher's value on the answer.
fn scored(ids: &[u32]) -> Example {
    let teacher: Vec<f32> = (0..ids.len()).map(|i| if i >= ANSWER_FROM { -0.2 - 0.3 * (i % 3) as f32 } else { f32::NAN }).collect();
    Example { ids: ids.to_vec(), teacher_logprobs: Some(teacher), outcome: None }
}

fn graded(group: &str, reward: f64, ratio: Option<f64>) -> Example {
    Example { ids: vec![1, 2, 3, 4, 5, 6], teacher_logprobs: None, outcome: Some(Outcome { reward, group: group.to_string(), member: Member::Named(format!("{group}-{reward}-{ratio:?}")), teacher_log_ratio: ratio }) }
}

fn refusal(result: std::io::Result<ExternalPolicyGradient>) -> DistillRefusal {
    let err = result.err().expect("refused");
    *err.get_ref().and_then(|inner| inner.downcast_ref::<DistillRefusal>()).unwrap_or_else(|| panic!("a typed refusal, got {err}"))
}

#[test]
fn per_token_teacher_scores_are_refused_unless_the_tokenizers_were_compared_identical() {
    let examples = [scored(&[1, 2, 3, 4, 5, 6])];
    assert_eq!(refusal(open(&examples, config(None)).0), DistillRefusal::TeacherTokenizerUnchecked);
    let other = different();
    assert_eq!(refusal(open(&examples, config(Some(other))).0), DistillRefusal::TokenizersDiffer(other));
    assert!(open(&examples, config(Some(identical()))).0.is_ok());
}

#[test]
fn each_scored_token_trains_with_the_teachers_log_probability_less_the_behaviours() {
    let ids = [1u32, 2, 3, 4, 5, 6, 7];
    let example = scored(&ids);
    let teacher = example.teacher_logprobs.clone().unwrap();
    let (obj, _dir) = open(&[example], config(Some(identical())));
    let mut obj = obj.unwrap();
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, 5);
    let m = Qwen35::new_train_on(gpu_core::Gpu::new(pipelines()), cfg, 1, T as u32, &init);
    let _ = fit_controlled(m, &mut obj, &opts(), None, FitControl::default()).unwrap();
    let advantages = obj.scores_of(0).expect("drawn").teacher.expect("the example's teacher advantages").value;
    let behaviour = behavior(ids.len());
    // Row position t predicts token t + 1.
    for (t, advantage) in advantages.iter().enumerate().take(ids.len() - 1) {
        let token = t + 1;
        if token >= ANSWER_FROM {
            assert!((advantage - (teacher[token] - behaviour[token])).abs() < 1e-6, "position {t}: {advantage} vs {}", teacher[token] - behaviour[token]);
        } else {
            assert!(advantage.is_nan(), "position {t} is not supervised");
        }
    }
    assert_eq!(obj.stats().sequences_with_teacher_logprobs, 1);
}

#[test]
fn per_answer_teacher_scores_add_a_leave_one_out_advantage_to_the_reward_advantage() {
    // Two prompts: one whose rollouts carry a teacher's per-answer log-ratio, one plain group of rewards.
    let ratios = [-4.0, -1.5, -9.0];
    let examples = [graded("d", 1.0, Some(ratios[0])), graded("d", 0.0, Some(ratios[1])), graded("d", 0.0, Some(ratios[2])), graded("r", 1.0, None), graded("r", 0.0, None)];
    let (obj, _dir) = open(&examples, config(None));
    let obj = obj.expect("the sequence-level estimator needs no tokenizer comparison");
    let distill = sequence_advantages(&ratios);
    let reward = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0];
    for e in 0..3 {
        let want = (reward[e] + distill[e]) as f32;
        assert!((obj.advantage(e).unwrap() - want).abs() < 1e-5, "example {e}: {} vs {want}", obj.advantage(e).unwrap());
    }
    assert_eq!((obj.advantage(3), obj.advantage(4)), (Some(0.5), Some(-0.5)), "a group without teacher scores keeps its reward advantage");
    assert_eq!(obj.stats().sequences_with_teacher_logprob, 3);
}

#[test]
fn a_group_whose_per_answer_scores_cannot_form_a_baseline_is_refused() {
    let mixed = [graded("d", 1.0, Some(-4.0)), graded("d", 0.0, None), graded("d", 0.5, Some(-2.0))];
    let why = open(&mixed, config(None)).0.err().expect("mixed").to_string();
    assert!(why.contains("teacher_scores") && why.contains("\"d\""), "{why}");
    let alone = [graded("d", 1.0, Some(-4.0)), graded("d", 0.0, None)];
    let why = open(&alone, config(None)).0.err().expect("one scored rollout").to_string();
    assert!(why.contains("teacher_scores"), "{why}");
}
