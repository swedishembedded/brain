// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does [`modernbert::LayaDecision`]'s training loop actually LEARN?
//!
//! `crates/gradcheck` already proves the backward is the right derivative and
//! that an AdamW step moves the parameters it should. Neither of those says
//! the assembled loop - pack a real `(state, question)` sequence, score it,
//! turn the scores into a policy gradient, seed the head's backward, step -
//! fits anything. A loop can be individually correct at every stage and still
//! be wired so that, say, the option order and the target disagree, and the
//! only symptom is a model that stays at chance.
//!
//! So this trains a DELIBERATELY TINY, randomly initialized model (4 layers,
//! `d_model` 64) on a small arbitrary label mapping and asserts it goes from
//! chance to fitted. Tiny and random is the point: there is no pretrained
//! prior to lean on, so the only way accuracy can move is if the loop is
//! genuinely learning from the examples.
//!
//! **The convergence run trains the TRUNK as well as the head, and it has
//! to.** A randomly initialized trunk cannot be frozen and still learn this
//! task, for a reason specific to the architecture rather than to the
//! optimizer: every option marker is the SAME `[MASK]` token, ModernBERT has
//! no learned position table (position reaches the model only through RoPE,
//! inside attention), and at random init attention is near-uniform - so all
//! `k` markers arrive at the head as nearly the same vector and no head can
//! tell the options apart. Measured, not assumed: at `d_model` 64 with a
//! frozen random trunk the four option logits come back as
//! `[0.030221, 0.030220, 0.030218, 0.030210]` and 1200 steps move accuracy
//! from 2/8 to 3/8, while unfreezing the trunk reaches 8/8 within 100 steps.
//! That is also the honest reason `Training::HeadOnly` is nevertheless the
//! right default for the REAL checkpoint, whose trunk already separates them.
//!
//! **The option order here is FIXED, and that is a scope statement rather
//! than a convenience.** Real training shuffles it every step
//! (`rl_common.py::encode_record`, and `brain::DecisionPipeline::
//! train_choices`'s own `OptionSampler` on both arms), which turns the task
//! from "map a state to a slot" into "read four option texts and pick the
//! matching one" - and a 64-wide 4-layer model trained from scratch on eight
//! examples does not get there (measured: 225 steps leave it at exactly
//! `ln 4`, a perfectly uniform report, which IS the optimal answer for a
//! model that cannot tell the options apart). Two other tests cover what
//! this one therefore cannot: `tests/build_sequence.rs` gates the permuted
//! packing itself against the real Python reference (including the shrink
//! path and the identity permutation), and the SDK's own
//! `real_laya_checkpoint_head_training_improves_held_out_accuracy` measures
//! shuffled-option training on the real trunk, which can learn it.
//!
//! It needs the real Laya TOKENIZER (a 11 MB `tokenizer.json`, not the 843 MB
//! checkpoint) because `build_sequence` is defined in terms of a real
//! byte-level BPE and its special-token ids; the weights are freshly seeded
//! here. Skips cleanly when the tokenizer is absent, per this workspace's
//! convention.

use modernbert::config::{LayerAttn, ModernBertConfig};
use modernbert::laya::LayaConfig;
use modernbert::sequence::{Question, State};
use modernbert::{LayaDecision, Training};
use rlcd::reinforce::{rlcd_loss, RlcdObjective};

const OPTIONS: [&str; 4] = ["alpha", "beta", "gamma", "delta"];
const INSTRUCTIONS: &str = "which bucket does this record belong in";

/// `(state, gold option index)`. The mapping from the text to the option is
/// ARBITRARY on purpose - "wind" is `delta`, not something a language model
/// could guess - so a model that has not learned from these examples cannot
/// do better than chance on them.
const TRAIN: &[(&str, usize)] = &[
    ("the wind is picking up outside", 3),
    ("heavy rain all afternoon", 3),
    ("simmer the sauce for twenty minutes", 1),
    ("preheat the oven before baking", 1),
    ("the invoice is thirty days overdue", 0),
    ("quarterly revenue beat the forecast", 0),
    ("he scored in the final minute", 2),
    ("the match went to extra time", 2),
];

fn tokenizer_path() -> Option<String> {
    let dir = brain_testutil::model_dir("convaiinnovations/laya")?;
    let p = format!("{dir}/tokenizer/tokenizer.json");
    std::path::Path::new(&p).exists().then_some(p)
}

/// A small random model over the REAL vocabulary. `max_positions` has to
/// cover `max_len`, and the local-attention window has to be smaller than a
/// span or the windowed layers are vacuous.
fn small_model(tok_path: &str, enc_seed: u64, head_seed: u64, mode: Training) -> LayaDecision {
    let tok = data::qwen_tokenizer::QwenBpe::from_file(tok_path).expect("load Laya tokenizer.json");
    let mut cfg = ModernBertConfig {
        vocab: tok.vocab_size() as u32,
        d_model: 64,
        n_layers: 4,
        n_heads: 4,
        d_ff: 128,
        max_positions: 192,
        eps: 1e-5,
        layer_types: (0..4).map(|l| if l % 2 == 0 { LayerAttn::Full } else { LayerAttn::Local }).collect(),
        rope_theta_full: 160_000.0,
        rope_theta_local: 10_000.0,
        window: 16,
        cls_token_id: 0,
        sep_token_id: 0,
        pad_token_id: 0,
        mask_token_id: 0,
    };
    cfg.cls_token_id = tok.special_id("[CLS]").expect("[CLS]");
    cfg.sep_token_id = tok.special_id("[SEP]").expect("[SEP]");
    cfg.pad_token_id = tok.special_id("[PAD]").expect("[PAD]");
    cfg.mask_token_id = tok.special_id("[MASK]").expect("[MASK]");

    let laya_cfg = LayaConfig::new(cfg.d_model);
    let enc_init = modernbert::init::init_weights(&cfg, enc_seed);
    let head_init = modernbert::init::init_weights_laya(&laya_cfg, head_seed);
    let gpu = gpu_core::testgpu::dev(modernbert::kern::PIPELINES);
    LayaDecision::new_on(gpu, cfg, laya_cfg, tok, 160, 64, &enc_init, &head_init, mode)
}

fn question() -> Question {
    Question::Choice {
        ins: INSTRUCTIONS.into(),
        options: OPTIONS.iter().map(|o| ((*o).to_string(), None)).collect(),
    }
}

/// Accuracy over `set`, in the canonical option order.
fn accuracy(m: &mut LayaDecision, set: &[(&str, usize)]) -> f32 {
    let q = question();
    let mut hit = 0;
    for (text, gold) in set {
        let (logits, _) = m.score(&State::Str((*text).into()), &q, None).expect("score");
        let arg = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("logits");
        if arg == *gold {
            hit += 1;
        }
    }
    hit as f32 / set.len() as f32
}

#[test]
#[ignore = "slow: ~200s of real training - run via `make test/slow`"]
fn a_tiny_laya_model_trains_from_chance_to_fitted() {
    let Some(tok_path) = tokenizer_path() else {
        brain_testutil::skip("Laya tokenizer absent - run `brain pull convaiinnovations/laya`");
        return;
    };
    let mut m = small_model(&tok_path, 0x1A9A_2026, 0xA10A_0001, Training::HeadAndTrunk);
    assert!(!m.trunk_frozen(), "this run has to move the trunk - see the module doc");

    let before = accuracy(&mut m, TRAIN);
    println!("accuracy before training: {before:.3} (chance {:.3})", 1.0 / OPTIONS.len() as f32);

    let cfg = RlcdObjective::default();
    let q = question();
    let mut rng = data::rng::Rng::new(0xD1CE);
    let steps = 200usize;
    let (mut first_ten, mut last_ten) = (0.0f32, 0.0f32);

    for step in 0..steps {
        let (text, gold) = TRAIN[(rng.next_u64() % TRAIN.len() as u64) as usize];
        // Fixed option order - see the module doc for why this test cannot
        // shuffle and what covers the shuffled path instead.
        let target = rlcd::proper::hard_target(OPTIONS.len(), gold);
        // sigma anneals over the run, the shape both published Laya runs use.
        let obj = cfg.at(rlcd::reinforce::anneal(0.4, 0.1, step as f32 / steps as f32));
        let l = m
            .train_step_with(&State::Str(text.into()), &q, None, 1e-3, 3e-3, |scores| {
                rlcd_loss(scores, &target, false, &obj, &mut rng)
            })
            .expect("train step");
        assert!(l.is_finite(), "loss went non-finite at step {step}: {l}");
        if step < 10 {
            first_ten += l / 10.0;
        }
        if step >= steps - 10 {
            last_ten += l / 10.0;
        }
    }

    let after = accuracy(&mut m, TRAIN);
    println!("loss {first_ten:.4} -> {last_ten:.4};  accuracy {before:.3} -> {after:.3}");

    assert!(last_ten < first_ten, "loss did not fall: {first_ten:.4} -> {last_ten:.4}");
    // Chance is 0.25 on four options. Fitting eight arbitrary examples is
    // easy IF the loop is wired correctly and impossible if it is not, which
    // is exactly the discrimination this test is for.
    assert!(
        after >= 0.875,
        "the model did not fit its own training set: {before:.3} -> {after:.3}"
    );
    assert!(after > before, "accuracy did not improve: {before:.3} -> {after:.3}");
}

/// A saved head must reload into a fresh model and answer IDENTICALLY.
/// Without this, "training works" and "you can keep what you trained" are two
/// different claims and only the first is checked.
#[test]
fn a_saved_head_reloads_and_reproduces_the_same_logits() {
    let Some(tok_path) = tokenizer_path() else {
        brain_testutil::skip("Laya tokenizer absent - run `brain pull convaiinnovations/laya`");
        return;
    };
    // Head-only, and the trunk seed is what the reloaded model must share:
    // a head reattached to a DIFFERENT trunk is a different model, which is
    // exactly what `save_head` refuses to pretend otherwise about.
    const TRUNK: u64 = 0x5A7E_2026;
    let mut m = small_model(&tok_path, TRUNK, 0xA10A_0001, Training::HeadOnly);
    let q = question();
    let cfg = RlcdObjective::default();
    let mut rng = data::rng::Rng::new(11);
    for _ in 0..12 {
        let (text, gold) = TRAIN[(rng.next_u64() % TRAIN.len() as u64) as usize];
        let target = rlcd::proper::hard_target(OPTIONS.len(), gold);
        m.train_step_with(&State::Str(text.into()), &q, None, 0.0, 3e-2, |scores| {
            rlcd_loss(scores, &target, false, &cfg, &mut rng)
        })
        .expect("train step");
    }
    let (want, want_act) = m.score(&State::Str(TRAIN[0].0.into()), &q, None).expect("score");

    let path = std::env::temp_dir()
        .join(format!("brain-laya-head-{}.safetensors", std::process::id()))
        .to_str()
        .expect("temp path")
        .to_string();
    m.save_head(&path).expect("save_head");

    // A FRESH model whose HEAD is seeded differently, so anything it
    // reproduces came from the file rather than from a coincidence of
    // initialization. Same trunk, since the file does not carry one.
    let mut reloaded = small_model(&tok_path, TRUNK, 0xFFFF_0001, Training::HeadOnly);
    let head = LayaDecision::read_head_file(&path).expect("read head file");
    let snapshot: Vec<(String, Vec<f32>)> = head.into_iter().collect();
    assert!(!snapshot.is_empty(), "the saved head has no tensors");
    reloaded.set_head_weights(&snapshot);
    let (got, got_act) = reloaded.score(&State::Str(TRAIN[0].0.into()), &q, None).expect("score");
    let _ = std::fs::remove_file(&path);

    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() <= 1e-4, "option logit[{i}] {a} vs {b} after reload");
    }
    for (i, (&a, &b)) in got_act.iter().zip(&want_act).enumerate() {
        assert!((a - b).abs() <= 1e-4, "act logit[{i}] {a} vs {b} after reload");
    }
}

/// `save_head` writes the HEAD only, so it must REFUSE on a model whose trunk
/// was trained rather than write a file that silently reattaches the head to
/// the published trunk.
#[test]
fn saving_a_head_whose_trunk_moved_is_refused() {
    let Some(tok_path) = tokenizer_path() else {
        brain_testutil::skip("Laya tokenizer absent - run `brain pull convaiinnovations/laya`");
        return;
    };
    let m = small_model(&tok_path, 3, 4, Training::HeadAndTrunk);
    let err = m.save_head("/dev/null").expect_err("should refuse");
    assert!(err.contains("trunk was trained"), "unhelpful refusal: {err}");
}
