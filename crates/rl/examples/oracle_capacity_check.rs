// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! E1 from the continual-learning follow-up research: does a rank-8,
//! `wq,wk,wv,wo`-only LoRA adapter over the study's frozen base have enough
//! CAPACITY to represent all 12 study rules, independent of GRPO, sequential
//! interference, or the cue-independent-shortcut failure the 12-cycle
//! continual-learning study measured?
//!
//! The study's own "capacity oracle" (`continual::joint_oracle`) does not
//! answer this: it gives the pooled run `cycles * steps_per_cycle` TOTAL
//! steps over `cycles` rules, which is the same per-rule budget one cycle
//! gives one rule (240), not the ~48,000-per-rule budget this exact
//! architecture is already known (from this same base's own pretraining
//! fixture, 16 rules at 0.995) to need. Its 0.071 therefore confounds
//! capacity with per-rule sample budget.
//!
//! This program removes every other variable: no RL (teacher-forced SFT via
//! `rl::objective::mixture::Anchor` on the oracle's own known-correct
//! completions - `curriculum::target_of`), no sequential interference (all
//! 12 study rules pooled into one dataset, one training run), and a per-rule
//! budget matched to what the base's own pretraining needed (6000 steps,
//! batch 128, ~500 steps/rule) rather than GRPO's 240. If THIS run cannot
//! reach a high score on the same 192 probes the study scores against,
//! capacity genuinely is a binding constraint on the loop's own result. If
//! it can, capacity is not the constraint, and the loop's failure to
//! accumulate is a training-regime/curriculum problem, not a sizing one.
//!
//! Run it:
//! ```text
//! cargo run --release -p brain-rl --features qwen3 --example oracle_capacity_check
//! ```
//! Reuses the continual-learning study's cached frozen base if present and
//! fingerprint-matched (same cache dir, same fingerprint scheme as
//! `examples/continual_learning.rs` and `tests/continual_study.rs`);
//! otherwise builds its own copy. Budget: a few minutes to pretrain the base
//! (cached afterwards), then well under a minute for this program's own SFT
//! run and scoring pass.
//!
//! Swedish Embedded AB builds diagnostic harnesses that isolate ONE variable
//! at a time before recommending a fix - capacity, curriculum, and training
//! regime are three different problems with three different solutions, and
//! conflating them wastes an engineering team's time on the wrong one. If
//! your team needs expertise in diagnosing why a training loop plateaus, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use model::rollout::RolloutParams;
use model::serve::SampleParams;
use qwen3::config::{LoraCfg, QwenConfig};
use qwen3::model::Qwen;
use rl::curriculum::{self, ContentSplit, PositionCopyEnv, PositionCopyVerifier, Rule};
use rl::env::Environment;
use rl::improve;
use rl::objective::mixture::Anchor;

const LORA_RANK: u32 = 8;
const LORA_ALPHA: f32 = 16.0;
const LORA_TARGETS: [&str; 4] = ["wq", "wk", "wv", "wo"];

const STUDY_RULES: usize = 12;
const EVAL_PER_RULE: usize = 16;

const PRETRAIN_SEQS: usize = 40_000;
const PRETRAIN_STEPS: u32 = 6_000;
const PRETRAIN_BATCH: u32 = 128;
const PRETRAIN_LR: f32 = 3e-3;
const PRETRAIN_RULES: usize = 16;
const PRETRAIN_SEED: u64 = 20;

/// E1's own SFT budget for the 12 study rules pooled - matched to the
/// pretraining base's own recipe (same seqs-per-step density), not to
/// GRPO's 240-step/rule cycle budget. This IS the variable under test.
const SFT_SEQS: usize = 30_000;
const SFT_STEPS: u32 = 6_000;
const SFT_BATCH: u32 = 128;
const SFT_LR: f32 = 3e-3;
const SFT_SEED: u64 = 1;

fn lora_cfg() -> LoraCfg {
    LoraCfg { rank: LORA_RANK, alpha: LORA_ALPHA, targets: LORA_TARGETS.iter().map(|s| s.to_string()).collect() }
}

fn pretrain_config() -> QwenConfig {
    QwenConfig {
        vocab: curriculum::VOCAB,
        block_size: curriculum::SEQ_LEN as u32,
        max_position_embeddings: curriculum::SEQ_LEN as u32,
        n_layers: 2,
        d_model: 64,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        d_ff: 256,
        tie_embeddings: true,
        qk_norm: true,
        lora: None,
        ..QwenConfig::tiny()
    }
}

fn study_config() -> QwenConfig {
    QwenConfig { lora: Some(lora_cfg()), ..pretrain_config() }
}

/// Identical scheme to `examples/continual_learning.rs`'s own
/// `fixture_fingerprint` - same inputs, same cache, so this program reuses
/// that one's base (or vice versa) whenever they agree, and rebuilds its own
/// under a fingerprint mismatch rather than overwriting a fixture another
/// run may still be measuring against.
fn fixture_fingerprint() -> String {
    format!(
        "v1|{}|{}|rules={PRETRAIN_RULES}|seqs={PRETRAIN_SEQS}|steps={PRETRAIN_STEPS}|bs={PRETRAIN_BATCH}|lr={PRETRAIN_LR}|seed={PRETRAIN_SEED}|rec={}",
        model::ModelConfig::to_json(&pretrain_config()),
        model::ModelConfig::to_json(&study_config()),
        curriculum::RECORD_LEN
    )
}

fn prepare_base(out: &Path) -> PathBuf {
    let fingerprint = fixture_fingerprint();
    let shared = std::env::temp_dir().join("brain-rl-continual-base");
    let shared_base = shared.join("base.safetensors");
    if shared_base.exists() && std::fs::read_to_string(shared.join("fingerprint.txt")).map(|s| s == fingerprint).unwrap_or(false) {
        println!("base: reusing the cached pretrained fixture at {} (fingerprint matches)", shared_base.display());
        return shared_base;
    }

    let cache = out.join("base");
    let base = cache.join("base.safetensors");
    let stamp = cache.join("fingerprint.txt");
    if base.exists() && std::fs::read_to_string(&stamp).map(|s| s == fingerprint).unwrap_or(false) {
        println!("base: reusing {} (fingerprint matches)", base.display());
        return base;
    }

    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).expect("create base cache");
    let data_dir = cache.join("pretrain-data");
    let rules = Rule::pretrain_rules(PRETRAIN_RULES);
    println!("base: pretraining on {PRETRAIN_RULES} rules whose cues no study cycle ever uses ({PRETRAIN_STEPS} steps, batch {PRETRAIN_BATCH}) - a few minutes, cached afterwards");
    curriculum::write_pretrain_dataset(&rules, PRETRAIN_SEQS, PRETRAIN_SEED, &data_dir).expect("write pretrain dataset");
    let opts = model::FitOpts {
        steps: PRETRAIN_STEPS,
        batch_size: PRETRAIN_BATCH,
        block_size: curriculum::SEQ_LEN as u32,
        lr: PRETRAIN_LR,
        min_lr: PRETRAIN_LR / 10.0,
        warmup: 100,
        decay_iters: PRETRAIN_STEPS,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: 0,
        eval_batches: 0,
        seed: PRETRAIN_SEED,
        checkpoint_secs: 0,
        align_to_lines: true,
        ..model::FitOpts::default()
    };
    let started = std::time::Instant::now();
    let pretrained = cache.join("pretrained.safetensors");
    let (before, after) = model::fit::<Qwen>(&data_dir, pretrain_config(), &opts, Some(&pretrained)).expect("pretrain");
    // Zero-delta LoRA overlay: load the pretrained full-parameter model,
    // re-save under the study's LoRA config with lora_a/lora_b at their
    // fresh (zero-delta) init, base weights untouched. Mirrors
    // `continual::overlay_adapter` exactly (that fn is crate-private).
    let init = qwen3::init::init_weights(&study_config(), PRETRAIN_SEED);
    let mut merged = init;
    let c = checkpoint::load(pretrained.to_str().unwrap());
    for (name, data) in c.by_role("") {
        merged.insert(name, data);
    }
    let m = Qwen::new(study_config(), 1, curriculum::SEQ_LEN as u32, &merged);
    m.save(base.to_str().unwrap());
    std::fs::write(&stamp, &fingerprint).expect("write fingerprint");
    println!("base: pretrained in {:.1}s, loss {before:.3} -> {after:.3}", started.elapsed().as_secs_f64());
    base
}

fn greedy_params() -> RolloutParams {
    RolloutParams { max_new: curriculum::OUT_LEN, sample: SampleParams::greedy(), eos: None }
}

/// The same 192-probe SHAPE the study scores ACC against: 16 held-out probes
/// per study rule x 12 rules, drawn from `ContentSplit::Eval` (content the
/// training data - here or in the study - never used). Not the identical
/// seeds the study's own private `probe_seeds` draws (that function is
/// crate-private), but the identical rule set, split, and per-rule count -
/// what makes the ACC comparison fair is the content-space partition, not
/// the seed value.
fn probes() -> Vec<rl::env::Task> {
    (0..STUDY_RULES)
        .flat_map(|k| {
            let e = PositionCopyEnv::new(Rule::for_cycle(k), ContentSplit::Eval);
            (0..EVAL_PER_RULE).map(move |i| e.tasks(9_000_000 + (k as u64) * 10_000 + i as u64).into_iter().next().expect("one task"))
        })
        .collect()
}

fn main() {
    let out = std::env::temp_dir().join("brain-rl-oracle-capacity-check");
    std::fs::create_dir_all(&out).expect("create out dir");

    let base = prepare_base(&out);

    let rules: Vec<Rule> = (0..STUDY_RULES).map(Rule::for_cycle).collect();
    println!("E1: pooling all {STUDY_RULES} study rules into one teacher-forced SFT dataset ({SFT_SEQS} seqs, {SFT_STEPS} steps, batch {SFT_BATCH}) - no RL, no sequential interference, per-rule budget matched to the base's own pretraining density");
    let data_dir = out.join("sft-data");
    curriculum::write_pretrain_dataset(&rules, SFT_SEQS, SFT_SEED, &data_dir).expect("write sft dataset");

    let opts = model::FitOpts {
        steps: SFT_STEPS,
        batch_size: SFT_BATCH,
        block_size: curriculum::SEQ_LEN as u32,
        lr: SFT_LR,
        min_lr: SFT_LR / 10.0,
        warmup: 100,
        decay_iters: SFT_STEPS,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: 0,
        eval_batches: 0,
        seed: SFT_SEED,
        checkpoint_secs: 0,
        align_to_lines: true,
        // SEP's itos char (curriculum::line_aligned_meta maps token id `i`
        // to `'a' + i`; SEP = 1 -> 'b') - supervise only the completion
        // (t0,t1,t2), not the prompt, which the loss would otherwise spend
        // most of its gradient trivially predicting.
        mask_before: Some('b'),
        mask_per_line: true,
    };

    let (train, _val, batch_cfg, vocab) = model::load_dataset(&data_dir, &opts).expect("load sft dataset");
    assert_eq!(vocab, curriculum::VOCAB, "sft dataset vocab must match the study's vocab");
    let anchor = Anchor::new(train, batch_cfg);

    let c = checkpoint::load(base.to_str().unwrap());
    let cfg = <Qwen as model::Model>::Config::from_json(&c.header["config"]);
    let init = c.by_role("");
    let model = Qwen::new(cfg, opts.batch_size, opts.block_size, &init);

    let started = std::time::Instant::now();
    let trained = out.join("e1-trained.safetensors");
    let (initial, last) = model::fit_with(model, anchor, &opts, Some(&trained)).expect("sft fit");
    println!("E1: trained in {:.1}s, loss {initial:.3} -> {last:.3}", started.elapsed().as_secs_f64());

    let tasks = probes();
    let (scores, _) = improve::score_checkpoint::<Qwen>(&trained, &tasks, &PositionCopyVerifier, &greedy_params());
    let acc: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

    println!(
        "\nE1 RESULT: capacity-only ACC {acc:.3} over {} probes ({STUDY_RULES} rules x {EVAL_PER_RULE} probes), \
         no RL, no sequential interference, {SFT_STEPS} pooled steps.\n\
         Compare against the 12-cycle GRPO study's ACC 0.271 (same probe shape, different seeds) and the joint \
         oracle's 0.071 (same total budget, but split {STUDY_RULES} ways at 240 steps/rule, not pooled at this budget).\n\
         PREDICTED (from the continual-learning follow-up research): ACC >= 0.85 means capacity is NOT the binding \
         constraint on the loop's own failure to accumulate - the constraint is the training regime/curriculum \
         (the cue-independent shortcut under replay_frac=0.0, and GRPO's ~237x under-supervision per rule at \
         240 steps/group-2). ACC below ~0.5 would falsify that and mean rank-8 attention-only LoRA genuinely cannot \
         represent {STUDY_RULES} cue-conditioned rules at this budget, which flips the diagnosis back toward capacity.",
        tasks.len()
    );
}
