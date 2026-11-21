// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning whose loss and gradients are checked against an independent reference
// for its clients. If your team needs expertise in verifying training numerics then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Numerical conformance of supervised fine-tuning's masked cross-entropy against an independent reference.
//!
//! Spec, for records `ids` with a per-token supervision mask:
//!
//! ```text
//! loss = - (1 / N) * sum over records r, positions j >= 1 with supervised[r][j] of log p(ids[r][j] | ids[r][..j]),
//! N    = the number of supervised tokens of the batch
//! ```
//!
//! so position `t` predicts token `t + 1`, a prompt or tool token never enters the loss or the gradient, a record's
//! padding is no target and no part of `N`, and a row attends to its own record only, from position 0.
//!
//! `golden/sft_ce.json` holds two chat-shaped records (a prompt, an answer, a tool result inside the record, a
//! second answer; the second record shorter than the row), the weights of a one-layer Qwen3-shaped decoder, and what
//! the spec gives for them, computed in float64 by `golden/sft_ce_generate.py` with torch autograd over a model it
//! writes from the published architecture (nothing of brain is run). The test drives the trainer's own path: the
//! records become rows through the dataset's example layout (`TokenDataset::new_examples`), and the model trains on
//! them through `Model::set_batch`, `forward` and `backward`.
//!
//! The comparison is staged in the order the step computes, and a failure names the earliest stage that diverged:
//! the per-position log-probabilities of the forward, the supervised loss, then each tensor's gradient from the head
//! back to the embedding. The band is the binary32 agreement bound of the step's longest chain of rounded operations
//! (`brain_testutil::training::step_bound`), stated beside the measured difference.
//!
//! The suite's power is shown, not assumed: each mutant below is a defect of the batch the trainer hands the model
//! (a label shifted by one, prompt or tool tokens supervised, padding counted in the denominator, records packed
//! into one row without a boundary), and each must fail the comparison.

use std::collections::HashMap;

use data::loader::{BatchConfig, TokenDataset};
use model::{Batch, Model};
use qwen3::{Qwen, QwenConfig, IGNORE};
use serde::Deserialize;

const FIXTURE: &str = include_str!("../golden/sft_ce.json");
/// The id a row is padded with, as the trainer's loader pads (`model::train`'s example rows).
const PAD: u32 = 0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    #[allow(dead_code)]
    provenance: serde_json::Value,
    config: Config,
    records: Vec<Record>,
    weights: HashMap<String, Vec<f32>>,
    expected: Expected,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    vocab: u32,
    d_model: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    d_ff: u32,
    rope_theta: f32,
    rms_eps: f32,
    block: u32,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Record {
    ids: Vec<u32>,
    supervised: Vec<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    supervised_tokens: usize,
    loss: f64,
    /// Per record, `log p(ids[j] | ids[..j])` for `j = 1..len`.
    logprobs: Vec<Vec<f64>>,
    grads: HashMap<String, Vec<f64>>,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("golden/sft_ce.json parses")
}

fn qwen_config(c: &Config) -> QwenConfig {
    QwenConfig {
        vocab: c.vocab,
        block_size: c.block,
        n_layers: c.n_layers,
        d_model: c.d_model,
        n_heads: c.n_heads,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        d_ff: c.d_ff,
        rope_theta: c.rope_theta,
        rms_eps: c.rms_eps,
        max_position_embeddings: c.block,
        tie_embeddings: true,
        qk_norm: true,
        attn_bias: false,
        lora: None,
        rope_scaling: None,
    }
}

/// The batch the trainer builds from `records`, one record a row: the dataset's own example layout.
fn trainer_batch(records: &[Record], block: u32) -> (Vec<u32>, Vec<u32>) {
    let (mut ids, mut mask, mut starts) = (Vec::new(), Vec::new(), Vec::new());
    for r in records {
        starts.push(ids.len());
        ids.extend(&r.ids);
        mask.extend(&r.supervised);
    }
    let cfg = BatchConfig { batch_size: records.len(), block_size: block as usize, ..Default::default() };
    let ds = TokenDataset::new_examples(ids, mask, &starts, PAD, &cfg).expect("every record fits a row");
    let picks: Vec<Option<usize>> = (0..records.len()).map(Some).collect();
    let (x, y) = ds.example_batch(&cfg, &picks).expect("an example-indexed split");
    (x, model::train::targets_to_u32(&y))
}

/// Every position of every record a target (the next token), padding none: the forward's log-probabilities at all
/// of a record's positions, whatever its mask.
fn every_position(records: &[Record], block: u32) -> Vec<u32> {
    let t = block as usize;
    let mut y = vec![IGNORE; records.len() * t];
    for (r, rec) in records.iter().enumerate() {
        for j in 1..rec.ids.len() {
            y[r * t + j - 1] = rec.ids[j];
        }
    }
    y
}

/// The longest chain of rounded binary32 operations in one forward and backward of the fixture's step: every
/// reduction a value passes through (norms, projections, attention scores and their mix, the head, the log-sum-exp,
/// the batch sum), forward and back.
fn chain(c: &Config, rows: usize) -> usize {
    let (d, ff, q, hd, t) = (c.d_model as usize, c.d_ff as usize, (c.n_heads * c.head_dim) as usize, c.head_dim as usize, c.block as usize);
    let layer = d + d + hd + hd + 2 * t + q + d + d + ff;
    2 * (c.n_layers as usize * layer + d + d + c.vocab as usize + rows * t)
}

/// The tensors in the order the backward produces their gradients: head and final norm first, embedding last.
fn backward_order(cfg: &QwenConfig) -> Vec<String> {
    let mut names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| n != "tok.weight").collect();
    names.reverse();
    names.push("tok.weight".to_string());
    names
}

/// One stage's comparison: its name, the measured difference and the band it was held to.
struct Stage {
    name: String,
    off: f64,
    band: f64,
}

impl Stage {
    /// Past the band, or not a number (a NaN difference is a divergence, never a pass).
    fn diverged(&self) -> bool {
        self.off.is_nan() || self.off > self.band
    }
}

fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    let diff: f64 = got.iter().zip(want).map(|(a, b)| (f64::from(*a) - b).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = want.iter().map(|b| b * b).sum::<f64>().sqrt();
    diff / norm.max(f64::MIN_POSITIVE)
}

/// Run the fixture's step on `x`/`y` (the trainer's batch, or a mutant of it) and compare every stage with the
/// reference, in the order the step computes them. `denominator` overrides the loss denominator the model takes from
/// the batch (a mutant that counts padding).
fn stages(fx: &Fixture, x: &[u32], y: &[u32], denominator: Option<f32>) -> Vec<Stage> {
    let cfg = qwen_config(&fx.config);
    let rows = fx.records.len();
    let block = fx.config.block;
    let m = Qwen::new(cfg.clone(), rows as u32, block, &fx.weights);
    let bound = brain_testutil::training::step_bound(chain(&fx.config, rows));
    let mut out = Vec::new();

    // Stage 1, the forward: log p(ids[j] | ids[..j]) at every position of every record, read from the training
    // forward with every next token a target.
    Model::set_batch(&m, Batch::Lm { tokens: x, targets: &every_position(&fx.records, block) });
    Model::forward(&m);
    let lp = m.batch_token_logprobs();
    let (mut off, mut worst) = (0f64, String::new());
    for (r, want) in fx.expected.logprobs.iter().enumerate() {
        for (j, w) in want.iter().enumerate() {
            let got = f64::from(lp[r * block as usize + j]);
            let d = (got - w).abs() / w.abs().max(1.0);
            if d > off {
                off = d;
                worst = format!(" (worst: record {r}, token {})", j + 1);
            }
        }
    }
    out.push(Stage { name: format!("forward log-probabilities{worst}"), off, band: bound });

    // Stage 2, the loss over the supervised tokens of the batch as given.
    Model::set_batch(&m, Batch::Lm { tokens: x, targets: y });
    if let Some(n) = denominator {
        m.set_loss_denominator(n);
    }
    m.zero_grads();
    let loss = f64::from(Model::forward(&m));
    out.push(Stage { name: format!("supervised loss ({loss} against {})", fx.expected.loss), off: (loss - fx.expected.loss).abs() / fx.expected.loss.abs().max(1.0), band: bound });

    // Stage 3, each gradient, head first.
    Model::backward(&m);
    m.poll_wait();
    for name in backward_order(&cfg) {
        let want = &fx.expected.grads[&name];
        out.push(Stage { name: format!("gradient of {name}"), off: rel_l2(&m.read_grad(&name), want), band: bound });
    }
    out
}

/// The earliest stage that diverged, as a failure message, or `None` when every stage is within its band.
fn earliest_divergence(stages: &[Stage]) -> Option<String> {
    stages.iter().find(|s| s.diverged()).map(|s| format!("earliest divergence: {}: off by {:e}, past the binary32 band {:e}", s.name, s.off, s.band))
}

/// The trainer's step on the golden records matches the independent reference at every stage: the forward's
/// log-probabilities, the masked loss (7 supervised tokens of 13, padding excluded) and every gradient.
#[test]
fn the_masked_cross_entropy_step_matches_the_independent_reference() {
    let fx = fixture();
    let (x, y) = trainer_batch(&fx.records, fx.config.block);
    assert_eq!(y.iter().filter(|&&t| t != IGNORE).count(), fx.expected.supervised_tokens, "the trainer supervises the records' supervised tokens");
    let s = stages(&fx, &x, &y, None);
    for st in &s {
        eprintln!("{:<60} off {:e} band {:e}", st.name, st.off, st.band);
    }
    if let Some(why) = earliest_divergence(&s) {
        panic!("{why}");
    }
}

/// A mutant of the trainer's batch, and the stage it must be caught at or before.
fn assert_killed(what: &str, fx: &Fixture, x: &[u32], y: &[u32], denominator: Option<f32>) {
    let s = stages(fx, x, y, denominator);
    let why = earliest_divergence(&s).unwrap_or_else(|| panic!("mutant {what:?} survived: every stage stayed within its band"));
    eprintln!("mutant {what:?} killed: {why}");
}

/// Mutant: labels shifted by one, so position `t` is trained on token `t + 2` (or on itself).
#[test]
fn a_label_shifted_by_one_is_caught() {
    let fx = fixture();
    let (x, y) = trainer_batch(&fx.records, fx.config.block);
    let t = fx.config.block as usize;
    let late: Vec<u32> = y.chunks(t).flat_map(|row| row[1..].iter().copied().chain([IGNORE])).collect();
    assert_killed("targets one position late", &fx, &x, &late, None);
    let early: Vec<u32> = y.chunks(t).flat_map(|row| [IGNORE].into_iter().chain(row[..t - 1].iter().copied())).collect();
    assert_killed("targets one position early", &fx, &x, &early, None);
    let own: Vec<u32> = x.iter().zip(&y).map(|(&xi, &yi)| if yi == IGNORE { IGNORE } else { xi }).collect();
    assert_killed("each position trained on its own token", &fx, &x, &own, None);
}

/// Mutant: the prompt and tool tokens enter the loss (the supervision mask ignored).
#[test]
fn prompt_and_tool_tokens_in_the_loss_are_caught() {
    let fx = fixture();
    let unmasked: Vec<Record> = fx.records.iter().map(|r| Record { ids: r.ids.clone(), supervised: (0..r.ids.len()).map(|j| j > 0).collect() }).collect();
    let (x, y) = trainer_batch(&unmasked, fx.config.block);
    assert_killed("every token supervised", &fx, &x, &y, None);
    // Only the tool result inside the first record leaks.
    let mut tool = fx.records.clone();
    tool[0].supervised[5] = true;
    let (x, y) = trainer_batch(&tool, fx.config.block);
    assert_killed("the tool result supervised", &fx, &x, &y, None);
}

/// Mutant: padding counted, as a target (the pad id predicted) or in the denominator.
#[test]
fn padding_in_the_loss_or_its_denominator_is_caught() {
    let fx = fixture();
    let (x, y) = trainer_batch(&fx.records, fx.config.block);
    let t = fx.config.block as usize;
    let len = fx.records[1].ids.len();
    let mut pad_targets = y.clone();
    for p in len - 1..t {
        pad_targets[t + p] = PAD;
    }
    assert_killed("padding a target", &fx, &x, &pad_targets, None);
    let padded_count = (fx.expected.supervised_tokens + (t - len)) as f32;
    assert_killed("padding in the denominator", &fx, &x, &y, Some(padded_count));
}

/// Mutant: the records packed into one row back to back, the second continuing the first's positions and attending
/// to it, as packing without a record boundary does.
#[test]
fn records_attending_across_a_boundary_are_caught() {
    let fx = fixture();
    let (x, y) = trainer_batch(&fx.records, fx.config.block);
    let t = fx.config.block as usize;
    // Row 1 now holds the tail of record 0 before record 1, so record 1 sits at later positions and sees record 0;
    // its targets move with it. Row 0 is unchanged.
    let (r0, r1) = (&fx.records[0], &fx.records[1]);
    let lead = t - r1.ids.len();
    let mut x2 = x.clone();
    let mut y2 = y.clone();
    for p in 0..t {
        x2[t + p] = if p < lead { r0.ids[r0.ids.len() - lead + p] } else { r1.ids[p - lead] };
        y2[t + p] = if p >= lead && p + 1 < t && r1.supervised[p + 1 - lead] { r1.ids[p + 1 - lead] } else { IGNORE };
    }
    assert_eq!(y2.iter().filter(|&&v| v != IGNORE).count(), fx.expected.supervised_tokens, "the same supervised tokens");
    assert_killed("record 1 packed after record 0 without a boundary", &fx, &x2, &y2, None);
}
