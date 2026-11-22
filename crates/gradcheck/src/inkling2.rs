// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness gates for models whose layers read their own earlier states and
// whose core runs several times per token, for its clients. If your team needs expertise in proving such a backward
// exact then you can procure our services by sending an email to info@swedishembedded.com.

//! Gradient checks of Inkling 2's training graph on the tiny looped configuration
//! ([`inkling2::train::tiny_config`]): a core of two depth-attention blocks run three times, messages between passes,
//! sliding and full layers, routed experts.
//!
//! What they exist for: a core parameter is read once per pass and the analytic gradient must be the sum over the
//! passes; a retained block state is read by several later depth attentions, the next pass's messages and the readout,
//! and its gradient must reach the partial it was retained from; the message queries are gathered from two published
//! tensors and their gradient scattered back. A backward that dropped one pass, one consumer or one row fails the
//! per-entry checks ([`check_inkling2_shared`]), which the directional check ([`check_inkling2`]) cannot be trusted to
//! catch (knowledge #187).

use std::collections::HashMap;

use inkling2::train::{init_weights, tiny_config, Inkling2Train, TrainConfig};
use model::Model;

use crate::{directional_check, elementwise_check_at, Report};

/// Rows and tokens of the checked batch: longer than the window and the relative extents, two rows.
const B: u32 = 2;
const T: u32 = 6;

/// Set the checked batch and pin the routing its forward takes: a finite-difference step must not flip a top-k
/// selection (the loss is not differentiable there), and with the selection replayed the router is still
/// differentiated through the weights it gives the pinned experts.
fn set_batch(m: &Inkling2Train, vocab: u32) {
    set_rows(m, vocab);
    let _ = m.forward();
    let taken = m.taken_routing().expect("the forward routed");
    model::training::TokenObjectives::set_routing_replay(m, Some(std::sync::Arc::new(taken))).expect("the model replays its own routing");
    set_rows(m, vocab);
}

fn set_rows(m: &Inkling2Train, vocab: u32) {
    let n = (B * T) as usize;
    let x: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 1) % vocab).collect();
    // Next-token targets, the last position of each row and one more unsupervised.
    let y: Vec<u32> = (0..n).map(|i| if i % T as usize == T as usize - 1 || i == 3 { model::IGNORE } else { (i as u32 * 5 + 2) % vocab }).collect();
    m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
}

fn full_model(seed: u64) -> (Inkling2Train, TrainConfig) {
    let tc = tiny_config();
    let init = init_weights(&tc.cfg, seed);
    let m = Inkling2Train::new(tc.clone(), B, T, &init);
    set_batch(&m, tc.cfg.valid_vocab);
    (m, tc)
}

/// Every parameter of a full fine-tune, by directional finite differences.
pub fn check_inkling2(seed: u64) -> Report {
    let (m, _) = full_model(seed);
    directional_check(&m, 1e-3, 6, seed ^ 0x1A7)
}

/// Every parameter of the shared core (the layers the passes revisit) and the message queries, entry by entry at a
/// strided subset of `per_tensor` entries (the first, the last and evenly spaced ones between).
pub fn check_inkling2_shared(seed: u64, per_tensor: usize) -> Report {
    let (m, tc) = full_model(seed);
    let core = tc.cfg.looped.expect("the tiny config is looped");
    let shared: Vec<String> = m
        .param_names()
        .into_iter()
        .filter(|n| n.contains("cross_pass") || (core.start..core.end).any(|l| n.starts_with(&format!("{}layers.{l}.", inkling2::names::TEXT_PREFIX)) || n.ends_with(&format!("_queries.{}", l.saturating_sub(1))) && n.contains("attn_queries") || n.ends_with(&format!("mlp_queries.{l}"))))
        .collect();
    let mut checks = Vec::new();
    for name in shared {
        let len = m.read_weight(&name).len();
        let picks: Vec<usize> = if len <= per_tensor { (0..len).collect() } else { (0..per_tensor).map(|i| i * (len - 1) / (per_tensor - 1)).collect() };
        checks.extend(elementwise_check_at(&m, &name, 1e-3, &picks).checks);
    }
    Report { checks }
}

/// A LoRA build over the frozen tiny base (every linear and merged site; the float expert banks take no adapter):
/// every adapter tensor, by directional finite differences, with `B` moved off its zero start so `A`'s gradient is not
/// trivially zero.
pub fn check_inkling2_lora(seed: u64) -> Report {
    let base = tiny_config();
    let targets: Vec<String> = inkling2::train::DEFAULT_TARGETS.iter().filter(|t| !t.starts_with("w13") && !t.starts_with("w2")).map(|t| t.to_string()).collect();
    let lora = model::adapter::device::LoraCfg::new(2, 4.0, targets);
    let tc = base.with_lora(lora).expect("every target is a site");
    let mut init: HashMap<String, Vec<f32>> = init_weights(&tc.cfg, seed);
    let mut rng = data::rng::Lcg::new(seed ^ 0x2B);
    for (name, numel) in tc.adapter_params() {
        init.insert(name, (0..numel).map(|_| rng.scaled(0.3)).collect());
    }
    let m = Inkling2Train::new(tc.clone(), B, T, &init);
    set_batch(&m, tc.cfg.valid_vocab);
    directional_check(&m, 1e-3, 6, seed ^ 0x3C)
}
