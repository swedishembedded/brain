// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training an adapter over a frozen base held packed (QLoRA and its bf16, FP8 and int8 siblings).
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, with adapters over a 4-bit base. If your team needs expertise in quantized-base training of
//! language models then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The question is whether the packed tier is executed right, not how much quality it costs: the adapter's loss
//! and gradients over the packed base must equal those of an f32 build over the values the tier's codec decodes
//! to (`Weight::round_trip`, held equal to the device's own decode by `model`'s tests). Any difference left is
//! the order of f32 accumulation.

use std::collections::HashMap;

use gpu_core::select::Dtype;
use model::adapter::device::LoraCfg;
use model::conformance_grad::randomize_adapters;
use model::ops::{TierPolicy, Weight};
use model::{Batch, Model};
use ricursive::train::default_targets;
use ricursive::{Ripple, TrainConfig, TrainOpts};

const ROWS: u32 = 2;
const SEQ: u32 = 6;

fn adapter_config() -> TrainConfig {
    TrainConfig::tiny().with_lora(LoraCfg::new(2, 4.0, default_targets()))
}

fn batch(vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let n = (ROWS * SEQ) as usize;
    let x: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 1) % vocab).collect();
    let mut y: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 9) % vocab).collect();
    y[5] = ricursive::train::IGNORE;
    y[11] = ricursive::train::IGNORE;
    (x, y)
}

/// Loss and the gradient of every trained tensor, on the batch.
fn loss_and_grads(m: &Ripple, x: &[u32], y: &[u32]) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(Batch::Lm { tokens: x, targets: y });
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    m.poll_wait();
    (loss, m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect())
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt()
}

/// `||a - b|| / ||b||` per tensor, the worst over the model, with its name.
fn worst_relative(got: &[(String, Vec<f32>)], want: &[(String, Vec<f32>)]) -> (f64, String) {
    let mut worst = (0.0, String::new());
    for ((name, g), (_, w)) in got.iter().zip(want) {
        let diff: Vec<f32> = g.iter().zip(w).map(|(a, b)| a - b).collect();
        let rel = norm(&diff) / norm(w).max(1e-12);
        if rel > worst.0 {
            worst = (rel, name.clone());
        }
    }
    worst
}

/// The base weights with every linear replaced by what `tier` decodes it to.
fn decoded(cfg: &TrainConfig, weights: &HashMap<String, Vec<f32>>, tier: Dtype) -> HashMap<String, Vec<f32>> {
    let mut out = weights.clone();
    for (name, n, k) in ricursive::weights::all_linears_of(&cfg.model) {
        out.insert(name.clone(), Weight::round_trip(&weights[&name], n, k, tier));
    }
    out
}

fn packed_opts(tier: Dtype) -> TrainOpts {
    TrainOpts { tiers: TierPolicy::uniform(tier), ..TrainOpts::default() }
}

/// Spec: an adapter trains over a base held at bf16, FP8 or NF4 exactly as it does over an f32 base holding the
/// decoded values; the int8 tier too wherever the backend runs it.
#[test]
fn an_adapter_over_a_packed_base_trains_like_one_over_its_decoded_values() {
    let cfg = adapter_config();
    let weights = cfg.tiny_weights(5);
    let (x, y) = batch(cfg.model.vocab_size as u32);
    for tier in [Dtype::BF16, Dtype::F8E4M3, Dtype::NF4, Dtype::I8] {
        let packed = match Ripple::build(cfg.clone(), ROWS, SEQ, &weights, &packed_opts(tier), true) {
            Ok(m) => m,
            Err(e) if tier == Dtype::I8 && e.contains("holds that tier as") => {
                brain_testutil::skip_unavailable("int8 packed base: this backend has no packed-int8 product");
                continue;
            }
            Err(e) => panic!("{tier:?}: {e}"),
        };
        let reference = Ripple::build(cfg.clone(), ROWS, SEQ, &decoded(&cfg, &weights, tier), &TrainOpts::default(), true).unwrap();
        randomize_adapters(&packed, 9);
        randomize_adapters(&reference, 9);
        let (loss, grads) = loss_and_grads(&packed, &x, &y);
        let (want_loss, want) = loss_and_grads(&reference, &x, &y);
        assert!(want.len() > 20 && want.iter().all(|(n, _)| n.ends_with(".lora_a") || n.ends_with(".lora_b")), "only the adapter trains");
        assert!(want.iter().all(|(n, g)| norm(g) > 0.0 || n.ends_with(".lora_a")), "every B factor has a live gradient");
        let (rel, name) = worst_relative(&grads, &want);
        eprintln!("{tier:?}: loss {loss} vs {want_loss}, worst adapter gradient error {rel:.2e} ({name})");
        assert!((loss - want_loss).abs() <= 1e-4 * want_loss.abs().max(1.0), "{tier:?}: loss {loss} vs {want_loss}");
        assert!(rel < 2e-3, "{tier:?}: {name}'s gradient is {rel} from the decoded-values build's");
    }
}

/// Spec: a full fine-tune cannot hold a packed linear (its weights must be trained), and says so by name.
#[test]
fn a_full_fine_tune_over_a_packed_base_is_refused() {
    let cfg = TrainConfig::tiny();
    let err = Ripple::build(cfg.clone(), ROWS, SEQ, &cfg.tiny_weights(5), &packed_opts(Dtype::NF4), true).err().expect("refused");
    assert!(err.contains("full fine-tune") && err.contains("adapter"), "{err}");
}
