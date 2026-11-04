// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of financial time-series models for its clients. If your
// team needs expertise in training market-data decoders within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Activation recompute for the Kronos decoder trainer, held to the contract `model::saved::check_recompute` states:
//! a trainer whose layers share one set of intermediates and rebuild a layer's from its input before its backward
//! gives the loss and every gradient of the trainer that keeps them, bit for bit, holding fewer device bytes, for a
//! full fine-tune and for LoRA. Narrow formats and going back to keeping are refused by name. One test in its own
//! binary, so nothing else allocates while the device bytes are compared.

use std::collections::HashMap;

use data::rng::Lcg;
use kronos::config::KronosConfig;
use kronos::train::{param_list_c, KronosTrain, LoraCfg, CAL, PIPELINES};
use model::saved::{SavedFormat, SavedPolicy};

const T: u32 = 8;
const B: u32 = 2;

/// The loss bits, every trained parameter's gradient bits, and the device bytes the trainer holds.
type Measured = (u32, Vec<(String, Vec<u32>)>, u64);

fn config() -> KronosConfig {
    KronosConfig { n_layers: 5, ..KronosConfig::tiny() }
}

/// Reference-named weights: the trainer splits `embedding.fusion_proj.weight` into its two halves itself.
fn init_weights(cfg: &KronosConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let d = cfg.d_model;
    let mut names: Vec<(String, usize)> = param_list_c(cfg).into_iter().filter(|(n, _)| n != "embedding.fusion_l" && n != "embedding.fusion_r").collect();
    names.push(("embedding.fusion_proj.weight".into(), d * 2 * d));
    names
        .into_iter()
        .map(|(name, numel)| {
            let norm_gain = name.ends_with("norm.weight") || name.ends_with("norm1.weight") || name.ends_with("norm2.weight");
            let data = if norm_gain { (0..numel).map(|_| 1.0 + rng.scaled(0.02)).collect() } else { rng.vec_scaled(numel, 0.06) };
            (name, data)
        })
        .collect()
}

/// One batch of `B` sequences of `T` tokens: s1/s2 context, five calendar columns, the sampled sibling and both
/// targets.
struct Batch {
    s1: Vec<u32>,
    s2: Vec<u32>,
    cal: Vec<Vec<u32>>,
    sampled: Vec<u32>,
    t1: Vec<u32>,
    t2: Vec<u32>,
}

fn batch(cfg: &KronosConfig, seed: u64) -> Batch {
    let mut rng = Lcg::new(seed);
    let n = (B * T) as usize;
    let mut column = |k: usize| -> Vec<u32> { (0..n).map(|_| rng.next_u32() % k as u32).collect() };
    Batch {
        s1: column(cfg.s1_vocab()),
        s2: column(cfg.s2_vocab()),
        cal: CAL.iter().map(|&(_, card)| column(card)).collect(),
        sampled: column(cfg.s1_vocab()),
        t1: column(cfg.s1_vocab()),
        t2: column(cfg.s2_vocab()),
    }
}

fn step(init: &HashMap<String, Vec<f32>>, b: &Batch, lora: Option<LoraCfg>, policy: SavedPolicy) -> Measured {
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let mut m = KronosTrain::with_lora_batch_on(gpu.share(), config(), T, B, init, lora);
    m.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    m.set_batch(&b.s1, &b.s2, &[&b.cal[0], &b.cal[1], &b.cal[2], &b.cal[3], &b.cal[4]], &b.sampled, &b.t1, &b.t2);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    m.poll_wait();
    let mut names = m.param_names();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), m.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (loss.to_bits(), grads, held)
}

fn check(tag: &str, init: &HashMap<String, Vec<f32>>, b: &Batch, lora: Option<LoraCfg>) {
    let (want, want_grads, keep) = step(init, b, lora.clone(), SavedPolicy::Keep);
    let (got, got_grads, recompute) = step(init, b, lora, SavedPolicy::Recompute);
    assert_eq!(got, want, "{tag}: the loss differs recomputing");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|x| f32::from_bits(*x) != 0.0)), "{tag}: no parameter received a gradient");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, c)) in want_grads.iter().zip(&got_grads) {
        assert!(a == c, "{tag}: {n}: gradient differs recomputing");
    }
    assert!(recompute < keep, "{tag}: recomputing holds {recompute} device bytes, keeping {keep}");
}

#[test]
fn kronos_recomputes_its_layers_exactly() {
    let cfg = config();
    let init = init_weights(&cfg, data::rng::random_seed());
    let b = batch(&cfg, data::rng::random_seed());
    check("full", &init, &b, None);
    check("lora", &init, &b, Some(LoraCfg::attn(2, 4.0)));

    // Narrow saved formats do not apply, and a recomputing trainer does not go back to keeping.
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut m = KronosTrain::with_lora_batch_on(gpu, cfg, T, B, &init, None);
    assert!(m.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("kronos"));
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("kronos"));
}
