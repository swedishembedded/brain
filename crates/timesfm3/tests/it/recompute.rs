// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of time-series foundation models for its clients. If
// your team needs expertise in training multivariate forecasters within a fixed device budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Activation recompute for the TimesFM-3 trainer, held to the contract `model::saved::check_recompute` states: a
//! trainer whose layers share one set of intermediates and rebuild a layer's from its input before its backward
//! gives every gradient of the trainer that keeps them, bit for bit, holding fewer device bytes, for a full
//! fine-tune and for LoRA. Narrow formats and going back to keeping are refused by name. One test in its own binary,
//! so nothing else allocates while the device bytes are compared.

use std::collections::HashMap;

use model::saved::{SavedFormat, SavedPolicy};
use timesfm3::config::Timesfm3Config;
use timesfm3::train::{fixed_input, init_weights, LoraCfg, Timesfm3Train, TRAIN_PIPELINES};

const B: usize = 1;
const V: usize = 2;
const N: usize = 3;

/// The gradient bits of every trained parameter, sorted by name, and the device bytes the trainer holds.
type Measured = (Vec<(String, Vec<u32>)>, u64);

fn config() -> Timesfm3Config {
    Timesfm3Config { num_layers: 5, ..Timesfm3Config::tiny() }
}

struct Fixture {
    weights: HashMap<String, Vec<f32>>,
    x: Vec<f32>,
    mask: Vec<bool>,
    /// The seed `dL/dlogits` of the proxy objective `<c, logits>`.
    c: Vec<f32>,
}

fn fixture() -> Fixture {
    let cfg = config();
    let seed = seed::random_seed();
    let mut mask = vec![false; B * V * N];
    mask[0] = true;
    Fixture {
        weights: init_weights(&cfg, seed),
        x: fixed_input(&cfg, B, V, N, seed ^ 1),
        mask,
        c: (0..B * V * N * cfg.head_out_dim()).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect(),
    }
}

fn trainer(gpu: gpu_core::Gpu, f: &Fixture, lora: Option<LoraCfg>) -> Timesfm3Train {
    match lora {
        None => Timesfm3Train::new_on(gpu, config(), &f.x, &f.mask, B, V, N, &f.weights),
        Some(lc) => Timesfm3Train::new_lora_on(gpu, config(), lc, &f.x, &f.mask, B, V, N, &f.weights),
    }
}

fn step(f: &Fixture, lora: Option<LoraCfg>, policy: SavedPolicy) -> Measured {
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let mut t = trainer(gpu.share(), f, lora);
    t.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    t.zero_grads();
    t.forward();
    t.backward(&f.c);
    t.poll_wait();
    let mut names = t.param_names();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), t.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

fn check(tag: &str, f: &Fixture, lora: Option<LoraCfg>) {
    let (want_grads, keep) = step(f, lora.clone(), SavedPolicy::Keep);
    let (got_grads, recompute) = step(f, lora, SavedPolicy::Recompute);
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|x| f32::from_bits(*x) != 0.0)), "{tag}: no parameter received a gradient");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, c)) in want_grads.iter().zip(&got_grads) {
        assert!(a == c, "{tag}: {n}: gradient differs recomputing");
    }
    assert!(recompute < keep, "{tag}: recomputing holds {recompute} device bytes, keeping {keep}");
}

#[test]
fn timesfm3_recomputes_its_layers_exactly() {
    let f = fixture();
    check("full", &f, None);
    check("lora", &f, Some(LoraCfg::attn(2, 4.0).with_output_head()));

    // Narrow saved formats do not apply, and a recomputing trainer does not go back to keeping.
    let mut t = trainer(gpu_core::testgpu::dev(TRAIN_PIPELINES), &f, None);
    assert!(t.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("timesfm3"));
    t.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(t.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("timesfm3"));
}
