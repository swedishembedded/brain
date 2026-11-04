// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of decision models for its clients. If your team needs
// expertise in training state encoders and calibrated option scorers within a fixed device budget, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Activation recompute for the decision encoder, held to the contract `model::saved::check_recompute` states: an
//! encoder whose layers share one set of intermediates and rebuild a layer's from its input before its backward gives
//! every gradient of the encoder that keeps them, bit for bit, holding fewer device bytes. Narrow formats and going
//! back to keeping are refused by name. One test in its own binary, so nothing else allocates while the device bytes
//! are compared.

use decide::config::EncoderConfig;
use decide::kern::PIPELINES;
use decide::model::Encoder;
use model::saved::{SavedFormat, SavedPolicy};

/// Ragged spans, so the per-span attention and the packed embedding scatter both run.
const SPANS: &[(u32, u32)] = &[(0, 7), (7, 5), (12, 6)];

/// The gradient bits of every parameter, sorted by name, and the device bytes the encoder holds after the policy is
/// set.
type Measured = (Vec<(String, Vec<u32>)>, u64);

fn config() -> EncoderConfig {
    EncoderConfig { n_layers: 5, ..EncoderConfig::tiny() }
}

fn rows() -> u32 {
    SPANS.iter().map(|&(_, l)| l).sum()
}

fn max_span() -> u32 {
    SPANS.iter().map(|&(_, l)| l).max().unwrap()
}

fn step(init: &decide::Tensors, policy: SavedPolicy) -> Measured {
    let cfg = config();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let mut enc = Encoder::new_train_on(gpu.share(), cfg.clone(), rows(), max_span(), init);
    enc.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    let ids: Vec<u32> = (0..rows()).map(|i| 1 + i % 9).collect();
    let types: Vec<u32> = SPANS.iter().enumerate().flat_map(|(i, &(_, l))| std::iter::repeat_n((i % 2) as u32, l as usize)).collect();
    enc.set_batch(&ids, &types, SPANS);
    enc.prepare_reverse();
    let seed: Vec<f32> = (0..rows() * cfg.d_model).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();
    enc.zero_grads();
    enc.forward();
    enc.backward(&seed);
    enc.poll_wait();
    let mut names: Vec<String> = cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), enc.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

#[test]
fn the_decision_encoder_recomputes_its_layers_exactly() {
    let init = decide::init::init_weights(&config(), data::rng::random_seed());
    let (want_grads, keep) = step(&init, SavedPolicy::Keep);
    let (got_grads, recompute) = step(&init, SavedPolicy::Recompute);
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }
    assert!(recompute < keep, "recomputing holds {recompute} device bytes, keeping {keep}");

    // Narrow saved formats do not apply, and a recomputing encoder does not go back to keeping.
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = Encoder::new_train_on(gpu, config(), rows(), max_span(), &init);
    assert!(enc.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("decide"));
    enc.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(enc.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("decide"));
}
