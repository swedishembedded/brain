// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of vision-language models for its clients. If your team
// needs expertise in trading activation memory for recomputation in an encoder-decoder, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Activation recompute for Florence-2's BART trainer, held to the contract `model::saved::check_recompute` states:
//! a trainer whose layers share one set of intermediates per tower and rebuild a layer's from its input before its
//! backward gives the loss and every gradient of the trainer that keeps them, bit for bit, holding fewer device bytes,
//! for a full fine-tune and for LoRA. Narrow formats and going back to keeping are refused by name. One test in its own
//! binary, so nothing else allocates while the device bytes are compared.

use std::collections::HashMap;

use florence2::text::{BartConfig, LoraCfg};
use florence2::train::{self, Florence2Trainer};
use model::saved::{SavedFormat, SavedPolicy};

const T_PROMPT: u32 = 6;
const MAX_T: u32 = 5;

fn cfg() -> BartConfig {
    BartConfig { encoder_layers: 3, decoder_layers: 3, ..BartConfig::tiny() }
}

fn build(gpu: &gpu_core::Gpu, lora: Option<LoraCfg>) -> Florence2Trainer {
    let names = train::param_list(&cfg(), T_PROMPT, MAX_T, lora.as_ref());
    let init: HashMap<String, Vec<f32>> = train::init_weights(&names, 0.15, 11);
    Florence2Trainer::new(gpu.share(), cfg(), T_PROMPT, MAX_T, lora, &init)
}

/// The loss bits, every trained parameter's gradient bits, and the device bytes the trainer holds.
type Measured = (u32, Vec<(String, Vec<u32>)>, u64);

fn step(lora: Option<LoraCfg>, policy: SavedPolicy) -> Measured {
    let gpu = gpu_core::testgpu::dev(train::PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let mut m = build(&gpu, lora);
    m.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    let v = cfg().vocab_size;
    let prompt: Vec<u32> = (0..T_PROMPT).map(|i| (i * 5 + 1) % v).collect();
    let dec: Vec<u32> = (0..MAX_T).map(|i| (i * 3 + 2) % v).collect();
    let targets: Vec<u32> = (0..MAX_T).map(|i| (i * 7 + 3) % v).collect();
    m.set_batch(&prompt, &dec, &targets);
    m.zero_grads();
    let loss = m.loss();
    m.backward();
    m.poll_wait();
    let mut names = m.param_names();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), m.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (loss.to_bits(), grads, held)
}

fn check(tag: &str, lora: Option<LoraCfg>) {
    let (want, want_grads, keep) = step(lora.clone(), SavedPolicy::Keep);
    let (got, got_grads, recompute) = step(lora, SavedPolicy::Recompute);
    assert_eq!(got, want, "{tag}: the loss differs recomputing");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "{tag}: no parameter received a gradient");
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{tag}: {n}: gradient differs recomputing");
    }
    assert!(recompute < keep, "{tag}: recomputing holds {recompute} device bytes, keeping {keep}");
}

#[test]
fn florence2_recomputes_its_layers_exactly() {
    check("full", None);
    let targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"].iter().map(|s| s.to_string()).collect();
    check("lora", Some(LoraCfg::new(2, 4.0, targets)));

    let gpu = gpu_core::testgpu::dev(train::PIPELINES);
    let mut m = build(&gpu, None);
    assert!(m.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("florence2"));
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("florence2"));
}
