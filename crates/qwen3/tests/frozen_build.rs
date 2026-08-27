// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A frozen build and a trainable build of the same weights must produce the
//! same forward, to the bit.
//!
//! [`qwen3::Qwen::new`] hardcodes `train = true`, which gives every parameter
//! `paramstore::Role::Trainable` - a gradient buffer and two Adam moments
//! allocated beside each weight, four times the model. A caller that only ever
//! runs the encoder forward (FLUX.2's text conditioning is the case this test
//! exists for: `flux2::pipeline::build_text_encoder`) wants
//! `new_shard(.., train = false, ..)`, which allocates the weights only. That
//! is a memory decision and it must not be a numerical one, so this pins it:
//! same source tensors, same tokens, identical hidden states.
//!
//! Checkpoint-free and tiny - the property is about role assignment, not about
//! any particular model - but with `n_layers` deep enough that a multi-layer
//! tap is a real one.

use std::collections::HashMap;

use qwen3::{Qwen, QwenConfig, Shard};

fn tiny() -> QwenConfig {
    QwenConfig {
        vocab: 64,
        block_size: 16,
        n_layers: 4,
        d_model: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_ff: 64,
        rope_theta: 1.0e6,
        rms_eps: 1e-6,
        max_position_embeddings: 16,
        tie_embeddings: false,
        qk_norm: true,
        attn_bias: false,
        lora: None,
    }
    .with_defaults()
}

/// Deterministic pseudo-random weights for every parameter the config names.
fn init(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    };
    cfg.param_list()
        .into_iter()
        .map(|(name, numel)| {
            // norm scales sit around 1; everything else small around 0
            let one = name.ends_with("norm.weight") || name.ends_with("_norm.weight");
            let v = (0..numel).map(|_| if one { 1.0 + next() } else { next() }).collect();
            (name, v)
        })
        .collect()
}

#[test]
fn frozen_and_trainable_builds_agree_on_the_forward() {
    let cfg = tiny();
    let w = init(&cfg);
    let t = 8u32;
    let ids: Vec<u32> = (0..t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let content = 5usize; // the rest are pads: exercises the masked-pad path
    let taps = [1usize, 3];

    let shard = Shard::whole(cfg.n_layers as usize);
    let trainable = Qwen::new_shard(cfg.clone(), 1, t, &w, true, shard);
    let a = trainable.encode_hiddens_padded(&ids, content, &taps);
    drop(trainable);
    let frozen = Qwen::new_shard(cfg.clone(), 1, t, &w, false, Shard::whole(cfg.n_layers as usize));
    let b = frozen.encode_hiddens_padded(&ids, content, &taps);

    assert_eq!(a.len(), taps.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.len(), t as usize * cfg.d_model as usize, "tap {i} shape");
        // Not a tolerance: same weights, same graph, same device - the only
        // difference is which buffers were allocated alongside.
        assert_eq!(
            x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "tap {i}: the frozen build's hidden state differs from the trainable build's"
        );
        // ...and the forward is a real one, so the equality above is not two
        // buffers of zeros agreeing with each other.
        assert!(x.iter().any(|v| v.abs() > 1e-6), "tap {i} is all zeros");
        assert!(x.iter().all(|v| v.is_finite()), "tap {i} is not finite");
    }
    // Different taps must be different states - a tap index that were ignored
    // would make every comparison above vacuous.
    assert_ne!(a[0], a[1], "the two taps must read different layers");
}
