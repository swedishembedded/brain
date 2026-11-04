// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of world models for its clients. If your team needs
// expertise in training diffusion world models on interactive-environment data within a fixed device budget, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Activation recompute for the DIAMOND UNet trainer, held to the contract `model::saved::check_recompute` states: a
//! trainer whose residual blocks share their internal activations and rebuild a block's from its inputs before its
//! backward gives the loss and every gradient of the trainer that keeps them, bit for bit, holding fewer device
//! bytes. A narrow saved format is refused by name. One test in its own binary, so nothing else allocates while the
//! device bytes are compared.

use std::collections::HashMap;

use diamond::play::NormalRng;
use diamond::train::DiamondTrainer;
use diamond::{DiamondConfig, Tensors};
use model::saved::{SavedFormat, SavedPolicy};

/// Two blocks per level over two levels (the second with attention), plus the mid blocks and the up path's joins.
fn config() -> DiamondConfig {
    DiamondConfig {
        img_channels: 3,
        num_steps_conditioning: 2,
        cond_channels: 16,
        depths: vec![2, 2],
        channels: vec![8, 8],
        attn_depths: vec![false, true],
        num_actions: 4,
        h: 8,
        w: 8,
        sigma_data: 0.5,
        sigma_offset_noise: 0.3,
    }
}

fn normals(rng: &mut NormalRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.normal() * scale).collect()
}

fn random_tensors(cfg: &DiamondConfig, rng: &mut NormalRng) -> Tensors {
    let mut t: Tensors = HashMap::new();
    for (name, shape) in cfg.param_list() {
        let n: usize = shape.iter().product();
        // Conv weights scale with fan-in so the deep stack does not blow up the loss.
        let scale = if name.contains(".weight") && shape.len() == 4 { 0.25 / ((shape[1] * shape[2] * shape[3]) as f32).sqrt() } else { 0.25 };
        t.insert(name, (shape, normals(rng, n, scale)));
    }
    t
}

/// The loss bits, every trained parameter's gradient bits and the bytes the graph's activations hold.
type Measured = (u32, Vec<(String, Vec<u32>)>, u64);

fn step(cfg: &DiamondConfig, tensors: &Tensors, seed: u64, policy: SavedPolicy) -> Measured {
    let mut tr = DiamondTrainer::from_tensors(cfg.clone(), tensors, None);
    tr.set_saved_policy(policy).unwrap();
    let mut rng = NormalRng::new(seed);
    let n_px = tr.n_px as usize;
    let obs = normals(&mut rng, n_px * cfg.num_steps_conditioning as usize, 0.8);
    let clean = normals(&mut rng, n_px, 0.8);
    let noise = normals(&mut rng, n_px, 1.0);
    tr.set_transition(&obs, &clean, &noise, 0.7, &[1, 3]);
    tr.zero_grads();
    let loss = tr.forward_loss();
    tr.backward();
    tr.gpu.poll_wait();
    let mut names: Vec<String> = diamond::train::trainable_list(cfg).into_iter().map(|(n, _)| n).collect();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), tr.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (loss.to_bits(), grads, tr.activation_bytes())
}

#[test]
fn the_unet_recomputes_its_residual_blocks_exactly() {
    let cfg = config();
    let seed = std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 0u8);
    let tensors = random_tensors(&cfg, &mut NormalRng::new(seed));
    let (want, want_grads, keep) = step(&cfg, &tensors, seed ^ 1, SavedPolicy::Keep);
    let (got, got_grads, recompute) = step(&cfg, &tensors, seed ^ 1, SavedPolicy::Recompute);
    assert_eq!(got, want, "the loss differs recomputing");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }
    assert!(recompute < keep, "recomputing holds {recompute} activation bytes, keeping {keep}");

    // A narrow saved format does not apply.
    let mut tr = DiamondTrainer::from_tensors(cfg, &tensors, None);
    assert!(tr.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("diamond"));
}
