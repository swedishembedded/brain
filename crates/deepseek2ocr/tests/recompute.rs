// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of document-OCR vision-language models for its clients.
// If your team needs expertise in training multi-tower OCR composites within a fixed device budget, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Activation recompute for the DeepSeek-OCR composite, held to the contract `model::saved::check_recompute` states:
//! a composite whose SAM tower, CLIP tower and decoder each share one set of intermediates and rebuild a layer's from
//! its input before its backward gives the loss, the input-image gradient and the gradient of every tower's
//! parameters of the composite that keeps them, bit for bit, holding fewer device bytes. A narrow format narrows the
//! decoder's residual stream while the vision towers recompute from f32 inputs. Going back to keeping is refused by
//! name. One test in its own binary, so nothing else allocates while the device bytes are compared. The weights are
//! drawn at random from the config's own manifests, so the test needs no fixture.

use std::collections::HashMap;

use data::rng::Rng;
use deepseek2ocr::{DeepseekOcr, DeepseekOcrConfig};
use model::saved::{SavedFormat, SavedPolicy};

fn config() -> DeepseekOcrConfig {
    let mut cfg = DeepseekOcrConfig::tiny();
    cfg.clip.shape.n_layers = 5;
    cfg.decoder.shape.n_layers = 5;
    cfg
}

/// Weights for every tensor of the four manifests, small enough that nothing saturates; norm gains near one.
fn init(cfg: &DeepseekOcrConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let manifests = cfg.sam.param_list().into_iter().chain(cfg.clip.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product()))).chain(cfg.glue_param_list()).chain(cfg.decoder.shape.param_list());
    manifests
        .map(|(name, numel)| {
            let gain = name.ends_with("norm.weight") || name.ends_with("norm1.weight") || name.ends_with("norm2.weight") || name.contains("layernorm");
            let data = (0..numel).map(|_| if gain { 1.0 + 0.1 * (rng.next_f32() - 0.5) } else { 0.2 * (rng.next_f32() - 0.5) }).collect();
            (name, data)
        })
        .collect()
}

/// The device bytes the three towers hold, each on its own handle.
fn live(m: &DeepseekOcr) -> u64 {
    let on = |gpu: &gpu_core::Gpu| gpu.memory_usage().expect("the test device reports its memory").live;
    on(m.gpu()) + on(&m.encoder().sam().gpu) + on(&m.encoder().clip().gpu)
}

/// The loss bits, the image gradient bits, every tower's parameter gradient bits and the device bytes held after the
/// policy is set.
type Measured = (u32, Vec<u32>, Vec<(String, Vec<u32>)>, u64);

fn step(cfg: &DeepseekOcrConfig, weights: &HashMap<String, Vec<f32>>, seed: u64, policy: SavedPolicy) -> Measured {
    let dev = |p| gpu_core::testgpu::dev(p);
    let seq = 12u32;
    let mut m = DeepseekOcr::new(&dev, cfg.clone(), weights, seed, seq, 2, true);
    m.set_saved_policy(policy).unwrap();
    let held = live(&m);
    let mut rng = Rng::new(seed ^ 1);
    let ids: Vec<u32> = (0..seq).map(|_| (rng.next_u64() % cfg.decoder.shape.vocab as u64) as u32).collect();
    let mut targets: Vec<u32> = ids[1..].to_vec();
    targets.push(ids[0]);
    m.set_tokens(&ids, &targets);
    let image: Vec<f32> = (0..3 * cfg.sam.image_h() as usize * cfg.sam.image_w() as usize).map(|_| rng.next_f32() - 0.5).collect();
    m.zero_grads();
    let loss = m.forward(&image);
    let d_image = m.backward();
    m.decoder().poll_wait();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    let mut grads = Vec::new();
    for n in m.decoder().param_names() {
        grads.push((format!("decoder.{n}"), bits(m.decoder().read_grad(&n))));
    }
    for n in m.encoder().sam().param_names() {
        grads.push((format!("sam.{n}"), bits(m.encoder().sam().read_grad(&n))));
    }
    for n in m.encoder().clip().dispatched_params() {
        grads.push((format!("clip.{n}"), bits(m.encoder().clip().read_grad(&n))));
    }
    for n in m.encoder().glue_param_names() {
        grads.push((format!("glue.{n}"), bits(m.encoder().read_glue_grad(&n))));
    }
    grads.sort();
    (loss.to_bits(), bits(d_image), grads, held)
}

#[test]
fn the_composite_recomputes_every_tower_exactly() {
    let cfg = config();
    let seed = data::rng::random_seed();
    let weights = init(&cfg, seed);
    let (want, want_image, want_grads, keep) = step(&cfg, &weights, seed, SavedPolicy::Keep);
    let (got, got_image, got_grads, recompute) = step(&cfg, &weights, seed, SavedPolicy::Recompute);
    assert_eq!(got, want, "the loss differs recomputing");
    assert!(want_image.iter().any(|b| f32::from_bits(*b) != 0.0), "no gradient reached the image");
    assert!(got_image == want_image, "the image gradient differs recomputing");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }
    assert!(recompute < keep, "recomputing holds {recompute} device bytes, keeping {keep}");

    // A narrow format narrows the decoder's residual stream; the vision towers recompute from f32 inputs.
    let (loss, ..) = step(&cfg, &weights, seed, SavedPolicy::Quantized(SavedFormat::Bf16));
    assert_ne!(loss, want, "bf16 boundaries left the loss untouched: the decoder never held them");

    // Going back to keeping is refused by name, from a recomputing composite.
    let dev = |p| gpu_core::testgpu::dev(p);
    let mut m = DeepseekOcr::new(&dev, cfg, &weights, seed, 12, 2, true);
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("deepseek2ocr"));
}
