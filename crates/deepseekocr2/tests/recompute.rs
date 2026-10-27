// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of document-OCR vision-language models for its clients.
// If your team needs expertise in training multi-view OCR composites within a fixed device budget, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Activation recompute for the DeepSeek-OCR-2 resampler and its composite. The resampler records its layers as a
//! `model::saved::BlockStack`, held to the contract `model::saved::check_block_recompute` states: recomputing keeps
//! only each layer's input and gives the projected output's gradient and every parameter's gradient of keeping every
//! layer's state, bit for bit. The composite forwards the policy to the resampler and the decoder, so the whole
//! composite's loss and gradients agree the same way. Narrow formats and going back to keeping are refused by name.

use std::collections::HashMap;

use data::rng::Rng;
use deepseek2::DeepseekV2Config;
use deepseekocr2::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};
use deepseekocr2::encoder::{Resampler, PIPELINES};
use deepseekocr2::model::DeepseekOcr2;
use deepseekocr2::rows::TileGrid;
use model::saved::{BlockStep, SavedFormat, SavedPolicy};
use sam1::SamViTConfig;

fn vision_config() -> DeepseekOcr2VisionConfig {
    let encoder = Qwen2EncoderConfig { d_model: 8, n_layers: 5, n_heads: 4, n_kv_heads: 2, ffn_hidden: 11, rms_eps: 1e-6, rope_theta: 10_000.0, n_query_local: 3, n_query_global: 5 };
    let decoder = DeepseekV2Config::tiny();
    DeepseekOcr2VisionConfig { sam: SamViTConfig { compress_out: 8, ..SamViTConfig::tiny() }, encoder, decoder_hidden: decoder.shape.d_model, lora: None }
}

fn bits(v: &[f32]) -> Vec<u64> {
    v.iter().map(|x| u64::from(x.to_bits())).collect()
}

#[test]
fn the_resampler_recomputes_its_layers_exactly() {
    let cfg = vision_config();
    let mut rng = Rng::new(data::rng::random_seed());
    let init: HashMap<String, Vec<f32>> = deepseekocr2::init::init_weights(&cfg, data::rng::random_seed());
    let e = &cfg.encoder;
    let sam: Vec<f32> = (0..e.n_query_global * e.d_model).map(|_| rng.next_f32() - 0.5).collect();
    let d_projected: Vec<f32> = (0..e.n_query_global * cfg.decoder_hidden).map(|_| rng.next_f32() - 0.5).collect();
    model::saved::check_block_recompute("deepseekocr2 resampler", |policy| {
        let mut r = Resampler::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &init, true);
        r.set_saved_policy(policy)?;
        r.zero_grads();
        let (projected, state) = r.forward_train(&sam, false);
        let held_caches = state.held_block_caches();
        let d_sam = r.backward_train(state, &d_projected, false);
        r.gpu().poll_wait();
        let loss: f64 = projected.iter().zip(&d_projected).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let mut grads: Vec<(String, Vec<u64>)> = cfg.param_list().into_iter().map(|(n, _)| (n.clone(), bits(&r.read_grad(&n)))).collect();
        grads.push(("d_sam".into(), bits(&d_sam)));
        grads.sort();
        Ok(BlockStep { loss: loss.to_bits(), grads, held_caches })
    })
    .unwrap();

    let mut r = Resampler::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, &init, true);
    assert!(r.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("deepseekocr2"));
}

/// The loss bits, each view's SAM-token gradient bits and each resampler parameter's gradient bits.
type Stepped = (u32, Vec<Vec<u64>>, Vec<(String, Vec<u64>)>);

/// The composite's loss, every view's SAM-token gradient and the resampler's gradients under `policy`, as bits.
fn composite_step(policy: SavedPolicy, seed: u64) -> Stepped {
    let vision_cfg = vision_config();
    let decoder_cfg = DeepseekV2Config::tiny();
    let grid = TileGrid::new(2, 1);
    let seq = decoder_cfg.block_size;
    let vision_init = deepseekocr2::init::init_weights(&vision_cfg, seed);
    let decoder_init = deepseek2::init_weights(&decoder_cfg, seed ^ 1);
    let mut m = DeepseekOcr2::new_on(gpu_core::testgpu::dev(PIPELINES), gpu_core::testgpu::dev(deepseek2::PIPELINES), vision_cfg.clone(), decoder_cfg.clone(), &vision_init, &decoder_init, grid, seq, 0, true);
    m.set_saved_policy(policy).unwrap();
    let mut rng = Rng::new(seed ^ 2);
    let e = &vision_cfg.encoder;
    let mut rnd = |n: u32| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
    let tiles: Vec<Vec<f32>> = (0..grid.tiles()).map(|_| rnd(e.n_query_local * e.d_model)).collect();
    let global = rnd(e.n_query_global * e.d_model);
    let ids: Vec<u32> = (0..seq).map(|i| (i * 3 + 1) % decoder_cfg.shape.vocab).collect();
    let mut targets: Vec<u32> = ids[1..].to_vec();
    targets.push(deepseek2::IGNORE);
    m.set_tokens(&ids, &targets);
    m.zero_grads();
    let (loss, state) = m.forward(&tiles, &global);
    let (d_tiles, d_global) = m.backward(state);
    m.decoder().poll_wait();
    let mut d_sam: Vec<Vec<u64>> = d_tiles.iter().map(|d| bits(d)).collect();
    d_sam.push(bits(&d_global));
    let mut grads: Vec<(String, Vec<u64>)> = vision_cfg.param_list().into_iter().map(|(n, _)| (n.clone(), bits(&m.encoder().read_grad(&n)))).collect();
    grads.sort();
    (loss.to_bits(), d_sam, grads)
}

#[test]
fn the_composite_recomputes_both_towers_exactly() {
    let seed = data::rng::random_seed();
    let want = composite_step(SavedPolicy::Keep, seed);
    let got = composite_step(SavedPolicy::Recompute, seed);
    assert_eq!(got.0, want.0, "the loss differs recomputing");
    assert!(want.2.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b as u32) != 0.0)), "no resampler parameter received a gradient");
    assert!(got.1 == want.1, "a view's SAM-token gradient differs recomputing");
    assert!(got.2 == want.2, "a resampler gradient differs recomputing");

    // Going back to keeping is refused by name, from a recomputing composite.
    let decoder_cfg = DeepseekV2Config::tiny();
    let vision_cfg = vision_config();
    let vision_init = deepseekocr2::init::init_weights(&vision_cfg, seed);
    let decoder_init = deepseek2::init_weights(&decoder_cfg, seed ^ 1);
    let mut m = DeepseekOcr2::new_on(gpu_core::testgpu::dev(PIPELINES), gpu_core::testgpu::dev(deepseek2::PIPELINES), vision_cfg, decoder_cfg.clone(), &vision_init, &decoder_init, TileGrid::new(2, 1), decoder_cfg.block_size, 0, true);
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("deepseekocr2"));
}
