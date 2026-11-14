// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training that gives the same model on every
// backend for its clients. If your team needs expertise in cross-backend
// numerical parity of mixture-of-experts training then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The qwen35moe trainer on CUDA trains what the CPU backend trains: a LoRA run
//! (the mixer projections) and a full fine-tune (experts and router included),
//! a few AdamW steps each on a tiny four-layer hybrid with random weights, give
//! the same loss and the same gradient of every trained tensor at every step,
//! and the same loss after the last update.
//!
//! Both backends compute in fp32 and are held to the round-off bound of
//! [`brain_testutil::training`]: a value passes through [`CHAIN`] reductions in
//! series in one forward and backward, each over at most the model's longest
//! contraction. A card whose library GEMMs run on TF32 tensor cores computes
//! something else (qwen35moe takes no fp32 pin) and skips, as does a box where
//! no CUDA device opens within [`OPEN_DEADLINE`] ([`gpu_core::testgpu::on_cuda`]).

use std::collections::HashMap;
use std::time::Duration;

use brain_testutil::training::{assert_same_training, step_bound, Trace};
use gpu_core::provider::library::{GemmPrecision, LibraryGemmProvider};
use qwen35moe::config::{lora_cfg, Qwen35Config};
use qwen35moe::model::{pipelines, Qwen35};

const T: u32 = 8;
const STEPS: u32 = 3;
const ADAM: optim::Adam = optim::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-4 };
/// Reductions in series through one forward and backward: per layer the input norm, the mixer's input projections,
/// its convolution or attention scores, its recurrence or attention-weighted values, its output norm and projection,
/// the post norm, the router, the expert and shared-expert gate/up and down (twelve), plus the final norm, the head
/// and the loss, forward and backward, over four layers.
const CHAIN: u32 = 2 * (12 * 4 + 3);
const OPEN_DEADLINE: Duration = Duration::from_secs(30);

fn config(lora: bool) -> Qwen35Config {
    Qwen35Config { n_layers: 4, block_size: T, max_position_embeddings: T, lora: lora.then(|| lora_cfg(2, 4.0)), ..Qwen35Config::tiny() }
}

fn bound(cfg: &Qwen35Config) -> f64 {
    let longest = [cfg.d_model, cfg.vocab, cfg.n_heads * cfg.head_dim, cfg.linear_num_value_heads * cfg.linear_value_head_dim, cfg.moe_intermediate_size * cfg.top_k, cfg.shared_expert_intermediate_size, T]
        .into_iter()
        .max()
        .unwrap();
    step_bound((CHAIN * longest) as usize)
}

fn train(model: &Qwen35) -> Trace {
    let vocab = model.cfg.vocab;
    let tokens: Vec<u32> = (0..T).map(|i| (i * 7 + 3) % vocab).collect();
    let targets: Vec<u32> = (0..T).map(|i| (i * 5 + 11) % vocab).collect();
    let names = model::Model::optimized_params(model).expect("a training build names its trained tensors");
    let mut trace = Trace::default();
    for step in 1..=STEPS {
        model.set_batch(&tokens, &targets);
        model.zero_grads();
        trace.losses.push(model.forward());
        model.backward();
        trace.grads.push(names.iter().map(|n| (n.clone(), model.read_grad(n))).collect());
        model.adamw_step(step, 1e-2, 0.01, ADAM, None, 1.0);
    }
    model.set_batch(&tokens, &targets);
    trace.losses.push(model.forward());
    trace
}

fn compare(lora: bool, what: &str) {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = config(lora);
    let init: HashMap<String, Vec<f32>> = qwen35moe::init::init_weights(&cfg, 0x0e5c + u64::from(lora));
    let tolerance = bound(&cfg);
    let on_cpu = train(&Qwen35::new_train_on(gpu_core::Gpu::new_cpu(pipelines()), cfg.clone(), 1, T, &init));
    let on_cuda = gpu_core::testgpu::on_cuda(pipelines(), OPEN_DEADLINE, move || {
        let model = Qwen35::new_train(cfg, 1, T, &init);
        assert_eq!(model.gpu.kind(), "cuda", "the build placed on the card runs on CUDA");
        let tf32 = LibraryGemmProvider::for_gpu(&model.gpu, GemmPrecision::Auto).tf32_enabled();
        (!tf32).then(|| train(&model))
    });
    match on_cuda {
        Ok(Some(on_cuda)) => assert_same_training(&on_cpu, &on_cuda, tolerance, what),
        Ok(None) => brain_testutil::skip_unavailable(&format!("qwen35moe train_cuda_matches_cpu ({what}): the card runs the library GEMMs on TF32 tensor cores; nothing was compared")),
        Err(why) => brain_testutil::skip_unavailable(&format!("qwen35moe train_cuda_matches_cpu ({what}): {why}; nothing was compared")),
    }
}

#[test]
fn the_cuda_trainer_trains_lora_like_the_cpu_backend() {
    compare(true, "lora");
}

#[test]
fn the_cuda_trainer_trains_a_full_fine_tune_like_the_cpu_backend() {
    compare(false, "full fine-tune");
}
