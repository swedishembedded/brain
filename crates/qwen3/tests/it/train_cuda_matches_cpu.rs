// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training that gives the same model on every
// backend for its clients. If your team needs expertise in cross-backend
// numerical parity of training then you can procure our services by sending
// an email to info@swedishembedded.com.

//! The qwen3 trainer on CUDA trains what the CPU backend trains: a LoRA run and
//! a full fine-tune, a few AdamW steps each on tiny random weights, give the
//! same loss and the same gradient of every trained tensor at every step, and
//! the same loss after the last update.
//!
//! Both backends compute in fp32 with the GEMMs pinned to the fp32 reference
//! tier, so they differ only in the order sums are formed, and are held to the
//! round-off bound of [`brain_testutil::training`]: a value passes through
//! [`CHAIN`] reductions in series in one forward and backward, each over at
//! most the model's longest contraction. AdamW's ε is far above the gradient's
//! round-off, which that bound needs.
//!
//! The CUDA side runs on a worker thread placed on a card
//! ([`gpu_core::testgpu::on_cuda`]). A box with no card, a card that does not
//! open, a device that takes longer than [`OPEN_DEADLINE`] to create (a loaded
//! card), or a process whose ambient backend is not CUDA skips, naming which.

use std::collections::HashMap;
use std::time::Duration;

use brain_testutil::training::{assert_same_training, step_bound, Trace};
use gpu_core::provider::library::GemmPrecision;
use qwen3::{LoraCfg, Qwen, QwenConfig, Shard};

const ROWS: u32 = 2;
const T: u32 = 12;
const STEPS: u32 = 3;
const ADAM: optim::Adam = optim::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-4 };
/// Reductions in series through one forward and backward of [`config`]: per
/// layer the two norms, q/k/v, the scores, the attention-weighted values, the
/// output projection, gate/up and down (eight), plus the final norm, the head
/// and the loss, forward and backward.
const CHAIN: u32 = 2 * (8 * 2 + 3);
/// How long creating the CUDA device may take before the card is called loaded and the comparison skipped.
const OPEN_DEADLINE: Duration = Duration::from_secs(30);

fn config(lora: Option<LoraCfg>) -> QwenConfig {
    QwenConfig { block_size: T, max_position_embeddings: T, lora, ..QwenConfig::tiny() }
}

/// The round-off bound of one forward and backward of `cfg`.
fn bound(cfg: &QwenConfig) -> f64 {
    let longest = [cfg.d_model, cfg.d_ff, cfg.vocab, cfg.n_heads * cfg.head_dim, ROWS * T].into_iter().max().unwrap();
    step_bound((CHAIN * longest) as usize)
}

fn train(model: &Qwen) -> Trace {
    let tokens: Vec<u32> = (0..ROWS * T).map(|i| (i * 7 + 3) % 23).collect();
    let targets: Vec<u32> = (0..ROWS * T).map(|i| (i * 5 + 11) % 23).collect();
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

fn build(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>) -> Qwen {
    Qwen::new_shard_gemm(cfg.clone(), ROWS, T, init, true, Shard::whole(cfg.n_layers as usize), GemmPrecision::Fp32Reference)
}

/// Train on the CPU backend and on CUDA and compare, or skip (printed, with the reason) when no CUDA device is there.
fn compare(cfg: QwenConfig, init: HashMap<String, Vec<f32>>, what: &str) {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let tolerance = bound(&cfg);
    let on_cpu = train(&gpu_core::devices::with_host_tier(|| build(&cfg, &init)));
    let on_cuda = gpu_core::testgpu::on_cuda(qwen3::model::pipelines(), OPEN_DEADLINE, move || {
        let model = build(&cfg, &init);
        assert_eq!(model.gpu.kind(), "cuda", "the build placed on the card runs on CUDA");
        train(&model)
    });
    match on_cuda {
        Ok(on_cuda) => assert_same_training(&on_cpu, &on_cuda, tolerance, what),
        Err(why) => brain_testutil::skip_unavailable(&format!("train_cuda_matches_cpu ({what}): {why}; nothing was compared")),
    }
}

#[test]
fn the_cuda_trainer_trains_lora_like_the_cpu_backend() {
    let cfg = config(Some(LoraCfg::attn(2, 4.0)));
    let init = qwen3::init_weights(&cfg, 0x7ea1);
    compare(cfg, init, "lora");
}

#[test]
fn the_cuda_trainer_trains_a_full_fine_tune_like_the_cpu_backend() {
    let cfg = config(None);
    let init = qwen3::init_weights(&cfg, 0xf011);
    compare(cfg, init, "full fine-tune");
}
