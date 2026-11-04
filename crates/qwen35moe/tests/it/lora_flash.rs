// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA fine-tuning of large mixture-of-experts models on a single
// GPU for its clients. If your team needs expertise in long-context fine-tuning within a fixed memory budget then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The qwen35moe training attention at a long sequence: where the device can run the causal GQA flash kernels the
//! trainer uses them (memory linear in the sequence, no `heads * T * T` probabilities), and the loss and every
//! adapter gradient agree with the materialised chain. The CPU backend cannot run the workgroup-cooperative
//! kernels, keeps the chain, and says so.

use gpu_core::Gpu;
use qwen35moe::config::{lora_cfg, Qwen35Config};
use qwen35moe::model::{pipelines, Qwen35};

const LONG: u32 = 256;

fn loss_and_grads(m: &Qwen35, x: &[u32], y: &[u32]) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(x, y);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    (loss, m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

#[test]
fn the_cpu_backend_keeps_the_materialised_chain() {
    let cfg = Qwen35Config { block_size: LONG, lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() };
    let init = qwen35moe::init::init_weights(&cfg, 7);
    let m = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg, 1, LONG, &init);
    assert!(!m.attention_is_flash(), "a device without workgroup reductions cannot run the flash trio");
}

#[test]
#[ignore = "needs a device that runs workgroup-cooperative kernels (a GPU); the CPU backend keeps the chain"]
fn flash_attention_trains_the_same_adapters_as_the_materialised_chain() {
    let cfg = Qwen35Config { block_size: LONG, lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() };
    let init = qwen35moe::init::init_weights(&cfg, 7);
    let x: Vec<u32> = (0..LONG).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..LONG).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    let chain = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 1, LONG, &init);
    chain.set_flash_min_t(u32::MAX);
    let flash = Qwen35::new_train_on(Gpu::new(pipelines()), cfg, 1, LONG, &init);
    assert!(!chain.attention_is_flash());
    assert!(flash.attention_is_flash(), "this device should run the flash kernels at {LONG} tokens");
    let (want_loss, want) = loss_and_grads(&chain, &x, &y);
    let (loss, got) = loss_and_grads(&flash, &x, &y);
    assert!((loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    let mut nonzero = 0;
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        if w.iter().all(|v| *v == 0.0) {
            continue;
        }
        nonzero += 1;
        assert!(cosine(g, w) > 0.9999, "{name}: adapter gradient cosine {}", cosine(g, w));
    }
    assert!(nonzero >= 4, "the adapters received gradient ({nonzero} tensors)");
}
