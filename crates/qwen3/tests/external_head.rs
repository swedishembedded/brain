// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of multimodal and generative
// language models for its clients. If your team needs expertise in
// attaching your own output heads to a pretrained decoder then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The decoder runs to its final-norm hidden states and takes their gradient
//! back from a head the caller owns (Janus-Pro's image-token head, say).
//! Applying the decoder's own head on the host through that seam gives the
//! adapter gradients the built-in head does: the seam is the built-in loss
//! with the head moved out.

use qwen3::{LoraCfg, Qwen, QwenConfig, IGNORE};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const TOKENS: [u32; 8] = [1, 5, 9, 2, 7, 3, 11, 4];
const TARGETS: [u32; 8] = [IGNORE, IGNORE, 9, 2, 7, 3, 11, 4];

fn config() -> QwenConfig {
    QwenConfig { block_size: 8, tie_embeddings: false, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
}

fn weights(cfg: &QwenConfig) -> std::collections::HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 13);
    // Adapters that matter: a zero B would make every A gradient vanish.
    for (n, v) in w.iter_mut().filter(|(n, _)| n.ends_with(".lora_b")) {
        v.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 5 + n.len()) % 9) as f32 * 0.03 - 0.12);
    }
    w
}

/// Mean cross-entropy over the supervised positions of `hidden @ head^T`, and
/// its gradient with respect to `hidden`.
fn host_head(hidden: &[f32], head: &[f32], d: usize, v: usize) -> (f32, Vec<f32>) {
    let n = hidden.len() / d;
    let supervised: Vec<usize> = (0..n).filter(|&i| TARGETS[i] != IGNORE).collect();
    let mut loss = 0.0f64;
    let mut d_hidden = vec![0.0f32; hidden.len()];
    for &i in &supervised {
        let h = &hidden[i * d..(i + 1) * d];
        let logits: Vec<f64> = (0..v).map(|j| (0..d).map(|k| h[k] as f64 * head[j * d + k] as f64).sum()).collect();
        let max = logits.iter().cloned().fold(f64::MIN, f64::max);
        let sum: f64 = logits.iter().map(|l| (l - max).exp()).sum();
        loss += -(logits[TARGETS[i] as usize] - max - sum.ln());
        for j in 0..v {
            let p = (logits[j] - max).exp() / sum - if j == TARGETS[i] as usize { 1.0 } else { 0.0 };
            for k in 0..d {
                d_hidden[i * d + k] += (p * head[j * d + k] as f64 / supervised.len() as f64) as f32;
            }
        }
    }
    ((loss / supervised.len() as f64) as f32, d_hidden)
}

#[test]
fn a_head_moved_out_of_the_decoder_gives_the_adapter_gradients_of_the_built_in_one() {
    if gpu_disabled() {
        return;
    }
    let cfg = config();
    let init = weights(&cfg);

    let inner = Qwen::new(cfg.clone(), 1, 8, &init);
    inner.set_batch(&TOKENS, &TARGETS);
    inner.zero_grads();
    let want_loss = inner.forward();
    inner.backward();

    let mut outer = Qwen::new(cfg.clone(), 1, 8, &init);
    outer.enable_external_head();
    outer.set_batch(&TOKENS, &TARGETS);
    outer.zero_grads();
    let hidden = outer.forward_hidden();
    assert_eq!(hidden.len(), 8 * cfg.d_model as usize);
    let head = outer.read_weight("lm_head.weight");
    let (loss, d_hidden) = host_head(&hidden, &head, cfg.d_model as usize, cfg.vocab as usize);
    outer.backward_hidden(&d_hidden);

    assert!((loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    let mut compared = 0;
    for name in model::Model::param_names(&inner).into_iter().filter(|n| n.contains(".lora_")) {
        let (a, b) = (inner.read_grad(&name), outer.read_grad(&name));
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let worst = a.iter().zip(&b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / scale));
        assert!(worst < 2e-3, "{name}: gradient differs by {worst}");
        compared += 1;
    }
    assert!(compared >= 8, "every adapter tensor was compared ({compared})");
}
