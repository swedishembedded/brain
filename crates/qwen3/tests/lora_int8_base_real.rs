// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parameter-efficient fine-tuning on
// consumer GPUs for its clients. If your team needs expertise in training
// large models within a fixed memory budget then you can procure our
// services by sending an email to info@swedishembedded.com.

//! What holding a real decoder's frozen base as int8 costs a LoRA run, on the
//! real DeepSeek-R1-Distill-Qwen-1.5B: the same batch through the bf16 base
//! and through the int8 base (weights quantised to a byte and a scale per 32,
//! activations left in fp32), with the same non-zero adapters. The loss and
//! the gradient of every adapter tensor stay close to the bf16 base's:
//! quantising the weight is the only difference. The bound is what the
//! measurement showed, with a margin: tensor cosines of about 0.997 on average and
//! 0.989 at worst (a 1.5B decoder is the most sensitive size), a relative error
//! of at most 0.15, and a loss within 1% - noise next to a training step's own.
//!
//! Ignored by default (reads the checkpoint); needs a GPU with about 8 GB
//! free. Skips when the checkpoint is absent.

use data::rng::Rng;
use qwen3::finetune::LoraInit;
use qwen3::{Dtype, LoraCfg, Qwen};

const REPO: &str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
    (diff / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>()).sqrt()
}

fn run(dt: Dtype, tokens: &[u32], targets: &[u32], dir: &str) -> Option<(f32, Vec<(String, Vec<f32>)>)> {
    let (mut cfg, base) = qwen3::open_checkpoint(dir).expect("open the checkpoint");
    cfg.lora = Some(LoraCfg::attn(8, 16.0));
    cfg.block_size = tokens.len() as u32;
    let init = LoraInit::fresh(&cfg, base, 3);
    let m = Qwen::new_lora_dt(cfg, 1, tokens.len() as u32, &init, dt);
    if m.linear_dtype() != Some(dt) {
        return None;
    }
    // Non-zero adapters (the same ones for both bases), so every gradient flows.
    let mut rng = Rng::new(7);
    let names = model::Model::param_names(&m);
    for name in names.iter().filter(|n| n.ends_with(".lora_b")) {
        let len = m.read_weight(name).len();
        m.write_weight(name, &(0..len).map(|_| (rng.next_f32() - 0.5) * 0.02).collect::<Vec<f32>>());
    }
    m.set_batch(tokens, targets);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    Some((loss, names.into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect()))
}

#[test]
#[ignore = "reads a real checkpoint"]
fn an_int8_base_trains_the_adapters_the_bf16_base_does() {
    let Some(dir) = brain_testutil::model_dir(REPO) else { return brain_testutil::skip(&format!("{REPO} not in the model store")) };
    let text = "The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step, and the best time to plant a tree was twenty years ago; the second best time is now. ".repeat(16);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(&dir).expect("tokenizer");
    let ids: Vec<u32> = data::tokenizer::Tokenizer::encode(&tok, &text).into_iter().take(257).collect();
    let (tokens, targets) = (ids[..256].to_vec(), ids[1..257].to_vec());

    let Some((want_loss, want)) = run(Dtype::BF16, &tokens, &targets, &dir) else { return };
    let Some((loss, got)) = run(Dtype::I8, &tokens, &targets, &dir) else { return };
    let (mut cosines, mut worst_rel) = (Vec::new(), 0.0f64);
    for ((_, g), (_, w)) in got.iter().zip(&want) {
        if w.iter().all(|x| *x == 0.0) {
            continue;
        }
        cosines.push(cosine(g, w));
        worst_rel = worst_rel.max(rel_l2(g, w));
    }
    let worst_cos = cosines.iter().cloned().fold(1.0f64, f64::min);
    let mean_cos = cosines.iter().sum::<f64>() / cosines.len() as f64;
    eprintln!("int8 vs bf16 base on {REPO}: loss {loss:.4} vs {want_loss:.4}; {} adapter tensors: cosine mean {mean_cos:.5}, worst {worst_cos:.5}; worst rel_l2 {worst_rel:.4}", cosines.len());
    assert!((loss - want_loss).abs() < 0.05 * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    assert!(mean_cos > 0.995 && worst_cos > 0.98 && worst_rel < 0.2, "cosine mean {mean_cos}, worst {worst_cos}");
}
