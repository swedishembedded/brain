// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA on Ripple: what a rank-r adapter trains is the merged-weight gradient seen through its factors; the base
//! stays untouched; the adapter written by a run serves, through the engine, the logits the training model computes;
//! and it exports as a PEFT adapter directory.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, and for serving the result. If your team needs expertise in parameter-efficient fine-tuning of
//! language models then you can procure our services by sending an email to info@swedishembedded.com.

use model::adapter::bank::{AdapterBankConfig, VerifiedAdapter};
use model::adapter::device::LoraCfg;
use model::conformance_grad::randomize_adapters;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use model::{Batch, Model};
use optim::Adam;
use ricursive::train::{adapter, default_targets, AdapterInit, LORA_LEAVES};
use ricursive::{open_safetensors, Ripple, TrainConfig, TrainOpts};

use crate::serve::{prefill_in_slices, serve_config, stream};
use crate::synthetic::{synthetic, Shape};
use crate::taps::scaled_error;

const ROWS: u32 = 2;
const SEQ: u32 = 6;

fn batch(vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let n = (ROWS * SEQ) as usize;
    let x: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 1) % vocab).collect();
    let mut y: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 9) % vocab).collect();
    y[5] = ricursive::train::IGNORE;
    y[11] = ricursive::train::IGNORE;
    (x, y)
}

fn grads(m: &Ripple, x: &[u32], y: &[u32]) -> (f32, std::collections::HashMap<String, Vec<f32>>) {
    m.set_batch(Batch::Lm { tokens: x, targets: y });
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    m.poll_wait();
    (loss, m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect())
}

/// `out[r, c] = sum_k a[r, k] b[k, c]`, `a` is `[rows, inner]`, `b` is `[inner, cols]`.
fn matmul(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for k in 0..inner {
            let x = f64::from(a[r * inner + k]);
            for c in 0..cols {
                out[r * cols + c] = (f64::from(out[r * cols + c]) + x * f64::from(b[k * cols + c])) as f32;
            }
        }
    }
    out
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..cols).flat_map(|c| (0..rows).map(move |r| a[r * cols + c])).collect()
}

/// Spec: with `W' = W + s B A` as the weights of a full fine-tune, the adapter's gradients are the merged-weight
/// gradient through the factors, to first order exactly: `dB = s dW' A^T`, `dA = s B^T dW'` (the forward of the
/// adapter model also equals the merged model's). This is the proof the adapter trains what full-weight training of
/// its delta would.
#[test]
fn an_adapters_gradient_is_the_merged_weight_gradient_through_its_factors() {
    let rank = 2;
    let lora = LoraCfg::new(rank, 4.0, default_targets());
    let scale = lora.scale();
    let cfg = TrainConfig::tiny().with_lora(lora);
    let weights = cfg.tiny_weights(5);
    let (x, y) = batch(cfg.model.vocab_size as u32);

    let adapted = Ripple::build(cfg.clone(), ROWS, SEQ, &weights, &TrainOpts::default(), true).unwrap();
    randomize_adapters(&adapted, 9);
    let (loss, got) = grads(&adapted, &x, &y);

    use model::adapter::sites::LoraSites;
    let mut merged = weights.clone();
    for site in cfg.lora_sites() {
        let (an, bn) = model::adapter::device::lora_names(&site.name);
        let (a, b) = (adapted.read_weight(&an), adapted.read_weight(&bn));
        let delta = matmul(&b, &a, site.spec.out, rank as usize, site.spec.inn);
        let w = merged.get_mut(&site.name).unwrap();
        w.iter_mut().zip(&delta).for_each(|(w, d)| *w += scale * d);
    }
    let full = Ripple::build(TrainConfig::tiny(), ROWS, SEQ, &merged, &TrainOpts::default(), true).unwrap();
    let (full_loss, dw) = grads(&full, &x, &y);
    assert!((loss - full_loss).abs() <= 1e-5 * full_loss.abs(), "the adapter model and the merged model compute the same loss: {loss} vs {full_loss}");

    let mut worst = 0.0f32;
    for site in cfg.lora_sites() {
        let (an, bn) = model::adapter::device::lora_names(&site.name);
        let (a, b) = (adapted.read_weight(&an), adapted.read_weight(&bn));
        let (out, inn, r) = (site.spec.out, site.spec.inn, rank as usize);
        let dwm = &dw[&site.name];
        let db: Vec<f32> = matmul(dwm, &transpose(&a, r, inn), out, inn, r).iter().map(|v| scale * v).collect();
        let da: Vec<f32> = matmul(&transpose(&b, out, r), dwm, r, out, inn).iter().map(|v| scale * v).collect();
        worst = worst.max(scaled_error(&format!("dB {}", bn.trim_start_matches("model.layers.")), &got[&bn], &db));
        worst = worst.max(scaled_error(&format!("dA {}", an.trim_start_matches("model.layers.")), &got[&an], &da));
    }
    eprintln!("worst factor-gradient error against the merged-weight gradient: {worst:.3e}");
    assert!(worst <= 5e-5, "{worst}");
}

/// Spec: an adapter run trains only the factors, learns, and leaves the base weights exactly as they were.
#[test]
fn an_adapter_run_learns_and_leaves_the_base_untouched() {
    let cfg = TrainConfig::tiny().with_lora(LoraCfg::new(4, 8.0, default_targets()));
    let weights = cfg.tiny_weights(5);
    let (x, y) = batch(cfg.model.vocab_size as u32);
    let m = Ripple::build(cfg.clone(), ROWS, SEQ, &weights, &TrainOpts::default(), true).unwrap();
    assert!(m.param_names().iter().all(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")), "only the factors train");
    assert_eq!(m.param_names().len(), 3 * 8 * 2, "an A and a B for each of the eight linears of three layers");
    m.set_batch(Batch::Lm { tokens: &x, targets: &y });
    let start = m.forward();
    for step in 1..=25 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 2e-2, 0.0, Adam::default(), Some(1.0), 1.0);
    }
    let end = m.forward();
    eprintln!("adapter loss {start:.4} -> {end:.4}");
    assert!(end < 0.8 * start, "the adapter lowers the loss: {start} -> {end}");
    for (name, want) in weights.iter().filter(|(n, _)| !n.ends_with(".lora_a") && !n.ends_with(".lora_b")) {
        assert_eq!(&m.read_weight(name), want, "{name} is frozen");
    }
}

/// Spec: an adapter a run wrote, attached to the serving engine over the same base, gives the logits the training
/// model computes (windowed layers wrapping their ring, global layer paged); detached, the engine gives the base's.
#[test]
fn a_trained_adapter_served_by_the_engine_gives_the_training_models_logits() {
    let model = synthetic(&["sliding_attention", "sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 });
    let layers: Vec<usize> = (0..model.cfg.num_hidden_layers).collect();
    let vocab = model.cfg.vocab_size as u32;
    let prompt = stream(5, 12, vocab);
    let targets: Vec<u32> = prompt[1..].iter().copied().chain([ricursive::train::IGNORE]).collect();

    let base = open_safetensors(&model.path(), &model.cfg, &layers).unwrap();
    let cfg = TrainConfig::new(model.cfg.clone(), 12).with_lora(LoraCfg::new(2, 8.0, default_targets()));
    let init = AdapterInit::fresh(&cfg, &base, 11);
    let trainer = Ripple::build(cfg, 1, 12, &init, &TrainOpts::default(), true).unwrap();
    trainer.set_batch(Batch::Lm { tokens: &prompt, targets: &targets });
    for step in 1..=8 {
        trainer.zero_grads();
        trainer.forward();
        trainer.backward();
        trainer.adamw_step(step, 3e-2, 0.0, Adam::default(), None, 1.0);
    }
    let last = |logits: &[f32]| logits[logits.len() - vocab as usize..].to_vec();
    let trained = last(&trainer.logits_all(&prompt).unwrap());

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("adapter.safetensors");
    trainer.save_adapter(file.to_str().unwrap(), "test/adapter", "test/base", None, None).unwrap();
    let loaded = adapter::read_adapter(file.to_str().unwrap()).unwrap();
    assert_eq!(loaded.rank, 2);

    let mut engine = model.engine(serve_config(4, 0));
    let serve_last = |engine: &mut ricursive::RicursiveEngine, ns: model::paged::KvNamespace| {
        let mut table = BlockTable::with_namespace(ns);
        let hidden = prefill_in_slices(engine, &mut table, &prompt, 5);
        let logits = engine.logits(&hidden);
        engine.release_table(&mut table);
        logits
    };
    let base_logits = serve_last(&mut engine, model::paged::BASE_KV_NAMESPACE);
    engine.enable_adapters(AdapterBankConfig::new(1, 2, &LORA_LEAVES)).unwrap();
    let id = engine.load_adapter(VerifiedAdapter { label: "test/adapter".into(), digest: "sha256:test".into(), adapter: loaded }).unwrap();
    assert_eq!(engine.adapter_availability(id), model::serve::AdapterAvailability::Resident);
    engine.pin_adapter(id);
    let served = serve_last(&mut engine, id.kv_namespace());
    engine.unpin_adapter(id);

    let moved = scaled_error("adapter vs base", &served, &base_logits);
    assert!(moved > 1e-3, "the adapter changes what the engine answers ({moved:.2e})");
    let err = scaled_error("served vs trained", &served, &trained);
    assert!(err <= 5e-6, "the engine's bank answers what the training model does: {err:.2e}");
    assert_eq!(serve_last(&mut engine, model::paged::BASE_KV_NAMESPACE), base_logits, "a base sequence beside the loaded adapter answers the base, bit for bit");
}

/// Spec: the log-probabilities the trainer scores tokens with (what a policy-gradient objective's ratio and KL read)
/// are the serving engine's for the same tokens, at the same sampling temperature, under a trained adapter: RL on the
/// trainer's forward optimises the policy the engine samples from.
#[test]
fn the_trainers_token_logprobs_are_the_serving_engines_under_a_trained_adapter() {
    let model = synthetic(&["sliding_attention", "sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 });
    let layers: Vec<usize> = (0..model.cfg.num_hidden_layers).collect();
    let vocab = model.cfg.vocab_size;
    let tokens = stream(9, 12, vocab as u32);
    let targets: Vec<u32> = tokens[1..].iter().copied().chain([ricursive::train::IGNORE]).collect();

    let base = open_safetensors(&model.path(), &model.cfg, &layers).unwrap();
    let cfg = TrainConfig::new(model.cfg.clone(), 12).with_lora(LoraCfg::new(2, 8.0, default_targets()));
    let init = AdapterInit::fresh(&cfg, &base, 11);
    let trainer = Ripple::build(cfg, 1, 12, &init, &TrainOpts::default(), true).unwrap();
    trainer.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
    for step in 1..=8 {
        trainer.zero_grads();
        trainer.forward();
        trainer.backward();
        trainer.adamw_step(step, 3e-2, 0.0, Adam::default(), None, 1.0);
    }
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("adapter.safetensors");
    trainer.save_adapter(file.to_str().unwrap(), "test/adapter", "test/base", None, None).unwrap();

    let mut engine = model.engine(serve_config(4, 0));
    engine.enable_adapters(AdapterBankConfig::new(1, 2, &LORA_LEAVES)).unwrap();
    let id = engine.load_adapter(VerifiedAdapter { label: "test/adapter".into(), digest: "sha256:test".into(), adapter: adapter::read_adapter(file.to_str().unwrap()).unwrap() }).unwrap();
    assert_eq!(engine.adapter_availability(id), model::serve::AdapterAvailability::Resident);
    engine.pin_adapter(id);
    // The adapter was fitted to `targets`, so those are all near probability one; the comparison scores other tokens,
    // the ones a sampler draws and a policy gradient pushes down.
    let probed: Vec<u32> = targets.iter().map(|&t| if t == ricursive::train::IGNORE { t } else { (t + 5) % vocab as u32 }).collect();
    // The engine's log-probability of the probed token after the first `n`, at a sampler temperature of 0.7.
    let temperature = 0.7f32;
    let served: Vec<f32> = (1..tokens.len())
        .map(|n| {
            let mut table = BlockTable::with_namespace(id.kv_namespace());
            let hidden = prefill_in_slices(&mut engine, &mut table, &tokens[..n], 5);
            let logits: Vec<f64> = engine.logits(&hidden).iter().map(|&z| f64::from(z / temperature)).collect();
            engine.release_table(&mut table);
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = max + logits.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
            (logits[probed[n - 1] as usize] - log_sum) as f32
        })
        .collect();
    let trained = model::logprobs::token_logprobs_scaled(&trainer, &tokens, &probed, 1.0 / temperature).expect("the trainer scores tokens");
    let trained = &trained[..served.len()];
    let worst = served.iter().zip(trained).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("trainer vs served token log-probabilities over {} positions: worst difference {worst:.3e}", served.len());
    assert!(served.iter().any(|lp| *lp < -0.5), "the log-probabilities are not all near zero, so the comparison means something");
    assert!(worst <= 1e-4, "the trainer's log-probabilities are the engine's: {worst:.3e}");
}

/// Spec: a refused adapter says why and changes nothing: the wrong rank of a site, an unknown site.
#[test]
fn an_adapter_that_does_not_fit_the_model_is_refused_by_name() {
    let model = synthetic(&["sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 });
    let mut engine = model.engine(serve_config(4, 0));
    engine.enable_adapters(AdapterBankConfig::new(1, 2, &LORA_LEAVES)).unwrap();
    let bad = model::lora::device_adapter::DeviceAdapter { rank: 2, alpha: 4.0, sites: vec![model::lora::device_adapter::AdapterSite { base: "model.layers.9.self_attn.q_proj.weight".into(), a: vec![0.0; 2 * 64], b: vec![0.0; 64 * 2] }], saved: Vec::new() };
    let err = engine.load_adapter(VerifiedAdapter { label: "bad".into(), digest: "sha256:bad".into(), adapter: bad }).expect_err("a layer the model does not have");
    assert!(err.to_string().contains("model.layers.9.self_attn.q_proj.weight"), "{err}");
}

/// Spec: the PEFT export carries every pair under PEFT's names and a config naming the rank, alpha and modules.
#[test]
fn an_adapter_exports_as_a_peft_directory() {
    let cfg = TrainConfig::tiny().with_lora(LoraCfg::new(2, 4.0, vec!["q_proj".into(), "down_proj".into()]));
    let m = Ripple::build(cfg.clone(), ROWS, SEQ, &cfg.tiny_weights(5), &TrainOpts::default(), true).unwrap();
    randomize_adapters(&m, 3);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("adapter.safetensors");
    m.save_adapter(file.to_str().unwrap(), "test/adapter", "ricursiveAI/Ripple-65B-Preview", None, None).unwrap();
    let out = dir.path().join("peft");
    adapter::export_peft(file.to_str().unwrap(), &out, None).unwrap();

    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!((config["r"].as_u64(), config["lora_alpha"].as_f64()), (Some(2), Some(4.0)));
    assert_eq!(config["base_model_name_or_path"], "ricursiveAI/Ripple-65B-Preview");
    let modules: Vec<&str> = config["target_modules"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(modules.len(), 6, "q_proj and down_proj of three layers");
    assert!(modules.contains(&"model.layers.2.mlp.down_proj"), "{modules:?}");
    let tensors = checkpoint::safetensors::read(out.join("adapter_model.safetensors").to_str().unwrap()).unwrap();
    let a = tensors.iter().find(|t| t.name == "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight").expect("PEFT's name for A");
    assert_eq!(a.data, m.read_weight("model.layers.1.self_attn.q_proj.weight.lora_a"));
    assert_eq!(a.shape, vec![2, 64]);
    assert!(LORA_LEAVES.contains(&"q_proj"));
}
