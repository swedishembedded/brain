// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mixed-precision loading and fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in running one model whose tensors are held in different numeric
// formats then you can procure our services by sending an email to
// info@swedishembedded.com.

//! End-to-end acceptance of a MIXED-precision Qwen3 checkpoint: attention FP8, MLP NF4, embeddings bf16,
//! norms f32, in one packed file. Spec (the same as `qwen35/tests/mixed_precision.rs`):
//!   * the precision report lists every tensor of the file with its stored format;
//!   * the model loads each tensor in that format, its own report of what is resident agrees with the file's
//!     row by row, and no tensor is widened to fp32;
//!   * its forward and LoRA gradients equal an fp32 build's over the decoded values;
//!   * LoRA over it trains, and the saved adapter reloads over the same packed base.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use checkpoint::packed::{convert_to_file, BoundedWorkspace, CodecRegistry, LoadConfig, Loader, MemoryBudget, PackedReader, PackedSource};
use checkpoint::TensorSource;
use gpu_core::provider::library::GemmPrecision;
use model::ops::{conversion_policy, encoding_for, stored_tier_policy, FP8_BLOCK};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

const BLOCK: u32 = 12;

fn batch() -> (Vec<u32>, Vec<u32>) {
    let n = BLOCK as usize;
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 22 + 1).collect();
    let targets = (0..n).map(|i| tokens[(i + 1) % n]).collect();
    (tokens, targets)
}

/// Every contraction width a multiple of the NF4 group (32).
fn config() -> QwenConfig {
    QwenConfig { block_size: BLOCK, d_model: 64, n_heads: 4, n_kv_heads: 2, head_dim: 16, d_ff: 128, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
}

/// Attention FP8, MLP NF4, embeddings bf16, norms f32.
fn tier_of(name: &str) -> Option<Dtype> {
    if name.contains(".lora_") {
        None
    } else if name.contains("attn.") && name.ends_with(".weight") && !name.contains("norm") {
        Some(Dtype::F8E4M3)
    } else if name.contains("mlp.") {
        Some(Dtype::NF4)
    } else if name == "tok.weight" || name == "lm_head.weight" {
        Some(Dtype::BF16)
    } else if name.contains("norm") || name.contains("ln1") || name.contains("ln2") {
        Some(Dtype::F32)
    } else {
        None
    }
}

fn leaves(cfg: &QwenConfig) -> Vec<String> {
    (0..cfg.n_layers).flat_map(|l| qwen3::q8::Q8::LINEARS.iter().map(move |leaf| format!("blocks.{l}.{leaf}"))).collect()
}

fn shape_of(cfg: &QwenConfig, name: &str, numel: usize) -> Vec<u64> {
    let (d, ff, hq, hkv) = (cfg.d_model as u64, cfg.d_ff as u64, cfg.q_dim() as u64, cfg.kv_dim() as u64);
    let leaf = name.split_once('.').and_then(|(_, rest)| rest.split_once('.')).map_or("", |(_, leaf)| leaf);
    match (name, leaf) {
        ("tok.weight" | "lm_head.weight", _) => vec![cfg.vocab as u64, d],
        (_, "attn.wq.weight") => vec![hq, d],
        (_, "attn.wk.weight" | "attn.wv.weight") => vec![hkv, d],
        (_, "attn.wo.weight") => vec![d, hq],
        (_, "mlp.gate.weight" | "mlp.up.weight") => vec![ff, d],
        (_, "mlp.down.weight") => vec![d, ff],
        _ => vec![numel as u64],
    }
}

struct Fixture {
    cfg: QwenConfig,
    dir: PathBuf,
    /// The values of every tensor before conversion (adapter `B` non-zero).
    init: HashMap<String, Vec<f32>>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

/// Write `values` as an f32 file, then convert it ONCE into the mixed packed file `out`.
fn write_mixed(cfg: &QwenConfig, dir: &Path, values: &HashMap<String, Vec<f32>>, out: &Path) {
    let plain = dir.join("plain.safetensors");
    let mut names: Vec<&String> = values.keys().collect();
    names.sort();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = names.into_iter().map(|n| (n.clone(), shape_of(cfg, n, values[n].len()), values[n].clone())).collect();
    checkpoint::st::save_safetensors(plain.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
    let reader = PackedReader::open(&plain).unwrap();
    let codecs = CodecRegistry::reference();
    let loader = Loader { reader: &reader, codecs: &codecs, workspace: BoundedWorkspace::new(MemoryBudget::new(8 << 20, 0)), config: LoadConfig::default() };
    let requests: Vec<_> = tensors
        .iter()
        .map(|(name, shape, _)| {
            let desc = checkpoint::packed::TensorDescriptor::new(name, shape.clone(), checkpoint::packed::QuantEncoding::f32());
            (name.clone(), conversion_policy(&desc, tier_of(name)))
        })
        .collect();
    convert_to_file(&loader, &requests, out, &serde_json::json!({})).expect("convert to the mixed file");
}

fn fixture(tag: &str) -> Fixture {
    let cfg = config();
    let dir = std::env::temp_dir().join(format!("qwen3-mixed-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let mut init = qwen3::init_weights(&cfg, data::rng::random_seed());
    let mut rng = data::rng::Rng::new(data::rng::random_seed());
    for (name, v) in init.iter_mut() {
        if name.ends_with(".lora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * 0.3);
        }
    }
    write_mixed(&cfg, &dir, &init, &dir.join("mixed.safetensors"));
    Fixture { cfg, dir, init }
}

impl Fixture {
    fn source(&self) -> PackedSource {
        PackedSource::identity(Box::new(PackedReader::open(&self.dir.join("mixed.safetensors")).unwrap()), CodecRegistry::reference()).unwrap()
    }

    fn model(&self, src: &dyn TensorSource) -> Qwen {
        let tiers = stored_tier_policy(src, leaves(&self.cfg).iter().map(String::as_str));
        Qwen::new_lora_tiers(self.cfg.clone(), 1, BLOCK, src, &tiers, GemmPrecision::Fp32Reference)
    }

    fn decoded(&self, src: &dyn TensorSource) -> HashMap<String, Vec<f32>> {
        self.cfg.param_list().into_iter().map(|(name, _)| { let mut v = Vec::new(); assert!(src.with_tensor(&name, &mut |d| v = d.to_vec()), "{name}"); (name, v) }).collect()
    }
}

fn loss_and_grads(m: &Qwen) -> (f32, Vec<(String, Vec<f32>)>) {
    let (tokens, targets) = batch();
    m.set_batch(&tokens, &targets);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

#[test]
fn the_report_lists_every_tensor_of_the_file_in_the_format_it_is_stored_in() {
    let fx = fixture("report");
    let report = fx.source().precision_report();
    let mut want: Vec<String> = fx.cfg.param_list().into_iter().map(|(n, _)| n).collect();
    want.sort();
    assert_eq!(report.rows().iter().map(|r| r.name.clone()).collect::<Vec<_>>(), want, "a row for every tensor of the model, and no other");
    for row in report.rows() {
        let tier = tier_of(&row.name).unwrap_or(Dtype::F32);
        assert_eq!(row.format, encoding_for(tier).unwrap().to_string(), "{}", row.name);
    }
    assert_eq!(report.by_format().len(), 4, "FP8, NF4, bf16 and f32 all appear: {:?}", report.by_format());
}

#[test]
fn every_tensor_reaches_the_device_in_its_stored_format() {
    let fx = fixture("resident");
    let src = fx.source();
    let m = fx.model(&src);
    let stored = src.precision_report();
    let resident = m.precision_report();
    for row in stored.rows() {
        let got = resident.get(&row.name).unwrap_or_else(|| panic!("{}: not resident", row.name));
        assert_eq!(got.format, row.format, "{}: resident format", row.name);
        let tier = tier_of(&row.name).unwrap_or(Dtype::F32);
        let fp32_bytes = row.shape.iter().product::<u64>() * 4;
        assert!(got.bytes < fp32_bytes || tier == Dtype::F32, "{}: {} B resident is not below the {fp32_bytes} B of an fp32 expansion", row.name, got.bytes);
        // The loader widens an FP8 file's bf16 block scales to the f32 the kernels read; that is the only difference.
        let wider = if tier == Dtype::F8E4M3 { row.shape[0].div_ceil(FP8_BLOCK as u64) * row.shape[1].div_ceil(FP8_BLOCK as u64) * 2 } else { 0 };
        assert_eq!(got.bytes, row.bytes + wider, "{}: resident bytes", row.name);
    }
    for name in leaves(&fx.cfg) {
        assert_eq!(m.linear_dtype_of(&name), tier_of(&name), "{name}: held as the file stores it");
    }
}

#[test]
fn the_mixed_model_computes_what_an_fp32_build_over_its_decoded_values_does() {
    let fx = fixture("parity");
    let src = fx.source();
    let (want_loss, want) = loss_and_grads(&Qwen::new(fx.cfg.clone(), 1, BLOCK, &fx.decoded(&src)));
    let (loss, got) = loss_and_grads(&fx.model(&src));
    // Same values summed in a different order by different kernels: loss within 1e-4 relative, adapter gradients
    // within cosine 0.9999.
    assert!((loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    let mut nonzero = 0;
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        if w.iter().all(|x| *x == 0.0) {
            continue;
        }
        nonzero += 1;
        let c = cosine(g, w);
        assert!(c > 0.9999, "{name}: adapter gradient cosine {c}");
    }
    assert!(nonzero >= 4, "the adapters received gradient ({nonzero} tensors)");
}

#[test]
fn lora_over_the_mixed_base_trains_and_its_adapter_saves_and_reloads() {
    let fx = fixture("train");
    let src = fx.source();
    let m = fx.model(&src);
    let (tokens, targets) = batch();
    m.set_batch(&tokens, &targets);
    let before = m.forward();
    for step in 1..=60 {
        m.zero_grads();
        m.forward();
        m.backward();
        model::Model::adamw_step(&m, step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let after = m.forward();
    assert!(after < before * 0.8, "LoRA over the mixed base did not learn: {before} -> {after}");
    let live = m.logits_all(&tokens);

    let adapter = fx.dir.join("adapter.safetensors");
    qwen3::lora::save_adapter(adapter.to_str().unwrap(), &m, "test/adapter", "test/base", None).expect("save the adapter");
    let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).unwrap();
    assert!(st.tensors.keys().all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")), "the adapter file holds only adapter tensors");

    // Reload: the saved factors over the same base, converted again into a mixed file, reproduce the trained model.
    let mut values = fx.init.clone();
    for (name, v) in &st.tensors {
        values.insert(name.clone(), v.clone());
    }
    write_mixed(&fx.cfg, &fx.dir, &values, &fx.dir.join("reloaded.safetensors"));
    let reloaded_src = PackedSource::identity(Box::new(PackedReader::open(&fx.dir.join("reloaded.safetensors")).unwrap()), CodecRegistry::reference()).unwrap();
    let reloaded = fx.model(&reloaded_src).logits_all(&tokens);
    let mean_abs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    assert!(mean_abs(&live, &reloaded) < 1e-5, "the reloaded adapter does not reproduce the trained logits: {}", mean_abs(&live, &reloaded));

    // Folding it into the decoded base gives the same function, and the untouched base does not.
    let mut folded = fx.decoded(&src);
    folded.retain(|n, _| !n.contains(".lora_"));
    qwen3::lora::fold_adapter_into(&mut folded, adapter.to_str().unwrap()).expect("fold");
    let plain_cfg = QwenConfig { lora: None, ..fx.cfg.clone() };
    let plain = Qwen::new(plain_cfg, 1, BLOCK, &folded).logits_all(&tokens);
    assert!(mean_abs(&live, &plain) < 1e-3, "folded base differs from the live adapter forward: {}", mean_abs(&live, &plain));
    let untouched = fx.model(&src).logits_all(&tokens);
    assert!(mean_abs(&live, &untouched) > 1e-3, "training left the logits where they were");
}
