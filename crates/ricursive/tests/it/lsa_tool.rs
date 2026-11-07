// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generic D-LSA tooling on Ripple: attention capture, restriction to the global layers, distillation and the
//! indexer artifact.
//!
//! Swedish Embedded AB implements long-context serving whose attention reads a
//! bounded set of keys per token for its clients. If your team needs
//! expertise in sparse attention for production LLM serving, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::path::Path;

use data::rng::Lcg;
use model::lsa::capture::{self, AttentionCapture, DistillOptions};
use model::lsa::indexer::Indexer;
use model::lsa::LsaConfig;
use ricursive::{open_safetensors, RicursiveConfig, RicursiveEngine, ServeConfig};

use crate::serve::{engine_with, stream, tiny_config, tiny_dir, weights};

fn tiny_engine(dir: &str) -> RicursiveEngine {
    engine_with(dir, ServeConfig { sparse_attention: true, ..ServeConfig::new(4, 200, 96, 2, 0, 8) })
}

fn lsa_config(top_p: usize, top_k: usize) -> LsaConfig {
    LsaConfig { index_dim: 4, pool_size: 1, bucket_size: 2, bucket_cap: 4, cluster_block: 8, top_p, top_k, sink: 1, window: 8, early: None }
}

/// `norm(x) * gain` rows, in f64.
fn rmsnorm(x: &[f32], gain: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    x.chunks(dim)
        .flat_map(|row| {
            let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ms + eps as f64).sqrt();
            row.iter().zip(gain.iter().cycle()).map(|(v, g)| (*v as f64 * inv) as f32 * g).collect::<Vec<_>>()
        })
        .collect()
}

/// `x[rows, k] @ w[n, k]^T`.
fn linear(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
    let n = w.len() / k;
    x.chunks(k).flat_map(|row| (0..n).map(|o| row.iter().zip(&w[o * k..(o + 1) * k]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32).collect::<Vec<_>>()).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The capture of a global layer is the layer's normed, unrotated queries and keys, as the step-1 forward computes
/// them from the layer's input; a sliding layer is not captured.
#[test]
fn the_capture_of_a_global_layer_is_its_normed_unrotated_queries_and_keys() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let ids = stream(3, 20, cfg.vocab_size as u32);
    let mut e = tiny_engine(&dir);
    assert_eq!(e.sparse_capable(), [false, false, true]);
    assert_eq!(e.index_geometry().rope_base, 0.0, "global layers apply no RoPE");
    let all: Vec<u32> = (0..ids.len() as u32).collect();
    let captured = e.capture_attention(&ids, &all).unwrap();
    assert!(captured[0].is_none() && captured[1].is_none());
    let got = captured[2].as_ref().unwrap();

    // The layer's input is the previous layer's output of the step-1 forward.
    let w = weights(&dir, &cfg);
    let out = ricursive::RicursiveModel::new(cfg.clone(), &w, None).forward(&ids).unwrap();
    let st = checkpoint::safetensors::read(&format!("{dir}/ricursive_tiny_weights.safetensors")).unwrap();
    let get = |name: &str| -> &[f32] { &st.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no {name}")).data };
    let (h, hd) = (cfg.hidden_size, cfg.head_dim);
    let xn = rmsnorm(&out.layer_outputs[1], get("model.layers.2.input_layernorm.weight"), h, cfg.rms_norm_eps);
    let q = rmsnorm(&linear(&xn, get("model.layers.2.self_attn.q_proj.weight"), h), get("model.layers.2.self_attn.q_norm.weight"), hd, cfg.rms_norm_eps);
    let k = rmsnorm(&linear(&xn, get("model.layers.2.self_attn.k_proj.weight"), h), get("model.layers.2.self_attn.k_norm.weight"), hd, cfg.rms_norm_eps);
    let (dq, dk) = (max_diff(&got.q, &q), max_diff(&got.k, &k));
    eprintln!("capture vs step-1 forward: queries max_abs {dq:.3e}, keys {dk:.3e}");
    assert_eq!((got.q.len(), got.k.len()), (ids.len() * cfg.q_dim(), ids.len() * cfg.kv_dim()));
    assert!(dq < 1e-4 && dk < 1e-4, "queries {dq}, keys {dk}");
}

/// Random, PCA and distilled indexers cover the global layers only.
#[test]
fn random_pca_and_distilled_indexers_hold_none_for_the_sliding_layers() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let mut e = tiny_engine(&dir);
    let g = e.index_geometry();
    let random = Indexer::random_for(lsa_config(3, 6), g, 1, &e.sparse_capable()).unwrap();
    assert_eq!(random.layers.iter().map(Option::is_some).collect::<Vec<_>>(), [false, false, true]);
    let docs: Vec<Vec<u32>> = (0..2).map(|i| stream(10 + i, 40, cfg.vocab_size as u32)).collect();
    let pca = capture::pca_indexer(&mut e, lsa_config(3, 6), &docs).unwrap();
    assert_eq!(pca.layers.iter().map(Option::is_some).collect::<Vec<_>>(), [false, false, true]);
    let docs: Vec<Vec<u32>> = (0..2).map(|i| stream(20 + i, 48, cfg.vocab_size as u32)).collect();
    let d = capture::distill(&mut e, &pca, &docs, &DistillOptions { epochs: 1, queries: 8, ..DistillOptions::default() }, &mut Lcg::new(3), |_, _, _| {}).unwrap();
    let distilled = d.indexer();
    assert_eq!(distilled.layers.iter().map(Option::is_some).collect::<Vec<_>>(), [false, false, true]);
    // A model with no sparse-capable layer has nothing to index.
    assert!(Indexer::random_for(lsa_config(3, 6), g, 1, &[false, false, false]).is_err());
}

fn corpus(vocab: u32, seed0: u32, n: usize, len: usize) -> Vec<Vec<u32>> {
    (0..n as u32).map(|i| stream(seed0 + i, len, vocab)).collect()
}

/// Distilled on a handful of synthetic texts, the indexer keeps more of the model's attention than a random one at
/// the same budget, on held-out texts.
#[test]
fn a_distilled_indexer_keeps_more_attention_than_a_random_one_at_equal_budget() {
    let Some(dir) = tiny_dir() else { return };
    let (index_dim, top_p, top_k, train_docs, epochs, lr) = (8usize, 3usize, 6usize, 24usize, 10usize, 5e-3f32);
    let cfg = tiny_config(&dir);
    let vocab = cfg.vocab_size as u32;
    let mut e = tiny_engine(&dir);
    let lsa = LsaConfig { index_dim, ..lsa_config(top_p, top_k) };
    let train = corpus(vocab, 100, train_docs, 64);
    let held = corpus(vocab, 900, 8, 64);
    let init = capture::pca_indexer(&mut e, lsa, &train[..2]).unwrap();
    let opts = DistillOptions { epochs, queries: 40, lr, ..DistillOptions::default() };
    let mut kls = Vec::new();
    let d = capture::distill(&mut e, &init, &train, &opts, &mut Lcg::new(7), |_, _, kl| kls.push(kl)).unwrap();
    let distilled = d.indexer();
    let random = Indexer::random_for(lsa, e.index_geometry(), 5, &e.sparse_capable()).unwrap();
    let r = capture::retention_report(&mut e, &held, &[&random, &init, &distilled], &lsa, 40, &mut Lcg::new(11)).unwrap();
    eprintln!("KL first {:.4} last {:.4}; retained share: window {:.4}, random {:.4}, pca {:.4}, distilled {:.4}", kls[0], kls[kls.len() - 1], r.local, r.indexers[0], r.indexers[1], r.indexers[2]);
    assert!(kls[kls.len() - 1] < kls[0], "the objective went down");
    assert!(r.indexers[2] > r.indexers[0], "distilled {} vs random {}", r.indexers[2], r.indexers[0]);
}

fn lsa_runs_here() -> bool {
    let runs = gpu_core::testgpu::dev(&[]).caps().workgroup_reductions;
    if !runs {
        brain_testutil::skip_unavailable("LSA: this backend does not run workgroup-barrier kernels (GPU-only)");
    }
    runs
}

/// With the distilled indexer attached, the model's next-token loss stays close to dense (the gap to the random
/// indexer's is printed).
#[test]
fn sparse_logits_with_a_distilled_indexer_stay_close_to_dense() {
    let Some(dir) = tiny_dir() else { return };
    if !lsa_runs_here() {
        return;
    }
    let cfg = tiny_config(&dir);
    let vocab = cfg.vocab_size as u32;
    let mut e = tiny_engine(&dir);
    let lsa = lsa_config(2, 4);
    let train = corpus(vocab, 100, 8, 64);
    let held = corpus(vocab, 900, 4, 64);
    let init = capture::pca_indexer(&mut e, lsa, &train[..2]).unwrap();
    let opts = DistillOptions { epochs: 40, queries: 40, lr: 2e-2, ..DistillOptions::default() };
    let distilled = capture::distill(&mut e, &init, &train, &opts, &mut Lcg::new(7), |_, _, _| {}).unwrap().indexer();
    let random = Indexer::random_for(lsa, e.index_geometry(), 5, &e.sparse_capable()).unwrap();
    let dense = capture::tail_loss(&mut e, &held, 24);
    let mut gaps = Vec::new();
    for ix in [&random, &distilled] {
        e.set_lsa(ix).unwrap();
        e.set_lsa_dense_until(0);
        gaps.push((capture::tail_loss(&mut e, &held, 24) - dense).abs());
    }
    eprintln!("tail loss: dense {dense:.4}; |sparse - dense| random {:.4}, distilled {:.4}", gaps[0], gaps[1]);
    assert!(gaps[1] < 0.25, "the distilled indexer's loss strays {} from dense", gaps[1]);
}

/// The real layers 0 and 4: capture works, and a PCA indexer for the global layer builds and round-trips through
/// the artifact with RoPE base 0 and only that layer sparse. (Layer 4 of the checkpoint is layer 1 of the two-layer
/// mapping.)
#[test]
fn a_pca_indexer_for_the_real_global_layer_round_trips_through_the_artifact() {
    let checkpoint = brain_testutil::testdata("ricursive/ripple-65b-partial/layers-0-4.safetensors");
    let config = brain_testutil::testdata("ricursive/ripple-65b-partial/config.json");
    let golden = brain_testutil::testdata("golden/ricursive/real");
    for path in [&checkpoint, &config] {
        if !Path::new(path).exists() {
            brain_testutil::skip(&format!("{path} absent: place the partial Ripple-65B checkpoint under testdata/ricursive/ripple-65b-partial/"));
            return;
        }
    }
    let taps = checkpoint::safetensors::read(&format!("{golden}/ricursive_real_reduced.safetensors"));
    let Ok(taps) = taps else {
        brain_testutil::skip(&format!("{golden} absent: run tools/goldens/ricursive_dump_reference.py real"));
        return;
    };
    let prompt: Vec<u32> = taps.iter().find(|t| t.name == "input_ids").unwrap().data.iter().map(|&v| v as u32).collect();
    let doc: Vec<u32> = prompt.iter().cycle().take(40).copied().collect();

    let layers = [0usize, 4];
    let cfg = RicursiveConfig::from_file(&config).unwrap().truncated(&layers).unwrap();
    let weights = ricursive::open_shared_checkpoint(&checkpoint, &cfg, &layers).unwrap();
    let serve = ServeConfig { sparse_attention: true, ..ServeConfig::new(8, 16, 64, 1, 0, 8) };
    let mut e = RicursiveEngine::from_source(cfg.clone(), weights, serve, None).unwrap();
    assert_eq!(e.sparse_capable(), [false, true]);

    let captured = e.capture_attention(&doc, &[5, 17, 39]).unwrap();
    assert!(captured[0].is_none());
    let layer = captured[1].as_ref().unwrap();
    assert_eq!((layer.q.len(), layer.k.len()), (3 * cfg.q_dim(), doc.len() * cfg.kv_dim()));
    assert!(layer.q.iter().chain(&layer.k).all(|v| v.is_finite()) && layer.k.iter().any(|v| *v != 0.0));

    let lsa = LsaConfig { index_dim: 16, pool_size: 1, bucket_size: 2, bucket_cap: 4, cluster_block: 8, top_p: 8, top_k: 16, sink: 1, window: 8, early: None };
    let indexer = capture::pca_indexer(&mut e, lsa, &[doc]).unwrap();
    assert_eq!(indexer.geometry.rope_base, 0.0);
    assert!(indexer.layers[0].is_none() && indexer.layers[1].is_some());

    let file = tempfile::NamedTempFile::new().unwrap();
    let path = file.path().to_str().unwrap();
    indexer.save(path).unwrap();
    let header = checkpoint::st::load_safetensors(path).unwrap().config();
    assert_eq!(header["geometry"]["rope_base"], 0.0);
    assert_eq!(header["sparse_layers"], serde_json::json!([1]));
    assert_eq!(Indexer::load(path).unwrap(), indexer);
    // The artifact attaches to the engine it was built for (where the device runs the kernels).
    match e.set_lsa(&indexer) {
        Ok(()) | Err(_) if !gpu_core::testgpu::dev(&[]).caps().workgroup_reductions => {}
        Ok(()) => {}
        Err(e) => panic!("attaching the round-tripped artifact: {e}"),
    }
}

#[test]
fn the_shared_opener_validates_a_directory_checkpoint_like_a_file() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    // A directory holding the weights file under the sharded-checkpoint name.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::copy(format!("{dir}/ricursive_tiny_weights.safetensors"), tmp.path().join("model.safetensors")).unwrap();
    std::fs::copy(format!("{dir}/config.json"), tmp.path().join("config.json")).unwrap();
    assert!(ricursive::open_checkpoint(tmp.path().to_str().unwrap(), &cfg, &layers).is_ok());
    assert!(open_safetensors(tmp.path().join("model.safetensors").to_str().unwrap(), &cfg, &layers).is_ok());
}
