// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAGNOSTIC: does the actual real-generation code path
//! (`crate::dit::forward_q_streamed`, what `RealDit::forward` calls, and
//! what every `brain ltxv t2v --dit-config ltx25_22b` run actually
//! executes) agree with [`LtxDit::forward`] (the eager path every existing
//! real-weight parity gate in this crate replays against) on the SAME real
//! weights, SAME real config (gated attention + embeddings connector both
//! ON), SAME inputs?
//!
//! This gap exists because every other real-weight gate in this crate uses
//! `LtxDit::forward` (`dit_parity.rs::real_weight`, this crate's own
//! `connector_real_parity.rs`) - `forward_q_streamed`'s only prior coverage
//! (`block_weight_cache.rs`) uses `random_tiny_weights`, so it has never
//! been checked against ANYTHING at real weights, not even against its own
//! crate's eager twin.

use std::path::Path;

use checkpoint::gguf::MmapGguf;
use ltxv::block::QTier;
use ltxv::config::LtxDitConfig;
use ltxv::dit::{dit_tensor_manifest, forward_q_streamed, load_head_tensors_from_source, LtxDit};
use ltxv::gguf_src::LtxvGgufSource;
use vae::blocks::Tensors;

const REPO: &str = "Lightricks/LTX-2.5";
const LAYERS: u32 = 2;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn real_dit_gguf_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DIT") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("Q8_0") && n.ends_with(".gguf")))
        .filter(|p| {
            MmapGguf::open(&p.to_string_lossy())
                .ok()
                .and_then(|g| g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string))
                .as_deref()
                == Some(ltxv::import::GGUF_ARCHITECTURE)
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

fn load_all_weights(mg: &MmapGguf, cfg: &LtxDitConfig) -> Tensors {
    dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let want: usize = shape.iter().product();
            let data = mg.tensor(&name).unwrap_or_else(|| panic!("real ltxv dit gguf: missing tensor {name}")).unwrap_or_else(|e| panic!("real ltxv dit gguf: {name}: {e}"));
            assert_eq!(data.len(), want, "real ltxv dit gguf: {name} has {} values, expected {want}", data.len());
            (name, (shape, data))
        })
        .collect()
}

#[test]
fn streamed_forward_matches_eager_forward_on_real_weights_with_connector_on() {
    let Some(gguf_path) = real_dit_gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    if !Path::new(&gguf_path).exists() {
        brain_testutil::skip(&format!("gguf {gguf_path} does not exist"));
        return;
    }

    let cfg = LtxDitConfig { num_layers: LAYERS, ..LtxDitConfig::ltx25_22b() };
    cfg.assert_supported();
    assert!(cfg.use_embeddings_connector, "this test's whole point is proving the connector-enabled real path");

    // ---- a small, real-shaped input: t=8 tokens, context_len a real
    // multiple of the real connector's register count (128) -------------
    let (t, context_len) = (8usize, 128usize);
    let dim = cfg.cross_attention_dim as usize;
    let context: Vec<f32> = (0..context_len * dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let mut context_valid = vec![0f32; context_len];
    context_valid[..20].fill(1.0);
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let timesteps = vec![0.7f32; t];
    let keyframes_mask = vec![0f32; t];
    let f = 2usize;
    let (h, w) = (2usize, 2usize);
    let mut positions = vec![0f32; 3 * t * 2];
    let mut tok = 0usize;
    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                for (axis, v) in [fi, hi, wi].into_iter().enumerate() {
                    positions[(axis * t + tok) * 2] = v as f32;
                    positions[(axis * t + tok) * 2 + 1] = v as f32 + 1.0;
                }
                tok += 1;
            }
        }
    }

    // ---- eager: LtxDit::forward, int8 tier via forward_q (this crate's
    // existing convention for testing int8 with the eager weight map) ---
    let mg = MmapGguf::open(&gguf_path).unwrap_or_else(|e| panic!("opening {gguf_path}: {e}"));
    let t0 = std::time::Instant::now();
    let w_eager = load_all_weights(&mg, &cfg);
    eprintln!("eager weight subset loaded ({} tensors) in {:.2}s", w_eager.len(), t0.elapsed().as_secs_f64());
    let model = LtxDit::new(cfg, w_eager, None);
    let t1 = std::time::Instant::now();
    let out_eager = model.forward_q(&latent, &timesteps, &positions, &keyframes_mask, &context, context_len, t, &context_valid, QTier::Int8).out;
    eprintln!("eager forward_q (int8) ran in {:.2}s", t1.elapsed().as_secs_f64());

    // ---- streamed: forward_q_streamed, the ACTUAL real-generation path -
    let src = LtxvGgufSource::open(&gguf_path).unwrap_or_else(|e| panic!("opening {gguf_path} as a streaming source: {e}"));
    let head = load_head_tensors_from_source(&src, &cfg);
    let cache = Default::default();
    let t2 = std::time::Instant::now();
    let out_streamed = forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &latent, &timesteps, &positions, &keyframes_mask, &context, context_len, t, &context_valid, &cache);
    eprintln!("streamed forward_q_streamed (int8) ran in {:.2}s", t2.elapsed().as_secs_f64());

    let c = cosine(&out_eager, &out_streamed);
    let m = max_abs(&out_eager, &out_streamed);
    eprintln!("eager vs streamed, real weights, connector ON: cosine={c:.9}  max_abs={m:.3e}  n={}", out_eager.len());
    assert!(c >= 0.999, "streamed forward diverged from eager forward on the SAME real weights: cosine {c:.9}");
}
