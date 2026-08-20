// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAGNOSTIC real-weight parity check for `video_embeddings_connector` -
//! the one component the real-generation pipeline routes EVERY block's
//! cross-attention context through, and whose real-weight forward has never
//! been checked against the reference at all (`dit_parity.rs::real_weight`'s
//! own `ltxv_real_dit_tiny_layers_matches_reference` deliberately runs with
//! `use_embeddings_connector: false`, a documented, tracked gap).
//!
//! Golden: `tools/goldens/ltxv_real_connector_dump_reference.py`, which
//! instantiates the reference's own `Embeddings1DConnector` directly (real
//! width: dim=4096, 8 layers, 128 registers, gated attention ON) loaded
//! with the REAL Q8_0 GGUF's connector weight subset, and dumps its output
//! on a fixed synthetic `[128, 4096]` input (right-padded, 20 real + 108
//! register-substituted positions) - the smallest input length that is
//! still a real multiple of the real register count.

use std::path::Path;

use checkpoint::gguf::MmapGguf;
use ltxv::block::{open_device, EmbeddingsConnector};
use ltxv::config::LtxDitConfig;
use ltxv::dit::dit_tensor_manifest;
use vae::blocks::Tensors;

const REPO: &str = "Lightricks/LTX-2.5";
const PREFIX: &str = "video_embeddings_connector";

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

/// The connector's own tensor subset, read straight off the real GGUF - the
/// same names [`dit_tensor_manifest`] emits for a connector-enabled config,
/// filtered to this one prefix (`LtxDit::forward` reads the rest of the
/// manifest too, but this test only exercises the connector standalone).
fn load_connector_weights(mg: &MmapGguf, cfg: &LtxDitConfig) -> Tensors {
    dit_tensor_manifest(cfg)
        .into_iter()
        .filter(|(name, _)| name.starts_with(PREFIX))
        .map(|(name, shape)| {
            let want: usize = shape.iter().product();
            let data = mg.tensor(&name).unwrap_or_else(|| panic!("real ltxv dit gguf: missing tensor {name}")).unwrap_or_else(|e| panic!("real ltxv dit gguf: {name}: {e}"));
            assert_eq!(data.len(), want, "real ltxv dit gguf: {name} has {} values, expected {want}", data.len());
            (name, (shape, data))
        })
        .collect()
}

#[test]
fn ltxv_real_connector_matches_reference() {
    let fx_path = brain_testutil::testdata("golden/ltxv/connector/connector_real.safetensors");
    if !Path::new(&fx_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_real_connector_dump_reference.py"));
        return;
    }
    let Some(gguf_path) = real_dit_gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };

    let mg = MmapGguf::open(&gguf_path).unwrap_or_else(|e| panic!("opening {gguf_path}: {e}"));
    let cfg = LtxDitConfig::ltx25_22b();

    let t0 = std::time::Instant::now();
    let w = load_connector_weights(&mg, &cfg);
    eprintln!("connector weight subset loaded ({} tensors) in {:.2}s", w.len(), t0.elapsed().as_secs_f64());

    let t = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let get = |name: &str| -> Vec<f32> { t.iter().find(|x| x.name == name).unwrap_or_else(|| panic!("no golden {name}")).data.clone() };
    let shape = |name: &str| -> Vec<usize> { t.iter().find(|x| x.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape.clone() };

    let hidden = get("hidden");
    let valid = get("valid");
    let s = shape("hidden")[0] as u32;
    let want_out = get("connector_out");

    let gpu = open_device(Some("gpu"));
    let connector = EmbeddingsConnector::on(
        gpu,
        &w,
        PREFIX,
        cfg.connector_inner_dim(),
        cfg.connector_num_attention_heads,
        cfg.connector_attention_head_dim,
        cfg.connector_num_layers,
        cfg.connector_num_learnable_registers,
        cfg.connector_apply_gated_attention,
        cfg.connector_norm_output,
        cfg.positional_embedding_theta,
        &cfg.connector_positional_embedding_max_pos,
        cfg.norm_eps,
    );

    let t1 = std::time::Instant::now();
    let got_out = connector.forward(&hidden, &valid, s);
    eprintln!("real-weight connector forward (s={s}, dim={}) ran in {:.2}s", cfg.connector_inner_dim(), t1.elapsed().as_secs_f64());

    let c = cosine(&got_out, &want_out);
    let m = max_abs(&got_out, &want_out);
    eprintln!("connector_out: cosine={c:.9}  max_abs={m:.3e}  n={}", got_out.len());
    assert!(c >= 0.999, "connector_out: cosine {c:.9} < 0.999");
}
