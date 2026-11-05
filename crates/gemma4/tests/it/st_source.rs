// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The checkpoint as LTX ships it (`gemma4-12b-with-proj-ltx-2.5-*.safetensors`)
//! read one layer at a time must encode exactly what the eager whole-file
//! import encodes: same names, same decode, same arithmetic.

use gemma4::{Gemma4Config, Gemma4Model, Precision};

/// A tiny-config checkpoint under the real file's own names (`model.`-prefixed
/// tower, bare `text_embedding_projection.*` head), small random weights.
fn write_checkpoint(path: &std::path::Path, cfg: &Gemma4Config) {
    let mut rng = data::rng::Lcg::random();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = gemma4::gemma4_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let len: usize = shape.iter().product();
            let values = (0..len).map(|_| rng.signed() * 0.1).collect();
            let name = if name.starts_with("text_embedding_projection.") { name } else { format!("model.{name}") };
            (name, shape.iter().map(|&d| d as u64).collect(), values)
        })
        .collect();
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
}

#[test]
fn a_streamed_safetensors_checkpoint_encodes_what_the_eager_import_encodes() {
    let cfg = Gemma4Config::tiny();
    let path = std::env::temp_dir().join(format!("gemma4-st-source-{}.safetensors", std::process::id()));
    write_checkpoint(&path, &cfg);
    let ids = [2u32, 17, 5, 9, 31, 4];

    let eager = gemma4::import_gemma4(checkpoint::safetensors::read(path.to_str().unwrap()).unwrap(), &cfg).unwrap();
    let want = Gemma4Model::new(cfg, eager, None).forward(&ids).hidden_states;

    let src = gemma4::open_safetensors(path.to_str().unwrap(), &cfg).unwrap();
    let got = gemma4::forward_streamed(&cfg, &src, None, Precision::Fp32, &ids).unwrap().hidden_states;
    let _ = std::fs::remove_file(&path);

    assert_eq!(got.len(), want.len(), "one hidden state per layer plus the embedding");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "hidden state {i}");
    }
}
