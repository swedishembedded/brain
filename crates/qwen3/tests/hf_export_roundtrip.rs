// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A checkpoint exported as a `transformers` directory reads back as the
//! same model: every decoder variant (Qwen3 with a tied head, Qwen2 with q/k/v
//! bias, Llama with linear and llama3 RoPE scaling), sharded or not, gives
//! back its configuration and every tensor - bit for bit at F32, to the
//! format's own rounding at BF16.

use checkpoint::TensorSource;
use model::rope_scaling::RopeScaling;
use qwen3::export::{export_hf, hf_config, HfDtype, HfExport};
use qwen3::QwenConfig;

fn variants() -> Vec<(&'static str, QwenConfig)> {
    let t = QwenConfig::tiny();
    vec![
        ("qwen3", t.clone()),
        ("qwen2", QwenConfig { qk_norm: false, attn_bias: true, tie_embeddings: false, ..t.clone() }),
        ("llama-linear", QwenConfig { qk_norm: false, tie_embeddings: false, rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }), ..t.clone() }),
        (
            "llama3",
            QwenConfig {
                qk_norm: false,
                tie_embeddings: false,
                rope_scaling: Some(RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 8 }),
                ..t
            },
        ),
    ]
}

fn read(src: &dyn TensorSource, name: &str) -> Vec<f32> {
    let mut out = Vec::new();
    assert!(src.with_tensor(name, &mut |x| out.extend_from_slice(x)), "{name} missing");
    out
}

#[test]
fn every_variant_round_trips_through_a_transformers_directory() {
    for (label, cfg) in variants() {
        let weights = qwen3::init_weights(&cfg, 7);
        for (dtype, shard_bytes) in [(HfDtype::F32, u64::MAX), (HfDtype::Bf16, 2_000)] {
            let dir = std::env::temp_dir().join(format!("brain-hf-export-{label}-{dtype:?}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            export_hf(&weights, &cfg, &dir, &HfExport { dtype, shard_bytes, tokenizer_dir: None }).unwrap();
            assert_eq!(dir.join("model.safetensors.index.json").exists(), shard_bytes != u64::MAX, "{label}: sharded as asked");

            let (back, src) = qwen3::open_checkpoint(dir.to_str().unwrap()).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(QwenConfig { block_size: cfg.block_size, ..back }, cfg, "{label}: the configuration reads back");
            for (name, _) in cfg.param_list() {
                let want: Vec<f32> = match dtype {
                    HfDtype::F32 => weights[&name].clone(),
                    _ => weights[&name].iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect(),
                };
                assert!(read(&*src, &name) == want, "{label} {dtype:?}: {name} differs");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// A configuration no `transformers` class describes is refused, not
/// exported under the nearest name.
#[test]
fn an_unrepresentable_configuration_is_refused() {
    let both = QwenConfig { attn_bias: true, ..QwenConfig::tiny() };
    assert!(hf_config(&both, HfDtype::F32).is_err());
    let lora = QwenConfig { lora: Some(qwen3::LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    assert!(hf_config(&lora, HfDtype::F32).unwrap_err().contains("adapter"));
}
