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
            export_hf(&weights, &cfg, &dir, &HfExport { dtype, shard_bytes, ..Default::default() }).unwrap();
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

/// An adapter folded in on the way out gives the checkpoint brain's own fold
/// gives: the weights it targets carry `(alpha/r)·B·A`, the rest are the base.
#[test]
fn a_folded_adapter_exports_as_the_folded_weights() {
    use model::Model;
    let base_cfg = QwenConfig::tiny();
    let lora_cfg = QwenConfig { lora: Some(qwen3::LoraCfg { rank: 2, alpha: 6.0, targets: vec!["wq".into(), "up".into()] }), ..base_cfg.clone() };
    let init = qwen3::init_weights(&lora_cfg, 9);
    let model = qwen3::Qwen::new(lora_cfg.clone(), 1, 4, &init);
    for name in model.param_names().into_iter().filter(|n| n.contains(".lora_")) {
        let n = model.read_weight(&name).len();
        model.write_weight(&name, &(0..n).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.02).collect::<Vec<_>>());
    }
    let dir = std::env::temp_dir().join(format!("brain-hf-export-fold-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let adapter = dir.join("adapter.safetensors");
    qwen3::lora::save_adapter(adapter.to_str().unwrap(), &model, "acme/lora", "acme/base", None).unwrap();

    let base: std::collections::HashMap<String, Vec<f32>> = base_cfg.param_list().into_iter().map(|(n, _)| (n.clone(), init[&n].clone())).collect();
    let out = dir.join("hf");
    export_hf(&base, &base_cfg, &out, &HfExport { dtype: HfDtype::F32, adapter: Some(adapter.to_str().unwrap()), ..Default::default() }).unwrap();

    let mut want = base.clone();
    qwen3::lora::fold_adapter_into(&mut want, adapter.to_str().unwrap()).unwrap();
    let (_, src) = qwen3::open_checkpoint(out.to_str().unwrap()).unwrap();
    let mut changed = 0;
    for (name, w) in &want {
        let got = read(&*src, name);
        let worst = got.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "{name}: {worst}");
        changed += (w != &base[name]) as usize;
    }
    assert_eq!(changed, 2 * base_cfg.n_layers as usize, "exactly the targeted weights moved");
    let _ = std::fs::remove_dir_all(&dir);
}
