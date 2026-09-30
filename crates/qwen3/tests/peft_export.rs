// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA adapter exported in PEFT's layout applies as brain applies it: read
//! back by brain's reader for third-party adapters (the same `.lora_A` /
//! `.lora_B` convention PEFT loads), it folds into the HF-named base exactly
//! as brain's own fold does into brain's names, and its `adapter_config.json`
//! carries the rank, alpha and target modules PEFT scales by.

use model::Model;
use qwen3::{LoraCfg, Qwen, QwenConfig};

#[test]
fn a_peft_export_folds_as_the_adapter_brain_trained() {
    let targets: Vec<String> = ["wq", "wv", "down"].iter().map(|s| s.to_string()).collect();
    // alpha == rank: PEFT keeps alpha in adapter_config.json, not in the
    // tensors, and the reader alone therefore scales by 1.
    let cfg = QwenConfig { lora: Some(LoraCfg { rank: 2, alpha: 2.0, targets: targets.clone() }), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 3);
    let model = Qwen::new(cfg.clone(), 1, 4, &init);
    // Non-zero factors on both sides, so the fold is not trivially the base.
    for name in model.param_names().into_iter().filter(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")) {
        let n = model.read_weight(&name).len();
        model.write_weight(&name, &(0..n).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.01).collect::<Vec<_>>());
    }
    let dir = std::env::temp_dir().join(format!("brain-peft-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let adapter = dir.join("adapter.safetensors");
    qwen3::lora::save_adapter(adapter.to_str().unwrap(), &model, "acme/tiny-lora", "acme/tiny", None).unwrap();

    let out = dir.join("peft");
    qwen3::export::export_peft(adapter.to_str().unwrap(), &out, None).unwrap();
    assert_eq!(serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap()).unwrap()["base_model_name_or_path"], "acme/tiny", "the base comes from the card");
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!((config["peft_type"].as_str(), config["r"].as_u64(), config["lora_alpha"].as_f64()), (Some("LORA"), Some(2), Some(2.0)));
    let mut modules: Vec<&str> = config["target_modules"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
    modules.sort();
    assert_eq!(modules, ["down_proj", "q_proj", "v_proj"]);

    // brain's fold, over brain's names...
    let base: std::collections::HashMap<String, Vec<f32>> = cfg.param_list().into_iter().filter(|(n, _)| !n.contains(".lora_")).map(|(n, _)| (n.clone(), init[&n].clone())).collect();
    let mut brain_folded = base.clone();
    qwen3::lora::fold_adapter_into(&mut brain_folded, adapter.to_str().unwrap()).unwrap();
    // ...and the PEFT file's, over the HF names it targets.
    let names = qwen3::hf::HfNames::CAUSAL_LM;
    let pairs = model::lora::read_external_adapter(out.join("adapter_model.safetensors").to_str().unwrap()).unwrap();
    assert_eq!(pairs.len(), 3 * cfg.n_layers as usize, "one pair per targeted linear");
    let mut peft_folded = base.clone();
    for p in &pairs {
        let hf = p.base_key.strip_prefix("base_model.model.").expect("PEFT's prefix");
        let brain = base.keys().find(|b| names.from_brain(b).as_deref() == Some(hf)).unwrap_or_else(|| panic!("{hf} targets no base weight"));
        p.add_delta(1.0, peft_folded.get_mut(brain).unwrap());
    }
    for (name, w) in &brain_folded {
        let worst = w.iter().zip(&peft_folded[name]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "{name}: folds differ by {worst}");
        if targets.iter().any(|t| name.contains(&format!(".{t}.")) || name.ends_with(&format!("{t}.weight"))) {
            assert_ne!(w, &base[name], "{name}: the adapter changed nothing");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
