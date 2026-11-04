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
    let cfg = QwenConfig { lora: Some(LoraCfg::new(2, 2.0, targets.clone())), ..QwenConfig::tiny() };
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
    assert_eq!(config["task_type"], "CAUSAL_LM");
    let modules: Vec<&str> = config["target_modules"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(modules.len(), 3 * cfg.n_layers as usize, "every targeted linear is named in full");
    let mut leaves: Vec<&str> = modules.iter().map(|m| m.rsplit('.').next().unwrap()).collect();
    leaves.sort();
    leaves.dedup();
    assert_eq!(leaves, ["down_proj", "q_proj", "v_proj"]);

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

/// Pairs of different rank and scale share one adapter file: each is padded
/// to the largest rank with its own `alpha / r` folded into `B`, so PEFT's
/// single `lora_alpha / r` applies every pair's own delta.
#[test]
fn pairs_of_different_rank_and_scale_keep_their_own_delta() {
    use qwen3::export::{PeftAdapter, PeftLora};
    let pair = |module: &str, r: usize, alpha: f32, n_in: usize, n_out: usize| PeftLora {
        module: module.into(),
        r,
        alpha,
        a: (0..r * n_in).map(|i| (i as f32 * 0.37).sin()).collect(),
        b: (0..n_out * r).map(|i| (i as f32 * 0.91).cos()).collect(),
    };
    let loras = vec![pair("tower.fc", 2, 8.0, 3, 4), pair("decoder.q_proj", 4, 4.0, 5, 2)];
    let delta = |l: &PeftLora, scale: f32| -> Vec<f32> {
        let (n_in, n_out) = (l.a.len() / l.r, l.b.len() / l.r);
        (0..n_out * n_in).map(|i| scale * (0..l.r).map(|k| l.b[(i / n_in) * l.r + k] * l.a[k * n_in + i % n_in]).sum::<f32>()).collect()
    };
    let want: Vec<Vec<f32>> = loras.iter().map(|l| delta(l, l.alpha / l.r as f32)).collect();
    let dir = std::env::temp_dir().join(format!("brain-peft-mixed-{}", std::process::id()));
    PeftAdapter { base_model: "acme/base".into(), task_type: None, loras, modules_to_save: vec!["head".into()], saved: vec![("head.weight".into(), vec![2], vec![1.0, 2.0])] }.write(&dir).unwrap();
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json")).unwrap()).unwrap();
    let (r, alpha) = (config["r"].as_u64().unwrap() as usize, config["lora_alpha"].as_f64().unwrap() as f32);
    assert_eq!((r, config["modules_to_save"].clone()), (4, serde_json::json!(["head"])));
    let st = checkpoint::st::load_safetensors(dir.join("adapter_model.safetensors").to_str().unwrap()).unwrap();
    assert_eq!(st.tensors["base_model.model.head.weight"], vec![1.0, 2.0], "a saved module's tensor under PEFT's prefix");
    for (module, want) in ["tower.fc", "decoder.q_proj"].iter().zip(&want) {
        let a = st.tensors[&format!("base_model.model.{module}.lora_A.weight")].clone();
        let b = st.tensors[&format!("base_model.model.{module}.lora_B.weight")].clone();
        let got = delta(&PeftLora { module: module.to_string(), r, alpha, a, b }, alpha / r as f32);
        let worst = got.iter().zip(want).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "{module}: delta differs by {worst}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
