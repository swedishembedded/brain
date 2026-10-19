// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for moving fine-tuned multimodal
// models between training and serving stacks for its clients. If your team
// needs expertise in vision-language model fine-tuning and deployment then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A tune directory ([`crate::train::Trainer::save_tune`]) as one PEFT
//! adapter over the Hugging Face checkpoint, so `transformers` + `peft`
//! serve what brain trained:
//!
//! * the decoder's LoRA pairs under the decoder's HF modules
//!   (`model.language_model.layers.N.…`);
//! * the tower's LoRA pairs under the tower's: a pair on the fused `qkv`
//!   becomes three, on `q_proj`, `k_proj` and `v_proj`, sharing `A` and each
//!   taking its row block of `B` - the same delta, exactly;
//! * a connector or tower trained in full as PEFT `modules_to_save`
//!   (`model.merger`, `model.vision_tower`), its tensors under their HF names.
//!
//! A decoder trained in full is no adapter (its every weight changed); its
//! weights stay in the tune directory and the export refuses it.

use std::collections::HashMap;
use std::path::Path;

use qwen3::export::{PeftAdapter, PeftLora};

use crate::config::MiniCpmVConfig;
use crate::train::{TUNE_DECODER_ADAPTER, TUNE_DECODER_FULL, TUNE_MANIFEST, TUNE_VISION};

/// The PEFT module that owns a canonical vision-side tensor when it trains in full.
fn module_to_save(name: &str) -> &'static str {
    if name.starts_with("merger.") {
        "model.merger"
    } else {
        "model.vision_tower"
    }
}

/// Split `v` into `n` equal row blocks.
fn row_blocks(v: &[f32], n: usize, what: &str) -> Result<Vec<Vec<f32>>, String> {
    if !v.len().is_multiple_of(n) {
        return Err(format!("{what}: {} values do not split into {n} row blocks", v.len()));
    }
    Ok(v.chunks(v.len() / n).map(<[f32]>::to_vec).collect())
}

/// Write the tune in `tune` (over the checkpoint in `base_dir`) as a PEFT
/// adapter directory `out`.
pub fn export_peft(tune: &Path, base_dir: &Path, out: &Path) -> Result<(), String> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(tune.join(TUNE_MANIFEST)).map_err(|e| format!("{}: {e}", tune.display()))?)
        .map_err(|e| format!("{}: {e}", tune.join(TUNE_MANIFEST).display()))?;
    if tune.join(TUNE_DECODER_FULL).is_file() {
        return Err(format!(
            "{}: the decoder trained in full, so this tune is no adapter - its decoder weights are {TUNE_DECODER_FULL}",
            tune.display()
        ));
    }
    let cfg = MiniCpmVConfig::from_dir(base_dir)?;
    let base_model = manifest["base"].as_str().ok_or_else(|| format!("{}: no base", tune.join(TUNE_MANIFEST).display()))?.to_string();
    let mut loras = Vec::new();
    let decoder = tune.join(TUNE_DECODER_ADAPTER);
    if decoder.is_file() {
        let hf = |n: &str| qwen35::import::hf_linear_name(n).and_then(|h| h.strip_suffix(".weight").map(str::to_string));
        loras.extend(qwen3::export::adapter_pairs(decoder.to_str().ok_or("non-UTF-8 tune path")?, &hf)?.0);
    }
    let (mut modules_to_save, mut saved) = (Vec::<String>::new(), Vec::new());
    let vision = tune.join(TUNE_VISION);
    if vision.is_file() {
        let p = vision.to_str().ok_or("non-UTF-8 tune path")?;
        let st = checkpoint::st::load_safetensors(p).map_err(|e| format!("{p}: {e}"))?;
        let vcfg = st.config();
        let sources: HashMap<String, Vec<String>> = crate::vision::hf_sources(&cfg).into_iter().collect();
        let srcs = |n: &str| sources.get(n).ok_or_else(|| format!("{p}: {n} is not a vision tensor of this base"));
        let mut names: Vec<&String> = st.tensors.keys().collect();
        names.sort();
        for name in names {
            let v = &st.tensors[name];
            if name.ends_with(".lora_b") {
                continue;
            }
            if let Some(base) = name.strip_suffix(".lora_a") {
                let (r, alpha) = match (vcfg["lora_rank"].as_u64(), vcfg["lora_alpha"].as_f64()) {
                    (Some(r), Some(a)) => (r as usize, a as f32),
                    _ => return Err(format!("{p}: LoRA factors without lora_rank and lora_alpha")),
                };
                let b = st.tensors.get(&format!("{base}.lora_b")).ok_or_else(|| format!("{p}: {name} has no lora_b"))?;
                let hf = srcs(base)?;
                for (src, b) in hf.iter().zip(row_blocks(b, hf.len(), base)?) {
                    let module = src.strip_suffix(".weight").ok_or_else(|| format!("{p}: {base} is not a weight"))?.to_string();
                    loras.push(PeftLora { module, r, alpha, a: v.clone(), b });
                }
                continue;
            }
            let module = module_to_save(name).to_string();
            if !modules_to_save.contains(&module) {
                modules_to_save.push(module);
            }
            let hf = srcs(name)?;
            for (src, block) in hf.iter().zip(row_blocks(v, hf.len(), name)?) {
                saved.push((src.clone(), vec![block.len() as u64], block));
            }
        }
    }
    if loras.is_empty() {
        return Err(format!(
            "{}: the tune trains no LoRA pair (only modules in full), and a PEFT LoRA adapter needs one",
            tune.display()
        ));
    }
    PeftAdapter { base_model, task_type: None, loras, modules_to_save, saved }.write(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_trained_tensors_belong_to_the_connector_or_the_tower() {
        assert_eq!(module_to_save("merger.fc1.weight"), "model.merger");
        assert_eq!(module_to_save("wm.fc1.weight"), "model.vision_tower");
        assert_eq!(row_blocks(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, "x").unwrap(), vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert!(row_blocks(&[1.0; 4], 3, "x").is_err());
    }
}
