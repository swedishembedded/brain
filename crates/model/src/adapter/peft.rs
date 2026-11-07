// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for moving fine-tuned adapters between training stacks and the
// ecosystem's tools without a converter per model family, for its clients. If your team needs expertise in
// parameter-efficient fine-tuning interchange then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A brain LoRA adapter file as a PEFT adapter directory (`adapter_model.safetensors` + `adapter_config.json`),
//! so `peft.PeftModel.from_pretrained` over the HF checkpoint computes what brain computes.
//!
//! The writer knows nothing about an architecture: a model supplies the map from its base weight names to HF
//! module paths (a model whose tensor names already are the HF ones strips `.weight`).

use std::path::Path;

use serde_json::{json, Value};

use super::kind::{LORA_A, LORA_B};

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// A LoRA adapter as `model::lora::device_adapter::save_adapter` writes it: rank, alpha, the
/// base it was trained on, and each targeted base weight's `A [r, in]` and
/// `B [out, r]`, sorted by base name.
pub struct LoraFile {
    pub r: usize,
    pub alpha: f32,
    pub base: Option<String>,
    pub pairs: Vec<(String, Vec<f32>, Vec<f32>)>,
}

pub fn read_lora(adapter_path: &str) -> Result<LoraFile, String> {
    let mut st = checkpoint::st::load_safetensors(adapter_path).map_err(|e| format!("{adapter_path}: {e}"))?;
    let card = st.card().ok_or_else(|| format!("{adapter_path}: no ModelCard"))?;
    let adapter = card.adapter.as_ref().ok_or_else(|| format!("{adapter_path}: its card describes no adapter"))?;
    if adapter.kind != "lora" {
        return Err(format!("{adapter_path}: a {:?} adapter is not a LoRA", adapter.kind));
    }
    let r = adapter.rank.filter(|&r| r > 0).ok_or_else(|| format!("{adapter_path}: the adapter has no rank"))? as usize;
    let mut bases: Vec<String> = st.tensors.keys().filter_map(|n| n.strip_suffix(LORA_A)).map(str::to_string).collect();
    bases.sort();
    if bases.is_empty() {
        return Err(format!("{adapter_path}: no adapter pairs"));
    }
    let mut pairs = Vec::with_capacity(bases.len());
    for base in bases {
        let a = st.tensors.remove(&format!("{base}{LORA_A}")).expect("listed");
        let b = st.tensors.remove(&format!("{base}{LORA_B}")).ok_or_else(|| format!("{adapter_path}: {base}{LORA_A} has no {LORA_B}"))?;
        if a.len() % r != 0 || b.len() % r != 0 {
            return Err(format!("{adapter_path}: {base}'s factors are not rank-{r} matrices"));
        }
        pairs.push((base, a, b));
    }
    Ok(LoraFile { r, alpha: adapter.alpha.unwrap_or(r as f32), base: adapter.base.clone(), pairs })
}

/// One LoRA pair under the HF module it applies to: `W += (alpha / r) B A`
/// with `A [r, in]` and `B [out, r]`.
pub struct PeftLora {
    /// The HF module path, e.g. `model.layers.0.self_attn.q_proj`.
    pub module: String,
    pub r: usize,
    pub alpha: f32,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/// A PEFT LoRA adapter directory's content: LoRA pairs, plus whole modules
/// trained in full (PEFT's `modules_to_save`, their tensors under HF names).
pub struct PeftAdapter {
    pub base_model: String,
    /// PEFT's `task_type` (`CAUSAL_LM` for a decoder); `None` leaves it out,
    /// for a model PEFT has no task wrapper for.
    pub task_type: Option<String>,
    pub loras: Vec<PeftLora>,
    /// Full-trained modules' HF paths, and every tensor of theirs as
    /// `(HF tensor name, shape, values)`.
    pub modules_to_save: Vec<String>,
    pub saved: Vec<(String, Vec<u64>, Vec<f32>)>,
}

impl PeftAdapter {
    /// Write `adapter_model.safetensors` and `adapter_config.json` into `out`,
    /// so `peft.PeftModel.from_pretrained` over the HF checkpoint computes
    /// exactly what brain computes. Target modules are named in full, so a
    /// leaf name shared by two towers never targets the other one. Pairs of
    /// different rank or scale are written at the largest rank with
    /// `lora_alpha == r` (scale 1): each pair's own `alpha / r` is folded
    /// into its `B` and the extra rank is zero, the same delta in any PEFT
    /// version, without per-module patterns.
    pub fn write(&self, out: &Path) -> Result<(), String> {
        let first = self.loras.first().ok_or("PEFT export: no LoRA pairs")?;
        let uniform = self.loras.iter().all(|l| l.r == first.r && l.alpha == first.alpha);
        let r = self.loras.iter().map(|l| l.r).max().expect("non-empty");
        let alpha = if uniform { first.alpha } else { r as f32 };
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        let mut modules: Vec<String> = Vec::new();
        for l in &self.loras {
            if l.r == 0 || l.a.len() % l.r != 0 || l.b.len() % l.r != 0 {
                return Err(format!("PEFT export: {}'s factors are not rank-{} matrices", l.module, l.r));
            }
            let (n_in, n_out) = (l.a.len() / l.r, l.b.len() / l.r);
            let (a, b) = if uniform {
                (l.a.clone(), l.b.clone())
            } else {
                let mut a = l.a.clone();
                a.resize(r * n_in, 0.0);
                let scale = l.alpha / l.r as f32;
                let mut b = vec![0f32; n_out * r];
                for o in 0..n_out {
                    for k in 0..l.r {
                        b[o * r + k] = l.b[o * l.r + k] * scale;
                    }
                }
                (a, b)
            };
            tensors.push((format!("base_model.model.{}.lora_A.weight", l.module), vec![r as u64, n_in as u64], a));
            tensors.push((format!("base_model.model.{}.lora_B.weight", l.module), vec![n_out as u64, r as u64], b));
            modules.push(l.module.clone());
        }
        for (name, shape, v) in &self.saved {
            if !self.modules_to_save.iter().any(|m| name.starts_with(&format!("{m}."))) {
                return Err(format!("PEFT export: {name} belongs to no module to save"));
            }
            tensors.push((format!("base_model.model.{name}"), shape.clone(), v.clone()));
        }
        std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
        checkpoint::st::save_safetensors(&out.join("adapter_model.safetensors").to_string_lossy(), &tensors, &Value::Null, None).map_err(|e| format!("{}: {e}", out.display()))?;
        let mut config = json!({
            "peft_type": "LORA",
            "base_model_name_or_path": self.base_model,
            "r": r,
            "lora_alpha": alpha,
            "target_modules": modules,
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": false,
            "inference_mode": true,
        });
        if !self.modules_to_save.is_empty() {
            config["modules_to_save"] = json!(self.modules_to_save);
        }
        if let Some(t) = &self.task_type {
            config["task_type"] = json!(t);
        }
        write_json(&out.join("adapter_config.json"), &config)
    }
}

/// A brain LoRA adapter file (`model::lora::device_adapter::save_adapter`'s
/// format) as PEFT pairs, each base weight name mapped to its HF module by
/// `hf_module` (`None`: an error naming the weight); also the base its card
/// records.
pub fn adapter_pairs(adapter_path: &str, hf_module: &dyn Fn(&str) -> Option<String>) -> Result<(Vec<PeftLora>, Option<String>), String> {
    let LoraFile { r, alpha, base, pairs } = read_lora(adapter_path)?;
    let loras = pairs
        .into_iter()
        .map(|(name, a, b)| {
            let module = hf_module(&name).ok_or_else(|| format!("{adapter_path}: no HF module for {name}"))?;
            Ok(PeftLora { module, r, alpha, a, b })
        })
        .collect::<Result<_, String>>()?;
    Ok((loras, base))
}
