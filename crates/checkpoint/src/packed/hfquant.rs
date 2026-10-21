// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading of quantized HuggingFace checkpoints for its clients. If
// your team needs expertise in serving compressed-tensors, ModelOpt and FP8 releases as published then you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a HuggingFace release *declares* about its quantization, parsed once.
//!
//! A release names its storage in `config.json`'s `quantization_config` (`quant_method`: `compressed-tensors`,
//! `fp8`, `modelopt`; `gptq`, `awq` and friends are recognised only to be refused by name) and, for NVIDIA
//! ModelOpt exports, in `hf_quant_config.json` (`quantization.quant_algo`, `quantized_layers`,
//! `exclude_modules`). [`HfQuant`] turns those documents into one answer per tensor: [`HfQuant::declared`] says
//! whether the weight `<module>.weight` is declared FP8, NVFP4 or left in its native dtype, so
//! [`PackedReader`](super::reader::PackedReader) builds each tensor's [`QuantEncoding`](super::QuantEncoding)
//! from the declaration plus the stored sibling tensors, and no model crate parses a quantization config.
//!
//! The declaration is checked against what is stored, never trusted over it: a weight declared FP8 whose dtype
//! is not `F8_E4M3` is an error naming both, not a silent reinterpretation.

use std::path::Path;

use serde_json::Value;

/// How one module's weight is declared to be stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeclaredWeight {
    /// Not quantized by the release (excluded, or not matched by any quantized group): native dtype.
    Native,
    /// 8-bit float weights with their scale tensors beside them (`weight_scale` or `weight_scale_inv`).
    /// `block` is the declared `weight_block_size` of a block-scaled release.
    Fp8 { block: Option<u32> },
    /// NVFP4 (E2M1 codes, E4M3 scale per `group`, f32 per-tensor scale).
    Nvfp4 { group: u32 },
    /// A method brain has no codec for; loading such a tensor fails with this message.
    Unsupported(String),
}

/// Quantization methods recognised only to refuse them by name: their packing (int32 words with zero points,
/// permutations, lookup tables) has no codec here.
const REFUSED_METHODS: [&str; 8] = ["gptq", "awq", "bitsandbytes", "marlin", "gptq_marlin", "awq_marlin", "hqq", "aqlm"];

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HfQuant {
    /// `quant_method` of the release (`compressed-tensors`, `fp8`, `modelopt`, ...), when it declares one.
    pub method: Option<String>,
    /// Declaration applied to every weight no pattern excludes (a whole-model `quant_algo`, or the weight
    /// format of a compressed-tensors config group).
    default: Option<DeclaredWeight>,
    /// ModelOpt `quantized_layers`: module (or module prefix, e.g. a fused-experts parent) to declaration.
    layers: Vec<(String, DeclaredWeight)>,
    /// Module name patterns the release leaves unquantized (`ignore` / `exclude_modules`); a trailing `*`
    /// matches any suffix, `re:` prefixed entries are regular expressions and are matched as plain prefixes of
    /// their literal head.
    excluded: Vec<String>,
    /// True when the declaration came with `quantized_layers`: then a module outside the list is native.
    layered: bool,
}

fn fp8_block(v: &Value) -> Option<u32> {
    let b = v.get("weight_block_size")?.as_array()?;
    match (b.first()?.as_u64()?, b.get(1)?.as_u64()?) {
        (r, c) if r == c && r > 0 => Some(r as u32),
        _ => None,
    }
}

fn modelopt_algo(algo: &str, group: Option<u64>) -> DeclaredWeight {
    match algo.to_ascii_uppercase().as_str() {
        "FP8" => DeclaredWeight::Fp8 { block: None },
        "NVFP4" | "W4A16_NVFP4" | "W4A4_NVFP4" => DeclaredWeight::Nvfp4 { group: group.unwrap_or(16) as u32 },
        "" | "MIXED_PRECISION" | "NONE" => DeclaredWeight::Native,
        other => DeclaredWeight::Unsupported(format!("ModelOpt quant_algo {other}")),
    }
}

impl HfQuant {
    /// Read the declaration of the checkpoint directory `dir` (`config.json` and `hf_quant_config.json` beside
    /// the weights). `Ok(None)` when the release declares nothing; a method brain cannot read is still `Ok`: the
    /// error is raised per tensor that needs it, so a release whose unsupported part is never asked for loads.
    pub fn from_dir(dir: &Path) -> Result<Option<HfQuant>, String> {
        let read = |file: &str| -> Result<Option<Value>, String> {
            match std::fs::read(dir.join(file)) {
                Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|e| format!("{}: {e}", dir.join(file).display())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(format!("{}: {e}", dir.join(file).display())),
            }
        };
        Ok(HfQuant::from_json(read("config.json")?.as_ref(), read("hf_quant_config.json")?.as_ref()))
    }

    /// [`Self::from_dir`] over already-parsed documents.
    pub fn from_json(config: Option<&Value>, hf_quant: Option<&Value>) -> Option<HfQuant> {
        let qc = config.and_then(|c| c.get("quantization_config").or_else(|| c.get("text_config").and_then(|t| t.get("quantization_config"))));
        let mut out = HfQuant::default();
        if let Some(h) = hf_quant.and_then(|h| h.get("quantization")) {
            out.method = Some("modelopt".to_string());
            let group = h.get("group_size").and_then(Value::as_u64);
            let algo = h.get("quant_algo").and_then(Value::as_str).unwrap_or("");
            out.default = Some(modelopt_algo(algo, group));
            if let Some(layers) = h.get("quantized_layers").and_then(Value::as_object) {
                out.layered = true;
                out.layers = layers
                    .iter()
                    .map(|(name, l)| (name.clone(), modelopt_algo(l.get("quant_algo").and_then(Value::as_str).unwrap_or(""), l.get("group_size").and_then(Value::as_u64).or(group))))
                    .collect();
            }
            out.excluded = h.get("exclude_modules").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
            return Some(out);
        }
        let qc = qc?;
        let method = qc.get("quant_method").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
        out.method = Some(method.clone());
        out.excluded = qc.get("ignore").or_else(|| qc.get("modules_to_not_convert")).and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
        out.default = Some(match method.as_str() {
            "fp8" => DeclaredWeight::Fp8 { block: fp8_block(qc) },
            "compressed-tensors" => Self::compressed_tensors_weight(qc),
            "modelopt" => modelopt_algo(qc.get("quant_algo").and_then(Value::as_str).unwrap_or(""), qc.get("group_size").and_then(Value::as_u64)),
            m if REFUSED_METHODS.contains(&m) => DeclaredWeight::Unsupported(format!("quantization method {m}")),
            "" => return None,
            m => DeclaredWeight::Unsupported(format!("quantization method {m}")),
        });
        Some(out)
    }

    /// The weight format of a compressed-tensors config: the first config group's `weights` entry.
    fn compressed_tensors_weight(qc: &Value) -> DeclaredWeight {
        let format = qc.get("format").and_then(Value::as_str).unwrap_or("");
        let weights = qc.get("config_groups").and_then(Value::as_object).and_then(|g| g.values().find_map(|g| g.get("weights")));
        let Some(w) = weights else { return DeclaredWeight::Native };
        let (bits, float) = (w.get("num_bits").and_then(Value::as_u64).unwrap_or(0), w.get("type").and_then(Value::as_str) == Some("float"));
        match (bits, float, format) {
            (8, true, _) => {
                let block = w.get("block_structure").and_then(Value::as_array).and_then(|b| b.first()?.as_u64()).map(|b| b as u32);
                DeclaredWeight::Fp8 { block }
            }
            (4, true, f) => DeclaredWeight::Unsupported(format!("compressed-tensors {f} (4-bit float with reciprocal global scales)")),
            (b, _, f) => DeclaredWeight::Unsupported(format!("compressed-tensors {b}-bit {f} weights")),
        }
    }

    fn is_excluded(&self, module: &str) -> bool {
        self.excluded.iter().any(|pat| match pat.strip_suffix('*') {
            Some(head) => module.starts_with(head),
            None => module == pat || module.ends_with(&format!(".{pat}")),
        })
    }

    /// How the weight tensor `name` (`<module>.weight`) is declared. `module` is `name` without `.weight`.
    pub fn declared(&self, name: &str) -> DeclaredWeight {
        let module = name.strip_suffix(".weight").unwrap_or(name);
        if self.is_excluded(module) {
            return DeclaredWeight::Native;
        }
        if self.layered {
            let hit = self.layers.iter().find(|(m, _)| module == m || module.strip_prefix(m.as_str()).is_some_and(|r| r.starts_with('.')));
            return hit.map_or(DeclaredWeight::Native, |(_, d)| d.clone());
        }
        self.default.clone().unwrap_or(DeclaredWeight::Native)
    }

    /// The declared FP8 `weight_block_size`, when the release has one.
    pub fn fp8_block(&self) -> Option<u32> {
        match self.default {
            Some(DeclaredWeight::Fp8 { block }) => block,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_modelopt_mixed_precision_export_declares_each_module_and_excludes_by_pattern() {
        let hf = json!({"quantization": {"quant_algo": "MIXED_PRECISION", "exclude_modules": ["mtp*"], "quantized_layers": {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
            "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}}});
        let q = HfQuant::from_json(None, Some(&hf)).unwrap();
        assert_eq!(q.declared("model.layers.0.self_attn.q_proj.weight"), DeclaredWeight::Fp8 { block: None });
        assert_eq!(q.declared("model.layers.0.mlp.experts.7.down_proj.weight"), DeclaredWeight::Nvfp4 { group: 16 });
        assert_eq!(q.declared("model.layers.0.mlp.gate.weight"), DeclaredWeight::Native);
        assert_eq!(q.declared("mtp.layers.0.self_attn.q_proj.weight"), DeclaredWeight::Native);
    }

    #[test]
    fn a_block_fp8_release_declares_its_block_and_a_whole_model_modelopt_algo_applies_everywhere() {
        let cfg = json!({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]}});
        let q = HfQuant::from_json(Some(&cfg), None).unwrap();
        assert_eq!((q.declared("a.weight"), q.fp8_block()), (DeclaredWeight::Fp8 { block: Some(128) }, Some(128)));
        assert_eq!(q.declared("lm_head.weight"), DeclaredWeight::Native);
        let all = json!({"quantization": {"quant_algo": "NVFP4", "group_size": 16}});
        assert_eq!(HfQuant::from_json(None, Some(&all)).unwrap().declared("x.weight"), DeclaredWeight::Nvfp4 { group: 16 });
    }

    #[test]
    fn compressed_tensors_fp8_is_read_and_gptq_awq_are_refused_by_name() {
        let ct = json!({"quantization_config": {"quant_method": "compressed-tensors", "format": "float-quantized", "ignore": ["lm_head"],
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float", "strategy": "tensor"}}}}});
        let q = HfQuant::from_json(Some(&ct), None).unwrap();
        assert_eq!((q.declared("a.weight"), q.declared("lm_head.weight")), (DeclaredWeight::Fp8 { block: None }, DeclaredWeight::Native));
        let gptq = json!({"quantization_config": {"quant_method": "gptq", "bits": 4}});
        let DeclaredWeight::Unsupported(why) = HfQuant::from_json(Some(&gptq), None).unwrap().declared("a.weight") else { panic!("gptq must be refused") };
        assert!(why.contains("gptq"), "{why}");
        assert!(HfQuant::from_json(Some(&json!({"architectures": ["X"]})), None).is_none(), "no declaration, no HfQuant");
    }
}
