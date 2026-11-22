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
//! A second, explicit sidecar schema declares every tensor by name: `hf_quant_config.json` holding one or more
//! sections (`model`, `mtp`, ...) of `parameter_groups`, each group naming a `weight_transform` (`cast`,
//! `nvfp4_nearest`, `mxfp8_nearest`), its dtypes, its scale format and block shape, and the exact `tensors` it
//! covers. It is recognised by that structure, never by the release's `quant_method` string, and its per-tensor
//! declarations override `config.json`'s global `quantization_config` entirely: a tensor it does not name is
//! [`DeclaredWeight::Undeclared`], and an unknown transform or scale format is refused by name.
//!
//! The declaration is checked against what is stored, never trusted over it: a weight declared FP8 whose dtype
//! is not `F8_E4M3` is an error naming both, not a silent reinterpretation.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::encoding::FloatFormat;

/// How one module's weight is declared to be stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeclaredWeight {
    /// Not quantized by the release (excluded, or not matched by any quantized group): native dtype.
    Native,
    /// 8-bit float weights with their scale tensors beside them (`weight_scale` or `weight_scale_inv`).
    /// `block` is the declared `weight_block_size` of a block-scaled release.
    Fp8 { block: Option<u32> },
    /// NVFP4 (E2M1 codes, E4M3 scale per `group`, and an f32 second level stored as `global` says).
    Nvfp4 { group: u32, global: Nvfp4Global },
    /// MXFP8: E4M3 codes with an E8M0 power-of-two scale per `block` elements of a row (stored as `U8`).
    Mxfp8 { block: u32 },
    /// Explicitly declared to be stored, and computed, in this float format (a `cast` parameter group).
    Cast(FloatFormat),
    /// The release declares its tensors one by one and this one is not among them: never loaded on a guess.
    Undeclared,
    /// A method brain has no codec for; loading such a tensor fails with this message.
    Unsupported(String),
}

/// Where the f32 second scale level of an NVFP4 tensor comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Global {
    /// A stored `<name>_scale_2` sidecar (ModelOpt): one value, or one per index of the leading axis.
    Stored,
    /// `<name>_amax / <name>_outer_scale_denominator` (one amax per index of the leading axis, or one for the
    /// tensor; a single denominator), folded once at open into the `_scale_2` level.
    AmaxOverDenominator,
}

/// The quantization the vendor's runtime applies to a layer's activations, as a parameter group declares it
/// (`required_activation_quantization`). Metadata for an opt-in numerics tier: brain's reference numerics keep
/// activations unquantized.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationQuant {
    pub format: String,
    pub block_size: u32,
    pub rounding: String,
    pub scale_dtype: String,
    #[serde(default)]
    pub amax_type: Option<String>,
    #[serde(default)]
    pub scale_selection_rule: Option<String>,
}

/// A declared activation quantization brain can apply ([`ActivationQuant::scheme`]): the numerics of the vendor's
/// W4A4 / W8A8 product, as an opt-in tier (`--linear-activations declared`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActivationScheme {
    /// NVFP4 (E2M1 values, an E4M3 scale per 16) with a per-token amax second level and the "four over six" block
    /// scale chosen by mean absolute error (`fouroversix_mae`), rounding to nearest.
    Nvfp4FourOverSixMae,
    /// MXFP8: E4M3 values with an E8M0 (power-of-two) scale per 32, rounding to nearest.
    Mxfp8E4m3,
}

impl ActivationScheme {
    /// Elements one scale covers along a row.
    pub fn block(self) -> u32 {
        match self {
            Self::Nvfp4FourOverSixMae => 16,
            Self::Mxfp8E4m3 => 32,
        }
    }
}

impl ActivationQuant {
    /// The scheme this declaration names, every field checked: a format, block, scale dtype, amax type, rounding or
    /// scale rule brain does not implement is an error naming it, never a nearby scheme.
    pub fn scheme(&self) -> Result<ActivationScheme, String> {
        let found = (self.format.as_str(), self.block_size, self.scale_dtype.as_str(), self.amax_type.as_deref(), self.rounding.as_str(), self.scale_selection_rule.as_deref());
        match found {
            ("nvfp4_e2m1", 16, "float8_e4m3fn", Some("per_token"), "nearest", Some("fouroversix_mae")) => Ok(ActivationScheme::Nvfp4FourOverSixMae),
            ("mxfp8_e4m3", 32, "ue8m0", None, "nearest", None) => Ok(ActivationScheme::Mxfp8E4m3),
            _ => Err(format!(
                "activation quantization {} (block {}, scale {}, amax {:?}, rounding {}, scale rule {:?}) is not implemented: brain runs nvfp4_e2m1/16/float8_e4m3fn/per_token/nearest/fouroversix_mae and mxfp8_e4m3/32/ue8m0/nearest",
                self.format, self.block_size, self.scale_dtype, self.amax_type, self.rounding, self.scale_selection_rule
            )),
        }
    }
}

/// Everything a `parameter_groups` release declares about one tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDeclaration {
    pub weight: DeclaredWeight,
    /// The safetensors dtype the tensor must be stored as (`output_dtype`): `F32`, `BF16`, `U8`, `F8_E4M3`, ...
    pub stored_dtype: &'static str,
    /// The dtype the group's forward output is declared in, when it differs from its compute dtype
    /// (`fwd_output_dtype`, e.g. f32 logits from a bf16 head).
    pub forward_output: Option<FloatFormat>,
    /// The vendor runtime's activation quantization for this tensor's matmul, if any.
    pub activation: Option<ActivationQuant>,
    /// The declared `weight_block_shape` (empty for a cast): its last extent is the scale block along a row; leading
    /// extents describe the producer's tile and must divide the corresponding axes.
    pub block_shape: Vec<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupDoc {
    compute_dtype: String,
    output_dtype: String,
    #[serde(default)]
    fwd_output_dtype: Option<String>,
    required_activation_quantization: Option<ActivationQuant>,
    weight_transform: String,
    #[serde(default)]
    weight_scale_format: Option<String>,
    weight_block_shape: Vec<u64>,
    tensors: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SectionDoc {
    parameter_groups: Vec<GroupDoc>,
}

/// A float dtype name of the schema (`float32`, `bfloat16`, `float16`) as a format and its safetensors tag.
fn float_dtype(name: &str) -> Option<(FloatFormat, &'static str)> {
    match name {
        "float32" => Some((FloatFormat::F32, "F32")),
        "bfloat16" => Some((FloatFormat::Bf16, "BF16")),
        "float16" => Some((FloatFormat::F16, "F16")),
        _ => None,
    }
}

impl GroupDoc {
    /// The declaration this group makes for each of its tensors. Unknown transforms and scale formats are refused by
    /// name per tensor ([`DeclaredWeight::Unsupported`]); dtypes contradicting the transform refuse the document.
    fn declaration(&self, at: &str) -> Result<TensorDeclaration, String> {
        let bad = |what: String| format!("hf_quant_config.json {at}: {what}");
        let block_last = self.weight_block_shape.last().copied();
        let (weight, stored_dtype) = match (self.weight_transform.as_str(), self.weight_scale_format.as_deref()) {
            ("cast", None) => {
                let (f, tag) = float_dtype(&self.compute_dtype).ok_or_else(|| bad(format!("cast to unsupported compute_dtype {}", self.compute_dtype)))?;
                if self.output_dtype != self.compute_dtype || !self.weight_block_shape.is_empty() {
                    return Err(bad(format!("a cast group stores its compute dtype with no blocks, got output_dtype {} block {:?}", self.output_dtype, self.weight_block_shape)));
                }
                (DeclaredWeight::Cast(f), tag)
            }
            ("nvfp4_nearest", Some("amax_denominator")) => {
                if (self.compute_dtype.as_str(), self.output_dtype.as_str()) != ("nvfp4", "uint8") || block_last != Some(16) {
                    return Err(bad(format!("nvfp4_nearest needs compute_dtype nvfp4, output_dtype uint8 and 16-wide blocks, got {} {} {:?}", self.compute_dtype, self.output_dtype, self.weight_block_shape)));
                }
                (DeclaredWeight::Nvfp4 { group: 16, global: Nvfp4Global::AmaxOverDenominator }, "U8")
            }
            ("mxfp8_nearest", None) => {
                if (self.compute_dtype.as_str(), self.output_dtype.as_str()) != ("mxfp8", "float8_e4m3fn") || block_last != Some(32) {
                    return Err(bad(format!("mxfp8_nearest needs compute_dtype mxfp8, output_dtype float8_e4m3fn and 32-wide blocks, got {} {} {:?}", self.compute_dtype, self.output_dtype, self.weight_block_shape)));
                }
                (DeclaredWeight::Mxfp8 { block: 32 }, "F8_E4M3")
            }
            ("cast" | "nvfp4_nearest" | "mxfp8_nearest", Some(f)) => (DeclaredWeight::Unsupported(format!("weight_scale_format {f} for weight_transform {}", self.weight_transform)), ""),
            ("nvfp4_nearest", None) => (DeclaredWeight::Unsupported("weight_transform nvfp4_nearest without a weight_scale_format".into()), ""),
            (t, _) => (DeclaredWeight::Unsupported(format!("weight_transform {t}")), ""),
        };
        let forward_output = match &self.fwd_output_dtype {
            None => None,
            Some(d) => Some(float_dtype(d).ok_or_else(|| bad(format!("unsupported fwd_output_dtype {d}")))?.0),
        };
        if self.weight_block_shape.contains(&0) {
            return Err(bad(format!("weight_block_shape {:?} has an empty extent", self.weight_block_shape)));
        }
        Ok(TensorDeclaration { weight, stored_dtype, forward_output, activation: self.required_activation_quantization.clone(), block_shape: self.weight_block_shape.clone() })
    }
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
    /// `layers` indexed by module: the position of each entry, so a weight's declaration is a lookup per prefix of its
    /// module name instead of a scan of every entry (a release lists hundreds, and a checkpoint asks for each of its
    /// thousands of tensors, again at every open).
    layer_index: std::collections::HashMap<String, usize>,
    /// Module name patterns the release leaves unquantized (`ignore` / `exclude_modules`); a trailing `*`
    /// matches any suffix, `re:` prefixed entries are regular expressions and are matched as plain prefixes of
    /// their literal head.
    excluded: Vec<String>,
    /// True when the declaration came with `quantized_layers`: then a module outside the list is native.
    layered: bool,
    /// A `parameter_groups` release: every tensor declared by name. When present it is the whole answer.
    tensors: Option<HashMap<String, TensorDeclaration>>,
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
        "NVFP4" | "W4A16_NVFP4" | "W4A4_NVFP4" => DeclaredWeight::Nvfp4 { group: group.unwrap_or(16) as u32, global: Nvfp4Global::Stored },
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
        HfQuant::from_json(read("config.json")?.as_ref(), read("hf_quant_config.json")?.as_ref())
    }

    /// [`Self::from_dir`] over already-parsed documents. An `hf_quant_config.json` of the `parameter_groups` shape
    /// that does not parse strictly is an error; so is one tensor declared twice.
    pub fn from_json(config: Option<&Value>, hf_quant: Option<&Value>) -> Result<Option<HfQuant>, String> {
        if let Some(groups) = hf_quant.filter(|h| Self::is_parameter_groups(h)) {
            return Self::from_parameter_groups(groups).map(Some);
        }
        Ok(Self::from_global_json(config, hf_quant))
    }

    /// The `parameter_groups` shape: a JSON object whose values (other than a `quant_method` string) are sections
    /// holding `parameter_groups`.
    fn is_parameter_groups(h: &Value) -> bool {
        h.as_object().is_some_and(|o| o.values().any(|v| v.get("parameter_groups").is_some()))
    }

    fn from_parameter_groups(h: &Value) -> Result<HfQuant, String> {
        let obj = h.as_object().ok_or("hf_quant_config.json: not a JSON object")?;
        let mut out = HfQuant::default();
        let mut tensors: HashMap<String, TensorDeclaration> = HashMap::new();
        for (key, v) in obj {
            if key == "quant_method" {
                let m = v.as_str().ok_or("hf_quant_config.json quant_method: not a string")?;
                out.method = Some(m.to_string());
                continue;
            }
            let section: SectionDoc = serde_json::from_value(v.clone()).map_err(|e| format!("hf_quant_config.json section {key}: {e}"))?;
            for (g, group) in section.parameter_groups.iter().enumerate() {
                let decl = group.declaration(&format!("{key}.parameter_groups[{g}]"))?;
                for t in &group.tensors {
                    if tensors.insert(t.clone(), decl.clone()).is_some() {
                        return Err(format!("hf_quant_config.json: tensor {t} is declared more than once"));
                    }
                }
            }
        }
        out.tensors = Some(tensors);
        Ok(out)
    }

    fn from_global_json(config: Option<&Value>, hf_quant: Option<&Value>) -> Option<HfQuant> {
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
                out.layer_index = out.layers.iter().enumerate().map(|(i, (m, _))| (m.clone(), i)).rev().collect();
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
            None => module.strip_suffix(pat.as_str()).is_some_and(|head| head.is_empty() || head.ends_with('.')),
        })
    }

    /// True for a release that declares each tensor by name (`parameter_groups`): then a tensor it does not name is
    /// [`DeclaredWeight::Undeclared`] and a refused declaration refuses the tensor whatever its dtype.
    pub fn explicit(&self) -> bool {
        self.tensors.is_some()
    }

    /// Everything an explicit release declares about tensor `name` (its exact stored name).
    pub fn declaration(&self, name: &str) -> Option<&TensorDeclaration> {
        self.tensors.as_ref()?.get(name)
    }

    /// The tensor names an explicit release declares (empty otherwise), in no particular order.
    pub fn declared_tensors(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().flat_map(|t| t.keys().map(String::as_str))
    }

    /// How the weight tensor `name` (`<module>.weight`) is declared. `module` is `name` without `.weight`.
    pub fn declared(&self, name: &str) -> DeclaredWeight {
        if let Some(t) = &self.tensors {
            return t.get(name).map_or(DeclaredWeight::Undeclared, |d| d.weight.clone());
        }
        let module = name.strip_suffix(".weight").unwrap_or(name);
        if self.is_excluded(module) {
            return DeclaredWeight::Native;
        }
        if self.layered {
            // The first listed entry that is the module or a dotted prefix of it: every prefix looked up, the lowest
            // position kept.
            let prefixes = std::iter::once(module).chain(module.match_indices('.').map(|(i, _)| &module[..i]));
            let hit = prefixes.filter_map(|m| self.layer_index.get(m)).min();
            return hit.map_or(DeclaredWeight::Native, |&i| self.layers[i].1.clone());
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
        let q = HfQuant::from_json(None, Some(&hf)).unwrap().unwrap();
        assert_eq!(q.declared("model.layers.0.self_attn.q_proj.weight"), DeclaredWeight::Fp8 { block: None });
        assert_eq!(q.declared("model.layers.0.mlp.experts.7.down_proj.weight"), DeclaredWeight::Nvfp4 { group: 16, global: Nvfp4Global::Stored });
        assert_eq!(q.declared("model.layers.0.mlp.gate.weight"), DeclaredWeight::Native);
        assert_eq!(q.declared("mtp.layers.0.self_attn.q_proj.weight"), DeclaredWeight::Native);
    }

    #[test]
    fn a_block_fp8_release_declares_its_block_and_a_whole_model_modelopt_algo_applies_everywhere() {
        let cfg = json!({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]}});
        let q = HfQuant::from_json(Some(&cfg), None).unwrap().unwrap();
        assert_eq!((q.declared("a.weight"), q.fp8_block()), (DeclaredWeight::Fp8 { block: Some(128) }, Some(128)));
        assert_eq!(q.declared("lm_head.weight"), DeclaredWeight::Native);
        let all = json!({"quantization": {"quant_algo": "NVFP4", "group_size": 16}});
        assert_eq!(HfQuant::from_json(None, Some(&all)).unwrap().unwrap().declared("x.weight"), DeclaredWeight::Nvfp4 { group: 16, global: Nvfp4Global::Stored });
    }

    #[test]
    fn compressed_tensors_fp8_is_read_and_gptq_awq_are_refused_by_name() {
        let ct = json!({"quantization_config": {"quant_method": "compressed-tensors", "format": "float-quantized", "ignore": ["lm_head"],
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float", "strategy": "tensor"}}}}});
        let q = HfQuant::from_json(Some(&ct), None).unwrap().unwrap();
        assert_eq!((q.declared("a.weight"), q.declared("lm_head.weight")), (DeclaredWeight::Fp8 { block: None }, DeclaredWeight::Native));
        let gptq = json!({"quantization_config": {"quant_method": "gptq", "bits": 4}});
        let DeclaredWeight::Unsupported(why) = HfQuant::from_json(Some(&gptq), None).unwrap().unwrap().declared("a.weight") else { panic!("gptq must be refused") };
        assert!(why.contains("gptq"), "{why}");
        assert!(HfQuant::from_json(Some(&json!({"architectures": ["X"]})), None).unwrap().is_none(), "no declaration, no HfQuant");
    }

    fn group(transform: &str, extra: Value, tensors: &[&str]) -> Value {
        let mut g = json!({"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "required_activation_quantization": null,
            "weight_block_shape": [], "weight_transform": transform, "tensors": tensors});
        for (k, v) in extra.as_object().unwrap() {
            g[k] = v.clone();
        }
        g
    }

    /// Spec: a `parameter_groups` sidecar declares each tensor by name, is chosen by its structure whatever its
    /// `quant_method`, and overrides `config.json`'s global algorithm: an unlisted tensor is undeclared, not NVFP4.
    #[test]
    fn parameter_groups_declare_each_tensor_and_override_the_global_config() {
        let nv = json!({"compute_dtype": "nvfp4", "output_dtype": "uint8", "weight_block_shape": [1, 16], "weight_scale_format": "amax_denominator"});
        let mx = json!({"compute_dtype": "mxfp8", "output_dtype": "float8_e4m3fn", "weight_block_shape": [32, 32]});
        let hf = json!({"quant_method": "anything", "model": {"parameter_groups": [
            group("cast", json!({}), &["emb.weight"]),
            group("nvfp4_nearest", nv, &["l0.w13_weight"]),
            group("mxfp8_nearest", mx, &["l1.w13_weight"])]},
            "draft": {"parameter_groups": [group("cast", json!({"compute_dtype": "float32", "output_dtype": "float32"}), &["fuse"])]}});
        let cfg = json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}});
        let q = HfQuant::from_json(Some(&cfg), Some(&hf)).unwrap().unwrap();
        assert!(q.explicit());
        assert_eq!(q.declared("emb.weight"), DeclaredWeight::Cast(FloatFormat::Bf16));
        assert_eq!(q.declared("l0.w13_weight"), DeclaredWeight::Nvfp4 { group: 16, global: Nvfp4Global::AmaxOverDenominator });
        assert_eq!(q.declared("l1.w13_weight"), DeclaredWeight::Mxfp8 { block: 32 });
        assert_eq!(q.declared("fuse"), DeclaredWeight::Cast(FloatFormat::F32));
        assert_eq!(q.declared("other.weight"), DeclaredWeight::Undeclared, "config.json's NVFP4 does not apply");
        assert_eq!(q.declaration("l1.w13_weight").unwrap().stored_dtype, "F8_E4M3");
    }

    /// Spec: the two activation quantizations a release declares are read as their schemes, field for field; any other
    /// value of any field is refused by name.
    #[test]
    fn declared_activation_quantization_is_read_exactly_and_anything_else_is_refused_by_name() {
        let nv = |rule: &str| ActivationQuant { format: "nvfp4_e2m1".into(), block_size: 16, rounding: "nearest".into(), scale_dtype: "float8_e4m3fn".into(), amax_type: Some("per_token".into()), scale_selection_rule: Some(rule.into()) };
        let mx = ActivationQuant { format: "mxfp8_e4m3".into(), block_size: 32, rounding: "nearest".into(), scale_dtype: "ue8m0".into(), amax_type: None, scale_selection_rule: None };
        assert_eq!(nv("fouroversix_mae").scheme(), Ok(ActivationScheme::Nvfp4FourOverSixMae));
        assert_eq!(mx.scheme(), Ok(ActivationScheme::Mxfp8E4m3));
        assert!(nv("fouroversix_mse").scheme().unwrap_err().contains("fouroversix_mse"));
        assert!(ActivationQuant { amax_type: Some("per_tensor".into()), ..nv("fouroversix_mae") }.scheme().unwrap_err().contains("per_tensor"));
        assert!(ActivationQuant { rounding: "stochastic".into(), ..mx.clone() }.scheme().unwrap_err().contains("stochastic"));
        assert!(ActivationQuant { block_size: 16, ..mx }.scheme().is_err());
    }

    /// Spec: unknown transforms and scale formats are refused by name on the tensors they declare; a malformed
    /// document (unknown field, dtypes contradicting the transform, a tensor declared twice) refuses the release.
    #[test]
    fn parameter_groups_refuse_unknown_values_by_name_and_malformed_documents_whole() {
        let doc = |groups: Vec<Value>| json!({"model": {"parameter_groups": groups}});
        let q = HfQuant::from_json(None, Some(&doc(vec![group("int4_awq", json!({}), &["a"]), group("cast", json!({"weight_scale_format": "e8m0"}), &["b"])]))).unwrap().unwrap();
        let DeclaredWeight::Unsupported(a) = q.declared("a") else { panic!("unknown transform must be refused") };
        let DeclaredWeight::Unsupported(b) = q.declared("b") else { panic!("unknown scale format must be refused") };
        assert!(a.contains("int4_awq") && b.contains("e8m0"), "{a} / {b}");
        let unknown_field = group("cast", json!({"surprise": 1}), &["a"]);
        assert!(HfQuant::from_json(None, Some(&doc(vec![unknown_field]))).unwrap_err().contains("surprise"));
        let twice = doc(vec![group("cast", json!({}), &["a"]), group("cast", json!({}), &["a"])]);
        assert!(HfQuant::from_json(None, Some(&twice)).unwrap_err().contains("more than once"));
        let wrong = group("mxfp8_nearest", json!({"compute_dtype": "mxfp8", "output_dtype": "uint8", "weight_block_shape": [1, 32]}), &["a"]);
        assert!(HfQuant::from_json(None, Some(&doc(vec![wrong]))).unwrap_err().contains("mxfp8_nearest"));
    }
}
