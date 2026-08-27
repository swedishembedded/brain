// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Safetensors-backed read/write (HF's official `safetensors` crate) plus a
//! [`ModelCard`] carried in the file's `__metadata__`.
//!
//! This is the forward-looking container that replaces the custom
//! `[len][json][f32 blob]` format in `lib.rs`. Everything is fp32 on the wire
//! for writes; reads dequantise F32/F16/BF16 to fp32. The card is stored as a
//! single JSON string under `brain.card` (the source of truth), with a few
//! well-known scalar fields mirrored as flat keys for human/tool inspection.

use std::collections::{BTreeMap, HashMap};
#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// LoRA / adapter descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adapter {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    /// LoRA scaling factor (the effective update is `(alpha/rank)·B·A`).
    /// Additive: absent on any card written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// Which base projections this adapter targets (matched by leaf name,
    /// e.g. `["wq","wk","wv","wo"]`), matching `qwen3::LoraCfg::targets`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<String>>,
    /// The bench dataset this adapter was trained on
    /// (`sha256:...`, bench's `dataset_id`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
}

/// Declared input/output modalities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Modalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

/// Portable model metadata carried inside a safetensors file's `__metadata__`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    pub schema_version: u32,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant_of: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Adapter>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Modalities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dim: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_params: Option<Value>,
    /// The `<vendor>` half of this model's fully-qualified reference
    /// (`brain_modelref::ModelRef`), e.g. `"Qwen"` for `Qwen/Qwen3-0.6B`.
    /// Additive (`#[serde(default)]`): absent on any card written before this
    /// field existed, which still deserializes fine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    /// The `<repo>` half of the reference (WITHOUT a quant suffix, matching
    /// `ModelRef::repo()`), e.g. `"Qwen3-0.6B"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// The upstream commit/revision these bytes were fetched at, when known
    /// (a HuggingFace commit SHA). Absent for a locally-produced or
    /// hand-imported checkpoint with no upstream provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Set only when `quant` was NOT produced from `vendor/repo` directly, but
    /// downloaded from a sibling quantization repo (e.g. a `Qwen/Qwen3-0.6B-Q8_0`
    /// ref resolving to a file inside the upstream `Qwen/Qwen3-0.6B-GGUF` repo
    /// rather than being quantized locally) — records where the bytes actually
    /// came from, for provenance and for `brain models migrate`-style tooling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
}

/// The single metadata key holding the whole card as a JSON string.
pub const CARD_KEY: &str = "brain.card";
/// The metadata key holding the model config as a JSON string.
pub const CONFIG_KEY: &str = "brain.config";

impl ModelCard {
    /// A minimal card: only the two required fields, everything else empty.
    pub fn new(id: impl Into<String>, family: impl Into<String>) -> Self {
        ModelCard {
            schema_version: 1,
            id: id.into(),
            display_name: None,
            family: family.into(),
            architecture: None,
            variant_of: None,
            adapter: None,
            capabilities: Vec::new(),
            modalities: None,
            context_length: None,
            param_count: None,
            license: None,
            created: None,
            owned_by: None,
            tokenizer: None,
            quant: None,
            embedding_dim: None,
            default_params: None,
            vendor: None,
            repo: None,
            revision: None,
            source_repo: None,
        }
    }

    /// A minimal card whose `id`/`vendor`/`repo`/`quant` come from a fully
    /// qualified `brain_modelref::ModelRef` (via its `Display` string and
    /// accessors, so `checkpoint` does not need a direct dependency on
    /// `brain-modelref` just for this constructor). `id` is the ref's full
    /// `Display` form (`"Qwen/Qwen3-0.6B-Q8_0"`), matching how the model store
    /// and the catalog key models by their canonical name.
    pub fn for_ref(id: &str, vendor: &str, repo: &str, quant: Option<&str>, family: impl Into<String>) -> Self {
        let mut card = ModelCard::new(id, family);
        card.vendor = Some(vendor.to_string());
        card.repo = Some(repo.to_string());
        card.quant = quant.map(str::to_string);
        card
    }

    /// Pack the card into safetensors `__metadata__` entries. `brain.card` is
    /// the JSON source of truth; well-known scalars are mirrored as flat keys.
    pub fn to_metadata(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert(CARD_KEY.to_string(), serde_json::to_string(self).unwrap());
        m.insert("id".to_string(), self.id.clone());
        m.insert("family".to_string(), self.family.clone());
        if let Some(v) = &self.architecture {
            m.insert("architecture".to_string(), v.clone());
        }
        if let Some(v) = &self.license {
            m.insert("license".to_string(), v.clone());
        }
        if let Some(v) = self.param_count {
            m.insert("param_count".to_string(), v.to_string());
        }
        if let Some(v) = self.context_length {
            m.insert("context_length".to_string(), v.to_string());
        }
        if let Some(v) = &self.vendor {
            m.insert("vendor".to_string(), v.clone());
        }
        if let Some(v) = &self.repo {
            m.insert("repo".to_string(), v.clone());
        }
        m
    }

    /// Recover the card from `__metadata__`, reading only `brain.card`.
    pub fn from_metadata(meta: &BTreeMap<String, String>) -> Option<ModelCard> {
        let raw = meta.get(CARD_KEY)?;
        serde_json::from_str(raw).ok()
    }
}

/// A safetensors model read into memory: fp32 tensors + raw metadata map.
pub struct StModel {
    pub tensors: HashMap<String, Vec<f32>>,
    pub metadata: BTreeMap<String, String>,
}

impl StModel {
    /// The model config, parsed from `brain.config` (or `Null` if absent).
    pub fn config(&self) -> Value {
        self.metadata
            .get(CONFIG_KEY)
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null)
    }

    /// The model card, if one was stored.
    pub fn card(&self) -> Option<ModelCard> {
        ModelCard::from_metadata(&self.metadata)
    }
}

/// Decode a safetensors `TensorView` byte buffer of the given dtype to fp32.
fn decode(dtype: safetensors::Dtype, raw: &[u8]) -> Result<Vec<f32>, String> {
    Ok(match dtype {
        safetensors::Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => raw
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => return Err(format!("st: unsupported dtype {other:?}")),
    })
}

/// Parse a safetensors byte buffer into fp32 tensors + metadata. Portable core
/// (the browser entry point passes fetched bytes; native `load_safetensors`
/// reads the file first).
pub fn parse_safetensors(bytes: &[u8]) -> Result<StModel, String> {
    let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| format!("st: {e}"))?;
    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        tensors.insert(name, decode(view.dtype(), view.data())?);
    }
    let (_, meta) = safetensors::SafeTensors::read_metadata(bytes).map_err(|e| format!("st: {e}"))?;
    let metadata = meta
        .metadata()
        .clone()
        .map(|m| m.into_iter().collect())
        .unwrap_or_default();
    Ok(StModel { tensors, metadata })
}

/// Read and parse a safetensors file from disk into fp32 tensors + metadata.
///
/// The file is **mapped**, not slurped. The result is unchanged - the same
/// [`parse_safetensors`] decodes the same bytes - but an owned `Vec<u8>` of
/// the whole file would be live at the same time as every decoded tensor, so
/// the eager route's peak was the file PLUS the model rather than the model.
/// A mapping's pages are not heap, and are dropped with the mapping, so what
/// survives this call is exactly the fp32 tensors the caller asked for.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_safetensors(path: &str) -> io::Result<StModel> {
    let file = std::fs::File::open(path)?;
    // SAFETY: weight files are treated as immutable for the mapping's lifetime,
    // the same contract `crate::mmap::MmapSafetensors::open` is built on.
    let mmap = unsafe { memmap2::Mmap::map(&file) }?;
    parse_safetensors(&mmap).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Write tensors as F32 safetensors. `config` is stored under `brain.config`
/// and `card` (if any) via [`ModelCard::to_metadata`]. The write is atomic
/// (tmp + rename), mirroring the custom-format `save`.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_safetensors(
    path: &str,
    tensors: &[(String, Vec<u64>, Vec<f32>)],
    config: &Value,
    card: Option<&ModelCard>,
) -> io::Result<()> {
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert(CONFIG_KEY.to_string(), serde_json::to_string(config)?);
    if let Some(c) = card {
        meta.extend(c.to_metadata());
    }

    // Own the little-endian byte buffers so the borrowed TensorViews outlive
    // serialize.
    let mut bufs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(tensors.len());
    for (name, shape, data) in tensors {
        let mut b = Vec::with_capacity(data.len() * 4);
        for v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        bufs.push((name.clone(), shape.iter().map(|&d| d as usize).collect(), b));
    }
    let views: Vec<(&str, safetensors::tensor::TensorView)> = bufs
        .iter()
        .map(|(name, shape, b)| {
            let v = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), b)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("st: {e}")))?;
            Ok((name.as_str(), v))
        })
        .collect::<io::Result<_>>()?;
    let out = safetensors::serialize(views, Some(meta))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("st: {e}")))?;

    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = format!("{path}.tmp");
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        file.write_all(&out)?;
        file.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Read only the leading `[u64 len][JSON header]` of a safetensors file, never
/// the tensor blob.
#[cfg(not(target_arch = "wasm32"))]
fn read_header_json(path: &str) -> io::Result<Value> {
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let hlen = u64::from_le_bytes(len_bytes) as usize;
    let mut hbytes = vec![0u8; hlen];
    file.read_exact(&mut hbytes)?;
    serde_json::from_slice(&hbytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read the `__metadata__` map from a safetensors header without loading tensors.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_metadata(path: &str) -> io::Result<BTreeMap<String, String>> {
    let header = read_header_json(path)?;
    let mut out = BTreeMap::new();
    if let Some(obj) = header.get("__metadata__").and_then(|m| m.as_object()) {
        for (k, v) in obj {
            if let Some(s) = v.as_str() {
                out.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(out)
}

/// Read the [`ModelCard`] from a safetensors header without loading tensors.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_card(path: &str) -> io::Result<Option<ModelCard>> {
    Ok(ModelCard::from_metadata(&read_metadata(path)?))
}

/// Sum tensor element counts from a safetensors header without loading tensors.
#[cfg(not(target_arch = "wasm32"))]
pub fn param_count_from_header(path: &str) -> io::Result<u64> {
    let header = read_header_json(path)?;
    let obj = header
        .as_object()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "st: header not an object"))?;
    let mut total: u64 = 0;
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let mut n: u64 = 1;
        if let Some(shape) = meta.get("shape").and_then(|s| s.as_array()) {
            for d in shape {
                n *= d.as_u64().unwrap_or(0);
            }
            total += n;
        }
    }
    Ok(total)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn scratch(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("brain-st-{}-{}.safetensors", name, std::process::id()))
            .to_str()
            .unwrap()
            .to_string()
    }

    fn sample_card() -> ModelCard {
        ModelCard {
            adapter: Some(Adapter {
                kind: "lora".into(),
                rank: Some(16),
                base: Some("qwen3".into()),
                alpha: Some(32.0),
                targets: Some(vec!["wq".into(), "wk".into(), "wv".into(), "wo".into()]),
                dataset_id: Some("sha256:abc123".into()),
            }),
            capabilities: vec!["text".into(), "chat".into()],
            modalities: Some(Modalities { input: vec!["text".into()], output: vec!["text".into()] }),
            context_length: Some(32768),
            param_count: Some(600_000_000),
            license: Some("apache-2.0".into()),
            architecture: Some("qwen3".into()),
            default_params: Some(serde_json::json!({"temperature": 0.7})),
            ..ModelCard::new("qwen3-0.6b", "qwen")
        }
    }

    #[test]
    fn old_card_json_without_vendor_repo_fields_still_deserializes() {
        // Exactly what a card written before vendor/repo/revision/source_repo
        // existed looks like on disk -- none of those four keys present.
        let old = serde_json::json!({
            "schema_version": 1,
            "id": "qwen3-0.6b",
            "family": "qwen",
        });
        let card: ModelCard = serde_json::from_value(old).expect("additive fields must not break old cards");
        assert_eq!(card.id, "qwen3-0.6b");
        assert_eq!(card.vendor, None);
        assert_eq!(card.repo, None);
        assert_eq!(card.revision, None);
        assert_eq!(card.source_repo, None);
    }

    #[test]
    fn for_ref_populates_vendor_repo_quant_and_mirrors_into_metadata() {
        let card = ModelCard::for_ref("Qwen/Qwen3-0.6B-Q8_0", "Qwen", "Qwen3-0.6B", Some("Q8_0"), "qwen");
        assert_eq!(card.id, "Qwen/Qwen3-0.6B-Q8_0");
        assert_eq!(card.vendor.as_deref(), Some("Qwen"));
        assert_eq!(card.repo.as_deref(), Some("Qwen3-0.6B"));
        assert_eq!(card.quant.as_deref(), Some("Q8_0"));

        let meta = card.to_metadata();
        assert_eq!(meta.get("vendor").map(String::as_str), Some("Qwen"));
        assert_eq!(meta.get("repo").map(String::as_str), Some("Qwen3-0.6B"));
        // brain.card round-trips the full struct, including the new fields.
        let roundtrip = ModelCard::from_metadata(&meta).unwrap();
        assert_eq!(roundtrip, card);
    }

    #[test]
    fn f32_roundtrip() {
        let p = scratch("f32");
        let cfg = serde_json::json!({"d_model": 8, "n_layers": 1});
        let tensors = vec![
            ("a".to_string(), vec![2u64, 2], vec![1.0f32, -2.5, 3.25, 4.0]),
            ("b".to_string(), vec![3u64], vec![0.1f32, 0.2, 0.3]),
        ];
        save_safetensors(&p, &tensors, &cfg, None).unwrap();
        let m = load_safetensors(&p).unwrap();
        assert_eq!(m.tensors["a"], vec![1.0, -2.5, 3.25, 4.0]);
        assert_eq!(m.tensors["b"], vec![0.1, 0.2, 0.3]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn config_roundtrip() {
        let p = scratch("cfg");
        let cfg = serde_json::json!({"d_model": 8, "n_layers": 2, "name": "x"});
        save_safetensors(&p, &[("w".to_string(), vec![1], vec![1.0f32])], &cfg, None).unwrap();
        let m = load_safetensors(&p).unwrap();
        assert_eq!(m.config(), cfg);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn card_metadata_roundtrip() {
        let card = sample_card();
        let meta = card.to_metadata();
        // Mirrored flat keys are present alongside the JSON source of truth.
        assert_eq!(meta["id"], "qwen3-0.6b");
        assert_eq!(meta["family"], "qwen");
        assert_eq!(meta["architecture"], "qwen3");
        assert_eq!(meta["param_count"], "600000000");
        assert_eq!(ModelCard::from_metadata(&meta).unwrap(), card);
    }

    #[test]
    fn card_file_roundtrip_via_load() {
        let p = scratch("card");
        let card = sample_card();
        let cfg = serde_json::json!({"d_model": 4});
        save_safetensors(&p, &[("w".to_string(), vec![2], vec![1.0f32, 2.0])], &cfg, Some(&card))
            .unwrap();
        let m = load_safetensors(&p).unwrap();
        assert_eq!(m.card().unwrap(), card);
        assert_eq!(m.config(), cfg);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn read_card_and_metadata_without_tensors() {
        let p = scratch("hdr");
        let card = sample_card();
        let cfg = serde_json::json!({"d_model": 4});
        let big = vec![7.0f32; 4096]; // a blob big enough to notice if it were read
        save_safetensors(&p, &[("w".to_string(), vec![4096], big)], &cfg, Some(&card)).unwrap();

        let meta = read_metadata(&p).unwrap();
        assert_eq!(meta["id"], "qwen3-0.6b");
        assert!(meta.contains_key(CARD_KEY));
        assert!(meta.contains_key(CONFIG_KEY));
        assert_eq!(read_card(&p).unwrap().unwrap(), card);
        assert_eq!(param_count_from_header(&p).unwrap(), 4096);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn f16_and_bf16_decode() {
        // Hand-build a safetensors with one F16 and one BF16 tensor.
        let f16 = [half::f16::from_f32(1.0), half::f16::from_f32(-2.5)];
        let bf16 = [half::bf16::from_f32(1.0), half::bf16::from_f32(-4.0)];
        let mut f16b = Vec::new();
        for v in f16 {
            f16b.extend_from_slice(&v.to_le_bytes());
        }
        let mut bf16b = Vec::new();
        for v in bf16 {
            bf16b.extend_from_slice(&v.to_le_bytes());
        }
        let a = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2], &f16b).unwrap();
        let b =
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &bf16b).unwrap();
        let bytes = safetensors::serialize(
            vec![("a".to_string(), a), ("b".to_string(), b)],
            None,
        )
        .unwrap();

        let m = parse_safetensors(&bytes).unwrap();
        assert_eq!(m.tensors["a"], vec![1.0, -2.5]);
        assert_eq!(m.tensors["b"], vec![1.0, -4.0]);
    }
}
