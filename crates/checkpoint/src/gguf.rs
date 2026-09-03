// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF v2/v3 reader: full KV metadata + dequant of all mainstream GGML quant
//! types to fp32.
//!
//! brain's engine is fp32-only, so the contract is dequant-on-load: every
//! tensor — F32/F16/BF16 or a per-tensor quantized block type — is decoded to a
//! flat `Vec<f32>`. Legacy blocks of 32 (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) and the
//! k-quant super-blocks of 256 (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K) are supported;
//! the codebook IQ*/TQ*/MXFP4 families return a clear "not yet implemented"
//! error (rare; a documented follow-up).
//!
//! The core [`parse_gguf`] works from a byte slice (no `Seek`, no fs) so it is
//! usable from wasm; the native [`read`] wrapper maps the file then calls it -
//! mirroring the `parse`/`load` split in `lib.rs` and `st.rs`. (There is a
//! whole-file `load_gguf` too, but it is `#[cfg(test)]`: see its own doc.)
//!
//! GGUF stores dims fastest-varying first (`ne[0]` = innermost); torch/brain
//! shapes are the reverse. The byte layout of the data is identical row-major,
//! so only the shape vector is reversed on read.

use std::collections::{BTreeMap, HashMap};

use crate::safetensors::StTensor;
use crate::st::ModelCard;

// ggml_type ids (see the GGUF spec's ggml_type enum). `pub(crate)` so
// `quant.rs`/`gguf_write.rs` dispatch on the same ids this reader does --
// one type-id table, not a second copy that can drift.
pub(crate) const T_F32: u32 = 0;
pub(crate) const T_F16: u32 = 1;
pub(crate) const T_Q4_0: u32 = 2;
pub(crate) const T_Q4_1: u32 = 3;
pub(crate) const T_Q5_0: u32 = 6;
pub(crate) const T_Q5_1: u32 = 7;
pub(crate) const T_Q8_0: u32 = 8;
pub(crate) const T_Q8_1: u32 = 9;
pub(crate) const T_Q2_K: u32 = 10;
pub(crate) const T_Q3_K: u32 = 11;
pub(crate) const T_Q4_K: u32 = 12;
pub(crate) const T_Q5_K: u32 = 13;
pub(crate) const T_Q6_K: u32 = 14;
pub(crate) const T_Q8_K: u32 = 15;
pub(crate) const T_BF16: u32 = 30;
pub(crate) const T_I8: u32 = 24;
pub(crate) const T_I16: u32 = 25;
pub(crate) const T_I32: u32 = 26;
pub(crate) const T_I64: u32 = 27;
pub(crate) const T_F64: u32 = 28;
pub(crate) const T_IQ4_NL: u32 = 20;
pub(crate) const T_IQ4_XS: u32 = 23;
pub(crate) const T_TQ1_0: u32 = 34;
pub(crate) const T_TQ2_0: u32 = 35;
pub(crate) const T_MXFP4: u32 = 39;

pub(crate) const QK_K: usize = 256;

/// A ggml tensor storage type, as declared per-tensor in a GGUF file.
///
/// This is the ONE vocabulary the reader dispatches on. `ggml_type_name`,
/// `block_geometry`, `tensor_nbytes` and `dequantize` used to be four
/// independent `match ty: u32` tables that could drift from each other; they
/// are now thin wrappers over this enum's methods, so a `match` over a
/// [`GgmlType`] gets a compiler error for an unhandled variant instead of a
/// silent `_ =>` arm - exactly where a newly-added format would otherwise
/// land silently instead of failing to compile. K-quant variants drop the
/// `_` before `K` (`Q4K`, not `Q4_K`) solely to satisfy
/// `non_camel_case_types`; [`GgmlType::name`] still spells them the
/// conventional way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    MXFP4,
    IQ4NL,
    IQ4XS,
    TQ1_0,
    TQ2_0,
}

impl GgmlType {
    /// Resolve a GGUF `ggml_type` id to the type this reader can decode.
    /// `None` for an id with no block-format entry here (a still-unsupported
    /// IQ grid codebook, or a genuinely unknown id) - never a guess.
    pub fn from_id(id: u32) -> Option<GgmlType> {
        Some(match id {
            T_F32 => GgmlType::F32,
            T_F16 => GgmlType::F16,
            T_BF16 => GgmlType::BF16,
            T_F64 => GgmlType::F64,
            T_I8 => GgmlType::I8,
            T_I16 => GgmlType::I16,
            T_I32 => GgmlType::I32,
            T_I64 => GgmlType::I64,
            T_Q4_0 => GgmlType::Q4_0,
            T_Q4_1 => GgmlType::Q4_1,
            T_Q5_0 => GgmlType::Q5_0,
            T_Q5_1 => GgmlType::Q5_1,
            T_Q8_0 => GgmlType::Q8_0,
            T_Q8_1 => GgmlType::Q8_1,
            T_Q2_K => GgmlType::Q2K,
            T_Q3_K => GgmlType::Q3K,
            T_Q4_K => GgmlType::Q4K,
            T_Q5_K => GgmlType::Q5K,
            T_Q6_K => GgmlType::Q6K,
            T_Q8_K => GgmlType::Q8K,
            T_MXFP4 => GgmlType::MXFP4,
            T_IQ4_NL => GgmlType::IQ4NL,
            T_IQ4_XS => GgmlType::IQ4XS,
            T_TQ1_0 => GgmlType::TQ1_0,
            T_TQ2_0 => GgmlType::TQ2_0,
            _ => return None,
        })
    }

    /// The GGUF `ggml_type` id this type is written as.
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => T_F32,
            GgmlType::F16 => T_F16,
            GgmlType::BF16 => T_BF16,
            GgmlType::F64 => T_F64,
            GgmlType::I8 => T_I8,
            GgmlType::I16 => T_I16,
            GgmlType::I32 => T_I32,
            GgmlType::I64 => T_I64,
            GgmlType::Q4_0 => T_Q4_0,
            GgmlType::Q4_1 => T_Q4_1,
            GgmlType::Q5_0 => T_Q5_0,
            GgmlType::Q5_1 => T_Q5_1,
            GgmlType::Q8_0 => T_Q8_0,
            GgmlType::Q8_1 => T_Q8_1,
            GgmlType::Q2K => T_Q2_K,
            GgmlType::Q3K => T_Q3_K,
            GgmlType::Q4K => T_Q4_K,
            GgmlType::Q5K => T_Q5_K,
            GgmlType::Q6K => T_Q6_K,
            GgmlType::Q8K => T_Q8_K,
            GgmlType::MXFP4 => T_MXFP4,
            GgmlType::IQ4NL => T_IQ4_NL,
            GgmlType::IQ4XS => T_IQ4_XS,
            GgmlType::TQ1_0 => T_TQ1_0,
            GgmlType::TQ2_0 => T_TQ2_0,
        }
    }

    /// This type's conventional spelling (`"Q4_K"`, `"BF16"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::F64 => "F64",
            GgmlType::I8 => "I8",
            GgmlType::I16 => "I16",
            GgmlType::I32 => "I32",
            GgmlType::I64 => "I64",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q8_1 => "Q8_1",
            GgmlType::Q2K => "Q2_K",
            GgmlType::Q3K => "Q3_K",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
            GgmlType::Q8K => "Q8_K",
            GgmlType::MXFP4 => "MXFP4",
            GgmlType::IQ4NL => "IQ4_NL",
            GgmlType::IQ4XS => "IQ4_XS",
            GgmlType::TQ1_0 => "TQ1_0",
            GgmlType::TQ2_0 => "TQ2_0",
        }
    }

    /// Elements per block.
    pub fn block_elems(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64 | GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::MXFP4 | GgmlType::IQ4NL => 32,
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8K | GgmlType::IQ4XS | GgmlType::TQ1_0 | GgmlType::TQ2_0 => QK_K,
        }
    }

    /// On-disk bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 | GgmlType::BF16 => 2,
            GgmlType::F64 => 8,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 36,
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
            GgmlType::Q8K => 292,
            // `block_mxfp4`: 1 (e8m0 scale) + QK_MXFP4/2 (16) = 17.
            GgmlType::MXFP4 => 17,
            // `block_iq4_nl`: sizeof(ggml_half) (2) + QK4_NL/2 (16) = 18.
            GgmlType::IQ4NL => 18,
            // `block_iq4_xs`: 2 (d) + 2 (scales_h) + QK_K/64 (4, scales_l) + QK_K/2 (128, qs) = 136.
            GgmlType::IQ4XS => 136,
            // `block_tq1_0`: QK_K/64 (4, qh) + (QK_K - 4*QK_K/64)/5 (48, qs) + 2 (d) = 54.
            GgmlType::TQ1_0 => 54,
            // `block_tq2_0`: QK_K/4 (64, qs) + 2 (d) = 66.
            GgmlType::TQ2_0 => 66,
        }
    }

    /// Whether this type's affine decode needs a per-sub-block minimum
    /// (`w = d*sc*q - dmin*m`), not just a scale (`w = d*sc*q`). Q4_K/Q5_K are
    /// affine; every other type this reader decodes is symmetric. Exists so a
    /// device-side K-quant kernel selector (a later milestone) can ask one
    /// place rather than re-deriving it from the type name.
    pub fn is_affine(self) -> bool {
        matches!(self, GgmlType::Q4K | GgmlType::Q5K)
    }

    /// This type's per-block decoder, for a caller expanding an arbitrary
    /// block range ([`block_expand`]). `None` for F32/F16/BF16: those are not
    /// "one block, one shared scale" formats in the sense [`block_expand`]
    /// serves - a caller wanting a range of them slices `dequantize`'s output
    /// directly, since there is no smaller independently-decodable unit.
    fn block_decoder(self) -> Option<fn(&[u8], &mut Vec<f32>)> {
        Some(match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64 | GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 => return None,
            GgmlType::Q4_0 => deq_q4_0,
            GgmlType::Q4_1 => deq_q4_1,
            GgmlType::Q5_0 => deq_q5_0,
            GgmlType::Q5_1 => deq_q5_1,
            GgmlType::Q8_0 => deq_q8_0,
            GgmlType::Q8_1 => deq_q8_1,
            GgmlType::Q2K => deq_q2_k,
            GgmlType::Q3K => deq_q3_k,
            GgmlType::Q4K => deq_q4_k,
            GgmlType::Q5K => deq_q5_k,
            GgmlType::Q6K => deq_q6_k,
            GgmlType::Q8K => deq_q8_k,
            GgmlType::MXFP4 => deq_mxfp4,
            GgmlType::IQ4NL => deq_iq4_nl,
            GgmlType::IQ4XS => deq_iq4_xs,
            GgmlType::TQ1_0 => deq_tq1_0,
            GgmlType::TQ2_0 => deq_tq2_0,
        })
    }
}

/// A lend of one tensor's ALREADY-QUANTIZED on-disk bytes, plus the block
/// format that decodes them - the payload of
/// [`crate::TensorSource::raw_blocks`]. `numel` is the tensor's element
/// count (not `bytes.len()`, which is `numel`'s ceiling to a whole number of
/// blocks): a caller reconstructing values needs to know where the real data
/// ends and any block-padding begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    pub ty: GgmlType,
    pub numel: usize,
}

impl BlockLayout {
    pub fn block_elems(&self) -> usize {
        self.ty.block_elems()
    }
    pub fn block_bytes(&self) -> usize {
        self.ty.block_bytes()
    }
}

/// A single GGUF metadata value. Mirrors the 13 `gguf_metadata_value_type`s;
/// arrays (including nested ones) are held as a `Vec<GgufValue>`.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Any unsigned/signed integer scalar as `u64` (negative → `None`).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) if v >= 0 => Some(v as u64),
            GgufValue::I16(v) if v >= 0 => Some(v as u64),
            GgufValue::I32(v) if v >= 0 => Some(v as u64),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    /// A string scalar as `&str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// The tokenizer embedded in a GGUF's `tokenizer.ggml.*` KV metadata.
///
/// A verbatim, typed view of the KV — no interpretation beyond pulling each key
/// out of the map. `model` is the ggml tokenizer scheme (`"gpt2"` for the GPT-2
/// / Qwen byte-level BPE, `"llama"`/`"bert"`/… for others). `tokens[id]` is the
/// token text for id `id`; `merges` are ranked `"a b"` pairs (index = priority);
/// `token_types[id]` is the ggml token-type enum (1 NORMAL, 2 UNKNOWN, 3
/// CONTROL, 4 USER_DEFINED, 5 BYTE, 6 UNUSED). The special-token ids are the
/// declared `*_token_id` scalars (absent → `None`).
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTokenizer {
    /// `tokenizer.ggml.model` — the tokenizer scheme (e.g. `"gpt2"`, `"llama"`).
    pub model: String,
    /// `tokenizer.ggml.pre` — the pre-tokenizer family (e.g. `"qwen2"`), if any.
    pub pre: Option<String>,
    /// `tokenizer.ggml.tokens` — token text indexed by id.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.merges` — ranked `"a b"` merge pairs (index = priority).
    pub merges: Vec<String>,
    /// `tokenizer.ggml.token_type` — per-id ggml token-type enum (may be empty).
    pub token_types: Vec<i32>,
    /// `tokenizer.ggml.bos_token_id`.
    pub bos: Option<u32>,
    /// `tokenizer.ggml.eos_token_id`.
    pub eos: Option<u32>,
    /// `tokenizer.ggml.unknown_token_id`.
    pub unk: Option<u32>,
    /// `tokenizer.ggml.padding_token_id`.
    pub pad: Option<u32>,
}

/// Pull a `GgufTokenizer` out of a KV map: `None` unless `tokenizer.ggml.model`
/// is present (a file without an embedded tokenizer). Shared by
/// [`GgufModel::tokenizer`] and [`MmapGguf::tokenizer`].
fn tokenizer_from_kv(kv: &BTreeMap<String, GgufValue>) -> Option<GgufTokenizer> {
    let model = kv.get("tokenizer.ggml.model").and_then(|v| v.as_str())?.to_string();
    let str_arr = |k: &str| match kv.get(k) {
        Some(GgufValue::Array(a)) => a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
        _ => Vec::new(),
    };
    let i32_arr = |k: &str| match kv.get(k) {
        Some(GgufValue::Array(a)) => a
            .iter()
            .map(|v| match *v {
                GgufValue::I8(x) => x as i32,
                GgufValue::I16(x) => x as i32,
                GgufValue::I32(x) => x,
                GgufValue::U8(x) => x as i32,
                GgufValue::U16(x) => x as i32,
                GgufValue::U32(x) => x as i32,
                _ => v.as_u64().map(|u| u as i32).unwrap_or(0),
            })
            .collect(),
        _ => Vec::new(),
    };
    let id = |k: &str| kv.get(k).and_then(|v| v.as_u64()).map(|v| v as u32);
    Some(GgufTokenizer {
        model,
        pre: kv.get("tokenizer.ggml.pre").and_then(|v| v.as_str()).map(|s| s.to_string()),
        tokens: str_arr("tokenizer.ggml.tokens"),
        merges: str_arr("tokenizer.ggml.merges"),
        token_types: i32_arr("tokenizer.ggml.token_type"),
        bos: id("tokenizer.ggml.bos_token_id"),
        eos: id("tokenizer.ggml.eos_token_id"),
        unk: id("tokenizer.ggml.unknown_token_id"),
        pad: id("tokenizer.ggml.padding_token_id"),
    })
}

/// A GGUF model read into memory: fp32 tensors + full KV metadata.
pub struct GgufModel {
    /// Tensor name → dequantized fp32 values (row-major, torch element order).
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor name → shape in torch order (`ne` reversed).
    pub shapes: BTreeMap<String, Vec<usize>>,
    /// Every metadata key-value pair, in the file's declared order.
    pub kv: BTreeMap<String, GgufValue>,
    /// Convenience view of the string-typed KV pairs (e.g. `general.name`).
    pub metadata_strings: BTreeMap<String, String>,
}

impl GgufModel {
    /// Total number of tensor elements (summed over shapes).
    pub fn param_count(&self) -> u64 {
        self.shapes.values().map(|s| s.iter().product::<usize>() as u64).sum()
    }

    /// Build a [`ModelCard`] from the standardized GGUF keys. Missing keys map
    /// to `None`; `general.architecture` fills both family and architecture,
    /// `general.name` fills id and display_name.
    pub fn model_card(&self) -> ModelCard {
        card_from_kv(&self.kv, self.param_count())
    }

    /// The tokenizer embedded in `tokenizer.ggml.*` KV, if the file declares one.
    pub fn tokenizer(&self) -> Option<GgufTokenizer> {
        tokenizer_from_kv(&self.kv)
    }
}

/// Build a [`ModelCard`] from a GGUF KV map plus a precomputed param count.
/// Shared by [`GgufModel::model_card`] and [`MmapGguf::model_card`].
fn card_from_kv(kv: &BTreeMap<String, GgufValue>, param_count: u64) -> ModelCard {
    let s = |k: &str| kv.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
    let arch = s("general.architecture");
    let name = s("general.name");
    let id = name.clone().or_else(|| arch.clone()).unwrap_or_else(|| "gguf".into());
    let family = arch.clone().unwrap_or_else(|| "gguf".into());

    let mut card = ModelCard::new(id, family);
    card.display_name = name;
    card.architecture = arch.clone();
    card.license = s("general.license");
    card.tokenizer = s("tokenizer.ggml.model");
    card.quant = quant_label(kv);
    card.param_count = Some(param_count);
    if let Some(a) = &arch {
        card.context_length = kv.get(&format!("{a}.context_length")).and_then(|v| v.as_u64());
    }
    card
}

/// A human label for the file's quantization, from `general.file_type` if
/// present AND recognized, else `general.quantization_version`, else `None`.
///
/// A `general.file_type` id this reader does not recognize must NOT stop
/// here and report "unknown" - that actively defeats the
/// `quantization_version` fallback for exactly the files most likely to need
/// it (a release quantized with a `llama_ftype` newer than this table).
fn quant_label(kv: &BTreeMap<String, GgufValue>) -> Option<String> {
    if let Some(ft) = kv.get("general.file_type").and_then(|v| v.as_u64()) {
        if let Some(name) = file_type_name(ft as u32) {
            return Some(name.to_string());
        }
    }
    kv.get("general.quantization_version").and_then(|v| v.as_u64()).map(|v| format!("qver{v}"))
}

/// `(id, name)` for every `general.file_type` (llama.cpp's `enum
/// llama_ftype`, the whole-file "mostly X" label) this reader recognizes -
/// the ONE table both [`file_type_name`] (read: id -> name) and
/// [`file_type_id`] (write: name -> id, `crate::quantize::Tier`'s own
/// `general.file_type` derivation) search, so the two directions cannot
/// drift from each other.
///
/// This is `llama_ftype`, NOT [`GgmlType`]/`ggml_type` - a separate llama.cpp
/// enum with its own numbering (file_type `Q8_0` is 7; the per-tensor
/// `ggml_type` `T_Q8_0` is 8). Hand-maintained, like [`GgmlType`], with no
/// upstream checkout in this repo to pin it against; ids past 37 (MXFP4 and
/// newer) are deliberately omitted rather than guessed.
const FILE_TYPES: &[(u32, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (7, "Q8_0"),
    (8, "Q5_0"),
    (9, "Q5_1"),
    (10, "Q2_K"),
    (11, "Q3_K_S"),
    (12, "Q3_K_M"),
    (13, "Q3_K_L"),
    (14, "Q4_K_S"),
    (15, "Q4_K_M"),
    (16, "Q5_K_S"),
    (17, "Q5_K_M"),
    (18, "Q6_K"),
    (19, "IQ2_XXS"),
    (20, "IQ2_XS"),
    (21, "Q2_K_S"),
    (22, "IQ3_XS"),
    (23, "IQ3_XXS"),
    (24, "IQ1_S"),
    (25, "IQ4_NL"),
    (26, "IQ3_S"),
    (27, "IQ3_M"),
    (28, "IQ2_S"),
    (29, "IQ2_M"),
    (30, "IQ4_XS"),
    (31, "IQ1_M"),
    (32, "BF16"),
    (36, "TQ1_0"),
    (37, "TQ2_0"),
];

/// Map a `general.file_type` id to its conventional name. `None` for an id
/// [`FILE_TYPES`] does not carry - never a fabricated "unknown" string that
/// shadows the `quantization_version` fallback in [`quant_label`].
fn file_type_name(ft: u32) -> Option<&'static str> {
    FILE_TYPES.iter().find(|&&(id, _)| id == ft).map(|&(_, name)| name)
}

/// The write-side inverse of [`file_type_name`]: the `general.file_type` id
/// for a name [`FILE_TYPES`] carries. `pub(crate)` for `crate::quantize::
/// Tier::file_type_id`, the only caller - deciding what to WRITE from the
/// same table this reads, rather than a second hand-maintained copy.
pub(crate) fn file_type_id(name: &str) -> Option<u32> {
    FILE_TYPES.iter().find(|&&(_, n)| n == name).map(|&(id, _)| id)
}

/// Forward-only cursor over a byte slice; every read is bounds-checked.
struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cursor { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("gguf: length overflow")?;
        if end > self.b.len() {
            return Err(format!("gguf: unexpected EOF at {} (+{n})", self.p));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
    }
    /// Read one metadata value of the given GGUF value type (arrays recurse).
    fn value(&mut self, ty: u32) -> Result<GgufValue, String> {
        Ok(match ty {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.i16()?),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.i32()?),
            6 => GgufValue::F32(self.f32()?),
            7 => GgufValue::Bool(self.u8()? != 0),
            8 => GgufValue::String(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.value(et)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.i64()?),
            12 => GgufValue::F64(self.f64()?),
            other => return Err(format!("gguf: unknown metadata value type {other}")),
        })
    }
}

/// One tensor's header info, resolved to torch shape.
struct Info {
    name: String,
    shape: Vec<usize>,
    ty: u32,
    offset: u64,
}

/// Parsed header: KV map, tensor infos, and the aligned tensor-data start.
type Header = (BTreeMap<String, GgufValue>, Vec<Info>, usize);

/// Parse header + KV + tensor infos, returning them plus the aligned data start.
fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    let mut c = Cursor::new(bytes);
    if c.take(4)? != b"GGUF" {
        return Err("gguf: bad magic".into());
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("gguf: unsupported version {version}"));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;

    let mut kv = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let val = c.value(ty)?;
        kv.insert(key, val);
    }

    let alignment = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32).max(1) as usize;

    let mut infos = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()? as usize;
        let mut ne = Vec::with_capacity(nd);
        for _ in 0..nd {
            ne.push(c.u64()? as usize);
        }
        ne.reverse();
        let ty = c.u32()?;
        let offset = c.u64()?;
        infos.push(Info { name, shape: ne, ty, offset });
    }

    let data_start = c.p.div_ceil(alignment) * alignment;
    Ok((kv, infos, data_start))
}

/// Validate a split GGUF's parts agree with each other and with the file set
/// [`MmapGguf::open`] actually found on disk, before a single tensor byte is
/// read. `part_paths`/`part_kv`/`part_infos` are one entry per part, in
/// filename order (`part_paths[i]` is 1-based part `i+1`).
///
/// Checked, per part: `split.no` equals the part's own 0-based index
/// (llama.cpp writes `split.no` 0-based against 1-based filenames - a
/// deliberate off-by-one this pins with a dedicated test rather than
/// re-deriving it from a spec each time); `split.count` equals the number of
/// files this open actually located (not just what any one part claims);
/// `general.architecture` agrees with part 1's. Checked once, across all
/// parts: if any part carries `split.tensors.count`, every part that
/// carries it agrees, and the summed tensor count across all parts matches
/// it exactly.
///
/// Every error names the offending part's path - the whole point of
/// validating up front is that a missing or mismatched part is reported as
/// itself, not discovered as "tensor X not found" after decoding has
/// already started.
#[cfg(not(target_arch = "wasm32"))]
fn validate_split(part_paths: &[String], part_kv: &[BTreeMap<String, GgufValue>], part_infos: &[Vec<Info>]) -> Result<(), String> {
    let arch0 = part_kv[0].get("general.architecture").and_then(|v| v.as_str());
    let mut declared_tensor_count: Option<u64> = None;
    let mut total_tensors = 0usize;
    for (i, kv) in part_kv.iter().enumerate() {
        let path = &part_paths[i];
        match kv.get("split.no").and_then(|v| v.as_u64()) {
            Some(no) if no as usize == i => {}
            Some(no) => return Err(format!("gguf: {path}: split.no={no} but this file is part {} of the split (0-based index should be {i})", i + 1)),
            None => return Err(format!("gguf: {path}: split file is missing split.no")),
        }
        match kv.get("split.count").and_then(|v| v.as_u64()) {
            Some(count) if count as usize == part_paths.len() => {}
            Some(count) => {
                return Err(format!("gguf: {path}: split.count={count} but {} part files were found for this split", part_paths.len()))
            }
            None => return Err(format!("gguf: {path}: split file is missing split.count")),
        }
        let arch = kv.get("general.architecture").and_then(|v| v.as_str());
        if arch != arch0 {
            return Err(format!("gguf: {path}: general.architecture {arch:?} disagrees with part 1's {arch0:?}"));
        }
        if let Some(declared) = kv.get("split.tensors.count").and_then(|v| v.as_u64()) {
            match declared_tensor_count {
                None => declared_tensor_count = Some(declared),
                Some(d) if d == declared => {}
                Some(d) => return Err(format!("gguf: {path}: split.tensors.count={declared} disagrees with an earlier part's {d}")),
            }
        }
        total_tensors += part_infos[i].len();
    }
    if let Some(want) = declared_tensor_count {
        if want as usize != total_tensors {
            return Err(format!(
                "gguf: {}: the {} part files together declare {total_tensors} tensors, but split.tensors.count={want}",
                part_paths[0],
                part_paths.len()
            ));
        }
    }
    Ok(())
}

/// Parse a GGUF byte buffer into fp32 tensors + full KV metadata. Portable core
/// (no fs/Seek); native `load_gguf`/`read` read the file first.
pub fn parse_gguf(bytes: &[u8]) -> Result<GgufModel, String> {
    let (kv, infos, data_start) = parse_header(bytes)?;

    let mut tensors = HashMap::with_capacity(infos.len());
    let mut shapes = BTreeMap::new();
    for info in infos {
        let numel: usize = info.shape.iter().product();
        let start = data_start
            .checked_add(info.offset as usize)
            .ok_or("gguf: tensor offset overflow")?;
        let nbytes = tensor_nbytes(info.ty, numel)
            .ok_or_else(|| format!("gguf: {} unknown type {}", info.name, info.ty))?;
        let raw = bytes
            .get(start..start + nbytes)
            .ok_or_else(|| format!("gguf: {} data out of range", info.name))?;
        let data = dequantize(info.ty, raw, numel)
            .map_err(|e| format!("gguf: {}: {e}", info.name))?;
        shapes.insert(info.name.clone(), info.shape);
        tensors.insert(info.name, data);
    }

    let metadata_strings = kv
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect();

    Ok(GgufModel { tensors, shapes, kv, metadata_strings })
}

/// Read and parse a GGUF file from disk into fp32 tensors + metadata.
///
/// **Test-only, and gated by the compiler rather than by convention.** This
/// slurps the file into an owned buffer and then dequantizes *every* tensor to
/// fp32 in host memory at once - two whole-model copies, the second several
/// times larger than the first for any real quantization tier. Nothing in a
/// production path may pay that: use [`MmapGguf`] (one tensor at a time, from
/// a mapping) or [`read`] (mapped, still whole-model but with no owned copy of
/// the file). The `#[cfg(test)]` is the gate - a production reference does not
/// compile, so this cannot regress into a load path by inattention the way a
/// doc-comment warning could.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub fn load_gguf(path: &str) -> Result<GgufModel, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("gguf: open {path}: {e}"))?;
    parse_gguf(&bytes)
}

/// Read every tensor of a GGUF file as [`StTensor`]s (name + shape + fp32),
/// dequantizing quantized types. Order is not significant (callers remap by
/// name). Retained for the FLUX.2 importer.
///
/// Reads through a mapping rather than slurping the file into an owned
/// buffer. The two differ only in where the quantized bytes live while they
/// are being decoded - every `dequantize` call sees the identical input span,
/// so the fp32 output is byte-identical (pinned by
/// `mapped_and_slurped_reads_agree_bit_for_bit`). What changes is the cost:
/// slurping copies the whole file into anonymous memory first, which on a
/// multi-GB checkpoint is a serial memcpy on the critical path of every
/// process start AND a second resident copy of the file alongside the fp32
/// expansion it is about to produce. Mapped, the decode faults pages in as it
/// reads them, from the page cache when the file is warm, and the peak drops
/// by the file's own size.
#[cfg(not(target_arch = "wasm32"))]
pub fn read(path: &str) -> Result<Vec<StTensor>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("gguf: open {path}: {e}"))?;
    // SAFETY: weight files are treated as immutable for the mapping's
    // lifetime - the same contract `MmapGguf::open` and
    // `safetensors::read_mmap` already rely on for every mapped checkpoint.
    let bytes = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("gguf: mmap {path}: {e}"))?;
    let (_, infos, data_start) = parse_header(&bytes)?;
    let mut out = Vec::with_capacity(infos.len());
    for info in infos {
        let numel: usize = info.shape.iter().product();
        let start = data_start
            .checked_add(info.offset as usize)
            .ok_or("gguf: tensor offset overflow")?;
        let nbytes = tensor_nbytes(info.ty, numel)
            .ok_or_else(|| format!("gguf: {} unknown type {}", info.name, info.ty))?;
        let raw = bytes
            .get(start..start + nbytes)
            .ok_or_else(|| format!("gguf: {} data out of range", info.name))?;
        let data = dequantize(info.ty, raw, numel)
            .map_err(|e| format!("gguf: {}: {e}", info.name))?;
        out.push(StTensor { name: info.name, shape: info.shape, data });
    }
    Ok(out)
}

/// A ggml type id's spelling, for the per-tensor dtype a cost model reads.
/// Thin wrapper over [`GgmlType::name`] - see its doc for why there is only
/// one table now. A type this reader can decode always has a name and one it
/// cannot never gets an invented one.
pub(crate) fn ggml_type_name(ty: u32) -> Option<&'static str> {
    GgmlType::from_id(ty).map(GgmlType::name)
}

/// Block geometry `(elements_per_block, bytes_per_block)` for a ggml type.
/// Thin wrapper over [`GgmlType::block_elems`]/[`GgmlType::block_bytes`].
/// `None` for a type id this reader has no block-format entry for (an
/// IQ/TQ/MXFP4 codebook, or an unknown id) - never a guessed geometry.
pub fn block_geometry(ty: u32) -> Option<(usize, usize)> {
    GgmlType::from_id(ty).map(|t| (t.block_elems(), t.block_bytes()))
}

/// Total on-disk byte count for `numel` elements of `ty`.
pub fn tensor_nbytes(ty: u32, numel: usize) -> Option<usize> {
    let (be, bb) = block_geometry(ty)?;
    Some(numel / be * bb + if numel.is_multiple_of(be) { 0 } else { bb })
}

/// The ggml type id of Q8_0, for a caller matching on
/// [`MmapGguf::raw_tensor_bytes`]'s reported type.
pub const TYPE_Q8_0: u32 = T_Q8_0;

/// Elements per Q8_0 block, and the block's on-disk byte size (a 2-byte fp16
/// scale followed by 32 int8 values).
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Expand elements `[e0, e1)` of a block-quantized tensor's raw bytes into
/// `out`, which is CLEARED first. Both bounds must be multiples of `ty`'s
/// [`GgmlType::block_elems`], because a quant block is the smallest
/// independently decodable unit - a caller wanting an unaligned range must
/// expand the enclosing blocks and slice.
///
/// This exists so that a caller reading a sub-range of a large quantized
/// tensor (one weight matrix's rows, or one column block of a fused matrix)
/// does not have to expand the whole tensor to reach it, and does not have to
/// reimplement the block layout to avoid that. It decodes through the SAME
/// per-block decoder every other read path uses (`GgmlType`'s internal
/// decoder lookup is private precisely so there is only one place that
/// mapping is made), so the values are identical to what [`dequantize`] would
/// have produced for those positions. `Err` for F32/F16/BF16 (no per-block
/// decoder - those are not "one block, one shared scale" formats) as well as
/// for a misaligned or inverted range or a range past the end of `raw`.
pub fn block_expand(ty: GgmlType, raw: &[u8], e0: usize, e1: usize, out: &mut Vec<f32>) -> Result<(), String> {
    let f = ty.block_decoder().ok_or_else(|| format!("gguf: block_expand: {} has no per-block decoder", ty.name()))?;
    let be = ty.block_elems();
    let bb = ty.block_bytes();
    if !e0.is_multiple_of(be) || !e1.is_multiple_of(be) {
        return Err(format!("gguf: block_expand range [{e0}, {e1}) is not block-aligned ({be})"));
    }
    if e1 < e0 {
        return Err(format!("gguf: block_expand range [{e0}, {e1}) is inverted"));
    }
    let (b0, b1) = (e0 / be, e1 / be);
    if b1 * bb > raw.len() {
        return Err(format!("gguf: block_expand needs {} bytes, tensor has {}", b1 * bb, raw.len()));
    }
    out.clear();
    out.reserve(e1 - e0);
    for b in b0..b1 {
        f(&raw[b * bb..(b + 1) * bb], out);
    }
    Ok(())
}

/// [`block_expand`] pinned to Q8_0 - kept as a thin, differently-named
/// wrapper because every existing caller (`gguf::int8_direct::try_i8_rect`)
/// already spells this name and there is no reason to touch call sites that
/// are not otherwise changing in this milestone.
pub fn q8_0_expand(raw: &[u8], e0: usize, e1: usize, out: &mut Vec<f32>) -> Result<(), String> {
    block_expand(GgmlType::Q8_0, raw, e0, e1, out)
}

/// f16 (ggml_half) at byte offset `i` in `b` → f32.
fn f16(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Dequantize a tensor's raw bytes to `numel` fp32 values. Dispatches through
/// [`GgmlType`] - the one type table - so this can never decode a type with a
/// geometry [`block_geometry`]/[`tensor_nbytes`] disagree about.
pub(crate) fn dequantize(ty: u32, raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    let Some(t) = GgmlType::from_id(ty) else {
        return match ty {
            // The remaining IQ grid codebooks (M18 covered MXFP4/IQ4_NL/
            // IQ4_XS/TQ1_0/TQ2_0; IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/
            // IQ3_S need a large NGRID lookup table each, deferred).
            16 | 17 | 18 | 19 | 21 | 22 | 29 => Err(format!("gguf: type {ty} (IQ grid codebook) dequant not yet implemented")),
            other => Err(format!("gguf: unsupported type {other}")),
        };
    };
    Ok(match t {
        GgmlType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        GgmlType::F16 => raw.chunks_exact(2).map(|b| crate::safetensors::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        GgmlType::BF16 => raw.chunks_exact(2).map(|b| crate::safetensors::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        // A plain scalar array, not a quantized block format - `numel`
        // elements read straight off the mapping with no scale to apply.
        // `as f32` is a lossy narrowing for I64/F64 outside f32's exact
        // integer/mantissa range, the same tradeoff this engine already
        // accepts everywhere else by being fp32-only; these ids almost never
        // carry model weights (auxiliary/index tensors, if present at all).
        GgmlType::F64 => raw.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        GgmlType::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        GgmlType::I16 => raw.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32).collect(),
        GgmlType::I32 => raw.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32).collect(),
        GgmlType::I64 => raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        GgmlType::Q4_0 => deq_blocks(raw, numel, 18, deq_q4_0),
        GgmlType::Q4_1 => deq_blocks(raw, numel, 20, deq_q4_1),
        GgmlType::Q5_0 => deq_blocks(raw, numel, 22, deq_q5_0),
        GgmlType::Q5_1 => deq_blocks(raw, numel, 24, deq_q5_1),
        GgmlType::Q8_0 => deq_blocks(raw, numel, 34, deq_q8_0),
        GgmlType::Q8_1 => deq_blocks(raw, numel, 36, deq_q8_1),
        GgmlType::Q2K => deq_blocks(raw, numel, 84, deq_q2_k),
        GgmlType::Q3K => deq_blocks(raw, numel, 110, deq_q3_k),
        GgmlType::Q4K => deq_blocks(raw, numel, 144, deq_q4_k),
        GgmlType::Q5K => deq_blocks(raw, numel, 176, deq_q5_k),
        GgmlType::Q6K => deq_blocks(raw, numel, 210, deq_q6_k),
        GgmlType::Q8K => deq_blocks(raw, numel, 292, deq_q8_k),
        GgmlType::MXFP4 => deq_blocks(raw, numel, 17, deq_mxfp4),
        GgmlType::IQ4NL => deq_blocks(raw, numel, 18, deq_iq4_nl),
        GgmlType::IQ4XS => deq_blocks(raw, numel, 136, deq_iq4_xs),
        GgmlType::TQ1_0 => deq_blocks(raw, numel, 54, deq_tq1_0),
        GgmlType::TQ2_0 => deq_blocks(raw, numel, 66, deq_tq2_0),
    })
}

/// Walk `raw` block by block, appending each block's dequantized values, then
/// truncate to `numel` (the last block may be partially used).
///
/// Blocks are independent - block `i` reads only `raw[i*block_bytes..]` and
/// owns exactly its own span of the output - so on native the walk is split
/// across the CPU scheduler's pool. The result is bit-identical to the serial
/// walk: every `f` call is unchanged and lands at the same output offset, only
/// the order in which the calls happen changes. `crates/checkpoint`'s own
/// `multi_block_dequant_lays_each_block_into_its_own_output_span` and
/// `a_partial_trailing_block_is_decoded_then_truncated` pin that mapping
/// against a per-block oracle, by bit pattern.
///
/// This is not micro-tuning: dequantizing a large quantized checkpoint to fp32
/// is on the critical path of every GGUF-backed model load, and for a streamed
/// int8 tier it runs again for every transformer block of every generation.
///
/// `GROUP` blocks are decoded into one small stack-local `Vec` before being
/// copied into the output. The temporary is what lets each `f` keep its
/// existing "append to a Vec" signature (all twelve decoders share it), and at
/// this size it stays in cache, so the copy costs far less than the
/// parallelism buys. On wasm (no threads, and `backend-cpu` does not build
/// there) the serial walk is kept verbatim.
#[cfg(not(target_arch = "wasm32"))]
fn deq_blocks(raw: &[u8], numel: usize, block_bytes: usize, f: fn(&[u8], &mut Vec<f32>)) -> Vec<f32> {
    const GROUP: usize = 64;
    let nblocks = raw.len() / block_bytes;
    if nblocks == 0 {
        return Vec::new();
    }
    // Element count per block, derived from this decoder rather than passed
    // in: the whole block stream produces the same count for every block, so
    // one decode of block 0 measures it exactly.
    let mut probe = Vec::new();
    f(&raw[..block_bytes], &mut probe);
    let be = probe.len();
    let mut out = vec![0f32; nblocks * be];
    backend_cpu::par::chunks_mut(&mut out, GROUP * be, |g, dst| {
        let first = g * GROUP;
        let mut tmp = Vec::with_capacity(GROUP * be);
        for i in first..(first + GROUP).min(nblocks) {
            f(&raw[i * block_bytes..(i + 1) * block_bytes], &mut tmp);
        }
        dst.copy_from_slice(&tmp);
    });
    out.truncate(numel);
    out
}

/// The serial walk [`deq_blocks`] documents - kept for wasm, which has neither
/// threads nor `backend-cpu`.
#[cfg(target_arch = "wasm32")]
fn deq_blocks(raw: &[u8], numel: usize, block_bytes: usize, f: fn(&[u8], &mut Vec<f32>)) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    for blk in raw.chunks_exact(block_bytes) {
        f(blk, &mut out);
    }
    out.truncate(numel);
    out
}

// ---- legacy blocks of 32 ----

fn deq_q4_0(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let qs = &b[2..18];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for j in 0..16 {
        lo[j] = ((qs[j] & 0x0F) as i32 - 8) as f32 * d;
        hi[j] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
    }
    out.extend_from_slice(&lo);
    out.extend_from_slice(&hi);
}

fn deq_q4_1(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let m = f16(b, 2);
    let qs = &b[4..20];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for j in 0..16 {
        lo[j] = (qs[j] & 0x0F) as f32 * d + m;
        hi[j] = (qs[j] >> 4) as f32 * d + m;
    }
    out.extend_from_slice(&lo);
    out.extend_from_slice(&hi);
}

fn deq_q5_0(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        lo[j] = (((qs[j] & 0x0F) | xh0) as i32 - 16) as f32 * d;
        hi[j] = (((qs[j] >> 4) | xh1) as i32 - 16) as f32 * d;
    }
    out.extend_from_slice(&lo);
    out.extend_from_slice(&hi);
}

fn deq_q5_1(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let m = f16(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..24];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        lo[j] = ((qs[j] & 0x0F) | xh0) as f32 * d + m;
        hi[j] = ((qs[j] >> 4) | xh1) as f32 * d + m;
    }
    out.extend_from_slice(&lo);
    out.extend_from_slice(&hi);
}

fn deq_q8_0(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    for j in 0..32 {
        out.push(b[2 + j] as i8 as f32 * d);
    }
}

/// Q8_1: `Q8_0`'s block plus a second fp16 field (`s = d * sum(qs)`), a
/// cached quantity ggml's fused matmul uses to skip a reduction - irrelevant
/// to plain dequant, which is `w = d*q[i]`, identical to Q8_0. Practically
/// never a STORED tensor type (it exists in ggml as an activation-quantize
/// target, not a checkpoint format), but a GGUF file is free to declare one,
/// and refusing to open the whole file over a tensor type nothing reads is
/// the worse failure.
fn deq_q8_1(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    for j in 0..32 {
        out.push(b[4 + j] as i8 as f32 * d);
    }
}

// ---- k-quant super-blocks of 256 ----

/// 6-bit packed scale/min extraction shared by Q4_K/Q5_K (ggml get_scale_min_k4).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn deq_q2_k(b: &[u8], out: &mut Vec<f32>) {
    let scales = &b[0..16];
    let qs = &b[16..80];
    let d = f16(b, 80);
    let dmin = f16(b, 82);
    let mut is = 0;
    let mut qoff = 0;
    for _n in (0..QK_K).step_by(128) {
        let mut shift = 0;
        for _j in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            for l in 0..16 {
                out.push(dl * ((qs[qoff + l] >> shift) & 3) as f32 - ml);
            }
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            for l in 0..16 {
                out.push(dl * ((qs[qoff + 16 + l] >> shift) & 3) as f32 - ml);
            }
            shift += 2;
        }
        qoff += 32;
    }
}

fn deq_q3_k(b: &[u8], out: &mut Vec<f32>) {
    const KM1: u32 = 0x0303_0303;
    const KM2: u32 = 0x0f0f_0f0f;
    let hmask = &b[0..32];
    let qs = &b[32..96];
    let s = &b[96..108];
    let d_all = f16(b, 108);

    // Unpack the 12 packed bytes into 16 signed 6-bit scales (ggml layout).
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    aux[1] = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
    aux[2] = u32::from_le_bytes([s[8], s[9], s[10], s[11]]);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
    aux[3] = ((aux[1] >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
    aux[0] = (aux[0] & KM2) | (((tmp) & KM1) << 4);
    aux[1] = (aux[1] & KM2) | (((tmp >> 2) & KM1) << 4);
    let mut sc = [0i8; 16];
    for (k, a) in aux.iter().enumerate() {
        let by = a.to_le_bytes();
        for j in 0..4 {
            sc[k * 4 + j] = by[j] as i8;
        }
    }

    let mut m: u8 = 1;
    let mut is = 0;
    let mut qoff = 0;
    for _n in (0..QK_K).step_by(128) {
        let mut shift = 0;
        for _j in 0..4 {
            let dl = d_all * (sc[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let sub = if hmask[l] & m != 0 { 0 } else { 4 };
                out.push(dl * (((qs[qoff + l] >> shift) & 3) as i32 - sub) as f32);
            }
            let dl = d_all * (sc[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let sub = if hmask[16 + l] & m != 0 { 0 } else { 4 };
                out.push(dl * (((qs[qoff + 16 + l] >> shift) & 3) as i32 - sub) as f32);
            }
            shift += 2;
            m <<= 1;
        }
        qoff += 32;
    }
}

fn deq_q4_k(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let dmin = f16(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    let mut is = 0;
    let mut qoff = 0;
    for _ in (0..QK_K).step_by(64) {
        let (sc, m) = scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for l in 0..32 {
            out.push(d1 * (qs[qoff + l] & 0x0F) as f32 - m1);
        }
        for l in 0..32 {
            out.push(d2 * (qs[qoff + l] >> 4) as f32 - m2);
        }
        qoff += 32;
        is += 2;
    }
}

fn deq_q5_k(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let dmin = f16(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    let mut is = 0;
    let mut qoff = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for _ in (0..QK_K).step_by(64) {
        let (sc, m) = scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out.push(d1 * ((ql[qoff + l] & 0x0F) as i32 + hi) as f32 - m1);
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out.push(d2 * ((ql[qoff + l] >> 4) as i32 + hi) as f32 - m2);
        }
        qoff += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn deq_q6_k(b: &[u8], out: &mut Vec<f32>) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = f16(b, 208);
    let mut yb = [0.0f32; QK_K];
    for ni in 0..2 {
        let qlo = ni * 64;
        let qho = ni * 32;
        let sco = ni * 8;
        let yo = ni * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[qlo + l] & 0x0F) as i32 | ((qh[qho + l] & 3) as i32) << 4) - 32;
            let q2 = ((ql[qlo + l + 32] & 0x0F) as i32 | (((qh[qho + l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[qlo + l] >> 4) as i32 | (((qh[qho + l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[qlo + l + 32] >> 4) as i32 | (((qh[qho + l] >> 6) & 3) as i32) << 4) - 32;
            yb[yo + l] = d * sc[sco + is] as i8 as f32 * q1 as f32;
            yb[yo + l + 32] = d * sc[sco + is + 2] as i8 as f32 * q2 as f32;
            yb[yo + l + 64] = d * sc[sco + is + 4] as i8 as f32 * q3 as f32;
            yb[yo + l + 96] = d * sc[sco + is + 6] as i8 as f32 * q4 as f32;
        }
    }
    out.extend_from_slice(&yb);
}

fn deq_q8_k(b: &[u8], out: &mut Vec<f32>) {
    let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    for j in 0..QK_K {
        out.push(b[4 + j] as i8 as f32 * d);
    }
}

// ---- codebook families (MXFP4, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0) ----
//
// Every struct layout, table and loop nesting below is transcribed from
// ggml's own `ggml-common.h` (block structs + `kvalues_*` tables) and
// `ggml-quants.c` (`dequantize_row_*`), fetched from
// `github.com/ggml-org/llama.cpp` at M18 time - this repo has no real
// MXFP4/IQ/TQ checkpoint to derive the byte layout from by inspection the
// way M8's K-quant work did, so the reference source is the ground truth
// instead. The loop NESTING in `deq_tq1_0`/`deq_tq2_0` is preserved exactly
// as ggml writes it (two byte-range chunks, each with the full inner
// digit/shift loop) rather than "simplified" into one pass over every byte
// - flattening it changes the OUTPUT ORDER, not just the code shape (caught
// while transcribing: a single merged loop over all 48 tq1_0 `qs` bytes
// produces a different permutation than ggml's own two-chunk split).

/// The E2M1 magnitude/sign table MXFP4 shares with NVFP4 (`kvalues_fp4` /
/// `kvalues_mxfp4` in `ggml-common.h`), values already 2x the true E2M1
/// magnitude (`[0, 0.5, 1, 1.5, 2, 3, 4, 6]` and their negatives) - ggml's
/// own comment says so, and [`e8m0_to_fp32_half`]'s scale is likewise
/// halved so the doubling cancels exactly.
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// ggml's non-linear 4-bit codebook (`kvalues_iq4nl` in `ggml-common.h`),
/// shared verbatim by IQ4_NL and IQ4_XS.
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// `ggml_e8m0_to_fp32_half` (`ggml-impl.h`): E8M0 (unsigned 8-bit exponent,
/// bias 127) decoded to HALF its true value (`2^(e-127)/2` = `2^(e-128)`) -
/// [`KVALUES_MXFP4`] is the true E2M1 magnitude DOUBLED, so halving the
/// scale here is what makes `scale * code` land on the real value. Ported
/// bit-for-bit via the same fp32-exponent-field placement ggml uses rather
/// than computed as `2f32.powi(e as i32 - 128)`: `e` in `{0, 1}` lands in
/// fp32's subnormal range, where a computed power is not guaranteed to
/// round identically to the direct bit pattern.
fn e8m0_to_fp32_half(e: u8) -> f32 {
    let bits: u32 = if e < 2 { 0x0020_0000u32 << e } else { (e as u32 - 1) << 23 };
    f32::from_bits(bits)
}

/// `block_mxfp4 { uint8_t e; uint8_t qs[16]; }`, `dequantize_row_mxfp4`.
fn deq_mxfp4(b: &[u8], out: &mut Vec<f32>) {
    let d = e8m0_to_fp32_half(b[0]);
    let qs = &b[1..17];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for j in 0..16 {
        lo[j] = KVALUES_MXFP4[(qs[j] & 0x0F) as usize] as f32 * d;
        hi[j] = KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32 * d;
    }
    out.extend_from_slice(&lo);
    out.extend_from_slice(&hi);
}

/// `block_iq4_nl { ggml_half d; uint8_t qs[16]; }`, `dequantize_row_iq4_nl`.
fn deq_iq4_nl(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let qs = &b[2..18];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for j in 0..16 {
        lo[j] = KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32 * d;
        hi[j] = KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32 * d;
    }
    out.extend_from_slice(&lo);
    out.extend_from_slice(&hi);
}

/// `block_iq4_xs { ggml_half d; uint16_t scales_h; uint8_t scales_l[4];
/// uint8_t qs[128]; }`, `dequantize_row_iq4_xs` - 8 sub-blocks of 32
/// elements, each with its own 6-bit signed scale (a 4-bit `scales_l`
/// nibble plus a 2-bit `scales_h` field, offset by -32) applied to
/// [`KVALUES_IQ4NL`], the same codebook IQ4_NL uses.
fn deq_iq4_xs(b: &[u8], out: &mut Vec<f32>) {
    let d = f16(b, 0);
    let scales_h = u16::from_le_bytes([b[2], b[3]]);
    let scales_l = &b[4..8];
    let qs = &b[8..136];
    for ib in 0..(QK_K / 32) {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32 | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let qblk = &qs[ib * 16..ib * 16 + 16];
        let mut lo = [0.0f32; 16];
        let mut hi = [0.0f32; 16];
        for j in 0..16 {
            lo[j] = KVALUES_IQ4NL[(qblk[j] & 0x0F) as usize] as f32 * dl;
            hi[j] = KVALUES_IQ4NL[(qblk[j] >> 4) as usize] as f32 * dl;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
}

/// One base-3 ternary digit from a `pow3`-multiplied byte
/// (`(uint16_t)(byte*pow3[n])*3 >> 8` in ggml, extracting digit `n` of a
/// byte packing 5 trits) minus 1, giving `{-1, 0, 1}`. `wrapping_mul`
/// reproduces C's implicit `uint8_t` truncation on overflow exactly - this
/// is not a defensive choice, the algorithm DEPENDS on the wraparound.
fn tq_trit(byte: u8, pow3: u16) -> f32 {
    let q = (byte as u16).wrapping_mul(pow3);
    let xi = (q as u32 * 3) >> 8;
    xi as f32 - 1.0
}

/// `block_tq1_0 { uint8_t qs[48]; uint8_t qh[4]; ggml_half d; }`,
/// `dequantize_row_tq1_0`. `qs` decodes in TWO chunks (`[0..32]` then
/// `[32..48]`), each running the full 5-digit loop before the next chunk
/// starts - not one pass over all 48 bytes per digit, which would emit a
/// different element order (48 is not a multiple of 32, which is why ggml's
/// own C source has two loops here instead of one).
fn deq_tq1_0(b: &[u8], out: &mut Vec<f32>) {
    const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];
    let qs = &b[0..48];
    let qh = &b[48..52];
    let d = f16(b, 52);
    for &chunk in &[&qs[0..32], &qs[32..48]] {
        for &p in &POW3[0..5] {
            for &byte in chunk {
                out.push(tq_trit(byte, p) * d);
            }
        }
    }
    for &p in &POW3[0..4] {
        for &byte in qh {
            out.push(tq_trit(byte, p) * d);
        }
    }
}

/// `block_tq2_0 { uint8_t qs[64]; ggml_half d; }`, `dequantize_row_tq2_0`.
/// `qs` decodes in two 32-byte chunks, each running all four 2-bit shift
/// positions before the next chunk - same "chunk outside, digit/shift loop
/// inside" shape as `deq_tq1_0`, for the same reason (preserving ggml's own
/// element order).
fn deq_tq2_0(b: &[u8], out: &mut Vec<f32>) {
    let qs = &b[0..64];
    let d = f16(b, 64);
    for chunk in qs.chunks_exact(32) {
        for l in 0..4 {
            for &byte in chunk {
                let q = ((byte >> (l * 2)) & 3) as f32;
                out.push((q - 1.0) * d);
            }
        }
    }
}

/// A memory-mapped GGUF file (or split file set) with an on-demand,
/// per-tensor dequant accessor.
///
/// [`open`](MmapGguf::open) mmaps the file and parses only the header (KV +
/// tensor infos); no tensor data is read. [`tensor`](MmapGguf::tensor)
/// dequantizes exactly one tensor's block range from the mmap, so peak host
/// memory is bounded by a single tensor's fp32 expansion — never the whole
/// model. Values are byte-identical to the eager [`parse_gguf`].
///
/// A split GGUF (`<base>-NNNNN-of-MMMMM.gguf`, one file per part, no tensor
/// straddling a part boundary) opens exactly like a single file - pass any
/// one part's path and every sibling is located and mapped alongside it.
/// `mmaps` therefore holds one mapping per part (`mmaps[0]` is always part
/// 1) instead of the single-file case's one-element vec, and `index` records
/// which part each tensor's bytes live in.
#[cfg(not(target_arch = "wasm32"))]
pub struct MmapGguf {
    mmaps: Vec<memmap2::Mmap>,
    kv: BTreeMap<String, GgufValue>,
    /// name → (part index into `mmaps`, ggml type, absolute byte start
    /// within that part, on-disk byte length, element count).
    index: HashMap<String, (usize, u32, usize, usize, usize)>,
    shapes: BTreeMap<String, Vec<usize>>,
    order: Vec<String>,
    /// Load-progress accounting: cumulative on-disk bytes walked, reported
    /// through [`crate::load_progress`].
    meter: crate::load_progress::LoadMeter,
}

#[cfg(not(target_arch = "wasm32"))]
impl MmapGguf {
    /// Open + mmap `path` (and, if it names one part of a split GGUF, every
    /// sibling part) and parse only the header(s) (no tensor bytes are read).
    pub fn open(path: &str) -> Result<MmapGguf, String> {
        let dir = std::path::Path::new(path).parent();
        let fname = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("gguf: {path}: not a valid file name"))?;
        let part_paths: Vec<String> = match crate::split::split_name(fname, "gguf") {
            Some((base, _part, count, width)) => (1..=count)
                .map(|p| {
                    let name = crate::split::split_sibling(base, p, count, width, "gguf");
                    match dir {
                        Some(d) if !d.as_os_str().is_empty() => d.join(&name).to_string_lossy().into_owned(),
                        _ => name,
                    }
                })
                .collect(),
            None => vec![path.to_string()],
        };

        let mut mmaps = Vec::with_capacity(part_paths.len());
        let mut part_kv = Vec::with_capacity(part_paths.len());
        let mut part_infos = Vec::with_capacity(part_paths.len());
        let mut part_data_start = Vec::with_capacity(part_paths.len());
        for p in &part_paths {
            let file = std::fs::File::open(p).map_err(|e| format!("gguf: open {p}: {e}"))?;
            // SAFETY: weight files are treated as immutable for the mapping's lifetime.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("gguf: mmap {p}: {e}"))?;
            let (kv, infos, data_start) = parse_header(&mmap)?;
            part_kv.push(kv);
            part_infos.push(infos);
            part_data_start.push(data_start);
            mmaps.push(mmap);
        }

        if part_paths.len() > 1 {
            validate_split(&part_paths, &part_kv, &part_infos)?;
        }

        let mut index = HashMap::new();
        let mut shapes = BTreeMap::new();
        let mut order = Vec::new();
        let mut tensor_bytes: u64 = 0;
        for (part_idx, infos) in part_infos.into_iter().enumerate() {
            let data_start = part_data_start[part_idx];
            let part_len = mmaps[part_idx].len();
            for info in infos {
                let numel: usize = info.shape.iter().product();
                let start = data_start
                    .checked_add(info.offset as usize)
                    .ok_or("gguf: tensor offset overflow")?;
                let nbytes = tensor_nbytes(info.ty, numel)
                    .ok_or_else(|| format!("gguf: {} unknown type {}", info.name, info.ty))?;
                if start + nbytes > part_len {
                    return Err(format!("gguf: {} data out of range", info.name));
                }
                if index.contains_key(&info.name) {
                    return Err(format!("gguf: {path}: tensor {} appears in more than one split part", info.name));
                }
                tensor_bytes += nbytes as u64;
                index.insert(info.name.clone(), (part_idx, info.ty, start, nbytes, numel));
                shapes.insert(info.name.clone(), info.shape);
                order.push(info.name);
            }
        }
        // Part 1's KV represents the file: `validate_split` already checked
        // every other part agrees on the keys that matter
        // (`general.architecture`, `split.count`, `split.tensors.count`).
        let kv = part_kv.into_iter().next().expect("at least one part");
        let meter = crate::load_progress::LoadMeter::new(path.to_string(), tensor_bytes);
        Ok(MmapGguf { mmaps, kv, index, shapes, order, meter })
    }

    /// Tensor names, in the file's declared order.
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// A tensor's shape (torch order), if present.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(|s| s.as_slice())
    }

    /// A tensor's own ggml quant type, as a name (`"F32"`, `"BF16"`, `"Q4_K"`,
    /// …), if present.
    ///
    /// Per TENSOR, not per file: a GGUF release routinely stores different
    /// layers at different precisions (attention kept wide, MoE experts
    /// squeezed), which is exactly why a placement/upload cost model must ask
    /// each tensor rather than the checkpoint. `None` for a type id this
    /// reader has no name for - never a guess.
    pub fn dtype(&self, name: &str) -> Option<&'static str> {
        ggml_type_name(self.index.get(name)?.1)
    }

    /// The full KV metadata map.
    pub fn kv(&self) -> &BTreeMap<String, GgufValue> {
        &self.kv
    }

    /// Total tensor element count (summed over shapes).
    pub fn param_count(&self) -> u64 {
        self.shapes.values().map(|s| s.iter().product::<usize>() as u64).sum()
    }

    /// The KV metadata as a JSON object (every key preserved; arrays recurse).
    pub fn config(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        for (k, v) in &self.kv {
            m.insert(k.clone(), gval_json(v));
        }
        serde_json::Value::Object(m)
    }

    /// The [`ModelCard`] derived from the standardized GGUF keys.
    pub fn model_card(&self) -> ModelCard {
        card_from_kv(&self.kv, self.param_count())
    }

    /// The tokenizer embedded in `tokenizer.ggml.*` KV, if the file declares one
    /// (byte-identical to [`GgufModel::tokenizer`] — both read the same KV map).
    pub fn tokenizer(&self) -> Option<GgufTokenizer> {
        tokenizer_from_kv(&self.kv)
    }

    /// Dequantize exactly one tensor to fp32, decoding on access from the mmap.
    /// `None` if the name is unknown; `Some(Err)` if the tensor exists but its
    /// quant type is unsupported (IQ/TQ/MXFP4 codebooks). Byte-identical to the
    /// eager [`parse_gguf`] for the same tensor.
    pub fn tensor(&self, name: &str) -> Option<Result<Vec<f32>, String>> {
        let &(part, ty, start, nbytes, numel) = self.index.get(name)?;
        self.meter.note(nbytes as u64);
        let raw = &self.mmaps[part][start..start + nbytes];
        Some(dequantize(ty, raw, numel).map_err(|e| format!("gguf: {name}: {e}")))
    }

    /// Dequantize elements `[start_elem, start_elem + len_elem)` of `name`'s
    /// FLAT, row-major layout - for a `[rows, cols]` tensor, one row is `cols`
    /// elements starting at `row * cols`. The GGUF twin of
    /// [`crate::mmap::MmapSafetensors::tensor_f32_range`], and it exists for
    /// the same caller: a handful of embedding rows out of a
    /// `[vocab, d_model]` table where decoding the whole thing (5.1 GB as f32
    /// at Qwen3.8-27B's 248320 x 5120) is not an option, and even
    /// [`crate::TensorSource::with_tensor_chunks`]' scan from offset 0 would
    /// read the entire table to reach a row near its end.
    ///
    /// Only the quant BLOCKS the range touches are decoded, so the cost is
    /// `O(len_elem)` regardless of where in the tensor the range sits. The
    /// requested range need not be block-aligned: the decode is widened to
    /// whole blocks and the result sliced back, so the values are bit-identical
    /// to the corresponding slice of [`Self::tensor`]'s whole-tensor output.
    ///
    /// `None` when `name` is absent, when the range falls even partially
    /// outside the tensor (never a silently truncated read), or when the ggml
    /// type has no known block geometry - the last of which is exactly the set
    /// [`dequantize`] cannot decode either, so it is "unsupported type", not a
    /// missing fast path.
    pub fn tensor_range(&self, name: &str, start_elem: usize, len_elem: usize) -> Option<Result<Vec<f32>, String>> {
        let &(part, ty, start, nbytes, numel) = self.index.get(name)?;
        let end_elem = start_elem.checked_add(len_elem)?;
        if end_elem > numel {
            return None;
        }
        let (block_elems, block_bytes) = block_geometry(ty)?;
        let b0 = start_elem / block_elems;
        let b1 = end_elem.div_ceil(block_elems);
        let span = &self.mmaps[part][start + b0 * block_bytes..(start + b1 * block_bytes).min(start + nbytes)];
        // `dequantize` truncates its output to the requested count exactly as
        // the whole-tensor path does, so asking for everything from this
        // range's first block boundary up to `end_elem` and slicing the
        // leading partial block off is byte-identical to a full decode.
        let lead = start_elem - b0 * block_elems;
        Some(dequantize(ty, span, lead + len_elem).map(|v| v[lead..].to_vec()).map_err(|e| format!("gguf: {name}: {e}")))
    }

    /// `name`'s RAW on-disk bytes (still quantized, if applicable) plus its
    /// ggml type id - for a caller that wants to dequantize INDEPENDENTLY of
    /// [`Self::tensor`]'s own [`dequantize`] (e.g. a quantization-exactness
    /// parity test that reimplements a quant format's dequant math from the
    /// GGUF spec, to check this reader's own output rather than trust it -
    /// `crates/ltxv/tests/gguf_quant_real.rs` is the first such caller).
    /// `None` if the name is unknown. Every other public accessor
    /// ([`Self::tensor`], the [`crate::TensorSource`] impl) still goes
    /// through [`dequantize`] unchanged - this is purely an additional read
    /// path, not a replacement for the decoded one.
    pub fn raw_tensor_bytes(&self, name: &str) -> Option<(&[u8], u32)> {
        // Deliberately NOT reimplemented on `raw_blocks` (below), even though
        // both read the same `self.index` entry: this accessor's contract is
        // "the bytes and the RAW type id, whatever it is" (its own doc: a
        // parity test needs this for a type the reader cannot even
        // dequantize), while `raw_blocks` declines for a type `GgmlType`
        // does not recognize - `raw_blocks`' contract is "block format that
        // DECODES them". `weightio::WeightReader::nbytes` depends on this
        // one working for an unsupported quant type too; narrowing it would
        // silently break that guarantee.
        let &(part, ty, start, nbytes, _numel) = self.index.get(name)?;
        Some((&self.mmaps[part][start..start + nbytes], ty))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl crate::TensorSource for MmapGguf {
    /// Streaming source: dequantize exactly one tensor from the mmap, lend it
    /// to `f`, drop it on return — the same peak-host-≈-one-tensor contract
    /// [`crate::weightio::WeightReader`]'s impl gives for safetensors, so a
    /// GGUF-backed import can share the identical streaming builder code
    /// (`ParamStore::new_with_roles_src` and friends) unchanged.
    ///
    /// A quant-type dequant failure (IQ/TQ/MXFP4 codebooks, `Some(Err(..))`)
    /// panics rather than returning `false` — per this repo's "validate at the
    /// boundary, fail loudly" rule (`AGENTS.md`), silently reporting "not
    /// found" here would surface as a two-way-coverage "missing tensor" error
    /// at the caller, masking a real dequant failure as a naming bug.
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.tensor(name) {
            Some(Ok(v)) => {
                f(&v);
                true
            }
            Some(Err(e)) => panic!("gguf: {name}: dequant failed: {e}"),
            None => false,
        }
    }

    /// Bounded chunked dequant: walk the tensor **block by block**, decoding
    /// at most `max_elems` elements into one scratch at a time.
    ///
    /// Without this a GGUF-backed source inherited the trait default, which
    /// materializes the whole tensor and hands it over as one chunk - so a
    /// device upload through `paramstore::upload` (which calls this precisely
    /// to stay bounded) silently paid a whole tensor per weight. On a real
    /// checkpoint the embedding table alone dominates that.
    ///
    /// A quant block is the smallest independently decodable unit, so the
    /// request is rounded DOWN to a whole number of blocks (never below one).
    /// Every chunk therefore starts on a block boundary and is decoded by the
    /// same [`dequantize`] the whole-tensor path uses, over the same bytes -
    /// which is why the streamed values are bit-identical to it, not merely
    /// close. Chunks arrive in order, contiguously, offset in elements.
    ///
    /// A ggml type with no known block geometry falls back to the whole-tensor
    /// path rather than guessing at a layout; today every type
    /// [`dequantize`] supports has geometry, so that arm is unreachable in
    /// practice and exists so an unsupported type fails in `dequantize`'s own
    /// error message rather than in a slicing panic here.
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(&(part, ty, start, nbytes, numel)) = self.index.get(name) else { return false };
        let Some((block_elems, block_bytes)) = block_geometry(ty) else {
            return self.with_tensor(name, &mut |d| f(0, d));
        };
        let per = if max_elems == 0 { numel } else { (max_elems / block_elems).max(1) * block_elems };
        let raw = &self.mmaps[part][start..start + nbytes];
        let mut e0 = 0usize;
        while e0 < numel {
            let e1 = (e0 + per).min(numel);
            // e0 is block-aligned by construction (`per` is a multiple of
            // `block_elems`); the trailing block may be partial, and
            // `dequantize` truncates it to the requested count exactly as the
            // whole-tensor path does.
            let (b0, b1) = (e0 / block_elems, e1.div_ceil(block_elems));
            let span = &raw[b0 * block_bytes..(b1 * block_bytes).min(nbytes)];
            match dequantize(ty, span, e1 - e0) {
                Ok(v) => f(e0 as u64, &v),
                Err(e) => panic!("gguf: {name}: dequant failed: {e}"),
            }
            e0 = e1;
        }
        // One event for the whole call, after the last chunk - the fallback
        // arm above reports through `with_tensor` -> `tensor` instead.
        self.meter.note(nbytes as u64);
        true
    }

    /// Zero-copy: `name`'s on-disk bytes reinterpreted as `u32` words, borrowed
    /// straight from the mapping. `None` unless the tensor's own ggml type is
    /// already `F32` (nothing to dequantize) AND its byte range is 4-byte
    /// aligned in the mapping - `bytemuck::try_cast_slice` is the alignment
    /// check, so a misaligned range cleanly falls through to `None` rather
    /// than panicking. A GGUF release commonly leaves 1-D tensors (norms,
    /// biases) in plain F32 while quantizing the 2-D weights, so this fires
    /// selectively per tensor, not per file.
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        let &(part, ty, start, nbytes, _numel) = self.index.get(name)?;
        if ty != T_F32 {
            return None;
        }
        bytemuck::try_cast_slice::<u8, u32>(&self.mmaps[part][start..start + nbytes])
            .ok()
            .inspect(|_| {
                self.meter.note(nbytes as u64);
            })
    }

    /// `name`'s already-quantized bytes, borrowed from the mapping, plus the
    /// block format that decodes them. `None` when the name is unknown OR
    /// when its ggml type has no [`GgmlType`] entry (an IQ/TQ/MXFP4
    /// codebook) - a caller wanting the raw bytes of an unrecognized type
    /// regardless wants [`Self::raw_tensor_bytes`], which has a wider
    /// contract on purpose (see its doc).
    fn raw_blocks(&self, name: &str) -> Option<(BlockLayout, &[u8])> {
        let &(part, ty, start, nbytes, numel) = self.index.get(name)?;
        let ty = GgmlType::from_id(ty)?;
        self.meter.note(nbytes as u64);
        Some((BlockLayout { ty, numel }, &self.mmaps[part][start..start + nbytes]))
    }

    /// Element count of `name`, without decoding - known from the header for
    /// every ggml type, quantized or not, since the caller only needs to know
    /// how many f32s a dequantized [`with_tensor`](Self::with_tensor) call
    /// will produce.
    fn numel(&self, name: &str) -> Option<usize> {
        self.index.get(name).map(|&(_, _, _, _, numel)| numel)
    }
}

/// Convert one GGUF metadata value to `serde_json::Value` (arrays recurse).
#[cfg(not(target_arch = "wasm32"))]
fn gval_json(v: &GgufValue) -> serde_json::Value {
    use serde_json::json;
    match v {
        GgufValue::U8(x) => json!(x),
        GgufValue::I8(x) => json!(x),
        GgufValue::U16(x) => json!(x),
        GgufValue::I16(x) => json!(x),
        GgufValue::U32(x) => json!(x),
        GgufValue::I32(x) => json!(x),
        GgufValue::F32(x) => json!(x),
        GgufValue::Bool(x) => json!(x),
        GgufValue::String(x) => json!(x),
        GgufValue::U64(x) => json!(x),
        GgufValue::I64(x) => json!(x),
        GgufValue::F64(x) => json!(x),
        GgufValue::Array(a) => serde_json::Value::Array(a.iter().map(gval_json).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF-backed device upload must be bounded the way a safetensors one
    /// already is. `paramstore::upload` pulls every weight through
    /// [`crate::TensorSource::with_tensor_chunks`] precisely so the host never
    /// holds a whole tensor; a source that has no chunked path inherits the
    /// trait default, which materializes the lot and quietly gives that
    /// guarantee back.
    ///
    /// It is not a theoretical cost. Qwen3-8B's `token_embd.weight` is
    /// 151936x4096, and the FLUX.2 text encoder loads it - the whole point of
    /// streaming that encoder was to stop holding tensors of that size.
    ///
    /// Two assertions, because either alone is worthless: the chunked read
    /// must be BOUNDED, and it must be **bit-identical** to the whole-tensor
    /// dequant. A bounded reader that decoded the wrong values would pass a
    /// memory test perfectly.
    #[test]
    fn gguf_chunked_reads_are_bounded_and_bit_identical_to_the_whole_tensor() {
        use crate::testalloc::peak_live;
        use crate::TensorSource;

        // A Q8_0 tensor big enough that one copy is unmistakable: 262144
        // elements = 8192 blocks = 1 MiB as f32.
        let numel = 262_144usize;
        let blocks = numel / Q8_0_BLOCK_ELEMS;
        let mut raw = Vec::with_capacity(blocks * 34);
        for b in 0..blocks {
            raw.extend(half::f16::from_f32(0.5 + (b % 7) as f32 * 0.125).to_le_bytes());
            raw.extend((0..32).map(|i| ((b + i) % 251) as i8 as u8));
        }
        let path = std::env::temp_dir().join(format!("brain-gguf-chunked-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        crate::gguf_write::write(
            &path,
            &[("general.architecture".to_string(), GgufValue::String("toy".to_string()))],
            &[crate::gguf_write::TensorOut { name: "w".to_string(), shape: vec![numel], ty: T_Q8_0, data: raw }],
            32,
        )
        .unwrap();

        let mg = MmapGguf::open(&path).unwrap();
        let whole = mg.tensor("w").unwrap().unwrap();
        assert_eq!(whole.len(), numel);

        let chunk = 4_096usize;
        let (streamed, peak_chunked) = peak_live(|| {
            let mut acc: Vec<f32> = Vec::with_capacity(numel);
            let found = mg.with_tensor_chunks("w", chunk, &mut |off, c| {
                assert_eq!(off as usize, acc.len(), "chunks must arrive in order, contiguously");
                acc.extend_from_slice(c);
            });
            assert!(found);
            acc
        });
        // Bits, not a tolerance: the chunked path must decode exactly what the
        // whole-tensor path does, or it is a different reader.
        assert_eq!(streamed, whole, "chunked dequant must be bit-identical to the whole-tensor dequant");

        // The accumulator above is itself a whole tensor, so measure the
        // bound on a run that retains nothing.
        let ((), peak_scan) = peak_live(|| {
            let mut sum = 0.0f64;
            assert!(mg.with_tensor_chunks("w", chunk, &mut |_, c| sum += c[0] as f64));
            assert!(sum.is_finite());
        });
        let tensor_bytes = numel * 4;
        assert!(
            peak_scan < tensor_bytes / 8,
            "chunked scan peak {peak_scan} is not << one tensor ({tensor_bytes}) - the whole tensor is being materialized"
        );
        assert!(peak_chunked >= tensor_bytes, "sanity: retaining every chunk necessarily holds a tensor");

        std::fs::remove_file(&path).ok();
    }

    fn put_str(v: &mut Vec<u8>, s: &str) {
        v.extend((s.len() as u64).to_le_bytes());
        v.extend(s.as_bytes());
    }
    fn h(x: f32) -> [u8; 2] {
        half::f16::from_f32(x).to_le_bytes()
    }

    // ---- dequant: one hand-built block per type, expected values derived
    //      directly from the GGML format definition (not via an encoder). ----

    /// Dequantize a single block of `ty` given exactly its `block_bytes`.
    fn one_block(ty: u32, elems: usize, blk: &[u8]) -> Vec<f32> {
        dequantize(ty, blk, elems).unwrap()
    }

    #[test]
    fn q4_0_block() {
        // d = 2.0; qs[0] = 0x3A -> low nibble 0xA=10 -> (10-8)*2 = 4 at y[0];
        // high nibble 0x3=3 -> (3-8)*2 = -10 at y[16]; other qs = 0x88 -> 0.
        let mut b = Vec::new();
        b.extend(h(2.0));
        b.push(0x3A);
        b.extend([0x88u8; 15]);
        let y = one_block(T_Q4_0, 32, &b);
        assert_eq!(y[0], 4.0);
        assert_eq!(y[16], -10.0);
        assert_eq!(y[1], 0.0);
    }

    #[test]
    fn q4_1_block() {
        // d=2.0, m=1.0; qs[0]=0x3A -> low 10 -> 10*2+1=21 (y0); high 3 -> 7 (y16);
        // rest qs=0x00 -> 0*2+1 = 1.
        let mut b = Vec::new();
        b.extend(h(2.0));
        b.extend(h(1.0));
        b.push(0x3A);
        b.extend([0u8; 15]);
        let y = one_block(T_Q4_1, 32, &b);
        assert_eq!(y[0], 21.0);
        assert_eq!(y[16], 7.0);
        assert_eq!(y[1], 1.0);
    }

    #[test]
    fn q5_0_block() {
        // d=2.0; qh=1 (bit0 set); qs[0]=0x0A.
        // y0: xh0=16 -> (10|16)-16=10 -> 20. y16: xh1=0 -> (0)-16=-16 -> -32.
        // y1: qs[1]=0, xh0=0 -> -16 -> -32.
        let mut b = Vec::new();
        b.extend(h(2.0));
        b.extend(1u32.to_le_bytes());
        b.push(0x0A);
        b.extend([0u8; 15]);
        let y = one_block(T_Q5_0, 32, &b);
        assert_eq!(y[0], 20.0);
        assert_eq!(y[16], -32.0);
        assert_eq!(y[1], -32.0);
    }

    #[test]
    fn q5_1_block() {
        // d=2.0, m=5.0; qh=1; qs[0]=0x0A.
        // y0: (10|16)*2 + 5 = 57. y16: (0)*2+5 = 5. y1: 0*2+5 = 5.
        let mut b = Vec::new();
        b.extend(h(2.0));
        b.extend(h(5.0));
        b.extend(1u32.to_le_bytes());
        b.push(0x0A);
        b.extend([0u8; 15]);
        let y = one_block(T_Q5_1, 32, &b);
        assert_eq!(y[0], 57.0);
        assert_eq!(y[16], 5.0);
        assert_eq!(y[1], 5.0);
    }

    #[test]
    fn q8_0_block() {
        // d=0.5; qs = [10, -4, 127, 0...] -> 5.0, -2.0, 63.5, 0.
        let mut b = Vec::new();
        b.extend(h(0.5));
        b.push(10u8);
        b.push((-4i8) as u8);
        b.push(127u8);
        b.extend([0u8; 29]);
        let y = one_block(T_Q8_0, 32, &b);
        assert_eq!(y[0], 5.0);
        assert_eq!(y[1], -2.0);
        assert_eq!(y[2], 63.5);
        assert_eq!(y[3], 0.0);
    }

    #[test]
    fn q2_k_block() {
        // scales[16], qs[64], d(f16), dmin(f16). dmin=0 so no min offset; d=1.
        // scales[0]=3 -> dl = 1*(3&0xF)=3 for the first group of 16; other
        // scales 0 -> dl 0. qs[0..3] = 1,2,3 (2-bit low, shift 0).
        let mut scales = [0u8; 16];
        scales[0] = 3;
        let mut qs = [0u8; 64];
        qs[0] = 1;
        qs[1] = 2;
        qs[2] = 3;
        let mut b = Vec::new();
        b.extend(scales);
        b.extend(qs);
        b.extend(h(1.0));
        b.extend(h(0.0));
        let y = one_block(T_Q2_K, 256, &b);
        assert_eq!(y[0], 3.0);
        assert_eq!(y[1], 6.0);
        assert_eq!(y[2], 9.0);
        assert_eq!(y[3], 0.0);
        assert_eq!(y[255], 0.0);
    }

    #[test]
    fn q3_k_block() {
        // hmask[32]=0xFF (all high bits set -> subtract 0); qs[64]; scales[12]=0
        // (all decoded scales 0 -> dl = d*(0-32) = -32); d=1.0.
        // qs[0]=1 -> y0 = -32*1 = -32; qs[1]=2 -> y1 = -32*2 = -64; rest 0.
        let hmask = [0xFFu8; 32];
        let mut qs = [0u8; 64];
        qs[0] = 1;
        qs[1] = 2;
        let scales = [0u8; 12];
        let mut b = Vec::new();
        b.extend(hmask);
        b.extend(qs);
        b.extend(scales);
        b.extend(h(1.0));
        let y = one_block(T_Q3_K, 256, &b);
        assert_eq!(y[0], -32.0);
        assert_eq!(y[1], -64.0);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[255], 0.0);
    }

    #[test]
    fn q4_k_block() {
        // d(f16),dmin(f16),scales[12],qs[128]. dmin=0; d=1.0; scales[0]=2 ->
        // first sub d1 = 1*2 = 2; all other subs 0. qs[0]=1 -> y0=2*1=2;
        // qs[1]=2 -> y1=2*2=4; rest 0.
        let mut scales = [0u8; 12];
        scales[0] = 2;
        let mut qs = [0u8; 128];
        qs[0] = 1;
        qs[1] = 2;
        let mut b = Vec::new();
        b.extend(h(1.0));
        b.extend(h(0.0));
        b.extend(scales);
        b.extend(qs);
        let y = one_block(T_Q4_K, 256, &b);
        assert_eq!(y[0], 2.0);
        assert_eq!(y[1], 4.0);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[255], 0.0);
    }

    #[test]
    fn q5_k_block() {
        // d,dmin,scales[12],qh[32],qs[128]. dmin=0; d=1.0; scales[0]=2 -> d1=2.
        // qs[0]=1, qh[0] bit0 set (u1=1) -> +16 -> y0 = 2*(1+16) = 34.
        // qs[1]=2, qh[1]=0 -> y1 = 2*2 = 4; rest 0.
        let mut scales = [0u8; 12];
        scales[0] = 2;
        let mut qh = [0u8; 32];
        qh[0] = 1;
        let mut qs = [0u8; 128];
        qs[0] = 1;
        qs[1] = 2;
        let mut b = Vec::new();
        b.extend(h(1.0));
        b.extend(h(0.0));
        b.extend(scales);
        b.extend(qh);
        b.extend(qs);
        let y = one_block(T_Q5_K, 256, &b);
        assert_eq!(y[0], 34.0);
        assert_eq!(y[1], 4.0);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[255], 0.0);
    }

    #[test]
    fn q6_k_block() {
        // ql[128],qh[64],scales[16] i8,d(f16). d=1.0; scales[0]=1, rest 0.
        // Quant q = ((ql&0xF)|((qh&3)<<4)) - 32. y[0..16] use sc[0]=1.
        // ql[0]=5, qh[0]=1 -> q1 = (5|(1<<4))-32 = 21-32 = -11 -> y0 = -11.
        // ql[1..]=0, qh[1..]=0 -> q1 = -32 -> y[1..16] = -32. rest 0.
        let ql = {
            let mut a = [0u8; 128];
            a[0] = 5;
            a
        };
        let qh = {
            let mut a = [0u8; 64];
            a[0] = 1;
            a
        };
        let mut scales = [0u8; 16];
        scales[0] = 1;
        let mut b = Vec::new();
        b.extend(ql);
        b.extend(qh);
        b.extend(scales);
        b.extend(h(1.0));
        let y = one_block(T_Q6_K, 256, &b);
        assert_eq!(y[0], -11.0);
        assert_eq!(y[1], -32.0);
        assert_eq!(y[15], -32.0);
        assert_eq!(y[16], 0.0);
        assert_eq!(y[64], 0.0);
        assert_eq!(y[255], 0.0);
    }

    #[test]
    fn q8_k_block() {
        // d = f32 0.5; qs[0]=4 -> 2.0, qs[1]=-2 -> -1.0, qs[255]=10 -> 5.0.
        let mut b = Vec::new();
        b.extend(0.5f32.to_le_bytes());
        let mut qs = [0u8; 256];
        qs[0] = 4;
        qs[1] = (-2i8) as u8;
        qs[255] = 10;
        b.extend(qs);
        b.extend([0u8; 32]); // bsums (ignored)
        let y = one_block(T_Q8_K, 256, &b);
        assert_eq!(y[0], 2.0);
        assert_eq!(y[1], -1.0);
        assert_eq!(y[255], 5.0);
    }

    #[test]
    fn iq_types_error_clearly() {
        // id 20 (IQ4_NL) moved OUT of "not yet implemented" in M18 - use a
        // still-genuinely-unimplemented IQ grid codebook id instead.
        let err = dequantize(16, &[0u8; 64], 32).unwrap_err();
        assert!(err.contains("not yet implemented"), "{err}");
    }

    // ---- dequant across MANY blocks -------------------------------------
    //
    // Every hand-built test above feeds exactly ONE block, which is
    // structurally blind to how blocks are laid into the output: a
    // reversed, overlapping, or off-by-one-block write is invisible at a
    // single block (lesson #4 - a degenerate fixture hides the bug class
    // the test exists for). These pin the block-to-output mapping across a
    // real multi-block run, which is what makes decoding blocks in parallel
    // a scheduling change rather than a correctness risk.

    /// Deterministic pseudo-random bytes - a decoder must be exercised over
    /// varied nibble/sign patterns, not a field of zeros that would agree
    /// under almost any indexing mistake.
    fn noise_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    /// The serial contract, stated independently of the implementation:
    /// block `i` of `raw` decodes to output elements
    /// `[i*elems_per_block, (i+1)*elems_per_block)`, and the result is
    /// truncated to `numel`. Built by decoding each block on its own, so it
    /// shares no traversal code with `dequantize`'s own block walk.
    fn oracle_by_block(ty: u32, raw: &[u8], block_bytes: usize, elems_per_block: usize, numel: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for blk in raw.chunks_exact(block_bytes) {
            out.extend(dequantize(ty, blk, elems_per_block).unwrap());
        }
        out.truncate(numel);
        out
    }

    /// Compare by BIT PATTERN, not by `==`. The fixture is random bytes, so a
    /// block's f16 scale can legitimately decode to NaN - and `NaN != NaN`
    /// would fail a value comparison between two byte-identical results. Bits
    /// are also the stronger claim: this is asserting "the same bytes", which
    /// is exactly the guarantee a scheduling change has to keep.
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn multi_block_dequant_lays_each_block_into_its_own_output_span() {
        // One representative of every block family: the legacy 32-element
        // types and the 256-element k-quants, since the two use different
        // block geometry and different decoders.
        for &ty in &[T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0, T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_Q8_K] {
            let (be, bb) = block_geometry(ty).expect("every type here has block geometry");
            // Enough blocks that a parallel split has something to split.
            // Block counts that straddle the decode's internal grouping, not
            // just fill one group: a count at the group size, one past it, and
            // one that leaves a short tail after several full groups. A list
            // that never exceeds one group would leave the group-index
            // arithmetic itself untested (lesson #4 again, one level down).
            for nblocks in [1usize, 2, 7, 64, 65, 130, 193] {
                let raw = noise_bytes(nblocks * bb, 0x5EED_0000 + ty as u64 * 7 + nblocks as u64);
                let numel = nblocks * be;
                let got = dequantize(ty, &raw, numel).unwrap();
                let want = oracle_by_block(ty, &raw, bb, be, numel);
                assert_eq!(got.len(), want.len(), "type {ty}, {nblocks} blocks: length");
                assert_eq!(bits(&got), bits(&want), "type {ty}, {nblocks} blocks: block-to-output mapping");
            }
        }
    }

    /// `numel` that is not a whole number of blocks: the last block is
    /// decoded in full and the result truncated. A parallel decode that
    /// sized its output from `numel` instead of the block count would drop
    /// or corrupt exactly this tail.
    #[test]
    fn a_partial_trailing_block_is_decoded_then_truncated() {
        for &ty in &[T_Q8_0, T_Q4_K] {
            let (be, bb) = block_geometry(ty).unwrap();
            let nblocks = 5usize;
            let raw = noise_bytes(nblocks * bb, 0xA5A5 + ty as u64);
            let numel = nblocks * be - be / 2; // half of the last block is real
            let got = dequantize(ty, &raw, numel).unwrap();
            assert_eq!(got.len(), numel, "type {ty}: truncated to numel");
            assert_eq!(bits(&got), bits(&oracle_by_block(ty, &raw, bb, be, numel)), "type {ty}: truncated tail must match");
        }
    }

    // ---- KV parsing + ModelCard mapping via an in-memory GGUF ----

    /// Emit a header with the given KV bytes and one F32 tensor `[n]`.
    fn build_gguf(kv: &[u8], n_kv: u64, vals: &[f32]) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend(1u64.to_le_bytes()); // 1 tensor
        h.extend(n_kv.to_le_bytes());
        h.extend(kv);
        // tensor info: name "w", 1 dim [n], type F32, offset 0
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes());
        h.extend((vals.len() as u64).to_le_bytes());
        h.extend(T_F32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        for v in vals {
            h.extend(v.to_le_bytes());
        }
        h
    }

    #[test]
    fn parse_kv_and_model_card() {
        let mut kv = Vec::new();
        // general.architecture = "llama" (string)
        put_str(&mut kv, "general.architecture");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "llama");
        // general.name = "my-model" (string)
        put_str(&mut kv, "general.name");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "my-model");
        // general.file_type = 15 (uint32) -> Q4_K_M
        put_str(&mut kv, "general.file_type");
        kv.extend(4u32.to_le_bytes());
        kv.extend(15u32.to_le_bytes());
        // llama.context_length = 4096 (uint64)
        put_str(&mut kv, "llama.context_length");
        kv.extend(10u32.to_le_bytes());
        kv.extend(4096u64.to_le_bytes());
        // tokenizer.ggml.model = "gpt2" (string)
        put_str(&mut kv, "tokenizer.ggml.model");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "gpt2");
        // an array of int32 to exercise array parsing
        put_str(&mut kv, "test.arr");
        kv.extend(9u32.to_le_bytes()); // array
        kv.extend(5u32.to_le_bytes()); // element type int32
        kv.extend(3u64.to_le_bytes()); // count
        kv.extend(1i32.to_le_bytes());
        kv.extend((-2i32).to_le_bytes());
        kv.extend(3i32.to_le_bytes());

        let bytes = build_gguf(&kv, 6, &[1.0, 2.0, 3.0, 4.0]);
        let m = parse_gguf(&bytes).unwrap();

        assert_eq!(m.tensors["w"], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.shapes["w"], vec![4]);
        assert_eq!(m.metadata_strings["general.architecture"], "llama");
        assert_eq!(m.kv["general.file_type"].as_u64(), Some(15));
        assert_eq!(
            m.kv["test.arr"],
            GgufValue::Array(vec![GgufValue::I32(1), GgufValue::I32(-2), GgufValue::I32(3)])
        );

        let card = m.model_card();
        assert_eq!(card.id, "my-model");
        assert_eq!(card.display_name.as_deref(), Some("my-model"));
        assert_eq!(card.family, "llama");
        assert_eq!(card.architecture.as_deref(), Some("llama"));
        assert_eq!(card.context_length, Some(4096));
        assert_eq!(card.tokenizer.as_deref(), Some("gpt2"));
        assert_eq!(card.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(card.param_count, Some(4));
    }

    #[test]
    fn tokenizer_accessor_extracts_tokens_merges_specials() {
        // Emit a header carrying a gpt2 tokenizer: model, tokens[], merges[],
        // token_type[], and bos/eos ids — then assert the typed accessor.
        fn put_str_arr(v: &mut Vec<u8>, key: &str, items: &[&str]) {
            put_str(v, key);
            v.extend(9u32.to_le_bytes()); // array
            v.extend(8u32.to_le_bytes()); // element type: string
            v.extend((items.len() as u64).to_le_bytes());
            for s in items {
                put_str(v, s);
            }
        }
        fn put_i32_arr(v: &mut Vec<u8>, key: &str, items: &[i32]) {
            put_str(v, key);
            v.extend(9u32.to_le_bytes()); // array
            v.extend(5u32.to_le_bytes()); // element type: int32
            v.extend((items.len() as u64).to_le_bytes());
            for x in items {
                v.extend(x.to_le_bytes());
            }
        }
        fn put_u32(v: &mut Vec<u8>, key: &str, x: u32) {
            put_str(v, key);
            v.extend(4u32.to_le_bytes()); // value type: uint32
            v.extend(x.to_le_bytes());
        }

        let mut kv = Vec::new();
        put_str(&mut kv, "tokenizer.ggml.model");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "gpt2");
        put_str(&mut kv, "tokenizer.ggml.pre");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "qwen2");
        put_str_arr(&mut kv, "tokenizer.ggml.tokens", &["<|endoftext|>", "h", "i", "hi"]);
        put_str_arr(&mut kv, "tokenizer.ggml.merges", &["h i"]);
        put_i32_arr(&mut kv, "tokenizer.ggml.token_type", &[3, 1, 1, 1]);
        put_u32(&mut kv, "tokenizer.ggml.bos_token_id", 0);
        put_u32(&mut kv, "tokenizer.ggml.eos_token_id", 0);

        let bytes = build_gguf(&kv, 7, &[1.0, 2.0]);
        let m = parse_gguf(&bytes).unwrap();
        let t = m.tokenizer().expect("tokenizer present");
        assert_eq!(t.model, "gpt2");
        assert_eq!(t.pre.as_deref(), Some("qwen2"));
        assert_eq!(t.tokens, vec!["<|endoftext|>", "h", "i", "hi"]);
        assert_eq!(t.merges, vec!["h i"]);
        assert_eq!(t.token_types, vec![3, 1, 1, 1]);
        assert_eq!(t.bos, Some(0));
        assert_eq!(t.eos, Some(0));
        assert_eq!(t.unk, None);

        // A file without tokenizer.ggml.model exposes no tokenizer.
        let plain = parse_gguf(&build_gguf(&[], 0, &[1.0])).unwrap();
        assert!(plain.tokenizer().is_none());
    }

    #[test]
    fn model_card_missing_keys_are_none() {
        let bytes = build_gguf(&[], 0, &[1.0, 2.0]);
        let card = parse_gguf(&bytes).unwrap().model_card();
        assert_eq!(card.family, "gguf");
        assert!(card.architecture.is_none());
        assert!(card.context_length.is_none());
        assert!(card.license.is_none());
        assert!(card.tokenizer.is_none());
        assert_eq!(card.param_count, Some(2));
    }

    /// Build a tiny two-tensor GGUF (one BF16, one F32) in memory and read it
    /// via the native `read` path (preserved for the FLUX.2 importer).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn synthetic_roundtrip() {
        use std::io::Write;
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend(2u64.to_le_bytes()); // tensors
        h.extend(1u64.to_le_bytes()); // kv
        put_str(&mut h, "general.architecture");
        h.extend(8u32.to_le_bytes());
        put_str(&mut h, "flux");
        // tensor 0: bf16 [2,3] (torch) -> gguf ne [3,2]
        put_str(&mut h, "a.weight");
        h.extend(2u32.to_le_bytes());
        h.extend(3u64.to_le_bytes());
        h.extend(2u64.to_le_bytes());
        h.extend(30u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        // tensor 1: f32 [4]
        put_str(&mut h, "b.scale");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(32u64.to_le_bytes()); // offset within data (aligned)
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        let vals = [1.0f32, -2.0, 0.5, 3.0, -0.25, 8.0];
        for v in vals {
            let bits = (v.to_bits() >> 16) as u16; // exact in bf16
            h.extend(bits.to_le_bytes());
        }
        h.resize(data_start + 32, 0);
        for v in [9.0f32, -1.5, 0.0, 2.5] {
            h.extend(v.to_le_bytes());
        }

        let dir = std::env::temp_dir().join(format!("gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        std::fs::File::create(&path).unwrap().write_all(&h).unwrap();

        let ts = read(path.to_str().unwrap()).unwrap();
        assert_eq!(ts.len(), 2);
        let a = ts.iter().find(|t| t.name == "a.weight").unwrap();
        let b = ts.iter().find(|t| t.name == "b.scale").unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, vals);
        assert_eq!(b.shape, vec![4]);
        assert_eq!(b.data, [9.0, -1.5, 0.0, 2.5]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `MmapGguf`'s `TensorSource` impl must stream the same values `tensor()`
    /// returns directly, and report absent names as `false` (not panic) so a
    /// two-way-coverage import check gets a real "missing" signal.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn mmap_gguf_tensor_source_streams_present_tensors_and_reports_absent_ones() {
        use crate::gguf_write::{write, TensorOut};
        use crate::TensorSource;

        let path = std::env::temp_dir()
            .join(format!("gguf-tensorsource-test-{}.gguf", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let data: Vec<u8> = (0..6).flat_map(|i| (i as f32 * 1.5 - 2.0).to_le_bytes()).collect();
        let tensors = vec![TensorOut { name: "w".to_string(), shape: vec![2, 3], ty: T_F32, data }];
        write(&path, &[], &tensors, 32).unwrap();

        let mg = MmapGguf::open(&path).unwrap();

        let mut seen = None;
        let found = mg.with_tensor("w", &mut |d| seen = Some(d.to_vec()));
        assert!(found, "with_tensor must find a present tensor");
        assert_eq!(seen, Some(vec![-2.0, -0.5, 1.0, 2.5, 4.0, 5.5]));

        let mut never_called = true;
        let found_missing = mg.with_tensor("does_not_exist", &mut |_| never_called = false);
        assert!(!found_missing, "with_tensor must report an absent name as false, not panic");
        assert!(never_called, "the closure must never run for an absent name");

        std::fs::remove_file(&path).ok();
    }

    /// A range read must be bit-identical to the corresponding slice of a
    /// whole-tensor decode, on a QUANTIZED tensor (where the range does not
    /// line up with the on-disk blocks) as well as a plain one, and must
    /// refuse an out-of-range request rather than truncating it.
    ///
    /// This is the accessor an embedding-row gather uses on a `[vocab,
    /// d_model]` table that cannot be decoded whole, so "reads only the
    /// blocks it needs" and "reads exactly the right values" are the same
    /// requirement.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn tensor_range_matches_the_whole_tensor_decode_including_mid_block_starts() {
        use crate::gguf_write::{write, TensorOut};

        let path = std::env::temp_dir()
            .join(format!("gguf-range-test-{}.gguf", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let f32_vals: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
        let f32_data: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Two Q8_0 blocks (32 elements each), so a range can straddle a block
        // boundary and start mid-block - the case a naive block-aligned read
        // gets wrong.
        let mut q_data = Vec::new();
        for blk in 0..2u8 {
            q_data.extend(half::f16::from_f32(0.5).to_le_bytes());
            q_data.extend((0..32u8).map(|i| i.wrapping_add(blk * 7)));
        }
        let tensors = vec![
            TensorOut { name: "f.weight".to_string(), shape: vec![3, 4], ty: T_F32, data: f32_data },
            TensorOut { name: "q.weight".to_string(), shape: vec![64], ty: T_Q8_0, data: q_data },
        ];
        write(&path, &[], &tensors, 32).unwrap();
        let mg = MmapGguf::open(&path).unwrap();

        for name in ["f.weight", "q.weight"] {
            let whole = mg.tensor(name).unwrap().unwrap();
            for (start, len) in [(0usize, 1usize), (1, 3), (5, 4), (whole.len() - 1, 1), (0, whole.len())] {
                let got = mg.tensor_range(name, start, len).unwrap_or_else(|| panic!("{name}[{start}..+{len}] must be readable")).unwrap();
                assert_eq!(got, whole[start..start + len], "{name}[{start}..+{len}] differs from the whole-tensor decode");
            }
            // Out of range is `None`, never a short read.
            assert!(mg.tensor_range(name, whole.len(), 1).is_none());
            assert!(mg.tensor_range(name, 0, whole.len() + 1).is_none());
            assert!(mg.tensor_range(name, usize::MAX, 1).is_none());
        }
        assert!(mg.tensor_range("nope", 0, 1).is_none());

        std::fs::remove_file(&path).ok();
    }

    /// `raw_words`/`numel` must not report a present tensor as missing (the D1
    /// bug: both used to default to `None` for every `MmapGguf` tensor, which
    /// made `raw_words(name).or_else(|| numel(name))` panic "missing" on a
    /// tensor that was actually there). `raw_words` fires zero-copy for the
    /// F32 tensor, declines (but still reports a `numel`) for the Q4_0 one.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn raw_words_and_numel_see_every_present_tensor() {
        use crate::gguf_write::{write, TensorOut};
        use crate::TensorSource;

        let path = std::env::temp_dir()
            .join(format!("gguf-rawwords-test-{}.gguf", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let f32_vals = [-2.0f32, -0.5, 1.0, 2.5, 4.0, 5.5];
        let f32_data: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // One Q4_0 block: d=2.0, qs[0]=0x3A, rest 0x88 (same fixture as q4_0_block).
        let mut q_data = Vec::new();
        q_data.extend(half::f16::from_f32(2.0).to_le_bytes());
        q_data.push(0x3A);
        q_data.extend([0x88u8; 15]);
        let tensors = vec![
            TensorOut { name: "f.weight".to_string(), shape: vec![2, 3], ty: T_F32, data: f32_data },
            TensorOut { name: "q.weight".to_string(), shape: vec![32], ty: T_Q4_0, data: q_data },
        ];
        write(&path, &[], &tensors, 32).unwrap();

        let mg = MmapGguf::open(&path).unwrap();

        // numel is known for BOTH tensors without decoding.
        assert_eq!(mg.numel("f.weight"), Some(6));
        assert_eq!(mg.numel("q.weight"), Some(32));
        assert_eq!(mg.numel("does_not_exist"), None);

        // raw_words zero-copies the F32 tensor and matches with_tensor's bits.
        let words = mg.raw_words("f.weight").expect("F32 tensor must be zero-copyable");
        let via_words: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
        assert_eq!(via_words, f32_vals);

        // raw_words declines the quantized tensor (nothing to bind as-is)
        // but with_tensor still dequantizes it correctly, and neither call
        // panics claiming the tensor is missing.
        assert!(mg.raw_words("q.weight").is_none());
        let mut seen = None;
        assert!(mg.with_tensor("q.weight", &mut |d| seen = Some(d.to_vec())));
        let seen = seen.unwrap();
        assert_eq!(seen.len(), 32);
        assert_eq!(seen[0], 4.0);
        assert_eq!(seen[16], -10.0);

        // The exact D1 caller pattern (crate::wan's upload_named) must now
        // resolve both tensors instead of panicking "missing".
        for name in ["f.weight", "q.weight"] {
            let resolved = mg.raw_words(name).map(|w| w.len()).or_else(|| mg.numel(name));
            assert!(resolved.is_some(), "{name} must resolve via raw_words or numel");
        }

        std::fs::remove_file(&path).ok();
    }

    /// [`MmapGguf::raw_tensor_bytes`] must return the EXACT on-disk block
    /// bytes (independently re-dequantizable) plus the right type id, for
    /// both an F32 and a Q8_0 tensor, and `None` for an absent name - the
    /// same three cases [`raw_words_and_numel_see_every_present_tensor`]
    /// exercises for the OTHER raw-access path.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn raw_tensor_bytes_returns_the_exact_on_disk_block_and_type() {
        use crate::gguf_write::{write, TensorOut};

        let path = std::env::temp_dir().join(format!("gguf-rawbytes-test-{}.gguf", std::process::id())).to_string_lossy().into_owned();
        let f32_vals = [1.0f32, -2.5, 3.0];
        let f32_data: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // One Q8_0 block (34 bytes: f16 scale + 32 int8), same fixture as `q8_0_block`.
        let mut q_data = Vec::new();
        q_data.extend(h(0.5));
        q_data.push(10u8);
        q_data.push((-4i8) as u8);
        q_data.push(127u8);
        q_data.extend([0u8; 29]);
        let tensors = vec![
            TensorOut { name: "f.weight".to_string(), shape: vec![3], ty: T_F32, data: f32_data.clone() },
            TensorOut { name: "q.weight".to_string(), shape: vec![32], ty: T_Q8_0, data: q_data.clone() },
        ];
        write(&path, &[], &tensors, 32).unwrap();

        let mg = MmapGguf::open(&path).unwrap();

        let (raw_f, ty_f) = mg.raw_tensor_bytes("f.weight").expect("f.weight must resolve");
        assert_eq!(ty_f, T_F32);
        assert_eq!(raw_f, f32_data.as_slice());

        let (raw_q, ty_q) = mg.raw_tensor_bytes("q.weight").expect("q.weight must resolve");
        assert_eq!(ty_q, T_Q8_0);
        assert_eq!(raw_q, q_data.as_slice());
        // The raw bytes must independently dequantize (by this file's own
        // `dequantize`) to the same values `tensor()` returns - proves
        // `raw_tensor_bytes` sliced the exact same block `tensor()` reads,
        // not an off-by-one range.
        assert_eq!(dequantize(T_Q8_0, raw_q, 32).unwrap(), mg.tensor("q.weight").unwrap().unwrap());

        assert!(mg.raw_tensor_bytes("does_not_exist").is_none());

        // `crate::weightio::WeightReader::nbytes` (built on `raw_tensor_bytes`
        // for the GGUF branch) reports the SAME exact sizes, and their sum
        // equals the tensor data actually on disk in this file - not a
        // fs::metadata(whole file) comparison, since the header/KV bytes
        // aren't tensor data, but the two tensor byte ranges must not overlap
        // or leave a gap, which their sum matching each raw slice's own
        // length independently confirms.
        let wr = crate::weightio::WeightReader::open(&path).unwrap();
        assert_eq!(wr.nbytes("f.weight"), Some(f32_data.len() as u64));
        assert_eq!(wr.nbytes("q.weight"), Some(q_data.len() as u64));
        assert_eq!(wr.nbytes("does_not_exist"), None);

        std::fs::remove_file(&path).ok();
    }

    /// Sanity-check the reader against a REAL quantized file.
    ///
    /// The fixture is `$BRAIN_GGUF_TESTFILE` if set, else the DeepSeek-OCR
    /// mmproj out of the model store. That fallback is the point: this used to
    /// name only the env var, which nothing in the repo ever sets and no fetch
    /// step ever produces, so `BRAIN_REQUIRE_FIXTURES=1` could not be satisfied
    /// on any machine - a fixture nobody can provide is a skip that never
    /// becomes a check. The mmproj is 427 MiB, is what several other suites
    /// already resolve, and `brain fetch ggml-org/DeepSeek-OCR-GGUF` produces
    /// it, so the demand is one a box can actually meet.
    ///
    /// Deliberately NOT "whatever `.gguf` is lying around": `load_gguf`
    /// dequantizes the whole file to fp32 in host memory, so picking an
    /// arbitrary one would let a 7 GB 14B checkpoint turn this smoke test into
    /// a 57 GB allocation.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn real_gguf_smoke() {
        const REPO: &str = "ggml-org/DeepSeek-OCR-GGUF";
        const MMPROJ: &str = "mmproj-DeepSeek-OCR-Q8_0.gguf";
        let from_store = || {
            let d = brain_testutil::model_dir(REPO)?;
            let p = std::path::Path::new(&d).join(MMPROJ);
            p.exists().then(|| p.to_string_lossy().into_owned())
        };
        let Some(path) = std::env::var("BRAIN_GGUF_TESTFILE").ok().filter(|p| !p.is_empty()).or_else(from_store)
        else {
            return brain_testutil::skip(&format!(
                "no real GGUF: set BRAIN_GGUF_TESTFILE, or `brain fetch {REPO}` so {MMPROJ} is in the store"
            ));
        };
        let m = load_gguf(&path).unwrap();
        assert!(!m.tensors.is_empty());
        assert!(m.param_count() > 0);
        // Every dequantized tensor must be finite.
        for (name, data) in &m.tensors {
            assert!(data.iter().all(|v| v.is_finite()), "non-finite in {name}");
        }
        let _ = m.model_card();
    }

    /// Every leaf read reports load progress against the file's tensor-byte
    /// total: one event per leaf call (a chunked read is one event, not one
    /// per chunk), cumulative across tensors, and a re-read adds its bytes
    /// again because the mapping was genuinely walked a second time. Same
    /// contract as the safetensors reader's test - a load-progress line is
    /// fed from either checkpoint format alike.
    #[test]
    fn leaf_reads_report_load_progress_to_the_observer() {
        use crate::TensorSource;

        let _serial = crate::load_progress::tests::observe_lock();
        let path = std::env::temp_dir().join(format!("brain-gguf-progress-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let mut data = Vec::new();
        for x in [1.0f32, -2.5, 3.25, 0.0, 7.0, -0.125] {
            data.extend_from_slice(&x.to_le_bytes());
        }
        crate::gguf_write::write(
            &path,
            &[],
            &[crate::gguf_write::TensorOut { name: "w".to_string(), shape: vec![2, 3], ty: T_F32, data }],
            32,
        )
        .unwrap();
        let got = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = got.clone();
        // The closure never panics and the observer is detached before the
        // assertions: a failed assert must not leave a stale observer behind
        // for the other tests' reads to trip over.
        crate::load_progress::observe(Box::new(move |e| {
            if let Ok(mut v) = sink.lock() {
                v.push((e.file.clone(), e.done, e.total));
            }
        }));
        let mg = MmapGguf::open(&path).unwrap();
        assert!(mg.with_tensor("w", &mut |_| {}));
        assert!(mg.raw_words("w").is_some());
        assert!(mg.with_tensor_chunks("w", 4, &mut |_, _| {}));
        let events = got.lock().unwrap().clone();
        crate::load_progress::clear();
        std::fs::remove_file(&path).ok();
        // The observer is process-global and the suite runs its tests
        // concurrently, so only this file's stream is asserted on.
        let mine: Vec<(u64, u64)> = events
            .iter()
            .filter(|(f, _, _)| f == &path)
            .map(|&(_, done, total)| (done, total))
            .collect();
        // "w" is F32, 6 elements = 24 bytes on disk, and every read is of the
        // whole tensor, so each one adds 24 to the cumulative counter.
        assert_eq!(mine, vec![(24, 24), (48, 24), (72, 24)]);
    }

    /// `GgmlType::from_id`/`id`/`name` must round-trip for every type this
    /// reader decodes, and must agree EXACTLY with the standalone
    /// `ggml_type_name`/`block_geometry` wrappers - those are now thin
    /// wrappers over this enum, so a disagreement here would mean the "one
    /// vocabulary" claim in `GgmlType`'s own doc is false.
    #[test]
    fn ggml_type_round_trips_and_agrees_with_the_wrapper_fns() {
        let all = [
            (T_F32, GgmlType::F32),
            (T_F16, GgmlType::F16),
            (T_BF16, GgmlType::BF16),
            (T_F64, GgmlType::F64),
            (T_I8, GgmlType::I8),
            (T_I16, GgmlType::I16),
            (T_I32, GgmlType::I32),
            (T_I64, GgmlType::I64),
            (T_Q4_0, GgmlType::Q4_0),
            (T_Q4_1, GgmlType::Q4_1),
            (T_Q5_0, GgmlType::Q5_0),
            (T_Q5_1, GgmlType::Q5_1),
            (T_Q8_0, GgmlType::Q8_0),
            (T_Q8_1, GgmlType::Q8_1),
            (T_Q2_K, GgmlType::Q2K),
            (T_Q3_K, GgmlType::Q3K),
            (T_Q4_K, GgmlType::Q4K),
            (T_Q5_K, GgmlType::Q5K),
            (T_Q6_K, GgmlType::Q6K),
            (T_Q8_K, GgmlType::Q8K),
            (T_MXFP4, GgmlType::MXFP4),
            (T_IQ4_NL, GgmlType::IQ4NL),
            (T_IQ4_XS, GgmlType::IQ4XS),
            (T_TQ1_0, GgmlType::TQ1_0),
            (T_TQ2_0, GgmlType::TQ2_0),
        ];
        for (id, ty) in all {
            assert_eq!(GgmlType::from_id(id), Some(ty), "id {id}");
            assert_eq!(ty.id(), id, "{ty:?} round trip");
            assert_eq!(Some(ty.name()), ggml_type_name(id), "{ty:?} vs ggml_type_name");
            assert_eq!(Some((ty.block_elems(), ty.block_bytes())), block_geometry(id), "{ty:?} vs block_geometry");
        }
        // Unknown ids (a still-unimplemented IQ grid codebook, a genuinely
        // invalid one) must decline, never guess. 9 and 24-28 moved out in
        // M17; 20, 23, 34, 35, 39 moved out in M18 - all real, recognized
        // types now.
        for unknown in [16u32, 29, 999] {
            assert_eq!(GgmlType::from_id(unknown), None, "id {unknown} must be unrecognized");
            assert_eq!(block_geometry(unknown), None);
            assert_eq!(ggml_type_name(unknown), None);
        }
    }

    /// `block_expand` (Q4_K, whose sub-block scales are ALSO exercised here
    /// since a naive generalization could silently drop the affine `dmin`
    /// term) must be bit-identical to the whole-tensor `dequantize` for the
    /// same range - the property the old `q8_0_expand`-only test suite never
    /// checked for any type but Q8_0.
    #[test]
    fn block_expand_matches_whole_tensor_dequantize_for_q4_k() {
        // Two super-blocks (512 elements), built with a real per-sub-block
        // scale AND min spread so a hoisted scale/min would be visible -
        // reuses the same encoder path `quantize.rs`'s round-trip test gates.
        let vals: Vec<f32> = (0..512).map(|i| ((i as i64 * 7 - 250) % 97) as f32 * 0.25).collect();
        let raw = crate::quant::quantize_par(T_Q4_K, &vals).unwrap();
        let whole = dequantize(T_Q4_K, &raw, 512).unwrap();
        assert_eq!(whole.len(), 512);

        // A sub-range starting mid-tensor (not block 0 - lesson: a relayout
        // that only gets block 0 right is exactly the bug this must catch).
        let mut got = Vec::new();
        block_expand(GgmlType::Q4K, &raw, 256, 512, &mut got).unwrap();
        assert_eq!(got, whole[256..512], "block_expand must match the whole-tensor decode bit-for-bit");

        // Misaligned/inverted/F32 refusals, never a silently-wrong geometry.
        let mut scratch = Vec::new();
        assert!(block_expand(GgmlType::Q4K, &raw, 0, 100, &mut scratch).is_err(), "100 is not a multiple of 256");
        assert!(block_expand(GgmlType::Q4K, &raw, 256, 0, &mut scratch).is_err(), "inverted range");
        assert!(block_expand(GgmlType::F32, &[0u8; 4], 0, 1, &mut scratch).is_err(), "F32 has no per-block decoder");

        // `q8_0_expand` is `block_expand` pinned to Q8_0 - prove it stayed a
        // real wrapper, not a copy that can drift.
        let q8_raw = crate::quant::quantize_par(T_Q8_0, &vals[..64]).unwrap();
        let q8_whole = dequantize(T_Q8_0, &q8_raw, 64).unwrap();
        let mut q8_got = Vec::new();
        q8_0_expand(&q8_raw, 0, 64, &mut q8_got).unwrap();
        assert_eq!(q8_got, q8_whole);
    }

    /// `quant_label`'s fallback bug: a `general.file_type` id this reader's
    /// table does not recognize must fall through to
    /// `general.quantization_version`, not report a fabricated "unknown"
    /// string that shadows the fallback.
    #[test]
    fn quant_label_falls_back_past_an_unrecognized_file_type() {
        // Recognized: reports the name, ignores quantization_version.
        let mut kv = BTreeMap::new();
        kv.insert("general.file_type".to_string(), GgufValue::U32(7)); // Q8_0
        kv.insert("general.quantization_version".to_string(), GgufValue::U32(2));
        assert_eq!(quant_label(&kv), Some("Q8_0".to_string()));

        // Unrecognized file_type id (e.g. a future MXFP4 ftype): falls back,
        // rather than the old behavior of returning "unknown" outright.
        let mut kv2 = BTreeMap::new();
        kv2.insert("general.file_type".to_string(), GgufValue::U32(9999));
        kv2.insert("general.quantization_version".to_string(), GgufValue::U32(2));
        assert_eq!(quant_label(&kv2), Some("qver2".to_string()));

        // Neither key present: None, not a fabricated label.
        assert_eq!(quant_label(&BTreeMap::new()), None);

        // file_type absent, quantization_version present: the pre-existing
        // fallback path, unchanged.
        let mut kv3 = BTreeMap::new();
        kv3.insert("general.quantization_version".to_string(), GgufValue::U32(3));
        assert_eq!(quant_label(&kv3), Some("qver3".to_string()));
    }

    /// M17: ggml ids 9 (Q8_1) and 24-28 (I8/I16/I32/I64/F64) used to fail
    /// `MmapGguf::open` OUTRIGHT (`tensor_nbytes` returned `None` for the
    /// whole file, not just that one tensor). `GgmlType::from_id` now
    /// recognizes all six, so a file carrying one opens and that tensor
    /// dequantizes - checked here with an oracle value chosen to exercise
    /// each type's full range, not just small/positive numbers.
    #[test]
    fn scalar_array_types_open_and_dequantize_to_their_exact_value() {
        fn open_one(ty: u32, data: Vec<u8>, numel: usize) -> Vec<f32> {
            let path = std::env::temp_dir().join(format!("brain-gguf-scalar-{ty}-{}.gguf", std::process::id()));
            let path = path.to_str().unwrap().to_string();
            crate::gguf_write::write(&path, &[], &[crate::gguf_write::TensorOut { name: "w".to_string(), shape: vec![numel], ty, data }], 32).unwrap();
            let mg = MmapGguf::open(&path).unwrap();
            let got = mg.tensor("w").unwrap().unwrap();
            std::fs::remove_file(&path).ok();
            got
        }

        // I8: full signed range, not just small values.
        let i8_vals: [i8; 4] = [-128, -1, 0, 127];
        let got = open_one(T_I8, i8_vals.iter().map(|&v| v as u8).collect(), 4);
        assert_eq!(got, vec![-128.0, -1.0, 0.0, 127.0]);

        // I16.
        let i16_vals: [i16; 3] = [-32768, 0, 32767];
        let got = open_one(T_I16, i16_vals.iter().flat_map(|v| v.to_le_bytes()).collect(), 3);
        assert_eq!(got, vec![-32768.0, 0.0, 32767.0]);

        // I32.
        let i32_vals: [i32; 3] = [i32::MIN, 0, i32::MAX];
        let got = open_one(T_I32, i32_vals.iter().flat_map(|v| v.to_le_bytes()).collect(), 3);
        assert_eq!(got, vec![i32::MIN as f32, 0.0, i32::MAX as f32]);

        // I64: a value OUTSIDE f32's exact-integer range, so the lossy `as
        // f32` narrowing this format accepts is exercised, not just an
        // in-range value that would pass even with a truncating bug.
        let i64_vals: [i64; 2] = [0, 1_000_000_000_000_000_000];
        let got = open_one(T_I64, i64_vals.iter().flat_map(|v| v.to_le_bytes()).collect(), 2);
        assert_eq!(got, vec![0.0, 1_000_000_000_000_000_000i64 as f32]);

        // F64: same lossy-narrowing point, with a fraction.
        let f64_vals: [f64; 2] = [0.1, -123456.789];
        let got = open_one(T_F64, f64_vals.iter().flat_map(|v| v.to_le_bytes()).collect(), 2);
        assert_eq!(got, vec![0.1f64 as f32, -123456.789f64 as f32]);
    }

    /// Q8_1's dequant is IDENTICAL to Q8_0's (`w = d*q[i]`) - the extra `s`
    /// field is a cached `d*sum(qs)` ggml's fused matmul uses, never read
    /// here. Proven by constructing two blocks with the SAME `d`/`qs` but a
    /// deliberately WRONG `s` (`0xFFFF`, not `d*sum(qs)`) and asserting the
    /// dequant still matches Q8_0's - if `s` were mistakenly folded into the
    /// decode, this would be the case that shows it.
    #[test]
    fn q8_1_dequantizes_identically_to_q8_0_ignoring_the_cached_sum_field() {
        let d = half::f16::from_f32(0.25);
        let qs: [i8; 32] = std::array::from_fn(|i| (i as i32 - 16) as i8);

        let mut q8_0_raw = Vec::new();
        q8_0_raw.extend(d.to_le_bytes());
        q8_0_raw.extend(qs.iter().map(|&v| v as u8));

        let mut q8_1_raw = Vec::new();
        q8_1_raw.extend(d.to_le_bytes());
        q8_1_raw.extend(0xFFFFu16.to_le_bytes()); // deliberately wrong `s`
        q8_1_raw.extend(qs.iter().map(|&v| v as u8));

        let want = dequantize(T_Q8_0, &q8_0_raw, 32).unwrap();
        let got = dequantize(T_Q8_1, &q8_1_raw, 32).unwrap();
        assert_eq!(got, want, "Q8_1 must decode exactly like Q8_0, ignoring the cached sum field");
    }

    /// Block geometry for the six M17 types, checked directly rather than
    /// only through a round trip - `tensor_nbytes`/`block_geometry` are
    /// `GgmlType`-derived, so this also pins them.
    #[test]
    fn m17_types_report_correct_block_geometry() {
        assert_eq!(block_geometry(T_Q8_1), Some((32, 36)));
        assert_eq!(block_geometry(T_I8), Some((1, 1)));
        assert_eq!(block_geometry(T_I16), Some((1, 2)));
        assert_eq!(block_geometry(T_I32), Some((1, 4)));
        assert_eq!(block_geometry(T_I64), Some((1, 8)));
        assert_eq!(block_geometry(T_F64), Some((1, 8)));
        assert_eq!(tensor_nbytes(T_Q8_1, 32), Some(36));
        assert_eq!(tensor_nbytes(T_I32, 10), Some(40));
    }

    /// MXFP4's E2M1 LUT and E8M0-half scale, both pinned exactly. `e=128`
    /// makes the scale exactly `1.0` (`e8m0_to_fp32_half`'s own doc works
    /// this out: `2^(128-128)`), so the decoded values equal
    /// [`KVALUES_MXFP4`] verbatim - packing code `j` into BOTH nibbles of
    /// byte `j` (`j | (j<<4)`) exercises every one of the 16 codes in a
    /// single block, in both the low-nibble (first 16 output elements) and
    /// high-nibble (last 16) position.
    #[test]
    fn mxfp4_dequantizes_using_the_e2m1_lut_and_e8m0_half_scale() {
        let mut raw = vec![128u8]; // e: scale = 1.0
        raw.extend((0u8..16).map(|j| j | (j << 4)));
        assert_eq!(raw.len(), 17);
        let got = dequantize(T_MXFP4, &raw, 32).unwrap();
        let want: Vec<f32> = KVALUES_MXFP4.iter().chain(KVALUES_MXFP4.iter()).map(|&v| v as f32).collect();
        assert_eq!(got, want);

        // Scale is genuinely applied, not just along for the ride: e=127
        // halves everything.
        let mut raw2 = vec![127u8];
        raw2.extend((0u8..16).map(|j| j | (j << 4)));
        let got2 = dequantize(T_MXFP4, &raw2, 32).unwrap();
        let want2: Vec<f32> = want.iter().map(|&v| v * 0.5).collect();
        assert_eq!(got2, want2);
    }

    /// IQ4_NL's own 16-entry non-linear codebook, same LUT-coverage
    /// construction as the MXFP4 test above; `d=1.0` (fp16) makes the
    /// decoded values equal [`KVALUES_IQ4NL`] verbatim.
    #[test]
    fn iq4_nl_dequantizes_using_its_own_codebook() {
        let mut raw = half::f16::from_f32(1.0).to_le_bytes().to_vec();
        raw.extend((0u8..16).map(|j| j | (j << 4)));
        assert_eq!(raw.len(), 18);
        let got = dequantize(T_IQ4_NL, &raw, 32).unwrap();
        let want: Vec<f32> = KVALUES_IQ4NL.iter().chain(KVALUES_IQ4NL.iter()).map(|&v| v as f32).collect();
        assert_eq!(got, want);
    }

    /// IQ4_XS's 8 sub-block scales - `ls = scales_l nibble | (scales_h
    /// 2-bit field << 4)`, `dl = d*(ls-32)` - checked against values
    /// independently hand-computed from the bit layout (not from this
    /// decoder's own output): sub-block 0 (nibble=5, hbits=2) -> ls=37 ->
    /// dl_factor=5; sub-block 3 (nibble=0xB, hbits=3, the HIGH nibble of
    /// `scales_l[1]`) -> ls=59 -> dl_factor=27; every other sub-block's
    /// scale bits are left zero -> ls=0 -> dl_factor=-32.
    #[test]
    fn iq4_xs_scale_extraction_matches_independently_computed_bits() {
        let d = 1.0f32;
        let mut scales_l = [0u8; 4];
        let mut scales_h: u16 = 0;
        scales_l[0] = 0x05; // sub-block 0's nibble (low nibble of scales_l[0])
        scales_h |= 2 << (2 * 0); // sub-block 0's 2-bit high field
        scales_l[1] = 0x0B << 4; // sub-block 3's nibble (HIGH nibble of scales_l[1])
        scales_h |= 3 << (2 * 3); // sub-block 3's 2-bit high field

        let mut raw = half::f16::from_f32(d).to_le_bytes().to_vec();
        raw.extend(scales_h.to_le_bytes());
        raw.extend(scales_l);
        // qs: code 1 in every low nibble, code 2 in every high nibble - so
        // each sub-block's 32 elements are exactly [KVALUES_IQ4NL[1]*dl;
        // 16] then [KVALUES_IQ4NL[2]*dl; 16], letting one assert per
        // sub-block cover both its scale AND its codebook lookup.
        raw.extend([0x21u8; 16].repeat(8)); // (2<<4)|1 = 0x21, 16 bytes per sub-block x 8
        assert_eq!(raw.len(), 136);

        let got = dequantize(T_IQ4_XS, &raw, QK_K).unwrap();
        let dl_factors = [5.0f32, -32.0, -32.0, 27.0, -32.0, -32.0, -32.0, -32.0];
        for (ib, &factor) in dl_factors.iter().enumerate() {
            let dl = d * factor;
            let base = ib * 32;
            for k in 0..16 {
                assert_eq!(got[base + k], KVALUES_IQ4NL[1] as f32 * dl, "sub-block {ib} low nibble, elem {k}");
                assert_eq!(got[base + 16 + k], KVALUES_IQ4NL[2] as f32 * dl, "sub-block {ib} high nibble, elem {k}");
            }
        }
    }

    /// TQ1_0 against values computed by an INDEPENDENT Python transcription
    /// of `dequantize_row_tq1_0` (not this Rust decoder, and not hand
    /// arithmetic) - the two-chunk `qs` split (`[0..32]` then `[32..48]`,
    /// each running the full 5-digit loop) is exactly the shape a
    /// naive single-pass "simplification" gets wrong (caught while writing
    /// [`deq_tq1_0`] the first time), so this pins real, non-trivial byte
    /// values at both chunk boundaries and in `qh`, not an all-zero
    /// degenerate case.
    #[test]
    fn tq1_0_matches_an_independently_computed_reference() {
        let mut qs = [0u8; 48];
        qs[0] = 5;
        qs[1] = 200;
        qs[31] = 77; // last byte of the FIRST chunk
        qs[32] = 9; // first byte of the SECOND chunk
        qs[47] = 250; // last byte of the second chunk
        let mut qh = [0u8; 4];
        qh[0] = 42;
        qh[3] = 123;
        let d = 2.0f32;

        let mut raw = Vec::new();
        raw.extend(qs);
        raw.extend(qh);
        raw.extend(half::f16::from_f32(d).to_le_bytes());
        assert_eq!(raw.len(), 54);

        let got = dequantize(T_TQ1_0, &raw, QK_K).unwrap();
        // index = n*32+m for the first chunk (n in 0..5, m in 0..32).
        assert_eq!(got[0 * 32 + 0], -2.0, "qs[0]=5, n=0");
        assert_eq!(got[0 * 32 + 1], 2.0, "qs[1]=200, n=0");
        assert_eq!(got[4 * 32 + 31], 144.0, "qs[31]=77, n=4");
        // index = 160 + n*16+m for the second chunk (n in 0..5, m in 0..16).
        assert_eq!(got[160 + 0 * 16 + 0], -2.0, "qs[32]=9, n=0");
        assert_eq!(got[160 + 4 * 16 + 15], 472.0, "qs[47]=250, n=4");
        // index = 240 + n*4+m for qh (n in 0..4, m in 0..4).
        assert_eq!(got[240], -2.0, "qh[0]=42, n=0");
        assert_eq!(got[240 + 3 * 4 + 3], 74.0, "qh[3]=123, n=3");
        // A zero byte's digit is -1 for every n, at every position -
        // pinned once for an untouched byte (qs[3]) as a baseline.
        assert_eq!(got[3], -2.0, "qs[3]=0 (untouched)");
    }

    /// TQ2_0 against the same independent-Python-reference discipline as
    /// TQ1_0 above; the two 32-byte chunks (`l` shift loop nested INSIDE
    /// each chunk, not the chunk nested inside `l`) is the analogous
    /// ordering trap.
    #[test]
    fn tq2_0_matches_an_independently_computed_reference() {
        let mut qs = [0u8; 64];
        qs[0] = 0b11100100; // l0=00(-1) l1=01(0) l2=10(1) l3=11(2)
        qs[31] = 0b01_10_11_00; // last byte of chunk 0
        qs[32] = 0b00_01_10_11; // first byte of chunk 1
        qs[63] = 0b10101010;
        let d = 3.0f32;

        let mut raw = Vec::new();
        raw.extend(qs);
        raw.extend(half::f16::from_f32(d).to_le_bytes());
        assert_eq!(raw.len(), 66);

        let got = dequantize(T_TQ2_0, &raw, QK_K).unwrap();
        // index = l*32+m within chunk 0 (m in 0..32).
        assert_eq!(got[0 * 32], -3.0, "qs[0] l=0");
        assert_eq!(got[1 * 32], 0.0, "qs[0] l=1");
        assert_eq!(got[2 * 32], 3.0, "qs[0] l=2");
        assert_eq!(got[3 * 32], 6.0, "qs[0] l=3");
        assert_eq!(got[0 * 32 + 31], -3.0, "qs[31] l=0");
        assert_eq!(got[3 * 32 + 31], 0.0, "qs[31] l=3");
        // index = 128 + l*32+m within chunk 1.
        assert_eq!(got[128], 6.0, "qs[32] l=0");
        assert_eq!(got[128 + 3 * 32], -3.0, "qs[32] l=3");
        assert_eq!(got[128 + 31], 3.0, "qs[63] l=0");
        assert_eq!(got[128 + 3 * 32 + 31], 3.0, "qs[63] l=3");
    }

    /// A GGUF carrying any of the five M18 types must open and dequantize
    /// through the real container path, not just the bare `dequantize`
    /// function - `raw_blocks`/`MmapGguf::open` are the paths a caller
    /// actually uses.
    #[test]
    fn m18_types_open_through_mmapgguf_and_dequantize() {
        for (ty, block_bytes, numel) in [
            (T_MXFP4, 17usize, 32usize),
            (T_IQ4_NL, 18, 32),
            (T_IQ4_XS, 136, QK_K),
            (T_TQ1_0, 54, QK_K),
            (T_TQ2_0, 66, QK_K),
        ] {
            let data = vec![0u8; block_bytes];
            let path = std::env::temp_dir().join(format!("brain-gguf-m18-{ty}-{}.gguf", std::process::id()));
            let path = path.to_str().unwrap().to_string();
            crate::gguf_write::write(&path, &[], &[crate::gguf_write::TensorOut { name: "w".to_string(), shape: vec![numel], ty, data }], 32).unwrap();
            let mg = MmapGguf::open(&path).unwrap();
            let got = mg.tensor("w").unwrap();
            assert!(got.is_ok(), "type {ty} must dequantize, not error: {got:?}");
            assert_eq!(got.unwrap().len(), numel);
            std::fs::remove_file(&path).ok();
        }
    }
}
