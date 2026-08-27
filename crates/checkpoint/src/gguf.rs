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
//! usable from wasm; the native [`load_gguf`]/[`read`] wrappers read the file
//! then call it — mirroring the `parse`/`load` split in `lib.rs` and `st.rs`.
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
pub(crate) const T_Q2_K: u32 = 10;
pub(crate) const T_Q3_K: u32 = 11;
pub(crate) const T_Q4_K: u32 = 12;
pub(crate) const T_Q5_K: u32 = 13;
pub(crate) const T_Q6_K: u32 = 14;
pub(crate) const T_Q8_K: u32 = 15;
pub(crate) const T_BF16: u32 = 30;

pub(crate) const QK_K: usize = 256;

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
/// present, else `general.quantization_version`.
fn quant_label(kv: &BTreeMap<String, GgufValue>) -> Option<String> {
    if let Some(ft) = kv.get("general.file_type").and_then(|v| v.as_u64()) {
        return Some(file_type_name(ft as u32).to_string());
    }
    kv.get("general.quantization_version").and_then(|v| v.as_u64()).map(|v| format!("qver{v}"))
}

/// Map a `general.file_type` enum value to its conventional name.
fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        _ => "unknown",
    }
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
#[cfg(not(target_arch = "wasm32"))]
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
/// The same id table [`block_geometry`] dispatches on, so a type this reader
/// can decode always has a name and one it cannot never gets an invented one.
pub(crate) fn ggml_type_name(ty: u32) -> Option<&'static str> {
    Some(match ty {
        T_F32 => "F32",
        T_F16 => "F16",
        T_BF16 => "BF16",
        T_Q4_0 => "Q4_0",
        T_Q4_1 => "Q4_1",
        T_Q5_0 => "Q5_0",
        T_Q5_1 => "Q5_1",
        T_Q8_0 => "Q8_0",
        T_Q2_K => "Q2_K",
        T_Q3_K => "Q3_K",
        T_Q4_K => "Q4_K",
        T_Q5_K => "Q5_K",
        T_Q6_K => "Q6_K",
        T_Q8_K => "Q8_K",
        _ => return None,
    })
}

/// Block geometry `(elements_per_block, bytes_per_block)` for a ggml type.
pub(crate) fn block_geometry(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        T_F32 => (1, 4),
        T_F16 | T_BF16 => (1, 2),
        T_Q4_0 => (32, 18),
        T_Q4_1 => (32, 20),
        T_Q5_0 => (32, 22),
        T_Q5_1 => (32, 24),
        T_Q8_0 => (32, 34),
        T_Q2_K => (256, 84),
        T_Q3_K => (256, 110),
        T_Q4_K => (256, 144),
        T_Q5_K => (256, 176),
        T_Q6_K => (256, 210),
        T_Q8_K => (256, 292),
        _ => return None,
    })
}

/// Total on-disk byte count for `numel` elements of `ty`.
pub(crate) fn tensor_nbytes(ty: u32, numel: usize) -> Option<usize> {
    let (be, bb) = block_geometry(ty)?;
    Some(numel / be * bb + if numel.is_multiple_of(be) { 0 } else { bb })
}

/// The ggml type id of Q8_0, for a caller matching on
/// [`MmapGguf::raw_tensor_bytes`]'s reported type.
pub const TYPE_Q8_0: u32 = T_Q8_0;

/// Elements per Q8_0 block, and the block's on-disk byte size (a 2-byte fp16
/// scale followed by 32 int8 values).
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Expand elements `[e0, e1)` of a Q8_0 tensor's raw bytes into `out`, which
/// is CLEARED first. Both bounds must be multiples of
/// [`Q8_0_BLOCK_ELEMS`], because a Q8_0 block is the smallest independently
/// decodable unit - a caller wanting an unaligned range must expand the
/// enclosing blocks and slice.
///
/// This exists so that a caller reading a sub-range of a large quantized
/// tensor (one weight matrix's rows, or one column block of a fused matrix)
/// does not have to expand the whole tensor to reach it, and does not have to
/// reimplement the block layout to avoid that. It decodes through the SAME
/// `deq_q8_0` every other read path uses, so the values are identical to
/// what [`dequantize`] would have produced for those positions.
pub fn q8_0_expand(raw: &[u8], e0: usize, e1: usize, out: &mut Vec<f32>) -> Result<(), String> {
    const BB: usize = 34;
    if !e0.is_multiple_of(Q8_0_BLOCK_ELEMS) || !e1.is_multiple_of(Q8_0_BLOCK_ELEMS) {
        return Err(format!("gguf: q8_0_expand range [{e0}, {e1}) is not block-aligned ({Q8_0_BLOCK_ELEMS})"));
    }
    if e1 < e0 {
        return Err(format!("gguf: q8_0_expand range [{e0}, {e1}) is inverted"));
    }
    let (b0, b1) = (e0 / Q8_0_BLOCK_ELEMS, e1 / Q8_0_BLOCK_ELEMS);
    if b1 * BB > raw.len() {
        return Err(format!("gguf: q8_0_expand needs {} bytes, tensor has {}", b1 * BB, raw.len()));
    }
    out.clear();
    out.reserve(e1 - e0);
    for b in b0..b1 {
        deq_q8_0(&raw[b * BB..(b + 1) * BB], out);
    }
    Ok(())
}

/// f16 (ggml_half) at byte offset `i` in `b` → f32.
fn f16(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Dequantize a tensor's raw bytes to `numel` fp32 values.
pub(crate) fn dequantize(ty: u32, raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    match ty {
        T_F32 => Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
        T_F16 => Ok(raw.chunks_exact(2).map(|b| crate::safetensors::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
        T_BF16 => Ok(raw.chunks_exact(2).map(|b| crate::safetensors::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
        T_Q4_0 => Ok(deq_blocks(raw, numel, 18, deq_q4_0)),
        T_Q4_1 => Ok(deq_blocks(raw, numel, 20, deq_q4_1)),
        T_Q5_0 => Ok(deq_blocks(raw, numel, 22, deq_q5_0)),
        T_Q5_1 => Ok(deq_blocks(raw, numel, 24, deq_q5_1)),
        T_Q8_0 => Ok(deq_blocks(raw, numel, 34, deq_q8_0)),
        T_Q2_K => Ok(deq_blocks(raw, numel, 84, deq_q2_k)),
        T_Q3_K => Ok(deq_blocks(raw, numel, 110, deq_q3_k)),
        T_Q4_K => Ok(deq_blocks(raw, numel, 144, deq_q4_k)),
        T_Q5_K => Ok(deq_blocks(raw, numel, 176, deq_q5_k)),
        T_Q6_K => Ok(deq_blocks(raw, numel, 210, deq_q6_k)),
        T_Q8_K => Ok(deq_blocks(raw, numel, 292, deq_q8_k)),
        16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 29 | 34 | 35 | 39 => {
            Err(format!("gguf: type {ty} (IQ/TQ/MXFP4 codebook) dequant not yet implemented"))
        }
        other => Err(format!("gguf: unsupported type {other}")),
    }
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

/// A memory-mapped GGUF file with an on-demand, per-tensor dequant accessor.
///
/// [`open`](MmapGguf::open) mmaps the file and parses only the header (KV +
/// tensor infos); no tensor data is read. [`tensor`](MmapGguf::tensor)
/// dequantizes exactly one tensor's block range from the mmap, so peak host
/// memory is bounded by a single tensor's fp32 expansion — never the whole
/// model. Values are byte-identical to the eager [`parse_gguf`].
#[cfg(not(target_arch = "wasm32"))]
pub struct MmapGguf {
    mmap: memmap2::Mmap,
    kv: BTreeMap<String, GgufValue>,
    /// name → (ggml type, absolute byte start, on-disk byte length, element count).
    index: HashMap<String, (u32, usize, usize, usize)>,
    shapes: BTreeMap<String, Vec<usize>>,
    order: Vec<String>,
}

#[cfg(not(target_arch = "wasm32"))]
impl MmapGguf {
    /// Open + mmap `path` and parse only its header (no tensor bytes are read).
    pub fn open(path: &str) -> Result<MmapGguf, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("gguf: open {path}: {e}"))?;
        // SAFETY: weight files are treated as immutable for the mapping's lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("gguf: mmap {path}: {e}"))?;
        let (kv, infos, data_start) = parse_header(&mmap)?;
        let mut index = HashMap::with_capacity(infos.len());
        let mut shapes = BTreeMap::new();
        let mut order = Vec::with_capacity(infos.len());
        for info in infos {
            let numel: usize = info.shape.iter().product();
            let start = data_start
                .checked_add(info.offset as usize)
                .ok_or("gguf: tensor offset overflow")?;
            let nbytes = tensor_nbytes(info.ty, numel)
                .ok_or_else(|| format!("gguf: {} unknown type {}", info.name, info.ty))?;
            if start + nbytes > mmap.len() {
                return Err(format!("gguf: {} data out of range", info.name));
            }
            index.insert(info.name.clone(), (info.ty, start, nbytes, numel));
            shapes.insert(info.name.clone(), info.shape);
            order.push(info.name);
        }
        Ok(MmapGguf { mmap, kv, index, shapes, order })
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
        ggml_type_name(self.index.get(name)?.0)
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
        let &(ty, start, nbytes, numel) = self.index.get(name)?;
        let raw = &self.mmap[start..start + nbytes];
        Some(dequantize(ty, raw, numel).map_err(|e| format!("gguf: {name}: {e}")))
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
        let &(ty, start, nbytes, _numel) = self.index.get(name)?;
        Some((&self.mmap[start..start + nbytes], ty))
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

    /// Zero-copy: `name`'s on-disk bytes reinterpreted as `u32` words, borrowed
    /// straight from the mapping. `None` unless the tensor's own ggml type is
    /// already `F32` (nothing to dequantize) AND its byte range is 4-byte
    /// aligned in the mapping - `bytemuck::try_cast_slice` is the alignment
    /// check, so a misaligned range cleanly falls through to `None` rather
    /// than panicking. A GGUF release commonly leaves 1-D tensors (norms,
    /// biases) in plain F32 while quantizing the 2-D weights, so this fires
    /// selectively per tensor, not per file.
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        let &(ty, start, nbytes, _numel) = self.index.get(name)?;
        if ty != T_F32 {
            return None;
        }
        bytemuck::try_cast_slice::<u8, u32>(&self.mmap[start..start + nbytes]).ok()
    }

    /// Element count of `name`, without decoding - known from the header for
    /// every ggml type, quantized or not, since the caller only needs to know
    /// how many f32s a dequantized [`with_tensor`](Self::with_tensor) call
    /// will produce.
    fn numel(&self, name: &str) -> Option<usize> {
        self.index.get(name).map(|&(_, _, _, numel)| numel)
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
        let err = dequantize(20, &[0u8; 64], 32).unwrap_err();
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
    /// Deliberately NOT "whatever `.gguf` is lying around": [`load_gguf`]
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
}
