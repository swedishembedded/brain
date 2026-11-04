// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real checkpoint's own embedded tokenizer.
//!
//! `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` carries its tokenizer as
//! ONE tensor, `tokenizer_json` (dtype `U8`, shape `[32169626]`, range-read
//! and confirmed this session at `data_offsets` derived from the header) -
//! the raw bytes of a standard HuggingFace `tokenizer.json`, not a separate
//! file. [`extract_tokenizer_json_bytes`] recovers those bytes from
//! [`checkpoint::safetensors::StTensor`]'s already-decoded `f32` form
//! (`U8` bytes decode losslessly to `f32`, see `checkpoint::safetensors`'s
//! own `parse`); [`load_tokenizer`] parses them with
//! [`data::llama_bpe::LlamaBpe`].
//!
//! Gemma-4's tokenizer is SentencePiece-style BPE, the scheme `LlamaBpe`
//! implements, not byte-level BPE: the normalizer replaces every space with
//! `▁`, the initial symbols are characters, and a character the vocabulary
//! lacks falls back to its UTF-8 bytes as `<0xNN>` pieces
//! (`model.byte_fallback`). Read as byte-level BPE, a space maps to a symbol
//! the vocabulary does not hold and is dropped, so every prompt reaches the
//! encoder with its words run together.

use checkpoint::safetensors::StTensor;
use data::llama_bpe::LlamaBpe;

/// The real checkpoint's `tokenizer_json` tensor's raw byte length -
/// `data_offsets` span `[26231604668, 26263774294)` in the real
/// `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` file, range-read and
/// confirmed this session. Pinned here the same way [`crate::config`]'s
/// `gemma4_12b` config values are pinned - a guard against a future
/// checkpoint revision silently shipping a differently-sized tokenizer
/// without this crate noticing.
pub const REAL_TOKENIZER_JSON_BYTE_LEN: usize = 32_169_626;

/// Recover `tokenizer_json`'s raw bytes from an already-`f32`-decoded
/// [`StTensor`] list (e.g. [`checkpoint::safetensors::read`]'s output).
/// Errors by name if the tensor is absent, and (defensively, since this is
/// untrusted checkpoint content) if any decoded value is out of `u8` range -
/// `checkpoint::safetensors::parse`'s own `U8 => raw.iter().map(|&b| b as
/// f32)` decode is exact for every legal byte value, so an out-of-range
/// value means the source was never really `U8` data, not a rounding
/// artifact.
pub fn extract_tokenizer_json_bytes(tensors: &[StTensor]) -> Result<Vec<u8>, String> {
    let t = tensors.iter().find(|t| t.name == "tokenizer_json").ok_or("gemma4 tokenizer: missing tokenizer_json tensor")?;
    bytes_from_f32(&t.name, &t.data)
}

/// The byte-recovery half of [`extract_tokenizer_json_bytes`], over a decoded
/// `f32` slice from any source. Shared with `crate::gguf_src`, which reaches
/// the same blob through a GGUF rather than a safetensors tensor list - one
/// range check, not two that can drift.
pub fn bytes_from_f32(name: &str, data: &[f32]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(data.len());
    for (i, &v) in data.iter().enumerate() {
        if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
            return Err(format!("gemma4 tokenizer: {name}[{i}] = {v} is not a valid byte value"));
        }
        bytes.push(v as u8);
    }
    Ok(bytes)
}

/// Extract + parse in one call - [`extract_tokenizer_json_bytes`] then [`parse_tokenizer`].
pub fn load_tokenizer(tensors: &[StTensor]) -> Result<LlamaBpe, String> {
    parse_tokenizer(&extract_tokenizer_json_bytes(tensors)?)
}

/// The tokenizer the raw `tokenizer.json` bytes declare.
pub fn parse_tokenizer(bytes: &[u8]) -> Result<LlamaBpe, String> {
    let json = std::str::from_utf8(bytes).map_err(|e| format!("gemma4 tokenizer: tokenizer_json is not UTF-8: {e}"))?;
    LlamaBpe::from_tokenizer_json(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny Gemma-shaped `tokenizer.json` (byte-fallback BPE with every
    /// `<0xNN>` piece, the `Replace` normalizer, one merge, the four specials)
    /// encoded the way `checkpoint::safetensors::parse` decodes a real `U8`
    /// tensor (each byte value stored as its own `f32`): proves the byte
    /// round-trip and that the recovered bytes parse, without the real 32 MB
    /// tensor.
    fn synthetic_tokenizer_json_tensor() -> StTensor {
        let mut vocab = serde_json::Map::new();
        for (i, name) in ["<pad>", "<eos>", "<bos>", "<unk>"].iter().enumerate() {
            vocab.insert(name.to_string(), (i as u32).into());
        }
        for b in 0u32..256 {
            vocab.insert(format!("<0x{b:02X}>"), (4 + b).into());
        }
        for (i, p) in ["h", "i", "hi"].iter().enumerate() {
            vocab.insert(p.to_string(), (260 + i as u32).into());
        }
        let json = serde_json::json!({
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            "pre_tokenizer": null,
            "post_processor": null,
            "added_tokens": [],
            "model": {"type": "BPE", "byte_fallback": true, "merges": [["h", "i"]], "vocab": vocab}
        })
        .to_string();
        StTensor { name: "tokenizer_json".into(), shape: vec![json.len()], data: json.bytes().map(|b| b as f32).collect() }
    }

    #[test]
    fn extract_recovers_exact_bytes() {
        let src = synthetic_tokenizer_json_tensor();
        let want: Vec<u8> = src.data.iter().map(|&v| v as u8).collect();
        let got = extract_tokenizer_json_bytes(std::slice::from_ref(&src)).expect("extract");
        assert_eq!(got, want);
        assert_eq!(String::from_utf8(got).unwrap(), String::from_utf8(want).unwrap());
    }

    #[test]
    fn extract_errors_by_name_when_absent() {
        let e = extract_tokenizer_json_bytes(&[]).unwrap_err();
        assert!(e.contains("tokenizer_json"), "{e}");
    }

    #[test]
    fn load_tokenizer_parses_the_recovered_bytes() {
        let src = synthetic_tokenizer_json_tensor();
        let tok = load_tokenizer(&[src]).expect("parse");
        assert_eq!(tok.bos_id(), 2);
        assert_eq!(tok.encode_raw("hi"), [262]);
    }

    /// Documentation pin, not a network check - see this module's doc for
    /// the fetch that established this value.
    #[test]
    fn real_tokenizer_json_length_is_pinned() {
        assert_eq!(REAL_TOKENIZER_JSON_BYTE_LEN, 32_169_626);
    }
}
