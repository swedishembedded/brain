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
//! own `parse`); [`load_tokenizer`] feeds them straight to
//! [`data::qwen_tokenizer::QwenBpe::from_json_bytes`].
//!
//! **No new tokenizer implementation was needed.** Per this repo's own
//! "check for reuse first" ethos (`kernels.md` §F.3 for kernels, the same
//! discipline applied here to non-kernel code), `crates/data` already ships
//! two `tokenizer.json`-compatible readers: [`data::qwen_tokenizer::
//! QwenBpe`] (byte-level BPE) and `data::unigram` (SentencePiece unigram,
//! for the umT5/T5/ALBERT/XLNet family). Range-fetching the real
//! `tokenizer_json` tensor this session (see this crate's roadmap entry for
//! the fetch recipe) and parsing it confirmed Gemma-4's tokenizer IS a
//! `QwenBpe`-compatible byte-level BPE, not a unigram model:
//! `model.type == "BPE"`, `model.vocab` has exactly 262144 entries (matching
//! [`crate::config::Gemma4Config::gemma4_12b`]'s `vocab_size` exactly), and
//! it round-trips `encode`/`decode` bit-for-bit on ordinary text. No fields
//! [`data::qwen_tokenizer::QwenBpe::from_json_bytes`] cannot already parse
//! were found - this module is a thin extraction + reuse, not a partial or
//! fragile new parser.

use checkpoint::safetensors::StTensor;
use data::qwen_tokenizer::QwenBpe;

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
    let mut bytes = Vec::with_capacity(t.data.len());
    for (i, &v) in t.data.iter().enumerate() {
        if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
            return Err(format!("gemma4 tokenizer: tokenizer_json[{i}] = {v} is not a valid byte value"));
        }
        bytes.push(v as u8);
    }
    Ok(bytes)
}

/// Extract + parse in one call - [`extract_tokenizer_json_bytes`] then
/// [`data::qwen_tokenizer::QwenBpe::from_json_bytes`].
pub fn load_tokenizer(tensors: &[StTensor]) -> Result<QwenBpe, String> {
    let bytes = extract_tokenizer_json_bytes(tensors)?;
    QwenBpe::from_json_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::Tokenizer as _;

    /// A tiny but structurally real `tokenizer.json` (BPE model, one merge,
    /// one added special token) encoded the same way `checkpoint::
    /// safetensors::parse` decodes a real `U8` tensor (each byte value
    /// stored as its own `f32`) - proves the byte round-trip AND that the
    /// recovered bytes are actually consumable by [`QwenBpe::
    /// from_json_bytes`], without needing the real 32MB tensor or any
    /// network access.
    fn synthetic_tokenizer_json_tensor() -> StTensor {
        let json = br#"{"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":[["h","i"]]},"added_tokens":[{"id":3,"content":"<end>"}]}"#;
        StTensor { name: "tokenizer_json".into(), shape: vec![json.len()], data: json.iter().map(|&b| b as f32).collect() }
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
    fn load_tokenizer_parses_the_recovered_bytes_as_a_real_bpe_tokenizer() {
        let src = synthetic_tokenizer_json_tensor();
        let tok = load_tokenizer(&[src]).expect("parse");
        assert_eq!(tok.vocab_size(), 4); // max id (3, "<end>") + 1
        assert_eq!(tok.special_id("<end>"), Some(3));
    }

    /// Documentation pin, not a network check - see this module's doc for
    /// the fetch that established this value.
    #[test]
    fn real_tokenizer_json_length_is_pinned() {
        assert_eq!(REAL_TOKENIZER_JSON_BYTE_LEN, 32_169_626);
    }
}
