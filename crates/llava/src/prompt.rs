// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The image-token splice: where the projected image embeddings enter the
//! text token sequence.
//!
//! Transcribed from upstream `llava/mm_utils.py::tokenizer_image_token`, not
//! guessed. `<image>` (the literal 7-byte substring `crate::template` writes
//! into the prompt) is **not** a vocab token in the base LLaMA/Vicuna
//! tokenizer - `data::llama_bpe::LlamaBpe` tokenizes it as ordinary text if
//! asked to. Upstream instead **splits the prompt string on it** before
//! tokenizing, so `<image>` never reaches the tokenizer at all:
//!
//! ```text
//! prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
//! # keep chunk[0]'s own BOS; every later chunk drops its own BOS
//! # and IMAGE_TOKEN_INDEX (-200) is inserted between consecutive chunks
//! ```
//!
//! `-200` is never a valid vocab id (LLaMA-2's vocab is `0..32000`), so it is
//! an unambiguous sentinel the decoder's embedding-splice step
//! ([`qwen3::model::Qwen::enable_mm_splice`] /
//! [`qwen3::model::PrefillInput::Embed`]) recognises and replaces with a row
//! of the projected image embeddings, in order.

use data::llama_bpe::LlamaBpe;

/// `IMAGE_TOKEN_INDEX` - LLaVA's out-of-vocabulary splice sentinel.
pub const IMAGE_TOKEN_INDEX: i64 = -200;

/// Tokenize `prompt`, replacing every literal [`crate::template::IMAGE_TOKEN`]
/// occurrence with one [`IMAGE_TOKEN_INDEX`] sentinel instead of BPE-encoding
/// it as text - the exact shape `tokenizer_image_token` produces (BOS kept
/// once, at the very front; dropped from every chunk after the first).
pub fn tokenize_with_image_splice(tok: &LlamaBpe, prompt: &str) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(prompt.len() / 3 + 1);
    for (i, chunk) in prompt.split(crate::template::IMAGE_TOKEN).enumerate() {
        if i > 0 {
            out.push(IMAGE_TOKEN_INDEX);
        }
        let ids = if i == 0 { tok.encode(chunk) } else { tok.encode_raw(chunk) };
        out.extend(ids.into_iter().map(i64::from));
    }
    out
}

/// Expand the `-200` sentinels in `ids` into `n_visual` consecutive
/// [`qwen3::model::PrefillInput::Embed`] rows sliced from `embeds`
/// (`[n_visual, d_model]`, row-major) and every other id into a
/// [`qwen3::model::PrefillInput::Token`]. LLaVA-1.5 splices exactly one image
/// per caption call, so exactly one `-200` run of length 1 is expected; a
/// prompt with none or more than one is a caller error, named rather than
/// silently mis-spliced.
pub fn splice_image_embeds<'a>(ids: &[i64], embeds: &'a [f32], n_visual: u32, d_model: u32) -> Result<Vec<qwen3::model::PrefillInput<'a>>, String> {
    let want = (n_visual * d_model) as usize;
    if embeds.len() != want {
        return Err(format!("llava: image embeds are {} floats, expected {want} ({n_visual} x {d_model})", embeds.len()));
    }
    let splices = ids.iter().filter(|&&id| id == IMAGE_TOKEN_INDEX).count();
    if splices != 1 {
        return Err(format!("llava: prompt has {splices} image splice point(s), expected exactly 1"));
    }
    let mut out = Vec::with_capacity(ids.len() - 1 + n_visual as usize);
    for &id in ids {
        if id == IMAGE_TOKEN_INDEX {
            let d = d_model as usize;
            for r in 0..n_visual as usize {
                out.push(qwen3::model::PrefillInput::Embed(&embeds[r * d..(r + 1) * d]));
            }
        } else {
            let t: u32 = id.try_into().map_err(|_| format!("llava: token id {id} out of range"))?;
            out.push(qwen3::model::PrefillInput::Token(t));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny, self-contained LlamaBpe fixture - see `data::llama_bpe`'s own
    // tests for the same shape; this module only needs BOS/content ids, not a
    // real 32k vocab.
    fn tiny_tok() -> LlamaBpe {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".into(), 0.into());
        vocab.insert("<s>".into(), 1.into());
        vocab.insert("</s>".into(), 2.into());
        for b in 0u32..256 {
            vocab.insert(format!("<0x{b:02X}>"), (3 + b).into());
        }
        let j = serde_json::json!({
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true}
            ],
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "\u{2581}"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}]},
            "pre_tokenizer": null,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                           {"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
            "model": {"type": "BPE", "byte_fallback": true, "merges": [], "vocab": vocab}
        });
        LlamaBpe::from_tokenizer_json(&j.to_string()).unwrap()
    }

    /// Byte-level fixture: with no merges, every char becomes its own
    /// byte-fallback token, so ids are trivially predictable and the splice
    /// structure (BOS placement, sentinel count/position) is what's tested.
    #[test]
    fn one_image_placeholder_splices_one_sentinel_after_the_kept_bos() {
        let tok = tiny_tok();
        let ids = tokenize_with_image_splice(&tok, "a<image>b");
        // chunk0 "a" WITH bos -> [bos, ...]; sentinel; chunk1 "b" WITHOUT bos.
        assert_eq!(ids[0], i64::from(tok.bos_id()));
        assert_eq!(ids.iter().filter(|&&x| x == IMAGE_TOKEN_INDEX).count(), 1);
        let pos = ids.iter().position(|&x| x == IMAGE_TOKEN_INDEX).unwrap();
        // Nothing after the sentinel encodes another bos.
        assert!(!ids[pos + 1..].contains(&i64::from(tok.bos_id())));
    }

    #[test]
    fn no_placeholder_is_plain_encode() {
        let tok = tiny_tok();
        let ids = tokenize_with_image_splice(&tok, "hello");
        assert_eq!(ids, tok.encode("hello").into_iter().map(i64::from).collect::<Vec<_>>());
        assert!(!ids.contains(&IMAGE_TOKEN_INDEX));
    }

    #[test]
    fn splice_image_embeds_expands_the_sentinel_into_n_visual_rows() {
        let ids = vec![1i64, 5, IMAGE_TOKEN_INDEX, 9];
        let embeds: Vec<f32> = (0..6).map(|i| i as f32).collect(); // 3 rows x d=2
        let out = splice_image_embeds(&ids, &embeds, 3, 2).unwrap();
        assert_eq!(out.len(), 3 + 3); // 3 tokens (minus sentinel) + 3 embed rows
        match &out[2] {
            qwen3::model::PrefillInput::Embed(row) => assert_eq!(*row, &[0.0, 1.0]),
            _ => panic!("expected the first spliced row"),
        }
        match &out[4] {
            qwen3::model::PrefillInput::Embed(row) => assert_eq!(*row, &[4.0, 5.0]),
            _ => panic!("expected the last spliced row"),
        }
        match &out[5] {
            qwen3::model::PrefillInput::Token(t) => assert_eq!(*t, 9),
            _ => panic!("expected the trailing text token"),
        }
    }

    #[test]
    fn splice_image_embeds_rejects_a_prompt_with_no_or_multiple_splices() {
        let embeds = vec![0.0f32; 2];
        let err = |r: Result<Vec<qwen3::model::PrefillInput<'_>>, String>| match r {
            Err(e) => e,
            Ok(_) => panic!("expected a rejection"),
        };
        assert!(err(splice_image_embeds(&[1, 2], &embeds, 1, 2)).contains("0 image splice"));
        assert!(err(splice_image_embeds(&[IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX], &embeds, 1, 2)).contains("2 image splice"));
    }
}
