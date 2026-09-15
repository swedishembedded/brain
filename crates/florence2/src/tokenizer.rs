// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2's tokenizer: the base BART/RoBERTa byte-level BPE the
//! checkpoint ships (`tokenizer.json`, 50265 ids) extended with 1024 tokens
//! Florence-2's own `Florence2Processor.__init__` adds PROGRAMMATICALLY at
//! load time - never present in the checkpoint's `tokenizer.json`
//! `added_tokens` field.
//!
//! Confirmed against the real `microsoft/Florence-2-base` repo, not assumed:
//! its `tokenizer.json`/`vocab.json` carry only the 5 base RoBERTa specials
//! (`<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`); `config.json` declares
//! `vocab_size: 51289` = 50265 + 1024, and the extra 1024 exist only as the
//! literal Python list in `processing_florence2.py` (`microsoft/Florence-2-base`,
//! lines 87-91), reproduced verbatim in [`additional_tokens`] below - order
//! matters, since ids are assigned sequentially in this exact order.
//!
//! Loading therefore needs one extra step beyond every other checkpoint this
//! repo imports: [`QwenBpe::from_dir`]'s generic `tokenizer.json` parser,
//! then [`QwenBpe::add_special_tokens`] (added there for this reason) with
//! [`additional_tokens`] in order.

use data::qwen_tokenizer::QwenBpe;

/// The 4 detection/OCR task-boundary markers, first in add order.
const TASK_MARKERS: &[&str] = &["<od>", "</od>", "<ocr>", "</ocr>"];

/// The remaining 20 task-boundary markers (captioning/grounding/segmentation/
/// region variants), added after the 1000 location tokens.
const OTHER_MARKERS: &[&str] = &[
    "<cap>",
    "</cap>",
    "<ncap>",
    "</ncap>",
    "<dcap>",
    "</dcap>",
    "<grounding>",
    "</grounding>",
    "<seg>",
    "</seg>",
    "<sep>",
    "<region_cap>",
    "</region_cap>",
    // Typo ("desciption") is the upstream checkpoint's own token spelling -
    // reproduced exactly, not a transcription error here.
    "<region_to_desciption>",
    "</region_to_desciption>",
    "<proposal>",
    "</proposal>",
    "<poly>",
    "</poly>",
    "<and>",
];

/// Number of quantized location bins (`<loc_0>` .. `<loc_999>`): pixel
/// coordinates normalized by image width/height then scaled by this many
/// bins, per Florence-2's location-token scheme.
pub const NUM_LOCATION_BINS: u32 = 1000;

/// The exact 1024 additional tokens, in the exact order
/// `Florence2Processor.__init__` adds them: 4 task markers, then 1000
/// location tokens, then 20 more task markers.
pub fn additional_tokens() -> Vec<String> {
    let mut v: Vec<String> = Vec::with_capacity(4 + NUM_LOCATION_BINS as usize + OTHER_MARKERS.len());
    v.extend(TASK_MARKERS.iter().map(|s| s.to_string()));
    v.extend((0..NUM_LOCATION_BINS).map(|x| format!("<loc_{x}>")));
    v.extend(OTHER_MARKERS.iter().map(|s| s.to_string()));
    v
}

/// Load Florence-2's tokenizer from a checkpoint directory: the base
/// `tokenizer.json` via [`QwenBpe::from_dir`], extended with
/// [`additional_tokens`] so `<loc_512>`/`<od>`/... encode/decode as single
/// atomic tokens exactly like the reference Python processor's.
pub fn load(dir: &str) -> Result<QwenBpe, String> {
    let mut bpe = QwenBpe::from_dir(dir)?;
    let extra = additional_tokens();
    let refs: Vec<&str> = extra.iter().map(String::as_str).collect();
    bpe.add_special_tokens(&refs);
    Ok(bpe)
}

/// The location-token content for bin `n` (`0..NUM_LOCATION_BINS`).
pub fn location_token(bin: u32) -> String {
    format!("<loc_{bin}>")
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additional_tokens_has_exactly_1024_entries_in_order() {
        let t = additional_tokens();
        assert_eq!(t.len(), 1024);
        assert_eq!(&t[0..4], &["<od>", "</od>", "<ocr>", "</ocr>"]);
        assert_eq!(t[4], "<loc_0>");
        assert_eq!(t[1003], "<loc_999>");
        assert_eq!(t[1004], "<cap>");
        assert_eq!(t[1023], "<and>");
    }

    #[test]
    fn location_token_formats_bin_index() {
        assert_eq!(location_token(0), "<loc_0>");
        assert_eq!(location_token(999), "<loc_999>");
    }

    /// `FLORENCE2_DIR` (checkpoint dir, e.g. the downloaded
    /// `microsoft/Florence-2-base` repo under this workspace's resources
    /// mount) - skips cleanly when unset, matching this repo's
    /// hardware/checkpoint-gated test convention.
    fn florence_dir() -> Option<String> {
        std::env::var("FLORENCE2_DIR").ok()
    }

    #[test]
    fn loads_real_checkpoint_tokenizer_and_extends_vocab_to_51289() {
        let Some(dir) = florence_dir() else {
            brain_testutil::skip("FLORENCE2_DIR unset");
            return;
        };
        let bpe = load(&dir).expect("load florence2 tokenizer");
        assert_eq!(bpe.vocab_size(), 51289, "50265 base + 1024 added");
        // Add order: 4 task markers, then 1000 loc bins, then 20 more markers.
        assert_eq!(bpe.special_id("<od>"), Some(50265));
        assert_eq!(bpe.special_id("<loc_0>"), Some(50269));
        assert_eq!(bpe.special_id("<loc_999>"), Some(51268));
        assert_eq!(bpe.special_id("<and>"), Some(51288));
    }

    #[test]
    fn pinned_reference_vectors() {
        let Some(dir) = florence_dir() else {
            brain_testutil::skip("FLORENCE2_DIR unset");
            return;
        };
        use data::tokenizer::Tokenizer;
        let bpe = load(&dir).expect("load florence2 tokenizer");
        // Ground truth from the real HF `transformers.AutoTokenizer` on the
        // downloaded `microsoft/Florence-2-base` checkpoint, replicating
        // `Florence2Processor.__init__`'s exact `add_special_tokens` call
        // (`encode(text, add_special_tokens=False)`).
        assert_eq!(
            bpe.encode("cat<loc_212><loc_345><loc_600><loc_800>"),
            vec![8729, 50481, 50614, 50869, 51069]
        );
        assert_eq!(
            bpe.encode("Locate Get started / Add card button in the image."),
            vec![574, 22486, 2315, 554, 1589, 4287, 1886, 6148, 11, 5, 2274, 4]
        );
        assert_eq!(bpe.encode("The capital of France is"), vec![133, 812, 9, 1470, 16]);
        assert_eq!(
            bpe.encode("Locate the phrases in the caption: a red car and a blue house"),
            vec![574, 22486, 5, 22810, 11, 5, 3747, 35, 10, 1275, 512, 8, 10, 2440, 790]
        );
    }

    #[test]
    fn real_checkpoint_location_tokens_round_trip() {
        use data::tokenizer::Tokenizer;
        let Some(dir) = florence_dir() else {
            brain_testutil::skip("FLORENCE2_DIR unset");
            return;
        };
        let bpe = load(&dir).expect("load florence2 tokenizer");
        let text = "cat<loc_212><loc_345><loc_600><loc_800>";
        let ids = bpe.encode(text);
        // The 4 location tokens must each survive as one atomic id, not get
        // BPE'd byte-by-byte (which would produce far more than 5 ids for
        // "cat" + 4 locs).
        let loc_lo = bpe.special_id(&location_token(0)).unwrap();
        let loc_hi = bpe.special_id(&location_token(NUM_LOCATION_BINS - 1)).unwrap();
        let loc_ids: Vec<u32> = ids.iter().copied().filter(|&id| (loc_lo..=loc_hi).contains(&id)).collect();
        assert_eq!(loc_ids.len(), 4, "all 4 loc tokens should be atomic: {ids:?}");
        assert_eq!(bpe.decode(&ids), text);
    }
}
