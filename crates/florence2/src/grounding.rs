// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parse Florence-2's generated location-token text into normalized
//! bounding boxes: `phrase<loc_a><loc_b><loc_c><loc_d>`, repeated - the
//! shape both `<CAPTION_TO_PHRASE_GROUNDING>` ("phrase_grounding") and
//! `<OD>`/`<DENSE_REGION_CAPTION>` ("description_with_bboxes") decode to
//! in the reference's own post-processor (`processing_florence2.py`'s
//! `box_pattern`, `r'([a-zA-Z0-9 ]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'`),
//! so one parser covers both task types this repo's UI-grounding use case
//! might end up using.
//!
//! Operates on DECODED TEXT (not raw token ids): `<loc_N>` decodes to its
//! literal content via the tokenizer's ordinary special-token decode path
//! (see `tokenizer.rs`), so a run of 4 adjacent `<loc_N>` tags in the
//! decoded string is exactly the model's box notation - no token-id-level
//! bookkeeping needed. A hand-written scanner, not a regex dependency,
//! matching this repo's tokenizer code's own style.

/// One grounded box: the phrase (or object category) text immediately
/// preceding its 4 location tags, and the box itself as `[x0, y0, x1, y1]`,
/// already normalized to `0.0..=1.0` (each coordinate is `bin / 1000.0`,
/// Florence-2's location-token scale), top-left/bottom-right corners.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedBox {
    pub phrase: String,
    pub bbox: [f32; 4],
}

/// Parse every `phrase<loc_a><loc_b><loc_c><loc_d>` group out of `text`.
/// A malformed trailing loc-tag run (not exactly a multiple of 4) yields no
/// box for that trailing partial run - it's dropped, not an error, since a
/// truncated generation (hit `max_new_tokens` mid-box) is a real, recoverable
/// case, not a parse failure the caller needs to handle specially.
pub fn parse_boxes(text: &str) -> Vec<GroundedBox> {
    let mut out = Vec::new();
    let mut phrase_start = 0usize;
    let mut bytes = text;
    let mut consumed = 0usize;
    let mut pending_bins: Vec<u32> = Vec::new();
    // Byte position where the CURRENT run's first tag starts - the phrase
    // for this run is `text[phrase_start..run_start]`, not up to the tag
    // that happens to be current when the 4th one completes the run.
    let mut run_start = 0usize;

    loop {
        match bytes.find("<loc_") {
            None => break,
            Some(rel_pos) => {
                let abs_pos = consumed + rel_pos;
                let after_tag = &bytes[rel_pos..];
                let Some((bin, tag_len)) = parse_one_loc_tag(after_tag) else {
                    // "<loc_" without a valid ">"-terminated digit run - skip
                    // past this occurrence so the loop makes progress.
                    bytes = &after_tag[5..];
                    consumed = abs_pos + 5;
                    continue;
                };

                if pending_bins.is_empty() {
                    run_start = abs_pos;
                }
                pending_bins.push(bin);

                if pending_bins.len() == 4 {
                    let phrase = text[phrase_start..run_start].trim().to_string();
                    let bbox = [pending_bins[0] as f32 / 1000.0, pending_bins[1] as f32 / 1000.0, pending_bins[2] as f32 / 1000.0, pending_bins[3] as f32 / 1000.0];
                    out.push(GroundedBox { phrase, bbox });
                    pending_bins.clear();
                    phrase_start = abs_pos + tag_len;
                }

                bytes = &after_tag[tag_len..];
                consumed = abs_pos + tag_len;
            }
        }
    }

    out
}

/// Parse one `<loc_N>` tag at the start of `s` (`s` must start with
/// `"<loc_"`). Returns `(N, byte_length_of_the_whole_tag)`.
fn parse_one_loc_tag(s: &str) -> Option<(u32, usize)> {
    let rest = s.strip_prefix("<loc_")?;
    let close = rest.find('>')?;
    let digits = &rest[..close];
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u32 = digits.parse().ok()?;
    Some((n, 5 + close + 1))
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_box_with_phrase() {
        let boxes = parse_boxes("a cat<loc_212><loc_345><loc_600><loc_800>");
        assert_eq!(boxes, vec![GroundedBox { phrase: "a cat".to_string(), bbox: [0.212, 0.345, 0.600, 0.800] }]);
    }

    #[test]
    fn multiple_boxes_in_one_string() {
        let boxes = parse_boxes("cat<loc_0><loc_0><loc_100><loc_100>dog<loc_200><loc_200><loc_300><loc_300>");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].phrase, "cat");
        assert_eq!(boxes[0].bbox, [0.0, 0.0, 0.1, 0.1]);
        assert_eq!(boxes[1].phrase, "dog");
        assert_eq!(boxes[1].bbox, [0.2, 0.2, 0.3, 0.3]);
    }

    #[test]
    fn no_loc_tags_returns_empty() {
        assert_eq!(parse_boxes("just a caption with no boxes"), vec![]);
    }

    #[test]
    fn trailing_partial_run_is_dropped_not_errored() {
        // Truncated generation: only 2 of 4 expected loc tags present.
        let boxes = parse_boxes("cat<loc_1><loc_2>");
        assert_eq!(boxes, vec![]);
    }

    #[test]
    fn empty_phrase_before_first_box() {
        let boxes = parse_boxes("<loc_10><loc_20><loc_30><loc_40>");
        assert_eq!(boxes, vec![GroundedBox { phrase: String::new(), bbox: [0.01, 0.02, 0.03, 0.04] }]);
    }

    #[test]
    fn max_bin_999() {
        let boxes = parse_boxes("x<loc_999><loc_999><loc_999><loc_999>");
        assert_eq!(boxes[0].bbox, [0.999, 0.999, 0.999, 0.999]);
    }

    #[test]
    fn malformed_tag_is_skipped_not_infinite_looped() {
        // "<loc_>" (no digits) and "<loc_12" (unterminated) must not hang the scanner.
        let boxes = parse_boxes("a<loc_><loc_12b<loc_1><loc_2><loc_3><loc_4>");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].bbox, [0.001, 0.002, 0.003, 0.004]);
    }
}
