// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Word error rate for speech recognition and speech round trips.
//!
//! Brain implements speech synthesis and recognition; whether an utterance
//! survives synthesis followed by recognition is measured with the functions
//! here, so every caller scores transcripts identically.

/// Word-level edit distance (substitutions, insertions, deletions) between a
/// reference and a hypothesis. Words are compared case-insensitively;
/// punctuation should already be stripped by [`normalize`].
pub fn word_edits(reference: &str, hypothesis: &str) -> usize {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hypothesis.split_whitespace().collect();
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, hw) in h.iter().enumerate() {
            let sub = prev[j] + usize::from(!rw.eq_ignore_ascii_case(hw));
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[h.len()]
}

/// Word error rate of one utterance: edits over reference length. An empty
/// reference scores 0 against an empty hypothesis and 1 against anything else.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let n = reference.split_whitespace().count();
    if n == 0 {
        return if hypothesis.split_whitespace().next().is_none() { 0.0 } else { 1.0 };
    }
    word_edits(reference, hypothesis) as f32 / n as f32
}

/// Strip punctuation and case: what a WER comparison should ignore.
pub fn normalize(s: &str) -> String {
    let cleaned: String = s.chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Corpus word error rate over `(reference, hypothesis)` pairs: total edits
/// over total reference words, after [`normalize`]. Long utterances weigh
/// more than short ones, unlike a mean of per-utterance rates. `None` when the
/// references hold no words, because an unmeasured rate is not zero.
pub fn corpus_wer<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<f32> {
    let (mut edits, mut words) = (0usize, 0usize);
    for (reference, hypothesis) in pairs {
        let reference = normalize(reference);
        edits += word_edits(&reference, &normalize(hypothesis));
        words += reference.split_whitespace().count();
    }
    (words > 0).then(|| edits as f32 / words as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_utterance_rate_matches_known_cases() {
        assert_eq!(word_error_rate("the cat sat", "the cat sat"), 0.0);
        assert_eq!(word_error_rate("the cat sat", "the cat"), 1.0 / 3.0);
        assert_eq!(word_error_rate("the cat sat", "the dog sat"), 1.0 / 3.0);
        assert_eq!(word_error_rate("THE Cat Sat", "the cat sat"), 0.0, "case-insensitive");
    }

    #[test]
    fn empty_reference_is_all_or_nothing() {
        assert_eq!(word_error_rate("", ""), 0.0);
        assert_eq!(word_error_rate("", "spurious"), 1.0);
    }

    #[test]
    fn normalize_ignores_case_and_punctuation() {
        assert_eq!(normalize("  The Quick, brown fox -- jumps! "), "the quick brown fox jumps");
    }

    #[test]
    fn corpus_rate_is_total_edits_over_total_reference_words_not_a_mean_of_rates() {
        // 1 edit over 2 words, then 1 edit over 8 words: the mean of the two
        // rates is 0.3125 but the corpus rate is 2 edits / 10 words.
        let pairs = [("a b", "a"), ("a b c d e f g h", "a b c d e f g x")];
        assert_eq!(corpus_wer(pairs), Some(0.2));
    }

    #[test]
    fn corpus_rate_is_absent_without_reference_words() {
        assert_eq!(corpus_wer(Vec::<(&str, &str)>::new()), None);
        assert_eq!(corpus_wer([("", "x")]), None);
    }
}
