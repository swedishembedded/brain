// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `normalizer` and `pre_tokenizer` of a Hugging Face `tokenizer.json`,
//! read as declared and applied the way the `tokenizers` library applies them.
//!
//! A byte-level BPE vocabulary is only half a tokenizer: the pre-tokenizer
//! decides where BPE may merge across, and two checkpoints sharing a merge
//! table tokenize differently when their splits differ. Every shape a file can
//! declare is either implemented here or refused by name at load - a pipeline
//! step that is silently skipped produces valid-looking ids for a different
//! tokenization.
//!
//! | step | declared as | effect |
//! |---|---|---|
//! | NFC | `normalizer: {"type": "NFC"}` | canonical composition |
//! | Split | `{"type": "Split", "pattern": {"Regex"/"String"}, "behavior", "invert"}` | splits every piece at the pattern's matches |
//! | Digits | `{"type": "Digits", "individual_digits"}` | isolates each numeric char, or each run |
//! | ByteLevel | `{"type": "ByteLevel", "add_prefix_space", "use_regex"}` | the GPT-2 split when `use_regex`; the byte mapping itself is the BPE model's |
//! | Sequence | `{"type": "Sequence", ...}` | each step re-splits every piece of the one before |
//!
//! A GGUF names its pre-tokenizer (`tokenizer.ggml.pre`) instead of declaring
//! it; [`for_gguf`] maps the names brain serves to the split sequence
//! llama.cpp runs for each.
//!
//! Swedish Embedded AB implements tokenizer parity for its clients' inference
//! engines. If your team needs text encoded exactly as a checkpoint was
//! trained on it, you can procure our services by sending an email to
//! info@swedishembedded.com.

use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// A `tokenizer.json` normalizer.
#[derive(Clone, Debug)]
pub enum Normalizer {
    Nfc,
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    /// Read `v` (`null` for none). An unimplemented type is an error naming it.
    pub fn from_json(v: &Value) -> Result<Option<Normalizer>, String> {
        if v.is_null() {
            return Ok(None);
        }
        match v["type"].as_str() {
            Some("NFC") => Ok(Some(Normalizer::Nfc)),
            Some("Sequence") => {
                let steps = v["normalizers"].as_array().ok_or("normalizer Sequence: no `normalizers`")?;
                let steps: Vec<Normalizer> = steps.iter().map(Normalizer::from_json).filter_map(Result::transpose).collect::<Result<_, _>>()?;
                Ok((!steps.is_empty()).then_some(Normalizer::Sequence(steps)))
            }
            other => Err(format!("tokenizer.json: normalizer {other:?} is not implemented")),
        }
    }

    pub fn apply(&self, text: &str) -> String {
        match self {
            Normalizer::Nfc => text.nfc().collect(),
            Normalizer::Sequence(steps) => steps.iter().fold(text.to_string(), |t, n| n.apply(&t)),
        }
    }
}

/// How a [`PreTokenizer::Split`] treats the delimiters its pattern finds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

/// What a [`PreTokenizer::Split`] looks for.
#[derive(Clone, Debug)]
pub enum Pattern {
    Regex(fancy_regex::Regex),
    Literal(String),
}

/// A `tokenizer.json` pre-tokenizer.
#[derive(Clone, Debug)]
pub enum PreTokenizer {
    Split { pattern: Pattern, behavior: SplitBehavior, invert: bool },
    Digits { individual: bool },
    ByteLevel { add_prefix_space: bool, use_regex: bool },
    Sequence(Vec<PreTokenizer>),
}

/// GPT-2's pattern, what `ByteLevel` splits by when `use_regex` is set.
const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Qwen2's pattern: its `tokenizer.json` split, and llama.cpp's `qwen2`.
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl PreTokenizer {
    /// Read `v` (`null` is no pre-tokenization: the whole text is one piece).
    /// An unimplemented type is an error naming it.
    pub fn from_json(v: &Value) -> Result<PreTokenizer, String> {
        if v.is_null() {
            return Ok(PreTokenizer::Sequence(Vec::new()));
        }
        match v["type"].as_str() {
            Some("Sequence") => {
                let steps = v["pretokenizers"].as_array().ok_or("pre_tokenizer Sequence: no `pretokenizers`")?;
                Ok(PreTokenizer::Sequence(steps.iter().map(PreTokenizer::from_json).collect::<Result<_, _>>()?))
            }
            Some("Split") => {
                let pattern = match (v["pattern"]["Regex"].as_str(), v["pattern"]["String"].as_str()) {
                    (Some(re), _) => Pattern::Regex(regex(re)?),
                    (None, Some(s)) => Pattern::Literal(s.to_string()),
                    _ => return Err(format!("pre_tokenizer Split: unreadable pattern {}", v["pattern"])),
                };
                let behavior = match v["behavior"].as_str() {
                    Some("Removed") => SplitBehavior::Removed,
                    Some("Isolated") => SplitBehavior::Isolated,
                    Some("MergedWithPrevious") => SplitBehavior::MergedWithPrevious,
                    Some("MergedWithNext") => SplitBehavior::MergedWithNext,
                    Some("Contiguous") => SplitBehavior::Contiguous,
                    other => return Err(format!("pre_tokenizer Split: behavior {other:?} is not implemented")),
                };
                Ok(PreTokenizer::Split { pattern, behavior, invert: v["invert"].as_bool().unwrap_or(false) })
            }
            Some("Digits") => Ok(PreTokenizer::Digits { individual: v["individual_digits"].as_bool().unwrap_or(false) }),
            Some("ByteLevel") => Ok(PreTokenizer::ByteLevel {
                add_prefix_space: v["add_prefix_space"].as_bool().unwrap_or(true),
                use_regex: v["use_regex"].as_bool().unwrap_or(true),
            }),
            other => Err(format!("tokenizer.json: pre_tokenizer {other:?} is not implemented")),
        }
    }

    /// Isolated splits by each pattern in turn: the shape llama.cpp gives
    /// every named pre-tokenizer.
    pub fn isolated_splits(patterns: &[&str]) -> Result<PreTokenizer, String> {
        Ok(PreTokenizer::Sequence(
            patterns.iter().map(|p| Ok(PreTokenizer::Split { pattern: Pattern::Regex(regex(p)?), behavior: SplitBehavior::Isolated, invert: false })).collect::<Result<_, String>>()?,
        ))
    }

    /// Split `text` into the pieces BPE encodes separately.
    pub fn split(&self, text: &str) -> Vec<String> {
        let mut pieces = vec![text.to_string()];
        self.apply(&mut pieces);
        pieces.retain(|p| !p.is_empty());
        pieces
    }

    fn apply(&self, pieces: &mut Vec<String>) {
        match self {
            PreTokenizer::Sequence(steps) => steps.iter().for_each(|s| s.apply(pieces)),
            PreTokenizer::Split { pattern, behavior, invert } => {
                *pieces = pieces.iter().flat_map(|p| split_piece(p, &spans(p, pattern), *behavior, *invert)).collect();
            }
            PreTokenizer::Digits { individual } => {
                let behavior = if *individual { SplitBehavior::Isolated } else { SplitBehavior::Contiguous };
                *pieces = pieces.iter().flat_map(|p| split_piece(p, &char_spans(p, char::is_numeric), behavior, false)).collect();
            }
            PreTokenizer::ByteLevel { add_prefix_space, use_regex } => {
                if *add_prefix_space {
                    for p in pieces.iter_mut().filter(|p| !p.starts_with(' ')) {
                        p.insert(0, ' ');
                    }
                }
                if *use_regex {
                    let gpt2 = Pattern::Regex(regex(GPT2_PATTERN).expect("GPT-2 pattern compiles"));
                    *pieces = pieces.iter().flat_map(|p| split_piece(p, &spans(p, &gpt2), SplitBehavior::Isolated, false)).collect();
                }
            }
        }
    }
}

/// The pre-tokenizer llama.cpp runs for a GGUF's `tokenizer.ggml.pre`, and
/// whether it encodes a word already in the vocabulary whole
/// (`ignore_merges`). `None` - a GGUF that names none - keeps the Qwen2
/// split the byte-level GGUF path has always used; any other unknown name is
/// refused rather than tokenized by a guess.
pub fn for_gguf(pre: Option<&str>) -> Result<(PreTokenizer, bool), String> {
    const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DEEPSEEK_LLM: [&str; 6] = [
        "[\r\n]",
        "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
        "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
        "\\s+$",
        "[一-龥ࠀ-一가-퟿]+",
        "\\p{N}+",
    ];
    const DEEPSEEK_CODER: [&str; 5] = ["[\r\n]", "\\s?\\p{L}+", "\\s?\\p{P}+", "[一-龥ࠀ-一가-퟿]+", "\\p{N}"];
    const DEEPSEEK_V3: [&str; 3] = [
        "\\p{N}{1,3}",
        "[一-龥぀-ゟ゠-ヿ]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    ];
    let (patterns, ignore_merges): (&[&str], bool) = match pre {
        None | Some("qwen2") => (&[QWEN2_PATTERN], false),
        Some("llama3" | "llama-v3" | "llama-bpe" | "lfm2") => (&[LLAMA3], true),
        Some("deepseek-llm") => (&DEEPSEEK_LLM, false),
        Some("deepseek-coder") => (&DEEPSEEK_CODER, false),
        Some("deepseek-v3") => (&DEEPSEEK_V3, false),
        Some("gpt-2") => (&[GPT2_PATTERN], false),
        Some(other) => return Err(format!("gguf tokenizer: pre-tokenizer {other:?} is not implemented")),
    };
    Ok((PreTokenizer::isolated_splits(patterns)?, ignore_merges))
}

fn regex(pattern: &str) -> Result<fancy_regex::Regex, String> {
    fancy_regex::Regex::new(pattern).map_err(|e| format!("pre_tokenizer pattern {pattern:?}: {e}"))
}

/// `text` cut into `(byte range, is a match)` spans that cover it exactly.
fn spans(text: &str, pattern: &Pattern) -> Vec<(std::ops::Range<usize>, bool)> {
    let mut matches: Vec<std::ops::Range<usize>> = Vec::new();
    match pattern {
        Pattern::Regex(re) => {
            // A pattern that fails at run time (backtrack limit) leaves the
            // piece whole rather than aborting the encode.
            for m in re.find_iter(text).flatten() {
                if !m.range().is_empty() {
                    matches.push(m.range());
                }
            }
        }
        Pattern::Literal(s) if !s.is_empty() => matches.extend(text.match_indices(s.as_str()).map(|(i, m)| i..i + m.len())),
        Pattern::Literal(_) => {}
    }
    cover(text.len(), matches)
}

/// One span per char satisfying `pred`, covering `text` with the gaps.
fn char_spans(text: &str, pred: fn(char) -> bool) -> Vec<(std::ops::Range<usize>, bool)> {
    cover(text.len(), text.char_indices().filter(|(_, c)| pred(*c)).map(|(i, c)| i..i + c.len_utf8()).collect())
}

fn cover(len: usize, matches: Vec<std::ops::Range<usize>>) -> Vec<(std::ops::Range<usize>, bool)> {
    let mut out = Vec::with_capacity(2 * matches.len() + 1);
    let mut at = 0;
    for m in matches {
        if m.start > at {
            out.push((at..m.start, false));
        }
        at = m.end;
        out.push((m, true));
    }
    if at < len {
        out.push((at..len, false));
    }
    out
}

/// Apply `behavior` to `text`'s spans, the matches being the delimiters (or,
/// with `invert`, the gaps between them).
fn split_piece(text: &str, spans: &[(std::ops::Range<usize>, bool)], behavior: SplitBehavior, invert: bool) -> Vec<String> {
    let spans = spans.iter().map(|(r, m)| (r.clone(), m ^ invert));
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    let mut previous_delim = false;
    match behavior {
        SplitBehavior::Removed => out.extend(spans.filter(|(_, d)| !d).map(|(r, _)| r)),
        SplitBehavior::Isolated => out.extend(spans.map(|(r, _)| r)),
        SplitBehavior::MergedWithPrevious => {
            for (r, d) in spans {
                match out.last_mut() {
                    Some(last) if d && !previous_delim => last.end = r.end,
                    _ => out.push(r),
                }
                previous_delim = d;
            }
        }
        SplitBehavior::MergedWithNext => {
            let all: Vec<_> = spans.collect();
            for (r, d) in all.into_iter().rev() {
                match out.last_mut() {
                    Some(last) if d && !previous_delim => last.start = r.start,
                    _ => out.push(r),
                }
                previous_delim = d;
            }
            out.reverse();
        }
        SplitBehavior::Contiguous => {
            for (r, d) in spans {
                match out.last_mut() {
                    Some(last) if d == previous_delim => last.end = r.end,
                    _ => out.push(r),
                }
                previous_delim = d;
            }
        }
    }
    out.into_iter().map(|r| text[r].to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn split(pattern: &str, behavior: &str, invert: bool, text: &str) -> Vec<String> {
        PreTokenizer::from_json(&json!({"type": "Split", "pattern": {"Regex": pattern}, "behavior": behavior, "invert": invert})).unwrap().split(text)
    }

    /// Every behavior, both ways round, as the `tokenizers` release splits
    /// the same strings.
    #[test]
    fn split_behaviors_match_the_reference() {
        let cases: [(&str, &[&str], &[&str]); 5] = [
            ("Removed", &["a", "b", "c"], &["-", "-", "-"]),
            ("Isolated", &["a", "--", "b", "x", "x", "-", "c"], &["a", "-", "-", "b", "-", "c"]),
            ("MergedWithPrevious", &["a--", "bx", "x", "-", "c"], &["a", "-", "-b", "-c"]),
            ("MergedWithNext", &["a", "--b", "x", "x", "-c"], &["a-", "-", "b-", "c"]),
            ("Contiguous", &["a", "--", "b", "xx-", "c"], &["a", "--", "b", "-", "c"]),
        ];
        for (behavior, plain, inverted) in cases {
            assert_eq!(split("-+|x", behavior, false, "a--bxx-c"), plain, "{behavior}");
            assert_eq!(split("-", behavior, true, "a--b-c"), inverted, "{behavior} inverted");
        }
    }

    #[test]
    fn digits_isolate_every_numeric_char_or_each_run() {
        let digits = |individual| PreTokenizer::from_json(&json!({"type": "Digits", "individual_digits": individual})).unwrap();
        assert_eq!(digits(true).split("a12٣½Ⅻ③b"), ["a", "1", "2", "٣", "½", "Ⅻ", "③", "b"]);
        assert_eq!(digits(false).split("a12٣½Ⅻ③b"), ["a", "12٣½Ⅻ③", "b"]);
    }

    #[test]
    fn byte_level_prefixes_each_piece_and_splits_by_gpt2() {
        let bl = PreTokenizer::from_json(&json!({"type": "ByteLevel", "add_prefix_space": true, "use_regex": true})).unwrap();
        assert_eq!(bl.split("hi there  you"), [" hi", " there", " ", " you"]);
        let seq = PreTokenizer::from_json(&json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": " "}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": true, "use_regex": false}]}))
        .unwrap();
        assert_eq!(seq.split("a b"), [" a", " ", " b"]);
    }

    #[test]
    fn an_unimplemented_step_is_refused_by_name() {
        let e = PreTokenizer::from_json(&json!({"type": "Metaspace"})).unwrap_err();
        assert!(e.contains("Metaspace"), "{e}");
        let e = Normalizer::from_json(&json!({"type": "NFKC"})).unwrap_err();
        assert!(e.contains("NFKC"), "{e}");
        assert!(for_gguf(Some("falcon")).unwrap_err().contains("falcon"));
    }

    #[test]
    fn nfc_composes() {
        let n = Normalizer::from_json(&json!({"type": "NFC"})).unwrap().unwrap();
        assert_eq!(n.apply("e\u{301}"), "é");
        assert!(Normalizer::from_json(&json!({"type": "Sequence", "normalizers": []})).unwrap().is_none());
    }
}
