// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Global LLM: a real Qwen3-8B architecture (`hidden=4096, layers=36,
//! heads=32, kv_heads=8, head_dim=128, vocab=200000, rope_theta=1e6` -
//! confirmed against the checkpoint's own `language_model/config.json`,
//! NOT the smaller published Qwen3-8B's `vocab=151936` preset), reused
//! VERBATIM from `crates/qwen3` rather than reimplemented - see this
//! crate's own module doc. [`import`]'s own doc explains why
//! `language_model/`, specifically, and not the repository's OTHER
//! same-shaped language-model directory (`qwen_7B/qwen_7B/`, MiniMax's
//! own native, non-Qwen3, architecture). This module owns only two
//! things: streamed import (so the ~18 GB checkpoint never needs to be
//! resident in host RAM at once) and the audio-code-restricted training
//! objective this port adds on top of `crates/qwen3`'s already-
//! gradchecked forward/backward - not a second copy of Qwen3 itself.
//!
//! # Prompt / audio-code token contract
//!
//! The checkpoint's own special-token convention (confirmed against the
//! reference `diffusers` PR's own `MiniMaxMusic3TextEncoderStep`/
//! `MiniMaxMusic3SemanticGenerationStep` - even whitespace-level changes to
//! the assembled prompt change the generated audio, per that reference's
//! own comment): the conditional prompt is assembled as
//! `<|im_start|><|caption_start|>{caption}<|caption_end|><|lyrics_start|>
//! {lyrics}<|lyrics_end|><|im_end|><|audio_start|>`, then one token per
//! 25 Hz audio frame - a semantic RVQ code as vocab id
//! `AUDIO_CODE_OFFSET + code` (`code` in `[0, SEMANTIC_VOCAB_SIZE)`), fed
//! back into the SAME `embed_tokens`/`lm_head` any ordinary text token
//! uses (audio codes are plain extra vocab ids, not a separate embedding
//! space - `Qwen`'s own weights need no change to read or write them),
//! until `AUDIO_END_TOKEN_ID` or a frame cap. Classifier-free guidance is
//! NOT a `Qwen3`-level feature: the reference runs the SAME ordinary
//! `Qwen3ForCausalLM` forward on a 2-row `[conditional, unconditional]`
//! batch (the unconditional row replaces every prompt token except the
//! first and the two trailing structure tokens with `AUDIO_CFG_TOKEN_ID`)
//! and blends the two logit rows on the host - pure orchestration on top
//! of `crates/qwen3::Qwen`'s ordinary forward, not new model-level API.
//! That orchestration (prompt assembly text, the AR sampling loop, the
//! depth-decoder feedback) is `crate` M7 scope (pipeline glue); this
//! module owns only the constants the contract is built from, so both
//! this milestone's training objective and the next milestone's sampling
//! loop read them from one place.

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::{Qwen, QwenConfig};

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const CAPTION_START: &str = "<|caption_start|>";
pub const CAPTION_END: &str = "<|caption_end|>";
pub const LYRICS_START: &str = "<|lyrics_start|>";
pub const LYRICS_END: &str = "<|lyrics_end|>";
pub const AUDIO_START: &str = "<|audio_start|>";
pub const AUDIO_END_TOKEN_ID: u32 = 151670;
pub const AUDIO_CFG_TOKEN_ID: u32 = 151654;
pub const AUDIO_CODE_OFFSET: u32 = 151675;
pub const SEMANTIC_VOCAB_SIZE: u32 = 16384;
/// The reference inference recipe's own fixed sampling parameters (M7's
/// AR sampling loop uses these; recorded here alongside the token
/// contract they parameterize).
pub const AR_CFG_SCALE: f32 = 1.5;
pub const AR_CFG_TOP_K: usize = 50;
pub const AR_SAMPLING_TOP_K: usize = 50;

/// The vocab id one semantic RVQ code occupies in the Global LLM's own
/// `vocab=200000` space.
pub fn audio_code_token_id(code: u32) -> u32 {
    assert!(code < SEMANTIC_VOCAB_SIZE, "audio_code_token_id: code {code} out of range [0, {SEMANTIC_VOCAB_SIZE})");
    AUDIO_CODE_OFFSET + code
}

/// Build the AR stage's two token-id prompts (conditional and CFG-
/// unconditional) from a raw caption/lyrics pair - `MiniMaxMusic3PromptStep`
/// in the reference. Assembles `<|im_start|><|caption_start|>
/// {clean_caption(caption)}<|caption_end|><|lyrics_start|>
/// {normalize_lyrics(lyrics)}<|lyrics_end|><|im_end|><|audio_start|>`,
/// tokenizes it once (`QwenBpe::encode` already special-cases the literal
/// `<|...|>` tokens - see its own `encode_with_specials`), then builds the
/// unconditional variant by replacing every token except the first and the
/// two trailing structure tokens with `AUDIO_CFG_TOKEN_ID` (`ids[1:-2]` in
/// the reference).
pub fn assemble_prompt(tokenizer: &QwenBpe, caption: &str, lyrics: &str) -> (Vec<u32>, Vec<u32>) {
    let clean = clean_caption(caption);
    let lyr = normalize_lyrics(lyrics);
    let prompt = format!("{IM_START}{CAPTION_START}{clean}{CAPTION_END}{LYRICS_START}{lyr}{LYRICS_END}{IM_END}{AUDIO_START}");
    let conditional_ids = tokenizer.encode(&prompt);
    let unconditional_ids = unconditional_variant(&conditional_ids);
    (conditional_ids, unconditional_ids)
}

fn unconditional_variant(ids: &[u32]) -> Vec<u32> {
    let n = ids.len();
    assert!(n >= 3, "unconditional_variant: prompt has {n} tokens, need at least 3 (im_start + ... + im_end + audio_start)");
    let mut out = ids.to_vec();
    for id in &mut out[1..n - 2] {
        *id = AUDIO_CFG_TOKEN_ID;
    }
    out
}

/// Streamed import of the real Global LLM checkpoint: `dir` is the
/// `language_model/` subfolder (`config.json` + a 4-shard
/// `model.safetensors.index.json` set) - a genuine `Qwen3ForCausalLM`/
/// `model_type: "qwen3"` re-export, standard fields throughout
/// (`attention_bias: false`, `rope_theta: 1e6`, no residual-scaling
/// extras). The checkpoint's OTHER language-model directory,
/// `qwen_7B/qwen_7B/`, is NOT this - its own `config.json` reads
/// `"architectures": ["AbabForCausalLM"]`, `"model_type": "mixtral"`,
/// with per-layer LayerNorm alpha/beta residual-scaling constants no
/// plain Qwen3 decoder layer has - MiniMax's native training-checkpoint
/// format, a materially different architecture despite matching
/// `hidden=4096, layers=36, heads=32, kv_heads=8, head_dim=128,
/// vocab=200000` on the surface. Only `language_model/` is safe to load
/// through `crates/qwen3::Qwen` verbatim; `qwen_7B/qwen_7B/`'s weights
/// would either fail `qwen3::import::hf_to_brain`'s name mapping outright
/// or, worse, load silently wrong (the alpha/beta scaling `qwen3`'s own
/// forward has no code path for). The tokenizer lives at a THIRD
/// location, `qwen_7B/qwen3-8B-tokenizer-music/` (not under
/// `language_model/`, which ships no tokenizer files of its own).
///
/// `checkpoint::weightio::WeightReader::open_hf_dir` mmaps only shard
/// HEADERS up front; `qwen3::import::hf_source` then resolves brain
/// parameter names against those headers with zero tensor bytes read;
/// `Qwen::new_shard_i8` pulls one tensor at a time straight to the
/// device (CPU-JIT-backend host buffers on this machine, since there is
/// no discrete GPU), quantizing to int8 (DP4A) as it goes and dropping
/// each tensor's transient f32 expansion before the next - peak host RAM
/// stays at "one tensor", never the whole ~18 GB bf16 checkpoint (which
/// would expand past this machine's ~21 GB usable RAM at fp32). Int8 is
/// not merely smaller-and-nice-to-have here: it is what makes an 8B
/// model resident on this machine's CPU backend possible at all - the
/// same load-bearing role this crate's plan recorded for it going in.
/// Inference-only (matches [`qwen3::Qwen::new_shard_i8`]'s own scope);
/// the audio-code training objective below runs at
/// [`qwen3::QwenConfig::tiny`] scale instead, where fp32 + a real
/// backward pass both fit comfortably.
pub fn import(dir: &str, b: u32, t: u32) -> Result<(QwenConfig, Qwen), String> {
    let config_path = std::path::Path::new(dir).join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(|e| format!("global_llm::import: reading {}: {e}", config_path.display()))?;
    let cfg = qwen3::import::config_from_hf(&config_json)?;
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(dir)).map_err(|e| format!("global_llm::import: {e}"))?;
    let src = qwen3::import::hf_source(&reader, &cfg)?;
    let qwen = Qwen::new_shard_i8(cfg.clone(), b, t, &src, model::Shard::whole(cfg.n_layers as usize));
    Ok((cfg, qwen))
}

/// Build a `Batch::LmWeighted` triple (`tokens`, `targets`, `weights`)
/// training the Global LLM to predict ONLY the audio-code targets that
/// follow `prompt_ids` - the training objective this milestone adds:
/// ordinary next-token cross-entropy (`crates/qwen3`'s own, already
/// gradchecked), restricted by POSITION via `model::Batch::LmWeighted`'s
/// existing per-position gradient weight (`0.0` on every position whose
/// TARGET still falls inside the prompt, `1.0` once the target is the
/// first audio-code token and thereafter) rather than a new loss kernel
/// or a vocab-subset mask - the same "reuse the existing weighted-CE
/// seam" choice `model::Batch::LmWeighted`'s own doc anticipates for
/// exactly this kind of prompt-masked training.
///
/// `prompt_ids`/`audio_code_ids` are caller-assembled token id sequences
/// (this function does no prompt-text assembly or offset arithmetic
/// itself - see [`audio_code_token_id`] for the offset, and this crate's
/// M7 milestone for prompt-text assembly). Returns `(tokens, targets,
/// weights)`, each `prompt_ids.len() + audio_code_ids.len() - 1` long (a
/// `Batch::Lm`-style shifted pair over the whole concatenated sequence).
pub fn audio_code_batch(prompt_ids: &[u32], audio_code_ids: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut seq = prompt_ids.to_vec();
    seq.extend_from_slice(audio_code_ids);
    let n = seq.len();
    assert!(n >= 2, "audio_code_batch: prompt_ids + audio_code_ids must have at least 2 tokens total");
    let tokens = seq[..n - 1].to_vec();
    let targets = seq[1..].to_vec();
    let prompt_len = prompt_ids.len();
    // Position i's target is seq[i+1]; that target is an audio code iff
    // i+1 >= prompt_len (the first audio-code token sits at seq[prompt_len]).
    let weights: Vec<f32> = (0..n - 1).map(|i| if i + 1 >= prompt_len { 1.0 } else { 0.0 }).collect();
    (tokens, targets, weights)
}

// ---------------------------------------------------------------------------
// Caption / lyrics text normalization
//
// The reference `diffusers` PR's `MiniMaxMusic3TextEncoderStep` runs two pure
// text transforms over the user-supplied caption and lyrics before either
// ever reaches a tokenizer: `_clean_caption` (Markdown-strip a free-text
// caption and rewrite `<|key value|>` tags into "key is value" prose) and
// `_normalize_lyrics` (collapse bracket-tag lines like `[verse]` down to just
// the tag, discarding any trailing text on that line, then re-punctuate the
// tag/lyric boundaries with newlines). Both are hand-ported below using only
// `std` string/char methods, not the `regex` crate: this workspace has zero
// `regex` dependents anywhere, and the reference's italic-stripping rule
// needs lookaround (a star not immediately preceded or followed by another
// star) that Rust's linear-time `regex` crate cannot express at all - a
// faithful port would need a heavier backtracking engine this workspace has
// never pulled in. Every one of these functions is a manual left-to-right
// scan that mirrors what the reference's `re.sub`/`re.match` backtracking
// would actually do character by character, not an approximation of it -
// verified byte-for-byte against the real Python reference (see this
// module's own tests).

/// Rewrite every `<\|key value\|>` span into `"key is value"` prose, and
/// strip Markdown structure (headings, bullets, bold, italic, horizontal
/// rules) line by line. Ported from the reference's `_clean_caption`; used
/// verbatim on the free-text caption half of the prompt contract above -
/// even whitespace-level differences from the reference's own output change
/// the generated audio, so this is a byte-for-byte port, not a paraphrase.
pub fn clean_caption(caption: &str) -> String {
    let text = rewrite_special_tags(caption);
    let mut lines_out: Vec<String> = Vec::new();
    for line in split_lines_like_python(&text) {
        let l1 = strip_heading(line);
        let l2 = strip_bullet(l1, &['*', '+', '-']);
        // Second bullet pass: the reference re-runs a `*`-only bullet strip
        // on the already-stripped line. Almost always a no-op given the
        // first pass already covers `*`, but kept as its own pass (not
        // folded away) to match the reference's own double-pass exactly.
        let l3 = strip_bullet(l2, &['*']);
        let l4 = strip_bold(l3);
        let l5 = strip_italic(&l4);
        lines_out.push(l5.trim_end().to_string());
    }
    for line in lines_out.iter_mut() {
        if is_horizontal_rule(line) {
            line.clear();
        }
    }
    let joined = lines_out.join("\n");
    let replaced = joined.replace("\u{2022} ", "").replace("    ", "");
    collapse_blank_lines(&replaced)
}

/// Reduce every bracket-tag line (`[verse]`, `[verse][chorus]`, ...) down to
/// just its leading tag run (dropping any trailing lyric text on that same
/// line - the reference's own module doc calls this out explicitly), then
/// re-punctuate tag/lyric boundaries with newlines and lowercase every
/// bracket's content. Ported from the reference's `_normalize_lyrics`; used
/// verbatim on the lyrics half of the prompt contract above.
pub fn normalize_lyrics(lyrics: &str) -> String {
    let mut output: Vec<String> = Vec::new();
    for line in lyrics.split('\n') {
        match leading_tags_prefix(line) {
            Some(prefix) => output.push(prefix.trim().to_string()),
            None => output.push(line.to_string()),
        }
    }
    let mut text = output.join("\n");
    text = text.replace("] ", "]\n");
    text = text.replace(" [", "\n[");
    text = text.replace(" ^ ", "\n");
    text = lowercase_bracket_contents(&text);
    format!("[start]\n{text}")
}

/// Split on the same boundaries as Python's `str.splitlines()` (not just
/// `\n`): `\n`, `\r\n`, `\r`, and a handful of other Unicode line-break
/// characters, with the terminator dropped from each returned piece and no
/// trailing empty piece for a string that ends in a terminator. Used only by
/// [`clean_caption`] - [`normalize_lyrics`] instead uses the reference's
/// OTHER, plainer split (`str.split("\n")`, i.e. plain [`str::split`] here),
/// which keeps empty strings between consecutive `\n` and does not treat
/// `\r` as a boundary at all; the two functions are not interchangeable.
fn split_lines_like_python(s: &str) -> Vec<&str> {
    fn is_py_line_boundary(c: char) -> bool {
        matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}')
    }
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while idx < chars.len() {
        let (i, c) = chars[idx];
        if is_py_line_boundary(c) {
            result.push(&s[start..i]);
            idx += if c == '\r' && chars.get(idx + 1).map(|&(_, c2)| c2) == Some('\n') { 2 } else { 1 };
            start = chars.get(idx).map(|&(j, _)| j).unwrap_or(s.len());
            continue;
        }
        idx += 1;
    }
    if start < s.len() {
        result.push(&s[start..]);
    }
    result
}

/// Rewrite every `<|inner|>` span (shortest run between `<|` and `|>` with no
/// `|` inside, matched left to right, non-overlapping) into `"key is rest"`
/// prose - `inner`, trimmed, split on the FIRST whitespace run into at most
/// two pieces; one piece (0 or 1 after the split) passes the trimmed inner
/// text through unchanged.
fn rewrite_special_tags(caption: &str) -> String {
    let mut out = String::with_capacity(caption.len());
    let bytes = caption.as_bytes();
    let len = caption.len();
    let mut i = 0usize;
    while i < len {
        if bytes[i] == b'<' && i + 1 < len && bytes[i + 1] == b'|' {
            if let Some(rel) = caption[i + 2..].find('|') {
                let pipe_pos = i + 2 + rel;
                if pipe_pos + 1 < len && bytes[pipe_pos + 1] == b'>' {
                    out.push_str(&rewrite_tag_inner(&caption[i + 2..pipe_pos]));
                    i = pipe_pos + 2;
                    continue;
                }
            }
        }
        let ch_len = caption[i..].chars().next().expect("i < len implies a char remains").len_utf8();
        out.push_str(&caption[i..i + ch_len]);
        i += ch_len;
    }
    out
}

fn rewrite_tag_inner(raw: &str) -> String {
    let inner = raw.trim();
    match split_whitespace_once(inner) {
        Some((first, rest)) => format!("{first} is {rest}"),
        None => inner.to_string(),
    }
}

/// Python's `str.split(None, 1)`: split on the first run of whitespace,
/// dropping the run itself, keeping everything after it (including any
/// further internal whitespace) as the second piece verbatim.
fn split_whitespace_once(s: &str) -> Option<(&str, &str)> {
    let ws_start = s.char_indices().find(|(_, c)| c.is_whitespace())?.0;
    let rest_start = s[ws_start..].find(|c: char| !c.is_whitespace()).map(|off| ws_start + off)?;
    Some((&s[..ws_start], &s[rest_start..]))
}

/// Byte offset just past the maximal run of `' '`/`'\t'` starting at `from`.
fn space_tab_run_end(s: &str, from: usize) -> usize {
    let mut end = from;
    for c in s[from..].chars() {
        if c == ' ' || c == '\t' {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Byte offset just past the maximal run of Unicode whitespace starting at
/// `from`.
fn whitespace_run_end(s: &str, from: usize) -> usize {
    let mut end = from;
    for c in s[from..].chars() {
        if c.is_whitespace() {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Strip a leading `^\s{0,3}#{1,6}\s+` (a Markdown heading marker) if
/// present; otherwise return `line` unchanged. At most 3 leading whitespace
/// chars, then 1-6 `#` chars (not more - 7+ never matches, since the
/// mandatory trailing `\s+` can never find whitespace while more `#` remain),
/// then 1+ whitespace chars, all removed together.
fn strip_heading(line: &str) -> &str {
    let ws_end = whitespace_run_end(line, 0);
    if line[..ws_end].chars().count() > 3 {
        return line;
    }
    let rest = &line[ws_end..];
    let hash_count = rest.chars().take_while(|&c| c == '#').count();
    if !(1..=6).contains(&hash_count) {
        return line;
    }
    let after_hash = &rest[hash_count..]; // '#' is always 1 byte
    let trailing_ws_end = whitespace_run_end(after_hash, 0);
    if trailing_ws_end < 1 {
        return line;
    }
    &after_hash[trailing_ws_end..]
}

/// Strip a leading `^\s*C\s+` where `C` is any char in `bullet_chars` -
/// covers the reference's two bullet passes (`[*+-]` then `*`-only).
fn strip_bullet<'a>(line: &'a str, bullet_chars: &[char]) -> &'a str {
    let ws_end = whitespace_run_end(line, 0);
    let rest = &line[ws_end..];
    let mut chars = rest.chars();
    let Some(first) = chars.next() else { return line };
    if !bullet_chars.contains(&first) {
        return line;
    }
    let after_bullet = &rest[first.len_utf8()..];
    let trailing_ws_end = whitespace_run_end(after_bullet, 0);
    if trailing_ws_end < 1 {
        return line;
    }
    &after_bullet[trailing_ws_end..]
}

/// Repeatedly replace `**<non-`*`content>**` with its content (a `while`
/// loop over full left-to-right passes, matching the reference's own
/// `while "**" in line: line = re.sub(...)` - stops once a pass makes no
/// change or no `**` remains).
fn strip_bold(line: &str) -> String {
    let mut current = line.to_string();
    while current.contains("**") {
        let (updated, changed) = bold_pass(&current);
        if !changed {
            break;
        }
        current = updated;
    }
    current
}

/// One left-to-right, non-overlapping pass replacing every
/// `\*\*([^*]+)\*\*` with its captured content.
fn bold_pass(line: &str) -> (String, bool) {
    let bytes = line.as_bytes();
    let len = line.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;
    let mut changed = false;
    while i < len {
        if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'*' {
            if let Some(rel) = line[i + 2..].find('*') {
                let close = i + 2 + rel;
                if close > i + 2 && close + 1 < len && bytes[close + 1] == b'*' {
                    out.push_str(&line[i + 2..close]);
                    i = close + 2;
                    changed = true;
                    continue;
                }
            }
        }
        let ch_len = line[i..].chars().next().expect("i < len implies a char remains").len_utf8();
        out.push_str(&line[i..i + ch_len]);
        i += ch_len;
    }
    (out, changed)
}

/// Single pass replacing every `\*([^*\n]+)\*` NOT immediately preceded or
/// followed by another `*` with its captured content - the reference's own
/// lookaround (`(?<!\*)...(?!\*)`), hand-rolled: since content may not
/// contain `*`, the only candidate closing star is the first `*` reached
/// after the opening one, so there is nothing to backtrack into - either
/// that candidate satisfies both lookarounds or the whole attempt at this
/// start position fails outright and the scan moves on one char.
fn strip_italic(line: &str) -> String {
    let bytes = line.as_bytes();
    let len = line.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;
    while i < len {
        if bytes[i] == b'*' && (i == 0 || bytes[i - 1] != b'*') {
            if let Some(rel) = line[i + 1..].find('*') {
                let close = i + 1 + rel;
                if close > i + 1 && !(close + 1 < len && bytes[close + 1] == b'*') {
                    out.push_str(&line[i + 1..close]);
                    i = close + 1;
                    continue;
                }
            }
        }
        let ch_len = line[i..].chars().next().expect("i < len implies a char remains").len_utf8();
        out.push_str(&line[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// `^\s*[-*_]{3,}\s*$` over an entire line: only whitespace, then 3+ chars
/// each `-`/`*`/`_` in any mix, then only whitespace.
fn is_horizontal_rule(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && trimmed.chars().count() >= 3 && trimmed.chars().all(|c| c == '-' || c == '*' || c == '_')
}

/// Collapse every run of 2+ consecutive `\n` into exactly one `\n`
/// (equivalent to collapsing every run of 1+ into one, since a lone `\n`
/// maps to itself either way).
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '\n' {
            while chars.peek() == Some(&'\n') {
                chars.next();
            }
        }
    }
    out
}

/// `^[ \t]*((?:\[[^\]]+\][ \t]*)+)`, `.match()`-style (anchored at the
/// start, need not consume the whole line): one or more consecutive
/// `[non-empty, non-`]`-content]` units, each optionally followed by
/// spaces/tabs, after skipping leading spaces/tabs - returns the matched
/// group (trimmed), or `None` if the line does not start with at least one
/// such bracket unit.
fn leading_tags_prefix(line: &str) -> Option<&str> {
    let group_start = space_tab_run_end(line, 0);
    let mut cur = group_start;
    let mut matched_end: Option<usize> = None;
    loop {
        if cur >= line.len() || line.as_bytes()[cur] != b'[' {
            break;
        }
        let Some(rel) = line[cur + 1..].find(']') else { break };
        let close = cur + 1 + rel;
        if close == cur + 1 {
            break; // empty `[]` content - [^\]]+ needs at least one char
        }
        let next = space_tab_run_end(line, close + 1);
        matched_end = Some(next);
        cur = next;
    }
    matched_end.map(|end| line[group_start..end].trim())
}

/// Lowercase the content of every `[...]` bracket pair (non-nested, content
/// may not contain `]`); brackets whose content is empty (`[]`) are left
/// untouched since the reference's `[^\]]+` requires at least one char.
fn lowercase_bracket_contents(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = s.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;
    while i < len {
        if bytes[i] == b'[' {
            if let Some(rel) = s[i + 1..].find(']') {
                let close = i + 1 + rel;
                if close > i + 1 {
                    out.push('[');
                    out.push_str(&s[i + 1..close].to_lowercase());
                    out.push(']');
                    i = close + 1;
                    continue;
                }
            }
        }
        let ch_len = s[i..].chars().next().expect("i < len implies a char remains").len_utf8();
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;
    use model::{Batch, Model};

    #[test]
    fn audio_code_token_id_offsets_into_the_real_vocab_range() {
        assert_eq!(audio_code_token_id(0), AUDIO_CODE_OFFSET);
        assert_eq!(audio_code_token_id(SEMANTIC_VOCAB_SIZE - 1), AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE - 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn audio_code_token_id_rejects_an_out_of_range_code() {
        audio_code_token_id(SEMANTIC_VOCAB_SIZE);
    }

    #[test]
    fn unconditional_variant_keeps_only_the_first_and_last_two_tokens() {
        let ids = vec![10u32, 11, 12, 13, 14, 15];
        let uncond = unconditional_variant(&ids);
        assert_eq!(uncond, vec![10, AUDIO_CFG_TOKEN_ID, AUDIO_CFG_TOKEN_ID, AUDIO_CFG_TOKEN_ID, 14, 15]);
        // The conditional ids themselves must be untouched.
        assert_eq!(ids, vec![10, 11, 12, 13, 14, 15]);
    }

    #[test]
    #[should_panic(expected = "need at least 3")]
    fn unconditional_variant_rejects_a_too_short_prompt() {
        unconditional_variant(&[1, 2]);
    }

    #[test]
    fn audio_code_batch_masks_out_every_prompt_position() {
        let prompt = vec![1u32, 2, 3, 4, 5];
        let audio = vec![100u32, 101, 102];
        let (tokens, targets, weights) = audio_code_batch(&prompt, &audio);
        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 100, 101]);
        assert_eq!(targets, vec![2, 3, 4, 5, 100, 101, 102]);
        // targets[0..4] are still prompt tokens (2,3,4,5) -> weight 0;
        // targets[4..] are the 3 audio codes (100,101,102) -> weight 1.
        assert_eq!(weights, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    /// The wiring this milestone actually adds: `Batch::LmWeighted`
    /// masking restricted to audio-code target positions must be
    /// trainable end to end, not just structurally well-formed - plain
    /// AdamW at `QwenConfig::tiny()` scale (where a real backward pass is
    /// cheap) must collapse the loss on a single fixed batch. `crates/
    /// qwen3` itself already gradchecks `LmWeighted`'s own gradient
    /// (weight=1 reproduces `Batch::Lm`'s gradient exactly, weight=0
    /// contributes exactly zero); this test proves only that THIS
    /// module's own position-masking construction is correct, not a
    /// second gradcheck of `qwen3`'s backward.
    #[test]
    fn audio_code_ce_training_overfits_a_single_batch() {
        let cfg = QwenConfig::tiny();
        let init = qwen3::init_weights(&cfg, 41);
        let mut r = Lcg::new(42);
        let prompt: Vec<u32> = (0..6).map(|_| r.next_u32() % cfg.vocab).collect();
        let audio: Vec<u32> = (0..4).map(|_| r.next_u32() % cfg.vocab).collect();
        let (tokens, targets, weights) = audio_code_batch(&prompt, &audio);

        let mut qwen = Qwen::new(cfg.clone(), 1, tokens.len() as u32, &init);
        qwen.enable_weighted_loss();
        Model::set_batch(&qwen, Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights });

        let loss0 = Model::forward(&qwen);
        let mut loss = loss0;
        for step in 1..=300u32 {
            Model::zero_grads(&qwen);
            Model::set_batch(&qwen, Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights });
            loss = Model::forward(&qwen);
            Model::backward(&qwen);
            Model::adamw_step(&qwen, step, 5e-2, 0.0, Some(1.0), 1.0);
            Model::poll_wait(&qwen);
        }
        assert!(loss < loss0 * 0.1, "audio-code CE training did not collapse the loss: start={loss0} end={loss} (300 steps)");
    }

    // clean_caption / normalize_lyrics: every expected string below is the
    // VERBATIM output of the reference Python (`_clean_caption`/
    // `_normalize_lyrics`, stdlib `re` only) run against the matching input,
    // captured directly from a throwaway interpreter session - not
    // hand-derived. Any mismatch here means this file's manual scan has
    // diverged from the reference's regex semantics, not that the expected
    // value needs adjusting.

    #[test]
    fn clean_caption_matches_the_python_reference() {
        assert_eq!(clean_caption(""), "");
        assert_eq!(clean_caption("# Heading\nSome text"), "Heading\nSome text");
        assert_eq!(clean_caption("  # indented heading\ntext"), "indented heading\ntext");
        assert_eq!(clean_caption("* bullet one\n+ bullet two\n- bullet three"), "bullet one\nbullet two\nbullet three");
        assert_eq!(clean_caption("**bold text** normal"), "bold text normal");
        assert_eq!(clean_caption("*italic text* normal"), "italic text normal");
        assert_eq!(clean_caption("**bold *nested italic* inside**"), "**bold nested italic inside**");
        assert_eq!(clean_caption("*italic **nested bold** inside*"), "italic nested bold inside");
        assert_eq!(clean_caption("---\ntext\n***\nmore\n___\nend"), "\ntext\nmore\nend");
        assert_eq!(clean_caption("line1\n\n\nline2"), "line1\nline2");
        assert_eq!(clean_caption("<|mood happy|>"), "mood is happy");
        assert_eq!(clean_caption("<|genre pop rock ballad|>"), "genre is pop rock ballad");
        assert_eq!(clean_caption("<|justoneword|>"), "justoneword");
        assert_eq!(
            clean_caption("text <|tempo fast|> more text <|key C major|> end"),
            "text tempo is fast more text key is C major end"
        );
        assert_eq!(clean_caption("\u{2022} bullet point\n\u{2022}  another"), "bullet point\n another");
        assert_eq!(clean_caption("H\u{e9}llo w\u{f6}rld caf\u{e9} *\u{e9}motion*"), "H\u{e9}llo w\u{f6}rld caf\u{e9} \u{e9}motion");
        assert_eq!(clean_caption("text with * unmatched asterisk"), "text with * unmatched asterisk");
        assert_eq!(clean_caption("*a* *b* *c*"), "a b c");
        assert_eq!(clean_caption("# Title **bold** *italic* [tag]\nnext"), "Title bold italic [tag]\nnext");
    }

    #[test]
    fn normalize_lyrics_matches_the_python_reference() {
        assert_eq!(normalize_lyrics(""), "[start]\n");
        assert_eq!(normalize_lyrics("[verse]\nHello world"), "[start]\n[verse]\nHello world");
        assert_eq!(normalize_lyrics("[verse][chorus]\nHello"), "[start]\n[verse][chorus]\nHello");
        assert_eq!(normalize_lyrics("[Verse 1] some lyrics here"), "[start]\n[verse 1]");
        assert_eq!(normalize_lyrics("[verse]   [chorus]  \ntext"), "[start]\n[verse]\n \n[chorus]\ntext");
        assert_eq!(normalize_lyrics("no tags here just text"), "[start]\nno tags here just text");
        assert_eq!(normalize_lyrics("[VERSE] Line one\n[Chorus] Line two"), "[start]\n[verse]\n[chorus]");
        assert_eq!(normalize_lyrics("a ] b [ c ^ d"), "[start]\na ]\nb\n[ c\nd");
        assert_eq!(normalize_lyrics("[tag] text ] after [ more ^ split"), "[start]\n[tag]");
        assert_eq!(normalize_lyrics("[a][b][c] trailing text dropped"), "[start]\n[a][b][c]");
        assert_eq!(normalize_lyrics("\n\n[verse]\ntext\n\n"), "[start]\n\n\n[verse]\ntext\n\n");
    }
}
