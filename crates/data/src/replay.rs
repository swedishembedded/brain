// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving agent sessions whose every request extends the tokens the
// model was given and sampled before, for its clients. If your team needs expertise in exact multi-turn prompts for
// training and serving language models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Replaying a conversation's sampled answers as the tokens they were.
//!
//! A chat reply carries [`Sampled`]: the ids the model generated for the turn, the ids of the generation prompt it
//! continued (`prefill_token_ids`: the template's opening of the assistant turn and any prefilled reasoning block),
//! how the generation ended, the tokenizer, and the length and hash of the prompt it was given. A client echoes the
//! object back unchanged on that assistant message. [`replay`] builds the next prompt by segments: the template writes
//! everything that is not a sampled answer (system, tools, user and tool turns, and the generation prompt), and each
//! answer that carries a valid [`Sampled`] is spliced in as its generation prompt and generated ids, with the turn's
//! end-of-turn marker added as context when the generation did not end on one. So the prompt of call `k + 1` is the
//! prompt of call `k`, the ids it generated and the turns after it, by construction, for any template.
//!
//! The template never sees a sampled answer: it renders a sentinel in its place, and the splice replaces the text from
//! where that turn begins to its end-of-turn marker. An answer without `sampled` (another client, a training record,
//! another tokenizer) is rendered by the template, as is every answer from the first one whose blob does not hold:
//! another tokenizer, a generation prompt the template cannot write there, or a context that is not the prompt built
//! so far (the conversation was edited). [`History`] reports whether the prompt is append-only and where it is not.
//!
//! Token ids travel as text: the unsigned LEB128 varint of each id, the whole run base64-encoded ([`encode_ids`]).
//! About a third of the bytes of a JSON array of the same ids.

use std::collections::HashSet;

use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

/// The token ids of one run, as the replay blob carries them: each id as an unsigned LEB128 varint, base64-encoded.
pub fn encode_ids(ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(ids.len() * 3);
    for &id in ids {
        let mut v = id;
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                bytes.push(byte);
                break;
            }
            bytes.push(byte | 0x80);
        }
    }
    crate::base64::encode(&bytes)
}

/// The ids [`encode_ids`] wrote. A varint longer than an id can be, or one cut off, is an error.
pub fn decode_ids(text: &str) -> Result<Vec<u32>, String> {
    let bytes = crate::base64::decode(text)?;
    let mut ids = Vec::with_capacity(bytes.len() / 2);
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for &b in &bytes {
        value |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            ids.push(u32::try_from(value).map_err(|_| "a token id past u32".to_string())?);
            value = 0;
            shift = 0;
        } else {
            shift += 7;
            if shift > 28 {
                return Err("a token id varint longer than five bytes".to_string());
            }
        }
    }
    if shift != 0 {
        return Err("the token ids end inside a varint".to_string());
    }
    Ok(ids)
}

/// `sha256:<hex>` of `ids` as little-endian u32s: the identity of a prompt a [`Sampled`] answer was generated after.
pub fn context_hash(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

/// What a model sampled for one assistant turn, as a reply reports it and a client echoes it back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sampled {
    /// The generated ids, the end-of-turn token included when one ended the generation.
    pub token_ids: Vec<u32>,
    /// The ids of the generation prompt the model continued: the template's opening of the assistant turn and any
    /// prefill (`<|im_start|>assistant\n<think>\n\n</think>\n\n` for Qwen3 without reasoning).
    pub prefill_token_ids: Vec<u32>,
    /// How the generation ended (`stop`, `length`, `cancelled`, `stop_sequence`, `tool_calls`).
    pub finish: String,
    /// `sha256:<hex>` of the tokenizer file the ids belong to; `None` for a tokenizer read from no file.
    pub tokenizer_digest: Option<String>,
    /// How many ids the prompt held, the generation prompt included.
    pub context_len: usize,
    /// [`context_hash`] of the prompt.
    pub context_hash: String,
}

impl Sampled {
    /// What a generation after `prompt` sampled: `token_ids` (the end-of-turn token included when one ended it), the
    /// last `prefill_len` ids of the prompt as its generation prompt.
    pub fn of(prompt: &[u32], prefill_len: usize, token_ids: Vec<u32>, finish: &str, tokenizer_digest: Option<&str>) -> Sampled {
        Sampled {
            token_ids,
            prefill_token_ids: prompt[prompt.len().saturating_sub(prefill_len)..].to_vec(),
            finish: finish.to_string(),
            tokenizer_digest: tokenizer_digest.map(str::to_string),
            context_len: prompt.len(),
            context_hash: context_hash(prompt),
        }
    }

    /// The blob as a reply's assistant message carries it.
    pub fn to_json(&self) -> Value {
        json!({
            "token_ids": encode_ids(&self.token_ids),
            "prefill_token_ids": encode_ids(&self.prefill_token_ids),
            "finish": self.finish,
            "tokenizer_digest": self.tokenizer_digest,
            "context_len": self.context_len,
            "context_hash": self.context_hash,
        })
    }

    /// A blob a client sent back. Every field is required; a malformed one is an error naming it.
    pub fn from_json(v: &Value) -> Result<Sampled, String> {
        let text = |key: &str| v.get(key).and_then(Value::as_str).ok_or_else(|| format!("sampled.{key}: missing or not text"));
        let ids = |key: &str| text(key).and_then(|t| decode_ids(t).map_err(|e| format!("sampled.{key}: {e}")));
        let tokenizer_digest = match v.get("tokenizer_digest") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => return Err("sampled.tokenizer_digest: not text".to_string()),
        };
        let context_len = v.get("context_len").and_then(Value::as_u64).ok_or("sampled.context_len: missing or not a count")?;
        Ok(Sampled {
            token_ids: ids("token_ids")?,
            prefill_token_ids: ids("prefill_token_ids")?,
            finish: text("finish")?.to_string(),
            tokenizer_digest,
            context_len: usize::try_from(context_len).map_err(|_| "sampled.context_len: too large")?,
            context_hash: text("context_hash")?.to_string(),
        })
    }
}

/// Why an assistant turn was rendered by the template instead of replayed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Divergence {
    /// The turn carries no `sampled`.
    Missing,
    /// Its `sampled` is malformed: unreadable, an id outside the vocabulary, or a turn marker inside the answer.
    Malformed,
    /// Its ids belong to another tokenizer.
    Tokenizer,
    /// Its generation prompt is not one the template writes at that point.
    Prefill,
    /// Its context is not the prompt built so far: an earlier turn was changed.
    Context,
    /// The template does not write that turn as a turn of its own followed by an end-of-turn marker.
    Template,
}

impl Divergence {
    pub fn as_str(self) -> &'static str {
        match self {
            Divergence::Missing => "missing",
            Divergence::Malformed => "malformed",
            Divergence::Tokenizer => "tokenizer",
            Divergence::Prefill => "prefill",
            Divergence::Context => "context",
            Divergence::Template => "template",
        }
    }
}

/// Whether a prompt replays every earlier answer as it was sampled (`append_only`), and if not, the first assistant
/// turn that was rendered by the template instead and why. Turn indices count the request's messages from 0.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct History {
    pub diverged: Option<(usize, Divergence)>,
}

impl History {
    pub fn append_only(&self) -> bool {
        self.diverged.is_none()
    }

    /// `{append_only, diverged_turn, reason}` as a reply's `brain.history` reports it.
    pub fn to_json(&self) -> Value {
        json!({
            "append_only": self.append_only(),
            "diverged_turn": self.diverged.map(|(turn, _)| turn),
            "reason": self.diverged.map(|(_, why)| why.as_str()),
        })
    }

    fn note(&mut self, turn: usize, why: Divergence) {
        if self.diverged.is_none_or(|(at, _)| turn < at) {
            self.diverged = Some((turn, why));
        }
    }
}

/// A prompt built by [`replay`]: its ids, how many of them at the end are the generation prompt, and its history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replayed {
    pub ids: Vec<u32>,
    pub prefill_len: usize,
    pub history: History,
}

/// A template rendering: turns, reasoning on or off, with or without the generation prompt.
pub type Render<'a> = dyn Fn(&[Value], bool, bool) -> Result<String, String> + 'a;

/// What [`replay`] needs of a model's prompt path.
pub struct PromptPath<'a> {
    /// The template's text for `messages` (OpenAI-shaped turns), with reasoning on or off (`thinking`), with or
    /// without the generation prompt (`generation`). The rendering with it must extend the rendering without it.
    pub render: &'a Render<'a>,
    /// The ids of a run of template text.
    pub encode: &'a dyn Fn(&str) -> Vec<u32>,
    /// The markers a generation ends with, each as text and as the ids of that text.
    pub end_of_turn: &'a [(String, Vec<u32>)],
    /// Ids that open or close a turn: an answer may hold none, except an end-of-turn id as its last.
    pub turn_markers: &'a HashSet<u32>,
    pub vocab: usize,
    pub tokenizer_digest: Option<&'a str>,
}

/// The prompt for `messages` under `path`, with reasoning `thinking`: every assistant turn that carries a valid
/// `sampled` replayed as its ids, everything else rendered by the template, the generation prompt at the end.
pub fn replay(messages: &[Value], path: &PromptPath, thinking: bool) -> Result<Replayed, String> {
    let mut history = History::default();
    let mut candidates: Vec<(usize, Sampled)> = Vec::new();
    for (at, m) in messages.iter().enumerate() {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blob) = m.get("sampled").filter(|v| !v.is_null()) else {
            history.note(at, Divergence::Missing);
            continue;
        };
        match Sampled::from_json(blob) {
            Err(_) => history.note(at, Divergence::Malformed),
            Ok(s) if s.tokenizer_digest.as_deref() != path.tokenizer_digest => history.note(at, Divergence::Tokenizer),
            Ok(s) if !well_formed(&s, path) => history.note(at, Divergence::Malformed),
            Ok(s) => candidates.push((at, s)),
        }
    }
    loop {
        match build(messages, &candidates, path, thinking)? {
            Ok((ids, prefill_len)) => return Ok(Replayed { ids, prefill_len, history }),
            // The first answer that does not hold is rendered by the template, and so is every later one: their
            // contexts held that answer as sampled.
            Err((j, why)) => {
                history.note(candidates[j].0, why);
                candidates.truncate(j);
            }
        }
    }
}

/// Every id in the vocabulary, and no turn marker inside the answer except an end-of-turn id as its last.
fn well_formed(s: &Sampled, path: &PromptPath) -> bool {
    let in_vocab = s.token_ids.iter().chain(&s.prefill_token_ids).all(|&id| (id as usize) < path.vocab);
    let inner = match s.token_ids.split_last() {
        Some((last, rest)) if path.end_of_turn.iter().any(|(_, ids)| ids.len() == 1 && ids[0] == *last) => rest,
        _ => &s.token_ids[..],
    };
    in_vocab && !inner.iter().any(|id| path.turn_markers.contains(id))
}

/// A sentinel that occurs in no message.
fn sentinel_stem(messages: &[Value]) -> String {
    let text = serde_json::to_string(messages).unwrap_or_default();
    let mut stem = String::from("BRAINREPLAYTURN");
    while text.contains(&stem) {
        stem.push('Q');
    }
    stem
}

type Built = Result<(Vec<u32>, usize), (usize, Divergence)>;

/// The prompt with `candidates` replayed, or the first candidate that does not hold and why.
fn build(messages: &[Value], candidates: &[(usize, Sampled)], path: &PromptPath, thinking: bool) -> Result<Built, String> {
    let stem = sentinel_stem(messages);
    let sentinel = |j: usize| format!("{stem}{j}Z");
    let mut shown: Vec<Value> = messages.to_vec();
    for (j, (at, _)) in candidates.iter().enumerate() {
        shown[*at] = json!({"role": "assistant", "content": sentinel(j)});
    }
    let base = (path.render)(&shown, thinking, false)?;
    let full = (path.render)(&shown, thinking, true)?;
    let generation = full.strip_prefix(base.as_str()).ok_or("the template's generation prompt does not extend the conversation it follows")?;
    let mut ids: Vec<u32> = Vec::new();
    // The hash of `ids[..hashed]`, extended as the prompt grows: each answer's context is a prefix of the next one's.
    let mut hasher = Sha256::new();
    let mut hashed = 0usize;
    let mut cursor = 0usize;
    for (j, (at, s)) in candidates.iter().enumerate() {
        let before = (path.render)(&shown[..*at], thinking, false)?;
        if before.len() < cursor || !base.starts_with(&before) {
            return Ok(Err((j, Divergence::Template)));
        }
        ids.extend((path.encode)(&base[cursor..before.len()]));
        let prompts = [thinking, !thinking].map(|t| -> Result<Vec<u32>, String> {
            let open = (path.render)(&shown[..*at], t, true)?;
            let closed = if t == thinking { before.clone() } else { (path.render)(&shown[..*at], t, false)? };
            Ok(open.strip_prefix(closed.as_str()).map(|g| (path.encode)(g)).unwrap_or_default())
        });
        if !prompts.into_iter().any(|p| p.is_ok_and(|p| p == s.prefill_token_ids)) {
            return Ok(Err((j, Divergence::Prefill)));
        }
        ids.extend(&s.prefill_token_ids);
        for id in &ids[hashed..] {
            hasher.update(id.to_le_bytes());
        }
        hashed = ids.len();
        if ids.len() != s.context_len || format!("sha256:{}", hex::encode(hasher.clone().finalize())) != s.context_hash {
            return Ok(Err((j, Divergence::Context)));
        }
        ids.extend(&s.token_ids);
        let mark = sentinel(j);
        let Some(found) = base[before.len()..].find(&mark).map(|p| before.len() + p + mark.len()) else {
            return Ok(Err((j, Divergence::Template)));
        };
        // The template's close of the turn runs to the first marker a generation ends with (`<|im_end|>`; a template
        // that writes typed messages closes the last message before it). A generation that ended on its own marker
        // holds that close; one that did not (cut at its length, cancelled, a stop string) gets it as context.
        let Some(closed) = path.end_of_turn.iter().filter_map(|(text, _)| base[found..].find(text.as_str()).map(|p| found + p + text.len())).min() else {
            return Ok(Err((j, Divergence::Template)));
        };
        let ended = s.token_ids.last().is_some_and(|last| path.end_of_turn.iter().any(|(_, e)| e.len() == 1 && e[0] == *last));
        if !ended {
            ids.extend((path.encode)(&base[found..closed]));
        }
        cursor = closed;
    }
    ids.extend((path.encode)(&base[cursor..]));
    let prefill = (path.encode)(generation);
    let prefill_len = prefill.len();
    ids.extend(prefill);
    Ok(Ok((ids, prefill_len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_through_their_text_encoding() {
        let ids = [0u32, 1, 127, 128, 16_383, 16_384, 151_643, u32::MAX];
        assert_eq!(decode_ids(&encode_ids(&ids)).unwrap(), ids);
        assert_eq!(decode_ids(&encode_ids(&[])).unwrap(), Vec::<u32>::new());
        assert!(decode_ids(&crate::base64::encode(&[0x80])).is_err(), "cut inside a varint");
        assert!(decode_ids(&crate::base64::encode(&[0xff, 0xff, 0xff, 0xff, 0xff, 0x01])).is_err(), "too long");
    }
}
