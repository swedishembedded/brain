// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Form extraction: answer a fixed set of questions about an image with a
//! probability for every allowed answer.
//!
//! A form is a JSON object of named fields: a choice (`{"enum": [...]}`), an
//! integer count (`{"type": "integer", "minimum": a, "maximum": b}`, exclusive
//! bounds accepted) or free text (`{"type": "string"}`). The prompt lists the
//! fields (`? animal=dog|cat|bird; n#0..5` plus a newline); the answer is
//! written field by field as `name=value`, each field ending in a newline and
//! the last in `</s>`. For every field each candidate answer (plus an
//! abstention `-` for choices) is scored by the summed log-probability of its
//! tokens and terminator, the scores are soft-maxed into probabilities, and
//! the most probable candidate is committed before the next field.

use std::fmt;

use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;
use serde_json::Value;

use crate::decoder::{CrossMemory, Decoder, Positions};
use crate::tokenizer::{Tokenizer, TokenizerError};

/// Most candidates a field may offer.
pub const MAX_CANDIDATES: usize = 256;
/// Characters a choice or field name may not contain.
const RESERVED: &[char] = &[';', '|', '=', '#', '@', '$', '*', '\n'];
/// Abstention answer of a choice field.
pub const ABSTAIN: &str = "-";

/// Why a form was refused or could not be answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormError {
    /// The form is malformed.
    Invalid(String),
    /// A field kind that is not implemented.
    Unsupported(String),
    /// Tokenising failed.
    Tokenizer(TokenizerError),
}

impl fmt::Display for FormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormError::Invalid(m) => write!(f, "mirage form: {m}"),
            FormError::Unsupported(m) => write!(f, "mirage form: {m}"),
            FormError::Tokenizer(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for FormError {}

impl From<TokenizerError> for FormError {
    fn from(e: TokenizerError) -> Self {
        FormError::Tokenizer(e)
    }
}

/// What a field asks for.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    /// One of several values (strings or numbers).
    Choice(Vec<Value>),
    /// An integer in an inclusive range.
    Count { min: i64, max: i64 },
    /// Free text of at most `max_length` characters.
    Text { max_length: usize },
}

/// A named field.
#[derive(Debug, Clone, PartialEq)]
pub struct FormField {
    /// Field name.
    pub name: String,
    /// What it asks for.
    pub kind: FieldKind,
}

fn invalid<T>(m: impl Into<String>) -> Result<T, FormError> {
    Err(FormError::Invalid(m.into()))
}

fn plain(name: &str, what: &str) -> Result<(), FormError> {
    if name.is_empty() || name.contains(RESERVED) || name.contains(char::is_whitespace) {
        return invalid(format!("{what} `{name}` is empty or contains whitespace or one of ; | = # @ $ *"));
    }
    Ok(())
}

/// Parse and validate a form.
pub fn parse_form(form: &Value) -> Result<Vec<FormField>, FormError> {
    let Some(map) = form.as_object() else { return invalid("a form is a JSON object of fields") };
    if map.is_empty() {
        return invalid("a form needs at least one field");
    }
    let mut fields = Vec::new();
    for (name, spec) in map {
        plain(name, "field name")?;
        let ty = spec.get("type").and_then(Value::as_str);
        let kind = if let Some(options) = spec.get("enum") {
            let Some(options) = options.as_array() else { return invalid(format!("field `{name}`: `enum` must be a list")) };
            if options.is_empty() || options.len() > MAX_CANDIDATES {
                return invalid(format!("field `{name}`: `enum` needs 1 to {MAX_CANDIDATES} values"));
            }
            let mut seen = std::collections::BTreeSet::new();
            for o in options {
                let text = match o {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    _ => return invalid(format!("field `{name}`: enum values are strings or numbers")),
                };
                plain(&text, &format!("field `{name}` value"))?;
                if text == ABSTAIN || !seen.insert(text.clone()) {
                    return invalid(format!("field `{name}`: value `{text}` is repeated or reserved"));
                }
            }
            FieldKind::Choice(options.clone())
        } else if ty == Some("integer") {
            let bound = |key: &str, ex: &str, delta: i64| -> Option<i64> {
                spec.get(key).and_then(Value::as_i64).or_else(|| spec.get(ex).and_then(Value::as_i64).map(|v| v + delta))
            };
            let (Some(min), Some(max)) = (bound("minimum", "exclusiveMinimum", 1), bound("maximum", "exclusiveMaximum", -1)) else {
                return invalid(format!("field `{name}`: an integer needs `minimum` and `maximum`"));
            };
            if min > max || (max - min) as usize >= MAX_CANDIDATES {
                return invalid(format!("field `{name}`: range {min}..{max} is empty or has more than {MAX_CANDIDATES} values"));
            }
            FieldKind::Count { min, max }
        } else if ty == Some("string") {
            if spec.get("format").is_some() {
                return Err(FormError::Unsupported(format!("field `{name}`: box fields are not implemented")));
            }
            FieldKind::Text { max_length: spec.get("maxLength").and_then(Value::as_u64).unwrap_or(32).min(256) as usize }
        } else {
            return invalid(format!("field `{name}`: expected `enum`, an integer or a string type"));
        };
        fields.push(FormField { name: name.clone(), kind });
    }
    Ok(fields)
}

fn spec_text(fields: &[FormField]) -> String {
    let parts: Vec<String> = fields
        .iter()
        .map(|f| match &f.kind {
            FieldKind::Choice(options) => {
                let names: Vec<String> = options.iter().map(value_text).collect();
                format!("{}={};", f.name, names.join("|"))
            }
            FieldKind::Count { min, max } => format!("{}#{min}..{max}", f.name),
            FieldKind::Text { max_length } => format!("{}${max_length}", f.name),
        })
        .collect();
    format!("? {}\n", parts.join(" "))
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One answered field.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The chosen value (`null` when the model abstained).
    pub value: Value,
    /// Probability of every candidate, in candidate order, abstention last.
    pub probabilities: Vec<(String, f64)>,
}

/// The extraction result.
#[derive(Debug, Clone, PartialEq)]
pub struct Extract {
    /// Answers in form order.
    pub fields: Vec<(String, Answer)>,
}

impl Serialize for Extract {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        struct Fields<'a>(&'a [(String, Answer)]);
        struct Probs<'a>(&'a [(String, f64)]);
        struct One<'a>(&'a Answer);
        impl Serialize for Probs<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                let mut m = s.serialize_map(Some(self.0.len()))?;
                for (k, v) in self.0 {
                    m.serialize_entry(k, v)?;
                }
                m.end()
            }
        }
        impl Serialize for One<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("value", &self.0.value)?;
                m.serialize_entry("probabilities", &Probs(&self.0.probabilities))?;
                m.end()
            }
        }
        impl Serialize for Fields<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                let mut m = s.serialize_map(Some(self.0.len()))?;
                for (k, v) in self.0 {
                    m.serialize_entry(k, &One(v))?;
                }
                m.end()
            }
        }
        let mut m = s.serialize_map(Some(1))?;
        m.serialize_entry("form", &Fields(&self.fields))?;
        m.end()
    }
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// Encode `text` as if it began a new word (a leading boundary mark).
fn encode_with_prefix(tok: &Tokenizer, text: &str) -> Result<Vec<u32>, TokenizerError> {
    tok.encode(&format!(" {text}"))
}

/// Answer `fields` about the image whose memory is `cross`.
pub fn extract(dec: &Decoder, cross: &CrossMemory, tok: &Tokenizer, prompt: &[u32], fields: &[FormField], max_tokens: usize) -> Result<Extract, FormError> {
    crate::decoder::with_decode_threads(|| extract_inner(dec, cross, tok, prompt, fields, max_tokens))
}

fn extract_inner(dec: &Decoder, cross: &CrossMemory, tok: &Tokenizer, prompt: &[u32], fields: &[FormField], max_tokens: usize) -> Result<Extract, FormError> {
    let vocab = dec.vocabulary();
    let mut st = dec.new_state();
    let mut tokens = prompt.to_vec();
    tokens.extend(encode_with_prefix(tok, &spec_text(fields))?);
    dec.forward(&mut st, cross, &tokens, Positions::Last);
    let newline = tok.encode("\n")?;
    let eos = [tok.eos_id()];
    let mut answers = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let terminator: &[u32] = if i + 1 == fields.len() { &eos } else { &newline };
        let head = tok.encode(&format!("{}=", field.name))?;
        let mut logits = dec.forward(&mut st, cross, &head, Positions::Last);

        let (labels, values): (Vec<String>, Vec<Value>) = match &field.kind {
            FieldKind::Choice(options) => (options.iter().map(value_text).collect(), options.clone()),
            FieldKind::Count { min, max } => ((*min..=*max).map(|v| v.to_string()).collect(), (*min..=*max).map(Value::from).collect()),
            FieldKind::Text { .. } => (Vec::new(), Vec::new()),
        };
        if let FieldKind::Text { max_length } = field.kind {
            let (text, probability) = greedy_text(dec, cross, tok, &mut st, &mut logits, terminator, max_length.min(max_tokens))?;
            answers.push((field.name.clone(), Answer { value: Value::String(text.clone()), probabilities: vec![(text, probability)] }));
            continue;
        }
        let mut candidates: Vec<(String, Value, Vec<u32>)> = Vec::new();
        for (label, value) in labels.iter().zip(&values) {
            candidates.push((label.clone(), value.clone(), tok.encode(label)?));
        }
        if matches!(field.kind, FieldKind::Choice(_)) {
            candidates.push((ABSTAIN.to_string(), Value::Null, tok.encode(ABSTAIN)?));
        }
        let first = log_softmax(&logits);
        let scores: Vec<f32> = candidates
            .iter()
            .map(|(_, _, toks)| {
                let mut seq = toks.clone();
                seq.extend_from_slice(terminator);
                let mut score = first[seq[0] as usize];
                if seq.len() > 1 {
                    let mut branch = st.clone();
                    let rows = dec.forward(&mut branch, cross, &seq[..seq.len() - 1], Positions::All);
                    for (t, row) in rows.chunks_exact(vocab).enumerate() {
                        score += log_softmax(row)[seq[t + 1] as usize];
                    }
                }
                score
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = scores.iter().map(|s| f64::from(s - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probabilities: Vec<(String, f64)> = candidates.iter().zip(&exps).map(|((l, _, _), e)| (l.clone(), e / total)).collect();
        let best = exps.iter().enumerate().fold(0, |b, (i, e)| if *e > exps[b] { i } else { b });
        let mut commit = candidates[best].2.clone();
        commit.extend_from_slice(terminator);
        dec.forward(&mut st, cross, &commit, Positions::Last);
        answers.push((field.name.clone(), Answer { value: candidates[best].1.clone(), probabilities }));
    }
    Ok(Extract { fields: answers })
}

/// Greedy free text until the terminator; returns the text and the product of token probabilities.
fn greedy_text(dec: &Decoder, cross: &CrossMemory, tok: &Tokenizer, st: &mut crate::decoder::DecoderState, logits: &mut Vec<f32>, terminator: &[u32], max_tokens: usize) -> Result<(String, f64), FormError> {
    let mut out = Vec::new();
    let mut probability = 1.0f64;
    for _ in 0..max_tokens.max(1) {
        let lp = log_softmax(logits);
        let next = lp.iter().enumerate().fold(0, |b, (i, v)| if *v > lp[b] { i } else { b }) as u32;
        probability *= f64::from(lp[next as usize]).exp();
        if terminator.first() == Some(&next) {
            *logits = dec.forward(st, cross, terminator, Positions::Last);
            return Ok((tok.decode(&out)?, probability));
        }
        out.push(next);
        *logits = dec.forward(st, cross, &[next], Positions::Last);
    }
    *logits = dec.forward(st, cross, terminator, Positions::Last);
    Ok((tok.decode(&out)?, probability))
}
