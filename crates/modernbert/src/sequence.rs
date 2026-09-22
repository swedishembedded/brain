// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laya's own packed sequence layout, reproduced byte-for-byte in logic from
//! the real `convaiinnovations/laya` `rl_common.py`'s `build_sequence`/
//! `render_options`/`serialize_state` (Apache-2.0, verified against the real
//! released source this session - not re-derived from the model card):
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```
//!
//! `render_options` renders a question's option TEXTS in label-index order:
//! - `choice`: `crit` is an ORDERED `label -> description` map; a falsy
//!   (missing or empty) description renders as the bare label, else
//!   `"label: description"`.
//! - `score`: `crit` is an ordered list of level descriptions, rendered
//!   `"level {i}: {text}"`.
//! - `noul`: always exactly two options, `["false: ...", "true: ..."]`
//!   (index 1 is always the "true" reading), falling back to a fixed default
//!   sentence when the criteria text is missing or empty.
//!
//! `build_sequence` then: tokenizes `"<type> question: <ins>"` as the head
//! text; tokenizes `" " + option_text` per option, capped at 48 tokens, with
//! a `[MASK]` token id prepended to each; shrinks every option EVENLY when
//! they do not fit `head_max_len` (`opt_budget < 16`); truncates the head
//! text to whatever budget is left (floor 8 tokens); packs
//! `[CLS] head [SEP] opt0 opt1 ... [SEP]`; tokenizes the serialized state
//! into whatever room remains under `max_len`; and appends a final `[SEP]`.
//! Marker positions are the packed row (0-based, within this ONE call's own
//! sequence) of each option's `[MASK]` token - a caller packing multiple
//! questions into one batch adds that question's `row0` to get the ABSOLUTE
//! row `LayaHead::set_call`'s `marker_rows` needs.
//!
//! **Any literal occurrence of the tokenizer's own `[MASK]` string inside
//! user-supplied instructions/option text/state is replaced with a space
//! before tokenizing** (`tok.decode(&[mask_token_id])`, not a hardcoded
//! `"[MASK]"` - the multilingual/typed-decisions variants use different
//! special-token spellings) - guarding against injecting a bogus marker
//! position from untrusted text, exactly as the Python reference does.
//!
//! **Key order matters and this workspace's `serde_json` does NOT preserve
//! it** (`Cargo.toml`'s own comment confirms `preserve_order` is
//! deliberately not enabled workspace-wide - verified this session, not
//! assumed): Python's `json.dumps(state, ensure_ascii=False)` serializes a
//! dict in INSERTION order, but a `serde_json::Value::Object` parsed or built
//! in this workspace iterates and serializes in SORTED key order instead. A
//! naive `serde_json::to_string` round-trip on a multi-key state would
//! therefore tokenize DIFFERENT bytes than the real reference on any state
//! whose keys are not already alphabetical - silently, since both sides
//! still produce valid JSON. [`OrderedJson`] exists ONLY to avoid that: it
//! is not a general JSON value, just enough of one to serialize a
//! `build_sequence` state (or a `choice` question's ordered `crit` map) with
//! the SAME key order the caller gave it, via [`write_json`]'s own
//! `json.dumps`-equivalent formatting (`", "`/`": "` separators, JSON string
//! escaping, `ensure_ascii=False` - non-ASCII passed through raw).

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer as _;

use crate::config::ModernBertConfig;

/// `rl_common.py`'s own `QTYPES` mapping - the index `LayaHead::set_call`'s
/// `qtype` slice needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn index(self) -> u32 {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }

    fn word(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// A minimal ordered JSON tree - see the module doc's "key order matters"
/// section for why this exists instead of `serde_json::Value`. `Object`
/// stores `(key, value)` pairs in caller-given order; nothing here re-sorts
/// or deduplicates them (a duplicate key is the caller's bug, reproduced
/// faithfully rather than silently fixed, matching `json.dumps`' own
/// last-write-wins behavior on a Python dict that can't even construct a
/// duplicate key in the first place).
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedJson>),
    Object(Vec<(String, OrderedJson)>),
}

impl OrderedJson {
    pub fn str(s: impl Into<String>) -> OrderedJson {
        OrderedJson::String(s.into())
    }
    pub fn int(n: i64) -> OrderedJson {
        OrderedJson::Number(n.into())
    }
    pub fn array(items: Vec<OrderedJson>) -> OrderedJson {
        OrderedJson::Array(items)
    }
    pub fn object(entries: Vec<(&str, OrderedJson)>) -> OrderedJson {
        OrderedJson::Object(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Read a JSON document, KEEPING the key order it was written in.
    ///
    /// [`write_json`]'s missing half. A request that arrives as text - over a
    /// pipe, a socket, or out of a file - has to reach [`build_sequence`] as
    /// the same bytes the Python reference would tokenize, and that is only
    /// true if nothing between the wire and the tokenizer sorts an object's
    /// keys. `serde_json::Value` does exactly that (this workspace
    /// deliberately does not enable its `preserve_order` feature - see this
    /// module's doc), so a parse through it would silently change what the
    /// model reads.
    ///
    /// Deserializing into [`OrderedJson`] instead keeps document order for
    /// free: `serde_json` yields an object's entries in the order it reads
    /// them, and this type stores them that way.
    pub fn parse(text: &str) -> Result<OrderedJson, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

/// Order-preserving by construction: `MapAccess` hands entries over in the
/// order they appear in the document, and [`OrderedJson::Object`] keeps them.
/// Written by hand rather than derived because this enum is JSON's own shape,
/// not a tagged Rust type - the same reason `serde_json::Value` implements it
/// by hand.
impl<'de> serde::Deserialize<'de> for OrderedJson {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<OrderedJson, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedJson;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("any JSON value")
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Bool(v))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<OrderedJson, E> {
                // JSON has no NaN/Inf literal, so this only rejects a
                // non-finite a non-JSON deserializer handed us.
                serde_json::Number::from_f64(v)
                    .map(OrderedJson::Number)
                    .ok_or_else(|| serde::de::Error::custom("a JSON number must be finite"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v.to_string()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<OrderedJson, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(OrderedJson::Array(items))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<OrderedJson, A::Error> {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((k, v)) = map.next_entry::<String, OrderedJson>()? {
                    entries.push((k, v));
                }
                Ok(OrderedJson::Object(entries))
            }
        }

        de.deserialize_any(V)
    }
}

/// `json.dumps(v, ensure_ascii=False)`-equivalent formatting: `", "` between
/// array/object items, `": "` between an object's key and value, JSON string
/// escaping via `serde_json`'s own string serializer (matches
/// `ensure_ascii=False`: only the JSON-mandatory characters are escaped, not
/// non-ASCII), reused rather than hand-rolled so it cannot drift from
/// `serde_json`'s own escaping rules.
pub fn write_json(v: &OrderedJson, out: &mut String) {
    match v {
        OrderedJson::Null => out.push_str("null"),
        OrderedJson::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        OrderedJson::Number(n) => out.push_str(&n.to_string()),
        OrderedJson::String(s) => out.push_str(&serde_json::to_string(s).expect("string serialization cannot fail")),
        OrderedJson::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(item, out);
            }
            out.push(']');
        }
        OrderedJson::Object(entries) => {
            out.push('{');
            for (i, (k, val)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).expect("string serialization cannot fail"));
                out.push_str(": ");
                write_json(val, out);
            }
            out.push('}');
        }
    }
}

/// A `build_sequence` state: either a plain string (used AS-IS, matching
/// `serialize_state`'s `isinstance(state, str)` branch) or a JSON value
/// (serialized via [`write_json`], matching its `json.dumps` branch).
#[derive(Clone, Debug)]
pub enum State {
    Str(String),
    Json(OrderedJson),
}

impl State {
    /// The exact text `build_sequence` tokenizes: a `Str` state as-is, a
    /// `Json` one through [`write_json`] (`json.dumps(..., ensure_ascii=
    /// False)`'s own formatting). Public because a caller holding one of
    /// these has to be able to hand the SAME bytes to a different backbone -
    /// `brain::DecisionPipeline`'s decide arm takes a `&str` state - rather
    /// than re-serializing it its own way and asking two models about two
    /// different states.
    pub fn serialize(&self) -> String {
        match self {
            State::Str(s) => s.clone(),
            State::Json(v) => {
                let mut out = String::new();
                write_json(v, &mut out);
                out
            }
        }
    }
}

/// One decision question, in `render_options`' three shapes.
#[derive(Clone, Debug)]
pub enum Question {
    /// `crit`: ordered `(label, description)` pairs. A falsy (`None` or
    /// empty-string) description renders as the bare label.
    Choice { ins: String, options: Vec<(String, Option<String>)> },
    /// `crit`: ordered level descriptions, rendered `"level {i}: {text}"`.
    Score { ins: String, options: Vec<String> },
    /// Always renders exactly `["false: ...", "true: ..."]` (index 1 is the
    /// "true" reading) - falsy criteria text falls back to a fixed default.
    Noul { ins: String, false_text: Option<String>, true_text: Option<String> },
}

impl Question {
    pub fn qtype(&self) -> QType {
        match self {
            Question::Choice { .. } => QType::Choice,
            Question::Score { .. } => QType::Score,
            Question::Noul { .. } => QType::Noul,
        }
    }

    fn ins(&self) -> &str {
        match self {
            Question::Choice { ins, .. } | Question::Score { ins, .. } | Question::Noul { ins, .. } => ins,
        }
    }

    /// Option texts in label-index order - `rl_common.py`'s own
    /// `render_options`.
    pub fn render_options(&self) -> Vec<String> {
        match self {
            Question::Choice { options, .. } => options
                .iter()
                .map(|(k, v)| match v {
                    Some(v) if !v.is_empty() => format!("{k}: {v}"),
                    _ => k.clone(),
                })
                .collect(),
            Question::Score { options, .. } => {
                options.iter().enumerate().map(|(i, c)| format!("level {i}: {c}")).collect()
            }
            Question::Noul { false_text, true_text, .. } => {
                let f = as_nonempty(false_text).unwrap_or("no, the statement does not hold");
                let t = as_nonempty(true_text).unwrap_or("yes, the statement holds");
                vec![format!("false: {f}"), format!("true: {t}")]
            }
        }
    }
}

fn as_nonempty(s: &Option<String>) -> Option<&str> {
    match s {
        Some(s) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    }
}

/// Floor division matching Python's `//` for a positive divisor (the only
/// case `build_sequence` ever exercises - `len(opt_ids) >= 1`).
fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// Reproduce `rl_common.py`'s `build_sequence` exactly: tokenize, budget,
/// pack, truncate - see the module doc for the full shape. `tok` provides the
/// encode; `cfg` provides the special-token ids (`cls_token_id`/
/// `sep_token_id`/`mask_token_id`, read from the real tokenizer at import
/// time - never hardcoded, see [`crate::import`]). Returns `(packed ids,
/// marker positions)`, both already truncated to `max_len` and filtered to
/// positions still inside it - the SAME two-step the Python reference ends
/// with (`ids[:max_len], [m for m in markers if m < max_len]`).
pub fn build_sequence(
    tok: &QwenBpe,
    cfg: &ModernBertConfig,
    state: &State,
    q: &Question,
    max_len: u32,
    head_max_len: u32,
    option_order: Option<&[usize]>,
    truncate_left: bool,
) -> (Vec<u32>, Vec<usize>) {
    let mask_tok = tok.decode(&[cfg.mask_token_id]);
    let opts = q.render_options();
    let order: Vec<usize> = match option_order {
        Some(o) => o.to_vec(),
        None => (0..opts.len()).collect(),
    };

    let ins = q.ins().replace(&mask_tok, " ");
    let head_text = format!("{} question: {}", q.qtype().word(), ins);
    let mut head_ids: Vec<u32> = tok.encode(&head_text);

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(order.len());
    for &i in &order {
        let opt_text = opts[i].replace(&mask_tok, " ");
        let mut piece = tok.encode(&format!(" {opt_text}"));
        piece.truncate(48);
        let mut o = Vec::with_capacity(1 + piece.len());
        o.push(cfg.mask_token_id);
        o.append(&mut piece);
        opt_ids.push(o);
    }

    let sum_opt: i64 = opt_ids.iter().map(|o| o.len() as i64).sum();
    let mut opt_budget: i64 = head_max_len as i64 - sum_opt;
    if opt_budget < 16 {
        // Too many / too long options: shrink every option text evenly.
        let per = floor_div(head_max_len as i64 - 16, (opt_ids.len().max(1)) as i64).max(4) as usize;
        for o in opt_ids.iter_mut() {
            o.truncate(per);
        }
        let sum_opt2: i64 = opt_ids.iter().map(|o| o.len() as i64).sum();
        opt_budget = head_max_len as i64 - sum_opt2;
    }
    let head_cap = opt_budget.max(8) as usize;
    head_ids.truncate(head_cap);

    let mut ids: Vec<u32> = Vec::new();
    ids.push(cfg.cls_token_id);
    ids.extend(head_ids);
    ids.push(cfg.sep_token_id);

    let mut markers: Vec<usize> = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o.iter().copied());
    }
    ids.push(cfg.sep_token_id);

    let room = (max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
    let state_text = state.serialize().replace(&mask_tok, " ");
    let mut st = tok.encode(&state_text);
    if truncate_left {
        if st.len() > room {
            st = st.split_off(st.len() - room);
        }
    } else {
        st.truncate(room);
    }
    ids.extend(st);
    ids.push(cfg.sep_token_id);

    ids.truncate(max_len as usize);
    let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len as usize).collect();
    (ids, markers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_options_choice_uses_bare_label_for_falsy_description() {
        let q = Question::Choice {
            ins: "x".to_string(),
            options: vec![
                ("refund".to_string(), Some("wants money back".to_string())),
                ("complaint".to_string(), Some(String::new())),
                ("question".to_string(), None),
            ],
        };
        assert_eq!(q.render_options(), vec!["refund: wants money back", "complaint", "question"]);
    }

    #[test]
    fn render_options_score_is_level_indexed() {
        let q = Question::Score { ins: "x".to_string(), options: vec!["bad".to_string(), "ok".to_string(), "great".to_string()] };
        assert_eq!(q.render_options(), vec!["level 0: bad", "level 1: ok", "level 2: great"]);
    }

    #[test]
    fn render_options_noul_defaults_when_criteria_missing() {
        let q = Question::Noul { ins: "x".to_string(), false_text: None, true_text: None };
        assert_eq!(
            q.render_options(),
            vec!["false: no, the statement does not hold", "true: yes, the statement holds"]
        );
        let q2 = Question::Noul {
            ins: "x".to_string(),
            false_text: Some("nope".to_string()),
            true_text: Some(String::new()), // empty -> falls back to default
        };
        assert_eq!(q2.render_options(), vec!["false: nope", "true: yes, the statement holds"]);
    }

    #[test]
    fn write_json_matches_python_json_dumps_separators_and_order() {
        let v = OrderedJson::object(vec![
            ("user", OrderedJson::str("alice")),
            ("turns", OrderedJson::int(3)),
            ("tags", OrderedJson::array(vec![OrderedJson::str("billing"), OrderedJson::str("urgent")])),
        ]);
        let mut out = String::new();
        write_json(&v, &mut out);
        assert_eq!(out, r#"{"user": "alice", "turns": 3, "tags": ["billing", "urgent"]}"#);
    }

    #[test]
    fn write_json_does_not_escape_non_ascii() {
        let v = OrderedJson::str("café");
        let mut out = String::new();
        write_json(&v, &mut out);
        assert_eq!(out, "\"café\""); // ensure_ascii=False: raw UTF-8, no \uXXXX
    }

    /// The half [`write_json`] was missing: reading a state back off the
    /// wire. A caller who receives a request as TEXT (a JSON API, a file, a
    /// pipe) must be able to hand its `state` to [`build_sequence`] and
    /// tokenize the same bytes the Python reference would - which is only
    /// true if the parse preserves the key order the sender wrote, since
    /// `json.dumps` serializes a dict in insertion order. `serde_json`'s own
    /// `Value` sorts its keys, so a round-trip through it would silently
    /// tokenize something else.
    #[test]
    fn parse_preserves_key_order_through_a_round_trip() {
        let text = r#"{"z": 1, "a": {"n": null, "m": [true, false]}, "b": "x"}"#;
        let v = OrderedJson::parse(text).expect("valid JSON must parse");
        let mut out = String::new();
        write_json(&v, &mut out);
        assert_eq!(out, r#"{"z": 1, "a": {"n": null, "m": [true, false]}, "b": "x"}"#);
    }

    #[test]
    fn parse_reads_every_json_shape() {
        let v = OrderedJson::parse(r#"{"s": "t", "i": -3, "f": 1.5, "b": true, "n": null, "a": [1, "two"]}"#).unwrap();
        let OrderedJson::Object(entries) = &v else { panic!("expected an object, got {v:?}") };
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["s", "i", "f", "b", "n", "a"]);
        assert_eq!(entries[0].1, OrderedJson::str("t"));
        assert_eq!(entries[3].1, OrderedJson::Bool(true));
        assert_eq!(entries[4].1, OrderedJson::Null);
        let mut out = String::new();
        write_json(&entries[5].1, &mut out);
        assert_eq!(out, r#"[1, "two"]"#);
    }

    /// Non-ASCII survives the round trip verbatim, matching
    /// `ensure_ascii=False` - the property the writer already has and the
    /// reader must not undo.
    #[test]
    fn parse_round_trips_non_ascii_and_escapes() {
        let v = OrderedJson::parse(r#"{"k": "caf\u00e9 \"q\"\n"}"#).unwrap();
        let mut out = String::new();
        write_json(&v, &mut out);
        assert_eq!(out, "{\"k\": \"caf\u{e9} \\\"q\\\"\\n\"}");
    }

    /// A malformed request is an error a caller can report, not a panic.
    #[test]
    fn parse_rejects_malformed_json_with_a_message() {
        let err = OrderedJson::parse("{\"a\": }").unwrap_err();
        assert!(!err.is_empty(), "the error must say something");
        assert!(OrderedJson::parse("").is_err());
    }

    /// A parsed state tokenizes as the state it came from: the reason the key
    /// order matters at all.
    #[test]
    fn a_parsed_state_serializes_like_the_object_it_came_from() {
        let parsed = State::Json(OrderedJson::parse(r#"{"turn": 2, "actor": "customer"}"#).unwrap());
        let built = State::Json(OrderedJson::object(vec![
            ("turn", OrderedJson::int(2)),
            ("actor", OrderedJson::str("customer")),
        ]));
        assert_eq!(parsed.serialize(), built.serialize());
        assert_eq!(parsed.serialize(), r#"{"turn": 2, "actor": "customer"}"#);
    }
}
