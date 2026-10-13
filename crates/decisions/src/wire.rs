// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The request wire: text in, typed [`Question`]s and a [`State`] out.
//!
//! Two spellings of the same questions are accepted and normalise to one
//! [`Request`]:
//!
//! * **native**, an array, each question carrying its own `id`:
//!
//! ```json
//! [{"id": "team",  "type": "choice", "instruction": "who handles this",
//!   "options": [{"id": "billing", "description": "payments, refunds"}, "tech"]},
//!  {"id": "anger", "type": "score",  "instruction": "how upset",
//!   "options": ["calm", "annoyed", "furious"]},
//!  {"id": "urgent","type": "noul",   "instruction": "is this urgent",
//!   "criteria": {"true": "someone is blocked", "false": "it can wait"}}]
//! ```
//!
//! * **Jev-style**, an object keyed by question name, whose `criteria` is an
//!   ordered label map (choice), an ordered list (score) or a true/false pair
//!   (noul); `instructions` is accepted for `instruction`.
//!
//! **Key order is preserved end to end** ([`OrderedJson`], not
//! `serde_json::Value`): the state's key order is part of what a model reads,
//! and an option's position is how a caller maps an answer back onto its own
//! list, so a sorted re-serialization would silently ask about different bytes.
//!
//! Every failure names the field it is about, because the caller is on the
//! other end of a wire and cannot see this process.
//!
//! Swedish Embedded AB implements typed decision endpoints, a model behind a
//! schema a service can rely on rather than free text a caller has to parse.
//! If your team needs one, you can procure our services by sending an email to
//! info@swedishembedded.com.

use decide::primitives::{Opt, Question};
use modernbert::{write_json, OrderedJson, State};

/// Most questions one request may carry.
pub const MAX_QUESTIONS: usize = 64;
/// Largest state, in bytes, one request may carry.
pub const MAX_STATE_BYTES: usize = 1 << 20;

/// How a request's `state` text is to be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateFormat {
    /// The text is the state, verbatim.
    Text,
    /// The text is a JSON document: a JSON string is a text state, anything
    /// else is structured state serialized in the caller's key order.
    Json,
}

impl StateFormat {
    pub fn parse(s: &str) -> Option<StateFormat> {
        match s {
            "text" => Some(StateFormat::Text),
            "json" => Some(StateFormat::Json),
            _ => None,
        }
    }
}

/// The three question types a decision model may be asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    pub const ALL: [QuestionType; 3] = [QuestionType::Choice, QuestionType::Score, QuestionType::Noul];

    pub fn of(q: &Question) -> QuestionType {
        match q {
            Question::Choice { .. } => QuestionType::Choice,
            Question::Score { .. } => QuestionType::Score,
            Question::Noul { .. } => QuestionType::Noul,
        }
    }

    /// The wire name.
    pub fn slug(self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }

    pub fn parse(s: &str) -> Option<QuestionType> {
        QuestionType::ALL.into_iter().find(|t| t.slug() == s)
    }
}

/// One question and the caller's name for it.
#[derive(Clone, Debug)]
pub struct NamedQuestion {
    pub id: String,
    pub question: Question,
}

/// A parsed request: what to judge, and what to ask about it.
#[derive(Clone, Debug)]
pub struct Request {
    pub state: State,
    pub questions: Vec<NamedQuestion>,
}

/// The state, read per `format`, refusing one over [`MAX_STATE_BYTES`].
pub fn parse_state(text: &str, format: StateFormat) -> Result<State, String> {
    if text.len() > MAX_STATE_BYTES {
        return Err(format!("`state` is {} bytes; at most {MAX_STATE_BYTES} are accepted", text.len()));
    }
    match format {
        StateFormat::Text => Ok(State::Str(text.to_string())),
        StateFormat::Json => Ok(match OrderedJson::parse(text).map_err(|e| format!("`state` is not valid JSON: {e}"))? {
            OrderedJson::String(s) => State::Str(s),
            other => State::Json(other),
        }),
    }
}

/// The questions of either spelling, validated against the published limits.
pub fn parse_questions(text: &str) -> Result<Vec<NamedQuestion>, String> {
    let root = OrderedJson::parse(text).map_err(|e| format!("`questions` is not valid JSON: {e}"))?;
    let named: Vec<(Option<String>, &OrderedJson)> = match &root {
        OrderedJson::Array(items) => items.iter().map(|b| (None, b)).collect(),
        OrderedJson::Object(entries) => entries.iter().map(|(k, b)| (Some(k.clone()), b)).collect(),
        _ => return Err("`questions` must be an array of questions or an object of name -> question".into()),
    };
    if named.is_empty() {
        return Err("`questions` must ask at least one question".into());
    }
    if named.len() > MAX_QUESTIONS {
        return Err(format!("`questions` carries {}; at most {MAX_QUESTIONS} are accepted", named.len()));
    }
    let mut out: Vec<NamedQuestion> = Vec::with_capacity(named.len());
    for (n, (key, body)) in named.into_iter().enumerate() {
        let label = key.clone().unwrap_or_else(|| format!("#{n}"));
        let (id, question) = parse_question(body, key).map_err(|e| format!("question {label:?}: {e}"))?;
        question.validate().map_err(|e| format!("question {id:?}: {e}"))?;
        if out.iter().any(|q| q.id == id) {
            return Err(format!("question {id:?}: the id is used twice; answers are matched by id"));
        }
        out.push(NamedQuestion { id, question });
    }
    Ok(out)
}

fn parse_question(body: &OrderedJson, key: Option<String>) -> Result<(String, Question), String> {
    let OrderedJson::Object(fields) = body else {
        return Err("a question must be an object".into());
    };
    let id = match (key, field(fields, "id")) {
        (Some(k), _) => k,
        (None, Some(OrderedJson::String(s))) if !s.is_empty() => s.clone(),
        (None, _) => return Err("`id` is required, and is a non-empty string".into()),
    };
    let Some(OrderedJson::String(kind)) = field(fields, "type") else {
        return Err("`type` is required, and is one of \"choice\", \"score\", \"noul\"".into());
    };
    let instructions = match field(fields, "instruction").or_else(|| field(fields, "instructions")) {
        Some(OrderedJson::String(s)) => s.clone(),
        // Jev documents instructions as "question text or structured
        // guidance": an object or list reaches the model as the text it
        // serializes to rather than being refused.
        Some(other) => render(other),
        None => return Err("`instruction` is required: it is what the model is being asked".into()),
    };
    let candidates = field(fields, "options").or_else(|| field(fields, "criteria"));

    let question = match kind.as_str() {
        "choice" => {
            let options = match candidates {
                // Jev: an ORDERED label -> description map.
                Some(OrderedJson::Object(entries)) => entries.iter().map(|(name, desc)| option(name, Some(desc))).collect(),
                Some(OrderedJson::Array(items)) => items.iter().map(option_from_item).collect::<Result<Vec<_>, _>>()?,
                _ => return Err("a choice needs `options`: an array of option ids (or {id, description}), or `criteria`: an ordered object of option id -> description".into()),
            };
            Question::Choice { instructions, options }
        }
        "score" => {
            let Some(OrderedJson::Array(items)) = candidates else {
                return Err("a score needs `options` (or `criteria`): an ordered list of level descriptions, worst first".into());
            };
            let levels = items
                .iter()
                .map(|l| match l {
                    OrderedJson::String(s) => s.clone(),
                    other => render(other),
                })
                .collect();
            Question::Score { instructions, levels }
        }
        "noul" => {
            // Criteria are optional: the proposition alone is a question.
            let (mut yes, mut no) = (None, None);
            if let Some(OrderedJson::Object(entries)) = field(fields, "criteria") {
                yes = text_of(entries, "true");
                no = text_of(entries, "false");
            }
            Question::Noul { instructions, yes, no }
        }
        other => return Err(format!("unknown question type {other:?}: expected \"choice\", \"score\" or \"noul\"")),
    };
    Ok((id, question))
}

/// An option from a native `options` item: a bare id, or `{id, description}`.
fn option_from_item(item: &OrderedJson) -> Result<Opt, String> {
    match item {
        OrderedJson::String(name) => Ok(Opt::new(name.as_str())),
        OrderedJson::Object(fields) => {
            let Some(OrderedJson::String(name)) = field(fields, "id").or_else(|| field(fields, "name")) else {
                return Err("an option object needs an `id`".into());
            };
            Ok(option(name, field(fields, "description")))
        }
        _ => Err("an option is a string or an object with an `id`".into()),
    }
}

fn option(name: &str, description: Option<&OrderedJson>) -> Opt {
    match description {
        // A falsy description means "the label says it all" - the rule the
        // model's own option renderer applies.
        None | Some(OrderedJson::Null) => Opt::new(name),
        Some(OrderedJson::String(d)) if d.is_empty() => Opt::new(name),
        Some(OrderedJson::String(d)) => Opt::described(name, d.as_str()),
        Some(other) => Opt::described(name, render(other)),
    }
}

/// A parsed `{state, questions}` pair as one value.
pub fn parse_request(state: &str, format: StateFormat, questions: &str) -> Result<Request, String> {
    Ok(Request { state: parse_state(state, format)?, questions: parse_questions(questions)? })
}

fn render(v: &OrderedJson) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn field<'a>(fields: &'a [(String, OrderedJson)], name: &str) -> Option<&'a OrderedJson> {
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn text_of(fields: &[(String, OrderedJson)], name: &str) -> Option<String> {
    match field(fields, name) {
        Some(OrderedJson::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NATIVE: &str = r#"[
      {"id": "team", "type": "choice", "instruction": "who handles this",
       "options": [{"id": "billing", "description": "payments, refunds"}, "tech", {"id": "sales", "description": ""}]},
      {"id": "anger", "type": "score", "instruction": "how upset", "options": ["calm", "annoyed", "furious"]},
      {"id": "urgent", "type": "noul", "instruction": "is this urgent",
       "criteria": {"true": "someone is blocked", "false": "it can wait"}}
    ]"#;

    const JEV: &str = r#"{
      "department": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "Payments", "technical": null, "sales": ""}},
      "frustration": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]},
      "is_urgent": {"type": "noul", "instructions": "Urgent?"}
    }"#;

    #[test]
    fn the_native_spelling_parses_into_three_typed_questions_in_order() {
        let qs = parse_questions(NATIVE).unwrap();
        let ids: Vec<&str> = qs.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["team", "anger", "urgent"]);
        match &qs[0].question {
            Question::Choice { instructions, options } => {
                assert_eq!(instructions, "who handles this");
                let names: Vec<&str> = options.iter().map(|o| o.name.as_str()).collect();
                assert_eq!(names, ["billing", "tech", "sales"], "option order is the caller's");
                assert_eq!(options[0].description.as_deref(), Some("payments, refunds"));
                assert_eq!(options[1].description, None);
                assert_eq!(options[2].description, None, "an empty description means the label says it all");
            }
            other => panic!("expected a choice, got {other:?}"),
        }
        match &qs[1].question {
            Question::Score { levels, .. } => assert_eq!(levels, &["calm", "annoyed", "furious"]),
            other => panic!("expected a score, got {other:?}"),
        }
        match &qs[2].question {
            Question::Noul { yes, no, .. } => {
                assert_eq!(yes.as_deref(), Some("someone is blocked"));
                assert_eq!(no.as_deref(), Some("it can wait"));
            }
            other => panic!("expected a noul, got {other:?}"),
        }
    }

    #[test]
    fn the_jev_spelling_parses_to_the_same_vocabulary_with_names_as_ids() {
        let qs = parse_questions(JEV).unwrap();
        let ids: Vec<&str> = qs.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["department", "frustration", "is_urgent"]);
        match &qs[0].question {
            Question::Choice { options, .. } => {
                let names: Vec<&str> = options.iter().map(|o| o.name.as_str()).collect();
                assert_eq!(names, ["billing", "technical", "sales"]);
                assert_eq!(options[0].description.as_deref(), Some("Payments"));
                assert_eq!(options[1].description, None, "null means the label says it all");
            }
            other => panic!("expected a choice, got {other:?}"),
        }
        assert!(matches!(&qs[2].question, Question::Noul { yes: None, no: None, .. }));
    }

    #[test]
    fn a_structured_state_keeps_its_key_order_and_a_json_string_is_text() {
        let s = parse_state(r#"{"turn": 2, "actor": "customer", "amount": 41.5}"#, StateFormat::Json).unwrap();
        assert_eq!(s.serialize(), r#"{"turn": 2, "actor": "customer", "amount": 41.5}"#);
        let t = parse_state(r#""just words""#, StateFormat::Json).unwrap();
        assert_eq!(t.serialize(), "just words");
    }

    #[test]
    fn a_text_state_is_taken_verbatim_even_when_it_looks_like_json() {
        let s = parse_state("{not json", StateFormat::Text).unwrap();
        assert_eq!(s.serialize(), "{not json");
        assert!(parse_state("{not json", StateFormat::Json).unwrap_err().contains("`state`"));
    }

    #[test]
    fn an_oversized_state_is_refused() {
        let big = "x".repeat(MAX_STATE_BYTES + 1);
        assert!(parse_state(&big, StateFormat::Text).unwrap_err().contains("at most"));
    }

    #[test]
    fn malformed_questions_are_refused_by_field_name() {
        let cases = [
            ("[]", "at least one"),
            ("{}", "at least one"),
            ("7", "array"),
            ("not json", "`questions`"),
            (r#"[{"type":"noul","instruction":"x"}]"#, "`id`"),
            (r#"[{"id":"q","instruction":"x"}]"#, "`type`"),
            (r#"[{"id":"q","type":"noul"}]"#, "`instruction`"),
            (r#"[{"id":"q","type":"vibes","instruction":"x"}]"#, "vibes"),
            (r#"[{"id":"q","type":"choice","instruction":"x"}]"#, "`options`"),
            (r#"[{"id":"q","type":"choice","instruction":"x","options":[{"description":"d"}]}]"#, "`id`"),
            (r#"[{"id":"q","type":"score","instruction":"x","options":["only"]}]"#, "levels"),
            (r#"[{"id":"q","type":"noul","instruction":"x"},{"id":"q","type":"noul","instruction":"y"}]"#, "twice"),
        ];
        for (req, needle) in cases {
            let err = parse_questions(req).unwrap_err();
            assert!(err.contains(needle), "{req}\n  expected {needle:?} in: {err}");
        }
        let err = parse_questions(r#"{"team": {"type": "choice", "instructions": "x"}}"#).unwrap_err();
        assert!(err.contains("\"team\""), "the failing question must be named: {err}");
    }

    #[test]
    fn the_published_limits_are_enforced_at_the_wire() {
        let many: Vec<String> = (0..256).map(|i| format!("\"o{i}\"")).collect();
        let over = format!(r#"[{{"id":"q","type":"choice","instruction":"x","options":[{}]}}]"#, many.join(","));
        assert!(parse_questions(&over).unwrap_err().contains("255"));
        let ok = format!(r#"[{{"id":"q","type":"choice","instruction":"x","options":[{}]}}]"#, many[..255].join(","));
        assert!(parse_questions(&ok).is_ok());

        let levels = |n: usize| {
            let l: Vec<String> = (0..n).map(|i| format!("\"l{i}\"")).collect();
            format!(r#"[{{"id":"q","type":"score","instruction":"x","options":[{}]}}]"#, l.join(","))
        };
        assert!(parse_questions(&levels(11)).unwrap_err().contains("10"));
        assert!(parse_questions(&levels(1)).is_err());
        assert!(parse_questions(&levels(2)).is_ok());
        assert!(parse_questions(&levels(10)).is_ok());
    }

    #[test]
    fn too_many_questions_are_refused() {
        let one = r#"{"id":"q%","type":"noul","instruction":"x"}"#;
        let qs: Vec<String> = (0..=MAX_QUESTIONS).map(|i| one.replace('%', &i.to_string())).collect();
        let err = parse_questions(&format!("[{}]", qs.join(","))).unwrap_err();
        assert!(err.contains(&MAX_QUESTIONS.to_string()), "{err}");
    }
}
