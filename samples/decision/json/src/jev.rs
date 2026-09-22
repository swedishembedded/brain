// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The wire format: a JEV-style request in, a JEV-style response out.
//!
//! Split from `main.rs` because it is the half that has nothing to do with a
//! model - text to [`Question`]s and [`Answer`]s back to text - and is
//! therefore the half that can be tested without loading 843 MB of weights.
//!
//! The shape is TypeSafe's Jev decision API, which brain's own decision
//! primitives already match one-for-one (`choice`/`score`/`noul`, criteria
//! that are an ordered label map, an ordered level list, or a true/false
//! pair - see `decide::primitives`):
//!
//! ```json
//! {"state": {"ticket": "payments failing"},
//!  "questions": {"team":  {"type": "choice", "instructions": "who handles this",
//!                          "criteria": {"billing": "payments, refunds", "tech": "outages"}},
//!                "anger": {"type": "score",  "instructions": "how upset",
//!                          "criteria": ["calm", "annoyed", "furious"]},
//!                "urgent":{"type": "noul",   "instructions": "is this urgent"}}}
//! ```
//!
//! **Key order is preserved end to end** - [`OrderedJson`] rather than
//! `serde_json::Value`, because the state's key order is part of what the
//! model reads (see `modernbert::sequence`'s own module doc) and a sorted
//! re-serialization would silently ask about different bytes.
//!
//! **Unknown fields are ignored**, including a request's own `model`: which
//! model answers is decided by `--model` on the command line, since the
//! process has exactly one loaded. The response reports the one that actually
//! answered.
//!
//! Swedish Embedded AB implements typed decision endpoints - a model behind a
//! schema a service can rely on, rather than free text a caller has to parse
//! and hope about - for its clients. If your team needs one, you can procure
//! our services by sending an email to info@swedishembedded.com.

use brain::decision::{Answer, Opt, OrderedJson, Question, State};

/// One parsed request: what to judge, and what to ask about it.
#[derive(Debug)]
pub struct Request {
    pub state: State,
    /// The caller's own question names, in the order they were written, with
    /// the question each names. A `Vec`, not a map: the response is keyed by
    /// these names and the answers come back positionally.
    pub questions: Vec<(String, Question)>,
}

/// Parse a JEV-style request.
///
/// Every failure names the field it is about, because the caller is on the
/// other end of a pipe and cannot see this process.
pub fn parse_request(text: &str) -> Result<Request, String> {
    let root = OrderedJson::parse(text)?;
    let OrderedJson::Object(fields) = &root else {
        return Err("a request must be a JSON object with `state` and `questions`".into());
    };
    let state = match field(fields, "state") {
        Some(OrderedJson::String(s)) => State::Str(s.clone()),
        // Anything else is structured state, serialized the way the model's
        // own reference does it (`json.dumps`, insertion order).
        Some(other) => State::Json(other.clone()),
        None => return Err("`state` is required: it is what the questions are about".into()),
    };
    let Some(OrderedJson::Object(qs)) = field(fields, "questions") else {
        return Err("`questions` is required, and is an object of name -> question".into());
    };
    if qs.is_empty() {
        return Err("`questions` must ask at least one question".into());
    }

    let mut questions = Vec::with_capacity(qs.len());
    for (name, body) in qs {
        let q = parse_question(body).map_err(|e| format!("question {name:?}: {e}"))?;
        // The published limits (255 options, 2..10 levels), enforced here so
        // the name of the offending question travels with the complaint.
        q.validate().map_err(|e| format!("question {name:?}: {e}"))?;
        questions.push((name.clone(), q));
    }
    Ok(Request { state, questions })
}

fn parse_question(body: &OrderedJson) -> Result<Question, String> {
    let OrderedJson::Object(fields) = body else {
        return Err("a question must be an object".into());
    };
    let Some(OrderedJson::String(kind)) = field(fields, "type") else {
        return Err("`type` is required, and is one of \"choice\", \"score\", \"noul\"".into());
    };
    let instructions = match field(fields, "instructions") {
        Some(OrderedJson::String(s)) => s.clone(),
        // Jev documents instructions as "question text or structured
        // guidance", so a caller may send an object or a list; it reaches the
        // model as the text it serializes to rather than being refused.
        Some(other) => render(other),
        None => return Err("`instructions` is required: it is what the model is being asked".into()),
    };
    let criteria = field(fields, "criteria");

    match kind.as_str() {
        "choice" => {
            let Some(OrderedJson::Object(entries)) = criteria else {
                return Err("a choice needs `criteria`: an object of option name -> description".into());
            };
            let options = entries
                .iter()
                .map(|(name, desc)| match desc {
                    // A falsy description means "the label says it all" - the
                    // same rule the model's own option renderer applies.
                    OrderedJson::Null => Opt::new(name.as_str()),
                    OrderedJson::String(d) if d.is_empty() => Opt::new(name.as_str()),
                    OrderedJson::String(d) => Opt::described(name.as_str(), d.as_str()),
                    other => Opt::described(name.as_str(), render(other)),
                })
                .collect();
            Ok(Question::Choice { instructions, options })
        }
        "score" => {
            let Some(OrderedJson::Array(items)) = criteria else {
                return Err("a score needs `criteria`: an ordered list of level descriptions, worst first".into());
            };
            let levels = items
                .iter()
                .map(|l| match l {
                    OrderedJson::String(s) => s.clone(),
                    other => render(other),
                })
                .collect();
            Ok(Question::Score { instructions, levels })
        }
        "noul" => {
            // Criteria are optional here: the proposition alone is a question.
            let (mut yes, mut no) = (None, None);
            if let Some(OrderedJson::Object(entries)) = criteria {
                yes = text_of(entries, "true");
                no = text_of(entries, "false");
            }
            Ok(Question::Noul { instructions, yes, no })
        }
        other => Err(format!("unknown question type {other:?}: expected \"choice\", \"score\" or \"noul\"")),
    }
}

/// Render one request's answers as the response document, keyed by the
/// caller's own question names.
///
/// `names` and `answers` are the two halves of the same request, in the same
/// order - [`brain::DecisionPipeline::decide`] answers positionally.
pub fn render_response(model: &str, names: &[String], answers: &[Answer]) -> String {
    let mut entries: Vec<(String, OrderedJson)> = Vec::with_capacity(answers.len());
    for (name, answer) in names.iter().zip(answers) {
        entries.push((name.clone(), render_answer(answer)));
    }
    let doc = OrderedJson::Object(vec![
        ("model".to_string(), OrderedJson::str(model)),
        ("answers".to_string(), OrderedJson::Object(entries)),
    ]);
    render(&doc)
}

/// A failed request, as a document rather than a log line: one JSON object
/// per request on stdout whether it worked or not, so a consumer (a `jq`
/// filter, a line-oriented client in `--jsonl` mode) never has to guess
/// which request a missing line belonged to.
pub fn render_error(model: &str, message: &str) -> String {
    render(&OrderedJson::Object(vec![
        ("model".to_string(), OrderedJson::str(model)),
        (
            "error".to_string(),
            OrderedJson::Object(vec![("message".to_string(), OrderedJson::str(message))]),
        ),
    ]))
}

fn render_answer(answer: &Answer) -> OrderedJson {
    match answer {
        Answer::Choice { choice, probabilities, confidence } => OrderedJson::Object(vec![
            ("type".to_string(), OrderedJson::str("choice")),
            ("choice".to_string(), OrderedJson::str(choice)),
            ("confidence".to_string(), number(*confidence)),
            (
                "probabilities".to_string(),
                OrderedJson::Object(probabilities.iter().map(|(n, p)| (n.clone(), number(*p))).collect()),
            ),
        ]),
        Answer::Score { score, legend, probabilities, confidence } => OrderedJson::Object(vec![
            ("type".to_string(), OrderedJson::str("score")),
            ("score".to_string(), number(*score)),
            ("confidence".to_string(), number(*confidence)),
            // Keyed by level INDEX, since that is what `score` is a position
            // on; the text is what the caller sent, echoed back so a reader of
            // the response alone can interpret the number.
            (
                "legend".to_string(),
                OrderedJson::Object(legend.iter().enumerate().map(|(i, t)| (i.to_string(), OrderedJson::str(t))).collect()),
            ),
            (
                "probabilities".to_string(),
                OrderedJson::Object(probabilities.iter().enumerate().map(|(i, p)| (i.to_string(), number(*p))).collect()),
            ),
        ]),
        // No `confidence`: a two-outcome probability already is its own
        // confidence - see `decide::primitives::Answer::Noul`.
        Answer::Noul { noul } => OrderedJson::Object(vec![
            ("type".to_string(), OrderedJson::str("noul")),
            ("noul".to_string(), number(*noul)),
        ]),
    }
}

/// Probabilities to 6 decimals: enough to see a 1e-5 tail, few enough that a
/// response diffs cleanly and does not advertise float noise as precision.
fn number(v: f32) -> OrderedJson {
    let rounded = (v as f64 * 1e6).round() / 1e6;
    serde_json::Number::from_f64(rounded).map(OrderedJson::Number).unwrap_or(OrderedJson::Null)
}

fn render(v: &OrderedJson) -> String {
    let mut out = String::new();
    brain::decision::write_json(v, &mut out);
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

    /// The request from Jev's own documentation, which is the shape this
    /// sample exists to accept: three question types at once, a plain-text
    /// state, criteria in three different notations.
    const JEV_EXAMPLE: &str = r#"{
      "model": "typesafe/jev",
      "state": "Help! My payments have been failing for 3 days and nobody answers support.",
      "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations",
                                    "sales": "Pricing, upgrades, new accounts"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]}
      }
    }"#;

    #[test]
    fn the_documented_jev_request_parses_into_three_typed_questions_in_order() {
        let req = parse_request(JEV_EXAMPLE).expect("the documented request must parse");
        let names: Vec<&str> = req.questions.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["is_urgent", "department", "frustration"]);

        match &req.questions[0].1 {
            Question::Noul { instructions, yes, no } => {
                assert_eq!(instructions, "Does this convey urgency?");
                assert!(yes.is_none() && no.is_none(), "absent criteria stay absent, so the model's own defaults apply");
            }
            other => panic!("expected a noul, got {other:?}"),
        }
        match &req.questions[1].1 {
            Question::Choice { options, .. } => {
                // Option ORDER is the caller's, not sorted: a caller maps the
                // answer back onto its own list.
                let names: Vec<&str> = options.iter().map(|o| o.name.as_str()).collect();
                assert_eq!(names, ["billing", "technical", "sales"]);
                assert_eq!(options[0].description.as_deref(), Some("Payments, invoicing, refunds"));
            }
            other => panic!("expected a choice, got {other:?}"),
        }
        match &req.questions[2].1 {
            Question::Score { levels, .. } => assert_eq!(levels, &["Calm".to_string(), "Frustrated".to_string(), "Very angry".to_string()]),
            other => panic!("expected a score, got {other:?}"),
        }
    }

    /// Structured state reaches the model as the JSON the caller wrote, in
    /// the order they wrote it - the property the whole `OrderedJson` path
    /// exists for.
    #[test]
    fn a_structured_state_keeps_its_key_order() {
        let req = parse_request(r#"{"state": {"turn": 2, "actor": "customer", "amount": 41.5},
                                    "questions": {"q": {"type": "noul", "instructions": "ok?"}}}"#)
            .unwrap();
        assert_eq!(req.state.serialize(), r#"{"turn": 2, "actor": "customer", "amount": 41.5}"#);
    }

    #[test]
    fn a_text_state_is_passed_through_verbatim() {
        let req = parse_request(r#"{"state": "just words", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}"#).unwrap();
        assert_eq!(req.state.serialize(), "just words");
    }

    #[test]
    fn a_nouls_criteria_become_its_two_readings() {
        let req = parse_request(
            r#"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "did it close?",
                "criteria": {"true": "the deal closed", "false": "it did not"}}}}"#,
        )
        .unwrap();
        match &req.questions[0].1 {
            Question::Noul { yes, no, .. } => {
                assert_eq!(yes.as_deref(), Some("the deal closed"));
                assert_eq!(no.as_deref(), Some("it did not"));
            }
            other => panic!("expected a noul, got {other:?}"),
        }
    }

    /// Every rejection names the field - and, when there is one, the question.
    #[test]
    fn malformed_requests_are_refused_by_field_name() {
        let cases = [
            (r#"{"questions": {"q": {"type": "noul", "instructions": "x"}}}"#, "state"),
            (r#"{"state": "s"}"#, "questions"),
            (r#"{"state": "s", "questions": {}}"#, "questions"),
            (r#"{"state": "s", "questions": {"q": {"instructions": "x"}}}"#, "type"),
            (r#"{"state": "s", "questions": {"q": {"type": "noul"}}}"#, "instructions"),
            (r#"{"state": "s", "questions": {"q": {"type": "vibes", "instructions": "x"}}}"#, "vibes"),
            (r#"{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x"}}}"#, "criteria"),
            (r#"{"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}}"#, "levels"),
            ("not json at all", "expected"),
        ];
        for (req, needle) in cases {
            let err = parse_request(req).unwrap_err();
            assert!(err.contains(needle), "{req}\n  expected an error mentioning {needle:?}, got: {err}");
        }
        // And the question's own name travels with the complaint.
        let err = parse_request(r#"{"state": "s", "questions": {"team": {"type": "choice", "instructions": "x"}}}"#).unwrap_err();
        assert!(err.contains("\"team\""), "the failing question must be named: {err}");
    }

    /// The response is one line of JSON, keyed by the caller's names, with
    /// each answer carrying the fields its type is documented to carry.
    #[test]
    fn the_response_is_keyed_by_the_callers_question_names() {
        let names = vec!["urgent".to_string(), "team".to_string(), "anger".to_string()];
        let answers = vec![
            Answer::Noul { noul: 0.9612345 },
            Answer::Choice {
                choice: "billing".into(),
                probabilities: vec![("billing".into(), 0.98), ("technical".into(), 0.02)],
                confidence: 0.97,
            },
            Answer::Score {
                score: 1.3,
                legend: vec!["calm".into(), "annoyed".into()],
                probabilities: vec![0.7, 0.3],
                confidence: 0.55,
            },
        ];
        let out = render_response("laya", &names, &answers);
        assert!(!out.contains('\n'), "a response is one line, so a stream of them is line-oriented: {out}");

        let back = OrderedJson::parse(&out).expect("the response must be valid JSON");
        let OrderedJson::Object(top) = &back else { panic!("expected an object") };
        assert_eq!(field(top, "model"), Some(&OrderedJson::str("laya")));
        let Some(OrderedJson::Object(ans)) = field(top, "answers") else { panic!("expected answers") };
        let keys: Vec<&str> = ans.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["urgent", "team", "anger"]);

        assert!(out.contains(r#""noul": 0.961235"#), "probabilities are rounded, not float noise: {out}");
        assert!(out.contains(r#""choice": "billing""#), "{out}");
        assert!(out.contains(r#""legend": {"0": "calm", "1": "annoyed"}"#), "a score echoes its own scale: {out}");
    }

    #[test]
    fn a_failed_request_is_itself_a_json_document() {
        let out = render_error("laya", "`state` is required");
        let back = OrderedJson::parse(&out).expect("an error must still be valid JSON");
        let OrderedJson::Object(top) = &back else { panic!("expected an object") };
        let Some(OrderedJson::Object(err)) = field(top, "error") else { panic!("expected an error object") };
        assert_eq!(field(err, "message"), Some(&OrderedJson::str("`state` is required")));
    }
}
