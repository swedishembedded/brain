// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tool-calling: a faithful Qwen function-call format, a dataset generator, and
//! a scorer — the domain layer for training AND evaluating a model's ability to
//! make intelligent tool calls.
//!
//! ## Format (matches Qwen2.5/3's `tool_call` convention)
//!
//! The system turn advertises the available functions inside `<tools>` tags as
//! JSON schemas; the assistant answers with a `<tool_call>` block naming one
//! function and its arguments:
//!
//! ```text
//! <|im_start|>system
//! # Tools
//! You may call one function to assist with the user query.
//! <tools>
//! {"type":"function","function":{"name":"get_weather","description":"…",
//!   "parameters":{"type":"object","properties":{"location":{"type":"string",…}},
//!   "required":["location"]}}}
//! …
//! </tools>
//! For each call, return a JSON object with the function name and arguments
//! inside <tool_call></tool_call>.<|im_end|>
//! <|im_start|>user
//! What's the weather in Paris?<|im_end|>
//! <|im_start|>assistant
//! <tool_call>
//! {"name":"get_weather","arguments":{"location":"Paris"}}
//! </tool_call><|im_end|>
//! ```
//!
//! The *skill* being trained/measured is **routing + argument filling**: given
//! several candidate tools (the right one plus distractors) and a natural-
//! language request, name the correct function and copy each argument's value
//! from the request into the tool's canonical argument. That is exactly what the
//! generator below produces and what the scorer checks.

use serde_json::{json, Value};

use crate::chat::ChatExample;
use crate::rng::Rng;

/// One parameter of a tool's signature.
#[derive(Clone, Debug)]
pub struct ToolParam {
    pub name: String,
    pub ty: &'static str, // "string" | "number" | ...
    pub description: String,
}

/// A function/tool the model may call.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: Vec<ToolParam>,
}

impl ToolSpec {
    /// The JSON schema object (`{"type":"function","function":{…}}`) advertised
    /// in the system `<tools>` block.
    pub fn schema(&self) -> Value {
        let mut props = serde_json::Map::new();
        for p in &self.params {
            props.insert(p.name.clone(), json!({ "type": p.ty, "description": p.description }));
        }
        let required: Vec<&str> = self.params.iter().map(|p| p.name.as_str()).collect();
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": { "type": "object", "properties": props, "required": required }
            }
        })
    }
}

/// A concrete call: a tool name plus a JSON object of arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// The assistant response body (goes inside `<|im_start|>assistant\n … <|im_end|>`).
    pub fn response_str(&self) -> String {
        // Compact, deterministic JSON (sorted keys via BTreeMap ordering in the
        // arguments Value if the caller built it sorted; we serialise as-is).
        format!(
            "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
            self.name,
            serde_json::to_string(&self.arguments).unwrap_or_else(|_| "{}".into())
        )
    }
}

/// One tool-call training/eval case: the candidate tools, the user request, and
/// the ground-truth call.
#[derive(Clone, Debug)]
pub struct ToolCase {
    pub tools: Vec<ToolSpec>,
    pub user: String,
    pub call: ToolCall,
}

/// The Qwen tool-call system prompt advertising `tools`.
pub fn system_prompt(tools: &[ToolSpec]) -> String {
    let mut s = String::from(
        "# Tools\n\nYou may call one function to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        s.push_str(&serde_json::to_string(&t.schema()).unwrap());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, return a json object with function name \
         and arguments within <tool_call></tool_call> XML tags.",
    );
    s
}

impl ToolCase {
    /// Render as a masked chat example (system=tools, user=request, assistant=call).
    pub fn to_chat_example(&self) -> ChatExample {
        ChatExample::with_system(system_prompt(&self.tools), self.user.clone(), self.call.response_str())
    }
}

/// Parse the first `<tool_call>{…}</tool_call>` from generated text into a
/// [`ToolCall`]. Tolerant of whitespace and a missing closing tag (greedy decode
/// may stop early). Returns `None` if no parseable call is present.
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
    let start = text.find("<tool_call>")? + "<tool_call>".len();
    let rest = &text[start..];
    let body = match rest.find("</tool_call>") {
        Some(e) => &rest[..e],
        None => rest,
    };
    // Extract the first balanced {...} JSON object.
    let obj_start = body.find('{')?;
    let mut depth = 0i32;
    let mut end = None;
    for (i, c) in body[obj_start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(obj_start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let obj = &body[obj_start..end?];
    let v: Value = serde_json::from_str(obj).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments").cloned().unwrap_or_else(|| json!({}));
    Some(ToolCall { name, arguments })
}

/// True iff two calls match: same function name and the same argument
/// name→value map (string-compared after trimming, order-independent).
pub fn calls_match(expected: &ToolCall, got: &ToolCall) -> bool {
    if expected.name != got.name {
        return false;
    }
    let (Some(e), Some(g)) = (expected.arguments.as_object(), got.arguments.as_object()) else {
        return expected.arguments == got.arguments;
    };
    if e.len() != g.len() {
        return false;
    }
    e.iter().all(|(k, ev)| g.get(k).map(|gv| val_eq(ev, gv)).unwrap_or(false))
}

/// Loose scalar equality: numbers compare numerically, strings after trim; falls
/// back to structural equality.
fn val_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) => x.trim() == y.trim(),
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::String(x), Value::Number(y)) | (Value::Number(y), Value::String(x)) => {
            x.trim().parse::<f64>().ok() == y.as_f64()
        }
        _ => a == b,
    }
}

// ---- deterministic generator ---------------------------------------------------

/// The fixed tool catalogue the generator draws from. Each has 1-2 args whose
/// values are copied from the request — the routing+filling skill. Distractor
/// tools (the ones NOT called) force the model to select by name, not position.
fn catalogue() -> Vec<(ToolSpec, Vec<(&'static str, &'static [&'static str])>)> {
    // (spec, per-arg value pool). Values are drawn from disjoint pools so a wrong
    // routing produces a wrong, checkable answer.
    let p = |name: &str, ty: &'static str, desc: &str| ToolParam {
        name: name.into(),
        ty,
        description: desc.into(),
    };
    vec![
        (
            ToolSpec { name: "get_weather".into(), description: "Get the current weather for a city.".into(),
                params: vec![p("location", "string", "City name")] },
            vec![("location", &["Paris", "Tokyo", "Cairo", "Oslo", "Lima", "Delhi"])],
        ),
        (
            ToolSpec { name: "set_timer".into(), description: "Start a timer for a number of minutes.".into(),
                params: vec![p("minutes", "number", "Duration in minutes")] },
            vec![("minutes", &["3", "5", "10", "15", "20", "45"])],
        ),
        (
            ToolSpec { name: "play_music".into(), description: "Play a song by a given artist.".into(),
                params: vec![p("artist", "string", "Artist name"), p("song", "string", "Song title")] },
            vec![("artist", &["Adele", "Queen", "Prince", "Bjork"]), ("song", &["Hello", "One", "Kiss", "Joga"])],
        ),
        (
            ToolSpec { name: "send_email".into(), description: "Send an email to a recipient.".into(),
                params: vec![p("to", "string", "Recipient"), p("subject", "string", "Subject line")] },
            vec![("to", &["alice", "bob", "carol", "dave"]), ("subject", &["lunch", "budget", "recap", "hello"])],
        ),
        (
            ToolSpec { name: "convert_currency".into(), description: "Convert an amount between currencies.".into(),
                params: vec![p("amount", "number", "Amount"), p("to", "string", "Target currency")] },
            vec![("amount", &["10", "50", "100", "250"]), ("to", &["USD", "EUR", "JPY", "GBP"])],
        ),
    ]
}

/// A natural-language request template per tool that mentions each arg value, so
/// the model must copy values from free text (not a fixed slot).
fn phrase(name: &str, args: &serde_json::Map<String, Value>) -> String {
    let g = |k: &str| args.get(k).and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| v.as_i64().map(|n| n.to_string()))).unwrap_or_default();
    match name {
        "get_weather" => format!("What's the weather like in {} right now?", g("location")),
        "set_timer" => format!("Please start a timer for {} minutes.", g("minutes")),
        "play_music" => format!("Can you play the song {} by {}?", g("song"), g("artist")),
        "send_email" => format!("Send an email to {} with the subject {}.", g("to"), g("subject")),
        "convert_currency" => format!("Convert {} into {} for me.", g("amount"), g("to")),
        _ => String::new(),
    }
}

/// Generate `n` tool-call cases. Each case presents `n_tools` candidate tools
/// (the target + distractors), a request, and the ground-truth call. Numeric
/// args are emitted as JSON numbers, string args as strings.
pub fn generate(n: usize, n_tools: usize, seed: u64) -> Vec<ToolCase> {
    let cat = catalogue();
    let n_tools = n_tools.clamp(1, cat.len());
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // pick the target tool + (n_tools-1) distractors
        let ti = rng.gen_range_inclusive(0, cat.len() as i64 - 1) as usize;
        let mut idxs = vec![ti];
        while idxs.len() < n_tools {
            let j = rng.gen_range_inclusive(0, cat.len() as i64 - 1) as usize;
            if !idxs.contains(&j) {
                idxs.push(j);
            }
        }
        // shuffle presentation order so position carries no signal
        for i in (1..idxs.len()).rev() {
            let j = rng.gen_range_inclusive(0, i as i64) as usize;
            idxs.swap(i, j);
        }
        let tools: Vec<ToolSpec> = idxs.iter().map(|&i| cat[i].0.clone()).collect();

        // sample argument values for the target
        let (spec, pools) = &cat[ti];
        let mut args = serde_json::Map::new();
        for (arg, pool) in pools {
            let v = pool[rng.gen_range_inclusive(0, pool.len() as i64 - 1) as usize];
            // numeric params -> JSON number
            let is_num = spec.params.iter().find(|p| &p.name == arg).map(|p| p.ty == "number").unwrap_or(false);
            let jv = if is_num { json!(v.parse::<i64>().unwrap_or(0)) } else { json!(v) };
            args.insert((*arg).to_string(), jv);
        }
        let user = phrase(&spec.name, &args);
        out.push(ToolCase {
            tools,
            user,
            call: ToolCall { name: spec.name.clone(), arguments: Value::Object(args) },
        });
    }
    out
}
