// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Qwen3 chat template, reproduced **exactly** (byte-for-byte) from the
//! checkpoint's own Jinja (`tokenizer_config.json`'s `chat_template`), plus a
//! streaming scanner that turns generated text back into content/reasoning/
//! tool-call events.
//!
//! This is a *second*, correct renderer alongside
//! [`crate::qwen_tokenizer::QwenBpe::apply_chat_template`]. That older function
//! is a plain-string approximation kept for backward compatibility (FLUX.2's
//! text-conditioning prompt, `zimage`, `qwen_cli`, `caps.rs`, `chat.rs` all
//! byte-depend on its exact — non-Jinja-exact — output) and MUST NOT change.
//! [`render`] is the one to reach for whenever the true Jinja semantics matter:
//! tool calls, multi-turn `reasoning_content` splitting, and the `<tool_call>`/
//! `<tool_response>` framing the older function never implemented.
//!
//! ## Two halves
//!
//! * [`render`] / [`render_for_generation`] — messages (+ tools) → prompt text,
//!   following the Jinja template's control flow line for line (see each
//!   function's doc comment for the exact rules).
//! * [`ChatScanner`] / [`split_output`] — generated text → `(content,
//!   reasoning, tool_calls)`, streaming-safe: a marker (`<think>`, `<tool_call>`,
//!   …) split across two [`ChatScanner::push`] calls — as happens with real
//!   token-by-token generation — is never leaked into the visible output.
//!
//! [`crate::toolcall::parse_tool_call`] (the training-time dataset scorer) is
//! rewired to delegate to [`split_output`] rather than duplicating a JSON
//! scanner.

use crate::toolcall::ToolCall;

// ===================== message model =====================

/// A chat turn's role, in Qwen3's four-role vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Tool,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One tool call in an assistant turn. `arguments` is the RAW JSON text exactly
/// as received/generated — never re-parsed-and-reserialized, which would lose
/// key order and separator style and could desync from what the model actually
/// emitted (the Jinja template itself takes this same "already a string" path:
/// `{%- if tool_call.arguments is string %}{{- tool_call.arguments }}`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ToolCallMsg {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One chat message. `reasoning_content` and `tool_calls` are assistant-only;
/// `tool_call_id` is meaningful only on [`Role::Tool`] (which call this answers
/// — not read by [`render`] itself, since the Jinja template groups consecutive
/// `tool` messages positionally rather than by id, but carried for callers that
/// need to correlate results back to requests).
#[derive(Clone, Debug, Default)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallMsg>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(c: impl Into<String>) -> ChatMessage {
        ChatMessage { role: Role::System, content: c.into(), ..Default::default() }
    }
    pub fn user(c: impl Into<String>) -> ChatMessage {
        ChatMessage { role: Role::User, content: c.into(), ..Default::default() }
    }
    pub fn assistant(c: impl Into<String>) -> ChatMessage {
        ChatMessage { role: Role::Assistant, content: c.into(), ..Default::default() }
    }
    pub fn tool(c: impl Into<String>) -> ChatMessage {
        ChatMessage { role: Role::Tool, content: c.into(), ..Default::default() }
    }
    pub fn with_tool_calls(mut self, calls: Vec<ToolCallMsg>) -> ChatMessage {
        self.tool_calls = calls;
        self
    }
}

/// Which upstream template generation [`render`] transcribes. Qwen rewrote
/// several behaviors between the Qwen3 and Qwen3.8 releases, and both remain
/// live (the dense Qwen3 checkpoints still ship the old template; Qwen3.8
/// ships the new one), so the port carries both, selected per model family:
///
/// | behavior | [`TemplateFlavor::Qwen3`] | [`TemplateFlavor::Qwen38`] |
/// |---|---|---|
/// | tools preamble | "You may call one or more functions…", system content BEFORE the tools block, JSON `<tool_call>` example | "You have access to the following functions:", system content AFTER the `<IMPORTANT>` block, `<function=…><parameter=…>` example |
/// | history tool calls | `{"name": …, "arguments": {…}}` JSON payload | `<function=name>` + one `<parameter=key>` block per argument |
/// | assistant reasoning source | `reasoning_content` field, else split out of an embedded `</think>` in `content` | `reasoning_content` field only |
/// | history reasoning framing | only after the last real user query, and only the final turn or a reasoned one | `preserve_thinking` kwarg: kept on EVERY turn by default; `false` strips everything up to the last real user query |
/// | whitespace | content emitted verbatim | `content`/`reasoning_content` trimmed (the Jinja `\|trim`) |
/// | generation prompt (thinking on) | nothing after `<\|im_start\|>assistant` | prefills an open `<think>\n` |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TemplateFlavor {
    /// The Qwen3-generation template (dense Qwen3 checkpoints, e.g. 0.6B).
    #[default]
    Qwen3,
    /// The Qwen3.8-generation template (Qwen3.8 checkpoints, e.g. 27B).
    Qwen38,
}

/// Rendering knobs the Jinja template reads from the caller's generation
/// config (`add_generation_prompt`, `enable_thinking`, `reasoning_effort`).
#[derive(Clone, Debug)]
pub struct TemplateOpts {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    /// Qwen3.8 `reasoning_effort`: `xhigh` (default when thinking enabled),
    /// `medium`, or `low`.  Injected as a system-prompt instruction before the
    /// tools/system content.  `None` means "use the template default" (which
    /// is `xhigh` when `enable_thinking` is true, ignored when false). Under
    /// [`TemplateFlavor::Qwen38`] the template's own `\|default('xhigh')` applies
    /// at render time, so `None` and `Some("xhigh")` render identically.
    pub reasoning_effort: Option<String>,
    /// Which template generation to transcribe (default [`TemplateFlavor::Qwen3`],
    /// the behavior this port was originally validated against).
    pub flavor: TemplateFlavor,
    /// Qwen3.8's `preserve_thinking` chat-template kwarg. `None` is the
    /// template's `undefined` and keeps reasoning on EVERY assistant turn;
    /// `Some(false)` strips reasoning from every assistant turn up to and
    /// including the last real (non-tool-response) user query. Ignored under
    /// [`TemplateFlavor::Qwen3`].
    pub preserve_thinking: Option<bool>,
}

impl Default for TemplateOpts {
    fn default() -> TemplateOpts {
        TemplateOpts { add_generation_prompt: true, enable_thinking: true, reasoning_effort: None, flavor: TemplateFlavor::Qwen3, preserve_thinking: None }
    }
}

// ===================== json_py: Python json.dumps(ensure_ascii=False) =====================

/// A hand-rolled, order-preserving JSON re-serializer matching Python's
/// `json.dumps(x, ensure_ascii=False)` byte-for-byte (given `x = json.loads(raw)`
/// with no numeric renormalization — see [`dumps`]).
///
/// `serde_json::Value` cannot be used for this: the workspace does not build
/// `serde_json` with the `preserve_order` feature, so its `Map` is a `BTreeMap`
/// and round-tripping an object through it silently sorts keys alphabetically —
/// exactly the kind of divergence from the real Jinja `tojson` output this
/// module exists to avoid.
mod json_py {
    /// An order-preserving JSON tree. Numbers keep their original lexical text
    /// (so `1.0` stays `1.0`, not renormalized to `1`) rather than round-tripping
    /// through a float — the client's tool schema is echoed back, not recomputed.
    #[derive(Clone, Debug, PartialEq)]
    pub(super) enum Node {
        Obj(Vec<(String, Node)>),
        Arr(Vec<Node>),
        Str(String),
        Num(String),
        Bool(bool),
        Null,
    }

    /// Re-serialize raw JSON text the way Python's `json.dumps(x,
    /// ensure_ascii=False)` would print `x = json.loads(raw)`: `", "` between
    /// elements, `": "` after keys, object key order preserved, non-ASCII left
    /// unescaped.
    pub fn dumps(raw: &str) -> Result<String, String> {
        let node = parse(raw)?;
        let mut out = String::new();
        write_node(&node, &mut out);
        Ok(out)
    }

    pub(super) fn write_node(n: &Node, out: &mut String) {
        match n {
            Node::Null => out.push_str("null"),
            Node::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Node::Num(s) => out.push_str(s),
            Node::Str(s) => write_str(s, out),
            Node::Arr(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_node(it, out);
                }
                out.push(']');
            }
            Node::Obj(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(k, out);
                    out.push_str(": ");
                    write_node(v, out);
                }
                out.push('}');
            }
        }
    }

    /// Python's `json.dumps` (ensure_ascii=False) escape table: `\`, `"`, and the
    /// six named control escapes get their short form; every other control char
    /// (< 0x20) gets `\u00xx`; everything else — including all non-ASCII — is
    /// written through literally.
    fn write_str(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0C}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
    }

    // ---- a small recursive-descent JSON parser, order- and lexeme-preserving ----

    pub(super) fn parse(raw: &str) -> Result<Node, String> {
        let c: Vec<char> = raw.chars().collect();
        let mut i = 0usize;
        skip_ws(&c, &mut i);
        let v = value(&c, &mut i)?;
        skip_ws(&c, &mut i);
        if i != c.len() {
            return Err(format!("trailing data at char {i}"));
        }
        Ok(v)
    }

    fn skip_ws(c: &[char], i: &mut usize) {
        while matches!(c.get(*i), Some(' ' | '\t' | '\n' | '\r')) {
            *i += 1;
        }
    }

    fn value(c: &[char], i: &mut usize) -> Result<Node, String> {
        skip_ws(c, i);
        match c.get(*i) {
            Some('{') => object(c, i),
            Some('[') => array(c, i),
            Some('"') => Ok(Node::Str(string(c, i)?)),
            Some('t') | Some('f') => boolean(c, i),
            Some('n') => null(c, i),
            Some(ch) if *ch == '-' || ch.is_ascii_digit() => number(c, i),
            other => Err(format!("unexpected {other:?} at char {i}")),
        }
    }

    fn object(c: &[char], i: &mut usize) -> Result<Node, String> {
        *i += 1; // '{'
        let mut fields = Vec::new();
        skip_ws(c, i);
        if c.get(*i) == Some(&'}') {
            *i += 1;
            return Ok(Node::Obj(fields));
        }
        loop {
            skip_ws(c, i);
            let key = string(c, i)?;
            skip_ws(c, i);
            if c.get(*i) != Some(&':') {
                return Err(format!("expected ':' at char {i}"));
            }
            *i += 1;
            let v = value(c, i)?;
            fields.push((key, v));
            skip_ws(c, i);
            match c.get(*i) {
                Some(',') => *i += 1,
                Some('}') => {
                    *i += 1;
                    break;
                }
                other => return Err(format!("expected ',' or '}}' at char {i}, got {other:?}")),
            }
        }
        Ok(Node::Obj(fields))
    }

    fn array(c: &[char], i: &mut usize) -> Result<Node, String> {
        *i += 1; // '['
        let mut items = Vec::new();
        skip_ws(c, i);
        if c.get(*i) == Some(&']') {
            *i += 1;
            return Ok(Node::Arr(items));
        }
        loop {
            let v = value(c, i)?;
            items.push(v);
            skip_ws(c, i);
            match c.get(*i) {
                Some(',') => *i += 1,
                Some(']') => {
                    *i += 1;
                    break;
                }
                other => return Err(format!("expected ',' or ']' at char {i}, got {other:?}")),
            }
        }
        Ok(Node::Arr(items))
    }

    fn string(c: &[char], i: &mut usize) -> Result<String, String> {
        if c.get(*i) != Some(&'"') {
            return Err(format!("expected '\"' at char {i}"));
        }
        *i += 1;
        let mut s = String::new();
        loop {
            match c.get(*i) {
                None => return Err("unterminated string".into()),
                Some('"') => {
                    *i += 1;
                    return Ok(s);
                }
                Some('\\') => {
                    *i += 1;
                    match c.get(*i) {
                        Some('"') => s.push('"'),
                        Some('\\') => s.push('\\'),
                        Some('/') => s.push('/'),
                        Some('b') => s.push('\u{08}'),
                        Some('f') => s.push('\u{0C}'),
                        Some('n') => s.push('\n'),
                        Some('r') => s.push('\r'),
                        Some('t') => s.push('\t'),
                        Some('u') => {
                            let cp = hex4(c, i)?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                if c.get(*i + 1) != Some(&'\\') || c.get(*i + 2) != Some(&'u') {
                                    return Err("expected low surrogate".into());
                                }
                                *i += 2;
                                let lo = hex4(c, i)?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err("invalid low surrogate".into());
                                }
                                let cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                s.push(char::from_u32(cp).ok_or("invalid codepoint")?);
                            } else {
                                s.push(char::from_u32(cp).ok_or("invalid codepoint")?);
                            }
                        }
                        other => return Err(format!("bad escape {other:?}")),
                    }
                    *i += 1;
                }
                Some(&ch) => {
                    s.push(ch);
                    *i += 1;
                }
            }
        }
    }

    /// Reads the 4 hex digits of a `\uXXXX` escape (leaves `i` on the last
    /// digit, matching `string`'s trailing `*i += 1`).
    fn hex4(c: &[char], i: &mut usize) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            *i += 1;
            let d = c.get(*i).and_then(|c| c.to_digit(16)).ok_or("bad \\u escape")?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn number(c: &[char], i: &mut usize) -> Result<Node, String> {
        let start = *i;
        if c.get(*i) == Some(&'-') {
            *i += 1;
        }
        while matches!(c.get(*i), Some(d) if d.is_ascii_digit()) {
            *i += 1;
        }
        if c.get(*i) == Some(&'.') {
            *i += 1;
            while matches!(c.get(*i), Some(d) if d.is_ascii_digit()) {
                *i += 1;
            }
        }
        if matches!(c.get(*i), Some('e') | Some('E')) {
            *i += 1;
            if matches!(c.get(*i), Some('+') | Some('-')) {
                *i += 1;
            }
            while matches!(c.get(*i), Some(d) if d.is_ascii_digit()) {
                *i += 1;
            }
        }
        if *i == start || (c[start..*i].len() == 1 && c[start] == '-') {
            return Err(format!("invalid number at char {start}"));
        }
        Ok(Node::Num(c[start..*i].iter().collect()))
    }

    fn boolean(c: &[char], i: &mut usize) -> Result<Node, String> {
        if c[*i..].starts_with(&['t', 'r', 'u', 'e']) {
            *i += 4;
            Ok(Node::Bool(true))
        } else if c[*i..].starts_with(&['f', 'a', 'l', 's', 'e']) {
            *i += 5;
            Ok(Node::Bool(false))
        } else {
            Err(format!("invalid literal at char {i}"))
        }
    }

    fn null(c: &[char], i: &mut usize) -> Result<Node, String> {
        if c[*i..].starts_with(&['n', 'u', 'l', 'l']) {
            *i += 4;
            Ok(Node::Null)
        } else {
            Err(format!("invalid literal at char {i}"))
        }
    }
}

// ===================== render: messages -> prompt text =====================

/// Render the Qwen3 chat template, byte-for-byte matching the checkpoint's
/// Jinja (`tokenizer_config.json`'s `chat_template`). See the module doc for
/// how this differs from [`crate::qwen_tokenizer::QwenBpe::apply_chat_template`].
///
/// Implements, in order:
/// 1. **Tools preamble** (only if `tools` is non-empty): `<|im_start|>system\n`,
///    then `messages[0]`'s content + `"\n\n"` if it is a system message (which
///    is CONSUMED here — never re-emitted by the per-message loop below), the
///    fixed `# Tools` preamble, each tool's schema via [`json_py::dumps`], and
///    the fixed tail describing the `<tool_call>` reply format.
/// 2. **No-tools system preamble**: `<|im_start|>system\n{content}<|im_end|>\n`
///    if `messages[0]` is a system message — and here it is NOT consumed; the
///    per-message loop below still sees index 0, it just has nothing left to do
///    for it (a system message at `loop.first` matches none of the loop's role
///    branches).
/// 3. **`last_query_index`**: scanning in reverse, the index of the last `user`
///    message whose content is not a synthetic `<tool_response>…</tool_response>`
///    wrapper. Defaults to `messages.len() - 1`. Gates which assistant turn(s)
///    get a `<think>…</think>` prefix: only an assistant turn strictly after the
///    last real user turn (and only if it's the very last message, or it has
///    non-empty reasoning) is rendered as a completed reasoning turn.
/// 4. **Per-message body**: `user` / non-first `system` → plain ChatML turn;
///    `assistant` → resolved `reasoning_content` (explicit field, or split out
///    of an embedded `</think>` in `content`) + tool calls; consecutive `tool`
///    messages → one wrapping `<|im_start|>user` turn with one `<tool_response>`
///    block per message.
/// 5. **Generation prompt**: `<|im_start|>assistant\n`, plus an empty
///    `<think>\n\n</think>\n\n` block when thinking is explicitly disabled.
pub fn render(msgs: &[ChatMessage], tools: &[String], opts: TemplateOpts) -> Result<String, String> {
    let mut out = String::new();
    let has_tools = !tools.is_empty();
    let first_is_system = msgs.first().map(|m| m.role) == Some(Role::System);

    // Resolve reasoning_effort. Qwen3 flavor: None means "no injection"
    // (callers resolve defaults). Qwen38 flavor: the template's own
    // `|default('xhigh')` applies at render time, so None renders as xhigh.
    let reasoning_directive = if opts.enable_thinking {
        let effort = if opts.flavor == TemplateFlavor::Qwen38 {
            Some(opts.reasoning_effort.as_deref().unwrap_or("xhigh"))
        } else {
            opts.reasoning_effort.as_deref()
        };
        match effort {
            Some("low") => Some("Reasoning effort is set to low. Keep your thinking brief and focused, \
                 moving directly to the conclusion without unnecessary elaboration."),
            Some("medium") => None, // medium: no injected instruction
            None => None, // not specified: no injection (Qwen3 flavor; callers resolve defaults)
            Some("xhigh") => Some("Reasoning effort is set to xhigh. Please think carefully through the task, \
                 validate key assumptions, consider plausible alternatives, and prioritize \
                 correctness, consistency, and clarity in the final answer."),
            Some(other) => return Err(format!("Unexpected reasoning effort '{other}'. Supported types are xhigh (default), medium, and low.")),
        }
    } else {
        None // thinking disabled: reasoning_effort is ignored
    };

    if has_tools {
        if opts.flavor == TemplateFlavor::Qwen38 {
            out.push_str("<|im_start|>system\n");
            if let Some(dir) = reasoning_directive {
                out.push_str(dir);
                out.push_str("\n\n");
            }
            out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in tools {
                out.push('\n');
                out.push_str(&json_py::dumps(tool)?);
            }
            out.push_str("\n</tools>");
            out.push_str(
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
                 <tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n\
                 </parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n\
                 that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
                 - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
                 - Required parameters MUST be specified\n\
                 - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
                 - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
                 </IMPORTANT>",
            );
            // System content comes AFTER the <IMPORTANT> block in Qwen3.8
            // (the Qwen3-generation template puts it before "# Tools").
            if first_is_system {
                let content = msgs[0].content.trim();
                if !content.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(content);
                }
            }
            out.push_str("<|im_end|>\n");
        } else {
            out.push_str("<|im_start|>system\n");
            if let Some(dir) = reasoning_directive {
                out.push_str(dir);
                out.push_str("\n\n");
            }
            if first_is_system {
                out.push_str(&msgs[0].content);
                out.push_str("\n\n");
            }
            out.push_str(
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
                 You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
            );
            for tool in tools {
                out.push('\n');
                out.push_str(&json_py::dumps(tool)?);
            }
            out.push_str(
                "\n</tools>\n\nFor each function call, return a json object with function name and \
                 arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": \
                 <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
            );
        }
    } else if opts.flavor == TemplateFlavor::Qwen38 {
        // Qwen3.8 no-tools system turn: trimmed content, the directive
        // separated from content by exactly one blank line (no trailing one
        // when the content is absent), and the whole turn skipped when both
        // are empty.
        if first_is_system {
            let content = msgs[0].content.trim();
            if !content.is_empty() {
                out.push_str("<|im_start|>system\n");
                if let Some(dir) = reasoning_directive {
                    out.push_str(dir);
                    out.push_str("\n\n");
                }
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            } else if let Some(dir) = reasoning_directive {
                out.push_str("<|im_start|>system\n");
                out.push_str(dir);
                out.push_str("<|im_end|>\n");
            }
        } else if let Some(dir) = reasoning_directive {
            out.push_str("<|im_start|>system\n");
            out.push_str(dir);
            out.push_str("<|im_end|>\n");
        }
    } else if first_is_system || reasoning_directive.is_some() {
        out.push_str("<|im_start|>system\n");
        if let Some(dir) = reasoning_directive {
            out.push_str(dir);
            out.push_str("\n\n");
        }
        if first_is_system {
            out.push_str(&msgs[0].content);
        }
        out.push_str("<|im_end|>\n");
    }

    // last_query_index: reverse scan, first (from the end) real user turn.
    // Qwen3.8 trims the content before the synthetic-tool-response check
    // (the Jinja `render_content(...)|trim` at the top of the scan).
    let mut last_query_index = msgs.len().saturating_sub(1);
    let mut multi_step_tool = true;
    for (index, m) in msgs.iter().enumerate().rev() {
        if multi_step_tool && m.role == Role::User {
            let content: &str = if opts.flavor == TemplateFlavor::Qwen38 { m.content.trim() } else { &m.content };
            if !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
                multi_step_tool = false;
                last_query_index = index;
            }
        }
    }

    for (index, m) in msgs.iter().enumerate() {
        let is_first = index == 0;
        let content = &m.content;
        if m.role == Role::User || (m.role == Role::System && !is_first) {
            out.push_str("<|im_start|>");
            out.push_str(m.role.as_str());
            out.push('\n');
            if opts.flavor == TemplateFlavor::Qwen38 {
                out.push_str(content.trim());
            } else {
                out.push_str(content);
            }
            out.push_str("<|im_end|>\n");
        } else if m.role == Role::Assistant {
            let (reasoning, content) = resolve_reasoning(m, opts.flavor == TemplateFlavor::Qwen38);
            if opts.flavor == TemplateFlavor::Qwen38 {
                // Qwen3.8: field-only reasoning, everything trimmed, and the
                // think block gated by preserve_thinking (default: keep on
                // EVERY assistant turn) rather than by position heuristics.
                let reasoning = reasoning.trim();
                let content = content.trim();
                if opts.preserve_thinking.unwrap_or(true) || index > last_query_index {
                    out.push_str("<|im_start|>assistant\n<think>\n");
                    out.push_str(reasoning);
                    out.push_str("\n</think>\n\n");
                    out.push_str(content);
                } else {
                    out.push_str("<|im_start|>assistant\n");
                    out.push_str(content);
                }
                for (ci, call) in m.tool_calls.iter().enumerate() {
                    if ci == 0 {
                        if !content.is_empty() {
                            out.push_str("\n\n");
                        }
                    } else {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.name);
                    out.push_str(">\n");
                    append_xml_parameters(&call.arguments, &mut out);
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            } else {
                let is_last = index == msgs.len() - 1;
                if index > last_query_index && (is_last || !reasoning.is_empty()) {
                    out.push_str("<|im_start|>assistant\n<think>\n");
                    out.push_str(reasoning.trim_matches('\n'));
                    out.push_str("\n</think>\n\n");
                    out.push_str(content.trim_start_matches('\n'));
                } else {
                    out.push_str("<|im_start|>assistant\n");
                    out.push_str(&content);
                }
                for (ci, call) in m.tool_calls.iter().enumerate() {
                    if ci != 0 || !content.is_empty() {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n{\"name\": \"");
                    out.push_str(&call.name);
                    out.push_str("\", \"arguments\": ");
                    out.push_str(&call.arguments);
                    out.push_str("}\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
        } else if m.role == Role::Tool {
            let run_start = is_first || msgs[index - 1].role != Role::Tool;
            let run_end = index == msgs.len() - 1 || msgs[index + 1].role != Role::Tool;
            if run_start {
                out.push_str("<|im_start|>user");
            }
            out.push_str("\n<tool_response>\n");
            if opts.flavor == TemplateFlavor::Qwen38 {
                out.push_str(content.trim());
            } else {
                out.push_str(content);
            }
            out.push_str("\n</tool_response>");
            if run_end {
                out.push_str("<|im_end|>\n");
            }
        }
        // A system message at index 0 matches none of the branches above (by
        // design — see point 2 in the doc comment) and is silently skipped.
    }

    if opts.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if !opts.enable_thinking {
            out.push_str("<think>\n\n</think>\n\n");
        } else if opts.flavor == TemplateFlavor::Qwen38 {
            // Qwen3.8 prefills the open think tag: the model continues
            // INSIDE the think block (the scanner must start reasoning-open).
            out.push_str("<think>\n");
        }
    }

    Ok(out)
}

/// `render` with `add_generation_prompt=true` — the common inference-time call.
pub fn render_for_generation(msgs: &[ChatMessage], tools: &[String], opts: TemplateOpts) -> Result<String, String> {
    render(msgs, tools, TemplateOpts { add_generation_prompt: true, ..opts })
}

/// Resolve an assistant message's `(reasoning, content)` pair.
///
/// Qwen3 flavor: the explicit `reasoning_content` field if present, else
/// split out of an embedded `</think>` in `content` (Jinja:
/// `content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')`
/// for the reasoning, `content.split('</think>')[-1].lstrip('\n')` for the
/// remaining content), else empty reasoning with `content` unchanged.
///
/// Qwen3.8 flavor: the `reasoning_content` field only
/// (`message.reasoning_content is string`), with `content` unchanged - the
/// new template never splits `<think>` markers out of content.
fn resolve_reasoning(m: &ChatMessage, qwen38: bool) -> (String, String) {
    if let Some(r) = &m.reasoning_content {
        return (r.clone(), m.content.clone());
    }
    if qwen38 {
        return (String::new(), m.content.clone());
    }
    let Some(split_at) = m.content.find("</think>") else {
        return (String::new(), m.content.clone());
    };
    let before = m.content[..split_at].trim_end_matches('\n');
    let reasoning = match before.rfind("<think>") {
        Some(p) => &before[p + "<think>".len()..],
        None => before,
    };
    let reasoning = reasoning.trim_start_matches('\n').to_string();
    let content = m.content[split_at + "</think>".len()..].trim_start_matches('\n').to_string();
    (reasoning, content)
}

/// Qwen3.8 history tool-call arguments: one `<parameter=key>\n{value}\n</parameter>\n`
/// block per key of the arguments object, in the object's own key order.
/// String values verbatim (the Jinja `| string`); every other JSON value
/// re-serialized Python-json-style (the Jinja `| tojson`). A payload that is
/// not an object (or does not parse) contributes no parameters - upstream
/// always sends objects, and the template's `arguments != ''` guard gives the
/// empty case nothing to iterate.
fn append_xml_parameters(arguments: &str, out: &mut String) {
    let Ok(node) = json_py::parse(arguments) else { return };
    let json_py::Node::Obj(entries) = node else { return };
    for (key, value) in entries {
        out.push_str("<parameter=");
        out.push_str(&key);
        out.push_str(">\n");
        match value {
            json_py::Node::Str(s) => out.push_str(&s),
            other => json_py::write_node(&other, out),
        }
        out.push_str("\n</parameter>\n");
    }
}

// ===================== ChatScanner: generated text -> events =====================

/// One unit of streaming-decoded output: visible text, hidden reasoning, or a
/// piece of a tool call.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatEvent {
    Content(String),
    Reasoning(String),
    ToolCallStart { index: u32, name: String },
    ToolCallArgs { index: u32, fragment: String },
    ToolCallEnd { index: u32 },
}

/// Internal scan state. The call states buffer raw text in
/// [`ChatScanner::pending`] rather than carrying their own buffer, since the
/// withholding logic that guards against a marker splitting across two
/// [`ChatScanner::push`] calls already lives there. `CallHeader`/`CallArgs`
/// are the Qwen3 JSON payload form; `XmlBetween`/`XmlKey`/`XmlValue` walk one
/// Qwen3.8 XML call (`<function=name>` → per-argument `<parameter=key>`
/// blocks → `</function>`).
#[derive(Debug)]
enum St {
    Content,
    Think,
    CallHeader,
    CallArgs { in_string: bool, escape: bool, depth: i32, started: bool },
    /// Between XML parameter blocks: awaiting the next `<parameter=` or the
    /// closing `</function>`. Whitespace between blocks is discarded; stray
    /// text is surfaced as content (a ToolCallStart has already streamed, so
    /// the invalid-header policy of re-labeling the whole call is unavailable).
    XmlBetween,
    /// Inside a `<parameter=` header, awaiting the `>` that ends the key.
    XmlKey,
    /// Inside a parameter value, awaiting `</parameter>`. The value is
    /// buffered whole (its JSON form - verbatim when it parses as JSON,
    /// escaped as a string otherwise - can only be chosen once it is complete).
    XmlValue { key: String },
    CallTail,
}

/// A streaming scanner over a Qwen3/Qwen3.8 generation: splits raw model
/// output into visible content, `<think>…</think>` reasoning, and
/// `<tool_call>…</tool_call>` blocks, one `push()` at a time - safe against a
/// marker being split across two calls (as real token-by-token streaming
/// does), down to a single byte per call. The tool-call PAYLOAD form inside
/// the fence follows the [`TemplateFlavor`]: Qwen3's JSON
/// `{"name": …, "arguments": {…}}`, or Qwen3.8's `<function=name>` +
/// `<parameter=key>` XML (reconstructed into the same JSON-arguments shape,
/// so callers and the render direction see one representation).
pub struct ChatScanner {
    state: St,
    flavor: TemplateFlavor,
    /// Raw text not yet resolved into an event: either the tail of `Content`/
    /// `Think` text that could still be the start of a marker, or (in the
    /// call states) everything since the state was entered that hasn't yet
    /// been consumed.
    pending: String,
    content: String,
    reasoning: String,
    /// Set when a literal `<think>` tag was seen mid-stream: the render
    /// frames a think block as `<think>\n...\n</think>`, so exactly one
    /// newline after the tag (and one before the close) is framing, not
    /// reasoning. False when the scan starts inside an already-open think
    /// block (prefilled `<think>\n`) - there the framing newline belonged
    /// to the prompt, not to the generation.
    think_lead_nl: bool,
    tool_calls: Vec<ToolCallMsg>,
    next_index: u32,
    cur_name: String,
    cur_args: String,
}

const MARKERS_CONTENT: [&str; 2] = ["<think>", "<tool_call>"];
const MARKERS_THINK: [&str; 1] = ["</think>"];
const MARKERS_TAIL: [&str; 1] = ["</tool_call>"];
/// Markers that end the current position inside a Qwen3.8 XML call: the next
/// parameter block, or the end of the parameter list.
const MARKERS_XML_BETWEEN: [&str; 2] = ["<parameter=", "</function>"];

impl ChatScanner {
    /// `thinking_open=true` seeds an already-open [`St::Think`] state (for a
    /// prompt whose generation-prefix already emitted a bare `<think>`, e.g.
    /// `enable_thinking` with a prefill); `false` is the normal case where the
    /// model must emit the literal `<think>` tag itself if it wants to think.
    /// Defaults to the [`TemplateFlavor::Qwen3`] tool-call form.
    pub fn new(thinking_open: bool) -> ChatScanner {
        ChatScanner::with_flavor(thinking_open, TemplateFlavor::Qwen3)
    }

    /// Same, with an explicit template flavor selecting the tool-call payload
    /// form the scanner expects ([`TemplateFlavor::Qwen38`] = the XML form).
    pub fn with_flavor(thinking_open: bool, flavor: TemplateFlavor) -> ChatScanner {
        ChatScanner {
            state: if thinking_open { St::Think } else { St::Content },
            flavor,
            pending: String::new(),
            content: String::new(),
            reasoning: String::new(),
            think_lead_nl: false,
            tool_calls: Vec::new(),
            next_index: 0,
            cur_name: String::new(),
            cur_args: String::new(),
        }
    }

    /// Feed the next chunk of generated text (a token, several tokens, or the
    /// whole output at once — chunking must not change the resulting events).
    pub fn push(&mut self, delta: &str, out: &mut Vec<ChatEvent>) {
        self.pending.push_str(delta);
        self.drain(out);
    }

    /// End of generation (EOS or max-tokens cut): flush any withheld tail as
    /// content/reasoning, and if a tool call was left open (JSON truncated
    /// mid-arguments, or the closing `</tool_call>` never arrived), close it out
    /// so the caller gets a (possibly invalid-JSON) result rather than a
    /// silently dropped call.
    pub fn finish(&mut self, out: &mut Vec<ChatEvent>) {
        match self.state {
            St::Content => self.flush_pending_as(out, false),
            St::Think => self.flush_pending_as(out, true),
            St::CallHeader => {
                // The header never resolved (truncated before `"arguments":`) —
                // there is no name to attach a tool call to, so surface the raw
                // text (including the `<tool_call>` tag already consumed off
                // `pending`) as content rather than silently dropping it.
                let text = format!("<tool_call>{}", std::mem::take(&mut self.pending));
                self.content.push_str(&text);
                out.push(ChatEvent::Content(text));
            }
            St::CallArgs { .. } => {
                if !self.pending.is_empty() {
                    let frag = std::mem::take(&mut self.pending);
                    self.cur_args.push_str(&frag);
                    out.push(ChatEvent::ToolCallArgs { index: self.next_index, fragment: frag });
                }
                self.close_call(out);
            }
            St::CallTail => {
                self.pending.clear();
                self.close_call(out);
            }
            St::XmlBetween | St::XmlKey { .. } => {
                // The call header already streamed as a ToolCallStart, so the
                // whole-call re-labeling the JSON header does here is
                // unavailable: discard whatever partial marker text is left
                // (never real prose - real text was surfaced as content as it
                // arrived), close the arguments object, and report the call.
                self.pending.clear();
                self.xml_close_object(out);
                self.close_call(out);
            }
            St::XmlValue { .. } => {
                // Cut mid-value: the buffered text is the value.
                let raw = std::mem::take(&mut self.pending);
                let value = raw.strip_prefix('\n').unwrap_or(&raw);
                let value = value.strip_suffix('\n').unwrap_or(value);
                let St::XmlValue { key } = &self.state else { unreachable!("XmlValue arm") };
                let key = key.clone();
                let fragment = xml_push_parameter_state(&mut self.cur_args, &key, value);
                out.push(ChatEvent::ToolCallArgs { index: self.next_index, fragment });
                self.xml_close_object(out);
                self.close_call(out);
            }
        }
        self.state = St::Content;
    }

    pub fn content(&self) -> &str {
        &self.content
    }
    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }
    pub fn tool_calls(&self) -> &[ToolCallMsg] {
        &self.tool_calls
    }

    fn flush_pending_as(&mut self, out: &mut Vec<ChatEvent>, reasoning: bool) {
        if self.pending.is_empty() {
            return;
        }
        let mut text = std::mem::take(&mut self.pending);
        if reasoning {
            // Unclosed think block at finish: the open-side framing strip
            // still applies; there is no `</think>` to strip a close-side
            // newline against.
            self.strip_think_framing(&mut text, false);
        }
        if text.is_empty() {
            return;
        }
        if reasoning {
            self.reasoning.push_str(&text);
            out.push(ChatEvent::Reasoning(text));
        } else {
            self.content.push_str(&text);
            out.push(ChatEvent::Content(text));
        }
    }

    /// The render frames a think block as `<think>\n{reasoning}\n</think>`,
    /// so the scan direction drops exactly one newline right after a seen
    /// `<think>` tag and one right before `</think>` - the inverse of the
    /// render's own trimming, which is what makes a rendered assistant turn
    /// scan back to the reasoning it was rendered from. `trailing` selects
    /// the close-side strip (only at the `</think>` transition, where the
    /// Think-state withholding guarantees the framing newline is still
    /// buffered); the open-side strip applies while [`Self::think_lead_nl`]
    /// is set, and only if the text actually begins with that newline.
    fn strip_think_framing(&mut self, text: &mut String, trailing: bool) {
        if trailing && text.ends_with('\n') {
            text.pop();
        }
        if self.think_lead_nl {
            if let Some(rest) = text.strip_prefix('\n') {
                let rest = rest.to_string();
                text.clear();
                text.push_str(&rest);
                self.think_lead_nl = false;
            } else if !text.is_empty() {
                self.think_lead_nl = false;
            }
        }
    }

    fn close_call(&mut self, out: &mut Vec<ChatEvent>) {
        out.push(ChatEvent::ToolCallEnd { index: self.next_index });
        self.tool_calls.push(ToolCallMsg {
            id: format!("call_{}", self.next_index),
            name: std::mem::take(&mut self.cur_name),
            arguments: std::mem::take(&mut self.cur_args),
        });
        self.next_index += 1;
    }

    /// End of a Qwen3.8 XML parameter list: close the JSON arguments object
    /// (`{}` when no parameter arrived) and stream the closing fragment, so
    /// the Args fragments still concatenate to exactly
    /// [`ToolCallMsg::arguments`]. The call itself is closed by the caller
    /// (via `</tool_call>`'s [`St::CallTail`], or [`ChatScanner::finish`]
    /// when the generation was cut).
    fn xml_close_object(&mut self, out: &mut Vec<ChatEvent>) {
        let tail = if self.cur_args.is_empty() { "{}" } else { "}" }.to_string();
        self.cur_args.push_str(&tail);
        out.push(ChatEvent::ToolCallArgs { index: self.next_index, fragment: tail });
    }

    /// Consume as much of `pending` as the current state can resolve
    /// unambiguously, emitting events as it goes, until either `pending` is
    /// exhausted or the remainder needs more input to resolve.
    fn drain(&mut self, out: &mut Vec<ChatEvent>) {
        loop {
            match &self.state {
                St::Content => {
                    if !self.scan_for_marker(out, &MARKERS_CONTENT, false) {
                        return;
                    }
                }
                St::Think => {
                    if !self.scan_for_marker(out, &MARKERS_THINK, true) {
                        return;
                    }
                }
                St::CallHeader => {
                    let chars: Vec<char> = self.pending.chars().collect();
                    let outcome = match self.flavor {
                        TemplateFlavor::Qwen3 => parse_header(&chars),
                        TemplateFlavor::Qwen38 => parse_xml_header(&chars),
                    };
                    match outcome {
                        HeaderOutcome::Incomplete => return,
                        HeaderOutcome::Invalid => {
                            let text = format!("<tool_call>{}", std::mem::take(&mut self.pending));
                            self.content.push_str(&text);
                            out.push(ChatEvent::Content(text));
                            self.state = St::Content;
                        }
                        HeaderOutcome::Done { name, consumed_chars } => {
                            let byte_off =
                                self.pending.char_indices().nth(consumed_chars).map(|(i, _)| i).unwrap_or(self.pending.len());
                            self.pending.drain(..byte_off);
                            self.cur_name = name.clone();
                            self.cur_args.clear();
                            out.push(ChatEvent::ToolCallStart { index: self.next_index, name });
                            self.state = match self.flavor {
                                TemplateFlavor::Qwen3 => St::CallArgs { in_string: false, escape: false, depth: 0, started: false },
                                TemplateFlavor::Qwen38 => St::XmlBetween,
                            };
                        }
                    }
                }
                St::XmlBetween => {
                    // Either the next parameter block or the end of the
                    // parameter list; anything before it is block separator
                    // whitespace (discarded) or stray model text (surfaced as
                    // content - see St's doc).
                    let found = MARKERS_XML_BETWEEN.iter().filter_map(|m| self.pending.find(m).map(|p| (p, *m))).min_by_key(|(p, _)| *p);
                    match found {
                        None => {
                            let keep = withheld_suffix_len(&self.pending, &MARKERS_XML_BETWEEN);
                            let rest: String = self.pending.drain(..self.pending.len() - keep).collect();
                            if rest.trim_start_matches([' ', '\t', '\n', '\r']).is_empty() {
                                // block separator whitespace - nothing
                            } else {
                                self.content.push_str(&rest);
                                out.push(ChatEvent::Content(rest));
                            }
                            return;
                        }
                        Some((pos, marker)) => {
                            let pre: String = self.pending.drain(..pos).collect();
                            if !pre.trim_start_matches([' ', '\t', '\n', '\r']).is_empty() {
                                self.content.push_str(&pre);
                                out.push(ChatEvent::Content(pre));
                            }
                            self.pending.drain(..marker.len());
                            if marker == "</function>" {
                                self.xml_close_object(out);
                                self.state = St::CallTail;
                            } else {
                                self.state = St::XmlKey;
                            }
                        }
                    }
                }
                St::XmlKey => match self.pending.find('>') {
                    None => return,
                    Some(gt) => {
                        let key: String = self.pending.drain(..gt).collect();
                        self.pending.drain(..1);
                        self.state = St::XmlValue { key };
                    }
                },
                St::XmlValue { key } => match self.pending.find("</parameter>") {
                    // The value buffers whole in `pending` - its JSON form
                    // (verbatim vs escaped) is only decidable at the block
                    // end, so nothing may be emitted or dropped early.
                    None => return,
                    Some(pos) => {
                        let raw: String = self.pending.drain(..pos).collect();
                        self.pending.drain(.."</parameter>".len());
                        let value = raw.strip_prefix('\n').unwrap_or(&raw);
                        let value = value.strip_suffix('\n').unwrap_or(value);
                        let fragment = xml_push_parameter_state(&mut self.cur_args, key, value);
                        out.push(ChatEvent::ToolCallArgs { index: self.next_index, fragment });
                        self.state = St::XmlBetween;
                    }
                },
                St::CallArgs { .. } => {
                    // The template's fixed `"arguments": ` separator can leave
                    // whitespace before the JSON value that has nothing to do
                    // with the value's own text; drop it here, before any
                    // depth/string scanning, so where a chunk boundary happens
                    // to fall can never change whether that whitespace ends up
                    // in the emitted argument fragment.
                    let started = matches!(&self.state, St::CallArgs { started: true, .. });
                    if !started {
                        let trim = self.pending.len() - self.pending.trim_start().len();
                        if trim > 0 {
                            self.pending.drain(..trim);
                        }
                        if self.pending.is_empty() {
                            return;
                        }
                        if let St::CallArgs { started, .. } = &mut self.state {
                            *started = true;
                        }
                    }
                    let (consumed, done) = self.scan_args();
                    if consumed > 0 {
                        let frag: String = self.pending.drain(..consumed).collect();
                        self.cur_args.push_str(&frag);
                        out.push(ChatEvent::ToolCallArgs { index: self.next_index, fragment: frag });
                    }
                    if done {
                        self.state = St::CallTail;
                    } else {
                        return;
                    }
                }
                St::CallTail => {
                    if !self.scan_tail(out) {
                        return;
                    }
                }
            }
        }
    }

    /// `Content`/`Think` scan step: find the earliest of `markers` in
    /// `pending`; emit everything before it as a content/reasoning event and
    /// transition state, or — if no marker is present — emit the safe prefix
    /// (everything except a suffix that could still be the start of a marker)
    /// and return `false` to signal "need more input".
    fn scan_for_marker(&mut self, out: &mut Vec<ChatEvent>, markers: &[&str], reasoning: bool) -> bool {
        let found = markers.iter().filter_map(|m| self.pending.find(m).map(|p| (p, *m))).min_by_key(|(p, _)| *p);
        match found {
            Some((pos, marker)) => {
                if pos > 0 {
                    let mut text: String = self.pending.drain(..pos).collect();
                    if reasoning {
                        // The marker is `</think>` (the only Think-state
                        // marker): the text right before it may end with the
                        // render's framing newline.
                        self.strip_think_framing(&mut text, true);
                        if !text.is_empty() {
                            self.reasoning.push_str(&text);
                            out.push(ChatEvent::Reasoning(text));
                        }
                    } else {
                        self.content.push_str(&text);
                        out.push(ChatEvent::Content(text));
                    }
                }
                if marker == "<think>" {
                    self.think_lead_nl = true;
                }
                self.pending.drain(..marker.len());
                self.state = if marker == "<think>" {
                    St::Think
                } else if marker == "</think>" {
                    St::Content
                } else {
                    debug_assert_eq!(marker, "<tool_call>");
                    St::CallHeader
                };
                true
            }
            None => {
                let keep = if reasoning {
                    withheld_think_suffix_len(&self.pending, markers)
                } else {
                    withheld_suffix_len(&self.pending, markers)
                };
                let emit_len = self.pending.len() - keep;
                if emit_len > 0 {
                    let mut text: String = self.pending.drain(..emit_len).collect();
                    if reasoning {
                        self.strip_think_framing(&mut text, false);
                        if !text.is_empty() {
                            self.reasoning.push_str(&text);
                            out.push(ChatEvent::Reasoning(text));
                        }
                    } else {
                        self.content.push_str(&text);
                        out.push(ChatEvent::Content(text));
                    }
                }
                false
            }
        }
    }

    /// `CallArgs` scan step: walk `pending` tracking brace/bracket depth and
    /// string/escape state (so a quoted `{`/`}`/`"` never perturbs the count),
    /// until the depth returns to 0 right after it first went positive — that's
    /// the end of the JSON value. Returns `(bytes consumed from pending, done)`.
    fn scan_args(&mut self) -> (usize, bool) {
        let St::CallArgs { in_string, escape, depth, .. } = &mut self.state else {
            unreachable!("scan_args called outside CallArgs");
        };
        let mut consumed = 0usize;
        for ch in self.pending.chars() {
            consumed += ch.len_utf8();
            if *in_string {
                if *escape {
                    *escape = false;
                } else if ch == '\\' {
                    *escape = true;
                } else if ch == '"' {
                    *in_string = false;
                }
                continue;
            }
            match ch {
                '"' => *in_string = true,
                '{' | '[' => *depth += 1,
                '}' | ']' => {
                    *depth -= 1;
                    if *depth == 0 {
                        return (consumed, true);
                    }
                }
                _ => {}
            }
        }
        (consumed, false)
    }

    /// `CallTail` scan step: consume up to and including `</tool_call>`,
    /// discarding whatever precedes it (Qwen emits a bare `\n` there) and
    /// closing out the call. Returns `false` ("need more input") if the marker
    /// isn't present yet.
    fn scan_tail(&mut self, out: &mut Vec<ChatEvent>) -> bool {
        match self.pending.find("</tool_call>") {
            Some(pos) => {
                self.pending.drain(..pos + "</tool_call>".len());
                self.close_call(out);
                self.state = St::Content;
                true
            }
            None => {
                let keep = withheld_suffix_len(&self.pending, &MARKERS_TAIL);
                let drop_len = self.pending.len() - keep;
                if drop_len > 0 {
                    self.pending.drain(..drop_len);
                }
                false
            }
        }
    }
}

/// The length (in bytes) of the longest suffix of `s` that is a proper prefix
/// of one of `markers` — i.e. text that must be withheld because the next
/// `push()` could complete a marker. Only called once `s` is known to contain
/// no marker as a substring, so any match here is necessarily shorter than the
/// marker itself.
fn withheld_suffix_len(s: &str, markers: &[&str]) -> usize {
    let max_len = markers.iter().map(|m| m.len().saturating_sub(1)).max().unwrap_or(0);
    for l in (1..=max_len.min(s.len())).rev() {
        let start = s.len() - l;
        if !s.is_char_boundary(start) {
            continue;
        }
        let suf = &s[start..];
        if markers.iter().any(|m| m.starts_with(suf)) {
            return l;
        }
    }
    0
}

/// Think-state withholding: in addition to a suffix that could still start a
/// marker, everything from the last newline on must stay buffered - a trailing
/// newline may be the render's framing `\n` before `</think>`, which the
/// close-time strip removes and which therefore can never be emitted as
/// reasoning text. Withholding from the newline also keeps any later
/// marker-prefix suffix intact.
fn withheld_think_suffix_len(s: &str, markers: &[&str]) -> usize {
    let keep = withheld_suffix_len(s, markers);
    match s.rfind('\n') {
        Some(i) => keep.max(s.len() - i),
        None => keep,
    }
}

enum HeaderOutcome {
    Incomplete,
    Invalid,
    Done { name: String, consumed_chars: usize },
}

/// Parse the fixed-shape `<tool_call>` header `{"name": "X", "arguments": `
/// (whitespace-tolerant around punctuation) from the start of `buf`. Returns
/// [`HeaderOutcome::Done`] with the parsed name and how many *chars* of `buf`
/// made up the header (everything through the colon + any following whitespace
/// — the arguments JSON value starts right after), [`HeaderOutcome::Incomplete`]
/// if `buf` is a valid-so-far prefix of the header (need more input), or
/// [`HeaderOutcome::Invalid`] if `buf` structurally cannot be this header (e.g.
/// the first key isn't `"name"`).
fn parse_header(buf: &[char]) -> HeaderOutcome {
    let mut i = 0usize;

    // Err(true) = ran out of input (incomplete); Err(false) = mismatch (invalid).
    fn skip_ws(c: &[char], i: &mut usize) {
        while matches!(c.get(*i), Some(' ' | '\t' | '\n' | '\r')) {
            *i += 1;
        }
    }
    fn expect(c: &[char], i: &mut usize, ch: char) -> Result<(), bool> {
        match c.get(*i) {
            None => Err(true),
            Some(&x) if x == ch => {
                *i += 1;
                Ok(())
            }
            Some(_) => Err(false),
        }
    }
    fn string(c: &[char], i: &mut usize) -> Result<String, bool> {
        match c.get(*i) {
            None => return Err(true),
            Some('"') => *i += 1,
            Some(_) => return Err(false),
        }
        let mut s = String::new();
        loop {
            match c.get(*i) {
                None => return Err(true),
                Some('"') => {
                    *i += 1;
                    return Ok(s);
                }
                Some('\\') => {
                    *i += 1;
                    match c.get(*i) {
                        None => return Err(true),
                        Some('"') => s.push('"'),
                        Some('\\') => s.push('\\'),
                        Some('/') => s.push('/'),
                        Some('b') => s.push('\u{08}'),
                        Some('f') => s.push('\u{0C}'),
                        Some('n') => s.push('\n'),
                        Some('r') => s.push('\r'),
                        Some('t') => s.push('\t'),
                        Some('u') => {
                            if c.len() <= *i + 4 {
                                return Err(true);
                            }
                            let hex: String = c[*i + 1..*i + 5].iter().collect();
                            let cp = u32::from_str_radix(&hex, 16).map_err(|_| false)?;
                            s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            *i += 4;
                        }
                        Some(_) => return Err(false),
                    }
                    *i += 1;
                }
                Some(&c2) => {
                    s.push(c2);
                    *i += 1;
                }
            }
        }
    }

    macro_rules! tri {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(true) => return HeaderOutcome::Incomplete,
                Err(false) => return HeaderOutcome::Invalid,
            }
        };
    }

    skip_ws(buf, &mut i);
    tri!(expect(buf, &mut i, '{'));
    skip_ws(buf, &mut i);
    let key1 = tri!(string(buf, &mut i));
    if key1 != "name" {
        return HeaderOutcome::Invalid;
    }
    skip_ws(buf, &mut i);
    tri!(expect(buf, &mut i, ':'));
    skip_ws(buf, &mut i);
    let name = tri!(string(buf, &mut i));
    skip_ws(buf, &mut i);
    tri!(expect(buf, &mut i, ','));
    skip_ws(buf, &mut i);
    let key2 = tri!(string(buf, &mut i));
    if key2 != "arguments" {
        return HeaderOutcome::Invalid;
    }
    skip_ws(buf, &mut i);
    tri!(expect(buf, &mut i, ':'));
    skip_ws(buf, &mut i);
    HeaderOutcome::Done { name, consumed_chars: i }
}

/// Parse the Qwen3.8 XML call header `<function=NAME>` (whitespace-tolerant
/// before the tag) from the start of `buf`, mirroring [`parse_header`]'s
/// outcome contract: [`HeaderOutcome::Done`] with the parsed name and how
/// many *chars* of `buf` made up the header, [`HeaderOutcome::Incomplete`]
/// while `buf` could still extend into a valid header, [`HeaderOutcome::Invalid`]
/// when it structurally cannot (e.g. a Qwen3 JSON payload under the 3.8
/// flavor).
fn parse_xml_header(buf: &[char]) -> HeaderOutcome {
    let mut i = 0usize;
    while matches!(buf.get(i), Some(' ' | '\t' | '\n' | '\r')) {
        i += 1;
    }
    for ch in "<function=".chars() {
        match buf.get(i) {
            None => return HeaderOutcome::Incomplete,
            Some(&c) if c == ch => i += 1,
            Some(_) => return HeaderOutcome::Invalid,
        }
    }
    let name_start = i;
    while matches!(buf.get(i), Some(c) if *c != '>') {
        i += 1;
    }
    match buf.get(i) {
        None => HeaderOutcome::Incomplete,
        Some('>') => HeaderOutcome::Done { name: buf[name_start..i].iter().collect(), consumed_chars: i + 1 },
        Some(_) => unreachable!("the scan above stops only at '>'"),
    }
}

/// One completed `<parameter=key>` value: append its JSON piece to the
/// arguments object being built and stream it as a fragment (the first
/// parameter opens the object, the object close comes from
/// [`ChatScanner`]'s XML end-of-call handling). A value that parses as JSON
/// is taken verbatim - the exact inverse of the render direction's
/// string-verbatim/others-tojson split (`append_xml_parameters`); anything
/// else is escaped as a JSON string.
fn xml_push_parameter_state(cur_args: &mut String, key: &str, value: &str) -> String {
    let kjson = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".into());
    let vjson = if serde_json::from_str::<serde_json::Value>(value).is_ok() {
        value.to_string()
    } else {
        serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into())
    };
    let kv = format!("{kjson}: {vjson}");
    let fragment = if cur_args.is_empty() { format!("{{{}", kv) } else { format!(", {}", kv) };
    cur_args.push_str(&fragment);
    fragment
}

/// One-shot convenience: run a whole generated string through a fresh
/// [`ChatScanner`] and return the accumulated `(content, reasoning,
/// tool_calls)`.
pub fn split_output(text: &str, thinking_open: bool) -> (String, String, Vec<ToolCallMsg>) {
    let mut scanner = ChatScanner::new(thinking_open);
    let mut events = Vec::new();
    scanner.push(text, &mut events);
    scanner.finish(&mut events);
    (scanner.content().to_string(), scanner.reasoning().to_string(), scanner.tool_calls().to_vec())
}

/// Parse the first `<tool_call>` block out of generated `text` via
/// [`split_output`] — the shared implementation behind
/// [`crate::toolcall::parse_tool_call`]. Kept crate-private since the public
/// return type ([`ToolCall`], with a parsed `serde_json::Value`) belongs to the
/// `toolcall` module's domain, not this one's.
pub(crate) fn first_tool_call(text: &str) -> Option<ToolCall> {
    let (_, _, calls) = split_output(text, false);
    let call = calls.into_iter().next()?;
    let arguments = serde_json::from_str(&call.arguments).ok()?;
    Some(ToolCall { name: call.name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- json_py::dumps ----------------------------------------------------

    #[test]
    fn json_py_dumps_matches_python_json_dumps() {
        // Nested object, unicode, arrays, and numbers in various lexical forms.
        // Expected string computed by hand per Python's json.dumps(x,
        // ensure_ascii=False) rules: ", " item sep, ": " key sep, key order
        // preserved, non-ASCII printed literally, numbers kept as-given.
        let raw = r#"{"b": 1, "a": [1, 2.50, -3, 1.0e10], "city": "Örebro", "nested": {"x": true, "y": null}}"#;
        let expected = r#"{"b": 1, "a": [1, 2.50, -3, 1.0e10], "city": "Örebro", "nested": {"x": true, "y": null}}"#;
        assert_eq!(json_py::dumps(raw).unwrap(), expected);
    }

    #[test]
    fn json_py_dumps_escapes_and_key_order() {
        let raw = r#"{"z": "line1\nline2\ttab\"quote", "a": 1}"#;
        let expected = r#"{"z": "line1\nline2\ttab\"quote", "a": 1}"#;
        assert_eq!(json_py::dumps(raw).unwrap(), expected);
        // key order preserved even though 'z' < 'a' alphabetically would sort differently
        let raw2 = r#"{"zeta": 1, "alpha": 2}"#;
        assert_eq!(json_py::dumps(raw2).unwrap(), r#"{"zeta": 1, "alpha": 2}"#);
    }

    #[test]
    fn json_py_dumps_empty_containers() {
        assert_eq!(json_py::dumps("{}").unwrap(), "{}");
        assert_eq!(json_py::dumps("[]").unwrap(), "[]");
        assert_eq!(json_py::dumps(r#"{"a": [], "b": {}}"#).unwrap(), r#"{"a": [], "b": {}}"#);
    }

    // ---- render: no tools ----------------------------------------------------

    #[test]
    fn render_user_only() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        assert_eq!(s, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn render_system_user() {
        let msgs = vec![ChatMessage::system("be nice"), ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        assert_eq!(
            s,
            "<|im_start|>system\nbe nice<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn render_system_user_assistant_user() {
        let msgs =
            vec![ChatMessage::system("sys"), ChatMessage::user("q1"), ChatMessage::assistant("a1"), ChatMessage::user("q2")];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        assert_eq!(
            s,
            "<|im_start|>system\nsys<|im_end|>\n\
             <|im_start|>user\nq1<|im_end|>\n\
             <|im_start|>assistant\na1<|im_end|>\n\
             <|im_start|>user\nq2<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    // ---- render: with tools ----------------------------------------------------

    // Already in canonical `json_py::dumps` spacing (`", "` / `": "`), so the
    // *whole* expected string below can be one hardcoded literal — these tests
    // check render()'s own preamble/separator/tail bytes independently of
    // json_py::dumps (covered on its own in the json_py tests above), rather
    // than re-deriving the expectation by calling the function under test.
    fn two_tools() -> Vec<String> {
        vec![
            r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}"#.to_string(),
            r#"{"type": "function", "function": {"name": "set_timer", "description": "Set a timer", "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}}, "required": ["minutes"]}}}"#.to_string(),
        ]
    }

    #[test]
    fn render_with_tools_no_system() {
        let msgs = vec![ChatMessage::user("weather in paris?")];
        let tools = two_tools();
        let s = render(&msgs, &tools, TemplateOpts::default()).unwrap();
        let expected = format!(
            "<|im_start|>system\n\
             # Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\
             {}\n{}\n\
             </tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": \
             <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\nweather in paris?<|im_end|>\n\
             <|im_start|>assistant\n",
            tools[0], tools[1]
        );
        assert_eq!(s, expected);
    }

    #[test]
    fn render_with_tools_and_system() {
        let msgs = vec![ChatMessage::system("You are helpful."), ChatMessage::user("weather in paris?")];
        let tools = two_tools();
        let s = render(&msgs, &tools, TemplateOpts::default()).unwrap();
        let expected = format!(
            "<|im_start|>system\nYou are helpful.\n\n\
             # Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\
             {}\n{}\n\
             </tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": \
             <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\nweather in paris?<|im_end|>\n\
             <|im_start|>assistant\n",
            tools[0], tools[1]
        );
        assert_eq!(s, expected);
    }

    // ---- render: tool_calls + tool responses ----------------------------------------------------

    #[test]
    fn render_two_tool_calls_then_two_tool_responses() {
        let calls = vec![
            ToolCallMsg { id: "call_0".into(), name: "get_weather".into(), arguments: r#"{"location": "Paris"}"#.into() },
            ToolCallMsg { id: "call_1".into(), name: "set_timer".into(), arguments: r#"{"minutes": 5}"#.into() },
        ];
        let msgs = vec![
            ChatMessage::user("weather and a timer please"),
            ChatMessage::assistant("").with_tool_calls(calls),
            ChatMessage::tool("22C sunny"),
            ChatMessage::tool("timer set"),
        ];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        // exactly one <|im_start|>user wrapping both tool_response blocks
        assert_eq!(s.matches("<|im_start|>user").count(), 2); // the real user turn + the tool-response wrapper
        let expected_tail = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n\
             <tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"minutes\": 5}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\n22C sunny\n</tool_response>\n<tool_response>\ntimer set\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n";
        assert!(s.ends_with(expected_tail), "got: {s:?}");
    }

    // ---- render: enable_thinking=false ----------------------------------------------------

    #[test]
    fn render_disable_thinking_appends_empty_think_block() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: true, enable_thinking: false , ..Default::default() }).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "got: {s:?}");
    }

    // ---- render: reasoning_content splitting ----------------------------------------------------

    #[test]
    fn render_reasoning_explicit_field() {
        let msgs = vec![
            ChatMessage::user("q"),
            ChatMessage {
                reasoning_content: Some("thinking hard".into()),
                ..ChatMessage::assistant("the answer")
            },
        ];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        assert!(s.contains("<think>\nthinking hard\n</think>\n\nthe answer"), "got: {s:?}");
    }

    #[test]
    fn render_reasoning_embedded_in_content() {
        let msgs = vec![ChatMessage::user("q"), ChatMessage::assistant("<think>\nthinking hard\n</think>\n\nthe answer")];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        assert!(s.contains("<think>\nthinking hard\n</think>\n\nthe answer"), "got: {s:?}");
    }

    // ---- render: last_query_index gating ----------------------------------------------------

    #[test]
    fn render_last_query_index_gates_think_prefix() {
        // assistant BEFORE the last real user turn: no think prefix (plain form).
        let msgs = vec![
            ChatMessage::user("q1"),
            ChatMessage::assistant("<think>\nR1\n</think>\n\nA1"),
            ChatMessage::user("q2"),
        ];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        // the first assistant turn (index 1) is NOT the last message and index(1) <= last_query_index(2)
        // so it must render as the plain (non-think) form, with content unmodified (reasoning splitting
        // still resolves `content`, but the <think> prefix itself is not re-added).
        assert!(s.contains("<|im_start|>assistant\nA1<|im_end|>\n"), "got: {s:?}");
        assert!(!s.contains("R1"), "reasoning before the last query must be dropped, got: {s:?}");
    }

    #[test]
    fn render_last_query_index_multi_step_tool_use() {
        // user -> assistant(tool_call) -> tool -> assistant(final answer).
        // last_query_index is the FIRST user message (index 0), since the tool
        // messages are synthetic <tool_response> wrappers only when authored as
        // role:Tool turned into <|im_start|>user by the template — but here the
        // "last real user turn" is not the last message at all: it's index 0.
        let calls = vec![ToolCallMsg { id: "call_0".into(), name: "f".into(), arguments: "{}".into() }];
        let msgs = vec![
            ChatMessage::user("q1"),
            ChatMessage::assistant("").with_tool_calls(calls),
            ChatMessage::tool("result"),
            ChatMessage::assistant("<think>\nfinal reasoning\n</think>\n\nfinal answer"),
        ];
        let s = render(&msgs, &[], TemplateOpts::default()).unwrap();
        // the final assistant turn (index 3, the last message) is after last_query_index(0) -> think prefix kept
        assert!(s.contains("<think>\nfinal reasoning\n</think>\n\nfinal answer"), "got: {s:?}");
    }

    // ---- scanner ----------------------------------------------------

    #[test]
    fn scanner_plain_text_passthrough() {
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        scanner.push("hello, world! no markers here.", &mut events);
        scanner.finish(&mut events);
        assert_eq!(events, vec![ChatEvent::Content("hello, world! no markers here.".to_string())]);
        assert_eq!(scanner.content(), "hello, world! no markers here.");
    }

    #[test]
    fn scanner_splits_think_block() {
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        scanner.push("before<think>reasoning here</think>after", &mut events);
        scanner.finish(&mut events);
        assert_eq!(
            events,
            vec![
                ChatEvent::Content("before".to_string()),
                ChatEvent::Reasoning("reasoning here".to_string()),
                ChatEvent::Content("after".to_string()),
            ]
        );
        assert_eq!(scanner.content(), "beforeafter");
        assert_eq!(scanner.reasoning(), "reasoning here");
    }

    // ---- scanner chunking invariance (the most important test) ----------------------------------------------------

    fn transcript() -> &'static str {
        "<think>\nplanning the call\n</think>\n\nSure, let me check.\
         <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\
         <tool_call>\n{\"name\": \"weird\", \"arguments\": {\"path\":\"a\\\"}b\"}}\n</tool_call>\
         Done."
    }

    /// Adjacent Content/Reasoning/(same-index)ToolCallArgs fragments
    /// concatenated, so runs split differently by chunking compare equal. Only
    /// the *text* boundaries are chunking-dependent (by design — a fragment is
    /// "however much of the value the current push resolved"); the marker
    /// transitions themselves (Start/End, and Content vs Reasoning vs Args) must
    /// not be, which is exactly what this test is checking.
    fn coalesce(events: Vec<ChatEvent>) -> Vec<ChatEvent> {
        let mut out: Vec<ChatEvent> = Vec::new();
        for e in events {
            match (&e, out.last_mut()) {
                (ChatEvent::Content(s), Some(ChatEvent::Content(prev))) => prev.push_str(s),
                (ChatEvent::Reasoning(s), Some(ChatEvent::Reasoning(prev))) => prev.push_str(s),
                (
                    ChatEvent::ToolCallArgs { index, fragment },
                    Some(ChatEvent::ToolCallArgs { index: prev_index, fragment: prev_fragment }),
                ) if index == prev_index => prev_fragment.push_str(fragment),
                _ => out.push(e),
            }
        }
        out
    }

    fn run_whole(text: &str) -> Vec<ChatEvent> {
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        scanner.push(text, &mut events);
        scanner.finish(&mut events);
        coalesce(events)
    }

    fn run_split_at(text: &str, at: usize) -> Vec<ChatEvent> {
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        scanner.push(&text[..at], &mut events);
        scanner.push(&text[at..], &mut events);
        scanner.finish(&mut events);
        coalesce(events)
    }

    fn run_byte_at_a_time(text: &str) -> Vec<ChatEvent> {
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        for b in text.as_bytes() {
            let s = std::str::from_utf8(std::slice::from_ref(b)).expect("ascii-only transcript");
            scanner.push(s, &mut events);
        }
        scanner.finish(&mut events);
        coalesce(events)
    }

    #[test]
    fn scanner_chunking_invariance() {
        let text = transcript();
        let whole = run_whole(text);
        assert!(!whole.is_empty());
        // every possible single split point
        for at in 0..=text.len() {
            if !text.is_char_boundary(at) {
                continue;
            }
            let split = run_split_at(text, at);
            assert_eq!(split, whole, "mismatch splitting at byte {at}");
        }
        // one byte at a time
        let byte_by_byte = run_byte_at_a_time(text);
        assert_eq!(byte_by_byte, whole, "mismatch feeding one byte at a time");
    }

    // ---- scanner: Qwen3.8 XML tool-call flavor ----------------------------------------------------

    /// Qwen3.8 emits a tool call as XML nested in the SAME `<tool_call>` fence
    /// the Qwen3 JSON form uses: `<function=name>` followed by one
    /// `<parameter=key>` block per argument. The scanner reconstructs a JSON
    /// arguments object from them - the same `ToolCallMsg.arguments` shape the
    /// Qwen3 flavor produces, and the exact inverse of the render direction
    /// (`append_xml_parameters`: string values verbatim, every other value
    /// re-serialized JSON).
    fn xml_transcript() -> &'static str {
        "<think>\nchecking the weather\n</think>\n\nLet me look that up.\
         <tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\
         Done."
    }

    /// How a transcript is fed to the scanner: one push, one split, or one
    /// byte per push - the chunking-invariance axes.
    enum Chunk {
        Whole,
        SplitAt(usize),
        ByteAtATime,
    }

    /// Scan `text` through a [`ChatScanner`] of `flavor` fed per `chunk`, with
    /// the same fragment-coalescing the Qwen3-flavor helpers above apply.
    fn run_flavor(text: &str, flavor: TemplateFlavor, chunk: Chunk) -> Vec<ChatEvent> {
        let mut scanner = ChatScanner::with_flavor(false, flavor);
        let mut events = Vec::new();
        match chunk {
            Chunk::Whole => scanner.push(text, &mut events),
            Chunk::SplitAt(at) => {
                scanner.push(&text[..at], &mut events);
                scanner.push(&text[at..], &mut events);
            }
            Chunk::ByteAtATime => {
                for b in text.as_bytes() {
                    let s = std::str::from_utf8(std::slice::from_ref(b)).expect("ascii-only transcript");
                    scanner.push(s, &mut events);
                }
            }
        }
        scanner.finish(&mut events);
        coalesce(events)
    }

    /// The full XML transcript must scan identically however the stream is
    /// chunked - the same streaming invariant the Qwen3 flavor holds to.
    #[test]
    fn scanner_qwen38_xml_chunking_invariance() {
        let text = xml_transcript();
        let whole = run_flavor(text, TemplateFlavor::Qwen38, Chunk::Whole);
        assert!(!whole.is_empty());
        for at in 0..=text.len() {
            if !text.is_char_boundary(at) {
                continue;
            }
            assert_eq!(run_flavor(text, TemplateFlavor::Qwen38, Chunk::SplitAt(at)), whole, "mismatch splitting at byte {at}");
        }
        assert_eq!(run_flavor(text, TemplateFlavor::Qwen38, Chunk::ByteAtATime), whole, "mismatch feeding one byte at a time");
    }

    #[test]
    fn scanner_qwen38_xml_tool_call_reconstructs_json_arguments() {
        let text = xml_transcript();
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::Qwen38);
        let mut events = Vec::new();
        scanner.push(text, &mut events);
        scanner.finish(&mut events);
        assert_eq!(scanner.content(), "\n\nLet me look that up.Done.");
        // The transcript's `<think>\n...\n</think>` framing is stripped: the
        // scanner is the exact inverse of the render, which writes one framing
        // newline on each side of the reasoning.
        assert_eq!(scanner.reasoning(), "checking the weather");
        let [call] = scanner.tool_calls() else { panic!("expected exactly one tool call, got {:?}", scanner.tool_calls()) };
        assert_eq!(call.id, "call_0");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"location": "Paris", "days": 3}"#);
        // Streaming events stay coherent: Start names the call, the adjacent
        // Args fragments concatenate to the same arguments string, End closes.
        assert_eq!(
            coalesce(events),
            vec![
                ChatEvent::Reasoning("checking the weather".to_string()),
                ChatEvent::Content("\n\nLet me look that up.".to_string()),
                ChatEvent::ToolCallStart { index: 0, name: "get_weather".to_string() },
                ChatEvent::ToolCallArgs { index: 0, fragment: r#"{"location": "Paris", "days": 3}"#.to_string() },
                ChatEvent::ToolCallEnd { index: 0 },
                ChatEvent::Content("Done.".to_string()),
            ]
        );
    }

    /// Value forms: a bare string that is not JSON (escaped), a number and a
    /// nested JSON literal (passed through verbatim - the inverse of the
    /// render's string-verbatim/others-tojson split), and a parameter-less
    /// call (empty object).
    #[test]
    fn scanner_qwen38_xml_value_forms() {
        let text = "<tool_call>\n<function=a>\n<parameter=s>\nline1\nline2 \"quoted\"\n</parameter>\n<parameter=n>\n42\n</parameter>\n<parameter=o>\n{\"k\": [1, 2]}\n</parameter>\n</function>\n</tool_call>\
                    <tool_call>\n<function=b>\n</function>\n</tool_call>tail";
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::Qwen38);
        let mut events = Vec::new();
        scanner.push(text, &mut events);
        scanner.finish(&mut events);
        let calls = scanner.tool_calls();
        assert_eq!(calls.len(), 2, "calls: {calls:?}");
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments, r#"{"s": "line1\nline2 \"quoted\"", "n": 42, "o": {"k": [1, 2]}}"#);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, "{}");
        assert_eq!(scanner.content(), "tail");
    }

    /// A call cut off mid-generation still closes out rather than dropping
    /// (the JSON flavor's finish policy): a mid-VALUE cut takes the buffered
    /// text as the value; a mid-KEY cut contributes nothing; both close the
    /// arguments object.
    #[test]
    fn scanner_qwen38_xml_truncated_call_still_closes() {
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::Qwen38);
        let mut events = Vec::new();
        scanner.push("<tool_call>\n<function=f>\n<parameter=k>\npartial value", &mut events);
        scanner.finish(&mut events);
        let [call] = scanner.tool_calls() else { panic!("call must close on finish, got {:?}", scanner.tool_calls()) };
        assert_eq!(call.name, "f");
        assert_eq!(call.arguments, r#"{"k": "partial value"}"#);

        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::Qwen38);
        let mut events = Vec::new();
        scanner.push("<tool_call>\n<function=f>\n<parameter=ke", &mut events);
        scanner.finish(&mut events);
        let [call] = scanner.tool_calls() else { panic!("call must close on finish, got {:?}", scanner.tool_calls()) };
        assert_eq!(call.name, "f");
        assert_eq!(call.arguments, "{}");
    }

    /// A Qwen3-JSON payload under the 3.8 flavor is not the expected header;
    /// it surfaces as content rather than a silently dropped call (the same
    /// policy as the JSON flavor's invalid-header path).
    #[test]
    fn scanner_qwen38_json_shaped_call_surfaces_as_content() {
        let text = "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>";
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::Qwen38);
        let mut events = Vec::new();
        scanner.push(text, &mut events);
        scanner.finish(&mut events);
        assert!(scanner.tool_calls().is_empty());
        assert_eq!(scanner.content(), "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>");
    }

    /// The scanner is the exact inverse of the render direction: an assistant
    /// tool_calls message rendered with the Qwen3.8 flavor scans back to the
    /// same calls (arguments compared as parsed JSON - the render's own
    /// `| tojson` spacing need not survive).
    #[test]
    fn render_scan_roundtrip_qwen38_tool_calls() {
        let calls = vec![ToolCallMsg { id: "call_0".into(), name: "get_weather".into(), arguments: r#"{"location": "Paris", "days": 3}"#.into() }];
        let msgs = vec![ChatMessage { reasoning_content: Some("hmm".into()), ..ChatMessage::assistant("Let me look.") }.with_tool_calls(calls)];
        let rendered = render(&msgs, &[], TemplateOpts { flavor: TemplateFlavor::Qwen38, add_generation_prompt: false, ..Default::default() }).unwrap();
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::Qwen38);
        let mut events = Vec::new();
        scanner.push(&rendered, &mut events);
        scanner.finish(&mut events);
        assert_eq!(scanner.reasoning(), "hmm");
        let [call] = scanner.tool_calls() else { panic!("rendered tool call must scan back, got {:?} from {rendered:?}", scanner.tool_calls()) };
        assert_eq!(call.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
            serde_json::from_str::<serde_json::Value>(r#"{"location": "Paris", "days": 3}"#).unwrap()
        );
    }

    // ---- scanner: tool call argument/index details ----------------------------------------------------

    #[test]
    fn scanner_argument_fragments_concatenate_to_valid_json() {
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\", \"days\": 3}}\n</tool_call>";
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        // split into many small chunks to exercise fragment accumulation
        for chunk in text.as_bytes().chunks(3) {
            scanner.push(std::str::from_utf8(chunk).unwrap(), &mut events);
        }
        scanner.finish(&mut events);
        assert_eq!(scanner.tool_calls().len(), 1);
        let call = &scanner.tool_calls()[0];
        assert_eq!(call.name, "get_weather");
        let v: serde_json::Value = serde_json::from_str(&call.arguments).expect("valid json");
        assert_eq!(v["location"], "Paris");
        assert_eq!(v["days"], 3);
    }

    #[test]
    fn scanner_parallel_calls_get_sequential_indices() {
        let text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\
                    <tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>";
        let mut events = Vec::new();
        let mut scanner = ChatScanner::new(false);
        scanner.push(text, &mut events);
        scanner.finish(&mut events);
        assert_eq!(scanner.tool_calls()[0].name, "a");
        assert_eq!(scanner.tool_calls()[1].name, "b");
        assert!(events.contains(&ChatEvent::ToolCallStart { index: 0, name: "a".into() }));
        assert!(events.contains(&ChatEvent::ToolCallStart { index: 1, name: "b".into() }));
        assert!(events.contains(&ChatEvent::ToolCallEnd { index: 0 }));
        assert!(events.contains(&ChatEvent::ToolCallEnd { index: 1 }));
    }

    #[test]
    fn scanner_finish_closes_unterminated_call() {
        // simulate a max_new_tokens cut mid-arguments (no closing brace, no </tool_call>)
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Par";
        let mut scanner = ChatScanner::new(false);
        let mut events = Vec::new();
        scanner.push(text, &mut events);
        assert!(scanner.tool_calls().is_empty(), "must not close early, before finish()");
        scanner.finish(&mut events);
        assert_eq!(scanner.tool_calls().len(), 1);
        let call = &scanner.tool_calls()[0];
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, "{\"location\": \"Par");
        assert!(matches!(events.last(), Some(ChatEvent::ToolCallEnd { index: 0 })));
    }

    #[test]
    fn scanner_thinking_open_seeds_think_state() {
        let mut scanner = ChatScanner::new(true);
        let mut events = Vec::new();
        scanner.push("already inside a think block</think>after", &mut events);
        scanner.finish(&mut events);
        assert_eq!(
            events,
            vec![
                ChatEvent::Reasoning("already inside a think block".to_string()),
                ChatEvent::Content("after".to_string()),
            ]
        );
    }

    // ---- render: reasoning_effort ----------------------------------------------------

    #[test]
    fn reasoning_effort_xhigh_injects_system_instruction() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts {
            enable_thinking: true,
            reasoning_effort: Some("xhigh".into()),
            ..Default::default()
        }).unwrap();
        assert!(s.contains("Reasoning effort is set to xhigh"), "xhigh should inject instruction: {s:?}");
        assert!(s.contains("validate key assumptions"), "xhigh should mention validation: {s:?}");
    }

    #[test]
    fn reasoning_effort_medium_injects_nothing() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts {
            enable_thinking: true,
            reasoning_effort: Some("medium".into()),
            ..Default::default()
        }).unwrap();
        assert!(!s.contains("Reasoning effort"), "medium should not inject instruction: {s:?}");
    }

    #[test]
    fn reasoning_effort_low_injects_brief_instruction() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts {
            enable_thinking: true,
            reasoning_effort: Some("low".into()),
            ..Default::default()
        }).unwrap();
        assert!(s.contains("Reasoning effort is set to low"), "low should inject instruction: {s:?}");
        assert!(s.contains("brief and focused"), "low should mention brevity: {s:?}");
    }

    #[test]
    fn reasoning_effort_none_means_no_injection() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts {
            enable_thinking: true,
            reasoning_effort: None,
            ..Default::default()
        }).unwrap();
        assert!(!s.contains("Reasoning effort"), "None should inject nothing (caller resolves defaults): {s:?}");
    }

    #[test]
    fn reasoning_effort_ignored_when_thinking_disabled() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts {
            enable_thinking: false,
            reasoning_effort: Some("xhigh".into()),
            ..Default::default()
        }).unwrap();
        assert!(!s.contains("Reasoning effort"), "should be ignored when thinking disabled: {s:?}");
    }

    #[test]
    fn reasoning_effort_invalid_value_errors() {
        let msgs = vec![ChatMessage::user("hi")];
        let result = render(&msgs, &[], TemplateOpts {
            enable_thinking: true,
            reasoning_effort: Some("high".into()),
            ..Default::default()
        });
        assert!(result.is_err(), "invalid effort should error");
        assert!(result.unwrap_err().contains("Unexpected reasoning effort"), "error message should name the value");
    }

    #[test]
    fn reasoning_effort_with_tools_goes_before_tools() {
        let tools = vec![r#"{"type":"function","function":{"name":"get_weather"}}"#.into()];
        let msgs = vec![ChatMessage::user("weather?")];
        let s = render(&msgs, &tools, TemplateOpts {
            enable_thinking: true,
            reasoning_effort: Some("xhigh".into()),
            ..Default::default()
        }).unwrap();
        let re_pos = s.find("Reasoning effort").unwrap();
        let tools_pos = s.find("# Tools").unwrap();
        assert!(re_pos < tools_pos, "reasoning should come before tools: re_pos={re_pos}, tools_pos={tools_pos}");
    }

    // ---- render: Qwen3.8 template flavor (Qwen38) ------------------------------------
    //
    // Every expectation below is transcribed from the real Qwen3.8
    // tokenizer_config.json chat_template and cross-validated against the
    // template itself in tests/chat_template_cross_check.rs (the
    // matches_qwen_chat_qwen38_* tests there) whenever that template is on
    // the box.

    fn qwen38_msgs_with_history_reasoning() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("q1"),
            ChatMessage { role: Role::Assistant, content: "a1".into(), reasoning_content: Some("thinking one".into()), ..Default::default() },
            ChatMessage::user("q2"),
            ChatMessage::assistant("a2"),
        ]
    }

    #[test]
    fn qwen38_generation_prompt_prefills_an_open_think_tag() {
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts { enable_thinking: true, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n"), "got: {s:?}");
        // Thinking disabled stays a closed empty block (same as Qwen3).
        let s = render(&msgs, &[], TemplateOpts { enable_thinking: false, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_preserve_thinking_default_keeps_history_reasoning() {
        // Default (kwarg undefined): EVERY assistant turn keeps its think
        // block - the pre-query turn, and the final turn that has no
        // reasoning at all (empty reasoning, still framed).
        let msgs = qwen38_msgs_with_history_reasoning();
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: false, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.contains("<|im_start|>assistant\n<think>\nthinking one\n</think>\n\na1<|im_end|>\n"), "got: {s:?}");
        assert!(s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_preserve_thinking_false_strips_pre_query_reasoning() {
        let msgs = qwen38_msgs_with_history_reasoning();
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: false, flavor: TemplateFlavor::Qwen38, preserve_thinking: Some(false), ..Default::default() }).unwrap();
        // a1 sits before the last real user query (index 1 < 2): stripped.
        assert!(s.contains("<|im_start|>assistant\na1<|im_end|>\n"), "got: {s:?}");
        assert!(!s.contains("thinking one"), "got: {s:?}");
        // The trailing assistant (after the last query) keeps its block.
        assert!(s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_preserve_thinking_true_matches_the_default() {
        let msgs = qwen38_msgs_with_history_reasoning();
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: false, flavor: TemplateFlavor::Qwen38, preserve_thinking: Some(true), ..Default::default() }).unwrap();
        assert!(s.contains("<|im_start|>assistant\n<think>\nthinking one\n</think>\n\na1<|im_end|>\n"), "got: {s:?}");
        assert!(s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_history_reasoning_and_content_are_trimmed() {
        let msgs = vec![
            ChatMessage::user("q"),
            ChatMessage { role: Role::Assistant, content: "\n answer \n".into(), reasoning_content: Some(" \n hard think \n".into()), ..Default::default() },
        ];
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: false, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.contains("<|im_start|>assistant\n<think>\nhard think\n</think>\n\nanswer<|im_end|>\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_reasoning_content_only_never_splits_embedded_think() {
        // The Qwen3.8 template reads the reasoning_content FIELD only; a
        // literal <think> block inside content stays in content verbatim
        // (the Qwen3 flavor's split-and-reframe fallback is gone).
        let msgs = vec![
            ChatMessage::user("q"),
            ChatMessage { role: Role::Assistant, content: "<think>old</think>answer".into(), ..Default::default() },
        ];
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: false, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n<think>old</think>answer<|im_end|>\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_history_tool_calls_render_xml_parameters() {
        let msgs = vec![
            ChatMessage::user("weather in Paris?"),
            ChatMessage {
                role: Role::Assistant,
                content: "checking".into(),
                tool_calls: vec![ToolCallMsg { id: "c1".into(), name: "get_weather".into(), arguments: r#"{"city": "Paris", "n": 2}"#.into() }],
                ..Default::default()
            },
            ChatMessage::tool("18C, sunny"),
            ChatMessage::assistant("It is 18C."),
        ];
        let s = render(&msgs, &[], TemplateOpts { add_generation_prompt: false, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        let expected = "<|im_start|>assistant\n<think>\n\n</think>\n\nchecking\n\n<tool_call>\n<function=get_weather>\n\
             <parameter=city>\nParis\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n";
        assert!(s.contains(expected), "got: {s:?}");
        // Tool results are unchanged from the Qwen3 flavor (modulo trimming).
        assert!(s.contains("<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response><|im_end|>\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_tools_preamble_matches_upstream() {
        let tools = vec![r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}"#.to_string()];
        let msgs = vec![ChatMessage::system("Be terse."), ChatMessage::user("hi")];
        let s = render(&msgs, &tools, TemplateOpts { add_generation_prompt: true, enable_thinking: false, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.starts_with("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"), "got: {s:?}");
        assert!(s.contains("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
             <tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n\
             </parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n\
             that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
             - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
             - Required parameters MUST be specified\n\
             - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
             - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
             </IMPORTANT>\n\nBe terse.<|im_end|>\n"), "system content must come AFTER the <IMPORTANT> block: {s:?}");
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "got: {s:?}");
    }

    #[test]
    fn qwen38_reasoning_directive_layout_matches_the_template() {
        // With system content: directive + exactly one blank line, then the
        // content. Without: the directive turn carries no trailing blank line.
        let low = "Reasoning effort is set to low. Keep your thinking brief and focused, \
             moving directly to the conclusion without unnecessary elaboration.";
        let msgs = vec![ChatMessage::system("Be terse."), ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts { enable_thinking: true, reasoning_effort: Some("low".into()), flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.starts_with(&format!("<|im_start|>system\n{low}\n\nBe terse.<|im_end|>\n")), "got: {s:?}");
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts { enable_thinking: true, reasoning_effort: Some("low".into()), flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert!(s.starts_with(&format!("<|im_start|>system\n{low}<|im_end|>\n")), "got: {s:?}");
    }

    #[test]
    fn qwen38_reasoning_effort_defaults_to_xhigh_at_render_time() {
        // The template's own |default('xhigh'): unlike the Qwen3 flavor,
        // None and Some("xhigh") render identically under Qwen38.
        let msgs = vec![ChatMessage::user("hi")];
        let s = render(&msgs, &[], TemplateOpts { enable_thinking: true, reasoning_effort: None, flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        let s_xhigh = render(&msgs, &[], TemplateOpts { enable_thinking: true, reasoning_effort: Some("xhigh".into()), flavor: TemplateFlavor::Qwen38, ..Default::default() }).unwrap();
        assert_eq!(s, s_xhigh);
        assert!(s.contains("Reasoning effort is set to xhigh"), "got: {s:?}");
    }
}
