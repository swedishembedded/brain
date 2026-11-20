<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 536. A tool call written inside an unclosed think block is lost as reasoning

**Lesson:** Qwen3-4B sometimes writes `<tool_call>{...}</tool_call>` before it closes `</think>`. Brain's output
scanner (`crates/data/src/qwen_chat.rs`, `ChatScanner`) recognised only `</think>` while in the think state, so such a
call became reasoning text: the reply carried no `tool_calls`, and the agent loop saw a turn that did nothing. Fixed:
a complete call now ends the reasoning, as the reference parser does, and is reported as
`tool_call_rescued_from_think`.

## Evidence (2026-11-20)

- **Source:** one accounting practice job of an agent runtime (48 trials, brain 1bdd5400c, a Qwen3-4B LoRA adapter).
  The agent's request log holds 1,259 responses.
- **Responses:** 59 of the 1,259 have no `tool_calls`. Of those 59, 14 contain a well-formed `<tool_call>` JSON object.
  13 of the 14 are in `reasoning` with empty `text`, so the think block was never closed.
- **Trials:** 11 of the 48 trials lost at least one call this way, and 2 ended on such a turn. The lost calls are real
  work: `shell` calls that run the computation or write `answer.json`.
- A reviewer of about 150 failed trials from the same campaign (agent side only) estimated that this pattern, a tool call written as text in
  the reasoning, appears in a quarter to a third of failures across all four domains. The count above is one accounting
  job, so it is a lower bound for that job only.

## The reference rule

- **vLLM's current Qwen3 parser** (`vllm/parser/qwen3.py`, main at 9b9bd049, 2026-11-08; the engine came in
  vllm-project/vllm#45413) has the grammar transition `(REASONING, TOOL_START) -> TOOL_PREAMBLE` emitting
  `REASONING_END, TOOL_CALL_START`, commented "Tool call directly from reasoning (implicit end)". `</think>` after it,
  in the content state, is absorbed.
- **vLLM's older split parsers** reach the same call by another road: `Qwen3ReasoningParser.extract_reasoning`
  (v0.10.0, v0.11.0) returns the whole output as content when `</think>` is missing, and the Hermes tool parser
  (`vllm/tool_parsers/hermes_tool_parser.py`) then extracts every `<tool_call>` from that content with `json.loads`,
  requiring `name` and `arguments`; a payload that fails to parse gives no call at all.
- **The Qwen3 chat template** renders every assistant turn that carries reasoning as
  `<think>\n{reasoning}\n</think>\n\n{content}` before its calls, so a parsed reply always re-renders with a
  `</think>`.

## The rule brain applies

- In the think state `<tool_call>` is a marker beside `</think>`. Text before it is reasoning, minus one newline right
  before it (the framing the template writes before a call).
- The call is buffered until its `</tool_call>`. It is a call only when that block, read by the same scanner a call
  after `</think>` goes through, gives exactly one call with a non-empty name, a JSON-object `arguments`, no text
  around it and no anomaly. Then the reasoning ends there, the block is read as an ordinary call, and the scan
  records `ToolCallRescuedFromThink` (`tool_call_rescued_from_think` in `scan_anomalies`). The Qwen3.8 XML call
  shares the path.
- Anything else stays reasoning, byte for byte, with `ToolCallInsideThink`: a payload that is not a call, a
  `</think>` before the `</tool_call>`, an output that ends inside the call (also `EndedInsideThink`), and a bare
  `<function=` without the `<tool_call>` fence (vLLM has no reasoning-state transition for it either).
- brain is stricter than vLLM in two places: an incomplete call (no `</tool_call>`) is not a call, and text after a
  rescued call is plain content, so a later `</think>` stays in the content rather than being dropped.
- Specs: the `a_*call*_think_block*` tests in `qwen_chat.rs`, each checked at every split point and one byte at a
  time.

## Training records

A served rescued turn has no `</think>`; its parsed message (reasoning plus calls) renders with one. A rollout record
that carries the sampled `token_ids` trains those ids: `check_sampled` admits them because they read back
(`ChatTemplate::reads_as_same_reply`) as the same reasoning and call, so the trainer trains what was served
(`a_call_written_inside_an_unclosed_think_block_trains_the_ids_that_were_served`). Before the fix the same record was
refused as `TokenIdsMismatch`. A record without `token_ids` is imitation from the message and renders the canonical
form, with `</think>`.

## Open

- **Re-measure** the same practice tasks on a build with the fix. The pass rate holds per serving build (the
  examiner's own finding), so compare within one build pair only.
- The thinking budget and the reasoning-token count are token-level and still count a rescued call's tokens as
  reasoning, since only `</think>` ends the block for them.
