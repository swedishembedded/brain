<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 536. A tool call written inside an unclosed think block is lost as reasoning

**Lesson:** Qwen3-4B sometimes writes `<tool_call>{...}</tool_call>` before it closes `</think>`. Brain's output
scanner (`crates/data/src/qwen_chat.rs`, `ChatScanner`) recognises only `</think>` while in the think state
(`MARKERS_THINK`). Such a call therefore becomes reasoning text: the reply carries no `tool_calls`, and the agent
loop sees a turn that did nothing. The fix is still open.

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

## Next

- **Decide the parse rule against the reference.** One option: a `<tool_call>` in the think state closes the think
  block implicitly and is parsed as a call. Check what the reference Qwen3 parsers do with an unclosed think block
  before choosing.
- **Spec first:** streaming and whole-output chunking must give the same events, an anomaly must be recorded so the
  rescue is visible (never silent), and the training records must render what was served.
- **Then re-measure** the same practice tasks on the fixed build. The pass rate holds per serving build (the
  examiner's own finding), so compare within one build pair only.
