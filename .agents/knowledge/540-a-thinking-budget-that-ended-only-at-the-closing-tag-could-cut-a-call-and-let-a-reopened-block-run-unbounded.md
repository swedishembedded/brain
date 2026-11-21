<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 540. A thinking budget that ended only at the closing tag could cut a call and let a reopened block run unbounded

**Lesson:** a token budget that an agent harness grades against must be a property of the turn, not of one think
block, and the reported reasoning count must come from the parse, not from a second token-level rule. The scheduler's
`ThinkState` (`crates/model/src/serve/sampler.rs`) and `qwen3::chat::reasoning_tokens` both treated `</think>` as the
only end of reasoning, so three things went wrong:

- **A call inside the block was cut.** A model that writes `<tool_call>` before `</think>` (#536) kept spending the
  budget on the call's tokens, and a budget that ran out inside the call forced the lead-in and `</think>` into the
  middle of its JSON: the call broke and the turn did nothing.
- **A second block was free.** Once the first block closed, `ThinkState` stopped watching, so a block the model
  opened again in the same turn ran with no budget; a nested `<think>` also reset the count. The budget did not bound
  the turn.
- **The count disagreed with the parse.** `reasoning_tokens` counted every token between the tags, so a call that the
  scanner reads as the end of the reasoning (#536) was reported as reasoning, and text a stop string later cut (first
  part already scanned inside a think block) was reported both in `reasoning_content` and in the count.

## The rules now

- **Per turn.** `thinking_budget` is the turn's: the model's own tokens inside every think block it opens, outside
  call fences, never exceed it. At the budget the lead-in (once a turn) and the closing sequence are forced; a block
  opened after the budget is spent is closed at once with the closing sequence alone.
- **Calls are actions.** The frontend hands the scheduler the tokens that open and close a call
  (`ThinkingBudget::with_call_fence`). Fence tokens are not counted and nothing is forced inside a fence. Once a fence
  has closed inside the block, a budget that runs out ends the turn with the request's end-of-turn token instead of
  closing the block, so the call stays the turn's last word.
- **Windows and speculation.** Outside a block the scheduler may decode a multi-token window or verify a draft
  again (it had to stay host-checked from the start before); every token still reaches `ThinkState::accept`, and a
  token the budget forces mid-window replaces that slot and ends the window, so the draws that followed the model's
  own choice are dropped with their cache rows.
- **Exact count.** `reasoning_tokens` is the number of generated tokens that start inside the byte ranges the scanner
  read as reasoning (`ChatScanner::reasoning_spans`, `tokens_in_spans`), with token starts recorded by
  `Utf8Decoder::feed_marked` as the text streams. A stop string's text is never reasoning: when a stop completes over
  several steps, the outcome is the scan of the text before it alone (`ChatScanner::rescan`).

## Specs

- `crates/model/src/serve/sampler.rs`: `a_thinking_budget_is_shared_by_every_block_of_the_turn`,
  `a_call_inside_the_block_is_never_cut_and_running_out_after_it_ends_the_turn`, and
  `a_thinking_budget_always_terminates` (2,000 random token streams: model reasoning outside fences never exceeds
  the budget).
- `crates/model/tests/it/serve_sampling.rs`: the same through the scheduler with a scripted decoder whose windows
  are input-sensitive (a window that ran on past a forced token shows the wrong draw).
- `crates/data/src/qwen_chat.rs`: `the_reasoning_spans_cover_what_the_parse_reads_as_reasoning` at every chunking.
- `crates/qwen3/src/chat.rs`: `reasoning_tokens_are_the_tokens_the_parse_reads_as_reasoning`.

## Limits

- A fenced payload the parser does not accept as a call stays reasoning in the parse, so the reported
  `reasoning_tokens` may exceed the budget by that payload's length; the model's own unfenced reasoning cannot.
- A frontend whose vocabulary lacks the call tags (no `<tool_call>` special token) gets no fence: its budget counts a
  call written inside the block like any reasoning, as before.
