<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 214. A tool conversation cannot be masked by message boundary

**Lesson:** A mask that depends on cutting one rendering of a whole conversation is only as sound
as the template's prefix stability. When a template is context-sensitive, render each supervised
turn after its own prefix and prove it against the serving renderer.

## What happened
Whole-record masking (`ChatSample::encode_with`) renders the conversation once and cuts it at the
byte ranges found by rendering every prefix. The boundary search refuses a conversation whose first
assistant turn renders differently as a prefix than inside the whole (`message 1 is not
prefix-stable`).

Every multi-step tool conversation (user, call, result, call, result, answer) has exactly that
shape: none could be trained, in either thinking mode.

## Why
Qwen3's template writes an empty `<think>\n\n</think>\n\n` block on an assistant turn only when it
is the last message. So in any conversation with two assistant turns after the last user turn, the
first one renders differently as a prefix than inside the whole.

`ChatSample::answers_as_asked` had the way out: one example per supervised turn, the prompt
rendered for the conversation before it and the completion the strict suffix after it. But it
returned `None` for a turn that called a tool, so the fine-tune fell back to the refused path.

## What to do
`answers_as_asked` takes assistant turns that carry tool calls, and the fine-tune uses it in
thinking mode whenever a supervised turn calls a tool (and, as before, for every record in no-think
mode). Plain-answer datasets in thinking mode still take the whole-record path, so their encoding
is unchanged.

The remaining shape was a record that supervises only its final answer after unsupervised tool
turns: no supervised turn calls a tool, so it still took the refused whole-record path.
`ChatTemplate::render_with_supervised_suffix` is the one boundary that needs:

- The context rendered with the generation prompt (same reasoning flags) must be a byte prefix of
  the full render.
- The supervised span is what follows.
- A no-think block the prompt pre-fills is context.

`answers_as_asked` is that call once per supervised turn; `ChatSample::final_turn_as_asked` is it
for a record that supervises its last message alone. A record that supervises two or more turns
without a tool call among them is still masked as a whole and still refused.

## Evidence
The six-message conversation above went from 1 refusal to 3 examples (one per assistant turn), and
for each, in thinking and no-think mode (6 checks), prompt and completion equal the serving
renderer's (`qwen_chat::render_for_generation` of the prefix, and `render` of the prefix plus the
turn minus that prompt) with 0 differing bytes.
