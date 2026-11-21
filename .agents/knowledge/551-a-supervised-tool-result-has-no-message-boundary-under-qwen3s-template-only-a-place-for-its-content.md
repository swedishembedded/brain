<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 551. A supervised tool result has no message boundary under Qwen3's template, only a place for its content

**Lesson:** to train a model on a tool's results, do not look for the message's byte range in the conversation.
Qwen3's template writes the `<|im_start|>user` header and the `<|im_end|>` of a tool turn only around a run of
consecutive tool messages, and it writes the reasoning block of an assistant turn only when that turn is last. So how
a tool message renders depends on its neighbours, and probing one prefix at a time
(`ChatTemplate::render_with_message_boundaries`) refuses the record as not prefix-stable. What is stable is where the
content sits in one rendering of the whole conversation. `ChatSample::observations_in_place` finds it by rendering
the conversation once with a unique sentinel in place of each supervised content. It then checks that putting the
real contents back gives the real rendering byte for byte, and refuses the record otherwise. The supervised span is
the content alone; the template's framing is context.

## Evidence (2026-11-22)

- Real Qwen3-4B tokenizer and template (`QWEN3_DIR`), on an editor session of three assistant turns with
  `reasoning_content`, each calling a tool, and two results, one weighted 0.3:
  `crates/data/tests/it/chat_sample_encode.rs`, `qwen3_supervised_observations_are_their_contents_in_the_real_rendering`.
  The supervised runs decode to exactly the two screen contents with their weights. The decoded example is the
  template's own rendering cut after the last result. It ends `<tool_response>\n` followed by the content.
- The whole-record path refuses the same shape of conversation as not prefix-stable
  (`crates/data/tests/it/chat_sample_encode.rs`, second test).
- A template that changes the content (an `upper` filter in the test template) is refused as "not placed
  verbatim". No boundary is guessed.
