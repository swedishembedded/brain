<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 458. An API surface that refuses what its main client always sends is not a subset, it is unusable

**Lesson:** Before calling a protocol surface "the implemented subset", capture what its real clients send. A
field the main client puts in every request (here `tools` and `thinking: adaptive`) is the floor of the subset, not an
extension of it; and a number the client reads to manage its own context (the prompt's token count) must be the real one
somewhere in every reply, streamed or not.

## What happened

The Anthropic surface (`/v1/messages`) refused any request with `tools` or `tool_use`/`tool_result` blocks with a 400
("full Anthropic tool-calling ... is a documented follow-up") and accepted `thinking` only as `enabled`/`disabled`. Its
tests and the serving docs presented that as a working Anthropic API, and `samples/shell/api/claude-with-brain` points
Claude Code at it. Requests captured from Claude Code 2.1.220 in `--bare -p` mode against a recording server
(2026-11-14) show that every request of a session carries `tools` (its Bash, Edit and Read tools, as JSON schemas),
`thinking: {"type": "adaptive"}`, `output_config`, `metadata` and `cache_control` markers, and from the second turn on
assistant `tool_use` and user `tool_result` blocks, often two calls in one turn. So no Claude Code turn could ever be
answered: the surface served only clients that use none of the protocol's agent features.

Claude Code streams every turn, and a stream's only prompt count was the `message_start` estimate (content characters /
4, before the prompt is rendered): the client sized its context on a number that leaves out the chat template and the
tool schemas, which on a tool-carrying request are most of the prompt.

## Fix and evidence

- `crates/apiserve/src/anthropic.rs` maps client tools and `tool_choice` onto the contract's (the OpenAI function shape
  the chat templates read, with the OpenAI surface's bounds; schemas keep the client's member order), `tool_use` blocks
  onto the assistant turn's `tool_calls`, `tool_result` blocks onto `tool` turns, and the model's calls back onto
  `tool_use` blocks, streamed as `input_json_delta`s with `stop_reason: "tool_use"`. `adaptive` thinking leaves the
  model's default. Hosted server tools and `disable_parallel_tool_use: true` are refused by name. Gated in
  `crates/apiserve/tests/it/api.rs` against the vendored Anthropic schema
  (`anthropic_tool_calls_nonstream_are_tool_use_blocks_with_object_inputs`,
  `anthropic_tool_calls_stream_as_tool_use_blocks_of_input_json_deltas`) and in-module
  (`a_tool_using_conversation_maps_onto_the_contract`, `unservable_tool_requests_are_refused`).
- The closing `message_delta` repeats the real `input_tokens`, and every reply's `usage` carries
  `cache_read_input_tokens` when the model reports its prefix cache's share
  (`anthropic_messages_stream_orders_events_and_concatenates`, `usage_carries_the_prefix_caches_share_of_the_prompt`).

## Not done

`count_tokens` and the `message_start` estimate are still characters / 4.
