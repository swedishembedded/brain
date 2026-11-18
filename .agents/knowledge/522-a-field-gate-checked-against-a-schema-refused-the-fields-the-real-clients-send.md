<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 522. A field gate checked against a schema refused the fields the real clients send

**Lesson:** A refusal of undeclared fields is gated by the requests the clients actually send, not by the dialect's
schema. Record a real client's body against a recording endpoint, keep it as a fixture, and run the client itself
where it is installed. The bodies change with every client release.

## What happened

41b63e673 refused every top-level body field a dialect's parser does not read, except a fixed table of ignored
fields. Its tests used the dialects' schemas and hand-written bodies. Four groups of requests then failed with a 400.

The real `claude` CLI, 2.1.325, against the Anthropic route of `brain serve --mock`. `tests/e2e/claude_code.bats`
failed 3 of its 5 tests. The recorded body carries:

- `context_management: {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}`
- `output_config: {"effort": "high"}`
- `thinking: {"type": "adaptive", "display": "omitted"}`
- a tool with `defer_loading: true`
- the `context-management-2025-06-27`, `advanced-tool-use-2025-11-20` and `effort-2025-11-24` betas

The first refusal was `'context_management' is not supported`, and `output_config` came after it.

The mock model's `generate` spec lacked `chat` (a raw `/v1/completions` prompt) and `thinking_budget` (Anthropic's
`budget_tokens`). Two `api_conformance.bats` cases therefore failed.

Declaring `thinking_budget` then showed a third failure: the OpenRouter model card listed `thinking_budget` and its
aliases in `supported_parameters`. OpenRouter's schema enumerates that field, so every real Qwen3.8 card had been
out of schema too.

An agent runtime's OpenAI driver and an exam harness both send `stream_options` on every request, and
`chat_template_kwargs.enable_thinking` and `thinking_budget`. These were admitted.

## Fix

- The Anthropic route reads `output_config.effort`. `high` (the default) and `max` leave the model's default.
  `medium` and `low` set `reasoning_effort`, and a model without that setting refuses it. `format` is refused by name.
- The Anthropic route reads `context_management`. The no-op `clear_thinking_20251015` with `keep: "all"` passes. Any
  edit that drops part of the prompt is refused by name.
- A `defer_loading` tool is not shown to the model, because brain runs no tool search.
- A request-wide `cache_control` is accepted and ignored, as block-level ones are.
- The mock declares `chat` and `thinking_budget`.
- OpenRouter cards list only the parameters OpenRouter's schema enumerates.
- `tests/e2e/api_conformance.bats` now expects a chat body's `__adapter` to be refused by name. That is the contract
  41b63e673 set; the old assertion expected the field to be ignored.

Tests:

- `catalog` `client_shapes` has one fixture per client shape, run against the served Qwen3.8 `generate` spec: the
  agent runtime's OpenAI and Anthropic drivers, the exam harness, and Claude Code 2.1.325 as recorded.
- `claude_code.bats` passes 5 of 5 and `api_conformance.bats` passes 49 of 49.

## Not covered

- `thinking.display: "omitted"` is accepted, and the reply still carries the thinking text.
- Claude Code's `anthropic-beta` header is not read.
