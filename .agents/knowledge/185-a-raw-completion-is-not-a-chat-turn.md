<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 185. A raw completion is not a chat turn

`/v1/completions` continues a prompt with no chat template. The first
version reused the chat path's per-sequence state, and with it the chat
scanner. The scanner reads `<think>…</think>` as reasoning and `<tool_call>`
as a call.

- **Code is full of that markup.** A completion over HTML, XML or a prompt
  about reasoning models loses whatever the scanner takes to be reasoning.
  The client gets back less text than the model generated, silently. A
  request with no template is now `raw` from render to finish
  (`RenderedPrompt::raw`, `ParsedRequest::raw`), and its sequence scans with
  `ChatScanner::raw`, which passes every byte through as text.

**Fill-in-the-middle is a property of the vocabulary, not of the
architecture.** The DeepSeek Llama-layout checkpoints do not all agree.

| Checkpoints | FIM tokens | Order |
|---|---|---|
| deepseek-coder 1.3b/6.7b | `<｜fim▁begin｜>` / `<｜fim▁hole｜>` / `<｜fim▁end｜>` | prefix-suffix-middle |
| R1-Distill-Qwen (Qwen vocabulary) | `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>` | prefix-suffix-middle |
| deepseek-coder-7b-v1.5, llm, math | none | - |

`data::fim::FimFormat::of` picks the format from the tokenizer's added
tokens. A vocabulary with no complete set refuses `suffix` with the fixed
`qwen3::chat::NO_FIM_TOKENS`. The API maps that to a 400 without passing the
resident's own error text through. A model whose `generate` does not
declare `suffix` at all is refused before dispatch
(`catalog::resolve_text`).

The vendored OpenAI schema describes `text_completion` as one shape "for
both the streamed and non-streamed response", with a non-null
`finish_reason`. The streamed frames before the last have no reason yet.
OpenAI and vLLM both send `null` there, and so does brain. The conformance
tests validate the frames that carry a reason against the schema and check
the others' shape directly.
