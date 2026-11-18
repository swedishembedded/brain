<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 510. Every Qwen3.8 image request was refused because the client-text escape ran after the image placeholders were written

**Lesson:** Escaping client text is a boundary step, and markup the server writes itself must be added after it, never
before. A front end that writes its own tokens into the client's messages and then hands them to a parser that
escapes the client's text loses those tokens. A gate that builds the prompt with the shared parser, a real
vocabulary and an image catches it on the CPU in a second; the only gate that built a real image prompt needed the
27B on a GPU.

## What was reported

An agent runtime driving `Qwen/Qwen3.8-27B-FP8` over the OpenAI chat API (`brain serve --openai`) sent screenshots
inside tool results: a `tool` message whose `content` holds `text` and `image_url` parts. Three of 25 benchmark tasks
failed on HTTP 422 "the prompt holds image runs of [] tokens, the images need [N]": the image was decoded and counted,
but the prompt held no placeholder for it.

## Why

- The API surfaces were right: the OpenAI `tool` message and the Anthropic `tool_result` block both reach the model
  as a `tool` turn of typed parts with a payload-free `image_url` marker, and the checkpoint's template places an
  image in a tool response exactly as in a user turn (`render_content` serves every role; only a system message
  raises).
- The Qwen3.8 image front end (`qwen35::vision_frontend`) replaced each image part with its run of
  `<|vision_start|><|image_pad|>...<|vision_end|>` text and handed the messages to `qwen3::chat::parse_request`,
  which since S21 (`.agents/roadmap/service-security.md`) escapes every string a client wrote so it cannot spell a
  control token. The front end's own runs were escaped with the client's text and encoded as plain text: zero image
  tokens, in every role. Tool results were simply where the benchmark sent its images; an image in a user turn was
  refused the same way.
- No gate saw it: the escape landed after the image path, the front end's unit tests stop before the parser, and the
  one test that built a real image prompt through the server's front end (`tests/hf_fp8_vision_real.rs`) needs the
  27B on a GPU.

## Fix

The front end escapes the client's text first (`Invocation::escape_client_text`), then writes the placeholder runs,
then parses with `qwen3::chat::parse_escaped_request` (the same parse without the escape). Placeholder text a client
types is still refused, and an image in a system message is refused naming `messages[i].content[j]`. DeepSeek-VL,
whose conversation format has no tool turn, refuses a `tool` message as an invalid request naming `messages[i]`.

The other vision front ends escape before they place images (qwen3vl, minicpmv, qwen3omnimoe, deepseekvl: their
templates or tag writers run on the escaped text), so the ordering defect was the Qwen3.8 path's alone.

## Evidence

`crates/qwen35/tests/it/vision_request.rs` (tiny tower, byte vocabulary with the Qwen3.8 specials, CPU): before the
fix four of six tests failed with the reported message (`runs of [] tokens, the images need [8]` for the tool
result, `[4]` for a user turn); after it the tool result renders
`<tool_response>\nScreenshot of /login:<|vision_start|>` + eight `<|image_pad|>` + `<|vision_end|>\n</tool_response>`,
as the checkpoint's template does. The API side is
`an_image_in_a_tool_result_keeps_its_place_in_the_tool_turn_on_both_dialects` (`crates/apiserve/tests/it/api.rs`).
