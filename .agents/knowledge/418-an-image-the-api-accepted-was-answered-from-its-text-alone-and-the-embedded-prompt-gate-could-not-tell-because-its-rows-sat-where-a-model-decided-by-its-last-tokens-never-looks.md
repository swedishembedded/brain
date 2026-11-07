<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 418. An image the API accepted was answered from its text alone, and the embedded-prompt gate could not tell, because its rows sat where a model decided by its last tokens never looks

**Lesson:** An input the API accepts but the model never reads gives an answer that looks informed,
so ask the model's manifest before queueing it. A gate that checks that the input decides the output
must put the input where the output is decided.

## What happened

Question (2026-11-01): a deployed brain answered "I can't see any image" to a chat request with a
PNG `image_url` part, with `prompt_tokens` 76 (text only). Where was the picture lost, and why did
no gate say so?

## Why

- The API decoded every `image_url` part to an image blob and attached it to the invocation
  unconditionally ("harmless for a model that declares no image input"). A model that declares none
  never reads it, and the reply read like a model that had looked. The one place that knew the
  answer, the model's manifest (`generate` has no `image` input), was never asked. Now
  `bridge::refuse_unread_images` asks it before the call is queued, on the one-shot, streamed and
  fetch-on-demand paths of both chat surfaces: 400 naming the model.
- The conformance property `embedded_prompts_batch_and_never_share_a_prefix` puts embedding rows at
  prompt positions 4..16 and requires that rows change the greedy output. On the conditioned tiny
  FP8 fixture of the Qwen3.8 engine the output (6 tokens) was the same for rows A, rows B and no
  rows, even when every position but the last carried rows: the model's greedy output there is
  decided by the last prompt token.

  The property passes on a decoder that ignores its rows entirely whenever its fixture has that
  shape, and fails (as it should) the moment rows reach the last positions. It now puts a second run
  of rows ending at the last position. A gate that checks "the input decides the output" must put
  the input where the output is decided.
- An image of `h x w` merged rows spans `max(h, w)` rotary positions, not `h * w`, so a KV row is
  not its rotary position after an image; the engine holds one `PromptRope` per slot and every table
  (prefill round, decode, verify, mixed) asks `rope_at(row)`. A uniform shift of all positions
  leaves a model unchanged (rotary embedding is relative), which the engine test uses for the
  plumbing; it cannot see a wrong decode-side shift on that fixture (the output ignores context), so
  the decode side is gated only by the real-weight parity on the released checkpoint
  (`tests/hf_fp8_vision_real.rs`, the continuation's logits).
- Two cargo worktrees that share one `CARGO_TARGET_DIR` collide: after a build of the other tree's
  `brain-capability` the first tree failed with "unresolved import" of items its own source has
  until a file was touched. One target dir per worktree.

## Evidence

Real weights against `transformers` (bf16 decoded, tower in float32), one 288 x 256 picture: tower
rel-L2 8e-6; prompt ids and the three-axis rotary positions exact; logits at the end of the prompt
cosine 0.9991 / rel-L2 0.044 (weights decoded to bf16) and 0.9917 / 0.130 (FP8 kernels; text on the
same weights: mean 0.034, worst 0.10).
