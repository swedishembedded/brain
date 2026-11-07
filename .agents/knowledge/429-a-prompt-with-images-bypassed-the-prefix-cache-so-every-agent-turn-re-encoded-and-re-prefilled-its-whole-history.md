<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 429. A prompt with images bypassed the prefix cache, so every agent turn re-encoded and re-prefilled its whole history

**Lesson:** A prefix cache for a model with non-token inputs must be keyed by the content of those
inputs as well as the tokens, with the digest taken over the prefix actually reused. Otherwise such
prompts bypass the cache and every turn re-encodes and re-prefills its whole history.

## What happened

Question (2026-11-02): serving Qwen3.8-27B-FP8 with vision to an agent harness, requests holding an
`image_url` reported `cached_tokens` 0 (28k to 117k input tokens each, a single unchanging picture
early in the history) while text-only turns of the same agent cached 95 % or more. Why?

## Why

- `Qwen35GgufEngine::begin_prefill` made a prompt with embedding rows non-cacheable
  (`embeds.is_none()` in `cacheable`): the cache was keyed by token ids only and an image's
  placeholder tokens say nothing about its rows. Such a prompt was neither looked up nor filed, and
  the scheduler charged it in full at admission. The vision front end also ran the tower for every
  picture of every request.

## What to do

- The key now covers the rows: `PromptEmbeds::content_key()` gives, for every prefix length, a
  digest of the rows and non-default rotary positions inside that prefix (one event per row and per
  rotary break, never per token). `SnapshotPrefixCache` entries carry the key of the prompt they
  were filed from and match a prompt only when the tokens and the digest at the entry's length
  agree. The text before an image is shared whatever the image; an entry cut inside an image matches
  only the same image; the same placeholders as plain text match nothing.
- The scheduler asks the decoder for adoptable blocks and for the entry to spare with the prompt's
  rows (`cached_prefix_blocks_embedded`, `reclaim_prefix_keeping_embedded`; a decoder that keeps
  rows out of its cache keeps the old default of charging in full).
- The vision front end keeps the rows of its 8 most recent pictures, compared byte for byte (never
  by a hash of client bytes), so a resent picture skips the tower and yields rows equal to those the
  cache was filed under.

### Rules

- A prefix cache of a model that takes non-token inputs is keyed by their content as well as the
  tokens, with the content digest taken over the prefix actually reused.
- Compare client-supplied bytes for equality when a cache hit serves another request's result; a
  short hash is a cross-request attack surface.
- Entries of a prompt with embedding rows are never valid for the MTP head (its rows would be those
  of the placeholders).

## Evidence

- Test (tiny FP8 checkpoint, host cores): turn 2 of a conversation with the same picture reports 128
  cached rows (the turn-1 checkpoint, past the picture) and its hidden state equals the cold
  prefill's (worst relative difference 0); another picture at the same place and the placeholders as
  text report 0.
