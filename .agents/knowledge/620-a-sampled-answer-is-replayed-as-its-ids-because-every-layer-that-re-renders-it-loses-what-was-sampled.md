<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 620. A sampled answer is replayed as its ids, because every layer that re-renders it loses what was sampled

**Lesson:** there was no canonical record of what a model sampled. The server parsed the generated ids into an
OpenAI message (lossy), the agent runtime re-serialised that message (lossy), and the next request re-rendered it
through the chat template (lossy, and dependent on later messages: Qwen3's drops the empty think block of every
assistant turn but the last). So no session was append-only in ids, and a session could never train as one record
(#619). The fix is to stop re-rendering: every chat reply carries the answer's ids as an opaque blob (`sampled`), the
client echoes it on the assistant message, and the next prompt splices those ids in where the turn goes. The
template writes only what was never sampled. No option: the data decides.

## What happened

178 recorded vim agent sessions (59 Qwen3-4B, 59 Qwen3-8B, 60 Qwen3.8-27B-FP8) rendered as served: 0/59, 0/59 and
58/60 append-only. Simulating a runtime that sends arguments as generated and a template that keeps the empty think
block gave 50/59, 41/59, 60/60; the rest were the model's own invalid or looping calls, which a runtime repairs or
drops, so a re-rendered history can never be exact. Replaying the sampled ids is exact by construction, whatever the
runtime or the template does with the parsed message.

## How it works (`data::replay`, `qwen3::chat::parse_request_as`, `llmserve`, `apiserve`)

- A chat reply's assistant message (and a stream's terminal chunk, `choices[0].delta.sampled`) carries `sampled`:
  `token_ids` (generated, the stop token included when one ended it), `prefill_token_ids` (the generation prompt:
  the assistant header and any prefilled think block), `finish`, `tokenizer_digest`, `context_len`, `context_hash`
  (`sha256` of the prompt ids). Ids are unsigned LEB128 varints, base64-encoded.
- The prompt is built by segments: the template renders the conversation with a sentinel in place of each sampled
  answer; the text from where that turn begins to the first end-of-generation marker after the sentinel is replaced
  by `prefill_token_ids ++ token_ids`, and by the template's close of the turn as context when the generation did not
  end on its marker (length, cancel, stop string). Same code for the Qwen port and any checkpoint's own Jinja template
  (Inkling 2's typed messages, ricursive, ouro all parse through `parse_request_as`).
- A turn is rendered by the template when its blob is missing, malformed (an id outside the vocabulary, a turn marker
  inside the answer), from another tokenizer, names a generation prompt the template cannot write there, or its
  context hash is not the prompt built so far; from the first such context mismatch on, every later turn too. Every
  reply reports `brain.history: {append_only, diverged_turn, reason}`.
- The prefill is checked against the generation prompts the template writes (reasoning on or off), so a client
  cannot smuggle a turn opener in through it; the answer may hold no turn marker except its final end of turn.
- Qwen3.8's `preserve_thinking` switch is gone: a replayed turn never reaches the template's framing, and an
  unreplayed one is rendered with the template's default.

## Evidence (2026-11-28)

- Real tokenizers and templates (`qwen3` `it::replay_sessions`): Qwen3-0.6B, 4B, 8B, Qwen3.8-27B and Inkling-2-Mini,
  thinking on and off, answers with 0, 1 and 2 calls, text around calls, a malformed call and a length cut: every
  prompt `k+1` starts with prompt `k` ++ `token_ids(k)`, `append_only` throughout. Missing, foreign-tokenizer,
  malformed and context-tampered blobs are reported by turn and reason. No Qwen3.5 checkpoint is in the model store.
- Over HTTP (`catalog` `it::replayed_sessions`, tiny Qwen3 and Qwen3.8 engines on CPU): the reply message and the
  stream's terminal chunk carry `sampled`; an echoed blob gives `append_only`; a missing one `reason: missing`, a
  tampered one `reason: context`.
- Prefix cache: the second call of a two-call session (prompt 67 ids, answer 6) was served 64 of 111 prompt ids
  from cache on the Qwen3 engine: the first prompt's full blocks. The engines index prompt blocks only, so the answer's
  ids and the turns after it are prefilled again (open: index the generated blocks at release).
- Cost on a 60-call session (Qwen3-4B tokenizer and template, 16,190 generated ids, 41 KB of assistant text, tool
  outputs 1-6 KB; `qwen3` `it::replay_sessions::the_cost_of_sampled_blobs_on_a_long_session`, `--ignored`):
  token ids cost 4.11 B/id as JSON integers, 5.33 as base64 of little-endian u32, 2.27 as varint+base64 (the
  standard; 0.89x the assistant text). A blob is at most 903 B. Requests over the session: 7.96 MB without blobs,
  9.56 MB with (+20%; the worst call +54 KB). Responses: 45.5 KB without, 100 KB with. Building the final request
  (91k ids) by replay takes 48 ms on the CPU (fast profile) against 39 ms to render and encode it whole.
