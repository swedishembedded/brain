<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# cancellation - roadmap

Goal: EVERY served action stops when its caller's `CancelToken` fires, the
manifest says so (`ActionSpec::cancellable`), and a behavioural gate proves
it. Today the token is cooperative (`crates/capability/src/lib.rs`,
`CancelToken`): an action checks it between bounded units of work and returns
`Err("cancelled")`, or, once it has streamed text, completes with the partial
text and `finish_reason: "cancelled"` (`.agents/rules/serving-contract.md`).
An action that never reads the token runs to completion whatever the caller
does, and nothing in the manifest tells the caller which kind it got.

`scripts/gates/check-cancellable-actions.sh` (`make check/scripts`) is the
interim ratchet: a per-crate text heuristic over `caps.rs` files that declare
a `.streaming()` action. It cannot see a non-streaming action, a resident path
that bypasses the provider's polling, or a crate where one action polls and
another does not (s3dit), and a `cancel` read in a crate's test code counts
as polling. It is a fast pre-check, not the gate this file
works toward.

## Milestones

- [x] **C0. The ratchet is green again.** It was red at HEAD 2026-10-06:
      deepseekvl `generate` and januspro `text2image` ignored the token and
      were not listed; lfm2 was listed only because a comment in its
      `caps.rs` mentions `.streaming()` (both its actions are one-shot
      encoders). deepseekvl's batched decode now stops a request's sequence
      at the next token once its token fires, and januspro's `text2image`
      polls per image token (it passed `&|| false`); both check on entry.
      The gate skips comment lines, so lfm2 is off the list.
- [ ] **C1. The bit.** `ActionSpec::cancellable: bool` (default false),
      emitted by `to_json`, the D-Bus manifest and `/v1/capabilities`, with
      the contract "checks the token on entry and at least once per bounded
      unit of work". A one-shot action meets it with the entry check plus
      size-bounded inputs.
- [ ] **C2. The behavioural gate.** For every served action: run it under a
      weight-free harness with the token fired before the call (one-shot) or
      after the first progress event (streaming) and assert it returns within
      a bound with the contract's error or `finish_reason`. The text heuristic
      stays only as a pre-check.
- [ ] **C3. Work the list below**, smallest first, declaring the bit as each
      action lands.
- [ ] **C4. Flip it.** A served action without `cancellable = true` fails
      `make check`; the KNOWN list in the script is deleted.

## Work list: actions that ignore the token (2026-10-06)

Checked against the tree, provider path (`crates/<model>/src/caps.rs`) and
served path (`crates/catalog/src/resident_*.rs`, `crates/serving/src/
executor.rs`) both. Effort: S = the token is already in reach, add a check
in an existing loop or replace a `CancelToken::default()`; M = thread a
token or a stop callback through one or two library functions; L = several
loops or a batch engine.

| effort | model | action(s) | streaming | where the loop is | note |
|---|---|---|---|---|---|
| S | deepseek2ocr | generate | yes | `caps.rs` `Session::generate`'s `on_token` closure (returns `bool`) | return `false` once `inv.cancel` fires |
| S | llava | caption | yes | `caps.rs` `for i in 0..max_new` inside `Action::run` | served as a stateless provider |
| S | fastvlm | caption | yes | `caps.rs` `for i in 0..max_new` inside `Action::run` | same shape as llava |
| S | qwen3tts | batch | yes | `batch.rs` `synth_batch` passes `CancelToken::default()` per session | `run_batch` already polls per session |
| S | s3dit | image2image, inpaint, outpaint | yes | `pipeline.rs` `generate_img` step loop | text2image polls; the resident hands the edit actions to the provider, which does not; mirror `generate`'s `cancel` param |
| S | imgpipe | run | no | `lib.rs` `Pipeline::run`, per stage | few stages, low value |
| M | qwen3 | lora_train | yes | `caps.rs` calls `finetune::finetune`, which has no callback | `finetune_lora_controlled(.., FitControl)` already supports stopping (the SDK uses it); the module doc saying no callback exists is stale |
| M | deepseekocr2 | generate | yes | `deepseek2::model::generate_greedy_cb`, callback returns `()` | it cannot stop at all: EOS only stops emitting, the model decodes all of `max_new`; make the callback `FnMut(u32) -> bool` |
| M | moondream3 | caption | yes | `model.rs` `generate_kv`, no callback | progress is replayed after generation, so the streaming is not real either |
| M | glmdsa | generate | yes | `sample.rs` `generate_kv`, no callback | provider and `resident_llm.rs` `GlmInstance` |
| M | gpt2 (resident only) | generate | - | `sample.rs` `generate`, no callback | `resident_llm.rs` `GptInstance` |
| M | florence2 | ground | no | `text/lm.rs` `Florence2Lm::generate` | short decode |
| M | flux1, pulid | text2image | no | `flux1::pipeline` step loop (pulid via `generate_injected`) | one fix covers both; `Session::run` takes no progress callback either |
| M | sdxlunet, controlnet | text2image | no | `sdxlunet::sampler::sample` step loop | one fix covers both; no progress either |
| M | cosyvoice | synth | yes | `pipeline.rs` `generate`: LLM decode then flow timesteps | two loops |
| M | kronos (forecast) | forecast | no | `generate.rs` `forecast*` `for step in 0..pred_len` | several entry points |
| M | nemotronasr | transcribe | yes | model `transcribe`/`transcribe_with_encoder` chunk loop | `transcribe_stream` is one bounded window, N/A |
| M | qwen3asr | transcribe | yes | `transcribe`/`transcribe_with_head` decode loop | fixed window |
| M | rrdbnet | upscale | no | `caps.rs` `upscale_with_halo` tile loop behind the `Upscaler` trait | trait signature changes |
| M/L | minimaxmusic3 | generate | yes | pipeline: LLM decode, DiT denoise, vocoder | multi-minute; three phases |
| L | qwen3omnimoe | generate, speak, converse | yes | `caps.rs` `generate_greedy`, `generate_multimodal`, `speak` (thinker, talker/MTP, code2wav), `converse`; `int8_thinker_resident.rs` `generate_with_embeds` | several decode loops plus the multi-device int8 path; `generate` emits only start and end progress |
| S (entry check only) | one-shot | lfm2 embed/fill_mask, sam2 segment, yolov8 detect, scrfd detect, arcface embed, clip embed_text/embed_image, zipdepth infer, codeformer restore_face, t5encoder encode, vqgan encode/decode, worldmirror2 reconstruct, horizon predict, qwen3 embed, chronos2/fincast/timesfm3 forecast, splat render, imageops, demo echo | no | single forward | C1's entry check plus size-bounded inputs |

Already polling (for C2 to prove, not to fix): the `qwen3::chat::SeqState`
users (qwen3, qwen35, qwen35moe, qwen3vl `generate`, and their residents),
qwen3 `lora_gate`, qwen3vl `lora_train`, s3dit `text2image` and `lora_train`,
flux2 (all), wan (all), ltxv `t2v`/`dfr`, qwen3tts `synth`/`clone`/`design`/
`speak`, supir `restore`, splat `fit`, deepseekvl `generate`, januspro
`generate`/`text2image`, the mock.

Not actions, but the same defect one layer up: the SDK entry points in
`crates/sdk` (`vlm.rs` overwrites `inv.cancel` with `CancelToken::default()`;
`tts.rs`, `text.rs`, `video.rs`, `pipeline.rs` pass `default()`) give their
callers no way to cancel even where the action polls.
