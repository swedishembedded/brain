<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# cancellation - roadmap

Scope: Cooperative cancellation of every served action (`CancelToken`, `ActionSpec::cancellable`).

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
another does not, and a `cancel` read in a crate's test code counts
as polling. It is a fast pre-check, not the gate this file
works toward. Its KNOWN list is down to minimaxmusic3.

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
      - [x] The harness: `capability_mock::cancellation` (`admission`,
            `mid_run`, `certify`, `Bound`). `admission` fires the token before
            the call and requires `Err(capability::failure::CANCELLED)` with no
            progress; `mid_run` fires it from the `fire_after`-th progress
            event and requires at most `within` more events, ending in the
            cancelled error or (after streamed text) `finish_reason:
            "cancelled"`, and refuses a workload too short to show a stop. An
            action reports with `CancelToken::check`.
      - [x] Applied to every action reworked in the list below (the
            "verified" column says how).
      - [ ] Applied to the "already polling" set below, and run over every
            served action from one place rather than per crate.
- [ ] **C3. Work the list below**, smallest first, declaring the bit as each
      action lands.
- [ ] **C4. Flip it.** A served action without `cancellable = true` fails
      `make check`; the KNOWN list in the script is deleted.

## Done (2026-11-14)

Every row: entry check before any weight is read or resident lock taken,
a poll at the stated granularity, the canonical `"cancelled"` literal (a
stopped training run used to answer `"cancelled after step N"`, which
`ReplyError::of` classed as a model failure; the step now travels as the last
progress event), and RAII release of guards and scratch on the early return.

| model | action(s) | polled | verified |
|---|---|---|---|
| llava, fastvlm | caption | entry, after the vision tower, per decoded token | `admission` weight-free; `mid_run` with the real checkpoint (fastvlm `apple/FastVLM-0.5B` passed; llava skips by name: no `liuhaotian/llava-v1.5-13b` in the store) |
| cosyvoice | synth (provider, resident, SDK) | entry, per speech token, per flow Euler step; per-step progress now emitted | `admission` weight-free; `mid_run` over the LM decode loop on a tiny random config; flow loop by reading (no tiny flow config) |
| moondream3 | caption | entry (before the resident lock), per decoded token; progress is now live, no longer replayed after generation | `admission` weight-free; tiny-model test that `generate_kv` stops at the token its caller refuses |
| glmdsa | generate (provider and `resident_llm.rs` `GlmInstance`), train, finetune | entry, per decoded token, per optimizer step | `certify` on generate and train with a tiny trained checkpoint |
| gpt2 | generate (provider and `resident_llm.rs` `GptInstance`), train | entry, per decoded token, per optimizer step | `certify` on generate with a tiny trained checkpoint; both sample loops stop at a refused token (tiny model); the resident by compile and reading |
| florence2 | ground | entry (before the import), per decoded token | `admission` weight-free |
| qwen3 | lora_train, train, finetune | entry, per optimizer step | the existing tiny train test now asserts the literal |
| ouro (`resident_ouro.rs`), decisions (`brain::decision_actions`) | train | entry (ouro), per optimizer step | `decision_lifecycle` asserts the literal; ouro by reading |
| qwen3tts | batch | entry, per round (already), per vocoded clip | entry check weight-free; mid-run was already tested (`run_batch`) |
| deepseek2ocr, deepseekocr2 | generate | entry (before the resident lock and the build); the decode is `llmserve`'s scheduler, which stops a request at its next token | `admission` weight-free through `ocrserve` with a toy model; the rows' old `on_token` notes were stale, these models are served through `ocrserve` |
| s3dit | image2image, inpaint, outpaint (and lora_train's entry) | entry, per denoise step, through `generate_core` shared with text2image | `admission` weight-free for all five actions; mid-run needs the real Qwen text encoder and DiT |
| flux1, pulid | text2image | entry (before the pipeline cache lock), per denoise step | `admission` weight-free; mid-run skips by name (needs FLUX.1, PuLID, ArcFace, EVA-CLIP and BiSeNet checkpoints) |
| sdxlunet, controlnet | text2image | entry, per denoise step in the shared `sampler::sample` | `admission` weight-free for both; `certify` over the real `sample` loop with a fake `Denoiser` |
| SDK | `VisionLanguagePipeline::ask_stream`, `TextGenerationPipeline::generate_stream`, `VideoPipeline::generate_with_progress`, `TtsPipeline::{speak_with, clone_voice_with, design_with}` | as the model underneath | text: fired-before and mid-stream on a synthetic checkpoint; video and TTS: fired-before on the resolve fixtures; VLM skips by name without `Qwen/Qwen3-VL-4B-Instruct` |

## Work list: actions that still ignore the token

Effort: S = the token is already in reach; M = thread a token or a stop
callback through one or two library functions; L = several loops or a batch
engine.

| effort | model | action(s) | streaming | where the loop is | why still open |
|---|---|---|---|---|---|
| S | imgpipe | run | no | `lib.rs` `Pipeline::run`, per stage | few stages, low value |
| M | kronos (forecast) | forecast | no | `generate.rs` `forecast*` `for step in 0..pred_len` | several entry points; not reached |
| M | nemotronasr | transcribe | yes | model `transcribe`/`transcribe_with_encoder` chunk loop | not reached; `transcribe_stream` is one bounded window, N/A |
| M | qwen3asr | transcribe | yes | `model.rs` `transcribe` decode loop under `caps.rs` `decode` | fixed window; not reached |
| M | rrdbnet | upscale | no | `caps.rs` `upscale_tiled` window loop behind the `Upscaler` trait | trait signature changes; not reached |
| M | flux1, pulid, sdxlunet, controlnet | text2image | no | `Session::run`, `resident_flux1`/`resident_sdxl`/`resident_controlnet` | they now stop, but emit no per-step progress, so a client cannot see a step boundary; adding it changes `Session::run` and three residents |
| M/L | minimaxmusic3 | generate | yes | pipeline: LLM decode, DiT denoise, vocoder | multi-minute; three phases; not reached |
| L | qwen3omnimoe | generate, speak, converse | yes | `caps.rs` `generate_greedy`, `generate_multimodal`, `speak` (thinker, talker/MTP, code2wav), `converse`; `int8_thinker_resident.rs` `generate_with_embeds` | several decode loops plus the multi-device int8 path; `generate` emits only start and end progress |
| S (entry check only) | one-shot | lfm2 embed/fill_mask, sam2 segment, yolov8 detect, scrfd detect, arcface embed, clip embed_text/embed_image, zipdepth infer, codeformer restore_face, t5encoder encode, vqgan encode/decode, worldmirror2 reconstruct, horizon predict, qwen3 embed, chronos2/fincast/timesfm3 forecast, splat render, imageops, demo echo | no | single forward | C1's entry check plus size-bounded inputs |

Real-weight mid-run proofs still to run on a box with the checkpoints: llava
caption (LLaVA-1.5-13B), the SDK's `ask_stream` (Qwen3-VL-4B-Instruct), flux1
and pulid text2image, s3dit's edit actions, cosyvoice's flow loop end to end.

Already polling (for C2 to prove, not to fix): the `qwen3::chat::SeqState`
users (qwen3, qwen35, qwen35moe, qwen3vl `generate`, and their residents),
qwen3 `lora_gate`, qwen3vl `lora_train`, s3dit `text2image` and `lora_train`,
flux2 (all), wan (all), ltxv `t2v`/`dfr`, qwen3tts `synth`/`clone`/`design`/
`speak`, supir `restore`, splat `fit`, deepseekvl `generate`, januspro
`generate`/`text2image`, the mock. Note for C2: a `SeqState` user cancelled
before it starts answers `Ok` with `finish_reason: "cancelled"` and no text,
where the contract says `Err("cancelled")`; `admission` rejects that, so the
SDK's text, chat and VLM forms check on entry themselves.

Still with an SDK entry point that passes `CancelToken::default()` and has no
cancellable form: none of the surfaces above; `pipeline.rs` and `finetune.rs`
pass it only from their convenience forms.
