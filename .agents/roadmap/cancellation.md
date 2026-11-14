<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# cancellation - roadmap

Scope: Cooperative cancellation of every served action (`CancelToken`, `ActionSpec::cancellable`).

Goal: EVERY served action stops when its caller's `CancelToken` fires, the
manifest says so (`ActionSpec::cancellable`), and a behavioural gate proves
it. The token is cooperative (`crates/capability/src/lib.rs`, `CancelToken`):
an action checks it on entry and between bounded units of work and returns
`Err("cancelled")`, or, once it has streamed text, completes with the partial
text and `finish_reason: "cancelled"` (`.agents/rules/serving-contract.md`).

## Milestones

- [x] **C0. The ratchet is green again** (2026-10-06): deepseekvl `generate`
      and januspro `text2image` poll; lfm2 left the list.
- [x] **C1. The bit.** `ActionSpec::cancellable` (default false, builder
      `.cancellable()`), emitted by `to_json` (D-Bus `Manifests()`, `brain caps
      --json`) and `GET /v1/capabilities`, as `false` too. Contract: the token
      is checked on entry, before any weight is read or lock taken, and at
      least once per bounded unit of work; a one-shot action meets it with the
      entry check plus size-bounded inputs.
- [x] **C2. The behavioural gate.** The harness is
      `capability_mock::cancellation::{admission, mid_run, certify, Bound}`.
      `crates/catalog/tests/it/cancellation.rs` runs from one place over
      `catalog::manifests()`: every listed action declares the bit, and every
      action whose provider builds from an assembly of absent paths refuses a
      token fired before the call with the literal and no progress. The test
      prints which providers need real weights to be built; those are covered
      by their own crates (table below). Mid-run proofs stay beside each
      model's loop.
- [x] **C3. Work the list**, declaring the bit as each action lands (tables
      below).
- [x] **C4. Flip it.** `scripts/gates/check-cancellable-actions.sh`
      (`make check/scripts` and a pre-commit hook; spec
      `tests/gates/cancellable_actions.bats`) fails any production
      `ActionSpec::new(..)` without `.cancellable()`; the KNOWN list is gone
      and there is no allow list.
- [ ] **C5. Close the entry-only loops** listed under "Open" below: actions
      that declare the bit and check on entry, but whose long loop lives in a
      library function with no stop hook yet.

## One terminal rule (2026-11-14)

`qwen3::chat::SeqState::finish` answers `Err("cancelled")` for a sequence
cancelled before it streamed any text (also during prefill), and `Ok` with
the partial text and `finish_reason: "cancelled"` after. It used to answer
`Ok` with an empty reply, which the harness's `admission` rejects and which
an HTTP caller read as a 200. Every chat model, `llmserve` (which also
refuses a fired token at admission), the SDK's `TextGenerationPipeline` and
`ChatPipeline` (now `Error::Cancelled` before the first token) and the omni
Thinker share it; HTTP answers 503 for a draining server and D-Bus the error
frame, as they already did for every other action.

## Done

Every row: entry check before any weight is read or resident lock taken, a
poll at the stated granularity, the canonical `"cancelled"` literal, RAII
release on the early return, and the bit declared.

| model | action(s) | polled | verified |
|---|---|---|---|
| shared adapters | catalog `LazyProvider`, `ResidentProvider` actions; `Instance::run_batch` default | before the build/load and lock; per queued job | `residency` unit test; the catalog C2 test runs through them |
| llava, fastvlm | caption | entry, after the vision tower, per decoded token | `admission` weight-free; `mid_run` with the real checkpoint (fastvlm passed; llava skips by name: no `liuhaotian/llava-v1.5-13b`) |
| cosyvoice | synth (provider, resident, SDK) | entry, per speech token, per flow Euler step | `admission` weight-free; `mid_run` over the LM decode on a tiny random config |
| moondream3 | caption | entry, per decoded token, live progress | `admission` weight-free; tiny-model stop test |
| glmdsa, gpt2 | generate, train, finetune | entry, per token, per optimizer step | `certify` with tiny trained checkpoints |
| florence2 | ground | entry, per decoded token | `admission` weight-free |
| qwen3, ouro, decisions | train, finetune, lora_train | entry, per optimizer step | tiny train tests assert the literal |
| qwen3tts | batch | entry, per round, per vocoded clip | entry weight-free; `run_batch` mid-run |
| deepseek2ocr, deepseekocr2 | generate | entry; `llmserve` per token | `admission` through `ocrserve` with a toy model |
| s3dit | text2image, image2image, inpaint, outpaint | entry, per denoise step | `admission` weight-free |
| flux1, pulid | text2image | entry, per denoise step; one progress step per denoise step through `Session::run` and the residents | `admission` weight-free; mid-run skips by name (FLUX.1 and the PuLID towers) |
| sdxlunet, controlnet | text2image | entry, per denoise step; one progress step per step through `Session::run` and the residents | `certify` and a per-step report test over the real `sampler::sample` with a fake `Denoiser` |
| imgpipe | run | entry, per stage; the token rides into every model stage | `certify` with stub stages |
| kronos | forecast (CPU, NPU), predict | entry, per decode step in every rollout form | `certify` over both injected-cores rollouts on a zero model; fired-token test for the host forms |
| every forecaster | predict (`forecastops::run_predict`), forecast entry | entry, per origin | `certify` with a persistence forecaster |
| nemotronasr | transcribe (provider, CPU and NPU residents, batched) | entry, per encoded frame of the RNN-T decode; per stream in a batch | `certify` on a tiny random config; `transcribe_stream` is one window, entry only |
| qwen3asr | transcribe (provider, CPU and NPU residents) | entry (before the model lock), per decoded token, each reported | real-weight `certify` `a_transcription_stops_at_the_token_its_cancel_fires` (ignored by default like its siblings; skips by name without `Qwen/Qwen3-ASR-1.7B` or `asr/audio/librispeech_mr_quilter.wav`) |
| rrdbnet | upscale | entry, per window of a tiled sweep | `certify` on synthetic weights |
| minimaxmusic3 | generate | entry, per AR frame, per Euler step, per vocoded chunk | `admission` weight-free; `certify` over the Euler loop on a tiny DiT; AR and vocoder by reading |
| qwen3omnimoe | generate (bf16 and int8), speak, converse, int8 raw generate | entry (the checkpoint now loads on first run, not on lookup), per token (`SeqState`), Talker and Code2Wav as before | `certify` over `serve_chat` with a canned Thinker |
| one-shots | lfm2 embed/fill_mask, sam2 segment, yolov8 detect, scrfd, arcface, clip, zipdepth infer, codeformer, t5encoder, vqgan, worldmirror2, splat render, imageops, demo, the mocks | entry | the catalog C2 test where the provider builds without weights |
| already polling | the `SeqState` models, qwen3vl lora_train, flux2, wan, ltxv, qwen3tts synth/clone/design, supir, splat fit, deepseekvl, januspro, sam2 track, recon | entry check added where it was missing | their existing tests; C2 for admission |

## Open

- **Entry check only, loop without a stop hook** (declared, but a fired token
  is seen only on entry): lfm2 `eval` (`train::evaluate`), diamond `bench` and
  `replay --verify` (`worldrun`), horizon `eval` and `calibrate`, toypid
  `rollout`/`profile`, toymoe `eval`/`generate`, zipdepth `calib`
  (activation statistics; image loading polls), splat `merge`/`prune`, the
  `import`/`export`/`convert` actions (qwen3, qwen35, qwen3tts, glmdsa,
  qwen3omnimoe, the forecasters, `resident_provider::import_spec`), qwen3
  eval actions, forecast `compare`, trainjob `dryrun`/`consolidate`. Each
  needs its library loop to take a keep-going callback, as florence2 and
  rrdbnet did.
- yolov8 and zipdepth `train` stop through their step callback but answer
  `Ok` with the partial run, not `"cancelled"`.
- Real-weight mid-run proofs to run on a box with the checkpoints: llava
  caption, the SDK's `ask_stream`, flux1 and pulid, s3dit's edit actions,
  cosyvoice's flow loop, qwen3asr (the test above, with the LibriSpeech clip).
- The C2 test proves admission only for providers that build without
  weights; residents are covered by the shared `run_batch` check and their
  own entry checks, not by a weight-free behavioural test.
