# tts — workstream ledger

Qwen3-TTS stack (Talker + MTP + Mimi-style codec + ECAPA speaker encoder),
pure-Rust voice synthesis. The architecture and CLI live in `readme.md`; the
optimization write-up (KV-cache, INT8, streaming codec, validated numbers) in
`acceleration.md`. This file is the workstream ledger — status facts only,
verified against code.

## Done

- **Import** — HF → brain `.safetensors`, 1:1 name remap, bf16→f32, no
  transpose. Talker + MTP (`tts/src/import.rs`), codec (`codec/src/import.rs`;
  271 decoder tensors → 253 after dropping 2 input_proj and collapsing 32
  codebook stats to 16 tables), speaker (`speaker/src/import.rs`, 76 tensors).
  CLI `brain tts import` writes `talker/mtp/codec/speaker.safetensors`.
- **Components** — Talker (`tts/src/talker.rs`, 28-layer Qwen3 dense, GQA 16/8,
  SwiGLU, RMSNorm, **M-RoPE `mrope_section=[24,20,20]`**); MTP (`tts/src/mtp.rs`,
  5-layer block filling codebooks 1..15); codec (`codec/src/{model,decode_stream,
  streaming,recon}.rs`, Mimi 12 Hz, encode + decode); speaker
  (`speaker/src/model.rs`, ECAPA-TDNN, 1024-d embed); audio (`crates/audio`,
  WAV, 1-D conv builders, linear resample, STFT/mel).
- **Synth / clone / design** (`tts/src/pipeline.rs`) — speaker-free `synth`;
  `clone` with an **x-vector-only** path (ECAPA timbre) and an **ICL** path
  (in-tree codec-encode of the reference, or external `--ref-codes`);
  `design` (instruct + preset speaker, 1.7B only).
- **Backends** — NPU (`tts/src/npu_gen.rs`: resident KV Talker, INT8/INT4
  weight-only, KvMtp/FusedMtp, stateful streaming codec); CPU
  (`gen_kv.rs`/`gen_kv_mtp.rs`, AVX2/FMA KV Talker); GPU = cache-free
  `TalkerGen::forward` (0.6B / parity / training).
- **Resident engine + adapter** — `tts/src/serve.rs` (`TtsEngine`, load-once
  resident KV graphs, streaming PCM callback) and
  `crates/cli/src/resident_tts.rs` (`TtsResident: ResidentModel`, `speak`
  action, env-gated on `BRAIN_TTS_WEIGHTS/CKPT/LANG/REF/REF_TEXT`), registered
  in `resident.rs`.
- **Capability** — `tts/src/caps.rs` (`TtsProvider`, `synth` action + manifest,
  24 kHz WAV blob).
- **Event/HFSM seam** — `crates/runtime` `SynthModel` trait + `Synthesizing`
  state; `crates/events` `UserSynthRequest` / `AudioChunk` over the JSONL
  protocol. **A real TTS model is intentionally not wired into `brain run`**
  (only `FakeSynthModel`), to keep the runtime crate's deps light; the full
  `UserSynthRequest → AudioChunk… → done` flow is exercised by
  `runtime/tests/controller.rs`.
- **Training/eval** — `tts/src/sft.rs` (Talker LoRA SFT, multi-codebook CE) +
  `brain tts finetune`; `brain tts sim` (speaker cosine);
  `crates/eval/src/tts.rs` (MCD, speaker_sim, log_mel_l1).

## Parity ladder

| Component | Metric | Result |
|---|---|---|
| Codec decode | max-abs vs HF waveform | 3.7e-2 (assert <6e-2) |
| Codec decode | log-mel L1 | 4.6e-3 (assert <1e-2) |
| Speaker encoder | x-vector cosine | 1.0 (assert ≥0.999) |
| Talker | codebook-0 top-1 match | exact |
| Talker | logit max-abs vs HF dump | <2.0 (assert) |
| Cache-free vs KV-cached codes | per-frame code equality | identical (`assert_eq`) |
| Talker analytic vs FD grads | atol 4e-3, rtol 8e-2 | passes |
| Codec streaming (NPU) vs CPU ref | max-abs | 9.78e-6 |
| MTP NPU vs CPU | max-abs | 0.0 (bit-identical) |
| KV-cache prefix hidden (0.6B fp32) vs CPU | max-abs | 3.05e-5 |
| CPU-vs-GPU forward logit (gradcheck) | max-abs | 5.4e-7 |

**Cross-backend gate**: `make parity` asserts CPU == Vulkan == NPU (the gradcheck
suite incl. in-process CPU-vs-GPU forward and TTS NPU codec vs CPU ref). Codec
and speaker are import-only (no backward/gradcheck); the Talker gradcheck lives
in `tts/tests/talker.rs` via `gradcheck::directional_check`.

## Serving contract (`docs/serving-contract.md`)

| # | Obligation | Status |
|---|---|---|
| 1 | Capability | **Met, with a gap** — `caps.rs`/`resident_tts.rs` expose `synth`/`speak` with manifest tests; **gap**: no `CancelToken` polling, so multi-minute `synth`/`clone` is not cancellable |
| 2 | Residency | **Met** — `TtsResident: ResidentModel`, env-gated, CPU decode, RAM footprint |
| 3 | Batching | **Sequential (default)** — `TtsInstance` implements `run` only, no `run_batch`; autoregressive decode is the stated justification, but unlike ASR there is no genuine batched forward and no documenting comment |
| 4 | D-Bus surface | **Partial** — `speak` is reachable over D-Bus via `brain serve --dbus`, **but** `brain tts serve` is a private Unix-socket JSONL server with its own Python clients — the side channel the contract warns against |
| 5 | Example | **Not met** — no `examples/tts/`; no D-Bus TTS demo |

## Remaining

- **gpu-core Talker is cache-free** — 0.6B / parity / training only; fast
  synthesis (NPU + `BRAIN_TTS_TALKER=cpu`) uses KV-cache.
- **MTP + codec are the remaining per-clip cost** after the Talker win — a fused
  single-infer MTP graph and further codec work are the next levers.
- **Codec windowed-mask for `T > 72` pending** — the sliding-window transformer
  assumes the full sequence fits the window; long-form decode needs the windowed
  mask.
- **Training is Phase 7** — this is the inference stack; Talker LoRA SFT exists
  (`sft.rs`), but from-scratch / codec / speaker training is tracked separately.
- **CPU codec is sub-real-time** — SEANet MAC count too high for CPU; the NPU is
  the real-time path.
- **Serving-contract gaps** — cancel token, `run_batch`, the `tts_serve` side
  channel, and the missing `examples/tts/` (above).

## See also

- `docs/models/tts/readme.md` — architecture, CLI, parity table, streaming protocol.
- `docs/models/tts/acceleration.md` — KV-cache/INT8/streaming-codec wins, env knobs, P40.
- `docs/serving-contract.md` — the five obligations.
- `AGENTS.md` → Models → Qwen3-TTS.
