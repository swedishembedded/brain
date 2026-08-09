# Nemotron 3.5 ASR Streaming & Qwen3-ASR (`crates/nemotron`, `crates/qwen-asr`)

Two speech-to-text models sharing one audio-in/text-out contract: Nemotron 3.5 ASR
Streaming 0.6B (FastConformer + RNN-T, frame-synchronous streaming) and Qwen3-ASR
1.7B (Whisper-style encoder + Qwen3 decoder, offline fixed-window).

## Model id and weights

- **Id:** `brain/nemotron`, `brain/qwen-asr` — reserved vendor `brain/`, never
  auto-fetched.
- **Weights:**
  - `BRAIN_NEMOTRON` — Nemotron 3.5 ASR checkpoint directory (HF layout).
  - `BRAIN_QWEN_ASR` — Qwen3-ASR checkpoint directory (HF layout).
  - `BRAIN_QWEN_ASR_WINDOW` — Qwen3-ASR audio window in seconds (default `30`).
  - `BRAIN_QWEN_ASR_MAXNEW` — Qwen3-ASR max generated tokens (default `200`).

## Surfaces

**D-Bus only.** Neither model is registered in `crates/cli/src/catalog.rs` (unlike
facenet/fastvlm), so `brain caps`/`brain do` do not know either id — there is no
CLI path at all, only `crates/cli/src/resident_asr.rs` wiring both into
`resident::build_executor` for `brain serve --dbus`. Not chat/image/embed-shaped
(action names are `transcribe`/`transcribe_stream`, not `generate`), so no HTTP
subsection either.

## Inference

### D-Bus

Shared schema (`audio::asr_caps`), audio always **raw mono f32 little-endian PCM
at 16 kHz**:

- `transcribe` (both models) — required `audio` blob in, `text` blob + `text`/
  `tokens`/`num_tokens` out; params `prompt_id` (default `0`), `sample_rate`
  (default `16000`, must be `16000`). Streaming (progress frames).
- `transcribe_stream` (**Nemotron only** — Qwen3-ASR's manifest does not advertise
  it) — one window of a live session; params `stream` (session id), `eos`
  (default `false`), `prompt_id`, `sample_rate`; `audio` blob optional (a final
  `eos`-only call flushes a closed mic). Concurrent same-prompt streams/windows
  batch through **one true `run_batch` forward** (`Encoder::transcribe_batch` /
  `StreamSessions::step_batch`); Qwen3-ASR's `run_batch` is the sequential
  default over its build-once, fixed-window instance.

```bash
BRAIN_NEMOTRON=/path/to/nemotron/hf dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/asr/transcribe_mic.py --model brain/nemotron --wav clip.wav'
```

Reference client: [`examples/asr/transcribe_mic.py`](../../../examples/asr/transcribe_mic.py)
(`--model brain/nemotron|brain/qwen-asr`, `--wav FILE` or live mic capture) — see
[`examples/asr/README.md`](../../../examples/asr/README.md) for the full
`StreamTranscribe` protocol and the benchmark script.

## Training / Fine-tune / LoRA

Nemotron has a real trainer (`crates/nemotron/src/train.rs`: `Transducer` +
`AcousticModel`, both finite-diff gradchecked) but no CLI verb — no command to
give. Qwen3-ASR has no trainer.

## Not supported

`training` (verb), `finetune`, `LoRA`, `QLoRA`, `batch > 1` (Qwen3-ASR only —
Nemotron batches for real, see above).

## See also

- Crates: `crates/nemotron`, `crates/qwen-asr`
- Workstream ledger: [`status.md`](status.md)
- [`examples/asr/README.md`](../../../examples/asr/README.md)
