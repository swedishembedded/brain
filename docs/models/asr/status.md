# ASR (Nemotron 3.5 ASR, Qwen3-ASR) — status ledger

Two speech-to-text models on the shared engine, imported 1:1 from HF and parity-gated,
then wired through brain's full serving stack (capability → residency → D-Bus).

## Models

| | Nemotron 3.5 ASR Streaming 0.6B | Qwen3-ASR 1.7B |
|---|---|---|
| arch | FastConformer encoder + RNN-T transducer | Whisper-style audio encoder + spliced Qwen3-1.7B decoder |
| crate | `crates/nemotron` | `crates/qwen-asr` (reuses `crates/qwen`) |
| streaming | **yes** — the streaming model | offline (fixed audio window) |
| parity | pooler 7.9e-6, RNN-T tokens EXACT vs HF | encoder 6.9e-6, greedy tokens byte-identical vs HF |
| trainable | full (gradchecked: kernels, attention, conv, block, encoder, RNN-T loss, LSTM BPTT) | by composition (qwen decoder + splice + vit block bwd) |

Transcription is correct end to end on the LibriSpeech "Mr. Quilter" clip
(*"Mr. Quilter is the apostle. Of the middle classes, and we are glad to welcome his
gospel."*).

## Serving (the contract, all five obligations met — see `docs/serving-contract.md`)

- **Capability**: shared audio-in/text-out schema in `audio::asr_caps` (one
  implementation), `nemotron::caps` / `qwen_asr::caps` providers; detok via
  `nemotron::tokenizer` (metaspace BPE) and the Qwen BPE.
- **Residency**: `cli::resident_asr` (`NemotronResident`, `QwenAsrResident`),
  env-gated `BRAIN_NEMOTRON` / `BRAIN_QWEN_ASR`, registered in `build_executor`.
  Build-once in `activate`.
- **Batching**: Nemotron `run_batch` is a **true batched forward** — concurrent
  same-prompt stream-windows encode in one FastConformer pass
  (`Encoder::transcribe_batch`, row-concatenated per-frame matmuls, bit-identical to
  the single-utterance path). Qwen3-ASR is offline/autoregressive → sequential
  `run_batch` on a build-once fixed-window instance (encoder amortised).
- **D-Bus + example**: new `StreamTranscribe(model, params, pcm_fd) -> (job, event_fd)`
  (`crates/dbus`) windows a continuous PCM fd into executor jobs and streams `segment`
  frames back; `examples/asr/{transcribe_mic.py, bench_streams.py}`.

## Performance (release, CPU backend, Core Ultra 7 155H, 22 threads; 5.86 s clip)

Nemotron concurrent-stream benchmark (`examples/asr/bench_streams.py`, 2 s windows):

| streams | wall (s) | aggregate RTF | per-stream RTF | first segment (ms) |
|--:|--:|--:|--:|--:|
| 1 (cold) | 13.8 | 0.43 | 0.43 | 9712 (incl. one-time 0.6B build) |
| 1 (warm) | 7.6 | 0.77 | 0.77 | 2070 |
| 2 | 11.4 | 1.03 | 0.55 | 3587 |
| 4 | 15.0 | 1.56 | 0.42 | 4302 |

Scheduler: `builds: 1` (built once), `max_batch: 3` (concurrent windows batched into
one forward), `batches: 18`.

Reading it: aggregate throughput **scales with concurrency** (0.77 → 1.56× real time as
streams go 1 → 4) because the batched encoder forward and the continuous-batching
scheduler share work across streams; per-stream RTF drops under contention (the usual
throughput/latency trade). A single CPU box sustains ≈1.5 concurrent real-time streams;
on a GPU the batched forward is where the win compounds.

Qwen3-ASR (1.7B, offline, 8 s fixed window, same clip) — served and correct
(*"Mr. Quilter. … To welcome his gospel."*) but heavy on CPU:

| streams | wall (s) | per-stream RTF | first segment (ms) |
|--:|--:|--:|--:|
| 1 | 93 | 0.06 | 56108 (incl. one-time ~50 s 1.7B build) |
| 2 | 144 | 0.05 | 55825 |

`builds: 1`, `max_batch: 1` — one resident instance; offline autoregressive decode
does not cross-stream batch (documented). At ~16× slower than real time on CPU, the
1.7B decoder is bandwidth-bound (fp32 weights per token); INT8 weights and a GPU are
the levers (the register-tiled/int8 matmul kernels fall back to a slow path on the CPU
JIT — see the serve log). Use Qwen3-ASR for accuracy/offline transcription, Nemotron
for streaming/real-time.


## Not yet

- Frame-synchronous streaming (the FastConformer runs offline/windowed here; true
  left-context-cached streaming is a follow-up).
- Qwen3-ASR variable-length serving (fixed window today; probe-per-length or a padded
  KV scheme would generalise it).
- NPU path (OpenVINO Conformer export); Vulkan runs bit-exact but is submit-bound.
