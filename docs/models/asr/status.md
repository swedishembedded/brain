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
  implementation: `transcribe` + the streaming `transcribe_stream`),
  `nemotron::caps` / `qwen_asr::caps` providers; detok via `nemotron::tokenizer`
  (metaspace BPE) and the Qwen BPE.
- **Residency**: `cli::resident_asr` (`NemotronResident`, `QwenAsrResident`),
  env-gated `BRAIN_NEMOTRON` / `BRAIN_QWEN_ASR`, registered in `build_executor`.
  Build-once in `activate`.
- **Batching**: Nemotron `run_batch` is a **true batched forward** for both actions —
  concurrent same-prompt whole utterances encode in one FastConformer pass
  (`Encoder::transcribe_batch`), and concurrent live sessions step through one
  batched encoder pass (`Encoder::stream_push_batch`; per-frame matmuls
  row-concatenated across streams, attention/conv per stream). Qwen3-ASR is
  offline/autoregressive → sequential `run_batch` on a build-once fixed-window
  instance (encoder amortised).
- **D-Bus + example**: `StreamTranscribe(model, params, pcm_fd) -> (job, event_fd)`
  (`crates/dbus`) reads a continuous PCM fd and streams `segment` frames back;
  `examples/asr/{transcribe_mic.py, bench_streams.py}`.

## Frame-synchronous streaming (`nemotron::stream`)

The FastConformer is cache-aware by design — `chunked_limited` attention (each
4-frame chunk sees 14 chunks of left context + itself) over causal convs — so
`crates/nemotron/src/stream.rs` streams it *exactly*: per layer it caches the
56-row K/V attention band and the 8-row causal-conv GLU tail; the mel front end
(`audio::asr_frontend::NemotronMelStream`), the three subsampling stages and the
RNN-T decoder (`DecodeState`) all carry state across pushes. A fixed 63-row
relative-position table per layer (offsets `i-j ∈ [-3, 59]`) replaces the offline
`[2T-1]` ladder.

**Parity**: streamed output equals the offline whole-utterance forward — pooler
frames **bit-for-bit** under `BRAIN_NO_FASTCONV=1` (shape-invariant kernels; the
AVX2 conv fast path's documented ≤1-ulp reassociation applies otherwise) and the
**token sequence exactly** either way, asserted by `stream::tests` (tiny
random-weight model, ragged pushes + batched-vs-single) and two `--ignored`
checkpoint tests (`streaming_e2e_matches_offline_transcribe`,
`stream_sessions_deltas_match_offline` on the Mr. Quilter clip).

**Serving**: `transcribe_stream` (session id + `eos`, newly-emitted text/tokens
per window) is served by `nemotron::caps::StreamSessions` from both the direct
`Provider` and the resident instance; `StreamTranscribe` on D-Bus auto-upgrades
to it whenever the model's manifest advertises the action (windows share one live
session — no per-window re-encode; `qwen-asr` keeps the offline per-window path).
Algorithmic latency is one chunk (~0.32 s) + the front end's 16 ms lookahead;
sessions idle >10 min are reaped. Evicting the resident instance drops its live
sessions (a restarted id starts a fresh session).

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

- Qwen3-ASR variable-length serving (fixed window today; probe-per-length or a padded
  KV scheme would generalise it).
- NPU path (OpenVINO Conformer export); Vulkan runs bit-exact but is submit-bound.
- Streaming-session survival across residency eviction (state is dropped with the
  instance today; serialising `StreamState` would allow swap-out mid-stream).

Done since the table above: **frame-synchronous streaming** (left-context-cached
FastConformer + stateful RNN-T — see the section above; the "streaming" row now
means it in the strict sense).
