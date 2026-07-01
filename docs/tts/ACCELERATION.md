# Qwen3-TTS acceleration: what changed, why it mattered, and how the backends compare

This documents the work that took brain's Qwen3-TTS from a slow, single-path
prototype to a fast, multi-backend system with a resident streaming server. Each
section says **what** changed, **why** it mattered, and the **validated** numbers.

> Measurement note: the per-stage figures below were captured on the Intel NPU
> (OpenVINO) / this host during clean runs. All are *warm* (graphs compiled +
> blob-cached) unless stated. "cold" = first compile of a graph for a given
> shape, paid once and cached thereafter.

---

## TL;DR — the wins

| Improvement | Before | After | Factor |
|---|---|---|---|
| Talker decode (1.7B, per frame) | 1320 ms (cache-free recompute) | **70–180 ms** (KV-cache decode graph) | **~7–19×** |
| Prompt prefill (194-token ICL prefix) | 34 s (token-by-token) | **0.6–1.5 s** (one prefill inference) | **~26×** |
| Design engine load | 450 s | **19.6 s** (warm) | **~23×** |
| 1.7B on the NPU at all | OOM (~20 GB fp32) | **fits** (INT8, ~1.4 GB/graph) | enabling |
| Codec streaming | full re-decode / windowed 2× | **stateful, decode-only-new-frames** (exact) | removes 2× + streams |
| Repeated server request | full graph reload each time | **resident** (load once) | amortized |

Fastest path today for the 1.7B clone/design models: **the NPU** (KV-cache INT8
talker + streaming codec), which is also the default when you pass `--device npu`.

---

## 1. Resident KV-cache Talker graph (the headline win)

**What.** The Talker is a 28-layer Qwen3 GQA decoder that autoregressively samples
one codec frame at a time. The original NPU path recomputed the *entire* growing
context every frame (`O(T²)`). We added two compiled graphs:

- a **decode-step** graph — one token + per-layer past K/V in, hidden + new K/V
  out (single-query GQA, offset RoPE fed as inputs, host-managed cache);
- a **prefill** graph — the whole prompt prefix → hidden + per-layer K/V in one
  inference, to seed the cache.

OpenVINO's stateful `ReadValue/Assign` isn't expressible through the pure-Rust
ONNX exporter, so the K/V cache is carried as **explicit past/present tensors**.

**Why.** Per-frame cost drops from "re-run the whole model" to "one token against
a cached context," and a 194-token voice-clone prefix no longer costs 34 s of
token-by-token warm-up.

**Validated.** Talker 1320 → **70–180 ms/frame** (~7–19×); prefill 34 s →
**0.6–1.5 s** (~26×); correctness gated **exact** on the 0.6B in fp32 vs the CPU
reference (`max-abs 3.05e-5`). Default for NPU talkers.

## 2. INT8 weight-only Talker (makes the 1.7B fit)

**What.** Per-output-channel symmetric INT8 for the large linears
(`DequantizeLinear → MatMul`, axis=1).

**Why.** fp32 for the 1.7B Talker is ~20 GB and OOM'd the NPU; the autoregressive
Talker is memory-bandwidth-bound (it streams the full decoder weights every
frame), so halving the weight bytes both **fits** it and **speeds** it.

**Validated.** ~1.4 GB compiled graph (vs OOM); 1.7B voice clones run on the NPU.

## 3. MTP on the NPU — measured, and deliberately *not* shipped as default

**What.** The residual code predictor (MTP) is the same Qwen3 block, so it can
reuse the decode graph. We built it (`KvMtp`, `BRAIN_TTS_MTP=npu`).

**Why it stayed opt-in.** It's **bit-identical** to the CPU MTP (`max-abs 0.0`)
but **not faster** (~227 ms/frame vs CPU ~225 ms): the MTP runs 15 *tiny*
substeps per frame, so per-substep NPU dispatch + K/V marshaling dominate the
small compute. **Lesson: the NPU wins for one-big-infer/frame (Talker), loses for
many-tiny-infers/frame (MTP).** CPU MTP remains the default.

## 4. Server mode — resident engines (amortize the graph load)

**What.** `brain tts serve`: a Unix-socket, JSONL server. A single executor
thread owns the resident engines (OpenVINO infer requests aren't thread-shared)
and pulls jobs from a channel; connection threads stream `audio_chunk`s. Python
clients (`scripts/voice-clone.py`, `scripts/voice-design.py`) play the PCM to the
speakers.

**Why.** Compiling/loading the ~1.4 GB INT8 graphs is a real per-process cost;
loading them **once** and serving many requests turns each request into
~compute-only.

**Validated.** Engine loads once, then request 1 = 71 s (load+gen) vs request 2 =
**45 s (gen only, no reload)**.

## 5. Design-engine load: 450 s → 19.6 s

**What.** Design/CustomVoice/Synth engines were cold-compiling **two** ~1.4 GB
graphs (decode + prefill). The prefill graph only pays off for clone's long
(~194-token) reference prefix; a design prompt's ~25-token prefix seeds the cache
token-by-token in ~1.8 s. So `KvTalker::load` gained `with_prefill`, and the
server skips the prefill graph for short-prefix engines; caps were trimmed
256 → 192.

**Why.** The first `brain tts serve` design request was unusably slow (450 s).

**Validated.** Design engine load **450 s → 19.6 s** (warm, one cached graph);
valid audio on NPU and CPU. The server also gained `catch_unwind` (a failing
engine returns an error instead of silently hanging every request) and
`BRAIN_TTS_NPU_DEVICE` support.

## 6. Stateful streaming codec (now the default)

**What.** The Mimi-style codec (RVQ → causal transformer → SEANet, ×1920
upsample) is conv-compute-bound and *slower than real-time*. First we streamed a
sliding *window* (re-decoding warmup each chunk = ~2× cost); then we built a
**stateful** decoder: the causal front runs once, and the SEANet/upsample back is
driven chunk-by-chunk carrying **per-conv left-context buffers + the transposed-
conv overlap** as graph I/O (`build_codec_back_stream_graph`, `BackStreamSession`,
`NpuStreamCodec`). Each chunk decodes only its *new* frames — no re-decode.

**Why.** Removes the 2× windowed overhead, is **exact** (no window-boundary
approximation), and streams at fine granularity for lower first-audio latency.

**Validated.** vs the bit-exact CPU reference: **max-abs 9.78e-6**; warm 25.4 s vs
28 s (windowed) for 40 frames; made the **default** (`BRAIN_TTS_CODEC=windowed`
forces the old path, `cpu-stream` the CPU reference). A note on the CPU reference:
even release + rayon it's **0.08× real-time** — the SEANet MAC count over the
upsampled sequence is too high for CPU, confirming the NPU is the real-time path.

## 7. CPU SIMD + Vulkan as a first-class backend

- **CPU:** AVX2/FMA `dot` (4 accumulators, 32 f32/iter, runtime-detected + scalar
  fallback) in the CPU KV Talker/MTP.
- **Vulkan:** promoted to a first-class `gpu-core` backend (ash + naga
  WGSL→SPIR-V), runtime-selected with graceful wgpu fallback — the portable GPU
  training/forward path.

## 8. Vulkan correctness — two real driver-class bugs found with validation layers

Installed the Khronos validation layers and wired `BRAIN_VK_VALIDATE`
(synchronization validation + a debug messenger; `=gpu` for GPU-assisted). Both
bugs below were **invisible to code inspection** and needed the layers +
controlled reproduction.

- **#12 — tiled-embedding parity (Intel ANV).** The vocab-tiled embedding/lm_head
  (`step_sliced`) intermittently (~15–25%) corrupted *unrelated* downstream
  results. Sync- and GPU-assisted validation were both **clean** and the kernels
  are race-free — the trigger is **sub-range (non-zero offset) descriptor
  bindings**: ANV doesn't make a prior dispatch's writes visible across a sliced
  binding via an in-command-buffer compute-compute barrier (even `ALL_COMMANDS`),
  but a **submit+fence boundary is honoured**. Fix: mark sliced steps and
  serialize batches that contain one (only vocab-tiled large models pay it).
  Validated 0/12 (was flaky).
- **#20 — host-coherent storage flake.** Storage buffers were
  `HOST_VISIBLE|HOST_COHERENT` and up/downloaded by direct map; GPU compute
  writing host-coherent memory under many per-call allocations flaked on ANV
  (deterministic in `mse_fd`; `mse_grad` was *correct*, but an `mse_value` call in
  the FD loop returned garbage). Fix: **`DEVICE_LOCAL` storage + a reusable
  host-visible staging buffer + GPU copy** (the standard pattern), host-visible
  fallback kept for llvmpipe. Validated: `mse_fd` 5/5 (was 5/5 fail); full
  gradcheck suite green on Vulkan.

## 9. Cross-backend parity gate (`make parity`)

One command asserts **CPU == Vulkan == NPU**: the gradcheck suite on CPU and on
Vulkan (incl. a *direct* in-process CPU-vs-GPU forward-logit comparison —
`max-abs 5.4e-7`), plus the TTS NPU codec graph vs the CPU reference. Result:
**PASS**.

---

## Backend comparison — clone & design

**Which backends apply.** The TTS *inference* Talker/codec run on the **NPU**
(OpenVINO) with the host handling sampling/MTP, or on the **CPU**
(`--device npu BRAIN_TTS_TALKER=cpu`, codec still on the NPU). The
**Vulkan/gpu-core** path is the cache-free forward used for training and the
smaller (0.6B) models; the 1.7B decoder (multi-GB) is deliberately *not* uploaded
to the GPU backend (that's the whole point of the NPU host path), so Vulkan is not
a viable 1.7B-TTS inference backend — it's validated for correctness via the
parity gate, not used for large-model synthesis.

**Per-stage cost (1.7B, warm, validated this session):**

| Stage | NPU (default) | Notes |
|---|---|---|
| Prefill (clone, 194-token prefix) | 0.6–1.5 s | one prefill inference (was 34 s) |
| Talker decode | 70–180 ms/frame | KV-cache INT8 (was 1320 ms cache-free) |
| MTP (residual codes) | 114–225 ms/frame | CPU (NPU measured equal → kept on CPU) |
| Codec | streaming, exact | stateful NPU decoder (default) |
| Design engine load | 19.6 s warm | (was 450 s) |

**Fastest = NPU.** The KV-cache INT8 Talker + stateful streaming codec is both the
fastest path and the default for `--device npu`. The CPU path
(`BRAIN_TTS_TALKER=cpu`) is the portable fallback (fits in host RAM, slower on the
talker; still uses the NPU codec). After the Talker was made fast, the dominant
remaining per-clip costs are the **MTP** (CPU, per-frame) and **codec** — the next
levers are a fused single-infer MTP graph and further codec optimization.

> A clean side-by-side *total-time* table (NPU vs CPU wall clock) is best produced
> with `scripts/` on an unloaded host: e.g. `TTS_PROFILE=1 brain tts clone
> --device npu …` vs the same with `BRAIN_TTS_TALKER=cpu`. During this write-up the
> shared host was at load-average ~100, which is not representative (a warm clone
> that is normally ~45 s took 850 s), so only the per-stage figures above —
> gathered on a usable machine — are reported.

---

## How to run

```bash
# One-shot CLI (per-stage timing with TTS_PROFILE=1):
brain tts clone  --device npu --ckpt <1.7B-Base>        --weights-dir out/tts-1b7 \
    --ref voice.wav --ref-text "$(cat voice.txt)" --text "hello"      --out clone.wav
brain tts design --device npu --ckpt <1.7B-VoiceDesign> --weights-dir out/tts-vd  \
    --instruct "a calm narrator" --text "in a world..."               --out design.wav

# Resident server + streaming Python clients (play to speakers):
brain tts serve
python scripts/voice-clone.py  "hi, this is my voice clone"
python scripts/voice-design.py --instruct "a deep cinematic narrator" --text "..."
```

### Env knobs

| Variable | Effect |
|---|---|
| `BRAIN_TTS_TALKER` | `npu-kv` (default), `npu-int8`, `npu` (fp32 cache-free), `cpu` |
| `BRAIN_TTS_MTP` | `npu` opts MTP onto the NPU (default CPU) |
| `BRAIN_TTS_CODEC` | `npu-stream` (default), `windowed`, `cpu-stream` (CPU reference) |
| `BRAIN_TTS_NPU_DEVICE` | `npu` (default), `cpu`, `gpu` — OpenVINO target device |
| `BRAIN_TTS_STREAM_CHUNK` / `_WIN` | streaming codec chunk / window size |
| `BRAIN_VK_VALIDATE` | `1` (sync validation), `gpu` (GPU-assisted) — Vulkan debugging |
| `BRAIN_VK_SERIAL` | force submit+fence per Vulkan dispatch (diagnostic) |
| `BRAIN_TILE_BUDGET_WORDS` | force vocab tiling on small models (test #12) |

Cross-backend parity: **`make parity`**.
