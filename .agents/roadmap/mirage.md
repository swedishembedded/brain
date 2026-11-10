# mirage - roadmap

Scope: Mirage visual perception model (`crates/mirage`, arch id `mirage`).

Mirage (`Cactus-Compute/mirage`, Apache-2.0 weights and source) is a
51.3M-parameter perception model: a convolutional stem, 16 gated-mixing +
HConvGLU body blocks, 12 attention blocks, a CenterNet-style dense head with
mask prototypes, a text reader, and an 8-layer decoder with hyper-connection
lanes and engram tables. It ships as F32 safetensors (Flax names) and as a
2-bit Cactus-Quant `.cact`. The goal is the whole product in brain: import,
inference, training, serving, quantised kernels and a profiled throughput
story. Every lane below is open unless marked done.

Implementation rule: brain's Rust is written from the checkpoint's own facts
(tensor names, shapes, published formulas) and proven against numerical
goldens; nothing is transcribed from the vendor's source or engine.

## Lane L1 - foundation

| Item | State |
|---|---|
| Arch row `mirage` (Vision, `brain-mirage`, default ref `Cactus-Compute/mirage`); no capabilities declared | done |
| Crate in the workspace, layer map and default members | done |
| `config`: typed `MirageConfig` / `CheckpointConfig`, `deny_unknown_fields`, invariants name the field | done |
| `spec`: canonical manifest, 583 tensors / 54,078,804 parameters, training-only tensors marked | done |
| `import`: header-first strict safetensors import (names, shapes, dtype, byte lengths, metadata), pinned SHA-256, `config.json` cross-check, exact export round trip | done |
| `cact` / `cactmap`: bounds-checked `.cact` reader, CQ2/3/4 decoder with the packed form kept (`CqMatrix`), positional map derived from the config (806 entries for the released one), layout transforms, Hadamard permutations, vision manifest, charset, tokenizer blob | done |
| `tokenizer`: SentencePiece-BPE blob parser, score-ordered merging (heap, `O(n log n)`), byte fallback, markers, decode; ids and decodes equal to the publisher's reference on a tracked golden of 24 multilingual strings (en de fr es it nl pl) | done |

## Lane L2a - vision forward

Implemented in `crates/mirage/src/vision/` as host `f32` operators (CPU, row-parallel through the shared
CPU parallel layer). Each stage is gated against FP32 goldens from an independent numpy reference of the
same graph (cosine >= 0.9999 required; measured cosine rounds to 1.00000000 at every tap).

| Stage | State | max_abs vs golden (CPU) |
|---|---|---|
| Preprocess: letterbox size rule, area / bilinear resize, canvas, valid masks | done, resized image identical to the reference | 0 differing samples |
| Stem (5 stages, silu on all) | done | 1.2e-5 |
| Body (gated mixer, sequency-Walsh HConvGLU with 65-tap bands, GRN) | done | 5.6e-4 |
| Top (registers, 2D RoPE, masked GQA, level memory) and encoder output | done | top 8.1e-2 (register-scale values), encoder 8.6e-5 |
| Dense head (pyramid, neck, blocks, 10 channels), head input | done | dense64 5.9e-3, head 4.7e-4 |
| Peaks and proposal order | done, cells, order and boxes equal the golden | boxes within 1e-3 |
| Mask prototypes and coefficients, per-object field | done | protos 1.1e-5, coefficients 3.3e-6 |
| Bitmap rendering | done, 5 of 5 engine-corpus objects reproduce every run exactly | exact |
| Reader forward, windows, join, CTC collapse, crop rule | done | logits 2.0e-5 |
| Decoder memory (cells + position code, slot rows) | done | 4.9e-4 |

Open in L2a:
- Execution is on the host, not on the GPU op layer: no WGSL kernels (Walsh transform, banded channel
  convolution, GRN, ZC norm, 2D RoPE) and no CUDA parity run exist yet. The staging (one function per
  stage, weights prepared once) is the shape the kernel port needs; backward is not written.
- Peaks never consider cells outside the resized image (the `valid_only` switch is gone); the goldens were
  regenerated to match.
- The reader's strings differ from the engine's by spacing and a letter or two (the engine runs 2-bit
  weights); the join across windows is exercised by the long line.
- Upscale rounding is half-pixel bilinear in `f64`; an off-by-one against the vendor engine on part of the
  pixels is a known open item.

## Lane L2b - decoder, generation, JSON, forms

All on the host in f32, in the same stage structure as the vision half.

| Item | State | Numbers |
|---|---|---|
| Decoder (engram, 4-lane hyper-connections with sinkhorn, self-attention with causal taps and RoPE, cross-attention, Hadamard MLP, tied logits) | done | teacher-forced logits cosine 1.00000000 on both fixtures, max_abs 1.2e-3 (cats) / 2.1e-3 (text); argmax agreement 238 of 238 and 170 of 170 |
| Hadamard permutations from first principles (legacy shuffle seeded 11 and 13) | done, equal to the two deployment-file entries | exact |
| Incremental decoding (KV cache, rolling tap and engram state, per-image cross K/V) | done | one-token steps equal the full recompute, max_abs 0 |
| Greedy generation with absent-slot masking | done | tokens equal the reference exactly: cats 236 tokens, text 168 tokens, both ending in `</s>` |
| Loop guard (rank order, duplicate keys, requested sections, token budget) | done | stops at the section boundary of a partial line; a valid output is never cut |
| Line grammar and JSON assembly | done | 6 recorded cases match exactly (bitmap, German, fields, max objects, text, text lines) |
| Polygon masks (largest / merged / polygons) | done, held to overlap | rasterised IoU against the engine 0.970 to 1.000 over all objects of 4 cases |
| Forms: choice, count, text; candidate scoring, abstention, softmax | done | structure and candidate order; values are not compared with the engine (2-bit weights) |
| Typed `Describe` / `Extract`, `Options` (unknown keys refused) | done | |

Open in L2b:
- Box fields of a form are refused as unsupported (the scoring rule against proposals is not pinned).
- Polygon vertex lists differ from the engine's (marching squares at the field's 128 grid, Douglas-Peucker
  tolerance 0.13 cell); only overlap is held. Merged polygons join parts at their nearest vertices.
- The loop guard's `key()` rule is inferred (scene and entity lines by id, others by text); event (`V`) lines
  are returned as raw text since no recorded case shows their structure.
- Form scoring prepends a space to the prompt text to model the dummy prefix; the released tokenizer
  itself adds none. Probabilities differ from the engine's, as expected from 2-bit weights.
- Decoding fans out over at most 16 threads (`DECODE_THREADS`): more only adds wake-up cost per token.

## Lane L2c - device path

Started; only the first operator is on the device. `crates/mirage/src/device/` holds `Device` (an `Ops`
façade over a `Gpu` chosen by the process: `BRAIN_DEVICE=cpu` runs the CPU backend's WGSL JIT,
`BRAIN_BACKEND=cuda BRAIN_DEVICE=gpuN` one CUDA card) and the host path stays the oracle.

| Item | State | Numbers |
|---|---|---|
| Tests cap the host pool at 8 threads (`common::cap_threads`) | done | whole suite 5.8 s wall, about 1 CPU-minute (was about 35 CPU-minutes) |
| `hconv_glu` WGSL kernel (gelu_tanh(gate + bias) * value from a fused projection) | done, catalogued | CPU backend exact; CUDA max_abs 2.4e-7 |
| HConvGLU on the device: ZC norm (`rmsnorm_rows` with weight 1 + scale), one GEMM into gate and value, `hconv_glu`, one GEMM out; the Walsh transforms and the 65-tap bands are multiplied out into the two weight matrices at upload (`HConv::fold`, accumulated in f64) | done for one mixer | released body_0 / body_15 / top_0 / top_11 against the host mixer: CPU backend cosine 1.0 (max_abs 3.3e-7 .. 5.6e-3, the top blocks have large magnitudes); CUDA gpu1 cosine 0.9999999 (max_abs 1.1e-4 .. 3.7 at the largest magnitudes) |

Design note: no Walsh or banded-convolution kernel was written. The CPU backend's WGSL JIT supports a single
workgroup barrier, which rules out a staged butterfly, and the dense folded form turns the mixer into two
selector-picked GEMMs. It spends more arithmetic than the banded form (a 512x1024 and a 512x512 product per
row) for the roofline-tuned GEMM; whether that is right at batch 8 is for the first profile to say.

Open in L2c (everything below is not started): depthwise 3x3 / 1x5 and the stem convolution (check the
existing conv kernels first), GRN, 2D RoPE and masked attention for the top blocks, level memory, the dense
head and mask branches, the reader, batching over images; decoder on the device (mHC maps and sinkhorn,
Hadamard MLP, causal taps, engram gather, cross-attention over per-image K/V, tied-logit GEMM, device argmax)
with the shared step loop; `MirageModel` with `describe_batch` / `extract_batch`; a weight-free smoke test of
every step kind; the first nsys profile of a batch-8 describe on CUDA.

## Facts the foundation established

- The released `.cact` has 806 entries: 159 two-bit and 4 four-bit CQ matrices
  (the embedding and the three hyper-connection routing matrices are 4-bit), the
  rest FP16/FP32, the two Hadamard permutations (entries 226 and 227, stored as
  `f32` indices), a 36-value vision manifest, the 306-code-point OCR charset and
  the tokenizer blob. Its order is derived from the config in `cactmap::layout`.
- Quantised matrices are stored `[out, in]` and every other tensor keeps its
  Flax orientation, whatever its dtype; the routing matrices are stored
  `[layers * lanes, width]`.
- The embedding has `text + 11 language tags + detections + regions` rows; the
  tag count belongs to the tokenizer, not to `config.json`.
- The Hadamard MLP factors (16 and 32 at width 512) follow from the width, and
  its conditioning rank (8) and the engram tap count (4) are structural
  constants; none of them is in `config.json`.
- The released tokenizer adds no dummy prefix and has byte fallback.
- Tracked goldens live beside the crate tests (`crates/mirage/tests/golden/`)
  because `testdata/` is not tracked; the dumper is under `tools/goldens/`.

## Open lanes

| Lane | Scope | State |
|---|---|---|
| Vision encoder forward | stem, body, attention top, level memory, dense head peaks | open |
| Decoder forward | lane stack, engrams, cross-attention, Hadamard MLP, logits | open |
| Heads | boxes, masks, slots, enum scoring | open |
| OCR | the reader and greedy collapse | open |
| Training | full / head / LoRA, multi-task, gradient checks | open |
| Serving | capability, CLI, SDK, HTTP, D-Bus, batching | open |
| Quantised kernels | native low-bit CQ matmul from the packed weights | open |
| Performance | profile-driven batched throughput | open |
