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
- The golden peaks keep padding cells as candidates (a flat bias-only plateau gives 32 text proposals on
  the cats image); `ProposalOptions::valid_only` (the default) excludes them, and the golden test turns it off.
- The reader's strings differ from the engine's by spacing and a letter or two (the engine runs 2-bit
  weights); the join across windows is exercised by the long line.
- Upscale rounding is half-pixel bilinear in `f64`; an off-by-one against the vendor engine on part of the
  pixels is a known open item.

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
