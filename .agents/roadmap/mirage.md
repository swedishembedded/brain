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

Done. Every stage of the model runs on the GPU op layer (`crates/mirage/src/device/`), batched over images, and
`MirageModel` (resident weights on a caller-chosen `Gpu`) answers `describe_batch` / `extract_batch`. The host
f32 path stays the parity oracle. The device is whatever the process selected (`BRAIN_DEVICE=cpu` runs the CPU
backend's WGSL JIT; `BRAIN_BACKEND=cuda BRAIN_DEVICE=gpuN` one CUDA card).

Layout: `prims.rs` (one method per kernel), `hconv.rs` (folded HConvGLU), `vision.rs` (stem, body, top, dense
head, mask branches, per-image masks), `reader.rs` (OCR reader and decoder-memory construction),
`decoder.rs` (device decoder and the shared step loop), `batch_model.rs` (`MirageModel`).

Design decisions:
- Padding cells are carried, not gathered. Each image owns the full grid; row masks zero the cells outside its
  resized image where the host operators do, and the attention top runs over `cells + registers` tokens with a key
  mask (every other top operator is row-wise, so a padding row never reaches a valid row). Image `i` of a batch
  therefore equals a batch of one.
- The Walsh transforms and 65-tap bands (HConvGLU) and the Kronecker rotations, permutations and diagonal scales
  (Hadamard MLP, `HadamardMlp::fold`) are multiplied out into dense matrices at upload, so the device runs
  selector-picked GEMMs. The band and rotation parameters stay in the host structs (`HConv`, `HadamardMlp`)
  for the backward lane to map dense gradients back.
- The decoder advances a batch in lockstep (one token per sequence at a shared position); memories of different
  lengths share a stride and a key mask; a finished sequence steps on a filler token until the batch is done.
  Hashing of n-grams, rendering and the loop guard stay on the host (one `[batch]` readback per step).
- Form extraction (`extract_batch`) runs the vision half and the memory on the device and scores branches with the
  host decoder: branch scoring clones decoder state and feeds multi-token chunks, which the lockstep device
  loop does not model. Open: a device state copy.
- New kernels (catalogued): `mirage_conv2d`, `mirage_dwconv`, `mirage_rows_mask`, `mirage_add_masked`,
  `mirage_gated_mix`, `mirage_grn_{energy,scale,apply}`, `mirage_chan_fma`, `mirage_axpy`, `mirage_swiglu_rows`,
  `mirage_mean_h`, `mirage_upsample2`, `mirage_rope2d`, `mirage_attn` (online-softmax masked GQA), `mirage_copy_rows`,
  `mirage_embed_lanes`, `mirage_gather_rows`, `mirage_engram_gate`, `mirage_mhc_{maps,mix,post}`, `mirage_taps`,
  `mirage_row_write`, `mirage_rope_pos`, `mirage_hada_mid`, `mirage_lane_mean`, `hconv_glu`. Reused: `rmsnorm_rows`
  (weight `1 + scale`), `silu`, `sigmoid`, `add`, `mul`, `bias_add`, `argmax_row` and the selector-aware GEMM.
  `softmax_rows` does not JIT on the CPU backend, so the rank-8 softmax of the Hadamard conditioning lives in
  `mirage_hada_mid`. All kernels are one thread per element (or per row/head) and untuned.

Parity (released weights; goldens `testdata/golden/mirage/goldens/{cats,text}`; cosine, max_abs):

| Stage | CPU backend | CUDA gpu1 |
|---|---|---|
| stem stages 256..16 | 1.0000000, 3.6e-6 .. 1.3e-5 | cos >= 0.999998 (max_abs <= 1e-2) |
| body | 1.0000000, 1.7e-3 | 0.9999999, 5.4e-1 |
| top (valid rows + registers) | 1.0000000, 1.4e-1 | 0.999998, 1.6e2 (register-scale values) |
| encoder memory | 1.0000000, 9.5e-5 | 0.999994, 4.9e-2 |
| head_in / head (10 channels) | 1.0000000, 7.8e-5 / 5.1e-4 | 0.999999, 7.2e-2 / 3.9e-1 |
| reader logits | 1.0000000, 2.0e-4 | 1.0000000, 8.8e-3 |
| decoder memory (cells + slots) vs host and golden | 1.0000000, 2.1e-4 | 0.99999998, 9.2e-2 |
| decoder teacher-forced logits (40 steps, 2 sequences) | 1.0000000, 4.0e-4; argmax 40/40 | 0.999998, 1.1; argmax 40/40 |
| greedy generation, both fixtures in one batch | tokens equal the reference: 236 and 168, `</s>` | equal: 236 and 168 |

End to end (`device_model` tests): `describe_batch` of the two fixtures equals the host image by image on the CPU
backend, exactly (proposals, tokens, reasoning, relations, polygon masks, bitmap run-length codes, text lines);
on CUDA everything but the third decimal of a score and the odd polygon vertex at the threshold is equal.
Batch of N equals N single runs: CPU exact; CUDA the GEMM selector may pick another kernel for another row count,
so head maps agree to cosine 0.99999987 (max_abs 0.16 at logit scale) and the described results agree; a batch
of 8 (4 + 4 duplicates) describes duplicates identically. `extract_batch` agrees with the host on values and
probabilities (CPU 1e-4, CUDA 2e-2).

Weight-free coverage: the tiny configuration runs every device step kind (all primitives, folded mixers, the
batched vision path with images of different valid regions, the reader, the decoder with engram site, taps,
lanes and unequal memories) against the host oracle in `device`, `device_vision`, `device_reader` and
`device_decoder`. There is no tiny `MirageModel` smoke test because it needs a tokenizer blob.

First profile (nsys, CUDA gpu1, test binary, batch-8 `describe_batch` with bitmap masks, 4 cats + 4 text, untuned;
`device_model::a_batch_of_eight_describes_duplicates_identically`, 12.9 s wall, 6.02 s of kernel time):

| kernel kind | time | share | launches |
|---|---|---|---|
| `mirage_attn` | 5424 ms | 90.0% | 3804 |
| `mirage_mhc_maps` | 123 ms | 2.0% | 1896 |
| `argmax_row` | 81 ms | 1.3% | 236 |
| `rmsnorm_rows` | 80 ms | 1.3% | 17574 |
| `max_abs_rows` + `quant_pack` (GEMM activation prep) | 94 ms | 1.6% | 59368 |
| cublasLt split-K reduce | 46 ms | 0.8% | 29532 |
| `mirage_taps` + `mirage_row_write` | 27 ms | 0.5% | 16116 |
| everything else (elementwise, copy, conv, ...) | < 1% each | | |

The decoder's attention (self over up to 240 keys and cross over 352 keys, 32 threads per launch, about 1.7 ms
each, 3792 launches) is the whole story: first tuning target is a key-parallel attention (workgroup per
sequence and head, cooperative softmax) registered as a candidate, then batching the 8-layer step into fewer
launches (a step issues about 600 dispatches), `mirage_mhc_maps` (one thread per row, serial Sinkhorn) and the
argmax. The report was `~/project/shared/tmp/mirage-l2c/prof8.nsys-rep` (outside the repo).

Open after L2c: the reader and mask rendering run per text line / on the host (device mask-field and bitmap
kernels are a throughput item); the proposal peak pick, slot box codes and pooling ranges are host work between
two device submissions; `extract_batch` scores forms on the host decoder; CUDA tolerances above are the
selector's tensor-core GEMMs (see the roofline-by-selection rule), not kernel defects; tuning is L6.

## Lane L6 - performance, round 1

Hardware: one NVIDIA GH200 (gpu1 of a shared node; another session's jobs run on the card, so wall times include
time slicing), test binary `brain-mirage --test it`, released weights, cats (480x640) and text (600x800) mix,
bitmap masks, up to 600 tokens, TF32 tensor-core products for the f32 weights (the repo default,
`GemmPrecision::Auto`; `Fp32Reference` is the precise opt-out). The parity ladder (the whole crate suite, 95 tests)
passes on the CPU backend and on CUDA gpu1 after every change below; parity numbers did not move (the decoder
logits cosines stayed 0.99999935 / 0.99999765, tokens identical).

| Change | batch-8 describe wall | 2-sequence generation | kernel time |
|---|---|---|---|
| L2c baseline | 12.9 s | 16.5 s | 6024 ms (attention 90%) |
| key-parallel attention (scores, row softmax, weighted sum; the serial kernel stays as `attention_reference`, tested equal) | 8.4 s | 7.6 s | 756 ms |
| dense f32 products read f32 activations (`act_f32`) | 7.8 s | 6.7 s | |
| one submission per step, argmax merged, tap histories written by the tap kernel, fused sigmoid gate | 3.1 s | 2.3 s | |
| token feedback and engram hashing on the device (`WINDOW` = 16 steps per host round trip), uninitialised temporaries | 2.8 s | 2.4 s | 1018 ms (127k launches) |

Hypotheses killed or found:
- GEMM activation packing (`max_abs_rows` + `quant_pack`, 59k launches): not a precision change. `Ops::act` packed an
  int8 copy of every activation whenever the width was a multiple of 32, and dense f32 weights never read it. Dense
  products now call `act_f32`; results are unchanged. The only reduced precision on the f32 path is TF32 inside the
  library GEMM (framework default).
- Host round trips per token: removing them (8 s of `cuStreamSynchronize` before, one submission per step now, a
  readback every 16 steps) cut decode a lot at first (16.5 s to 2.3 s with the attention and submission work), but the
  last change (device-side feedback, 16 steps per readback) did not move decode time: a step costs about 9.5 ms
  whatever the batch, so the step is bound by about 500 launches of 3-50 us kernels, not by syncs.

Throughput curve (gpu1, `describe_throughput_over_batch_size`; vision = letterbox, upload, encoder, dense head;
memory = peaks and slot construction; decode = prefill and steps with the guard; post = mask fields, reader, JSON):

| batch | wall s | images/s | vision s | memory s | decode s | post s |
|---|---|---|---|---|---|---|
| 1 | 2.40 | 0.42 | 0.11 | 0.00 | 2.26 | 0.03 |
| 8 | 2.84 | 2.82 | 0.18 | 0.00 | 2.41 | 0.21 |
| 32 | 4.15 | 7.71 | 0.34 | 0.02 | 2.55 | 1.10 |
| 64 | 5.87 | 10.90 | 0.55 | 0.02 | 2.98 | 2.03 |

Latency of a batch equals its wall time (p50 equals p95 for one batch). Vendor engine CPU baseline: 265-271 ms per
image at every thread count, so batch 64 (92 ms per image) is 2.9x the engine, batch 32 is 2.0x, and batch 1 (2.4 s)
is 9x slower than one engine image. The decode is the floor: about 236 steps at 9.5 ms.

Next (round 2): per-step launch count (about 500): concatenated GEMMs sharing an input (phi pre/post/res; self q/k/v
plus gate; cross q plus gate; MLP cond plus first rotation), `mirage_attn_apply` (48 us a launch, serial over keys),
`mirage_mhc_maps` (65 us, a thread per sequence), `argmax_row` (343 us, a thread per sequence), then step replay;
batch-N post-processing (`post` grows with N: mask fields are host work) overlapped with the next batch; device form
scoring (`extract_batch` still scores candidates on the host decoder; it needs a device state copy and chunked
prefill) is not started. The nsys reports are in `~/project/shared/tmp/mirage-l2c/` (outside the repo).

## Lane L6 - performance, round 2

Same hardware and rules as round 1 (GH200 gpu1, shared; rank by the minimum of three runs). The parity ladder (95
tests) passes on the CPU backend and CUDA after the changes below; decoder logits cosines 0.9999994 / 0.9999977, tokens
identical (236 and 168).

Step replay exists: `Gpu::begin_tape` / `end_tape` / `replay_tape` (gpu-core) turn a repeated submission into a CUDA
graph. The contract is that nothing varying is a dispatch parameter, so the decoder step now keeps its position in a
device word (`BatchState::posbuf`, advanced by a one-thread kernel at the end of the step); every kernel that used the
position reads it from that buffer (`mirage_tok_at/put`, `mirage_engram_rows`, `mirage_taps`, `mirage_row_write`,
`mirage_rope_pos`, attention scores/apply with `visible = u32::MAX`). The step is recorded once per generation and
replayed per token; recording declines the vendor library GEMM (`set_declines_library`), so a replayed step runs the
in-house f32 kernels (exact f32, not TF32).

| Change | batch-1 describe | batch-8 describe | batch-64 describe |
|---|---|---|---|
| round 1 end (eager steps, 16 per readback) | 2.40 s | 2.84 s | 5.87 s |
| step recorded once, replayed as a graph | 0.76 s | 1.15 s | 4.10 s |
| exponent-domain Sinkhorn, workgroup argmax, unrolled apply | 0.69 s | 1.06 s | 3.99 s |
| fused projections (lane maps 3 to 1, q/k/v+gate 4 to 1, MLP cond+rotation 2 to 1) | 0.65 s | 0.95 s | 3.08 s |
| four-lane Sinkhorn in registers | 0.58 s | | |
| reader batched over all text lines, parallel host assembly | 0.55 s | 0.83 s | 2.60 s |
| attention weighted sum split over key groups | 0.47 s | 0.76 s | 2.52 s |

Throughput curve (final; `describe_throughput_over_batch_size`, min of 3, TF32 off in the decoder):

| batch | wall s | images/s | vision | memory | decode | post (device part) |
|---|---|---|---|---|---|---|
| 1 | 0.47 | 2.14 | 0.03 | 0.00 | 0.43 | 0.01 (0.01) |
| 8 | 0.76 | 10.59 | 0.10 | 0.02 | 0.57 | 0.03 (0.03) |
| 32 | 1.31 | 24.48 | 0.25 | 0.01 | 0.82 | 0.09 (0.07) |
| 64 | 2.52 | 25.44 | 0.42 | 0.02 | 1.63 | 0.16 (0.13) |

Per decoder step (about 236 steps per sequence): 1.8 ms at batch 1, 2.4 ms at batch 8, 3.5 ms at batch 32, 6.9 ms at
batch 64. Against the engine (265-271 ms per image on CPU): batch 8 is 2.8x, batch 32 and 64 about 6.5x faster per image;
batch 1 (470 ms) is 1.8x slower. The batch-1 and batch-64 per-step targets (well under 2 ms) are met at batch 1
(1.8 ms) and not at batch 64.

Kernel profile after the changes (nsys with `--cuda-graph-trace=node`, all four batch sizes together, ms of kernel time):
`mirage_attn_apply` 1447 (13%, 29 us avg), `rmsnorm_rows` 1205 (11%, 231k launches, 5 us), in-house GEMV
`matmul_gemv_reg` 1098 + 520 + 207 (17%), `mirage_attn_scores` 707, `mirage_mhc_maps` 663 (26 us), softmax 301,
`mirage_taps` 234. Hypotheses: the replayed GEMMs run the in-house gemv kernels at 9-20 us where a 1 MB weight read is
under 1 us, so the per-step floor at large batch is those kernels' occupancy; the vendor GEMM is not capturable in a
graph, so the next candidate is an in-house register-tiled batched GEMM for M of 8 to 64 registered with the selector.
`mirage_attn_apply` is still latency bound (a flash-decode kernel over the cached keys would fuse scores, softmax and
apply into one launch). Not done this round: ncu roofline fractions for the top five kernels; device form scoring
(`extract_batch` still scores on the host decoder); cross-attention q + gate concatenation (the per-head `rmsnorm_rows`
needs a contiguous q).

## Lane L4 - serving

Done 2026-11-10. Model id `Cactus-Compute/mirage` (case-exact upstream name, a deliberate carve-out in
`crates/cli/tests/it/model_ids.rs`); user-facing page `docs/reference/models/mirage.md`.

| Item | State | Where |
|---|---|---|
| Capability `Provider`: actions `describe` and `extract`, typed params with ceilings, image blob via `capability::blob`, no param names a path (weights come from the store through `MirageSpec`), cheap `check_request` for the scheduler's door | done | `crates/mirage/src/caps.rs` |
| One session for every surface: request checks, grouping of equal options into device batches (at most 64), cancellation, usage accounting (`usage`: images, batch, decoder tokens, latency split) | done | `MirageSession::run_batch` |
| Resolver spec (`weights` role = the checkpoint directory; hand-placed files and a pulled manifest both classify by `model_type: "mirage"`) | done | `crates/mirage/src/resolver.rs` |
| Pull recipe: `config.json`, `mirage.safetensors`, `mirage.cact` fetched as published, role `weights` = `.` | done | `crates/modelstore/src/recipe.rs` (`mirage` row) |
| Residency adapter: one instance, `batch_limit` 64, `check_job` at the door, measured footprint, metrics (`mirage_batches`, `mirage_images`, `mirage_widest_batch`, `mirage_memory_hits`, cache size); catalog entry | done | `crates/catalog/src/resident_mirage.rs`, `lib.rs` |
| Decoder-memory reuse between `describe` and `extract`: key = image content hash + preprocessing version + weights revision + proposal options; byte-bounded LRU (256 MiB) owned by the served instance | done | `crates/mirage/src/memory_cache.rs` |
| Batch control in the model: `describe_batch_with` / `extract_batch_with` (decoder tokens, kept memories, cancellation between decoder windows), `ModelError::Cancelled` | done | `crates/mirage/src/device/batch_model.rs`, `decoder.rs` |
| SDK `PerceptionPipeline` (`describe`, `extract`, options builder, `models_dir`, typed `objects()` / `text_lines()`, `Usage`) | done | `crates/sdk/src/perception.rs` |
| CLI through the generic action path: `brain mirage describe\|extract`; `brain caps` lists both | done | `crates/cli/src/resolve.rs`, `resolver_cli.rs` |
| HTTP `/v1/run` and D-Bus `Run` through the generic routes, with security tests | done | `crates/catalog/tests/it/mirage_http.rs` |
| brain-py `describe` / `extract` (`Perception` = JSON dict + `.usage`), Python sample, Rust sample | done | `brain-py/brain_py/base.py`, `samples/python/vision/mirage-perception`, `samples/vision/perception` |
| Docs page and model index row | done | `docs/reference/models/mirage.md` |

Acceptance evidence (released weights, CPU backend unless stated):

- CLI JSON equals SDK JSON on the cats golden for `describe` (bitmap masks, 600 tokens) and `extract`:
  `crates/cli/tests/it/mirage_cli.rs`.
- HTTP `/v1/run` equals the session answer for both actions: `mirage_http::a_served_describe_and_extract_equal_the_session_answers`.
- CLI, HTTP, D-Bus and brain-py return the same JSON for one picture, for both actions, on CUDA gpu1 through a real
  `brain serve --dbus --openai` on a private bus: `tests/e2e/mirage.bats` (`MIRAGE_FLAGS="--device gpu1 --backend cuda"`).
- 16 requests in flight are one device batch whose answers equal their single-request answers: `mirage_served::sixteen_concurrent_requests_are_one_device_batch_and_equal_their_single_answers`
  (the scheduler's `load` shows 17 running in one group, `mirage_widest_batch` is 16) and, at the session,
  `serving::concurrent_requests_share_one_batch_and_keep_their_answers` (`BatchStats::sizes`). On CUDA over real
  D-Bus connections the 16 arrive as groups of 4 and 12 (connection timing); answers equal up to accelerator rounding.
- Cancellation frees the slot: `mirage_served::a_cancelled_request_takes_no_slot_and_the_rest_are_unaffected` and
  `serving::a_cancelled_request_fails_alone_and_does_not_join_the_batch`.
- Unload releases memory (CUDA gpu1, driver per-process figure): 564 MiB with the context alone, 948 MiB with the
  model loaded and a batch of 64 described, 570 MiB after the session is dropped:
  `serving::dropping_the_session_releases_device_memory`. Host: 1.26 GiB resident at that point.
- Reuse: an `extract` after a `describe` (or a second `extract`) of the same image skips the vision half and gives
  the same answer; other proposal options are another key: `serving::extract_reuses_the_memory_a_describe_left_and_gives_the_same_answer`.
- Merging pays on an accelerator: 16 requests merged take 2.0 s (8.1 images/s), served one by one 18.4 s (0.9 images/s)
  on CUDA gpu1, which another session's jobs were also using (`serving::a_merged_batch_is_faster_than_serving_the_same_requests_serially`).

Design notes:

- Batching is the executor's closed group (`Instance::run_batch`, `batch_limit` 64), not `model::serve::Scheduler`:
  the decoder is a lockstep device loop over a handful of hundred steps (no paged KV, hyper-connection lanes), so
  continuous admission mid-group would need a different decoder; a request arriving mid-group waits for the next one.
  Requests with different options are separate `describe_batch` calls of one group (the model takes one `Options`).
- `describe` keeps the decoder memory of each image (one device read per batch) and `extract` consumes it; `describe`
  never consumes one, because it also needs the mask maps and reader windows of the vision half.
- Cancellation granularity: before the group starts (no slot), and between decoder windows (16 steps); a group in
  which every request is cancelled stops costing steps. A request cancelled mid-window still pays that window.
- `extract` scoring is host work per image (see L2c "Open after L2c"); a form is bounded at 64 KiB, 32 fields and
  1024 candidate values so a request cannot ask for unbounded host decoder passes.
- The D-Bus server reads an input blob's `w`/`h` from the `in_meta` entry itself while the JSONL transport reads a
  nested `meta`; `brain_py` sends both for images. (Other Python samples that send only the nested form to the bus
  would fail the same way; not touched here.)

Open after L4: device form scoring (L6), Python/HTTP `describe` for a batch of images in one call (each request is one
image), box form fields (L2b), quantised `.cact` execution (L5), training (L3). `make test/e2e` does not run
`tests/e2e/mirage.bats` (it needs the checkpoint and `MIRAGE_IMAGE`); `tests/e2e/model_table_check.py` reports a
README section that no longer exists (pre-existing).

## Lane L3 - training

Host training of every task, 2026-11-10 (`crates/mirage/src/train/`). The objective is brain's own: the vendor published
no recipe (the config carries only `form_share`, `trace_share`, `name_loss`, `seg_loss`, `head_grad`, `heat_alpha`).
Device backward is not done (see Open).

Design decisions:
- One graph, two instantiations. The training graph (`train::net`, `train::vision`, `train::decoder`) is generic over the
  scalar (`train::real::Real`: `f32` trainer, `f64` finite-difference oracle) and built on a reverse-mode tape
  (`train::tape`) whose every primitive has a hand-written backward (`ops`, `spatial`, `seq`, `loss`). A model's backward
  is the chain rule over primitives that are each checked against central finite differences of their own forward, so
  the oracle shares no code with the analytic backward. The existing inference operators stay as they were (f32,
  parity-proven); the graph equals them stage by stage (cosine 1.0, test below), it does not replace them.
- Everything dense goes through `Net::proj`, the one place a LoRA adapter attaches. Parameters stay in checkpoint (Flax)
  layout, so a trained tensor is saved without a transpose and the HConvGLU bands and Hadamard rotations are trained
  as their own parameters (the unfolded form: no dense-matrix gradient mapping is needed on the host).
- Shared machinery reused: `optim::NamedMoments` + `Adam::update_slice` + `grad_multiplier` (host AdamW; a `restore`
  was added for exact resume), `model::adapter::{pair_file, KeyStyle}` for the adapter file, `import::export/load` for
  weights, `imaging` for decoding, `data::rng::Lcg` for adapter init. No Mirage-only optimizer, adapter or checkpoint
  format: a run directory is a model directory (`train::prepare::make_servable`), and `Mirage::open` folds an
  `adapter.safetensors` found beside the weights.
- Task-presence semantics (`train::data`): a label family is present when its key is present (an empty list is
  labelled-empty, a missing key unknown). Missing families carry no heatmap negatives; the stop token is supervised only
  for `closed` records; the first token of a section after an unlabelled one is not supervised (otherwise a missing E
  section would teach "no objects").
- `head_grad` is applied as an explicit `grad_scale` op on the encoder memory and the stem skips entering the dense head
  and mask branches; the decoder's cross-attention memory is not scaled (an interpretation: the config says only "the
  gradient the heads send back into the encoder").
- `seg_loss` was 0 in the released configuration; here the mask term is an explicit `TrainWeights::mask`.
- Residual-routing parameters (`stack/mhc_{a,b,phi}_res`): after 20 Sinkhorn rounds their gradient is below what a
  finite difference resolves (analytic about 1e-18 against an f64 difference floor of 1e-9). The Sinkhorn primitive is
  verified alone; the three tensors are effectively frozen by the architecture, not by a filter.

Gradient checks (f64, central differences, relative to the largest gradient entry; block gate 1e-4, model gate 1e-3):

| Level | Checked | Worst error |
|---|---|---|
| Primitives (`train_ops`) | add, mul, scale, bias_add, chan_mul, scalar_mul, sigmoid, silu, gelu, grad_scale (identity forward, scaled backward), linear, linear_nt, ZC norm, RMS unit, mask_rows, softmax, row_dot, row_mul, plan (gather/scatter/concat/slice/upsample/pool), permute, depthwise (strides 1x1, 2x2, 2x1, 1xk), dense conv (3 strides), sequency Walsh (self-adjoint, self-inverse), banded channel conv (dx and dband), GRN scale, 2D/1D RoPE, masked GQA self and cross attention, causal dilated taps, 20-round Sinkhorn, lane mixing (3 shapes), Kronecker rotation | 2.4e-9 (CTC), all others below 1e-9 |
| Losses | token CE with weights, CenterNet focal, weighted L1, BCE+Dice, CTC (also against a brute-force alignment sum) | 2.4e-9 |
| Whole model (`train_graph`, tiny config) | every tensor except the three residual-routing ones, through all five heads in one backward, nonzero gradient to every encoder tensor | 3.8e-4 (hadamard `cond_u`, gradient 1e-7) |
| Objective (`train_fit`) | every term live (heat, reg, mask, text, ctc), every tensor outside the decoder stack resolved | below 1e-3 |

Other tests: f32 graph equals the host operators stage by stage (stem, top, encoder, head_in, head, protos, coefs, decoder
memory with slots, reader logits, decoder logits; cosine >= 0.999999 each); full fine-tune overfits one tiny sample;
head-only changes only its heads; LoRA is an exact no-op at init (B = 0), leaves the base untouched, learns, and its
folded weights give the live adapter's loss; exact resume of weights, moments and adapter (bitwise) for full and LoRA;
the run directory loads through the importer; record schema refusals; unlabelled families never become targets.

Acceptance experiments (released weights, host f32 on 32 threads of a shared Grace node; `train_real.rs`, `#[ignore]`d,
run with `--ignored --nocapture`; 120 train and 40 held-out records, batch 4, 60 steps, AdamW lr 3e-4, clip 1):

| Experiment | Mode | Before | After |
|---|---|---|---|
| Detection (COCO subset, heat + size/offset terms) | head-only dense | AP50 0.2193, loss 3.0292 | AP50 0.2715, loss 2.6316 |
| Masks (COCO polygons, BCE + Dice) | head-only dense | IoU 0.6508, loss 0.7443 | IoU 0.6576, loss 0.7251 |
| OCR (synthetic lines, CTC) | head-only reader | CER 0.2943, CTC 1.7011 | CER 0.2639, CTC 1.4003 |
| LoRA on text CE, served | rank 8 on attention/slot/engram projections | held-out text loss 3.6492 | held-out text loss 1.6490; served-vs-training head cosine 1.00000000, decoder logits cosine 1.00000000 (max_abs 4.0e-4) |
| Exact resume on the released weights | head-only | - | weights after step 3 bitwise identical to the uninterrupted run |

The LoRA row used 60 train and 12 held-out records, 30 steps of batch 2 at lr 2e-3 (text term only, rank 8, alpha 16); the
mask gain is small (+0.007 IoU, loss -0.019) because the released mask branch is already good on the labelled boxes;
the text loss falls mostly by learning brain's line format (`S caption=...`, `E<n> <name> *<ref>`), which is not the
vendor's.

Training throughput (host f32, 32 threads of a shared Grace node, no GPU; samples/s over the whole run): detection
1.46, masks 1.19, OCR 5.63 (reader only: the encoder is skipped), LoRA text 0.80 (batch 2, encoder + decoder + adapters).

Not done in this lane:
- Device backward and CUDA training. The host graph is the trainer; the device forward kernels have no `*_bwd`
  counterparts for Walsh, banded channel conv, GRN, ZC norm, 2D RoPE, mHC + Sinkhorn, Hadamard MLP, engram, taps,
  slot construction, CTC. No host-vs-device backward cosine, no nsys profile of a training step. Next steps: write
  those backward kernels (the host primitives in `train/{ops,spatial,seq,loss}.rs` are the specification and the
  FD oracle), keep HConvGLU and the Hadamard MLP unfolded on the device for training (the folded dense matrices cost
  `n^2` parameters per mixer; mapping their gradients back is a contraction with the Walsh/band structure), then
  compare against `Net::<f32>` gradients at cosine >= 0.9999 on the CPU backend and one CUDA card.
- The capability action (`train` / `lora_train`): no other model in the tree exposes training as a served action (their
  fine-tunes run through the SDK study/fine-tune path), so none was added; a served training action would need a
  dataset blob contract and a long-running action contract that do not exist yet.
- Form fine-tunes: records carry no forms and `Extract` scoring runs on the host decoder; the form NLL gate is open.
- The `name_head`, MTP block and `mtp_*` tensors are training-only in the release and are carried through unchanged
  (no loss uses them); `name_loss`, `form_share` and `trace_share` are honoured only as the weights of trace/form
  tokens, which the prepared datasets do not contain.
- Host training memory keeps the whole tape (one image per sample); the saved-activation policy is not implemented
  (`scripts/gates/saved-policy.allow`, mirage entry, shrinks when the device trainer takes the policy).
- The two datasets are small (300 train records each); the held-out numbers are 40-record validation sets and show
  that the machinery learns, not what a full recipe would reach. The OCR set had boxes overhanging the image (a
  generator defect); `read_jsonl_clamped` is the explicit opt-in reader, the strict one refuses them.

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
| Training | full / head / LoRA, multi-task, gradient checks | host trainer done (L3); device backward open |
| Serving | capability, CLI, SDK, HTTP, D-Bus, batching | done (L4) |
| Quantised kernels | native low-bit CQ matmul from the packed weights | open |
| Performance | profile-driven batched throughput | open |
