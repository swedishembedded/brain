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
- Box fields (`format: box`, `x-source` a choice field) are generated like free text after the source (L7: the engine trace shows the prompt carries `name$60` for a box, the source only fixes the place); the earlier prompt tokenisation of `? ` carried a stray space (fixed in L7).
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

## Lane L6 - performance, round 3

Hardware: one GH200 (gpu1 of a shared node; at the time of measurement the other session held 89 GB of the card and the
card was idle, 0 % utilisation), crate test binary, released weights, minimum of three replays.

### Step 1: the regression hunt (no regression)

L6 round 2 (mixed cats + text test, `describe_throughput_over_batch_size`) measured 2.14 / 10.59 / 24.48 / 25.44 images/s at
batch 1 / 8 / 32 / 64; lane L5 (`describe_throughput_packed_against_fp32`: cats only, no bitmap masks) measured 0.89 / 5.5 / 15.0 for
FP32. Both tests re-run at HEAD (e826f1f57) on the idle-utilisation card: mixed test 2.14 / 11.12 / 18.44 / 26.63 (batch 32 carried
a 0.75 s vision outlier, repeat 24.5), cats test FP32 2.19 / 11.18 / 28.59 at batch 1 / 8 / 64 and packed CQ 2.29 / 8.94 / 29.64. The
device sources changed between the last round-2 commit (44b6a4ff6) and HEAD only in `batch_model.rs` (serving hooks), `mod.rs`,
`decoder.rs` (cancel and memory hooks), `reader.rs` and a one-line `vision.rs`; the kernels did not change. Conclusion: the L5 numbers
were taken while the shared card was busy with the other session's jobs (the same code gives the round-2 numbers again). Hypothesis
"something regressed" killed with the measurement; no bisect needed.

### Step 2: resident memory (structured mixers, CQ row gather)

The HConvGLU mixers and the Hadamard MLPs no longer exist as dense `n^2` matrices. `WeightForm::Structured` (the default; `Folded` is
kept as the gated candidate and the parity oracle, `Device::new_with`, `MirageModel::new_with`) runs:
- `mirage_kron_rows`: one operator for the sequency Walsh transform (Sylvester +-1 factors 16 x 32, permutation bit-reversed Gray
  code, scale 1/sqrt(n)) and the three Hadamard rotations (learned `a`, `b` factors, permutations, diagonals, the residual add fused into
  the last one): a 512-thread workgroup per row, the 32-term stage into workgroup memory, one barrier (so the CPU JIT runs it), the
  16-term stage with permutation, scale and accumulate;
- `mirage_hconv_band_glu` (gate and value 65-tap bands with bias and gated product in one launch) and `mirage_band_cols` (output band),
  tap-major weights;
- `mirage_hada_mid` reads logits and z as two strided inputs; `mirage_cq_gather`: a row gather straight from a packed CQ matrix (workgroup
  per output row and 128-lane group, decode into workgroup memory, one barrier, 128-term signed sum), used for the tied embedding (it
  shares the logits matrix: one resident copy) and for the two engram tables (37.7 MB dense each, 2.5 MB packed).

Parity: the structured and folded forms both match the host mixer (cosine 1.0 / 0.9999999, max_abs 6.6e-7 structured against 1.5e-4
folded on body mixers: the structured form is exact f32, the folded form runs TF32 GEMMs), the Walsh kernel matches the host transform at
widths 2..512, the packed gather matches the host dequantiser bit for bit at 2/3/4 bits including repeats and the absent marker,
`a_packed_model_describes_like_the_same_model_dequantised` still equals the dequantised model, `structured_and_folded_mixers_describe_alike`
and the two decoder tests (teacher-forced logits cosines 0.9999994 / 0.9999977, tokens identical) run both forms. CPU backend: every test
module passes (run per module; running the whole binary at once on the CPU backend aborts the process under the load of 190 parallel
JIT tests, a pre-existing resource effect, not a failure of any test).

Resident device bytes, measured as the backend's live-allocation delta around the upload (`MirageModel::resident_bytes`):

| model | before | after |
|---|---|---|
| FP32 safetensors | 307.5 MB | 205.7 MB (all parameters, no folds) |
| packed CQ archive | 117.2 MB | 29.8 MB (13.8 packed + 11.5 band taps and factors in f32 + small tensors; archive file 21.6 MB) |

Throughput (same card state, min of 3; mixed test then cats test):

| batch | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| FP32 before (mixed) | 2.14 | 11.12 | 24.5 | 26.63 |
| FP32 after (mixed) | 1.98 | 10.87 | 25.09 | 27.76 |
| FP32 before (cats) | 2.19 | 11.18 | | 28.59 |
| FP32 after (cats) | 1.86 | 10.59 | | 26.2 - 29.5 |
| CQ before (cats) | 2.29 | 8.94 | | 29.64 |
| CQ after (cats) | 2.27 | 8.30 | | 30.14 |

Selection: encoder plus dense head of two images, structured against folded: 0.049 s against 0.051 s (CUDA), 1.52 s against 2.14 s (CPU
backend), so the structured form is the resident default. The Hadamard MLP is the exception at batch 1: the structured form adds
launches (logits GEMV, three rotation kernels, middle stage against two GEMV-class products) and costs about 0.04 s of the 0.42 s
batch-1 decode (step 1.8 ms to 2.0 ms) while winning at batch 64 (decode 1.64 s to 1.49 s); it stays the default for the 94 MB it saves.
nsys (FP32 batch 1 and 64, `fp32_describe_for_profiling`): `matmul_f32_tiled` (the native CUDA 64 x 64 tile GEMM, M = 64) is 39 % of
kernel time at 47-63 us per launch for 1 MB weights (8 workgroups for a 512-row weight; roofline is about 1 us), `mirage_kron_rows`
16-18 us per launch with 128 threads before the 512-thread version, `mirage_attn_apply` 11.7 %, `rmsnorm_rows` 9.5 % (89k launches).
Remaining (step 3 work): small-batch GEMM spread over the card, rmsnorm fusion, flash-decode attention, mhc_maps.

### Step 1 addendum: the card state explains L5's numbers

While measuring step 3 the other session's jobs took gpu1 to 99 % utilisation (87 GB held): the same binary then gave FP32 0.98 /
2.57 / 15.1 images/s and packed CQ 1.08 / 4.29 / 18.3 at batch 1 / 8 / 64 (decode 0.86 s at batch 1 against 0.42 s idle), the
numbers lane L5 recorded. No code regression exists; throughput figures are only comparable at equal card load, so every table in this
round states it (idle-card numbers above).

### Step 3 (decoder step floor): one hypothesis killed, the rest not started

Hypothesis: an in-house f32 product with one thread per activation row and four weight rows per workgroup (grid N/4 x M/64, scalar
then vec4 loads, `Device::linear` inside a replayed step) beats the 64 x 64 tile GEMM (`matmul_f32_tiled`, 39 % of kernel time, 8
workgroups for a 512-row weight). Measured under load (min of 30 replays per product, 64 x 512 x 512): 58 us scalar, 73 us vec4, against
the selector 47-63 us in the replayed step; and the library GEMM 6-22 us outside a replay. Killed: the kernel was not committed. A
useful replacement needs shared-memory staging of the activation tile and split-K for the 8-workgroup weights (a real register tile
with ~100 workgroups); roofline for that product is about 1 us. Still open from the nsys table: `matmul_f32_tiled` 39 %,
`mirage_attn_apply` 11.7 %, `rmsnorm_rows` 9.5 % (fuse with the following projection or residual), `mirage_mhc_maps` 5.3 %, in-house
`mirage_kron_rows` 11 % (a fused one-launch MLP would remove two launches per layer on CUDA). Targets (>30 images/s at batch 64,
step < 2 ms at batch 1) are not met: 27.8 images/s and 1.9-2.0 ms on the idle card.

### Step 4: serving test on CUDA

`concurrent_requests_share_one_batch_and_keep_their_answers` failed on CUDA only because it compared a merged batch with the
single-request answers bit for bit, while the GEMM selector picks other kernels (TF32 library products against in-house tiles) for other
row counts, so scores and the odd threshold mask pixel differ in the last places. Decision: the serving contract promises the same
structure and values to rounding, not bit identity across batch sizes; forcing one kernel for every row count would give up the
library's tensor-core products for batched work. The test now asserts exact equality on the CPU backend and, on accelerators, equal
structure (every box, name, description and text) with scores dropped and masks reduced to their foreground size.

### Step 5: precision on the serving surface and the quantize verb

`describe` and `extract` take `precision` (`fp32` | `cq`); omitted, the directory decides as before (safetensors when present). The
provider keeps one session per (directory, weight source) and replaces it when the other precision is asked for; `precision` is
not a batching option (it never joins a batch group). `brain mirage quantize --out DIR [--weights DIR]` calls
`mirage::model::quantize_dir` (FP32 checkpoint with its adapter folded, plan from the directory's `mirage.cact`, atomic write, config
copied); the CLI re-quantisation of the released weights differs from the vendor's file only in the 0.6 % of index bytes recorded in
L5. Tests: `precision_picks_the_weights_and_is_not_a_batching_option`, `quantize_dir_writes_a_loadable_archive`. The residency
adapter keys one instance per weight source (`default` | `fp32` | `cq`; L7 item 3): a `precision` request is served by an instance
loaded with exactly those weights. Tests: `the_precision_option_selects_the_instance_and_its_weights`,
`a_cq_request_is_served_by_the_packed_archive_beside_the_fp32_instance`.

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

Device training (lane L3b, in progress): `device::train::DevTape` implements `train::graph::Graph<f32>` over device buffers with
parameters resident in a `paramstore::ParamStore` (a stacked decoder layer is one entry `name#layer`; adapters are registered
pairs). Steps are recorded and submitted lazily (readback, `backward`); backward kernels accumulate with one owner thread per
element (`mirage_t_*`, no atomics); values carry `needs_grad`, so frozen parameters and constant inputs record no backward.
Packages A (core, elementwise, linear, norms, plans/gather/concat/slice, softmax, row ops, loss sum), B (depthwise and dense
convolutions, upsample, Walsh as a GEMM against its matrix, banded channel convolution with dx and dband, response norm), C (rope,
masked grouped-query attention, causal taps, Sinkhorn, lane mixing, Kronecker rotations, column permutation) and D (token
cross-entropy, focal heat, L1, BCE+Dice, CTC) are checked against the host tape in `device_train` (cosine >= 0.99999, relative
error <= 1e-4, losses 1e-5 in value; CPU backend). The whole tiny-config graph (`Net::resident` over `DevTape`, parameters in a
`ParamStore` via `device::train::params`, decoder layers as `name#layer`) gives the host gradients at group cosine 1.0000 for every
parameter group (encoder stem, bodies, tops, dense head, mask branches, reader, decoder layers, embedding); only the residual
routing tensors (`stack/mhc_a_res`, `mhc_phi_res`, peak gradient ~1e-8) differ in direction, below f32 resolution as on the host.
Untuned: attention keeps the probabilities (`tq*hq*tk` values) for the backward; channel, bias, band-weight and kernel-gradient
reductions loop over rows in one thread per output; CTC and Sinkhorn run one thread per sequence/block.

Device trainer (lane L3b, package E): `device::train::trainer::DevTrainer` takes the host trainer's `TrainConfig`/`Mode`
(full, head-only, LoRA) on any `Device`. Parameters live in a `ParamStore` (frozen tensors are `Role::Frozen`: no gradient or
moment buffer, and a frozen encoder records no backward), gradients accumulate over the batch's images, and the shared device
`optim::Optim` applies AdamW with the global clip. A run directory is the host trainer's (`mirage.safetensors` in checkpoint layout,
`adapter.safetensors`, `state.safetensors` with the moments by checkpoint name), so a run resumes on either evaluator, and the
HConvGLU bands and Hadamard rotations are saved unfolded and folded when the model opens (training never touches the folded
matrices). Both trainers implement `train::trainer::FineTuner`. Tests: `device_trainer` (steps follow the host trainer in every mode:
loss to 1e-4, weight-update cosine >= 0.999 on the CPU backend; exact resume for full and LoRA; a device run directory resumes on
the host; LoRA no-op at init, learns, merge equals live).

CUDA (gpu1 of a shared node, `BRAIN_BACKEND=cuda BRAIN_DEVICE=gpu1`): `device_train` and `device_trainer` pass (39 tests) with the
bounds widened to cosine 0.9999 / 2e-3 because the selector picks tensor-core GEMMs. Whole tiny graph, group cosine of the
parameter gradients against `Net::<f32>`: stem 0.999996, body_0 0.9999986, body_1 0.9999985, top_0 0.9999986, top_1 0.9999987,
dense 0.9999990, reader 0.9999998, decoder layers 0.9999994, embedding 0.99999975, engrams 0.9999997 (the residual routing
tensors, peak gradient ~1e-8, are below f32 resolution as on the host).

Acceptance experiments on the device (CUDA gpu1, same configs and data as the host runs; the OCR subset was regenerated after the
generator fix, so its host numbers were re-measured; host = 32 threads of a shared Grace node):

| Experiment | Before | After (device) | After (host) | samples/s device / host |
|---|---|---|---|---|
| Detection, head-only dense | AP50 0.2193, loss 3.0282 | AP50 0.2765, loss 2.6377 | AP50 0.2715, loss 2.6316 | 6.90 / 1.46 |
| Masks, head-only dense | IoU 0.6508, loss 0.7446 | IoU 0.6575, loss 0.7254 | IoU 0.6576, loss 0.7251 | 3.52 / 1.19 |
| OCR, head-only reader | CER 0.2814, CTC 1.6543 | CER 0.2472, CTC 1.3549 | CER 0.2472, CTC 1.3548 | 13.7 / 7.65 |
| LoRA text, rank 8 | loss 3.6489 | loss 1.6474; served-vs-training head cosine 1.00000000, logits 1.00000000 (max_abs 6.0e-4) | loss 1.6490 | 1.76 / 0.80 |
| Exact resume on the released weights (device) | - | weights after step 3 bitwise identical: true | - | - |

(The detection and mask host columns are the L3 measurements; the OCR host column was rerun on the regenerated subset.)

First nsys profile of a full-model device step (released weights, batch 2, CUDA gpu1, untuned kernels): 3326 ms of kernel time over
three steps, 1.11 samples/s. Top rows: `mirage_t_col_red` 32.5% (1332 launches), `dwconv_dk` 14.4%, `rms_dscale` 12.5%, `conv_dk`
10.1%, `attn_ds` 8.5%, shared `matmul_dw` 6.1%, `attn_probs` 3.8%. Fix: the one-thread-per-output reductions became tiled
workgroup reductions (a workgroup owns 8 adjacent columns, 8 row lanes, one barrier; narrow periods use all 64 threads as row
lanes) and the attention probability and score-gradient kernels run one workgroup per (query, head) row. After: 971 ms kernel
time, 1.87 samples/s; remaining top rows `matmul_dw` 20.6%, `col_red` 13.0%, `rms_dscale` 8.2%, `dwconv_dk` 7.9%, `conv_dk` 5.5%.
Open: attention still stores the probabilities (`tq*hq*tk`); the saved-activation policies (recompute, narrow boundaries) are not
implemented for the Mirage graph, so `scripts/gates/saved-policy.allow` keeps its `mirage` entry with an updated reason.

Not done in this lane:
- (Done, see above.) Device backward and CUDA training. The host graph is the trainer; the device forward kernels have no `*_bwd`
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
  generator defect); the generator is fixed and the strict reader is the only reader.

## L5: native low-bit execution of the deployment archive

Generic tier `Dtype::Cq2/Cq3/Cq4` (`Weight::Cq`, host codec `model::cq`): groups of 128 input lanes, a Lloyd-Max codebook, one
fp16 norm per group, the group rotated by the natural-order orthonormal Walsh matrix, so `y = sum_g norm_g * (codebook[idx_g] . (H x_g))`.
The `.cact` bytes upload unchanged (index words, norms two to a word, codebook levels as f32). Kernels in the shared catalogue:
`cq_rotate` (barrier-free, one thread per group) and `cq_rotate_wg` (a workgroup per group), `matmul_cq_gemv` (16 lanes per weight row,
four rows per workgroup, up to eight activation rows per workgroup, more rows as more workgroups), `matmul_cq_gemm` (barrier-free, one
thread per weight row and block of eight activation rows; the CPU backend and the fallback), `cq_dequant_rotated` (the weight in the
rotated domain as dense f32, for the dense GEMM at many rows), and a `weight_decode` arm per width (the dequantiser of tests,
training and `read_host`: 128 terms per element, not an inference path). Selection is the shared selector (`select::candidates`,
`Op::MatMul`, Cq dtypes): the cooperative product up to 96 rows where the device has workgroup barriers, rotated dequantisation plus
the dense GEMM above (crossover measured by `cq_tier::cq_crossover`, `--ignored`), the barrier-free kernel everywhere else. The native
autotuner (`native_tune`) is not wired to this tier: the choice is the selector's rule, not a timed probe. Coverage gate
(`tier-gemm.map`): GEMV `matmul_cq_gemv`, batched path `library` (scratch tile plus vendor GEMM), `fused` listed as a gap.

Kernel parity against the dequantised f32 matmul, worst error as a fraction of the root of the summed squared terms: 3e-7 to 9e-7
(cooperative and barrier-free kernels, every width, 1 to 33 rows, k = 200 and 256) and 1.2e-6 (dense-GEMM path, 100 rows), CUDA gpu1;
the CPU backend holds the 2e-5 bound (`model/tests/it/cq_tier.rs`; the tile decode equals the host dequantiser to 1e-5).

Mirage on the archive: `WeightSource::Cact` (`Mirage::open_with`, `MirageSession::load_with`, `mirage::model::load_dir`) keeps every
CQ matrix a `Linear` reads packed on the device (`Linear::cq`, `Checkpoint::packed`); the default stays FP32 safetensors when it exists,
the archive is the default only when it is all the directory holds, and an adapter refuses the archive. Resident matrix weights of the
released model: 8.75 MB packed (projections, routing matrices, tied embedding as the logits matrix) next to 132.5 MB dense, against
229.7 MB dense for the same model dequantised. The dense remainder is: the folded HConvGLU matrices (88 MB), the Hadamard-MLP
rotations and `cond_rot` (about 50 MB), the embedding as a gather table (17 MB) and the convolution kernels the archive stores as CQ
(the file holds 13.76 MB of CQ matrices: 8.75 MB are `Linear`s kept packed, the other 5.0 MB are convolutions expanded to dense). Done in L6 round 3: the CQ row gather (embedding and engram tables) and the HConvGLU/Hadamard folds replaced by structured kernels;
open: the convolution kernels on CQ.

Agreement with the reference engine on the recorded corpus (9 describe cases, 42 boxes; CUDA gpu1; the engine rounds activations to
int8, brain keeps them f32, so exact tokens are not expected): packed CQ model 40 of 42 boxes exact, 42 of 42 at IoU 0.5, 39 of 42
identical name/position/description lines, 4 of 9 documents identical apart from masks and scores. The FP32 model on the same cases:
0 exact, 29 of 42 at IoU 0.5, 8 identical lines (`device_cact_accept::the_packed_model_agrees_with_the_reference_engine`). The A8 emulation of
the engine's activation rounding is the opt-in `cq-a8` precision (L7 item 5).

Quality of the archive against FP32 (the 2-bit quantisation itself, teacher forced on the cats and text goldens, host path):
top-1 agreement of the decoder logits 216/237 and 148/169 positions, mean KL(FP32 || CQ) 0.107 and 0.162 nats, next-token accuracy
215/237 and 147/169 for CQ against 236/237 and 168/169 for FP32. Proposals: of the FP32 model's boxes the packed model has 3 of 4
(cats) and 7 of 7 (text) at IoU 0.5 and no exact one (`quality_against_fp32_on_the_goldens`, `proposals_of_the_packed_model_cover_the_fp32_proposals`).

Throughput of `describe_batch`, cats 640x480, best of three, CUDA gpu1 shared with another session's jobs: FP32 0.89 / 5.5 / 15.0 images/s
at batch 1 / 8 / 64; packed CQ 1.08 / 4.35 / 16.0. Before the cooperative rotation kernel the packed model ran 0.36 / 2.3 / 5.8:
one thread per activation group left the device idle through a serial 7 x 64 butterfly in every decode-step linear
(`describe_throughput_packed_against_fp32`). ncu of `matmul_cq_gemv` (CQ2, 2048 x 512, one row): 8.5 us, DRAM 1.3 % of peak, grid
0.12 waves: the 262 KB weight is L2 resident and the kernel is launch- and latency-bound, so the roofline at Mirage's sizes is the
launch floor, not memory; a bandwidth-bound shape needs a taller weight than any Mirage projection.

Quantising (`CactModel::quantize`, `CactBuilder`): the container writer rebuilds the released `mirage.cact` byte for byte (header,
codebooks, 44-byte records, 64-byte blob alignment, no padding after the last blob). A group's stored norm is the L2 norm of the
rotated group and a lane's index is the nearest codebook level of `u / norm` (the codebooks are on the unit sphere of a group); with
that rule, quantising the released FP32 weights against the released archive's plan writes a file of the same size whose 163 CQ
entries agree with the vendor's on 99.4 % of the index bytes and 99.9 % of the norm bytes (differences are ties and fp16 rounding)
and whose other 643 entries are identical. The reference engine's harness loads the re-quantised file and describes the cats photograph
(scene, tags, two tabby cats, boxes within a pixel of the original archive's). The plan (which entries are CQ at which width, the
codebooks, the derived entries and the tokenizer) is taken from a template archive, because `config.json` records the widths per tensor
family but not per entry; a `brain mirage quantize` verb is not wired (the library function and its tests are the surface).

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

## Lane L7 - close-out

### Item 1: the full test binary (done 2026-11-10)

The "abort after ~157 tests" was `gpu_core::testgpu::exclusive()` hard-exiting with code 101 after a WAITER had queued for
600 s behind healthy holders (65 tests of the binary take the lock and many run for minutes on the CPU backend). The limit
measured the queue, not a wedged holder. `exclusive()` now returns a `DeviceGuard` and aborts only when the CURRENT holder has
held the lock for 600 s (acquisition instant tracked in an atomic; the FATAL line names the waiting test and the holder's hold
time). Gate: `testgpu::tests::the_device_lock_limit_measures_the_holder_not_the_queue` (four waiters behind 150 ms holds with a
300 ms limit never abort; a holder past the limit is reported; a poisoned lock is free).
Evidence: `make test/one P=brain-mirage T=it` in one process: CPU backend 181 passed, 15 ignored, 903 s on the final tree (an earlier run, 178 passed in 853.9 s, had one failure
was the engine-corpus test picking up the new `cats_form_*` cases, fixed); CUDA gpu1 (`BRAIN_BACKEND=cuda BRAIN_DEVICE=gpu1`,
card shared with another session): 180 passed, 15 ignored, 103.5 s (before the device form test was added). Tests that only need a device keep `exclusive()`: they build
their own `Gpu` from a kernel list that is not `'static`, so `testgpu::dev` sharing does not apply without a larger test refactor.

### Item 3: residency adapter precision (done)

`resident_mirage` keys one instance per weight source (`default` | `fp32` | `cq`) and loads the archive for `cq`.
Tests: `the_precision_option_selects_the_instance_and_its_weights`, `a_cq_request_is_served_by_the_packed_archive_beside_the_fp32_instance`.

### Item 2: device form scoring and box fields (done)

Traced the released engine with gdb (token streams of `forward_chunk`). Facts, all now implemented and tested:
- A box field is asked exactly like free text: the prompt carries `name$60` (length token "60"; the default `maxLength` of a text
  field is 60 as well, not 32), the answer is generated greedily after `name=` up to the terminator and reported as
  `{"value": text, "probability": product of token probabilities}`. `x-source` is not in the prompt; it only fixes the place of the
  box (after the committed answer of that choice field). Text and box answers have `probability`, not a candidate table.
- The prompt was tokenised with a stray space before `?` (token 7915 instead of 8142); fixed (`tok.encode(spec)` as is).
- Engine agreement (host model on the packed weights, `cats_form*` cases recorded with the vendor harness):
  `text_and_box_fields_answer_like_the_reference_engine`: committed values equal, every probability within 0.08 (choice/count: the
  shift measured is up to 0.07; the box value `cat` has probability 0.265 against the engine's 0.280). The engine also rounds
  activations to 8 bits (item 5), which explains the residual. The free-text case (`what`) is a coin flip between the first word of
  the question (0.39) and the abstention (0.32; engine 0.51): only its shape is held against the engine.
- Device scoring (`device/form.rs`): `DeviceDecoder::extract_forms`. The images answering the same form are one device batch
  (cohort). A field's candidates are scored by rewinding the batch state to the end of the field head, teacher-forcing each candidate
  and reading the logits of each step; prefix state computed once per field. When the committed answers of the cohort differ in token
  length the cohort splits per length (state rows copied through the host, `select_state`; cross-attention rebuilt for the subset);
  free text and box values are generated per image. Host and device share `form::{candidates, settle, Terminators, log_softmax}`.
  Test `a_device_batch_scores_every_field_kind_as_the_host_does` (two images, two forms, all four kinds): values equal, probabilities
  within 1e-4 on CPU and 3e-2 tolerance on accelerators; passes on CPU and CUDA gpu1.
- Not batched across candidates: the decoder steps one token per sequence at a position the whole batch shares and the
  cross-attention indexes memory by batch row, so candidates run one after the other on the rewound state (images are the batch).
  A candidate-batched variant needs a per-row position or a cross-batch index in the attention kernels.

### Item 5: A8 activation precision (done 2026-11-10, opt-in `cq-a8`)

What the vendor engine does, read from its binary and checked against its running state (gdb on the read-only harness,
nothing written): every CQ product (`cq_matmul_batch`, `quantize_activation[_batch]`, `cq_row*_dp`, `cq_rows4_*_dp`) rotates each
128-lane group of the activation row (natural-order Walsh, times 1/sqrt(128)), takes the group's absmax, sets
`scale = absmax * 0.007874016` (the f32 nearest 1/127, a multiply, bits 0x3c010204; scale 1 for an all-zero group), quantises with
`fcvtas` (round half AWAY from zero, scale reciprocal then multiply) saturating to int8, and multiplies int8 against an int8
codebook table (`sdot`): `table_scale = max|level| / 127` (f32), `table[i] = round_away(level[i] / table_scale)`. Read from the live engine:
2-bit [-127,-38,38,127], 3-bit [-127,-79,-44,-14,15,45,79,127], 4-bit [-126,-95,-74,-57,-43,-30,-17,-5,7,19,31,45,59,76,97,127], scales
0.0010508697 / 0.0014946931 / 0.0019076669. A group's `i32` dot (at most 128*127*127, exact in f32) is turned into
`((table_scale * norm) * f32(dot)) * act_scale` and the groups are added in order (the engine's single-row and 4-row kernels
associate the three factors differently and fuse the last add; that is f32 rounding, not a rule). The training-side fake quantiser in the
vendor's python (per whole row, not per group, half-even) is not what the engine runs.

Implementation (shared tier, nothing Mirage specific): `model::cq::{CqActivations, a8_table, a8_quantize_group, Cq::matmul_a8}` (host
oracle), kernels `cq_rotate_a8` / `cq_rotate_a8_wg` (rotation + quantisation, int8 lanes four to a word, scales per group),
`matmul_cq_a8_gemm` (barrier-free, CPU backend and fallback), `matmul_cq_a8_gemv` (cooperative: 16 threads a row each take a 32-lane
quarter of four groups at a time and add the quarter dots as integers in workgroup memory), the codebook table built inside the kernels
from the stored levels. `Ops::set_cq_activations` (default F32); `Ops::cq_product` picks the A8 variants on the same selector rule
(cooperative up to 96 rows where barriers exist, barrier-free above and on the CPU; there is no dense A8 product, the many-row regime does
not expand the weight). A model-specific flag was not added (`LinearActivations` is the NVFP4 knob and would have silently ignored `int8`).
Mirage: `model::Precision {Fp32, Cq, CqA8}` (`cq-a8` on the `precision` option of describe and extract, the residency adapter's instance
key, `MirageSession::load_with`, `Mirage::open_precision`, `Checkpoint::cq_activations` for the host `Linear`s); the CLI takes it through the
capability's enum. Never a default.

Measured (kernel tests in `model/tests/it/cq_tier.rs`, Mirage in `mirage/tests/it/device_cact_accept.rs` and `pipeline.rs`):
- Kernel parity against the exact host integer emulation, widths 2/3/4, 1 to 33 rows, k 256 and 700: rotation kernels bit-exact (bytes and scales,
  both variants, CPU backend and CUDA gpu1); `matmul_cq_a8_gemm` bit-exact on the CPU backend and on CUDA; `matmul_cq_a8_gemv` bit-exact
  up to four groups per row and 1.1e-7 to 1.2e-7 of the largest output beyond (the partial sums are added in four slots), CUDA gpu1. Through `Ops::matmul`
  (m = 1, 4, 9, 100) within 4e-6.
- The emulation against the engine itself: a product recorded from the engine (260 activation rows, k = 256, 512 weight rows of a decoder
  projection) is reproduced to 7e-8 to 2.7e-7 of the largest output by the numpy form of the rule and 1.4e-8 by `Cq::matmul_a8`, the f32-activation
  product is 2.4e-3 to 7.9e-3 away. A 2 x 8 slice is the tracked fixture `tests/golden/vendor_cq_a8_matmul.json`
  (`the_a8_product_reproduces_a_recorded_engine_product`).
- Decoder logits teacher forced on the goldens (host path), against FP32 (cats 237 positions / text 169): top-1 agreement CQ f32-act 216 / 148, CQ A8
  215 / 148; mean KL(FP32 || model) CQ f32-act 0.10676 / 0.16194, CQ A8 0.10704 / 0.16227; A8 against CQ f32-act: top-1 236/237 and 168/169, KL 0.00004 /
  0.00006; next-token accuracy FP32 236 / 168, CQ 215 / 147, CQ A8 214 / 147. Int8 activations cost nothing measurable next to the 2-bit weights.
- Vendor agreement on the engine corpus (9 describe cases, 42 boxes, CUDA gpu1): CQ f32 activations 40 exact, 42 at IoU 0.5, 39 identical lines,
  4 identical documents (apart from masks and scores); CQ A8 34 exact, 42 at IoU 0.5, 39 identical lines, 1 identical document. A8 is NOT closer
  to the engine's boxes. Reason (analysis, not yet bisected): the engine rounds the activations of every product; brain's A8 covers only the products
  over packed `Linear`s, while the packed convolutions (5.0 MB, dense f32 here), the query and the keys and values (the vendor python quantises q and
  k, v to int8 per head, `kv_bits` in the archive header) stay f32, so the rounding errors are independent of the engine's instead of identical. The
  one-matmul check above is exact, so the arithmetic of a product is right; what remains is coverage.
- Form probabilities (host, packed weights; cats_form, cats_form_box, cats_form_text): largest probability difference from the engine 0.0685 with
  f32 activations, 0.0693 with A8 (committed values equal in both); `PROBABILITY_TOLERANCE` stays 0.08. The free-text case stays the coin flip
  described in item 3.
- Throughput of `describe_batch`, cats 640x480, best of three, CUDA gpu1 (shared card, idle during the run): images/s at batch 1 / 8 / 64: FP32
  1.85 / 11.28 / 29.49, packed CQ 2.44 / 8.77 / 31.36, packed CQ A8 2.40 / 8.49 / 32.13. A8 costs no throughput after the cooperative kernel
  was widened (a first version with four threads a row and whole groups per thread was 1.2x / 2.3x / 1.4x slower at 1 / 8 / 64: 16 threads a row
  keep the 128 workgroups of the f32 GEMV busy). The FP32 batch-1 figure varies 0.9 to 1.9 between runs on this shared card.

Not done, in order: (1) run the packed convolutions and the Hadamard/structured mixers through the same A8 products (im2col rows against the packed
kernel; today they are dense f32) and add the engine's per-head int8 query/key/value rounding to the attention kernels, then re-measure exact
boxes against the corpus; bisect with the existing engine dumps (`dump*.py`) which stage first diverges; (2) tighten `PROBABILITY_TOLERANCE` only if (1)
brings the form probabilities closer; (3) ncu roofline of the A8 kernels (not profiled; launch bound at Mirage sizes like the f32 ones).

### Remaining (not done in L7)

- Item 4, saved-activation policy. Findings so far: the host trainer is an eager whole-tape (`train/tape.rs`: `Tape::op` stores every
  value and a boxed backward closure; `backward` walks all nodes), the device trainer (`device/train/*`) records the same graph on
  the device. Steps: (1) a checkpointed block API on the host tape (run a block's forward on a scratch tape, keep only its boundary
  value, rebuild the block's tape inside the backward from the stored boundary and run its backward closures, then drop it), per
  decoder block and per vision stage/body block; (2) the same for the device tape (boundary buffers kept, block steps replayed);
  (3) attention backward recomputing the probabilities per query tile instead of storing `tq*hq*tk` (the `masked_grouped_query_attention`
  and vision attention ops in `train/ops.rs` / `device/train/seq.rs`); (4) parity: the existing `device_trainer`/`train_graph` gradient
  gates must hold under `Recompute` bit-for-bit on the CPU backend (recompute is deterministic); (5) delete the `mirage` line of
  `scripts/gates/saved-policy.allow`; (6) peak memory before/after with the process-level device counter used by
  `dropping_the_session_releases_device_memory`.
- Item 6, step floor: GEMM selection at M 1-64 in the shared layer (`matmul_f32_tiled` at 8 workgroups for M=64), fused rmsnorm,
  flash-decode attention, `mhc_maps` workgroup kernel, fewer launches in the structured Hadamard MLP; measure on an idle card only
  (gpu1 carries another session's load now: 87 GB used), min over replays, ncu roofline for the top 5 kernels.

## Open lanes

| Lane | Scope | State |
|---|---|---|
| Vision encoder forward | stem, body, attention top, level memory, dense head peaks | open |
| Decoder forward | lane stack, engrams, cross-attention, Hadamard MLP, logits | open |
| Heads | boxes, masks, slots, enum scoring | open |
| OCR | the reader and greedy collapse | open |
| Training | full / head / LoRA, multi-task, gradient checks | host trainer done (L3); device backward open |
| Serving | capability, CLI, SDK, HTTP, D-Bus, batching | done (L4) |
| Quantised kernels | native low-bit CQ matmul from the packed weights | done (L5) and the opt-in A8 mode (L7 item 5), see above |
| Performance | profile-driven batched throughput | open |

### Item 4: saved-activation policy and recomputing attention - done (2c04421fc, 521622a18, 9fbad69bf)

Recompute blocks: stem stages, body blocks, top layers, dense head, mask branches, reader stages and mixers, engram
sites, decoder layers. Gradients under `Recompute` equal `Keep` bit for bit on the CPU backend; the `mirage` line left
`scripts/gates/saved-policy.allow`. Attention backward rebuilds probabilities per row from q, k and the saved
log-sum-exp (no `tq*hq*tk` matrix).

One training step, full model, released weights, batch 2 (`train_real::{host,device}_step_memory_{keep,recompute}`,
CUDA gpu1 idle, 2026-11-10):

| trainer | policy | peak memory | step | samples/s |
|---|---|---|---|---|
| device (CUDA) | keep | device peak live 2242 MiB (782 MiB resident) | 0.50 s | 3.96 |
| device (CUDA) | recompute | device peak live 1175 MiB | 0.52 s | 3.88 |
| host (CPU) | keep | host peak 2954 MiB | 1.39 s | 1.44 |
| host (CPU) | recompute | host peak 1769 MiB | 2.63 s | 0.76 |

Real-weight experiments under `Recompute` on the device reproduce the `Keep` results exactly: detection held-out AP50
0.2766 / loss 2.6378, mask IoU 0.6575, OCR CER 0.2472, LoRA held-out text loss 1.6474, exact resume true. With the
workgroup-split attention kernels of 9fbad69bf the device runs are also faster than before: detection 10.25 samples/s
(3.92 before), LoRA 2.42 (1.17), OCR 31.3 (13.7), masks 1.74.

### Item 6: decoder step floor (L8c, 2026-11-10)

Hardware: GH200 gpu1 of a shared node. The other session's jobs held every card at 99-100 % utilisation and 83-86 GB for
most of this lane (one 20 minute window at the start was idle: 1-2 % utilisation, 26 GB); every wall number below that
was taken in that load is marked "loaded" and is only comparable within its own table. Kernel times come from
`Gpu::profile_tape` (minimum over 20 replays of the recorded step, each dispatch timed from the previous one's completion)
and from ncu (`--clock-control base`, cold caches: it prices a kernel at about 1.3-2x its in-graph time).
Probes (ignored tests in `crates/mirage/tests/it/device_decoder.rs`): `decoder_step_floor_probe` / `_cq` (step time at batch
1/8/32/64 plus the per-kernel table), `decoder_gemm_shapes_probe` (products at the decoder's shapes),
`decoder_step_for_profiler_b1` / `_b64` (one replay, for ncu).

Baseline (FP32, idle card, position 150, min over replays): step 1.77 / 2.18 / 3.00 / 5.89 ms at batch 1 / 8 / 32 / 64;
profile 2.77 / 3.15 / 4.04 / 6.87 ms. At batch 64 `native:matmul_f32_tiled` was 58 % (4.0 ms over 61 products). The ncu of
its worst product shows why: the mHC gate projection (24 weight rows, k = 2048) launched ONE block of the 64 x 64 tile and
took 220 us (8 per step = 1.76 ms); every 512-row projection launched 8 blocks. Batch 1 was already under the 2 ms target.

Changes (each its own verified step):
1. `matmul_f32_skinny` (crates/kernels-cuda/cu/matmul_f32_skinny.cu), chosen by `CudaProvider` for 16..128 rows when the
   64 x 64 grid would fill less than half the multiprocessors: a 32 x 8 output tile per block, the contraction split over
   the block's eight warps (each stages its own 32-wide chunk, next chunk prefetched into registers), fixed-order fold.
   Every model with an f32 `Ops::matmul` inherits it; gate `a_starved_tile_grid_runs_the_skinny_kernel_and_agrees_with_an_f64_oracle`
   (kernel name per shape, f64 oracle with the sqrt(k) summation bar, tails in m / n / k), the shared parity cases keep their
   kernels. ncu per product at 64 rows: 24 x 2048 220 -> 23 us, 1120 x 512 (gemv at 2 x 32 rows: 33 us each) -> 24 us,
   512 x 512 (22.6 us per 32 rows) -> 9.6 us, 384 x 512 -> 9.9 us.
2. CQ selector (`backend_api::select`): the cooperative CQ product serves 16 rows (was 96) before the decoded weight
   meets the dense product, which now ends on the skinny kernel; a weight shorter than the dense tiles (the 24-row gate)
   keeps the cooperative kernel at every row count (the barrier-free `matmul_cq_gemm` took 1.1 ms per gate at 32 rows).
   Int8-activation (A8) products keep the cooperative kernel to 96 rows (no dense int8 form exists). CQ step profile at
   batch 64: 9.83 -> 4.72 ms.
3. `mirage_mhc_maps`: the Sinkhorn loop divided every lane (32 divisions an iteration on one thread per row); one
   reciprocal per row and one vector reciprocal per column sweep: 29 -> 14 us per launch.
4. `mirage_attn_apply`: 4 lanes x 16 key groups instead of 16 lanes x 4 groups (the key loop is the latency chain):
   batch 1 0.356 -> 0.166 ms per step, batch 64 0.418 -> 0.372.
