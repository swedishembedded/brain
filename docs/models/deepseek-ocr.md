# DeepSeek-OCR (document image → text/markdown)

Point it at a scanned page, an invoice, a screenshot of a table - anything
whose content is text laid out on a page - and it reads the whole thing back
as text or markdown. Unlike a classic OCR engine it is a full
vision-language model, so it follows an instruction: "convert the document to
markdown", "read the table", and with the `<|grounding|>` marker it emits
bounding boxes alongside the text it read.

The architecture is a **DeepEncoder** - a SAM ViT-B tower at 1024², a 16×
convolutional token compressor, then CLIP-L/14 with its patch embedding
bypassed in favour of those compressed tokens - projected into a
**DeepSeek-V2 MoE decoder** (12 layers, 64 routed experts top-6 plus 2 shared,
plain MHA). One 1024×1024 page becomes 256 projector rows, interleaved with 16
learned `image_newline` rows and one `view_separator` row: a 273-row image
block spliced over the decoder's `<image>` placeholders.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain do`)      | [x] |
| HTTP API              | [x] |
| D-Bus                 | [x] |
| Batched serving       | [ ] |

The backward pass exists end to end (the decoder's cross-entropy gradient
reaches the input pixels through the splice, the projector, CLIP and the whole
SAM tower) and is gradient-checked.

**LoRA is wired too, but not yet a production feature** - which is why
"training" stays unchecked above. `deepseekv2::config::DeepseekV2Config::lora`
freezes the decoder's base weights (embeddings, norms, the MoE
router/experts/shared expert, the untied head, and the four attention
projections' own base matrices) and adds a rank-`r` low-rank adapter on those
four projections (`q_proj`/`k_proj`/`v_proj`/`o_proj`), composed entirely from
the same `matmul`/`axpy`/`grad_scale` kernels the base decoder's own
forward/backward already dispatch - no new kernel. It is gradient-checked
(`deepseekv2/tests/gradcheck.rs::grads_match_finite_differences_lora`) and
gated by a descent smoke test
(`deepseekocr/tests/tiny_ref.rs::composite_lora_backward_freezes_the_base_and_descends`)
that trains the composite with the decoder's base frozen and ONLY its
`.lora_a`/`.lora_b` adapters trainable, and asserts a plain gradient step on
those adapters alone measurably lowers the loss - proof the training path is
wired correctly and actually descends, not a production fine-tune. Still
missing: a `finetune`-style CLI verb, a masked-dataset training loop, and
adapter save/load through `brain do` - see `crates/deepseekocr/src/train.rs`
for the composite-level merge helper (`lora_init_map`) that exists so far.

## Getting the weights

Model id: `deepseek-ai/DeepSeek-OCR` - not auto-fetched. The checkpoint is the
GGUF pair `ggml-org/DeepSeek-OCR-GGUF` publishes:

```text
<dir>/mmproj-DeepSeek-OCR-Q8_0.gguf     448 MB   SAM + CLIP + projector
<dir>/DeepSeek-OCR-Q8_0.gguf            3.1 GB   the decoder, and the tokenizer
```

Point `BRAIN_DEEPSEEK_OCR_DIR` at the directory holding **both**. If either is
missing the model does not register at all - `brain caps` still lists it (the
manifest is weights-free) but `brain do` and `brain serve` say which file is
missing rather than failing mid-request.

On first use brain writes one derived file beside them:

```text
<dir>/DeepSeek-OCR-brain-fp32.safetensors   11.7 GB   the decoder's fp32 expansion
```

That expansion is not a convenience. A `WeightReader` streams it one tensor at
a time into the model's buffers, whereas dequantizing the Q8_0 file into a host
map would hold the same 11.7 GB *twice*. Building it takes a few minutes and
happens once; delete it to reclaim the disk and it is rebuilt on the next
activation.

## Running it

There is no dedicated `brain deepseekocr` verb - one generic action,
`generate`:

```bash
brain caps deepseek-ai/DeepSeek-OCR

BRAIN_DEEPSEEK_OCR_DIR=<dir> \
  brain do deepseek-ai/DeepSeek-OCR generate \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max_new 10 --in image=page.ppm --json
```

```json
{"completion_tokens":10,"finish_reason":"length","prompt_tokens":283,
 "text":"<|ref|>image<|/ref|><|det|>[[48, 0,"}
```

That is a real run on a scanned-page image: 283 prompt tokens is BOS + the
273-row image block + the 9-token instruction, and the text is grounding mode
opening a reference and its detection box. `--max_new 10` cut it off mid-box,
which is what `finish_reason: "length"` says.

Over D-Bus and over the OpenAI/Anthropic HTTP surfaces it is the same action
with the same params - the model is chat-capable-shaped (`generate`, streaming,
`messages`/`prompt`, a text output), so `/v1/chat/completions` reaches it with
an image attached and streams the decoded text back token by token, with real
`prompt_tokens` / `completion_tokens` / `finish_reason`.

```bash
BRAIN_DEEPSEEK_OCR_DIR=<dir> \
  dbus-run-session -- bash -c 'brain serve --dbus & sleep 5
    python3 examples/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8'
```

Reference client: [`examples/vision/deepseek-ocr/`](../../examples/vision/deepseek-ocr/README.md).

## Options

| Param | Effect |
|---|---|
| `prompt` | the instruction after the image. Default `<\|grounding\|>Convert the document to markdown.` - the reference model's own prompt |
| `messages` | flattened chat messages (JSON array string); the last user turn becomes the instruction |
| `max_new` | tokens to generate, default 32. **Every token is a full recompute** - see below |
| `weights` | override `BRAIN_DEEPSEEK_OCR_DIR` for one request |

The reserved markers are ordinary text in the instruction and are tokenized
atomically: `<|grounding|>` turns on grounding mode (the model then emits
`<|ref|>…<|/ref|><|det|>…<|/det|>` spans), and `<|ref|>`, `<|det|>` and the
`<td>`/`<tr>` table tags are all single ids in this vocabulary.

## Hardware and limits

**Split backend: vision on wgpu, decoder on CPU.** `crates/sam1`'s ViT tower
used to corrupt its per-block buffers on the wgpu backend at 1024x1024 once
the graph held three or more blocks - a tracked correctness bug that produced
plausible-looking garbage rather than an error. That bug is now fixed and
confirmed at real-weight scale (5/5 clean parity runs plus 32/32 clean trials
under induced heavy contention reproducing the original failure conditions),
so `caps::Session::load` builds the vision encoder (SAM+CLIP+glue) on
`gpu_core::Gpu::new_wgpu` and the decoder on `gpu_core::Gpu::new_cpu` (the
decoder has no wgpu-corruption history and no measured wgpu benefit, so it
stays put). This model still declares a RAM-only `MemCost` (`vram == 0`),
which is the residency scheduler's own vocabulary for "never place me on a
GPU" - that has NOT been updated to account for the vision tower's real wgpu
VRAM usage, a known, deliberately deferred gap (a residency/budget-accounting
fix, not a `crates/deepseekocr` change). It does not mutate `BRAIN_DEVICE`; a
server-lifetime resident must not change the backend other models build on.

**~22 GiB resident.** Measured, not estimated: the real-weight composite gate
reports a 21.32 GiB peak RSS for this exact build, read off `/proc/self/status`.
The served instance is sized for a 512-token context (the 273-row image block,
BOS, the instruction, and room to generate), which costs a little more than the
test's ~260. A box with less than ~24 GiB free will not activate it.

**KV-cached decode.** Decode used to be `O(T²)` recompute with no KV cache -
every generated token re-ran the whole sequence through 12 MoE layers, ~22 s
per token. `DeepseekV2::generate_greedy_kv` closed that: the prompt still
pays one batched forward (which also seeds the persistent per-layer K/V
cache), but every token after that is one `O(1)` incremental decode step
(`model::block::gqa_decode_step` plus a single-row MoE/dense FFN pass), not a
full re-run. Measured on 22 CPU cores, release build, a real document page
(a 3-page CV rendered to PNG, first page, 1654x2340) and the default prompt
(283 prompt tokens), `--max_new 32`:

| build | wall time | tokens/s (decode only) |
|---|---|---|
| `O(T²)` recompute (previous) | ~12 min (extrapolated from the `--max_new 2`/`--max_new 10` numbers below) | ~0.045 |
| `O(T)` KV-cache (current) | ~123-147 s | ~0.65 (32 tokens / ~49 s of decode-only kernel time) |

The `O(T²)` numbers this replaces, for reference (same machine, a synthetic
1600x1131 page): `--max_new 2` 1 min 54 s, `--max_new 10` 4 min 54 s (~22 s
per additional token, confirming the quadratic blowup). See
`docs/performance/overview.md`'s case study for the full `BRAIN_PROFILE`
per-kernel breakdown.

**Model load + vision encoding were the dominant cost, and the vision
encoder got its own profiling pass.** Of the ~123-147 s total run above:
~18 s was one-off model construction, ~50 s was the KV-cached decode, and
the remaining ~75-80 s was the vision encoder (SAM ViT-B at 1024x1024 ->
CLIP-L/24 -> compressor -> projector), then unbroken down per kernel. A
follow-up pass added per-stage instrumentation and found (and partly fixed)
the encoder's real cost: `attn_apply_cross` (SAM's attention-weighted-V
step) was 70-71% of the tower's own CPU forward, 20-30x more wall time than
a same-FLOP-count sibling kernel, traced to a cache-hostile V-transpose now
tiled for locality. The CPU backend `crates/sam1` used to be pinned to (the
wgpu backend used to corrupt this tower's output at production scale) costs
roughly 3.6x over wgpu on this tower alone in isolation, measured directly
rather than assumed. With both landed, real-weight runs measured the vision
encoder directly at 25-34 s, and a clean single-tenant run measured the whole
pipeline at 83.1 s (down from the ~95.6-97.1 s the same pass measured under
concurrent-agent machine load) - both numbers from BEFORE the vision encoder
moved onto wgpu for real (see below).

**Since then: the vision encoder moved onto wgpu, `silu_mul`/`scale_add`
gained AVX2 fast paths, and both were validated together on real pages.**
`silu_mul`/`scale_add` (the SwiGLU activation core and the MoE combine step)
were found running the generic scalar JIT path at 15-16% EACH of the decode
loop's profiled CPU time, the same missing-fast-path bug class
`moe_linear_gated` and the self-attention family were; fixed the same way,
and confirmed on a real page: both dropped to 1.1%/0.3% of the profile.
`crates/sam1`'s wgpu correctness fix (above) let the vision encoder finally
move off the CPU pin for real (`caps::Session::load` now builds SAM+CLIP+glue
on `gpu_core::Gpu::new_wgpu`) - honestly mixed in practice: SAM's own forward
dropped from 23.8-33.0 s to 13.3-27.0 s but inconsistently, while CLIP
regressed from a 1.2-8.9 s (median 1.4-1.5 s) range to a consistently slower
7.6-12.3 s - its ops are small enough that per-dispatch GPU overhead on this
Intel Arc iGPU outweighs the compute saved. **The current real number**: five
real pages, one resident instance, `max_new=128`, warm-instance median
69.4 s/page (mean 68.7 s) - roughly 12% faster than the prior 78.5 s median,
though the decode-side AVX2 fixes and the vision-wgpu split are not cleanly
separable since both landed together. At `max_new=32` head-to-head against
`llama-mtmd-cli` on the same real page, brain now measures 33.4 s against
`llama.cpp`'s own 57.3-64.7 s (two runs) - roughly **1.8x faster than
`llama.cpp`** on this comparison, a reversal of the 1.57x deficit the prior
pass measured. Model construction's 20-28 s (a one-time, per-activation cost,
unaffected by any of this) remains open, and the CLIP-on-wgpu regression is a
real, named follow-up (keep CLIP on CPU while only SAM+glue move to wgpu).
See `docs/performance/overview.md`'s case study for the full per-stage and
per-kernel tables and the honest caveats on what could and could not be
pinned down on a shared machine.

**No early stop.** The greedy loop always runs `max_new` steps. The output is
truncated at the first end-of-sentence id and `finish_reason` reports `stop`
honestly, but the wall time is always the full budget - stopping early needs a
fallible callback in `crates/deepseekv2`, not a wrapper here.

**One image, one view, batch 1.** The decoder's splice takes exactly one
contiguous `(row0, n_rows)` run, so only DeepSeek-OCR's *global* (overview) view
is served: the multi-tile "Base"/"Gundam" modes, which interleave a second token
stream at 640², need one splice call per run. The row layout for them already
exists and is unit-tested (`deepseekocr::rows`), and the row gather is already
indifferent to tiles - `deepseekv2::enable_mm_splice` is the piece that is not.
`run_batch` is therefore the serial default: two concurrent requests share no
work (each image needs its own encoder pass, and the decoder has no batch axis).

**Greedy only** - no temperature, top-k or top-p - and fp32 weights only (no
INT8 path).

**No multimodal oracle for the decode loop.** The text decoder alone is matched
token for token against llama.cpp on these weights; the image+decoder loop is
gated on completing, on finite logits and on causal self-consistency, because
llama.cpp's debug callback segfaults inside this model's CLIP graph and no
post-image token-id capture exists to compare against. So "brain decodes the
same tokens as the reference for a real page" is **not** claimed.
