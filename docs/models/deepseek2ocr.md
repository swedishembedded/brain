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
| CLI (`brain <arch> <action>`)      | [x] |
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

Model id: `deepseek-ai/DeepSeek-OCR`. **Either published release works**, and
brain recognizes whichever one it finds on disk.

The default is the GGUF pair `ggml-org/DeepSeek-OCR-GGUF` publishes, which
auto-fetches (⤓, opt-in `--autofetch`) on first CLI use, no env var needed:

```text
<dir>/mmproj-DeepSeek-OCR-Q8_0.gguf     448 MB   SAM + CLIP + projector
<dir>/DeepSeek-OCR-Q8_0.gguf            3.1 GB   the decoder, and the tokenizer
```

`brain pull deepseek-ai/DeepSeek-OCR` fetches the upstream `transformers`
release instead:

```text
<dir>/config.json                                the shape, cross-checked against the tensors
<dir>/tokenizer.json                             the real merges and specials
<dir>/model-00001-of-000001.safetensors  6.7 GB  every tensor, BF16
```

That one is used **as downloaded** - no conversion step and no derived file
beside it, because the tensor names map onto brain's exactly and the decoder
reads straight from the shard. It costs more disk than the quantized pair
(6.7 GB vs 3.5 GB) but skips the pair's one-off ~12 GB fp32 expansion, so it
is the smaller footprint overall and the faster first load. It also carries
the real `tokenizer.json` rather than the GGUF's `tokenizer.ggml.*` KV, which
names the pre-tokenizer instead of defining it.

Either checkpoint is discovered automatically anywhere under the models
directory, resolved through `deepseek2ocr::spec::Deepseek2ocrSpec` on **real
header content, never by filename**: for the GGUF pair, the LM's own
`general.architecture == "deepseek2-ocr"` paired with a sibling vision GGUF
declaring `general.architecture == "clip"` and `clip.projector_type ==
"deepseekocr"`; for the upstream release, a `config.json` declaring
`architectures: ["DeepseekOCRForCausalLM"]`. If a required file is missing the
model does not register at all - `brain caps` still lists it (the manifest is
weights-free) but the CLI and `brain serve` say what is missing rather than
failing mid-request. `weights` still names a directory outright, per request.

On first use of **the GGUF pair** brain writes one derived file beside it (the
upstream release needs no such expansion - see above):

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

brain deepseek2ocr generate \
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
dbus-run-session -- bash -c 'brain serve --dbus & sleep 5
  python3 examples/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8'
```

Reference client: [`examples/vision/deepseek-ocr/`](../../examples/vision/deepseek-ocr/README.md).

## Options

| Param | Effect |
|---|---|
| `prompt` | the instruction after the image. Default `<\|grounding\|>Convert the document to markdown.` - the reference model's own prompt |
| `messages` | flattened chat messages (JSON array string); the last user turn becomes the instruction |
| `max_new` | tokens to generate, default 2048 (`deepseek2ocr::caps::DEFAULT_MAX_NEW`). Decode is KV-cached and stops early on EOS - see below |
| `weights` | override the model-store resolver's own pick for one request |

`BRAIN_DEEPSEEK_OCR_NGRAM_SIZE` (default 30) and `BRAIN_DEEPSEEK_OCR_WINDOW_SIZE`
(default 90) tune the decode loop's n-gram anti-repetition filter: a
generated n-gram already seen within the trailing `WINDOW_SIZE` tokens is
suppressed. Set `NGRAM_SIZE=0` to disable the filter entirely.

The reserved markers are ordinary text in the instruction and are tokenized
atomically: `<|grounding|>` turns on grounding mode (the model then emits
`<|ref|>…<|/ref|><|det|>…<|/det|>` spans), and `<|ref|>`, `<|det|>` and the
`<td>`/`<tr>` table tags are all single ids in this vocabulary.

## Hardware and limits

**Both halves on the GPU, on a host that has one.** `caps::Session::load`
builds the vision encoder (SAM+CLIP+glue) with `gpu_core::Gpu::new_wgpu` and
the decoder on whichever device `caps::decoder_device` picks. That is a real
card by default: on two Tesla P40s the served build places the tower on one
and the DeepSeek-V2 MoE decoder on the other.
<!-- perf-number: the comparison this default rests on, measured on one host -->
One real page (283-token prompt, 512 generated tokens) takes **62.4 s** that
way against **91.8 s** with the decoder on the 48-thread CPU Cranelift JIT -
**1.47x**, with byte-identical decoded markdown, and the 283-row prefill is
**3.2 s against 5.6 s**.

Both figures above predate two changes that cut the page substantially: the
CLIP tower was dispatching a naive GEMM (see `crates/clip`), and the decode
loop rebuilt one bind group per dispatch per token instead of replaying the
one it had already recorded (`gpu_core::Gpu::enable_step_cache`). A rendered
1240x1754 text page at the same 512-token budget, both halves on ONE P40,
the two builds run back to back on an idle host:

| | before | after |
|---|---|---|
| generate (encode + prefill + 512 decode) | 77.50 s | 49.45 s | <!-- perf-number: one page, one card, one revision -->
| of that, CLIP forward | 5.02 s | 0.15 s | <!-- perf-number: one stage of that same page -->
| bind groups built | 1 945 491 | 33 384 |
| dispatches | 1 945 491 | 1 945 491 |

The decoded markdown is byte-identical across both, and the dispatch count is
unchanged - the GPU does exactly the work it did before.

What dominates the remainder is that dispatch COUNT, not any kernel: this
decoder dispatches all 64 experts of every MoE layer every token whatever the
router picked, which is 1 083 456 of those 1 945 491 dispatches.

<!-- perf-number: the CPU side of that comparison is not this repo's default build -->
That CPU figure is the FAST one: it needs `crates/backend-cpu`'s rayon/GEMV
scheduling fix, without which the same page takes 539 s. Comparing the card
against an unfixed CPU would have overstated the win about six-fold, which is
why the number quoted here is the fixed one.

Speed is not the whole reason the GPU is the default. The CPU decoder spends
all 48 cores and 13.5 GiB of host RAM to reach its throughput; the GPU
decoder beats it on roughly one host thread and 2.7 GiB, which on a serving
box is 47 cores and 11 GiB returned to every other resident.

The decoder is correct on both, and gated as such:
`crates/deepseek2/tests/{parity,generate}.rs` gate the real 2.9 B decoder
against the SAME llama.cpp reference on each backend (it reproduces
llama.cpp's continuation token for token on both),
`crates/deepseek2/tests/backend_parity.rs` gates the two against each other at
the fp32 noise floor, and `crates/deepseek2/tests/decode_throughput.rs`
asserts they decode the same ids at the served `ctx = 8192`/`chunk = 512`
shape while reporting each one's cost.

**`$BRAIN_DEEPSEEK_OCR_DECODER_DEVICE` is the operator knob**, alongside
`$BRAIN_DEEPSEEK_OCR_CTX`/`$BRAIN_DEEPSEEK_OCR_CHUNK`: `cpu`, `gpu` (the
vision tower's own card), `gpu<i>` or a bare index (that card), or `auto`
(the default). `auto` prefers a second discrete card, falls back to sharing
the tower's card when one is big enough for both, and falls back to the CPU
when there is no discrete GPU at all - on such a box `Gpu::new_wgpu` resolves
to a software rasteriser whose buffers ARE host RAM, which is strictly worse
than the Cranelift JIT already there. It reasons about each card's capacity,
not its free memory (wgpu exposes no portable free-VRAM query), so on a
shared box name the card outright. The CPU decoder is a supported placement,
not a deprecated one.

<!-- perf-number: hardware requirement, not a throughput claim -->
**~7 GiB + ~14 GiB of VRAM and ~3 GiB of RAM**, measured on the served build
(`nvidia-smi memory.used` per card and `/proc/self/status` VmHWM, sampled for
the life of a real page): **6.27 GiB** on the vision card, **13.25 GiB** on
the decoder card, **2.68 GiB** of host RSS. With the decoder on the CPU
instead it is 6.27 GiB of VRAM and **13.54 GiB** of RAM. Each of those is its
own direct measurement, so `crates/cli/src/resident_deepseekocr.rs` names
every device the instance really occupies with its own figure rather than
splitting one number across devices - which is what it used to do, from a
21.32 GiB all-CPU reading taken at the long-superseded 512-token flat-tape
shape.

The decoder's ~11.4 GiB is fp32 weights (2234 tensors, 2.9 B parameters); the
rest is ~1.0 GiB of KV cache at the checkpoint's real 8192-token context
(`$BRAIN_DEEPSEEK_OCR_CTX`, clamped to that ceiling), ~0.5 GiB of per-expert
MoE activation scratch at a 512-row prefill round, and ~0.3 GiB of attention
slabs. A smaller context shrinks only the KV cache - the weights do not move -
so those figures are close to a floor rather than an average. The long-standing
reason context used to be capped near 512 (every extra row cost a
`[seq, 129280]` logit slab in a flat batched tape) no longer applies:
`DeepseekV2::prefill_chunked` prefills in bounded rounds
(`$BRAIN_DEEPSEEK_OCR_CHUNK`, default 512, which already covers the whole real
prompt in one round) against a KV cache sized to the context instead of a tape
sized to it.

**KV-cached, CHUNKED decode.** Decode used to be `O(T²)` recompute with no KV
cache - every generated token re-ran the whole sequence through all 12 MoE
layers. `DeepseekV2::generate_greedy_kv` closed that with a persistent
per-layer K/V cache; `DeepseekV2::prefill_chunked` (used whenever the
composite is built `batched = false`, which is what serving now does) then
replaced the SINGLE batched forward that used to seed that cache with
several bounded rounds, so the prompt's own cost no longer requires the flat
tape either. Every generated token after prefill is still one `O(1)`
incremental decode step (`model::block::gqa_chunk_step` at one new row, plus
a single-row MoE/dense FFN pass), not a full re-run of the sequence.

**One grouped dispatch per MoE layer, not one per expert.** The checkpoint
stores each MoE projection's 64 experts as ONE `[64, out, in]` tensor
(`blk.N.ffn_*_exps.weight`); brain imports that layout unchanged instead of
unpacking it into 64 separate parameters, so the whole routed half of a layer
binds as three weight operands. Both prefill rounds and single-token decode
then run it through `model::moe::expert_fwd_grouped`, which permutes the
routed `(row, expert)` pairs into per-expert contiguous ranges on the device
and issues ONE grouped GEMM per projection - about 9 dispatches per MoE layer
regardless of expert count, against the `5 x 64 = 320` the per-expert loop
issued for a top-6 router. It also stops a prefill round writing (and then
discarding) an all-experts-by-all-rows activation slab: the grouped scratch is
sized by `rows * top_k`, not `rows * n_experts`.

For the actual measured numbers behind these changes - wall-clock deltas,
per-kernel profiles, and the full history of what was tried and what didn't
pan out - see `.agents/roadmap/deepseek2ocr.md`, or measure your own build
with `BRAIN_PROFILE=1` and `brain perf run` (see
[Performance](../performance/overview.md)); numbers measured on one machine
at one point in this model's development are not a promise for yours.

**Real EOS early stop.** `DeepseekV2::generate_greedy_kv_stream`'s callback
returns `false` on the model's end-of-sentence id, which stops the loop from
dispatching any further decode steps - wall time now tracks how early the
model actually stopped, not always the full `max_new` budget.

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
