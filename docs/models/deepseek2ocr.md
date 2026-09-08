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
| `max_new` | tokens to generate, default 32. **Every token is a full recompute** - see below |
| `weights` | override the model-store resolver's own pick for one request |

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
stays put). Because it holds real bytes on two devices at once, it is the one
served model the scheduler claims through its **multi-device** path: it names
`(GPU, ~6 GiB)` for the vision tower and `(CPU, ~16 GiB)` for everything
host-side, so each device is checked against its own budget instead of one of
them being invisible. On a host whose GPU shares physical RAM with the CPU,
both names draw from the one shared memory pool, so nothing is double-counted.
It does not mutate `BRAIN_DEVICE`; a server-lifetime resident must not change
the backend other models build on, so the card is chosen by scoped registry
selection instead.

<!-- perf-number: hardware requirement, not a throughput claim -->
**~22 GiB resident (measured at the old 512-token flat-tape shape; pending
re-measurement on the chunked one - see `crates/cli/src/
resident_deepseekocr.rs`'s `COMPOSITE_PEAK_BYTES` doc).** The served instance
is now sized for the checkpoint's real 8192-token context by default
(`$BRAIN_DEEPSEEK_OCR_CTX`, clamped to that ceiling), not a fixed 512 - the
long-standing reason for the 512 cap (every extra row of context cost a
`[seq, 129280]` logit slab in a flat batched tape) no longer applies:
`DeepseekV2::prefill_chunked` prefills in bounded rounds
(`$BRAIN_DEEPSEEK_OCR_CHUNK`, default 512, which already covers the whole
real prompt in one round) against a KV cache sized to the context instead of
a tape sized to it, so a wider context grows the KV cache and the round
scratch, not a `[ctx, vocab]` slab. A box with less than ~24 GiB free
probably still activates it, likely with room to spare - the 22 GiB figure
predates this change and has not yet been re-measured downward.

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

**Model construction and vision encoding (SAM ViT-B at 1024x1024 ->
CLIP-L/24 -> compressor -> projector) were profiled and optimized across
several passes**, including AVX2/AVX-512 fast paths for the decode loop's
dominant CPU kernels and moving the vision encoder onto the wgpu backend once
a `crates/sam1` correctness bug that used to block it there was fixed. The
decoder itself stays on the CPU backend (no measured wgpu benefit for that
stage).

For the actual measured numbers behind these changes - wall-clock deltas,
per-kernel profiles, and the full history of what was tried and what didn't
pan out - see `.agents/roadmap/deepseek-ocr.md`, or measure your own build
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
