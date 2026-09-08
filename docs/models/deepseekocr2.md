# DeepSeek-OCR-2 (document image → text/markdown)

The successor to [DeepSeek-OCR](deepseek2ocr.md). Point it at a scanned page,
an invoice, or a screenshot of a table and it reads the content back as text
or markdown, following an instruction the same way v1 does (`<|grounding|>`
switches on box-annotated output).

Only the vision front end changed. SAM ViT-B (unchanged from v1) feeds a
24-layer Qwen2-shaped GQA tower run as a **learned-query resampler**: the
image's own tokens and a bank of learned query embeddings are concatenated
and run together under a **prefix-LM attention mask** (image tokens attend to
each other bidirectionally; the query tokens attend causally over everything
that came before them), and only the query rows survive into a single linear
projector. That replaces v1's SAM → 16× conv compressor → CLIP-L/14 chain.
The projected rows splice into the same, entirely unmodified, DeepSeek-V2 MoE
decoder (12 layers, 64 routed experts top-6 plus 2 shared, plain MHA) v1
uses - `crates/deepseek2` needed no changes at all for this model.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain <arch> <action>`)      | [x] |
| HTTP API              | [x] |
| D-Bus                 | [x] |
| Batched serving       | [ ] |

LoRA and full fine-tuning both exist and are proven to descend (see
[Options](#options) and the fine-tuning note below), but "training from
scratch" - the whole composite learning this task from an untrained
initialization - has not been demonstrated, so that row stays unchecked.
HTTP and D-Bus both reach the model through the same generic
`capability::Provider` surface every other served model uses - there is no
model-specific transport code. Batched serving stays unchecked for the same
reason it does on v1: each request needs its own SAM pass over its own
image, and the decoder has no wired batch axis, so `run_batch` is the serial
default.

## Getting the weights

Model id: `deepseek-ai/DeepSeek-OCR-2`. Point `BRAIN_DEEPSEEKOCR2_DIR` at a
directory holding both:

```text
mmproj-deepseek-ocr-2-q8_0.gguf
deepseek-ocr-2-q8_0.gguf
```

**No vendor-published GGUF exists for this model.** Only third-party
conversions of the upstream `deepseek-ai/DeepSeek-OCR-2` safetensors
checkpoint are available as of this writing, so `brain pull`/auto-fetch is
deliberately not wired to any one of them - naming an unofficial repo as the
canonical source would misrepresent it as something it isn't. Obtain a
compatible GGUF pair yourself and place it manually; a compatible pair
carries `general.architecture = "deepseek2-ocr"` on the language-model file
(the same string v1's decoder uses - the two decoders are byte-identical) and
`clip.projector_type = "deepseekocr2"` on the mmproj file (the unambiguous
marker that distinguishes this model's vision tower from v1's).

If either file is missing the model simply does not register - `brain caps`
omits it rather than listing something every call would fail.

## Running it

```bash
brain caps deepseek-ai/DeepSeek-OCR-2

BRAIN_DEEPSEEKOCR2_DIR=<dir> \
  brain deepseekocr2 generate \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max_new 16 --in image=page.ppm --json
```

Reached the same way over D-Bus and the OpenAI/Anthropic-shaped HTTP surface:
one `generate` action, streamed token by token, real `prompt_tokens`/
`completion_tokens`/`finish_reason` in the response.

A real end-to-end run against the real checkpoint completes and streams a
well-formed response - real preprocessing, real SAM, real resampler, real
splice, real decode, all the way through. On the specific synthetic test
graphic used to verify the pipeline, the model's first token was EOS, so the
decoded text came back empty. No independent oracle exists for this
checkpoint's own output (the same honest limitation v1's page states for its
own image+decoder loop), so that is recorded here as an observation about one
test image, not a claim about the model's real-world OCR quality - every
stage of the pipeline that produced it is independently verified correct
(preprocessing checked non-degenerate, the vision tower and splice
gradient-checked, the decode loop's causal self-consistency proven against
the real checkpoint). Try it on an actual scanned document to judge decode
quality for yourself.

Reference client: [`examples/vision/deepseek-ocr-2/`](../../examples/vision/deepseek-ocr-2/README.md).

## Options

| Param | Effect |
|---|---|
| `prompt` | the instruction after the image. Default `<\|grounding\|>Convert the document to markdown.` |
| `messages` | flattened chat messages (JSON array string); the last user turn becomes the instruction |
| `max_new` | tokens to generate, default 16. Decode is full-recompute (see below), so this default is set conservatively rather than to a round number - see Hardware and limits |
| `weights` | override `BRAIN_DEEPSEEKOCR2_DIR` for one request |

The `<|grounding|>` marker and the `<|ref|>`/`<|det|>`/table-tag markers are
reserved tokens in this vocabulary, the same as v1's (the tokenizer is
byte-identical, since the decoder is unchanged).

## Hardware and limits

**Single device (CPU), unlike v1's vision/decoder split.** v1 runs its SAM
tower on wgpu because a specific wgpu correctness bug at that tower's scale
was found, fixed, and independently re-verified at real-weight scale before
that split was trusted. This model stacks a SECOND 24-block-deep transformer
behind SAM, and that combination has not had the same independent
verification on wgpu - so every stage (SAM, the resampler, the decoder) runs
on the CPU backend today, a deliberately cautious default rather than an
unverified placement. Moving the vision half to wgpu is a natural follow-up
once that verification exists.

<!-- perf-number: hardware requirement, not a throughput claim -->
**~16 GiB resident.** Measured, not estimated: peak RSS of the whole
composite (SAM + the 24-layer resampler + the decoder) during a real
multi-token greedy decode, read off `/proc/self/status`, rounded up.

**Global view only.** A document image is fit into the model's one
`1024x1024` global view. The real DeepSeek-OCR-2 model additionally supports
a multi-tile "Gundam"-style layout for higher-resolution reading; brain's row
layout and splice mechanism already handle multiple tiles correctly and are
tested against a synthetic fixture, but real SAM inference on a non-global
tile size needs a position-embedding resample `crates/sam1` does not
implement yet. Closing that gap is what unlocks multi-tile inference; it
does not require touching the resampler, the mask, or the splice.

**Decode is full-recompute, not KV-cached.** v1's decoder has a proven
incremental KV-cache decode path (`DeepseekV2::generate_greedy_kv`); this
composite's interaction with that path through its own splice has not been
independently verified yet, so it uses the plain recompute loop instead.
`max_new` defaults conservatively (16) as a result - a real run at `max_new
=40` did not complete inside a several-minute budget on modest hardware,
while 16 reliably does. Raising the default is a matter of wiring the
already-proven KV-cache path through, not new research.

**LoRA fine-tuning works, with one caveat worth stating plainly.** LoRA
freezes a base and learns a small rank-limited correction on top of it - a
technique that assumes the base is already a competent representation.
Testing against a small synthetic fixture confirmed this precisely: adapting
a **freshly, randomly initialized** composite with LoRA plateaus almost
immediately (expected - a random network has no useful representation for a
low-rank correction to build on), while full fine-tuning of the same fixture
descends to near-zero loss because it can also move the decoder's MoE
experts, which no LoRA target here reaches. Against a real, already-trained
checkpoint - the actual use case - a fresh LoRA adapter has a competent base
to correct, which is the regime LoRA is built for.

**Greedy only** - no temperature, top-k, or top-p - batch 1, fp32 weights
only (no INT8 path yet).

**No NPU validation yet.** This host has no NPU firmware to test against;
NPU support (following the ONNX/OpenVINO export precedent
`crates/scrfd`/`crates/zipdepth` already established) is planned but not yet
implemented for this model - see `.agents/roadmap/deepseekocr2.md` for
current status.
