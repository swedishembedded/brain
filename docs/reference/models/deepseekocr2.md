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
| Batched serving       | [x] |

LoRA and full fine-tuning both exist and are proven to descend (see
[Options](#options) and the fine-tuning note below), but "training from
scratch" - the whole composite learning this task from an untrained
initialization - has not been demonstrated, so that row stays unchecked.
HTTP and D-Bus reach the model through the same generic
`capability::Provider` surface every other served model uses. Requests are
continuously batched through the shared decoder engine (the same one
DeepSeek-OCR uses), and a replica runs on every card that holds the whole
model.

## Getting the weights

Model id: `deepseek-ai/DeepSeek-OCR-2`.

```bash
brain pull deepseek-ai/DeepSeek-OCR-2     # 6.3 GiB, the vendor's own safetensors
brain serve --openai
```

The upstream `transformers` release is served **in place**: `config.json`,
`tokenizer.json` and the safetensors are read as published, with no
conversion step and no copy of the weights on disk. The checkpoint is
recognised by its `config.json` (`DeepseekOCR2ForCausalLM`), and its shape is
derived from the tensors and refused unless it is the documented one.

A community GGUF pair (`mmproj-deepseek-ocr-2-q8_0.gguf` +
`deepseek-ocr-2-q8_0.gguf`, language model `general.architecture =
"deepseek2-ocr"`, mmproj `clip.projector_type = "deepseekocr2"`) placed in
the models directory is recognised too. `brain pull` fetches only the
vendor's release.

## Running it

```bash
brain caps deepseek-ai/DeepSeek-OCR-2

brain deepseekocr2 generate \
    --prompt "<|grounding|>Convert the document to markdown." \
    --in image=page.ppm --json
```

Reached the same way over D-Bus and the OpenAI/Anthropic-shaped HTTP surface
(`brain serve`): one `generate` action, streamed token by token, real
`prompt_tokens`/`completion_tokens`/`finish_reason` in the response.

Reference client: [`samples/python/vision/deepseek-ocr/`](../../../samples/python/vision/deepseek-ocr/README.md)
(the same script serves both DeepSeek-OCR and DeepSeek-OCR-2 - see that README's
"v2" section).

## Options

| Param | Effect |
|---|---|
| `prompt` | the instruction after the image. Default `<\|grounding\|>Convert the document to markdown.` |
| `messages` | flattened chat messages (JSON array string); the last user turn becomes the instruction |
| `max_new` | tokens to generate, default 2048 (a page that finishes earlier stops at its end-of-sequence token) |
| `max_tiles` | the most 768 px tiles a large page is cut into (default 6); `0` reads the page as one 1024 px global view; over the chat API it is the `max_tiles` field, top-level or under `chat_template_kwargs`, an integer 0 to 64 |
| `weights` | a checkpoint directory for one request, instead of the model store's own pick |

The `<|grounding|>` marker and the `<|ref|>`/`<|det|>`/table-tag markers are
reserved tokens in this vocabulary, the same as v1's (the tokenizer is
byte-identical, since the decoder is unchanged).

## Hardware and limits

The vision encoder (SAM and the resampler) builds on one card and the
decoder on a card with room for it, the same card, or the host; a server
places it inside the `--device` set it was given.

**How a page is read.** An image no larger than 768 px on both sides is one aspect-fitted `768 x 768` view through
the local query bank (145 image rows). A larger image is read the way the reference's dynamic crops do: the grid of
768 px tiles closest to the page's aspect ratio (two to six; Pillow's bicubic resize and pad, byte for byte), each
through SAM at its own 48x48 patch grid and the resampler's 144 local queries, then the whole page as one 1024 px
global view (256 queries), then one separator. The embedding order is the reference's: local tiles, global view,
separator. `max_tiles = 0` reads the page as the global view alone (257 rows). Grounding boxes are relative to the
original image in every reading.

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

**Greedy only** - no temperature, top-k, or top-p.

**No NPU validation yet.** This host has no NPU firmware to test against;
NPU support (following the ONNX/OpenVINO export precedent
`crates/scrfd`/`crates/zipdepth` already established) is planned but not yet
implemented for this model - see `.agents/roadmap/deepseekocr2.md` for
current status.
