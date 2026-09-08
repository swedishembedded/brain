# DeepSeek-OCR-2 - a page in, text out

DeepSeek-OCR-2 is the successor to [DeepSeek-OCR](../deepseek-ocr/README.md),
reusing the same decoder and the same `generate` action shape - only the
vision front end changed. It is driven through the same `Run`/`Subscribe`
methods on `com.swedishembedded.Brain1` as every other model, with no
model-specific transport code.

| model | action | weights env |
|---|---|---|
| `deepseek-ai/DeepSeek-OCR-2` | `generate` - image + instruction → streamed text | `BRAIN_DEEPSEEKOCR2_DIR` (the directory holding **both** shipped GGUFs) |

```bash
brain caps deepseek-ai/DeepSeek-OCR-2
```

## Run it

No vendor-published GGUF exists for this model yet (see the model catalog's
DeepSeek-OCR-2 entry for what a compatible checkpoint pair must carry), so
obtain one yourself and point `BRAIN_DEEPSEEKOCR2_DIR` at the directory
holding both files. The CLI reaches the model directly, with no bus needed:

```bash
BRAIN_DEEPSEEKOCR2_DIR=<dir> \
  brain deepseekocr2 generate \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max_new 16 --in image=page.ppm --json
```

The same `deepseek-ai/DeepSeek-OCR`
[`ocr_document.py`](../deepseek-ocr/ocr_document.py) client works over D-Bus
against this model too - it is the same `generate` action with the same
image/prompt parameters - once its `MODEL` constant is pointed at
`deepseek-ai/DeepSeek-OCR-2` and `BRAIN_DEEPSEEK_OCR_DIR` is replaced with
`BRAIN_DEEPSEEKOCR2_DIR`. A dedicated copy of that script is not shipped here
to avoid two files drifting out of sync over one three-line difference.

## What differs from v1

Today, this model serves its **global view only** (a document image fit into
one 1024×1024 view) - the multi-tile layout v1 also doesn't serve over D-Bus
needs the same SAM position-embedding work either model would need. Decode
is full-recompute rather than KV-cached, so `--max_new` defaults
conservatively; see the model catalog's own page for the current, honest
limits.

---

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements document-understanding systems for teams that
cannot upload their documents to a third party. If your team needs expertise
in OCR, document layout analysis, or vision-language models running
on-premise, you can procure our services by sending an email to
**info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
