# LFM2.5

LFM2.5-Encoder from LiquidAI: a bidirectional text encoder built from a
hybrid of short-convolution and attention layers, with a masked-language-
model head tied to its token embedding. It was pretrained with 30% token
masking and handles a usable context of up to 8,192 tokens. Reach for it when
you need text embeddings or cloze-style mask-filling, not generative chat -
that's what Qwen3 is for.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| INT8                   | [ ] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [x] |

## Getting the weights

`LiquidAI/LFM2.5-350M` (also available as the smaller `-230M` variant) is
auto-fetched on first use. To import a checkpoint you already have locally:

```bash
brain lfm import --hf <hf_dir> --out lfm-350m.safetensors
```

To serve a checkpoint, point `BRAIN_LFM2` (and `BRAIN_LFM2_TOKENIZER`) at it.

## Running it

```bash
brain lfm2 embed     --weights F --tokenizer tokenizer.json --text "..."
brain lfm2 fill-mask --weights F --tokenizer tokenizer.json \
    --text "The capital of France is <|mask|>." --topk 5
brain lfm2 finetune  --weights F --tokenizer tokenizer.json --data <dir> [...]
brain lfm2 eval      --weights F --tokenizer tokenizer.json --data <dir>
brain lfm2 data      --input corpus.txt --tokenizer tokenizer.json --out data/lfm
```

lfm2 also has a capability manifest (`brain caps brain/lfm2` lists it, and it
is reachable over D-Bus/HTTP) - but it dispatches through the `brain lfm2 ...`
verbs above on the CLI specifically, same as every other architecture with
its own dedicated CLI module.

To serve it over HTTP or D-Bus:

```bash
BRAIN_LFM2=lfm-350m.safetensors BRAIN_LFM2_TOKENIZER=tokenizer.json brain serve --dbus --openai
```

Embeddings are reachable at the standard `POST /embeddings` endpoint once
serving.

## Options

- `--seq T` - sequence length for `embed` (up to the 8,192-token usable
  context).
- `--topk K` - number of mask-fill candidates to return.
- `BRAIN_LFM2_BATCH` - batched-forward slots per serving instance (default 2).

## Hardware and limits

Usable context is 8,192 tokens. Attention is bidirectional (non-causal),
which means unmasked padding corrupts every real token's encoding - requests
are built at their exact length rather than padded into a batch, so
padding-aware batched serving is a separate piece of future work. There's no
training-from-scratch path today; you import a released checkpoint and
finetune it.
