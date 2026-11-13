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

`LiquidAI/LFM2.5-350M` (also available as the smaller `-230M` variant). Import
the release once into brain's checkpoint and keep it, with its
`tokenizer.json`, in the model store, where every action finds it:

```bash
brain pull LiquidAI/LFM2.5-350M
S=~/.local/share/brain/models
brain lfm2 import --source "$S/LiquidAI/LFM2.5-350M" --dest "$S/brain/lfm2/lfm2-350m.safetensors"
cp "$S/LiquidAI/LFM2.5-350M/tokenizer.json" "$S/brain/lfm2/"
```

`--weights` and `--tokenizer` name either outright.

## Running it

`brain lfm2 <action>` runs the served actions; `sample-models-lfm2`
(`make samples/models/lfm2/build`) is the same program on its own.
`<action> --help` lists every parameter with its default.

```bash
brain lfm2 embed --text "..." --out embeddings=emb.f32
brain lfm2 fill_mask --text "The capital of France is <|mask|>." --topk 5
brain lfm2 tokenize --corpus corpus.txt --dest data/lfm
brain lfm2 finetune --data data/lfm --save ft.safetensors [--steps N --batch B --seq T --activations keep|recompute] --seed $RANDOM
brain lfm2 eval --data data/lfm --weights ft.safetensors --seed $RANDOM
```

A long document comes from a file with `--in text=doc.txt`.

To serve it over HTTP or D-Bus:

```bash
BRAIN_LFM2=lfm-350m.safetensors BRAIN_LFM2_TOKENIZER=tokenizer.json brain serve --dbus --openai
```

Embeddings are reachable at the standard `POST /embeddings` endpoint once
serving; the serving process batches `fill_mask` and `embed` and runs the
other actions as they are.

## Options

- `--max-tokens T` - truncate the `embed` input (up to the 8,192-token usable
  context).
- `--topk K` - number of mask-fill candidates to return.
- `--normalize true` - L2-normalize the mean-pooled embedding.
- `BRAIN_LFM2_BATCH` - batched-forward slots per serving instance (default 2).

## Hardware and limits

Usable context is 8,192 tokens. Attention is bidirectional (non-causal),
which means unmasked padding corrupts every real token's encoding - requests
are built at their exact length rather than padded into a batch, so
padding-aware batched serving is a separate piece of future work. There's no
training-from-scratch path today; you import a released checkpoint and
finetune it.
