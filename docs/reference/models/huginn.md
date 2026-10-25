<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Huginn

A recurrent-depth ("looped") language model: a two-block prelude embeds the
tokens, four shared core blocks are applied a configurable number of times
with the embedded input re-injected through a linear adapter, and a two-block
coda un-embeds the final latent state into the tied head. Blocks are sandwich
RMSNorm (`x = norm_2(attn(norm_1(x)) + x); x = norm_4(mlp(norm_3(x)) + x)`),
fused QKV with a per-head Q/K bias, interleaved-pair RoPE (base 50 000) and a
fused gate/up SwiGLU. The released checkpoint is 3.57 B parameters, width
5280, 55 heads of width 96, and is published under Apache-2.0.

```bash
# Import the Hugging Face checkpoint (four fp32 safetensors shards).
brain huginn import --hf <hf checkpoint dir> --out huginn.safetensors

# Run the paged serving engine locally (greedy).
brain huginn generate --checkpoint <hf checkpoint dir> --prompt "The capital of France is" --max-new 32
```

`brain serve` picks up an imported `huginn` checkpoint from the model store
and exposes the same `generate` action as the other text models over HTTP,
D-Bus and the SDK. Chat requests are rendered with the checkpoint's own Jinja
`chat_template` (`<|begin_header|>User<|end_header|>` turns, `<|end_turn|>`),
which ships with the checkpoint - nothing is transcribed per family here.

The released template has no slot for tool schemas and the model was not
post-trained for tool calling, so a request's `tools` are dropped rather than
described in a system turn (measured: a schema block derails the answer). A
`<tool_call>` block the model emits anyway is parsed as usual. A checkpoint
whose template does carry a tools slot is unaffected.

## Recurrence and staging

`num_steps` is a serving setting (`--steps` on the CLI action). Every step
applies the same four core blocks to a different residual stream, so each
*stage* - a prelude block, a core block within one step, a coda block - owns
its own attention history. There are `n_prelude + steps * n_recurrent +
n_coda` stages. The engine can also wrap the recurrent stages around a ring
of histories (the released model's own `compress-sN` KV strategy), which cuts
the pool but diverges from the full history - it is available
(`Engine::open_with_sharing`) and not yet the served default.

## Cost

Weights are fp32 (14.3 GB) whatever the file holds, and each stage owns a
paged plane of the whole context: `stages * context * 2 * width * 4` bytes.
On a 24 GB card, with 2 GB kept free, whatever is left after the weights -
about 7.7 GB - is the pool. That is what sets the depth a card can serve:

| steps | stages | pool at 4096 | total |
|---:|---:|---:|---:|
| 4 | 20 | 3.5 GB | 17.8 GB |
| 6 | 28 | 4.8 GB | 19.1 GB |
| 12 | 52 | 9.0 GB | 23.3 GB |
| 32 (released) | 132 | 22.8 GB | 37.1 GB |

The resident serves 12 steps. At a 6144-token serving context (an agent's
system prompt and tool schemas) only 4-6 steps fit with the full history, or
more with the sharing ring; a card with more memory serves the released
depth. Prefill allocates a slab of the prompt's length, so a prompt close to
the context capacity needs that much headroom on top of the numbers above.

## Using it from an agent runtime

An agent runtime that speaks the OpenAI chat API reaches the served model
through brain's endpoint; the checkpoint's own template renders the
conversation. On a 24 GB card the weights leave room for about 7.7 GB of KV,
so serve with a context the depth fits (12 steps at 4096), and give the
runtime a system prompt that fits that window - a stock agent system prompt
is larger than 4096 tokens:

```bash
brain huginn import --hf <hf checkpoint dir> \
    --out <models>/brain/huginn-0125/model.brain.safetensors --id brain/huginn-0125
cp <hf checkpoint dir>/tokenizer*.json <hf checkpoint dir>/generation_config.json \
    <models>/brain/huginn-0125/
brain serve --device gpu --models-dir <models> --ctx 4096 --reserve-gb 0 \
    --openai 8795 --api-keys-out /tmp/keys.json

curl -s http://127.0.0.1:8795/v1/chat/completions \
    -H "Authorization: Bearer $(jq -r .openai /tmp/keys.json)" \
    -H 'Content-Type: application/json' \
    -d '{"model": "brain/huginn-0125",
         "messages": [{"role": "user", "content": "What is the capital of France?"}],
         "max_tokens": 24}'
```

A near-greedy request (temperature 0.2, an agent runtime's usual default)
loops on these checkpoints; ask for 0.7-0.8. The model answers chat questions
through this path. It does not call tools
(they are not part of its training), so a coding-agent loop over this
checkpoint is out of reach at the depths a 24 GB card can serve; the
released depth needs a larger card.

## Licence

The checkpoint is published under Apache-2.0. Check the model card before
production use.
