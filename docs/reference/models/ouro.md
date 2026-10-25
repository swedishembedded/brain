<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Ouro

A looped language model: one 48-layer transformer stack is executed a
configurable number of times (four in the released checkpoint) for every
token. The embedding is applied once; each pass consumes the previous pass's
final-normed state, ends with an exit gate, and appends its own attention
history. The head is untied and the released weights use sandwich RMSNorm,
multi-head attention with half-split RoPE, and a SwiGLU FFN.

```bash
# Import the Hugging Face checkpoint (single BF16 safetensors + tokenizer).
brain ouro import --hf <hf checkpoint dir> --out ouro.safetensors

# Run the paged serving engine locally (greedy).
brain ouro generate --checkpoint <hf checkpoint dir> --prompt "The capital of France is" --max-new 32
```

`brain serve` picks up an imported `ouro` checkpoint from the model store and
exposes the same `generate` action as the other text models over HTTP, D-Bus
and the SDK. The checkpoint directory must carry its `tokenizer.json`; the
ChatML template declared by the checkpoint is used for chat requests.

## Cost

Attention history is per pass, not shared: a sequence with `T` recurrent
passes owns `T` KV histories, one block table per pass. Each pass has its own
paged plane of the full context capacity, so KV memory is `T` times a
conventional decoder's at the same context, and admission is the per-pass page
count against the tightest plane.

`brain serve --ctx N` sets that capacity. At 4096 tokens the 1.4B checkpoint's
four passes need about 6.4 GB of pool beside 5.4 GB of weights; the 2.6B
checkpoint needs about 12.9 GB beside 10.7 GB, so on a 24 GB card serve it
with `--ctx 2048` (or import and serve the 1.4B when a long prompt, such as an
agent's tool schema, has to fit).

## Using it from an agent runtime

An agent runtime that speaks the OpenAI chat API reaches the served model
through brain's endpoint, tool calls included - the checkpoint is
instruction-tuned and calls tools in the ChatML form its template teaches:

```bash
brain serve --device gpu --models-dir <models> --ctx 6144 --openai 8792 \
    --api-keys-out /tmp/keys.json

curl -s http://127.0.0.1:8792/v1/chat/completions \
    -H "Authorization: Bearer $(jq -r .openai /tmp/keys.json)" \
    -H 'Content-Type: application/json' \
    -d '{"model": "brain/ouro-1.4b",
         "messages": [{"role": "user", "content": "What is the capital of France?"}],
         "tools": [{"type": "function", "function": {"name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
         "max_tokens": 256}'
```

A request with a short budget and reasoning left on can spend the whole
budget inside a reasoning block; the served request's `enable_thinking:
false` switch turns that off where the runtime sets it.

## Licence

The checkpoint is published under Apache-2.0, with the model card stating that
the weights are intended for research purposes only. Check the card before
production use.
