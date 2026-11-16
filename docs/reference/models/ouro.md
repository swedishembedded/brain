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
brain ouro import --source <hf checkpoint dir> --dest ouro.safetensors

# Run the paged serving engine locally: the checkpoint the store holds, or the
# one --weights names; --ctx and --block size the engine.
brain ouro generate --prompt "The capital of France is" --max-new 32 --temp 0

# Train it in place: AdamW at a constant rate over windows of the text, the released objective.
brain ouro train --steps 32 --seq 128 --batch 2 --lr 2e-5 --save trained.safetensors
```

`--activations recompute` is what `train` already does: ouro rebuilds every
layer of every pass from its residual in the backward, so `--activations keep`
is refused by name before the first step.

## Training

The objective is the released one: every pass's cross-entropy weighted by
that pass's exit probability, summed over passes, averaged over supervised
rows, through the analytic backward, run by brain's one training loop (the
same loop, exact resume and evaluation every architecture's trainer runs on).
`brain ouro train` draws windows of its text from a seed (`--seed`, random and
reported when omitted) and reports the loss every step; the gradient itself is gated by finite differences over
every parameter on cpu, wgpu and cuda. Measured: the tiny fixture checkpoint
falls from 3.37 to 1.99 nats in 40 steps; the released 2.6 B checkpoint falls
from 1.76 to 0.20 in 6 steps on a single 64-token batch (it overfits by
construction, and the six-step update visibly shifts its style - it answers a
completion request with tool syntax). The saved checkpoint loads and serves
through the OpenAI endpoint like any other; serve it at a context the card
holds (the 2.6 B's four-pass KV needs about 3 GB per 1 k tokens on top of its
13 GB of weights).

Full fine-tuning of the released checkpoint does not fit one 24 GB card -
weights, gradients and AdamW moments are four times its 10.7 GB of fp32
weights - so a real run updates on the host tier; the gradients and the
update are the same arithmetic on any backend.

`brain serve` picks up an imported `ouro` checkpoint from the model store and
exposes the same `generate` action as the other text models over HTTP, D-Bus
and the SDK. The checkpoint directory must carry its `tokenizer.json`; a chat
request is rendered by the chat template the checkpoint ships
(`tokenizer_config.json`, `chat_template.jinja` or `chat_template.json`),
whatever markup it writes, and generation stops at the checkpoint's declared
end-of-turn token. A checkpoint installed by the training service as
`local/<name>` is served under that name. The advertised context is the pool's
capacity (`--ctx`), at most the model's positions.

## Chat fine-tuning and training from scratch

Ouro trains through the generic full-parameter trainer (`brain::ChatFullFineTune`,
the training service's `full_train`): every weight trains on
`generic-messages-v2` records, each supervised turn masked by the template's own
rendering, f32 weights and moments. There is no LoRA trainer for the looped
stack; an adapter run is refused by name.

It also declares a from-scratch initialization, so the whole sequence runs on it
(and on every architecture `brain::from_scratch_architectures` lists, qwen3
included):

1. `brain::ModelInit` (served: `init_model`) learns a byte-level BPE vocabulary
   on a text corpus with the special tokens you declare, checks your chat
   template and writes a seeded random checkpoint of your configuration;
2. `brain::Pretrain` (served: `pretrain`) trains every weight on raw text with
   held-out evaluation;
3. `brain::ChatFullFineTune` (served: `full_train`) teaches it the chat format
   and tool calls.

Each served step can be resumed from its exact state with a `resume_key`; see
[Training as a served job](../../guides/training/served-training.md).
`samples/models/ouro` runs steps 1 and 2 on a CPU in seconds.

### The chat template a from-scratch model meets

`init_model` refuses a template that does not hold to what training and serving
rely on:

- it compiles, and a conversation renders with its last assistant turn a
  suffix of the prompt the model is asked from (the boundary training masks
  by, and the prompt serving renders);
- the text it writes after an assistant turn contains the `eos_token`, so the
  model learns to stop and serving stops it;
- when it has a place for tools (`tools` in its source), it writes an assistant
  tool call in a form the serving parser reads back:
  `<tool_call>{"name": "<function>", "arguments": {...}}</tool_call>` (the
  arguments a JSON object; read them with `arguments | tojson`), or the
  `<function=...>` form when the template writes it. A served reply then
  carries OpenAI `tool_calls`.

A training record and a served request with the same messages and tools render
to the same tokens.

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

Sampling matters for these checkpoints: a near-greedy request (temperature
0.2, an agent runtime's usual default) loops on repeated tokens over a
multi-turn tool conversation, while 0.7-0.8 answers cleanly. A request with
a short budget and reasoning left on can also spend the whole budget inside
a reasoning block; the served request's `enable_thinking: false` switch turns
that off where the runtime sets it.

## Licence

The checkpoint is published under Apache-2.0, with the model card stating that
the weights are intended for research purposes only. Check the card before
production use.
