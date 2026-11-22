<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Inkling 2

A mixture-of-experts decoder whose sublayers do not read a plain residual
stream: each reads a softmax over depth of the model's own block states (one
state per six-layer block plus the current partial sum), mixed by learned
queries. The released `thinkingmachines/Inkling-2-Mini`
(`Inkling2LoopForConditionalGeneration`) runs its middle twelve layers three
times per token; passes after the first restart from the prelude's states and
read the previous pass only through gated messages. The plain family member
(`Inkling2ForConditionalGeneration`) is the same recipe without the loop.

| Part | Released configuration |
|---|---|
| Decoder | 24 layers, hidden 2304, 48 layer visits per token (6 + 3 x 12 + 6) |
| Attention | 24 query and 8 KV heads of width 128, no RoPE: a hidden-conditioned relative bias over a learned distance bank, per-token sink logits, a sigmoid output gate; 20 sliding layers (window 1024) and 4 full ones |
| Convolutions | depthwise, 4 taps, after K, V, attention and MoE; dynamic per-token taps on the full layers' K and V |
| MoE | 128 experts, top 8, sigmoid router with a selection bias, a softcapped gated activation |
| Head | untied, 200,058 valid rows, logits softcapped at 30 |
| Weights | experts NVFP4 (layers 0-19) and MXFP8 (20-23), everything else BF16 or F32, as `hf_quant_config.json` declares per tensor |

The checkpoint is read as published: `config.json` is parsed strictly (an
unknown key or a layer list the reference would read differently is refused
by name), and every tensor of the file must be one the configuration accounts
for, with its shape and an encoding its role admits.

## Serving

```bash
brain pull thinkingmachines/Inkling-2-Mini
brain serve --dbus --openai 8788      # served under its pulled id
brain do inkling2 generate --prompt "Hello" --max_new 64   # in process, no server
```

The pulled directory is read in place: the expert banks stay in their NVFP4 and
MXFP8 encodings, the other weights at their stored tier, and activations are
f32, the reference's numerics. Requests go through the same continuous-batching
scheduler as every other model: prompts are read in chunks of 512 tokens under
the scheduler's prefill budget, every running sequence decodes in one batched
step, and the expert products are grouped by expert while reading a prompt and
gathered per slot while decoding.

Each of the 48 layer visits keeps its own state per sequence: the 8 full-attention
visits keep keys and values in a paged pool (one block table covers all of them),
the 40 sliding visits a ring of exactly their 1024-token window, and every visit
the inputs its four short convolutions read next. A sequence therefore costs
64 KiB per token (f32 keys and values of the 8 full visits) plus a fixed state
row of about 340 MB (the rings and the convolution states), whatever its length.
`brain serve --kv-cache bf16` keeps every key and value row (the paged planes
and the rings) in bf16 instead: half the bytes per token, at a measured cost
in precision against f32 (the default, the reference numerics); `int8` and
`fp8` are refused by name. `brain serve --ctx N` caps one
sequence (default 16384 tokens; the model's positions reach 1M) and
`--max-batch N` is how many run at once; the pool holds that many contexts and
as many state rows again for cached prompts.

A prompt of 32 tokens or more is cached 16 tokens short of its end, keyed by its
tokens (and by the content of any embedding rows), so the next turn of a
conversation, which re-renders the last tokens of this one, resumes from it and
reads in only what is new.

Chat is the checkpoint's own template. Replies are typed messages: a thinking
message, the answer, tool calls; the reply carries the reasoning in
`reasoning_content`, the answer as the text and calls in `tool_calls`. A
generation stops only on the sampling end the checkpoint declares
(`<|content_model_end_sampling|>`): the message boundary closes the thinking
message before the answer too, so it never stops one. `reasoning_effort` is the
template's own variable (`low`, `medium`, `high` or a number from 0 to 1); any
other value is refused by name.

## Images

The vision tower is loaded onto the decoder's card and reads the images of a
chat request: an OpenAI `image_url` part or an Anthropic `image` block with an
inline `data:` URL (PNG, JPEG or PPM; no URL is fetched), or over D-Bus the
blobs `image`, `image1`, ... in the order of the messages' image parts. Each
image part becomes the template's placeholder, expanded to one token per
40x40-pixel region of its image (the image is padded, never resized), and the
tower's rows stand at those tokens. An image may be at most 4096 pixels a side,
a request carries at most 8, and all of them together must need fewer tokens
than the context holds; an image part without its image (or the reverse), an
image in a raw prompt (`chat: false`) or in a system or assistant message is
refused as the client's error. Transparency does not reach the model: the HTTP
surfaces decode images to RGB.

```bash
curl -s localhost:8788/v1/chat/completions -H "Authorization: Bearer $KEY" -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 photo.png)"'"}},
    {"type": "text", "text": "What is in this picture?"}]}]}'
```

Adapters: the engine serves LoRA adapters of the trainer's linear and expert
sites per sequence (`inkling2::serve::Engine::enable_adapters`, `load_adapter`),
but `brain serve` does not yet offer an Inkling 2 adapter as a model of its
own. Not served yet: that.

## Speculative decoding

The release ships a drafter beside the decoder, `mtp.safetensors` (636 MB): a
three-layer block drafter that proposes the next 7 tokens of a sequence in one
forward. It reads the decoder's own hidden states at four logical depths (the
state entering visits 5, 23 and 41, and the final normalised row), denoises a
block of the last token plus 7 learned mask rows with attention that is not
causal, and a small recurrent head walks the block left to right into one
coherent proposal. The scheduler verifies every sequence's proposals in one step
and keeps the longest prefix the model agrees with, so greedy output is the same
as without it; a sampled request goes through rejection sampling.

```bash
brain serve --openai 8788 --speculative mtp     # the release's drafter
brain serve --openai 8788 --speculative ngram   # prompt lookup, no second model
```

Speculation is opt-in for this model (`auto` and `off` decode plainly): what it
buys depends on how much cheaper verifying several rows is than decoding them one
at a time on the device, and that is not measured yet. With it, every running
sequence keeps a second state row (a verify writes ahead into the sliding
windows and convolution states and is rolled back when proposals are rejected),
the drafter's weights and caches are resident, and the four hidden states of every
cached position are kept in host memory (18 KB per position). `dflash` is refused:
this model has no DFlash2 draft model. Adapter requests decode without speculation.

## Training

Inkling 2 trains through the generic fine-tuners (`brain finetune`, the SDK's
`ChatFineTune`, `PreferenceFineTune`, `ChatFullFineTune` and policy gradient;
the architecture is read from the checkpoint's `config.json`, or named with
`.arch("inkling2")`):

```bash
# LoRA over the release as published (QLoRA: the expert banks stay NVFP4/MXFP8, never dequantized)
brain finetune thinkingmachines/Inkling-2-Mini --lora 16 --dataset train.jsonl --out runs/sft
# DPO (also --loss ipo|simpo|orpo), KTO, distillation
brain finetune thinkingmachines/Inkling-2-Mini --lora 16 --objective preference --dataset pairs.jsonl --out runs/dpo
# GRPO-family policy gradient on rollout records (reward, group)
brain finetune thinkingmachines/Inkling-2-Mini --lora 16 --objective policy-gradient --dataset rollouts.jsonl --out runs/pg
```

- The adapter covers every site: the attention's q, k, v, o, output-gate,
  relative-state and sink linears, the depth, readout and message queries and the
  message gates (as `W + s B A` pairs over those small matrices), and one pair
  per expert over the stacked expert banks, applied on the rows routed to it.
  The dense linears are held in bf16 as published (`--base-format bf16`) or
  widened to f32 (`--base-format f32`); the router always routes in f32.
- Every layer visit is recomputed from its saved input in the backward (the
  core's three passes accumulate into one set of gradients), so activation
  memory is the visits' input rows and the block states.
- MoE routing replay: a rollout's recorded expert choices are forced on the
  trainer's forward (`GrpoOptions::routing_replay`), so a policy gradient trains
  the function that sampled.
- Packed rows (several documents per row, attention and every convolution
  state restarting at each document) and image-text rows (the vision tower's
  features spliced at the image placeholders, the tower frozen or adapted) train
  too.
- A full fine-tune trains every weight in f32 for a checkpoint whose experts are
  stored as floats; the released 26B is refused for it by name (it would need a
  streamed, sharded or expert-parallel full fine-tune).
- A trained adapter is not served yet (the engine holds no adapter bank, and the
  expert pairs cannot be folded into the quantized banks).

The example `samples/python/llm/inkling2/inkling2.py` asks one question at a low
and a high reasoning effort over D-Bus, OpenAI-compatible or Anthropic-compatible
HTTP.

## Numerics

By default the experts multiply their published NVFP4 and MXFP8 weights with
f32 activations: the reference implementation's numerics, which brain is
parity-gated against. The release also declares the activation quantization
its own runtime applies to each expert product (`required_activation_quantization`
in `hf_quant_config.json`: NVFP4 with a per-token scale and the "four over six"
block-scale rule on layers 0-19, MXFP8 on 20-23). `brain serve
--linear-activations declared` (in the library,
`Inkling2::set_linear_activations(LinearActivations::Declared)`) runs those
numerics in every product, while reading a prompt and while decoding; it is
lower precision, so it is never the default. How far it moves the next-token
distribution from the reference numerics (KL and top-1 agreement on the released
checkpoint) is measured by the `activation_tier` test of `crates/inkling2`;
measure it on your own prompts before relying on it.
