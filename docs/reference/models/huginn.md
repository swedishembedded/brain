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

The released template has no slot for tool schemas. A request that carries
tools gets them described in a system turn, with the reply asked for in
`<tool_call>` JSON - the wire form brain's scanner reads. A checkpoint whose
template does carry a tools slot is unaffected.

## Recurrence and staging

`num_steps` is a serving setting (`--steps` on the CLI action, the
checkpoint's `mean_recurrence` when absent). Every step applies the same four
core blocks to a different residual stream, so each *stage* - a prelude
block, a core block within one step, a coda block - owns its own attention
history. There are `n_prelude + steps * n_recurrent + n_coda` stages.

## Cost

KV memory is `stages` times a conventional decoder's at the same context:
each stage has a paged plane of the whole context capacity. At 2048 tokens
the 3.57 B checkpoint's 20 stages (4 recurrent steps) need about 1.7 GB of
pool beside 14.3 GB of weights and fit one 24 GB card; the released default
of 32 steps is 132 stages, about 11.4 GB at the same context, and does not
fit beside fp32 weights - serve the released depth with a larger card, or a
shorter `num_steps` (the reference itself treats the depth as a runtime
choice).

## Licence

The checkpoint is published under Apache-2.0. Check the model card before
production use.
