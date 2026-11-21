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
