<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 490. The tensor-core flash attention was written for heads of 128 and 256, so a 64-wide head trained on the portable kernel at a fifth of the rate

**Lesson:** A native kernel whose width is a compile-time constant serves exactly the widths someone instantiated;
a model of another width falls back without a word and the profile shows the portable kernel at the top. Derive
every count from the width, instantiate the widths models use, and gate each against the portable kernel.

## What happened

After [489](489-the-looped-decoder-trained-on-the-card-through-cuda-core-gemms-and-a-quadratic-loss-gradient-so-a-tiny-model-ran-at-a-tenth-of-the-card.md)
the default from-scratch Ouro (hidden 256, four heads of 64) spent 85% of its training step in the portable flash
trio: `flash_causal_gqa_fwd_step` took head widths 128 and 256, `flash_causal_gqa_bwd_steps` 128 only, and both
answered `None` for 64.

## What changed

- `flash_prefill_f16_hd256.cu` (the dense training forward) and `flash_attn_bwd_f16.cu` take `BRAIN_HD` 64: their
  tile counts already derived from the width, so only the softmax scale was per width; the backward's `dO` scale
  pass reads the width from its uniform, so one instance serves every width.
- `flash_attn_fwd_f16_hd64`, `flash_bwd_dq_hd64` and `flash_bwd_dkv_hd64` are registered and selected by width;
  `cuda_flash_attn_{fwd,bwd}` gate them against the portable kernels and an f64 oracle at 64 as at 128 (forward
  6e-3 of the output RMS, backward cosine 1.000000).
- Measured on gpu3 (`samples/models/ouro`, default size, rows of 1024, step rate after the first step): batch 8
  34,400 -> 173,600 tokens/s; the step losses move by at most 4e-4 from the portable kernel's (fp16 operands).
  The batch sweep flattens at about 250k tokens/s: rows of 1024 at batch 8 174k, 16 215k, 32 235k, 64 245k,
  128 254k, 256 254k; rows of 512 at 16 185k, 64 254k, 256 270k (256 x 1024 tokens a step fit the card).
