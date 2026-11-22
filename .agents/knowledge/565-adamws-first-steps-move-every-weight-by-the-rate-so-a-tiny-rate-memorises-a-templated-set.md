<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 565. AdamW's first steps move every weight by the rate, so a tiny rate memorises a templated set

**Lesson:** AdamW normalises each coordinate, so its early steps move nearly every weight by close to `lr`,
whatever the gradient's scale: the update's norm is about `lr * sqrt(P)` and its first-order loss decrease about
`lr * ||g||_1`, which grows with the parameter count. At `lr` 1e-6 on Qwen3-4B (P = 4.02e9) one step moves the model
by 0.059 (L2) and can take nats off the loss. A served Qwen3-4B `full_train` (4 cards, ZeRO-2, 20 steps, lr 1e-6)
whose loss fell from 4.96 to 0.0009 looked like a multi-card defect (a rate scaled by the world size, an update
applied per rank). It was not: one card falls the same way (to 0.0019, with a quarter of the records a step), the
rate is applied exactly, and the 256 training records are one sentence template ("N plus M is S. We add the ones,
then the tens, ...") that the model has learned once the template is. Two real defects were in the way of seeing it:
a data-parallel step reported rank 0's loss (one quarter of the step's records), not the step's, so the 4-card curve
was not comparable with any 1-card curve; and nothing held N ranks to one rank over the same records.

**How to apply:** judge a full fine-tune's rate by the update it applies (`|delta|` against `lr * sqrt(P)`, per
element against `lr`), not by intuition from LoRA or from SGD. A "too fast" loss drop at a small rate on a small or
templated set is the expected memorisation; test generalisation on held-out records instead. With bf16 weights in
the forward (`train_precision weights=bf16`), updates far below the bf16 spacing of a weight (about 1e-4 at 0.02)
reach the forward only when the f32 master crosses a rounding boundary, so the same run falls only to 1.4. A
data-parallel step's reported loss is the group's (`model::reduce_grads_weighted` and `ZeroOptimizer::reduce` reduce
it with the step's weights); `model::conformance_zero` holds N replicated ranks to one rank accumulating the same N
records a step.

## Evidence (2026-11-22, n442, 4 GH200, served `full_train`, Qwen3-4B, seed 7, max_block 512, 20 steps)

Step-1 update, exported f32 checkpoints against the bf16 base: 1 card accumulating 4 records, 4 cards replicated,
optimizer- and gradient-sharded all moved |delta|_2 = 0.058872 (lr * sqrt(P) = 0.063423), 93.7% of the weights
moved, 82.9% by lr within 5%, max |delta| 1.9e-6 (one f32 ulp at magnitude 16); the 4-card updates agree with the
1-card one to cosine 0.999976 and with each other to 8e-6 relative.

Loss at steps 1/2/3/5/8/10/12/15/20 (the step's loss over its records, after the fix):

| run | curve |
|---|---|
| 1 card, 4 records a step | 4.961 4.299 2.832 1.380 0.648 0.094 0.037 0.0049 0.0009 |
| 4 cards replicated | 4.961 4.298 2.832 1.380 0.648 0.094 0.037 0.0049 0.0009 |
| 4 cards optimizer / gradients | same to 1e-3 |
| 1 card, 1 record a step | 4.960 4.125 2.847 1.593 0.785 0.235 0.066 0.0092 0.0019 |
| 1 card, lr 1e-7 | 4.960 4.894 5.022 4.921 4.803 4.551 4.928 4.198 4.165 |
| 1 card, lr 0 | 4.960 4.899 5.041 4.973 5.007 4.840 5.156 4.872 4.921 |
| 4 cards parameter-sharded, weights=bf16 | 4.963 4.887 4.600 3.655 3.137 1.753 1.819 1.614 1.368 |
| 1 card, weights=bf16 | 4.959 4.782 4.633 3.986 3.420 1.915 1.999 1.699 1.414 |
