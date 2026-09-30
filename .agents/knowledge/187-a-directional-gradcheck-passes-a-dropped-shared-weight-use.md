<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 187. A directional gradcheck passes a shared weight whose second use is dropped

DeepSeek-VL's SAM tower runs its two compressor convs twice per forward:
once on the main path, once on the HD branch, whose output is added scaled
by a learned `hd_alpha`. The weight gradient is the sum over both uses. In
`sam1` the sum comes for free, because `conv2d_dw` accumulates into the
ParamStore gradient and both heads' backwards run.

To check that the gate can see a missing use, the HD branch's share was
dropped (both compressor gradients restored to their pre-HD-backward values).
At `hd_alpha = 0.7` the HD share is a large fraction of each gradient. Even
so, `directional_check` (best of four ±1 directions) caught it on only one
of the two tensors:

| tensor | directional | per entry |
|---|---|---|
| `compress.conv2.weight` | red, rel 0.43 | red |
| `compress.conv1.weight` | **pass** | red (first entries rel 0.74, 0.85, 1.86) |

A ±1 projection of a gradient with one use missing is still a projection of
something the right shape and sign. Taking the best of four projections
favours whichever one hides the gap. This is the same blindness the windowed
pad rows showed for `attn.qkv.bias` in `crates/sam1/tests/gradcheck.rs`.

Rule: a weight read at more than one site (shared, tied, applied twice) gets
an `elementwise_check`, not only a directional one. Run the check at a
coupling that is well away from zero. At the checkpoint's own
`hd_alpha ≈ 5.6e-3`, or at its zero init, the second use is below
finite-difference resolution, so the check passes whether or not that use is
accumulated. The gate is `gradcheck::check_sam_hd_shared`.
