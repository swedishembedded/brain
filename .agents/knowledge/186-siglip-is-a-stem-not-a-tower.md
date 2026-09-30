<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 186. SigLIP is a stem around CLIP's blocks, not a second tower

timm's SigLIP (DeepSeek-VL's low-resolution tower, Janus-Pro's
understanding tower) and Moondream's vision encoder are both the pre-LN
ViT block that `ClipVision` already runs. What differs sits outside the
blocks:

| | CLIP stem | SigLIP stem |
|---|---|---|
| class token | prepended, position row 0 | none |
| patch conv | `bias=False` | biased |
| before block 0 | `pre_norm` | nothing |
| after the last block | nothing | `post_norm` (timm `norm`) |
| MLP activation | quick-GELU | erf GELU (timm), tanh GELU (Moondream) |

So `ClipVisionConfig` has a `stem` field, and the activation comes from the
config. The blocks, the position-table resample and the backward are one
implementation. SigLIP-L/16@384 on the real DeepSeek-VL weights gives
cosine 1.0000000 and relative L2 3.9e-5 after the final norm.

Four facts that are easy to get wrong:

- **`siglip_large_patch16_384` is not a timm registry name.** It is a row
  in DeepSeek-VL's own `siglip_vit.SigLIP_MODEL_CONFIG`, read by its own
  copy of timm's `VisionTransformer` (`create_siglip_vit`). That copy
  hard-codes `nn.LayerNorm(eps=1e-6)` and `nn.GELU` whatever the caller
  passes. Janus carries the same file.
- **`select_layer` truncates the depth, and the final norm still runs.**
  `create_siglip_vit` builds `layers + select_layer + 1` blocks, and the
  encoder returns `forward_features`, which ends with `norm`. This is not
  HF CLIP's `hidden_states[-k]`, which taps a raw block output before any
  norm. At `select_layer=-1` the output is the post-norm of all 24 blocks.
- **The checkpoints carry a MAP head that never runs.** `global_pool="map"`
  builds `attn_pool.*` (13 tensors), but both models construct the tower
  with `ignore_head=True`. The importer skips those names from an explicit
  list, so any other stray tensor still stops the import.
- **Moondream's patch embedding is a conv with permuted columns.** Its
  `patch_emb` is `Linear(588, 1152)` over patches flattened `(y, x, c)`.
  A `Conv2d(3, 1152, 14, stride=14)` whose weight runs `(c, y, x)` computes
  the same per-patch dot product. The importer permutes the columns once
  (`patch_linear_to_conv`), and crops reach the tower as planar images.

The shared tower resolves every kernel by name (`VitKernelIds::by_name`)
because its callers register their own kernel lists. On such a device a
position in the tower's own list selects a different pipeline.
