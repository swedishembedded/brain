<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 191. DeepSeek-VL's processor does not normalize; each tower branch does

DeepSeek-VL's `preprocessor_config.json` lists CLIP's mean and standard
deviation, but it also sets `do_normalize: false`. The processor's
`pixel_values` are the padded square rescaled to `[0, 1]`, and nothing more.
The mean is used for one thing only: the padding colour
`int(image_mean * 255)` (122, 116, 104).

Normalization happens inside the hybrid tower, once per branch and with each
branch's own statistics:

| branch | resize before it | mean / std |
|---|---|---|
| SAM-B, 1024 px | none | CLIP's (0.481, 0.458, 0.408) / (0.269, 0.261, 0.276) |
| SigLIP-L, 384 px | torch bilinear, antialiased, of the `[0, 1]` square | 0.5 / 0.5 |

So the low branch resizes the unnormalized image, and the padding the
processor adds is the unnormalized background colour. Normalizing in the
processor, as `do_normalize: true` would, would give both branches CLIP's
statistics and SAM a differently coloured border.

Two more facts the composite depends on:

- **`language_config` is partial.** It names only the depth (30), the
  vocabulary (102400) and the context (16384). The width, heads and
  feed-forward size are `transformers.LlamaConfig`'s defaults (4096, 32,
  11008), and so is `rms_norm_eps` (1e-6). `qwen3::hf::decoder_config` fills
  them in from the class defaults, per architecture.
- **The checkpoint carries `high_layer_norm` and `low_layer_norm`, which the
  forward never applies.** `HybridVisionTower.__init__` builds them and
  `forward` ignores them. `deepseekvl::import::UNUSED` lists them by name, so
  any other tensor that no component reads still fails the coverage check.

The gate is `crates/deepseekvl/tests/composite_parity.rs`. It checks each
stage on the real checkpoint and requires 16 greedy tokens identical to the
fp32 reference's.
