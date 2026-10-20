<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 338. Serving past the declared window derives Qwen's YaRN recipe, and auto-sizing no longer extends silently

Two defects in how `QwenResident` (`crates/catalog/src/resident_llm.rs`) turned a context past the
checkpoint's `max_position_embeddings` into RoPE scaling.

1. The derived scaling was not Qwen's recipe. It used `factor = ctx / max_position_embeddings` with
   `original = max_position_embeddings`. Qwen3 declares 40960 (32768 pretrained plus 8192 of generation headroom), so
   `--qwen-ctx 131072` got factor 3.2 over 40960; the published recipe is `rope_type yarn`, `factor 4.0`,
   `original_max_position_embeddings 32768`. The correction range (`find_correction_range`) and the attention factor
   (`0.1 ln factor + 1`, 1.1386 for 4.0) both move with those two numbers. Now `original` is the largest power of two not
   above the declared window (`pretrained_window`: 40960 -> 32768, power-of-two windows unchanged) and `factor` is
   `ctx / original`, so 131072 on a Qwen3 is exactly the recipe. A test pins the derived config at 131072 and the
   frequency table it produces against `model::yarn`'s reference values (head_dim 128, theta 1e6: first channels
   1, 0.80584, 0.64938, ...; attention factor 1.1386294), which were re-derived independently from the
   `transformers` `_compute_yarn_parameters` formulas (correction range 23..40 of 64 channels).
2. Auto-sizing (`brain serve` without `--qwen-ctx`) was uncapped, so a card with room picked 131072 or 262144 and the
   derived YaRN then rescaled every request of a server nobody asked to extend (short prompts change: on the 8B, 0 of 12
   short summaries were byte-identical between no YaRN and YaRN(4, 32768)). Auto-sizing now stops at the declared
   window; going past it needs `--qwen-ctx`.

Evidence on the 8B (runs/b8/yarn-s2.json auto-derived factor 3.2, yarn-s3.json recipe declared in config.json): the needle
test (8k, 32k, 64k, 100k, 125k tokens at depths 0.1/0.5/0.9) retrieved 15/15 in BOTH arms, so that test does not
discriminate the two scalings and neither was ever shown to be worse; the recipe is adopted because it is the documented,
reference-checked one. Short-context answers differ between the arms (YaRN rescales all lengths). 128k stays an explicit,
off-by-default choice. Not re-run after this change: serving 131072 on the 8B needs about 76 GB (fp32 weights, fp32 KV).
Check to run in a card window: serve the plain checkpoint with `--qwen-ctx 131072 --qwen-max-prefill 128` and compare
with the config-declared recipe arm; the answers must now be byte-identical.
