<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 175. A scaling read by one spelling drops every other

`qwen3`, `qwen35` and `lfm2` each had their own copy of the `rope_scaling`
reader. All three kept the object only when `type == "yarn"` and turned
anything else into `None`, which means plain RoPE. The reader never errored,
so each of these checkpoints loaded and ran at frequencies it never declared:

- **`linear`** scaling. deepseek-coder 1.3b/6.7b declare `{"type": "linear",
  "factor": 4.0}`. Unscaled, every position rotates at 4× the rate the model
  was trained with.
- **`llama3`** scaling. R1-Distill-Llama-8B and every Llama 3.1+ checkpoint
  declare it; the low-frequency channels ran at up to 8× their trained
  wavelength.
- **YaRN under the current spelling.** Current transformers writes
  `rope_type`, not `type`, so a YaRN checkpoint it saved was dropped too.
  That is the one scheme all three readers claimed to support.

A test in `qwen3` and in `qwen35` named `rope_scaling_ignores_a_non_yarn_type` asserted
the drop as the intended behaviour. Its reasoning was that YaRN's formula
must not run under another name. That is true, but refusing the config would
have achieved it without running the model unscaled.

`model::rope_scaling::RopeScaling` is now the only reader, for all three
crates:

- It reads `rope_type`, then falls back to `type`.
- It errors on any type it does not implement, and on a missing required
  field.
- Each checked config reader (`from_json_checked`) returns that error.

Consumers take the result as one inverse-frequency table
(`RopeScaling::inv_freq`, through each config's `rope_table()`), so a new
scheme is one match arm here and not three.

Tolerance of the table: `crates/model/tests/rope_scaling_golden.rs` compares
it with the `rotary_emb.inv_freq` buffer that transformers builds for each
DeepSeek text checkpoint. The bound is 2 f32 ULP. A wrong formula misses by
orders of magnitude more than that.
