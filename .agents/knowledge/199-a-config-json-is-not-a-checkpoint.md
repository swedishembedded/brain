<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 199. A `config.json` beside GGUF files is metadata, not a checkpoint

`brain pull unsloth/Qwen3.8-27B-GGUF-Q8_0` failed with "no weights found
(safetensors or pytorch_model.bin ...)", though the repo plainly has the file.
`GgufRecipe::matches` refused any repo with a root `config.json`, on the
reasoning that a bare `config.json` means a transformers checkpoint owns the
repo. Current GGUF releases ship the model's `config.json` beside their
quantizations, so the GGUF recipe declined, the planner fell through to the
transformers catch-all, and that recipe looked for safetensors in a repo that
has none.

The signal that another family owns a repo is a weight file in another format
(`.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx`) or a `model_index.json`, not a
metadata file. `.bin` was also missing from that list, so a `pytorch_model.bin`
repo with a stray `.gguf` would have been claimed wrongly; it is there now.

Regression test: `gguf_recipe_claims_a_gguf_release_that_ships_a_config_json`,
built from the real listing shape (nested `BF16/` and `MTP/` directories,
multimodal projectors, `UD-` quantizations).
