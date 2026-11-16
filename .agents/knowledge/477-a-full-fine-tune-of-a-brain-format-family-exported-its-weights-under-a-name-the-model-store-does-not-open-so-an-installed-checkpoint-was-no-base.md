<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 477. A full fine-tune of a brain-format family exported its weights under a name the model store does not open, so an installed checkpoint was no base

**Lesson:** A writer and the reader that later resolves what it wrote share one name for the file, defined once next
to the writer. A round trip test goes through the reader a caller uses (`vendor/repo` in the store), not through a
path the test already knows.

## What happened

`model::finetune::export_brain_checkpoint` (the export of every family without a `transformers` writer: qwen35,
qwen35moe, ricursive, ouro) wrote the trained weights as `model.safetensors`. The model store opens a directory
without a `config.json` by `model.brain.safetensors` only (`Store::local`), so a checkpoint installed by `full_train
install_as` resolved to nothing: neither a later job nor `brain serve` could name it. The qwen3 path did not show it,
because qwen3 exports a `transformers` directory whose `config.json` the store reads. The existing tests opened the
exported file by its path, never by store name.

The same directory also could not be told apart by architecture: with no `config.json`, the SDK's fine-tuners took
any such directory for `qwen3`.

## What changed

- `export_brain_checkpoint` writes `model::finetune::BRAIN_WEIGHTS_FILE` (`model.brain.safetensors`), and the SDK's
  full fine-tune reports and digests whichever weights file the export wrote.
- `sdk::study::arch_of_checkpoint` tells a directory without `config.json` by the card of its brain checkpoint (the
  registry row whose family the card names).
- `chat_full_finetune_ouro` trains an Ouro base, then trains its exported checkpoint again by directory, through the
  store's resolution.
