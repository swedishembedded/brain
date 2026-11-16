<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# ouro - a looped decoder from random weights

Trains a tiny Ouro looped decoder from nothing with the SDK and samples it:

1. `brain::ModelInit` learns a byte-level BPE vocabulary on the text, declares the chat and tool-call tokens
   (`<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<|endoftext|>`) and a chat template, checks the
   template against the contract training and serving rely on, and writes a randomly initialised checkpoint;
2. `brain::Pretrain` trains every weight on the text, scoring a held-out tenth of it every quarter of the run, and
   the sample prints the training rate in tokens a second;
3. the trained checkpoint continues a prompt greedily through the model's own decode step.

The same calls create, pretrain and fine-tune any architecture that declares a from-scratch initialization
(`brain::from_scratch_architectures`); the trained directory is a base `brain serve` serves and the served training
actions (`init_model`, `pretrain`, `full_train`) take by name.

## Run it

```bash
make samples/models/ouro/run ARGS="--out out/ouro"                                  # tiny: seconds on a CPU
make samples/models/ouro/run ARGS="--out out/ouro-5m --size default --steps 100"    # hidden 256, 4 layers, 4 passes
make samples/models/ouro/run ARGS="--out out/ouro-5m --size default --device gpu0 --batch 32 --text my-text.txt"
```

`--size default` is about 5.3M parameters: hidden 256, 4 shared layers run for 4 passes, 4 heads, vocabulary 4096,
rows of 1024 tokens; `--size small` is about 1.3M (hidden 128, vocabulary 2048, rows of 512). `--block` and
`--batch` override a size's row length and rows per step. It needs no weights and no download; with no `--text` it
trains on a short built-in text.

## Build

`make samples/models/ouro/build`. It links the SDK's `study` surface and the `brain-ouro` crate for the decode step.
