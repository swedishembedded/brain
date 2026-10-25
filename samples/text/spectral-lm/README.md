<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# spectral-lm - a language model with a Fourier-spectrum hidden state

SpectralLM is a decoder whose residual stream is the lowest `m` coefficients
of a real Fourier basis of a longer virtual signal, and whose feed-forward
sublayers are spectral mixers: per-mode filters around a gated product taken
in the signal domain (`crates/spectrallm`). This sample trains it from scratch
on a corpus of documents, trains the conventional control beside it (the same
decoder with Qwen3's SwiGLU feed-forward network, which is the Qwen3
architecture), scores both and the pretrained Qwen3-0.6B in bits per byte on
the same held-out documents, and samples text from a trained model.

It is the program behind the measurements of the spectral-lm paper
(`papers/spectral-lm`).

## What it needs

- A GPU for training at any useful size (the CPU backend runs the same graph,
  slowly).
- The TinyStories corpus, prepared by brain's generic dataset command:

  ```sh
  mkdir -p data/tinystories && cd data/tinystories
  R=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
  curl -L -o train.txt $R/TinyStoriesV2-GPT4-train.txt
  curl -L -o valid.txt $R/TinyStoriesV2-GPT4-valid.txt
  cd ../.. && brain data gen tinystories --out data/tinystories
  ```

  This learns a 4096-token byte-level BPE on the first 30,000 training stories
  and writes `train.bin`, `val.bin`, `meta.json`, `tokenizer.json` and
  `stats.json`. TinyStories is distributed under CDLA-Sharing-1.0.
- For the Qwen3 comparison, the Qwen3-0.6B checkpoint (`brain pull
  Qwen/Qwen3-0.6B`).

## Build and run

```sh
make samples/text/spectral-lm/build
S=./target/release/sample-text-spectral-lm

# SpectralLM: 256 coefficients of a 512-sample signal, 24 mixer channels.
$S train data/tinystories --out runs/spectral.safetensors \
   --d-model 256 --layers 6 --heads 8 --signal-len 512 --channels 24 --steps 4000
# The control: the same decoder with a SwiGLU of hidden width 1024.
$S train data/tinystories --out runs/swiglu.safetensors \
   --mixer swiglu --d-model 256 --layers 6 --heads 8 --d-ff 1024 --steps 4000

# Bits per byte on the held-out stories.
$S eval --checkpoint runs/spectral.safetensors \
   --tokenizer data/tinystories/tokenizer.json --text data/tinystories/valid.txt --max-doc-bytes 1500
$S qwen-eval --weights ~/.local/share/brain/models/Qwen/Qwen3-0.6B/model.brain.safetensors \
   --text data/tinystories/valid.txt --max-doc-bytes 1500

# Text.
$S generate --checkpoint runs/spectral.safetensors \
   --tokenizer data/tinystories/tokenizer.json --prompt "Once upon a time"

# Parameters and analytic arithmetic of a shape, without training it.
$S cost --d-model 256 --layers 6 --heads 8 --signal-len 512 --channels 24
```

`qwen-corpus` encodes the same stories with Qwen3's tokenizer and
`qwen-train` continues Qwen3-0.6B on them with the same schedule flags, for
the fine-tuned reference.

Every training and scoring subcommand ends with one JSON line holding the
numbers a comparison records (losses, tokens, time, parameter counts,
arithmetic per token, bits per byte and what it was computed over). The seed
of every run is random unless `--seed` is given, and is always printed.

## What it demonstrates

That the spectral mixer trains through brain's generic training loop with its
analytic backward (gradient-checked in `crates/gradcheck`), that its FFT
transform path computes the same model as the dense basis product, that the
model decodes incrementally from a key/value cache, and how it compares with
a width-matched conventional decoder and a pretrained one on the same text.
