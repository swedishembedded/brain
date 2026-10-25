<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# spectrallm

Scope: SpectralLM, the decoder with a truncated real-Fourier residual stream and per-mode spectral mixers (`crates/spectrallm`), its operators, and the comparisons its paper reports (`papers/spectral-lm`).

The paper (`papers/spectral-lm/`) holds the method, the registered protocol
and every measured result; this ledger holds what is built, what is open and
why the scope is what it is.

## Decisions

- **The basis only matters through basis-structured operators.** A decoder
  whose residual-stream maps are dense absorbs any orthogonal basis (the
  paper's basis-absorption proposition). A "rotated" transformer and a
  truncated-spectrum transformer with dense maps are therefore Qwen3-shaped
  decoders of width `D` and `m`, trained with `brain qwen3 train`; they get no
  code of their own.
- **The spectral part is the feed-forward mixer**: per-mode filters, a fixed
  synthesis/analysis basis, SwiGLU in the signal domain. Attention, norms,
  RoPE, the KV cache, the optimizer and the training loop are the shared
  `model::block` / `model::fit` pieces, so SpectralLM and its controls differ
  only in the mixer.
- **Two transform paths.** A dense product with the stored basis is the
  reference; a real FFT kernel is the fast path, gated against it. The FFT is
  in scope because a dense-product mixer costs `6CDm` per token, which removes
  the arithmetic advantage the architecture exists for.
- **Corpus is TinyStories with a small byte-level BPE**, metric bits per byte,
  so Qwen3-0.6B (its own tokenizer) is scored on the same text.
- **Out of scope:** exact spectral reparameterization of Qwen3, distillation
  into a spectral student, Qwen3-8B, bf16, CUDA-native kernels, distributed
  training, dynamic mode budgets, HTTP/D-Bus serving, FourierFT adapters.

## Status

| Item | State |
|---|---|
| Paper skeleton, registered protocol, verified bibliography | done |
| Real Fourier basis host math (`model::hostmath`) | open |
| Mixer kernels: channel expand/contract/filter-grad, synthesis/analysis (dense + FFT) | open |
| `crates/spectrallm`: config, init, forward, backward, decode, save/load, arch row, CLI | open |
| Gradient checks, CPU == GPU parity, fixed-batch memorization | open |
| TinyStories dataset + byte-level BPE in `data::prepare` | open |
| Bits-per-byte evaluation | open |
| Comparison runs (SpectralLM vs narrow/wide Qwen3-shaped controls, 3 seeds) | open |
| Qwen3-0.6B: zero-shot BPB, fine-tuned BPB | open |
| Qwen3-0.6B hidden-state spectral structure vs PCA / random bases | open |
| Transform cost: dense vs FFT mixer | open |
