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
| Real Fourier basis host math (`model::hostmath::real_fourier_basis`) | done |
| Mixer kernels: `chan_scale_{expand,contract,grad}`, `rdft_{synth,analysis}` (FFT) beside the dense basis product | done |
| `crates/spectrallm`: config, init, forward, backward, KV decode, save/load, arch row, docs page | done |
| Gradient checks (both mixers, both transforms, tied embedding elementwise), CPU == GPU incl. the flash path, Qwen3 parity of the SwiGLU control, decode == forward, fixed-batch memorization | done |
| TinyStories + 4096-token byte-level BPE (`brain data gen tinystories`), bits-per-byte scorer (`model::score`) | done |
| `samples/text/spectral-lm` (train, eval, generate, cost, profile, Qwen3 eval/corpus/train/spectrum) and the paper's driver (`papers/spectral-lm/scripts/run.py`) | done |
| Learning-rate selection, primary comparison (3 seeds), pilots | open |
| Qwen3-0.6B: zero-shot BPB, fine-tuned BPB, residual-stream spectrum study | open |
| Transform cost table (dense vs FFT mixer) for the paper | open |

## Measured while building

- At m=256, D=512, C=24, 6 layers, batch 16 x 512 on a Tesla P40 (Vulkan),
  the first profile spent most of a training step in reference kernels the
  shared blocks already have faster siblings for: the materialized attention
  backward, the one-thread-per-row cross-entropy gradient and softmax.
  Selecting flash attention (`block::FlashGqaIds::for_training`) and the
  row-cooperative cross-entropy (`ce_value_stats_rows` + `ce_grad_stats`)
  raised the step from about 3.4k to 7.9k tokens/s; the SwiGLU control at
  f=1024 runs about 14k. The FFT pair is then the largest cost (about 7.4 ms
  per call over 196k rows of 512).
- A per-workgroup twiddle table shared by 2048/D rows per workgroup made the
  FFT slower, not faster (about 10 ms per call at 24 KiB of workgroup memory,
  9.5 ms at 12 KiB), so the kernels keep one row per workgroup with the
  twiddles computed in the butterfly.

## Next: ranked by what the measurements support

The registered study measures the feature axis only. What it found decides
the order of what follows:

- The spectral feed-forward mixer buys parameter efficiency, not arithmetic
  efficiency: it beats a SwiGLU with its parameter count and loses clearly to
  one with its arithmetic, and the FFT pair runs far below the tuned GEMM on
  the same card. More structured-operator work (butterfly/Monarch factors in
  place of dense projections) needs both multi-factor operators and kernel
  work before it can pay; it is not the lead bet.
- At a 512-token context attention is a minority-to-large share of a step
  (about a quarter of SpectralLM's forward, about two thirds of the dense
  control's), and projections plus feed-forward hold the rest, so replacing
  attention alone cannot give large end-to-end factors at ordinary context.

In order:

1. **Sequence-axis state mixing with occasional attention.** Put the existing
   Gated DeltaNet mixer (`model::gdn_mixer`, `gdn_segment`: selective
   recurrent state, chunk-parallel prefill, fixed-state decode, backward) in
   this crate's decoder as the token mixer, with an attention layer every few
   blocks; a diagonal complex-mode recurrence (per-mode decay and rotation)
   is an ablation of the same family. Measure held-out bits per byte, the
   recall and copying tasks of `crates/bench` (MQAR, selective copy, recall),
   and prefill tokens/s, time to first token and decode tokens/s from 2k to
   16k context, against the attention-only decoder.
2. **Adaptive depth.** brain already runs recurrent-depth decoders (`huginn`,
   `ouro`); measure FLOPs to a fixed held-out quality with adaptive recurrence
   on the same corpus.
3. **FLOPs to a target quality as the standard metric.** Record evaluation
   curves (periodic held-out loss) so every comparison can be read as compute
   to a fixed quality, not only quality at a fixed token budget.
4. Multiscale or innovation-gated states only after 1 or 2 move the
   quality-per-FLOP frontier; brain has no mechanism for them yet.
