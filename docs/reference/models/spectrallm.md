<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# SpectralLM

brain's own decoder whose residual stream is the lowest `m` coefficients of
the real orthonormal Fourier basis of a virtual `D`-sample signal, and whose
feed-forward sublayers are spectral mixers. There is no upstream checkpoint:
models are trained from scratch.

Everything in a block except the feed-forward sublayer is Qwen3's: RMSNorm,
grouped-query attention with per-head QK-norm and half-split RoPE, a tied
head. The spectral mixer with `C` channels computes

```text
M(h) = Σ_c W_c ⊙ Sᵀ[ SiLU(S(G_c ⊙ h)) ⊙ S(U_c ⊙ h) ]
```

with learned per-mode filters `G, U, W` of shape `[C, m]` and `S` the
`[D, m]` synthesis basis. It has `3Cm` parameters per block. The transforms
run as a fast Fourier transform on a GPU that executes workgroup barriers and
as a dense product with the stored basis everywhere else; both compute the
same model. With the SwiGLU mixer instead (`Mixer::SwiGlu`) the crate is the
Qwen3 decoder, which is how the conventional control is trained under the
same code.

## Configuration

| Field | Meaning |
|---|---|
| `d_model` | retained Fourier coefficients `m`, the residual width |
| `mixer.signal_len` | virtual signal length `D` (even, at least `d_model`; a power of two up to 2048 for the FFT path) |
| `mixer.channels` | spectral mixer channels `C` |
| `mixer.d_ff` | SwiGLU hidden width, for the conventional control |
| `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`, `rope_theta`, `rms_norm_eps`, `tie_embeddings` | as for Qwen3 |

A checkpoint records the mixer and the coefficient convention
(`basis: real-fourier-orthonormal-frequency-ordered`); a loader refuses a
basis it does not implement.

## Training, scoring and sampling

Training runs through the shared training loop (`model::fit`) with the
analytic backward, gradient-checked against finite differences for both
mixers and both transform paths. Incremental decoding uses a key/value cache.
`samples/text/spectral-lm` trains a model on a document corpus, scores it in
bits per byte, and generates text:

```bash
brain data gen tinystories --out data/tinystories   # after fetching train.txt and valid.txt
make samples/text/spectral-lm/build
./target/release/sample-text-spectral-lm train data/tinystories --out spectral.safetensors \
    --d-model 256 --layers 6 --heads 8 --signal-len 512 --channels 24
./target/release/sample-text-spectral-lm generate --checkpoint spectral.safetensors \
    --tokenizer data/tinystories/tokenizer.json --prompt "Once upon a time"
```

The measurements this architecture has been put through are in the
spectral-lm paper (`papers/spectral-lm`).

SpectralLM is not served by `brain serve`.
