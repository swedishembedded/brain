# vqgan — roadmap

VQ autoencoder (`crates/vqgan`) behind CodeFormer blind face restoration:
encoder, nearest-neighbour vector quantizer, and generator. Forward and
backward are both implemented and verified against the reference
implementation; the basic serving contract (`encode`/`decode` over D-Bus) is
in place.

## Not yet done

- [ ] Gradient checkpointing for training — every activation is currently
      kept live, so training memory scales with the fully-tapped graph
- [ ] Selective tap recording — the graph is currently either fully tapped
      (every intermediate pinned, no buffer pooling) or fully pooled with no
      taps at all; there's no way to record only some stages
- [ ] Training-time statistics: perplexity, mean codebook distance, codebook
      loss (mean distance in particular needs the full assignment distance
      matrix, which the argmin kernel deliberately doesn't emit)
- [ ] CLI subcommand and a generic model-trait implementation
- [ ] INT8 quantization
- [ ] Batched serving (`run_batch` over D-Bus) - requests are still served
      serially there; the forward graph itself is no longer the blocker (see
      below)
- [ ] Shared/pooled scratch buffers between the encode and decode graphs
      (each currently owns its own)
- [ ] Integration with the imaging pipeline — callers must already supply a
      correctly-shaped tensor

**Batch > 1 in the forward graph is done.** The earlier note here ("batch
size is hardcoded to 1 in the shared block builder") was imprecise: every
batch-sensitive kernel the builder dispatches (`conv_bias_reg`, `gn_apply`,
`attn_scores_bidir`/`_softmax`/`_apply`, `concat2`, `upsample2`, the
`nchw_nlc`/`nlc_nchw` permutation pair, ...) has ALWAYS taken an `N`/`bsz`
parameter and indexed per-sample correctly. The actual gap was purely on the
Rust side: `vae::blocks::Builder` hardcoded that parameter to `1` at every
dispatch call site and sized every activation buffer for one image. That is
now fixed - `Builder::set_batch(n)` threads a real batch count through every
NCHW-shaped block and the training tape's adjoints - and `Vqgan::new_batched`
builds one graph over `n` images (the codebook search/gather kernels needed
no change at all: they were already one-invocation-per-query, so `n * t`
queries covers the whole batch with the same two kernels `Vqgan::new` uses at
`n = 1`).

One exception, NOT touched by this fix: `add_chan` (the per-image,
per-channel broadcast add `sdxlunet`'s resnets use for the timestep
embedding) stays pinned to batch 1, because its only caller uploads a
`[C]`-shaped bias that is only correct at `N = 1`. Irrelevant to this crate
(vqgan never records an `AddChan`), but worth knowing before assuming every
op the shared builder registers is now batch-general - it is except that one,
and `sdxlunet` batching is its own future item.
