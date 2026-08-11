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
- [ ] Batch > 1 support and batched serving (`run_batch`) — requests are
      served serially today
- [ ] Shared/pooled scratch buffers between the encode and decode graphs
      (each currently owns its own)
- [ ] Integration with the imaging pipeline — callers must already supply a
      correctly-shaped tensor

Batch size is hardcoded to 1 in the shared block builder that this crate and
the diffusion VAE both use, so true batching needs to land in that shared
builder rather than as a vqgan-only fork.
