# world-models — roadmap

Action-conditioned video world models playable in real time from a window
(WASD control, session recording, full and LoRA fine-tuning): DIAMOND
(Atari-style diffusion world model) is playable end to end on CPU/GPU/NPU;
GenieRedux-CoinRun's tokenizer, dynamics, and sampler are forward-complete
and parity-verified against the reference but not yet wired into interactive
play.

## Not yet done

- [ ] CoinRun data ingest (frames + actions) into the world-model data
      pipeline
- [ ] Wrap GenieRedux's tokenizer + dynamics + sampler into a real-time
      interactive play loop (rolling frame/token window, SDL/WASD)
- [ ] Move GenieRedux's forward pass onto a single on-device graph / GPU
      backend — it currently round-trips through the host between every op
      and uses a naive matmul, which is far from real-time
- [ ] Generalize the VQ/EDM host math that currently lives in the
      DIAMOND-specific code into the shared world-model core, for reuse by
      other architectures
- [ ] `wm play`'s Enter/reset should restore the initial seed context and
      reseed the RNG deterministically — it currently resets to zeros,
      producing a random continuation rather than restarting from the
      original context
- [ ] INT8 post-training quantization for the NPU graph (only an fp32 export
      exists today)
- [ ] Reduce per-inference-step GPU submission overhead by batching multiple
      dispatches into a single submission; use fp16 cooperative-matrix
      paths where available
- [ ] Batched (n > 1) training, and GPU tiling for the convolution backward
      kernels, which are currently a naive implementation
- [ ] iVideoGPT and open-oasis pretrained-model support (only DIAMOND and
      GenieRedux-CoinRun are implemented)

GenieRedux's checkpoint conversion and per-run forward pass are slow enough
on CPU that interactive play isn't practical yet without the on-device-graph
work above; this is an engineering/performance gap, not a correctness one —
its outputs already match the reference exactly.
