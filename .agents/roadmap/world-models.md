# world-models - roadmap

Scope: Playable world models: DIAMOND and GenieRedux-G (`crates/diamond`, `genieredux`, `wm-core`, `wm-display`).

Action-conditioned video world models playable in real time from a window
(WASD control, session recording, full and LoRA fine-tuning): DIAMOND
(Atari-style diffusion world model) is playable end to end on CPU/GPU/NPU;
GenieRedux-CoinRun's tokenizer, dynamics, and sampler are forward-complete
and parity-verified against the reference, and `genieredux::GenieReduxWorldModel`
plays them as a `wm_core::WorldModel` (rolling token window, `--model genieredux`
in `brain diamond play|replay|bench`, written against the CLI and compiled only
by the CI lanes).

GenieRedux inference follows the reference sampler (`Dynamics.sample` in
`models/dynamics.py`: cosine schedule, overwrite-every-pass rule, confidence
re-masking, decaying Gumbel temperature, last frame dropped; rules and the one
tie-break choice in `crates/genieredux/src/maskgit.rs`). The dynamics and the
tokenizer decoder run through a per-layer key/value and PEG-history cache
(`crates/genieredux/src/cache.rs`), bit-identical to the full recompute on the
CPU backend: at the released config with a 4-frame context a generated frame
costs 41.6 GFLOP against 166.5 for the full pass (`cached_forward_flops`; the
saving grows with the context, 15 frames: 41.7 against 624.8). The window slides
by dropping the oldest frame, which rebuilds the caches because the positional
embedding is absolute. Training of the tokenizer and the dynamics is out of
this crate's scope: it ports inference.

Gates: `crates/genieredux/tests/it/{maskgit,kv_cache,world_model}.rs` on a tiny
random model (no checkpoints); the real-checkpoint parity tests
(`parity_*.rs`) are ignored and skip without them.

## Not yet done

- [ ] CoinRun data ingest (frames + actions) into the world-model data
      pipeline
- [ ] Run the multi-pass sampler (`inference_steps > 1`) against reference
      goldens: only the single-pass checkpoint default is gated against real
      weights (`parity_maskgit.rs`, ignored without them); the loop rules are
      gated on the tiny model
- [ ] Move GenieRedux's forward pass onto a single on-device graph / GPU
      backend — it currently round-trips through the host between every op
      and uses a naive matmul, which is far from real-time
- [ ] Generalize the VQ/EDM host math that currently lives in the
      DIAMOND-specific code into the shared world-model core, for reuse by
      other architectures
- [ ] INT8 post-training quantization for the NPU graph (only an fp32 export
      exists today)
- [ ] Reduce per-inference-step GPU submission overhead by batching multiple
      dispatches into a single submission; use fp16 cooperative-matrix
      paths where available
- [ ] Batched (n > 1) training, and GPU tiling for the convolution backward
      kernels, which are currently a naive implementation
- [ ] iVideoGPT and open-oasis pretrained-model support (only DIAMOND and
      GenieRedux-CoinRun are implemented)

GenieRedux's forward pass is host-orchestrated (each op round-trips through
the host, naive matmul), so real-time play at the released size waits on the
on-device-graph item above; that is a performance item, outputs match the
reference.

Enter/reset is done - `diamond::play`'s `reset` snapshots the seed context
(`init_obs`/`init_act`) and re-seeds `NormalRng::new(self.seed)`, and
`reset_initial` restores both, so the interactive Enter key rewinds to the
exact original context and noise stream rather than a random continuation.
Gated bit-exactly by `crates/diamond/tests/it/reset.rs`.
