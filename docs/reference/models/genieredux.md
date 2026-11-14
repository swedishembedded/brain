# GenieRedux-G

A CoinRun world model alongside [DIAMOND](diamond.md) (compared on the
[world models overview](world-models.md)): a spatiotemporal transformer
(QK-normalized biased attention, GEGLU feed-forward, causal 3D position
encoding) over a tokenizer and a guided MaskGIT dynamics model. Frames are 64x64
RGB; the tokenizer turns each into a 16x16 grid of codes, the dynamics predicts
the next frame's codes from the previous frames' codes and the action, and the
tokenizer's decoder turns them back into pixels. Model id: package
`brain-genieredux`.

## Playing it

```
brain diamond play --model genieredux --weights TOKENIZER.pt,DYNAMICS.pt
```

The two arguments are the released CoinRun checkpoints, imported directly from
torch with full tensor coverage checking. The same `--model genieredux` works for
`replay` and `bench`. In code it is `genieredux::GenieReduxWorldModel`, a
`wm_core::WorldModel`:

- `reset(frames, actions)` encodes the context frames (pixels in `[0, 1]`, the
  newest few that fit the window; none means one grey frame) and re-seeds the
  sampler, so the same context always starts the same session; `reset_initial`
  returns to that start and replays the same frames.
- `step(action)` returns the next frame. An action outside `0..num_actions()` is
  clamped to the last one. The window of tokens rolls forward; generated tokens
  are not re-encoded.

## Sampler

Generation follows the reference `Dynamics.sample` (GenieRedux
`models/dynamics.py`): each pass re-masks the least confident positions by a
schedule (cosine by default), predicts every position again (positions the
schedule kept are re-predicted, not frozen), and samples with a temperature that
decays to zero on the last pass. The released checkpoint uses one greedy pass.
The rules and the one tie-break choice are documented at the top of
`crates/genieredux/src/maskgit.rs`.

## Cost per generated frame

The dynamics and the decoder keep a per-layer cache of the temporal attention
keys and values (and the two preceding frames each position encoding reads), so
a new frame costs one frame of projections and feed-forward plus its attention
against the cache. At the released size with a 4-frame context that is 41.6
GFLOP per frame against 166.5 GFLOP recomputing the context; with a 15-frame <!-- perf-number: FLOP counts computed by genieredux::cached_forward_flops, not timings -->
context (the position table's limit) 41.7 against 624.8
(`genieredux::cached_forward_flops`, multiply-adds counted as two). Cached and
recomputed outputs are bit-identical on the CPU backend. When the window is full
the oldest frame is dropped and the caches rebuild, since the positional
embedding is absolute.

Inference only: training the tokenizer or the dynamics is outside this crate.
The forward pass is orchestrated from the host (each op reads back to the host),
which keeps it slower than real time at full size.
