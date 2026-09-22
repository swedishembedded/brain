<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 48. A parity gate built from the pipeline's own simplified positions cannot catch the pipeline getting positions wrong

A real-weight video DiT produced finite, well-behaved, non-NaN output at
every shape tried - and looked like fine-grained textured noise every time,
regardless of resolution, regardless of `eta` (ancestral vs. deterministic),
regardless of a from-scratch real-image-conditioning feature built and
verified against the reference source line by line. Every other component on
the path - the VAE decode (bit-exact against the reference at real width),
the text encoder (cosine 1.0 through 6 real layers, both attention types),
the embeddings connector (cosine 0.999999973 on real weights), the streamed
vs. eager forward (bit-identical) - had already been proven correct against
the actual reference, independently, several different ways. None of it
mattered, because the RoPE position VALUES the whole stack was built on were
wrong.

**The bug:** the pipeline built RoPE positions as raw latent-grid integers
(`token → [i, i+1)` per axis). The reference builds them in **pixel space**:
`get_pixel_coords` multiplies each latent bound by the VAE's own downsample
factor (time×8, height×32, width×32), applies a causal fix to the temporal
axis (the first latent frame covers exactly one pixel frame, every later one
covers eight), and only then divides the temporal axis by fps. Latent index 3
on the height axis is `[3, 4)` in one convention and `[96, 128)` in the
other - the SAME formula shape (`precompute_freqs_cis` over `theta`/
`max_pos`), fed a value 32x off. No crash follows from that: the model just
runs its trained frequency table over the wrong slice of it, per token,
uniformly, which is indistinguishable from "producing noise" until you know
to look for a scale, and is exactly why `positional_embedding_max_pos:
[20, 2048, 2048]` (unmistakably pixel-scale) sat unremarked next to a
latent-integer position builder for an entire porting pass.

**Why the existing real-weight gates could not have caught it.** Every
reduced-depth real-weight DiT test in this port replayed the reference's own
golden dumper - which itself builds positions from a raw `torch.meshgrid`,
not from `get_pixel_coords`/`VideoLatentTools`, as a deliberate, documented
scope cut ("prove the op sequence," not "prove the pipeline's own position
construction"). Brain's Rust replay used the identical raw-meshgrid
convention to match. Both sides agreed, at cosine ≥ 0.999999, because both
sides were self-consistently wrong the same way - a parity gate cannot see a
divergence that its own golden was built to not exercise. This is a
different shape of blindness than #47's (a real tap made invisible by
something downstream of it): here the tap was fine, the INPUT the gate fed
it was never the one the real pipeline actually constructs.

**A second bug hid in the same reference method, same root cause.** The
reference marks `keyframes_mask` on the first latent frame's tokens
*unconditionally* - the causal VAE gives that frame its own narrower pixel
window, "the same token class as a generated keyframe slot," independent of
whether any real image conditioning is present. The port had reasoned "no
image conditioning → every token is genuinely noise → mask stays zero,"
which is true and irrelevant: the mask marks a structural token CLASS, not
"this token is externally supplied." A real per-token positional-embedding
add was silently skipped on every real generation.

**Consequences worth keeping:**

* **A parity gate that builds its own inputs is only as strong as how those
  inputs were built.** Ask, specifically: does this golden exercise the
  pipeline's REAL position/coordinate/shape-construction code, or a
  simplified stand-in that happens to be self-consistent across both sides?
  "Both sides use the same formula" is not evidence the formula is the real
  one.
* **Cosine similarity (and even a tight `max_abs`) cannot catch a systematic
  input error that both the implementation and its own test were built
  around identically.** The fix is not a better metric - it's exercising the
  ACTUAL pipeline construction path in the gate, end to end, not a purpose-
  built substitute.
* **A finite, non-NaN, well-behaved-looking wrong output is the hardest kind
  to root-cause by staring at pixels.** Every visual symptom here (fine
  mesh texture, no global coherence, indifferent to resolution/eta/image
  conditioning) was consistent with "the model just needs more tokens" -
  a plausible, false lead that survived several independent, correctly-
  reasoned investigations before the actual cause surfaced from re-reading
  the reference's own coordinate-construction code end to end, not from
  another round of comparing pixels.
* **A config field's own stated range is a clue.** `positional_embedding_
  max_pos: [20, 2048, 2048]` had been read and repeated in this port's own
  documentation for an entire phase without anyone asking why a max position
  of 2048 would make sense for a token grid that never exceeds a few hundred
  entries. It would not - because the position space was never meant to be
  the token grid's own index space.
* **When adding position/coordinate handling for any new model, verify units
  against the reference's actual coordinate-construction function (not its
  RoPE math, which is downstream and unit-agnostic), and build at least one
  gate that exercises that construction for real** - a synthetic/simplified
  golden proves the op sequence, never the units.
