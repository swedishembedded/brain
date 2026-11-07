<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 116. A whole-model parity test is weak evidence about a backend's plumbing

**Lesson:** A model-level parity test covers only the paths that model, at that shape, happens to
exercise. Each mechanism a backend implements needs its own assertion that fails when the mechanism
is removed, and the removal has to be tried.

## What happened
A cross-backend forward-logit comparison is the strongest statement available about a backend's
*arithmetic*, and it was: a dense decoder's forward agreed with both the CPU JIT and Vulkan to
8.9e-8 the first time it ran.

It is a much weaker statement about everything around the arithmetic, and this was measured, not
assumed. With each of these applied one at a time, that model test still passed:

* storage handed back unzeroed (the model writes every buffer before reading);
* a sub-range binding's offset dropped (the shape never binds a sub-range);
* the dispatch grid laid out at a hardcoded 64 rather than the kernel's own work-group size (every
  kernel the shape touches declares 64, and an OVER-dispatch is harmless anyway - every kernel
  self-masks);
* a `write_at` offset dropped (the shape only ever writes from word 0);
* `--fmad=false` flipped to `true` (at a tiny shape one rounding versus two is far below the 1e-6
  floor the assertion uses - the per-kernel golden gate does catch it).

## Why
The tempting conclusion "it runs a real model, so the backend works" is the mistake. Four of these
five looked covered and were not.

## What to do
Give each mechanism an assertion that fails when the mechanism is removed, and try the removal.

Corollaries worth keeping:

- An over-dispatched grid is invisible in a catalogue where every kernel bounds itself, so the
  discriminating mutation is the one that UNDER-dispatches.
- A 256-wide kernel launched with 64 threads still computes the right answer whenever the data fits
  inside one stride, so the shape has to be wider than the wrong block before the test has any
  teeth.
