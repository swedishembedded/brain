<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 59. A feature can be complete, gated, benchmarked and unreachable, and only the RUN'S OWN BANNER says which

Two commits landed a day apart. One built a quantized, device-resident
audio-visual DiT block for `ltxv` - a new block type, a new cached-weight
type, a new residency session, real-weight parity gates, and a measured
several-times speedup at production width. The other made the audio latent
cross a long-form window seam. Both were correct, both were tested, and
`--audio` used neither: the pipeline's denoiser factory still constructed the
old host-fp32 object, one branch above where the new one would have gone.

Nothing in the workspace could see it.

* The **parity suite** was green, because it drives the new forward directly.
* The **benchmark** reported the new numbers, because it drives the new
  forward directly.
* **Nothing tested which arm the pipeline picks**, and there was no assertion
  anywhere that could have.
* The stale routing even carried a **log line asserting its own reason** ("the
  audio stream has no streamed/quantized path yet") that had been false since
  the previous commit, so reading the code near the defect actively confirmed
  it.

What found it was running the feature and reading the first line it printed:
the banner said `(host fp32, both streams)`.

Three things generalize, and the third is the fix:

* **A gate that drives a function directly cannot see whether anything calls
  it.** Coverage of an implementation is not coverage of its adoption. Whenever
  a change adds a faster/cheaper implementation ALONGSIDE an existing one, the
  question "which one does the product take?" is a separate claim needing a
  separate check.
* **File partitioning between concurrent authors creates exactly this seam.**
  Give one worker the pipeline and another the block, and the wiring belongs
  to neither. If work is partitioned that way, someone has to own the join,
  explicitly.
* **Make the banner DERIVED, not written.** The CLI used to spell the arm out
  itself from `(dit_config, audio)`; it now calls the same pure function the
  factory routes on (`ltxv::pipeline::av_arm_label`), so the line a reader
  checks and the object the run builds cannot disagree. A hand-written status
  line is a duplicate of a fact the code already decides, and it goes stale in
  exactly the direction that hides the defect (see also §54).

**Corollary, from the same run: `secs_per_step` in this pipeline's traces is a
RUNNING MEAN, not a per-step cost.** It is `elapsed / steps_done`, so the
first value of a stage carries the whole cold start and the last value times
the step count is the stage's span. Quoting the first value as "the step time"
overstates a warm step by the entire warm-up, and summing the printed values
overstates a stage by several times. A field whose name reads as an
instantaneous quantity and whose value is cumulative will be misread; either
the name or the doc has to say so before anyone quotes it.
