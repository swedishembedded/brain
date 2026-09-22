<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 45. A stage absent from the timing struct is absent from every conclusion

`ltxv::pipeline::Timings` carries `build_dit`, `denoise` and `vae`. It does
not carry the text encode. So a real `brain ltxv t2v` run printed
`964.3s total (build 10.84s, denoise 440.4s, vae 21.1s)` - three numbers that
sum to 472s of a 964s run - and every performance discussion built on that
line spent its effort on `denoise`, the second-largest stage, while the
largest one (the text encoder, **51% of the wall clock**) had never been
measured at all, by anyone, once.

Nothing was wrong. Each printed number was correct. The line simply did not
claim to be exhaustive, and no reader checked whether it added up.

Two things generalise, and the second is the one that cost the time:

* **A timing breakdown must either account for its own total or say what it
  is missing.** The same rule this file already records for a *rate*
  (#28: a partial FLOP numerator over a full denominator under-reports in
  silence) applies to a *timeline*. The fix shape is the same too: print the
  unattributed remainder as its own row rather than letting it be invisible.
  A breakdown whose parts sum to 49% of its own total is not a rounding
  issue, it is a missing stage.
* **Optimizing the stage that IS instrumented is the predictable
  consequence.** Two prior optimization passes on this model attributed,
  extrapolated and optimized inside `denoise` in real detail - correctly,
  and to real effect - because that is where the numbers were. Instrumenting
  the pipeline end to end (`--trace-ltxv`, one span per stage) is what made
  the largest stage visible, and it was visible immediately, on the first
  run, with no analysis at all.
