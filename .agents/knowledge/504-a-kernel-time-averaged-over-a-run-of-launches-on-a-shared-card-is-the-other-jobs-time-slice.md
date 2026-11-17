<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 504. A kernel time averaged over a run of launches on a shared card is the other job's time slice

**Lesson:** device events bracket wall time on the card, so a launch another context is time-sliced into reads as the
slice. Averaging a run of launches folds one stretched launch into the run, and the minimum of a few such averages
does not undo it when every run holds one. Time each launch on its own (one launch a submission, read back before the
next), take the median for a kernel and the fastest launch for a ceiling, and count the stretched launches: past a
third the number is not the kernel's, and a table measured that way must not be stored.

## What happened

`brain perf roofline --kernels` on gpu0 of a GH200 node (another experiment's server on the same card) listed one-row
int8 GEMVs at 1-2.5 ms and fractions of the roof of 100-21000 %. The table came from the store and had been tuned
the same way: `best_launch_ms` averaged chunks of launches and kept the best average, and the launch floor
(`measure_read_roof`, the average of 64 tiny launches, best of six) came out near 1.9 ms. The class bound was
`max(memory, compute, floor)`, so a 17 us kernel graded against a 1.9 ms "bound" read as 11145 %.

`launch_stats` now times every launch alone; the tuner keeps the median, the fastest and the share stretched past
1.25x the fastest (`Verdict::Timed`, store format v4), and the floor probe's stretched share is the card's busy share
(a table measured with more than a quarter stretched is used but not stored). On gpu3 with no other context the
floor is 6.2 us, a one-row 5120 x 4096 GEMV 14.6 us, 0 % stretched. The floor is no longer part of the bound: a class
whose roof time is under it is `launch`-bound, shown with the floor in its own column and kept out of the gap list.
Every roof is the largest rate measured for it (probe, device rows, any timed candidate), so no fraction passes one.

## What to do

- Grade with per-launch device times (`gpu_core::launch_stats`), never with a mean over a batch of launches.
- A roof that a kernel beats is the wrong roof: raise it to the kernel and say so, rather than print a fraction above one.
- The store format is part of the store key, so a build writing a new format does not make a running build of the
  old format re-tune at every start.
