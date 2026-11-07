<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 10. Normalisation conventions are asymmetric and silent

**Lesson:** Normalisation conventions are asymmetric and silent: nothing structural catches a
wrongly normalised input, so verify numerically against the golden.

## What happened
PuLID's `id_cond` is `ArcFace(raw) ‖ EvaClip(L2-normalised)`. The reference reads insightface's
`face_info['embedding']`, *not* `normed_embedding`, and divides only the CLIP half.

brain's `arcface` `embed` action normalises, because its output is meant to be cosine-ready, so
wiring it in is the natural move. Nothing structural catches it: same length, same dtype, same
finite values, first 512 components ~20× too small.

## Evidence
Verified numerically against the golden before writing code: `‖id_cond[:512]‖ = 20.11`,
`‖id_cond[512:]‖ = 1.0000`.
