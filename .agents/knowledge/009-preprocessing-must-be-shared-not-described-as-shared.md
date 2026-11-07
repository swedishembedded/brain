<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 9. Preprocessing must be shared, not described as shared

**Lesson:** Preprocessing must be shared, not described as shared.

## What happened
`brain depth calib` letterboxed into a padded square while `Predictor::begin` does an
aspect-preserving resize with no pad. The loader's doc-comment claimed they were "the same
transform the predictor uses at inference". INT8 scales were fitted to a distribution that never
occurs.

## What to do
The fix is structural: one extracted `preprocess_chw` that both call, so the claim cannot drift
again.
