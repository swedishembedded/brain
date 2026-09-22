<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 9. Preprocessing must be shared, not described as shared

`brain depth calib` letterboxed into a padded square while `Predictor::begin`
does an aspect-preserving resize with no pad - and the loader's doc-comment
claimed they were "the same transform the predictor uses at inference". INT8
scales were fitted to a distribution that never occurs. The fix is structural:
one extracted `preprocess_chw` both call, so the claim cannot drift again.
