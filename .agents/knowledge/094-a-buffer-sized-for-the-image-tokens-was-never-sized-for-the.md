<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 94. A buffer sized for "the image tokens" was never sized for the forward call it actually gets

`Flux1::n_max` returned `(h/16)*(w/16)` - image tokens only - with a comment
admitting "no txt/refs headroom yet: text2image only". But
`Flux1Model::run`'s own bound is `nt + ni <= n_max`: the joint sequence,
not the image half of it. Nothing enforced the two staying in sync, because
nothing had ever driven a real T5-XXL context through this path before -
every prior exercise of `Flux1Model` supplied `n_max` directly at the tiny
sizes its own tests chose. The first real conditioned forward (`text_encoder_2`
producing a real 512-row context) panicked "sized for 1024 joint tokens, got
1536" - `n_max` was measuring half of what it was named for.

The fix adds the missing term (`n_max = image_tokens + MAX_TXT_LEN`), but the
more durable part is what `MAX_TXT_LEN` replaced: `flux1::caps` and
`pulid::caps` each separately hardcoded `512` as their `max_len` param's
upper bound, matching `n_max`'s missing headroom only by coincidence - two
literals that happened to agree until someone changed one. They now both
read `flux1::pipeline::MAX_TXT_LEN`, so a param bound and the buffer it
bounds cannot drift apart again. The general shape: any time a "how big can
this get" constant is asserted in one place and re-declared as a literal
in another, the second declaration is a silent promise the first will hold
- make it a shared name instead of a coincidence.
