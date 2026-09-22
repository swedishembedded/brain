<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 19. Registration split across N lists is a defect waiting for its turn

Adding a served model meant editing three lists with no link between them -
`caps_cli::static_manifests()` (what `brain caps` lists),
`caps_cli::build_registry()` (what `brain do` can run) and
`resident::build_executor()` (what the transports serve). Each omission fails
SILENTLY and differently: undiscoverable, or listed-then-"unknown model", or
invisible to D-Bus.

It caught a model that had everything else right. `Real-ESRGAN` shipped with a
manifest, a provider, a residency adapter, a parity ladder at cosine 1.0 and 16
green tests, and `brain caps <id>` still said "unknown model" - because only the
third list had been edited. No test could see it, because no test related the
lists to each other.

The fix is not "remember to edit three places", it is to make three places
impossible: one `ModelEntry` per model holding the manifest, the provider
constructor and the residency adapter, with the other lists DERIVED. Then the
invariant becomes testable, and the test that matters is the one that reproduces
the original failure - *every model this lists must be constructible by name*.

Generalises past this repo: whenever a thing must be declared in more than one
place for a feature to work, the duplication is not a style problem, it is an
unexploded defect. Look for it wherever a "registry", a "catalog" and a
"dispatch table" name the same set.
