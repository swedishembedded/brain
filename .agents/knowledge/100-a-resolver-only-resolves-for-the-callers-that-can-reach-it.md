<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 100. A resolver only resolves for the callers that can reach it

`brain_modelstore::resolve` - the `ArchSpec`/`Confidence`/`Ambiguity` machinery
that reads real file headers, picks a candidate per role, and refuses to guess -
had been the one weight resolver for a while, with 22 architectures carrying a
`spec.rs`. The served path had none of it. Every `crates/cli/src/resident_*.rs`
still opened with `std::env::var("BRAIN_FLUX1_DIR").ok()?` and returned `None`,
so `brain serve --dbus` served nothing for a model whose weights were sitting
unambiguously in the store, while `brain flux2 generate` found the identical
files on its own. The example pipeline had to hardcode nine `BRAIN_*` paths to
work around it.

Nothing looked duplicated, which is why it lasted. There was no second copy of
the scanning or classification logic to notice - the served path simply had no
resolution logic at all, and "no implementation" does not trip a
one-implementation review the way a second implementation does. The tell was on
the other side: `ArchSpec` existed for exactly the architectures with a
`_cli.rs`, and for none of the six this pipeline actually served.

Two things were required to close it, and only one was the seam. The seam itself
was small (`resolver_cli::served_assembly`: env var per role first, unconditional
and never re-derived, then the same `try_resolve` the CLI uses; ambiguity
rendered as `VAR=<path>` because a daemon has no terminal to retype a flag at).
The real work was that the resolver could not answer for these architectures at
all - five had no `ArchSpec`, and `.onnx` had no `ArtifactKind`, so insightface's
antelopev2 pair was invisible to `inventory::scan` and no classifier could ever
have seen it. A capability that cannot describe an artifact is not a resolver
those callers can share, however generic its core is.

Watch for the honest asymmetry this leaves. FLUX.2 still needs an explicit
`BRAIN_FLUX2_DIT` on a box holding four real klein candidates, and that is the
resolver working: it prints all four and serves nothing until one is named.
"Auto-resolution" is not "always resolves" - it is "never guesses", and a
refusal that names every candidate is the feature, not a gap in it.
