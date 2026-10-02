<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 208. What lives in the binary is outside every library invariant

The residency adapters, which say how much memory a model needs and how to
load and run it, lived in `crates/cli`, the `brain` binary. Two costs followed,
and only one was visible.

The visible one: a binary cannot be linked, so no other process could serve a
model under a memory budget. The only other consumer held every model resident
forever, with no budget and no eviction.

The invisible one: the catalog's invariants (every model declares how it finds
its weights, no served manifest asks a remote caller for a path, every listed
model is constructible) were tested over the catalog crate's list, and the
models whose entries had to stay in the CLI were simply not in that list. When
they joined it, the test failed at once: Kronos advertised a `checkpoint` path
to remote callers. Nothing had ever checked it.

A second leak of the same class sat in `/v1/capabilities` and `/v1/run`: the
listing offered every param including host-resolved weights paths, and a call
that named one was accepted, so any caller with a key could point a model at a
file of their choosing. The dialect routes build their params themselves, which
hid it; the dialect-neutral routes take them from the caller.

## What to do instead

- Put the adapter in the crate that owns the model entry, so one list produces
  the manifest, the provider and the residency constructor, and every invariant
  over that list covers all of them.
- Validate a remote caller's params against the spec a caller sees
  (`for_serving`) first, then against the full spec, which is where the host
  fills in its own answers.
- When a check passes only because some entries are out of its reach, count the
  entries it reaches. A test over a list that silently excludes the awkward
  members is a test of the easy ones.
