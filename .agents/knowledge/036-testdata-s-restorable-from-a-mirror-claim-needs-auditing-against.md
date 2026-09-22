<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 36. `testdata/`'s "restorable from a mirror" claim needs auditing against a real machine, not assumed from the script

`scripts/data/fetch-testdata.sh`'s own header states the design goal
plainly: `testdata/` is disposable, gitignored, and `make fetch/testdata`
repopulates it from local mirrors (`BRAIN_*_MIRROR` env vars, each defaulting
to a fixed path - the ONE place a machine-specific path may appear in this
repo, per AGENTS.md). An audit of a real machine found several of the
referenced mirror roots simply absent, and several newer fixture trees (an
imaging workstream's fixtures among them) with no entry in the script at all,
mirror or otherwise.

Net: on that machine, most of `testdata/`'s contents were not actually
recoverable by running the documented recovery command. The tree LOOKS
disposable (gitignored, a script claims to repopulate it) but can function as
irreplaceable local state whenever the mirrors it depends on were either
never copied to that machine or were never wired into the script for newer
fixtures. `rm -rf testdata/` under that condition would be silent, uncontested
data loss, not the "idempotent, re-fetchable" operation the script's own doc
comment promises.

**The general shape to watch for**: any "regenerate me" claim (a script, a
doc comment, a design note) that depends on an external resource (a mirror
directory, a network endpoint, a service) is a claim about a DIFFERENT
machine's state until it's been exercised on the one you're sitting at. The
fix isn't code; it's process: before treating any gitignored/"disposable"
tree as safe to clear, run the stated recovery path for real (or audit
mirror-existence + script-coverage per subtree) rather than trusting the
tree's own claimed disposability.
