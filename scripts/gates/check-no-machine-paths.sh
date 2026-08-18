#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# No baked-in absolute machine paths under crates/.
#
# A weight or fixture location belongs in an env var or a CLI flag. A literal
# like "/data/workspace/resources/..." resolves on exactly one machine; on every
# other it fails the `.exists()` check and the test skips itself, which reads as
# "the fixture is absent" rather than "the path is wrong". That is the worst
# possible failure mode for a parity test, because a skipped test is green.
#
# The rule long predates this script and is written down in AGENTS.md, which
# even quotes the grep. Nothing ran it, so two violations accumulated - one of
# them in a test written the same week. A gate nobody executes is a comment.
#
# scripts/ and tools/ get the equivalent check from check-scripts.sh, which also
# allows the sanctioned overridable-default forms (`${VAR:-/path}`,
# `os.environ.get(V, "/path")`) that make sense for a human-run utility and do
# not make sense inside crates/.
#
# Usage: scripts/gates/check-no-machine-paths.sh [file ...]
#   With no arguments, scans the whole crates/ tree. With arguments (how the
#   pre-commit hook calls it), scans only those, ignoring anything outside
#   crates/.
#
# BOTH modes look at crates/**/*.rs and nothing else, which they have to: the
# rule is about how brain's own source resolves a path. The per-file mode used
# to scan every staged file under crates/, so a vendored third-party fixture
# with a machine path in it (crates/apiserve/tests/specs/openrouter.json is
# OpenRouter's published OpenAPI document, and its request examples carry an
# upstream "/root/worker") failed the hook while `make check/paths` over the
# same tree passed. Two modes of one gate disagreeing is worse than either
# answer: the one that is easy to run says OK.
set -uo pipefail
cd "$(dirname "$0")/../.."

# A string literal that STARTS an absolute machine path. A /data/ substring
# mid-string is not a filesystem path - a URL, or a torch-archive-internal
# "archive/data/0" key - and is deliberately not matched.
PATTERN='"/(data|home|tmp|opt|mnt|root)/'

if [ "$#" -gt 0 ]; then
  files=()
  for f in "$@"; do
    case "$f" in
    crates/*.rs) [ -f "$f" ] && files+=("$f") ;;
    esac
  done
  [ "${#files[@]}" -eq 0 ] && exit 0
  hits=$(grep -nE "$PATTERN" "${files[@]}" 2>/dev/null)
else
  hits=$(grep -rnE "$PATTERN" crates --include='*.rs' 2>/dev/null)
fi

[ -z "$hits" ] && { [ "$#" -eq 0 ] && echo "check-no-machine-paths: OK"; exit 0; }

echo "check-no-machine-paths: absolute machine path baked into crates/:"
echo "$hits" | sed 's/^/  /'
cat <<'EOF'

Resolve it from the environment instead:
  - model weights   -> the architecture's own BRAIN_*_ env var, else
                       BRAIN_MODELS_DIR (the model store root)
  - test fixtures   -> brain_testutil::testdata, which honours BRAIN_TESTDATA
  - in-repo outputs -> concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/...")

and skip the test when it resolves to nothing. A literal path makes a
misconfigured run look like a missing fixture, and a skipped test is green.
EOF
exit 1
