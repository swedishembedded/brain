#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# No baked-in absolute machine paths under crates/, docs/, or .agents/.
#
# Two distinct failure modes, two checks:
#
#  1. crates/**/*.rs: a weight or fixture location belongs in an env var or a
#     CLI flag. A literal like "/data/workspace/resources/..." resolves on
#     exactly one machine; on every other it fails the `.exists()` check and
#     the test skips itself, which reads as "the fixture is absent" rather
#     than "the path is wrong". That is the worst possible failure mode for a
#     parity test, because a skipped test is green. This also now catches a
#     BARE (unquoted) machine path inside a `//`/`///`/`//!` comment line -
#     e.g. a usage example in a doc comment - which the original
#     quoted-string-literal-only pattern missed (real incident: four
#     `crates/qwen35/tests/*.rs` usage-example doc comments shipped a literal
#     "/data/workspace/..." that no gate ever ran over).
#  2. docs/**/*.md and .agents/**/*.md: these are prose/example text, not
#     compiled code, so the failure mode is different (a misleading example
#     shown to a human, not a silently-skipped test) but the fix is the same
#     - show a placeholder like `[path/to/qwen3.8]`, never a literal that only
#     resolves on the machine that happened to write it. Real incident: a
#     freshly-written docs/models/qwen35.md example shipped
#     "/data/workspace/resources/qwen3.8" verbatim, uncaught, because this
#     gate only ever looked at crates/.
#
# The rule long predates this script and is written down in AGENTS.md, which
# even quotes the grep. Nothing ran it for years, so violations accumulated
# quietly - a gate nobody executes is a comment, not a gate; the same lesson
# repeated itself when this script's own scope turned out to be narrower than
# anyone remembered.
#
# scripts/ and tools/ get the equivalent check from check-scripts.sh, which also
# allows the sanctioned overridable-default forms (`${VAR:-/path}`,
# `os.environ.get(V, "/path")`) that make sense for a human-run utility and do
# not make sense inside crates/.
#
# Usage: scripts/gates/check-no-machine-paths.sh [file ...]
#   With no arguments, scans the whole crates/, docs/, and .agents/ trees.
#   With arguments (how the pre-commit hook calls it), scans only those,
#   ignoring anything outside those three.
#
# The per-file mode used to scan every staged file under crates/ regardless of
# extension, so a vendored third-party fixture with a machine path in it
# (crates/apiserve/tests/specs/openrouter.json is OpenRouter's published
# OpenAPI document, and its request examples carry an upstream "/root/worker")
# failed the hook while `make check/paths` over the same tree passed. Two
# modes of one gate disagreeing is worse than either answer: the one that is
# easy to run says OK. Stay scoped to *.rs (crates/) and *.md (docs/,
# .agents/) - brain's own source and its own prose, never a vendored fixture.
set -uo pipefail
cd "$(dirname "$0")/../.."

# A string literal that STARTS an absolute machine path. A /data/ substring
# mid-string is not a filesystem path - a URL, or a torch-archive-internal
# "archive/data/0" key - and is deliberately not matched.
QUOTED_PATTERN='"/(data|home|tmp|opt|mnt|root)/'
# A BARE path (no quote required) that starts right after whitespace, a
# backtick, '(', or '=' - i.e. actually starts a path/flag value - not a
# substring buried mid-word (a URL host+path, an archive-internal key).
BARE_PATTERN='(^|[[:space:]`(=])/(data|home|tmp|opt|mnt|root)/'
# Same, but for docs/.agents prose: /tmp is a universal, portable scratch
# convention used constantly in illustrative shell examples everywhere (it
# does not "resolve differently on every machine" the way a bespoke project
# mount like /data/workspace does) - excluded here on purpose, kept in
# BARE_PATTERN/QUOTED_PATTERN above for crates/ code, where a hardcoded
# /tmp/fixture path IS a real portability problem for an automated test.
MD_PATTERN='(^|[[:space:]`(=])/(data|home|opt|mnt|root)/'

rs_hits=""
md_hits=""

if [ "$#" -gt 0 ]; then
  rs_files=()
  md_files=()
  for f in "$@"; do
    case "$f" in
    crates/*.rs) [ -f "$f" ] && rs_files+=("$f") ;;
    docs/*.md | .agents/*.md) [ -f "$f" ] && md_files+=("$f") ;;
    esac
  done
  if [ "${#rs_files[@]}" -gt 0 ]; then
    quoted=$(grep -nE "$QUOTED_PATTERN" "${rs_files[@]}" 2>/dev/null)
    bare_comment=$(grep -nE '^\s*(///?|//!)' "${rs_files[@]}" 2>/dev/null | grep -E "$BARE_PATTERN")
    rs_hits=$(printf '%s\n%s' "$quoted" "$bare_comment" | sed '/^$/d')
  fi
  [ "${#md_files[@]}" -gt 0 ] && md_hits=$(grep -nE "$MD_PATTERN" "${md_files[@]}" 2>/dev/null)
else
  rs_hits=$(grep -rnE "$QUOTED_PATTERN" crates --include='*.rs' 2>/dev/null)
  md_hits=$(grep -rnE "$MD_PATTERN" docs .agents --include='*.md' 2>/dev/null)
fi

hits="${rs_hits}${rs_hits:+$'\n'}${md_hits}"

[ -z "$hits" ] && { [ "$#" -eq 0 ] && echo "check-no-machine-paths: OK"; exit 0; }

echo "check-no-machine-paths: absolute machine path baked in:"
echo "$hits" | sed 's/^/  /'
cat <<'EOF'

In crates/, resolve it from the environment instead:
  - model weights   -> the architecture's own BRAIN_*_ env var, else
                       BRAIN_MODELS_DIR (the model store root)
  - test fixtures   -> brain_testutil::testdata, which honours BRAIN_TESTDATA
  - in-repo outputs -> concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/...")

In docs/.agents/, show a placeholder instead: [path/to/qwen3.8], not a
literal that only resolves on the machine that wrote it.

And skip the test when it resolves to nothing. A literal path makes a
misconfigured run look like a missing fixture, and a skipped test is green.
EOF
exit 1
