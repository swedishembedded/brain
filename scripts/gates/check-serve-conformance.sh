#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements continuous-batching LLM serving for clients
# whose agents resend long conversation histories. If your team needs
# expertise in LLM serving throughput then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# Serving-engine conformance gate (`make check/serve-conformance`).
#
# Every `impl PagedDecoder for <Engine>` under crates/*/src is an engine the
# shared scheduler serves (`brain serve`). It must be run through the generic
# suite in `model::serve::conformance` (continuous batching, no head-of-line
# blocking, prefix reuse, adapter KV isolation): some test of the engine's
# crate must call `conformance::run_all`. An engine that cannot (it needs a
# checkpoint too large for CI) carries `conformance-exempt: <reason>` on the
# line before its impl, and the reason names the real-model test that covers it.
#
# Usage: scripts/gates/check-serve-conformance.sh [--root DIR]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
if [ "${1:-}" = "--root" ]; then root="$2"; fi
rc=0
while IFS=: read -r file line _; do
  crate_dir="$(echo "$file" | sed -E 's#^(crates/[^/]+)/.*#\1#')"
  [ "$crate_dir" = "crates/model" ] && continue
  prev="$(sed -n "$((line - 1))p" "$root/$file")"
  case "$prev" in *"conformance-exempt:"*) continue ;; esac
  engine="$(sed -n "${line}p" "$root/$file" | sed -E 's/.* PagedDecoder for ([A-Za-z0-9_]+).*/\1/')"
  # A file of the engine's crate that runs the suite AND names this engine type.
  covered=0
  for f in $(grep -rlE 'conformance::run_all' "$root/$crate_dir/tests" "$root/$crate_dir/src" 2>/dev/null); do
    if grep -qE "\\b$engine\\b" "$f"; then covered=1; break; fi
  done
  [ "$covered" -eq 1 ] && continue
  echo "check-serve-conformance: $file:$line: $engine implements PagedDecoder but no test in $crate_dir names it and calls model::serve::conformance::run_all" >&2
  echo "  add a test with a tiny engine, or mark the impl '// conformance-exempt: <reason and the covering real-model test>'" >&2
  rc=1
done < <(cd "$root" && grep -rnE 'impl[^{]* PagedDecoder for ' crates/*/src --include=*.rs)
[ "$rc" -eq 0 ] && echo "check-serve-conformance: every PagedDecoder engine runs the conformance suite"
exit "$rc"
