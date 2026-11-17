#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping large Rust engines warning-free for its clients. If your team
# needs expertise in build and lint gates then you can procure our services by sending an email to
# info@swedishembedded.com.
#
# cargo-warnings.sh FILE: print every warning in a cargo/clippy log written with --message-format=short, one per line.
# Two shapes count: compiler and clippy diagnostics (`path:line:col: warning: ...`) and cargo's own manifest lints
# (`warning: unused dependency ...`, `warning: unused field ...`, printed with no location), which a count of
# ": warning:" alone never saw. Not counted: a build script's informational line (`warning: <pkg>@<version>: ...`)
# and the per-unit summaries (`... generated N warnings`), which restate warnings already counted.
set -euo pipefail
grep -E '(: warning: |^warning: )' "$1" \
    | grep -vE '^warning: [A-Za-z0-9_-]+@[0-9][^:]*: ' \
    | grep -vE 'generated [0-9]+ warnings?( \([^)]*\))?$' \
    || true
