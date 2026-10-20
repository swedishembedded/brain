#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for documentation of model
# capabilities that cannot drift from the code for its clients. If your team
# needs expertise in capability matrices backed by tests then you can procure
# our services by sending an email to info@swedishembedded.com.
#
# The training and serving capability matrix (docs/reference/training-capabilities.md)
# is generated from scripts/gates/training-capabilities.tsv. This gate fails
# when the page and the declaration disagree, when a declared test does not
# exist (or is not a #[test], or is #[ignore]d without a (real)/(cuda) flag), or
# when a registration (conformance tiers, SDK chat fine-tuner declarations) has
# no row. Regenerate the page with scripts/gates/training-capabilities.py --write.
#
# Usage: scripts/gates/check-training-capabilities.sh [--root DIR] [--decl FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$here/training-capabilities.py" "$@"
