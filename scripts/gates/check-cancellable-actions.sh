#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# check-cancellable-actions.sh - every action a crate builds declares that it
# observes cancellation (`make check/scripts`, and the pre-commit hook).
#
# `capability::Invocation` carries a `CancelToken` on EVERY call, so every
# action accepts one whether or not it honours it. `ActionSpec::cancellable`
# is the declaration that it does, and it travels with the manifest to every
# surface. An action that ignored its token and still said so would be worse
# than one that never offered: the caller builds request lifetime on it. So
# the declaration is required, and the contract behind it (an entry check
# before any weight is read or lock taken, a poll per bounded unit of work) is
# proved behaviourally by `crates/catalog/tests/it/cancellation.rs` and by each
# model's own `capability_mock::cancellation` tests.
#
# This gate is the static half: every `ActionSpec::new(..)` in production
# code must chain `.cancellable()` (scripts/gates/cancellable-actions.py has
# the matching rules). There is no allow list: a new action that ignores
# cancellation cannot land.
#
# Usage: scripts/gates/check-cancellable-actions.sh [ROOT]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${1:-$(cd "$here/../.." && pwd)}"
exec python3 "$here/cancellable-actions.py" "$root"
