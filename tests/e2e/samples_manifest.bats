#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# The completeness half of samples.bats's harness, split into its own file on
# purpose: this is a static file-list comparison, needs no dbus-daemon, no
# brain binary, no jeepney - so it must not sit behind samples.bats's
# setup_file() skip conditions, or a missing dependency on the machine would
# silently stop catching a new, unwired sample.

@test "every tracked sample script is accounted for in tests/e2e/samples/manifest.tsv" {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  cd "$REPO"
  local manifest="tests/e2e/samples/manifest.tsv"
  local tracked listed
  tracked="$(git ls-files 'samples/shell/*.py' 'samples/shell/*.sh' 'samples/python/*.py' 'samples/python/*.sh' | sort)"
  listed="$(tail -n +2 "$manifest" | cut -f1 | sort)"

  local missing extra
  missing="$(comm -23 <(echo "$tracked") <(echo "$listed"))"
  extra="$(comm -13 <(echo "$tracked") <(echo "$listed"))"

  if [ -n "$missing" ]; then
    echo "sample scripts not listed in $manifest:" >&3
    echo "$missing" >&3
  fi
  if [ -n "$extra" ]; then
    echo "$manifest lists paths that no longer exist / aren't tracked:" >&3
    echo "$extra" >&3
  fi
  [ -z "$missing" ]
  [ -z "$extra" ]
}
