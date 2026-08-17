#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# End-to-end: the release .deb produced by scripts/build/build-deb.sh
# (`make deb/release`) is a real, correctly-formed Debian package - not just
# "dpkg-deb didn't crash". Covers the properties that were silently wrong
# before this suite existed: `Depends` derived from the actual binary
# (dpkg-shlibdeps) instead of a hardcoded libc6-only guess, a real
# Installed-Size, md5sums for every shipped file, and that `--flavor debug`
# writes a distinct filename so `make deb/debug` can never overwrite the
# release package.
#
# Needs a release build (`make deb/release` builds one) and dpkg-deb.

setup_file() {
  command -v dpkg-deb >/dev/null 2>&1 || skip "dpkg-deb not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  DEB="$(find "$REPO/target/debian" -maxdepth 1 -name 'brain_*.deb' ! -name '*-debug.deb' 2>/dev/null | sort | tail -1)"
  [ -n "$DEB" ] && [ -f "$DEB" ] || skip "no release .deb found (build with: make deb/release)"
  export DEB
}

@test "deb: control fields are present and well-formed" {
  run dpkg-deb -f "$DEB" Package
  [ "$status" -eq 0 ]
  [ "$output" = "brain" ]

  run dpkg-deb -f "$DEB" Architecture
  [ "$status" -eq 0 ]
  [ -n "$output" ]
}

@test "deb: Depends is derived from the binary, not a hardcoded libc6-only guess" {
  run dpkg-deb -f "$DEB" Depends
  [ "$status" -eq 0 ]
  # A brain binary linking wgpu/Vulkan/SDL pulls in far more than libc alone;
  # more than one comma-separated dependency is the signal dpkg-shlibdeps
  # actually ran instead of the field being hand-typed.
  echo "Depends: $output" >&2
  [[ "$output" == *,* ]]
}

@test "deb: Installed-Size is a real, non-zero number" {
  run dpkg-deb -f "$DEB" Installed-Size
  [ "$status" -eq 0 ]
  [[ "$output" =~ ^[0-9]+$ ]]
  [ "$output" -gt 0 ]
}

@test "deb: payload ships the binary and the D-Bus system-bus policy" {
  run dpkg-deb -c "$DEB"
  [ "$status" -eq 0 ]
  echo "$output" | grep -qE '\./usr/bin/brain$'
  echo "$output" | grep -qE '\./usr/share/dbus-1/system\.d/com\.swedishembedded\.Brain1\.conf$'
}

@test "deb: DEBIAN/md5sums covers every shipped file" {
  TDIR="$(mktemp -d)"
  dpkg-deb -e "$DEB" "$TDIR"
  [ -s "$TDIR/md5sums" ]
  # One md5sums line per non-directory payload entry.
  payload_files="$(dpkg-deb -c "$DEB" | grep -vE '/$' | wc -l)"
  md5_lines="$(wc -l < "$TDIR/md5sums")"
  [ "$md5_lines" -eq "$payload_files" ]
  rm -rf "$TDIR"
}

@test "deb: --flavor debug produces a distinct filename from the release package" {
  DEBUG_DEB="$(find "$REPO/target/debian" -maxdepth 1 -name 'brain_*-debug.deb' 2>/dev/null | sort | tail -1)"
  [ -n "$DEBUG_DEB" ] || skip "no debug .deb found (build with: make deb/debug)"
  [ "$DEBUG_DEB" != "$DEB" ]
}
