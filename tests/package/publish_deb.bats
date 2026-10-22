#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible, signed software
# releases for its clients. If your team needs expertise in release
# engineering and Debian packaging then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# The spec of scripts/build/publish-deb.sh (`make package/deb/publish`), run against a
# scratch git repository, a scratch GnuPG home and a fake `gh` that keeps the
# GitHub release in files. Every refusal names what to do next; the happy
# paths create the release when it is missing, replace the package already
# attached to it, and leave a signature that verifies.

SCRIPT="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)/scripts/build/publish-deb.sh"

setup() {
  T="$BATS_TEST_TMPDIR"
  STATE="$T/github"
  mkdir -p "$STATE/uploaded" "$T/bin"
  : >"$STATE/calls"
  : >"$STATE/assets"
  touch "$STATE/logged-in"

  # The fake gh: the release, its assets and the login live under $STATE;
  # every call is appended to $STATE/calls.
  cat >"$T/bin/gh" <<GH
#!/usr/bin/env bash
state="$STATE"
GH
  cat >>"$T/bin/gh" <<'GH'
printf '%s\n' "$*" >>"$state/calls"
case "$1 $2" in
"auth status")
  [ -f "$state/logged-in" ] && exit 0
  echo "You are not logged into any GitHub hosts." >&2; exit 1 ;;
"repo view") echo WRITE ;;
"api repos/"*)
  [ -f "$state/remote-sha" ] && { cat "$state/remote-sha"; exit 0; }
  echo "gh: Not Found (HTTP 404)" >&2; exit 1 ;;
"release view")
  [ -f "$state/release" ] || { echo "release not found" >&2; exit 1; }
  case " $* " in *" assets "*) cat "$state/assets" ;; esac ;;
"release create") touch "$state/release" ;;
"release upload")
  [ -f "$state/release" ] || { echo "release not found" >&2; exit 1; }
  shift 3
  for f in "$@"; do
    [ -f "$f" ] || continue
    n=$(basename "$f")
    grep -qxF "$n" "$state/assets" || echo "$n" >>"$state/assets"
    cp "$f" "$state/uploaded/$n"
  done ;;
"release delete-asset")
  grep -vxF "$4" "$state/assets" >"$state/assets.new"; mv "$state/assets.new" "$state/assets" ;;
*) echo "fake gh: unexpected call: $*" >&2; exit 64 ;;
esac
GH
  chmod +x "$T/bin/gh"
  PATH="$T/bin:$PATH"

  export GNUPGHOME="$T/gnupg"
  mkdir -m 700 "$GNUPGHOME"
  gpg --batch --quiet --passphrase '' --quick-gen-key 'Release Test <release@example.invalid>' ed25519 sign never 2>/dev/null

  REPO="$T/repo"
  git init -q "$REPO"
  git -C "$REPO" -c user.name=t -c user.email=t@example.invalid commit -q --allow-empty -m release
  git -C "$REPO" tag v1.2.3
  git -C "$REPO" remote add origin git@github.com:owner/demo.git
  git -C "$REPO" rev-parse HEAD >"$STATE/remote-sha"
  cd "$REPO"

  DEB=$(make_deb 1.2.3)
}

teardown() {
  gpgconf --kill gpg-agent 2>/dev/null || true
}

# make_deb VERSION - a minimal real package named demo, printed path.
make_deb() {
  local root="$T/deb-$1"
  mkdir -p "$root/DEBIAN" "$root/usr/share/doc/demo"
  echo demo >"$root/usr/share/doc/demo/README"
  printf 'Package: demo\nVersion: %s\nArchitecture: amd64\nMaintainer: t <t@example.invalid>\nDescription: demo\n' "$1" >"$root/DEBIAN/control"
  mkdir -p "$T/out"
  dpkg-deb --build --root-owner-group "$root" "$T/out/demo_$1_amd64.deb" >/dev/null
  echo "$T/out/demo_$1_amd64.deb"
}

@test "publish: explains how to install gh when it is missing" {
  # A PATH holding every tool the system has, except gh.
  mkdir "$T/nogh"
  IFS=: read -ra dirs <<<"$PATH"
  for d in "${dirs[@]}"; do [ "$d" = "$T/bin" ] || ln -sf "$d"/* "$T/nogh/" 2>/dev/null || true; done
  rm -f "$T/nogh/gh"
  PATH="$T/nogh" run "$T/nogh/bash" "$SCRIPT" --version 1.2.3 "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"https://cli.github.com"* ]]
  [[ "$output" == *"gh auth login"* ]]
}

@test "publish: explains gh auth login when not logged in" {
  rm "$STATE/logged-in"
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"gh auth login"* ]]
  ! grep -q '^release' "$STATE/calls"
}

@test "publish: refuses when HEAD is not the release tag" {
  git -c user.name=t -c user.email=t@example.invalid commit -q --allow-empty -m after
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"git checkout v1.2.3"* ]]
}

@test "publish: refuses when the version has no tag yet" {
  run bash "$SCRIPT" --version 1.2.4 "$(make_deb 1.2.4)"
  [ "$status" -ne 0 ]
  [[ "$output" == *"v1.2.4"* ]]
  [[ "$output" == *"git tag"* ]]
}

@test "publish: refuses a working tree with uncommitted changes" {
  echo x >file && git add file
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"git status"* ]]
}

@test "publish: explains how to push the tag when GitHub does not have it" {
  rm "$STATE/remote-sha"
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"git push origin v1.2.3"* ]]
}

@test "publish: explains how to create a signing key when there is none" {
  gpgconf --kill gpg-agent
  rm -rf "$GNUPGHOME" && mkdir -m 700 "$GNUPGHOME"
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"gpg --full-generate-key"* ]]
}

@test "publish: refuses a package whose version is not the tag's" {
  run bash "$SCRIPT" --version 1.2.3 "$(make_deb 1.2.2)"
  [ "$status" -ne 0 ]
  [[ "$output" == *"1.2.2"* ]]
  ! grep -q '^release upload' "$STATE/calls"
}

@test "publish: --check verifies everything up front and publishes nothing" {
  run bash "$SCRIPT" --version 1.2.3 --check
  [ "$status" -eq 0 ]
  ! grep -qE '^release (create|upload|delete-asset)' "$STATE/calls"
}

@test "publish: creates the missing release and uploads a package whose signature verifies" {
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  echo "$output"
  [ "$status" -eq 0 ]
  grep -q '^release create v1.2.3 .*--repo owner/demo.*--verify-tag' "$STATE/calls"
  grep -q '^release upload v1.2.3 .*--clobber' "$STATE/calls"
  grep -qxF demo_1.2.3_amd64.deb "$STATE/assets"
  grep -qxF demo_1.2.3_amd64.deb.asc "$STATE/assets"
  gpg --verify "$STATE/uploaded/demo_1.2.3_amd64.deb.asc" "$STATE/uploaded/demo_1.2.3_amd64.deb"
}

@test "publish: replaces the package already on the release and keeps other assets" {
  touch "$STATE/release"
  printf '%s\n' demo_1.2.3_amd64.deb demo_1.2.3_amd64.deb.asc demo_1.2.3-1_amd64.deb demo_1.2.3-1_amd64.deb.asc demo-1.2.3.tar.gz >"$STATE/assets"
  echo stale >"$STATE/uploaded/demo_1.2.3_amd64.deb"
  run bash "$SCRIPT" --version 1.2.3 "$DEB"
  echo "$output"
  [ "$status" -eq 0 ]
  ! grep -q '^release create' "$STATE/calls"
  cmp "$DEB" "$STATE/uploaded/demo_1.2.3_amd64.deb"
  [ "$(LC_ALL=C sort "$STATE/assets")" = "$(printf '%s\n' demo-1.2.3.tar.gz demo_1.2.3_amd64.deb demo_1.2.3_amd64.deb.asc)" ]
}

@test "publish: --remote picks the GitHub repository and the push it asks for" {
  git remote add upstream https://github.com/other/demo.git
  rm "$STATE/remote-sha"
  run bash "$SCRIPT" --version 1.2.3 --remote upstream "$DEB"
  [ "$status" -ne 0 ]
  [[ "$output" == *"git push upstream v1.2.3"* ]]
  grep -q 'repos/other/demo/commits/v1.2.3' "$STATE/calls"
}
