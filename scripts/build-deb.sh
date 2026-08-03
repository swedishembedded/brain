#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Build a standalone release Debian package for the Brain CLI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
OUT_DIR="${ROOT}/target/debian"
BINARY="${ROOT}/target/release/brain"
ARCH="$(dpkg --print-architecture)"

while (($#)); do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ -f "${BINARY}" ]] || { printf 'error: release binary not found: %s\n' "${BINARY}" >&2; exit 1; }
command -v dpkg-deb >/dev/null 2>&1 || { echo 'error: dpkg-deb is required' >&2; exit 1; }
VERSION="$(grep '^version' "${ROOT}/Cargo.toml" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
PACKAGE="brain_${VERSION}_${ARCH}"
STAGING="${ROOT}/target/debian-staging/${PACKAGE}"
rm -rf "${STAGING}"
install -d "${STAGING}/DEBIAN" "${STAGING}/usr/bin" "${STAGING}/usr/share/doc/brain"
install -m 0755 "${BINARY}" "${STAGING}/usr/bin/brain"

cat >"${STAGING}/DEBIAN/control" <<EOF
Package: brain
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Swedish Embedded <info@swedishembedded.com>
Depends: libc6 (>= 2.35)
Section: utils
Priority: optional
Description: Swedish Embedded model training and inference runtime
 Brain is the native CLI and API server for Swedish Embedded's model runtime.
EOF
cat >"${STAGING}/usr/share/doc/brain/copyright" <<'EOF'
Copyright: 2026 Swedish Embedded
License: MIT
EOF
mkdir -p "${OUT_DIR}"
dpkg-deb --build --root-owner-group "${STAGING}" "${OUT_DIR}/${PACKAGE}.deb"
printf '%s\n' "${OUT_DIR}/${PACKAGE}.deb"
