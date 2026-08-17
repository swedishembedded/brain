#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Build a standalone Debian package for the Brain CLI without cargo-deb.
#
# Depends are computed from the actual binary via dpkg-shlibdeps (brain links
# wgpu/Vulkan/SDL - a hardcoded libc6-only guess would be wrong), the control
# file carries Installed-Size, and the package ships md5sums plus a Debian
# changelog. Pass --flavor debug for a debug build so its output filename
# cannot overwrite the release .deb.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${ROOT}/target/debian"
BINARY="${ROOT}/target/release/brain"
ARCH="$(dpkg --print-architecture)"
FLAVOR=""

while (($#)); do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --flavor) FLAVOR="$2"; shift 2 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ -f "${BINARY}" ]] || { printf 'error: binary not found: %s\n' "${BINARY}" >&2; exit 1; }
command -v dpkg-deb >/dev/null 2>&1 || { echo 'error: dpkg-deb is required' >&2; exit 1; }
VERSION="$(grep '^version' "${ROOT}/Cargo.toml" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
PACKAGE="brain_${VERSION}_${ARCH}${FLAVOR:+-${FLAVOR}}"
STAGING="${ROOT}/target/debian-staging/${PACKAGE}"
rm -rf "${STAGING}"
install -d "${STAGING}/DEBIAN" "${STAGING}/usr/bin" "${STAGING}/usr/share/brain" \
    "${STAGING}/usr/share/doc/brain" "${STAGING}/usr/share/dbus-1/system.d"
install -m 0755 "${BINARY}" "${STAGING}/usr/bin/brain"
cp -a "${ROOT}/examples/." "${STAGING}/usr/share/brain/examples/"
# examples/ is a live working tree, not a release artifact - drop whatever
# gitignored Python bytecode cache happens to exist there at build time so
# the package payload isn't machine/interpreter-version dependent.
find "${STAGING}/usr/share/brain/examples" -name '__pycache__' -type d -prune -exec rm -rf {} +
# Vetted system-bus policy for `brain serve --dbus-system`: without a shipped
# default, operators hand-write the path-of-least-resistance allow-everyone
# policy, which grants every local user model execution + auto-fetch. This one
# restricts calls to root and the `brain` group (see the file's own comments).
install -m 0644 "${SCRIPT_DIR}/com.swedishembedded.Brain1.conf" "${STAGING}/usr/share/dbus-1/system.d/com.swedishembedded.Brain1.conf"

# Runtime library dependencies, derived from the binary itself. Requires
# dpkg-shlibdeps (dpkg-dev); fall back to a libc-only floor with a loud
# warning so a minimal build host still produces a package.
DEPENDS="libc6 (>= 2.35)"
if command -v dpkg-shlibdeps >/dev/null 2>&1; then
    SHLIBS_TMP="$(mktemp -d)"
    trap 'rm -rf "${SHLIBS_TMP}"' EXIT
    mkdir -p "${SHLIBS_TMP}/debian"
    cat >"${SHLIBS_TMP}/debian/control" <<EOF
Source: brain
Package: brain
Architecture: any
Description: stub control for dpkg-shlibdeps
EOF
    DEPENDS="$(cd "${SHLIBS_TMP}" \
        && dpkg-shlibdeps -O "${STAGING}/usr/bin/brain" 2>/dev/null \
        | sed -n 's/^shlibs:Depends=//p')"
    [[ -n "${DEPENDS}" ]] || { echo 'error: dpkg-shlibdeps produced no dependencies' >&2; exit 1; }
else
    echo 'warning: dpkg-shlibdeps not found (install dpkg-dev); using libc-only Depends' >&2
fi

INSTALLED_SIZE="$(du -sk --exclude=DEBIAN "${STAGING}" | cut -f1)"

cat >"${STAGING}/DEBIAN/control" <<EOF
Package: brain
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Swedish Embedded <info@swedishembedded.com>
Installed-Size: ${INSTALLED_SIZE}
Depends: ${DEPENDS}
Section: utils
Priority: optional
Homepage: https://github.com/swedishembedded/brain
Description: Swedish Embedded model training and inference runtime
 Brain is the native CLI and API server for Swedish Embedded's model runtime:
 training, evaluation and serving for the architectures listed by \`brain caps\`.
 .
 Bundled example scripts are installed under /usr/share/brain/examples/; the
 D-Bus system-bus policy for \`brain serve --dbus-system\` is installed under
 /usr/share/dbus-1/system.d/.
EOF
cat >"${STAGING}/usr/share/doc/brain/copyright" <<'EOF'
Copyright: 2026 Swedish Embedded
License: MIT
EOF
cat >"${STAGING}/usr/share/doc/brain/README.Debian" <<'EOF'
brain for Debian
----------------

Run `brain --help` for the full CLI reference and `brain caps` to list the
architectures this build can train/eval/serve.

Bundled data installed under /usr/share/brain/:
  examples/  runnable example scripts (see each subdirectory's own docs)

The D-Bus system-bus policy for `brain serve --dbus-system`
(com.swedishembedded.Brain1) is installed under
/usr/share/dbus-1/system.d/. It restricts calls to root and the `brain`
group; add an operator with `usermod -aG brain <user>` (create the group
first: `groupadd --system brain`).

Model weights are not shipped in this package - point the relevant
BRAIN_*_WEIGHTS / BRAIN_*_DIR environment variable (see `brain --help` and
docs/models/) at a local checkpoint, or let auto-fetch retrieve one.
EOF
gzip -9n >"${STAGING}/usr/share/doc/brain/changelog.Debian.gz" <<EOF
brain (${VERSION}) unstable; urgency=medium

  * Upstream release ${VERSION}. See https://github.com/swedishembedded/brain/releases for release notes.

 -- Swedish Embedded <info@swedishembedded.com>  $(date -R)
EOF

(cd "${STAGING}" && find usr -type f | LC_ALL=C sort | xargs md5sum >DEBIAN/md5sums)

mkdir -p "${OUT_DIR}"
dpkg-deb --build --root-owner-group "${STAGING}" "${OUT_DIR}/${PACKAGE}.deb"
printf '%s\n' "${OUT_DIR}/${PACKAGE}.deb"
