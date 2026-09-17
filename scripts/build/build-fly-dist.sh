#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Package the fly so it runs on a machine with no Rust toolchain.
#
# A tarball rather than a .deb: this has to unpack and run as an ordinary user
# on a box where nobody is going to be granted root to install a package, and
# where `cargo` is not going to exist. Everything the fly needs at run time
# that is not a system library goes in, including MuJoCo, which is dlopen'd
# rather than linked and so can simply be carried along.
#
# What is NOT bundled: the connectome. It is ~95 MB of third-party data under
# its own licence, it changes on its own release schedule, and a machine
# running the fly twice should not hold two copies. `--with-data` overrides
# that for an air-gapped target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
OUT_DIR="${ROOT}/target/dist"
WITH_DATA=0
WITH_MUJOCO=1
DATA_DIR="${BUZZFLY_DIR:-}"

while (($#)); do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --with-data) WITH_DATA=1; shift ;;
        --no-mujoco) WITH_MUJOCO=0; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

VERSION="$(grep '^version' "${ROOT}/Cargo.toml" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
ARCH="$(uname -m)"
NAME="brain-fly-${VERSION}-${ARCH}"
STAGE="${ROOT}/target/dist-staging/${NAME}"
rm -rf "${STAGE}"
install -d "${STAGE}/bin" "${STAGE}/lib" "${STAGE}/share" "${STAGE}/LICENSES"

# The demonstration, and the experiments behind it. Named here rather than
# globbed so that adding an example does not silently change what ships.
EXPERIMENTS=(cpg conditioning calibrate replay walk_search smell_check gait_report)

echo "building (release, this is the only step that needs cargo) ..."
( cd "${ROOT}" && cargo build --release --offline -p sample-fly-interactive )
for e in "${EXPERIMENTS[@]}"; do
    ( cd "${ROOT}" && cargo build --release --offline -p brain-fly --example "$e" )
done

install -m 0755 "${ROOT}/target/release/sample-fly-interactive" "${STAGE}/bin/"
for e in "${EXPERIMENTS[@]}"; do
    install -m 0755 "${ROOT}/target/release/examples/${e}" "${STAGE}/bin/${e}"
done

# MuJoCo is loaded with dlopen at run time, so carrying the shared object is
# enough and no relinking is involved. Apache-2.0, so redistributable with its
# licence, which is copied beside it.
if ((WITH_MUJOCO)); then
    MJ="${BRAIN_MUJOCO_DIR:-${MUJOCO_DIR:-}}"
    if [[ -n "${MJ}" && -d "${MJ}/lib" ]]; then
        cp -a "${MJ}/lib/"libmujoco.so* "${STAGE}/lib/" 2>/dev/null || true
        for l in LICENSE LICENSES/*; do
            [[ -f "${MJ}/${l}" ]] && cp -a "${MJ}/${l}" "${STAGE}/LICENSES/MUJOCO-LICENSE" && break
        done
        echo "bundled MuJoCo from ${MJ}"
    else
        echo "warning: no MuJoCo found (set BRAIN_MUJOCO_DIR); the target machine will need its own" >&2
    fi
fi

if ((WITH_DATA)); then
    [[ -n "${DATA_DIR}" ]] || { echo 'error: --with-data needs --data-dir or $BUZZFLY_DIR' >&2; exit 1; }
    echo "bundling the connectome and body from ${DATA_DIR} (this is the large part) ..."
    install -d "${STAGE}/share/buzzfly"
    cp -a "${DATA_DIR}/connectome" "${DATA_DIR}/body" "${STAGE}/share/buzzfly/"
    [[ -d "${DATA_DIR}/reference" ]] && cp -a "${DATA_DIR}/reference" "${STAGE}/share/buzzfly/"
fi

# Any tuning that has been measured travels with the binaries: a demo with no
# parameters shows the connectome as imported, which barely moves, and that
# reads as "it does not work" rather than "you did not give it the answer".
install -d "${STAGE}/share/tuning"
shopt -s nullglob
for t in "${ROOT}"/testdata/fly/*.txt; do install -m 0644 "$t" "${STAGE}/share/tuning/"; done
shopt -u nullglob

install -m 0644 "${ROOT}/LICENSE" "${STAGE}/LICENSES/BRAIN-LICENSE" 2>/dev/null || true

# The launcher's own licence header is written here rather than inside the
# heredoc below: a literal second "SPDX-License-Identifier" line in this file
# is indistinguishable, to the repository's header check, from this script
# having two of its own.
{
    printf '#!/usr/bin/env bash\n'
    printf '# %s: Apache-2.0\n' 'SPDX-License-Identifier'
    printf '# Copyright (c) 2026 Martin Schroder <info@swedishembedded.com>\n'
} >"${STAGE}/fly"

cat >>"${STAGE}/fly" <<'LAUNCHER'
#
# One entry point for the packaged fly. No cargo, no repository, no toolchain.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export LD_LIBRARY_PATH="${HERE}/lib:${LD_LIBRARY_PATH:-}"
[[ -d "${HERE}/lib" ]] && export BRAIN_MUJOCO_DIR="${BRAIN_MUJOCO_DIR:-${HERE}}"

# Data: bundled if it was packaged, otherwise wherever the operator put it.
DATA="${BUZZFLY_DIR:-${HERE}/share/buzzfly}"
export BRAIN_CONNECTOME_DIR="${BRAIN_CONNECTOME_DIR:-${DATA}/connectome}"
export BRAIN_FLYBODY_XML="${BRAIN_FLYBODY_XML:-${DATA}/body/floor.xml}"
export BRAIN_FLYBODY_FRUITFLY_XML="${BRAIN_FLYBODY_FRUITFLY_XML:-${DATA}/body/fruitfly.xml}"
export BRAIN_DEVICE="${BRAIN_DEVICE:-cpu}"

usage() {
    cat <<USAGE
usage: ./fly <command> [options]

  demo [args...]      the fly in a window, driven by its own connectome
  cpg                 is the walking rhythm in the wiring? (experiment)
  conditioning        can the mushroom body learn an odour? (experiment)
  calibrate           fit the physiology the connectome does not contain
  train [args...]     search the fly's parameters, writing a tuning
  replay <tuning>     re-measure a tuning against its controls
  env                 print where this is looking for MuJoCo and the data

The connectome is not bundled unless this package was built --with-data.
Point \$BUZZFLY_DIR at a directory holding connectome/ and body/, or set
\$BRAIN_CONNECTOME_DIR and \$BRAIN_FLYBODY_XML directly.

Anything after the command is passed through, so:
  ./fly demo --tuning share/tuning/walk.txt --food 2,0,0 --seek
  ./fly cpg            # honours TICKS=, CURRENTS=, DN=
  ./fly train OUT=/tmp/t.txt GENERATIONS=40
USAGE
}

need_data() {
    [[ -d "${BRAIN_CONNECTOME_DIR}" ]] || {
        echo "error: no connectome at ${BRAIN_CONNECTOME_DIR}" >&2
        echo "       set \$BUZZFLY_DIR to a directory holding connectome/ and body/" >&2
        exit 2
    }
}

# Experiments take their settings as NAME=value, so anything of that shape
# before the arguments is exported rather than passed along.
split_env() {
    ENVARGS=()
    REST=()
    for a in "$@"; do
        if [[ "$a" == *=* && "$a" != -* ]]; then ENVARGS+=("$a"); else REST+=("$a"); fi
    done
}

cmd="${1:-}"
shift || true
case "${cmd}" in
    demo)
        need_data
        exec "${HERE}/bin/sample-fly-interactive" \
            --connectome "${BRAIN_CONNECTOME_DIR}" \
            --body "${BRAIN_FLYBODY_FRUITFLY_XML}" "$@"
        ;;
    cpg|conditioning|calibrate)
        need_data
        split_env "$@"
        exec env "${ENVARGS[@]:-}" "${HERE}/bin/${cmd}" "${REST[@]:-}"
        ;;
    train)
        need_data
        split_env "$@"
        exec env "${ENVARGS[@]:-}" "${HERE}/bin/walk_search" "${REST[@]:-}"
        ;;
    replay)
        need_data
        [[ $# -ge 1 ]] || { echo "usage: ./fly replay <tuning.txt>" >&2; exit 2; }
        split_env "$@"
        exec env "${ENVARGS[@]:-}" "${HERE}/bin/replay" "${REST[@]:-}"
        ;;
    env)
        echo "BRAIN_MUJOCO_DIR     = ${BRAIN_MUJOCO_DIR:-<unset>}"
        echo "BRAIN_CONNECTOME_DIR = ${BRAIN_CONNECTOME_DIR}"
        echo "BRAIN_FLYBODY_XML    = ${BRAIN_FLYBODY_XML}"
        echo "BRAIN_DEVICE         = ${BRAIN_DEVICE}"
        [[ -d "${BRAIN_CONNECTOME_DIR}" ]] && echo "connectome: found" || echo "connectome: MISSING"
        ls "${HERE}/lib"/libmujoco.so* >/dev/null 2>&1 && echo "mujoco: bundled" || echo "mujoco: not bundled, using the system one"
        ;;
    ""|-h|--help|help) usage ;;
    *) echo "unknown command: ${cmd}" >&2; usage; exit 2 ;;
esac
LAUNCHER
chmod 0755 "${STAGE}/fly"

cat >"${STAGE}/README" <<README
brain-fly ${VERSION} (${ARCH})

A real Drosophila connectome driving a real body. No toolchain required:
everything here is already built.

    ./fly env          check what it can find
    ./fly demo         the fly in a window
    ./fly cpg          the experiment behind the walking rhythm

The connectome is $( ((WITH_DATA)) && echo "bundled under share/buzzfly." || echo "NOT bundled. Set \$BUZZFLY_DIR to a directory
holding connectome/ and body/, or set \$BRAIN_CONNECTOME_DIR and
\$BRAIN_FLYBODY_XML directly." )

System libraries still expected on the target: libc, and for ./fly demo an
EGL-capable GPU stack. The experiments run on the CPU backend and need
neither a GPU nor a display.

Licences for everything bundled are in LICENSES/.
README

install -d "${OUT_DIR}"
TARBALL="${OUT_DIR}/${NAME}.tar.gz"
tar -czf "${TARBALL}" -C "$(dirname "${STAGE}")" "${NAME}"
printf '\nwrote %s (%s)\n' "${TARBALL}" "$(du -h "${TARBALL}" | cut -f1)"
printf 'unpack on the target and run ./fly env\n'
