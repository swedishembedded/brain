#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Assemble everything a fly simulation needs into ONE directory.
#
# The pieces come from four unrelated places with four different licences: a
# connectome export, a biomechanical model and its 85 meshes, a recorded
# kinematics dataset, and a physics engine. Running the fly currently means
# setting four environment variables at four paths nobody can remember, and the
# failure mode when one is wrong is not an error - it is a fly that stands
# still, or falls through the floor, or flaps without flying.
#
# So this collects them, records where each came from, and writes an `env.sh`
# that sets every variable at once. Nothing is transformed: the files are
# copied verbatim so that what runs is what was published.
#
# Usage:
#   tools/buzzfly/collect.sh [destination]
#
# Sources, each overridable and each REQUIRED to exist:
#   BUZZFLY_CONNECTOME  directory holding neurons.csv.gz + connections_princeton.csv.gz
#   BUZZFLY_BODY        the fruit-fly MJCF assets directory (fruitfly.xml and its meshes)
#   BUZZFLY_REFERENCE   converted walking reference (see tools/convert/)
#   BUZZFLY_MUJOCO      MuJoCo install root (the directory holding lib/libmujoco.so)
set -euo pipefail

dest="${1:-$HOME/Downloads/buzzfly}"

die() { printf 'buzzfly: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

# Each source is checked for the FILE that proves it is the right directory,
# not merely for the directory existing. A path that points at an empty folder
# is the mistake this is here to catch.
need_dir() {
	local var="$1" path="$2" proof="$3"
	[ -n "$path" ] || die "\$$var is not set and has no default on this machine"
	[ -e "$path/$proof" ] || die "\$$var ($path) does not contain $proof"
}

conn="${BUZZFLY_CONNECTOME:-}"
body="${BUZZFLY_BODY:-}"
ref="${BUZZFLY_REFERENCE:-}"
mj="${BUZZFLY_MUJOCO:-${BRAIN_MUJOCO_DIR:-${MUJOCO_DIR:-$HOME/.mujoco/mujoco-3.12.0}}}"

need_dir BUZZFLY_CONNECTOME "$conn" neurons.csv.gz
need_dir BUZZFLY_BODY "$body" fruitfly.xml
need_dir BUZZFLY_MUJOCO "$mj" lib/libmujoco.so
[ -n "$ref" ] && [ -f "$ref" ] || die "\$BUZZFLY_REFERENCE ($ref) is not a file"

echo "buzzfly: collecting into $dest"
mkdir -p "$dest/connectome/manc" "$dest/body" "$dest/reference"

note "connectome"
cp -f "$conn/neurons.csv.gz" "$conn/connections_princeton.csv.gz" "$dest/connectome/manc/"
[ -f "$conn/SOURCE.txt" ] && cp -f "$conn/SOURCE.txt" "$dest/connectome/manc/"

note "body and its meshes"
# Everything, not a hand-picked list: the MJCF names 85 mesh and texture files
# and a copy missing one of them fails at load with a message about a file,
# which is a much worse way to find out than copying 150 MB.
cp -rf "$body/." "$dest/body/"

note "recorded kinematics"
cp -f "$ref" "$dest/reference/$(basename "$ref")"

# MuJoCo is NOT copied. It is a versioned install with its own layout and its
# own licence, and a second copy is a second thing to keep current; env.sh
# points at wherever it already lives.
note "MuJoCo stays where it is ($mj)"

# Assembled rather than written literally: this script must contain exactly
# one licence line of its own, and a heredoc holding a second one reads as a
# duplicate to anything scanning the file.
tag="SPDX-License-Identifier"

# env.sh is written to be PORTABLE, because this directory gets copied between
# machines and a file full of one machine's absolute paths is a file that works
# exactly once. It finds itself, and it searches for MuJoCo rather than
# recording where MuJoCo happened to be the day it was generated.
cat > "$dest/env.sh" <<'ENV'
# TAG_PLACEHOLDER: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Source this, then run the fly:
#
#   . /path/to/buzzfly/env.sh
#   make samples/fly/interactive/run
#
# Written by brain's tools/buzzfly/collect.sh; edit that rather than this.

# Where this file is, whatever anyone renamed or moved the directory to. The
# data lives beside it, so nothing here needs to know an absolute path.
if [ -n "${BASH_SOURCE:-}" ]; then
	_buzzfly_self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
	_buzzfly_self="${(%):-%x}"
else
	# Sourced by a shell that does not say which file it is reading. The
	# caller's $0 is the best guess and is usually wrong, so say so rather
	# than silently resolving to their shell's directory.
	_buzzfly_self="$0"
fi
BUZZFLY_DIR="$(cd "$(dirname "$_buzzfly_self")" && pwd)"
unset _buzzfly_self
if [ ! -d "$BUZZFLY_DIR/connectome" ]; then
	echo "buzzfly: $BUZZFLY_DIR does not look like a buzzfly directory." >&2
	echo "buzzfly: source this file by its own path, e.g. '. /path/to/buzzfly/env.sh'." >&2
fi
export BUZZFLY_DIR

export BRAIN_CONNECTOME_DIR="$BUZZFLY_DIR/connectome"
export BRAIN_FLYBODY_XML="$BUZZFLY_DIR/body/floor.xml"
export BRAIN_FLYBODY_FRUITFLY_XML="$BUZZFLY_DIR/body/fruitfly.xml"
export BRAIN_FLY_REFERENCE="$BUZZFLY_DIR/reference/REFERENCE_PLACEHOLDER"

# MuJoCo is SEARCHED FOR rather than recorded. It is installed per machine,
# outside this directory, and a path captured when this file was generated is
# the one thing here guaranteed to be wrong somewhere else. An existing
# setting always wins, so a machine with MuJoCo somewhere unusual is not
# overridden.
#
# The conventional roots are spelled as overridable defaults rather than
# literals so that a machine that installs MuJoCo elsewhere can redirect the
# SEARCH without having to know its exact version-suffixed directory name (the
# glob is appended outside the expansion, so `${BRAIN_MUJOCO_OPT}` names a
# parent, not a match).
if [ -z "${BRAIN_MUJOCO_DIR:-}" ]; then
	for _buzzfly_mj in \
		"${MUJOCO_DIR:-}" "${MUJOCO_PATH:-}" \
		"${BRAIN_MUJOCO_HOME:-$HOME/.mujoco}"/mujoco-* \
		"${BRAIN_MUJOCO_OPT:-/opt}"/mujoco-* \
		"${BRAIN_MUJOCO_LOCAL:-/usr/local}"/mujoco-*; do
		if [ -n "$_buzzfly_mj" ] && [ -f "$_buzzfly_mj/lib/libmujoco.so" ]; then
			export BRAIN_MUJOCO_DIR="$_buzzfly_mj"
			break
		fi
	done
	unset _buzzfly_mj
fi
if [ -z "${BRAIN_MUJOCO_DIR:-}" ]; then
	echo "buzzfly: no MuJoCo found. Install it and set \$BRAIN_MUJOCO_DIR to the" >&2
	echo "buzzfly: directory holding lib/libmujoco.so, e.g. ~/.mujoco/mujoco-3.12.0." >&2
fi
ENV
# The licence tag and the reference filename are substituted in afterwards: the
# heredoc above is quoted so that everything else in it survives verbatim.
sed -i "s|TAG_PLACEHOLDER|$tag|; s|REFERENCE_PLACEHOLDER|$(basename "$ref")|" "$dest/env.sh"

{
	echo "# buzzfly"
	echo
	echo "Everything a fruit-fly simulation needs, in one place. Assembled by"
	echo "brain's \`tools/buzzfly/collect.sh\`; nothing here is transformed, the"
	echo "files are verbatim copies of what was published."
	echo
	echo '```sh'
	echo ". $dest/env.sh"
	echo "make samples/fly/interactive/run"
	echo '```'
	echo
	echo "## What is here, and where it came from"
	echo
	echo "| path | source | licence |"
	echo "|---|---|---|"
	echo "| \`connectome/manc/\` | $conn | CC-BY 4.0 (MANC, Janelia) |"
	echo "| \`body/\` | $body | Apache-2.0 (flybody model) |"
	echo "| \`reference/\` | $ref | GPL-3.0+ (flybody datasets) |"
	echo
	echo "MuJoCo is **not** copied here: it is a versioned install with its own"
	echo "layout and licence, and a second copy is a second thing to keep"
	echo "current. \`env.sh\` points at the one already installed at \`$mj\`."
	echo
	echo "## Licences are not uniform, and it matters"
	echo
	echo "The body model is Apache-2.0 and the recorded kinematics beside it are"
	echo "GPL-3.0+. They are different licences on files that sit in adjacent"
	echo "directories, so anything redistributing this directory as a whole"
	echo "inherits the stricter one."
	echo
	echo "## Contents"
	echo
	echo '```'
	(cd "$dest" && find . -maxdepth 2 -not -path './body/*' | sort | sed 's|^\./||')
	echo "body/  ($(find "$dest/body" -type f | wc -l) files, $(du -sh "$dest/body" | cut -f1))"
	echo '```'
} > "$dest/README.md"

echo "buzzfly: $(du -sh "$dest" | cut -f1) in $dest"
echo "buzzfly: run '. $dest/env.sh' and every BRAIN_* variable is set"
