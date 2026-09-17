#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Every directory under crates/ - and every samples/<category>/<name>/ - must
# be a crate whose manifest PARSES. samples/shell/ and samples/python/ are
# EXCLUDED (root Cargo.toml's `exclude`): they hold shell/Python client
# samples, never a Cargo.toml, and are skipped here for the same reason
# cargo itself skips them.
#
# The workspace takes its members from the globs `members = ["crates/*",
# "samples/*/*"]`, so one bad manifest under EITHER does not fail one crate -
# it makes the whole workspace unresolvable, at manifest-parse time, before
# any build script, test or gate can run. A half-finished sample directory is
# therefore exactly as fatal as a half-finished crate.
#
# Two failure modes, and they are not equally well reported:
#
#   1. A directory with no Cargo.toml. Cargo names it correctly, so this check
#      only gets there first.
#
#   2. A Cargo.toml that does not PARSE - most often unresolved merge/stash
#      conflict markers. Cargo reports this as a chain of "failed to load
#      manifest for dependency ..." lines rooted at whichever member it
#      happened to be resolving, so the path it prints is an INNOCENT crate
#      and the broken file is never named at all:
#
#        error: failed to load manifest for workspace member `crates/apiserve`
#        Caused by: failed to load manifest for dependency `brain-imaging`
#        Caused by: failed to load manifest for dependency `brain-vision`
#        ... four more, none of them the actual file ...
#
#      That is the one that cost a working day. This check names the file and
#      the line.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 2

fail=0

for d in crates/*/ samples/*/*/; do
    [ -d "$d" ] || continue
    case "$d" in samples/shell/*|samples/python/*) continue ;; esac
    if [ ! -f "${d}Cargo.toml" ]; then
        echo "check-workspace-members: FAIL - ${d} has no Cargo.toml"
        echo "    The workspace globs crates/* and samples/*/*, so this makes EVERY"
        echo "    member unresolvable. Finish it, remove the directory, or move it out."
        fail=1
    fi
done

# Conflict markers first: a more specific and far more likely diagnosis than
# "invalid TOML", and worth saying by name.
if markers=$(grep -rln --include=Cargo.toml -E '^(<<<<<<< |=======$|>>>>>>> )' crates/ samples/ Cargo.toml 2>/dev/null); then
    if [ -n "$markers" ]; then
        echo "check-workspace-members: FAIL - unresolved conflict markers in a manifest:"
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            echo "    $f:$(grep -nE '^(<<<<<<< |=======$|>>>>>>> )' "$f" | head -1 | cut -d: -f1)"
        done <<< "$markers"
        echo
        echo "    Cargo cannot parse these, and its own error will name an unrelated crate."
        fail=1
    fi
fi

# Then anything else that simply is not valid TOML.
bad=$(python3 - <<'PY'
import glob, tomllib
for f in sorted(glob.glob("crates/*/Cargo.toml") + glob.glob("samples/*/*/Cargo.toml")) + ["Cargo.toml"]:
    try:
        with open(f, "rb") as fh:
            tomllib.load(fh)
    except FileNotFoundError:
        pass
    except tomllib.TOMLDecodeError as e:
        print(f"{f}: {e}")
PY
)
if [ -n "$bad" ]; then
    echo "check-workspace-members: FAIL - manifest does not parse as TOML:"
    echo "$bad" | sed 's/^/    /'
    fail=1
fi

[ "$fail" -ne 0 ] && exit 1
echo "check-workspace-members: exit 0, $(ls -d crates/*/ | wc -l | tr -d ' ') crates + $(find samples -mindepth 3 -maxdepth 3 -name Cargo.toml 2>/dev/null | wc -l | tr -d ' ') samples, all manifests parse"
exit 0
