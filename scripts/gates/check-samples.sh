#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Gate for samples/ - the standalone applications built on the public `brain`
# SDK. Invoked by `make check/samples`.
#
# Enforces samples/README.md. The manifest checks are cheap reads; the two that
# matter are measurements:
#
#   CLOSURE     - a sample's real `cargo tree` graph contains nothing reachable
#                 ONLY through an SDK surface it did not enable. This is the
#                 check that literally enforces "pull in only its required
#                 dependencies", and the surface -> crate mapping is DERIVED
#                 from the SDK's own manifest, so it cannot drift from what the
#                 features do.
#   INCREMENTAL - editing a sample's own sources and rebuilding compiles
#                 exactly one crate, and none of them a brain crate.
#
# A declared `[package.metadata.brain] max-brain-crates` budget per sample turns
# closure growth into a decision someone makes in a commit rather than a drift
# nobody notices.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Pin CARGO_HOME exactly as the Makefile does. Cargo records the ABSOLUTE
# source path of every registry crate in its fingerprints, so building under a
# different CARGO_HOME shares no artifacts with what `make` produced and the
# measurements below would report a cold third-party graph that has nothing to
# do with samples.
export CARGO_HOME="${CARGO_HOME_OVERRIDE:-$HOME/.cargo}"

fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; fail=1; }

mapfile -t manifests < <(find samples -mindepth 3 -maxdepth 3 -name Cargo.toml | sort)
if [ "${#manifests[@]}" -eq 0 ]; then
	echo "check/samples: no samples yet - nothing to check"
	exit 0
fi
echo "check/samples: ${#manifests[@]} sample(s)"

# ---------------------------------------------------------------- manifests
for m in "${manifests[@]}"; do
	dir="$(dirname "$m")"
	rel="${dir#samples/}"
	want="sample-${rel//\//-}"
	got="$(sed -n 's/^name = "\(.*\)"/\1/p' "$m" | head -1)"

	# The Makefile's samples/%/{build,run} rules resolve the package this way.
	[ "$got" = "$want" ] || bad "$m: package is '$got', path implies '$want'"

	# Rule 1: the only brain dependency is the SDK facade.
	while read -r dep; do
		case "$dep" in
			brain) : ;;
			brain-*) bad "$dep: $dir reaches past the SDK into an engine crate (samples/README.md rule 1)" ;;
		esac
	done < <(sed -n '/^\[dependencies\]/,/^\[/p' "$m" \
		| sed -n 's/^\([A-Za-z0-9_-]*\)[ .=].*/\1/p' | grep '^brain' || true)

	[ -f "$dir/README.md" ] || bad "$dir: no README.md"
	while read -r src; do
		head -3 "$src" | grep -q 'SPDX-License-Identifier' || bad "$src: no SPDX header"
	done < <(find "$dir/src" -name '*.rs' 2>/dev/null || true)
done

# --------------------------------------------------- shell/python samples
# These trees carry no Cargo closure to police (root Cargo.toml excludes
# them); the contract is structural: a README, an SPDX-headed entry point
# (run via `bash script.sh` / `python3 script.py`, so the executable bit
# itself is not required - and is inconsistent in this tree already), and no
# committed fixture.
mapfile -t script_dirs < <(find samples/shell samples/python -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort)
if [ "${#script_dirs[@]}" -gt 0 ]; then
	echo "check/samples: ${#script_dirs[@]} shell/python sample(s)"
	fixture_re='\.(csv|ply|wav|mp3|mp4|png|jpe?g|ppm|pt|pth|safetensors|weights|gguf|bin|npy|npz)$'
	for d in "${script_dirs[@]}"; do
		[ -f "$d/README.md" ] || bad "$d: no README.md"

		mapfile -t entries < <(find "$d" -maxdepth 1 \( -name '*.py' -o -name '*.sh' \) 2>/dev/null | sort)
		[ "${#entries[@]}" -gt 0 ] || bad "$d: no .py or .sh entry point"
		for e in "${entries[@]}"; do
			head -3 "$e" | grep -q 'SPDX-License-Identifier' || bad "$e: no SPDX header"
		done

		while read -r f; do
			bad "$f: looks like a committed fixture - samples fetch/generate their own input (see samples/README.md), never commit one"
		done < <(find "$d" -maxdepth 1 -type f -regextype posix-extended -regex ".*$fixture_re" 2>/dev/null)
	done
fi

# ------------------------------------------------------------------ images
# A README pointing at an image the repo does not CARRY renders as a broken box
# for everyone who clones it, and the author is the one person who cannot see
# it: the untracked file is sitting right there in their working tree. The
# mirror of it - an image nothing points at - is weight every clone pays for and
# no reader ever sees, which is what a generated chart written into a tracked
# directory turns into. Neither survives review reliably, so both are measured.
python3 - <<'PY' || fail=1
import pathlib, re, subprocess, sys

tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split())
IMG = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
ref_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
src_re = re.compile(r"<img[^>]+src=\"([^\"]+)\"")
root = pathlib.Path.cwd()
ok = True
referenced = set()

# Every tracked markdown under samples/, not just README.md - a sample may
# carry a second document (arena's CRITERIA.md), and scanning only READMEs
# would report an image that document legitimately uses as an orphan.
for rm in sorted(pathlib.Path(p) for p in tracked if p.startswith("samples/") and p.lower().endswith(".md")):
    if not rm.exists():
        continue
    text = rm.read_text()
    for ref in ref_re.findall(text) + src_re.findall(text):
        if ref.startswith(("http://", "https://", "data:")):
            continue
        target = (rm.parent / ref).resolve()
        if not target.is_relative_to(root):
            print(f"FAIL {rm}: image {ref!r} escapes the repository")
            ok = False
            continue
        rel = str(target.relative_to(root))
        referenced.add(rel)
        if not target.exists():
            print(f"FAIL {rm}: image {ref!r} does not exist")
            ok = False
        elif rel not in tracked:
            print(f"FAIL {rm}: image {ref!r} exists but is NOT in git - it would be "
                  f"a broken image for everyone who clones it (git add it)")
            ok = False

for f in sorted(p for p in tracked if p.startswith("samples/") and pathlib.Path(p).suffix.lower() in IMG):
    if f not in referenced:
        print(f"FAIL {f}: committed but no sample README references it - a generated "
              f"chart belongs in an ignored output directory, not in git")
        ok = False

sys.exit(0 if ok else 1)
PY

# ------------------------------------------------- surfaces, closure, budget
# One python pass: read the SDK's feature table, then check every sample's
# declaration and real dependency graph against it.
python3 - "${manifests[@]}" <<'PY' || fail=1
import subprocess, sys, tomllib

sdk = tomllib.load(open("crates/sdk/Cargo.toml", "rb"))
feats = sdk.get("features", {})

# Infrastructure tiers are selected BY a surface and must never be named by a
# sample; `full` is the "everything" alias and is equally not a surface.
TIERS = {"device", "resolve", "full", "default"}
surfaces = {f: v for f, v in feats.items() if f not in TIERS}

def crates_of(feature, seen=None):
    """Every `dep:` crate a feature pulls in, transitively through the table."""
    seen = seen if seen is not None else set()
    out = set()
    for entry in feats.get(feature, []):
        if entry.startswith("dep:"):
            out.add(entry[4:])
        elif "/" in entry:
            out.add(entry.split("/", 1)[0])
        elif entry not in seen:
            seen.add(entry)
            out |= crates_of(entry, seen)
    return out

surface_crates = {s: crates_of(s) for s in surfaces}
ok = True

for m in sys.argv[1:]:
    d = m.rsplit("/", 1)[0]
    pkg = "sample-" + d[len("samples/"):].replace("/", "-")
    man = tomllib.load(open(m, "rb"))
    dep = man.get("dependencies", {}).get("brain")

    if not isinstance(dep, dict) or not dep.get("features"):
        print(f"FAIL {m}: must NAME the SDK surfaces it uses, e.g. "
              f'brain = {{ workspace = true, features = ["image"] }} '
              f"(samples/README.md rule 2)")
        ok = False
        continue

    named = set(dep["features"])
    for f in sorted(named):
        if f in TIERS:
            print(f"FAIL {m}: names '{f}', an infrastructure tier rather than a "
                  f"surface. Name a surface; a surface selects its own tiers.")
            ok = False
        elif f not in surfaces:
            print(f"FAIL {m}: names '{f}', which is not an SDK surface. "
                  f"Known surfaces: {', '.join(sorted(surfaces))}")
            ok = False

    tree = subprocess.run(
        ["cargo", "tree", "-p", pkg, "--edges", "normal", "--prefix", "none"],
        capture_output=True, text=True)
    if tree.returncode != 0:
        print(f"FAIL {pkg}: cargo tree failed:\n{tree.stderr.strip()[:400]}")
        ok = False
        continue
    present = {ln.split()[0] for ln in tree.stdout.splitlines()
               if ln.strip().startswith("brain")}

    # THE closure check: nothing REACHABLE ONLY through a surface this sample
    # did not enable.
    #
    # "Reachable only" is the whole subtlety, and getting it wrong produces
    # false positives that are indistinguishable from real leaks. Two
    # subtractions are needed, for two different reasons.
    #
    # First, surfaces share infrastructure tiers (`image` and a future `text`
    # both select `resolve`), so a surface's raw crate set is not exclusively
    # its own - without this, enabling `image` would be reported as leaking
    # `resolve`'s crates "from the text surface".
    #
    # Second, and this is what the `creature` surface exposed: a surface's
    # `dep:` crates have dependencies of their own, and those can be crates
    # another surface names directly. `brain-wm-display` depends on
    # `brain-imaging`, which depends on `brain-vision`, which depends on
    # `brain-model` - both of which the `image` surface names. Reading the
    # manifest alone, those look like image-surface crates that leaked into a
    # creature-only sample. They are nothing of the sort; they are the
    # transitive closure of a dependency the sample legitimately asked for.
    # So the enabled set is expanded through its REAL graph before subtracting.
    allowed = set()
    for f in named:
        allowed |= crates_of(f)
    for crate in sorted(allowed):
        t = subprocess.run(
            ["cargo", "tree", "-p", crate, "--edges", "normal", "--prefix", "none"],
            capture_output=True, text=True)
        if t.returncode == 0:
            allowed |= {ln.split()[0] for ln in t.stdout.splitlines()
                        if ln.strip().startswith("brain")}
    for s in sorted(set(surfaces) - named):
        leaked = sorted((surface_crates[s] - allowed) & present)
        if leaked:
            print(f"FAIL {pkg}: links {', '.join(leaked)}, reachable only through the "
                  f"'{s}' surface, which it did not enable")
            ok = False

    budget = man.get("package", {}).get("metadata", {}).get("brain", {}).get("max-brain-crates")
    if budget is None:
        print(f"FAIL {m}: no [package.metadata.brain] max-brain-crates budget "
              f"(samples/README.md rule 8)")
        ok = False
    elif len(present) > budget:
        print(f"FAIL {pkg}: links {len(present)} brain crates, budget is {budget}. "
              f"Raise the budget in the same commit as the dependency, or drop it.")
        ok = False
    else:
        print(f"  {pkg}: {len(present)} brain crates "
              f"(budget {budget}, surfaces: {', '.join(sorted(named))})")

sys.exit(0 if ok else 1)
PY

# --------------------------------------------------------------- incremental
if [ "${BRAIN_SAMPLES_CHECK_BUILD:-1}" = "0" ]; then
	note "build measurement skipped (BRAIN_SAMPLES_CHECK_BUILD=0)"
elif ! command -v cargo >/dev/null 2>&1; then
	note "build measurement skipped (no cargo)"
else
	for m in "${manifests[@]}"; do
		rel="$(dirname "$m")"; rel="${rel#samples/}"
		pkg="sample-${rel//\//-}"
		note "building $pkg"
		# Narrow selection on purpose: a selection that also contained the
		# engine's default members would union this sample's declared features
		# with the SDK's own defaults and silently ignore the declaration.
		cargo build --release -p "$pkg" >/dev/null 2>&1 || bad "$pkg: does not build"
		cargo clippy --release -p "$pkg" --all-targets --message-format=short \
			>/dev/null 2>&1 || bad "$pkg: clippy failed"
	done

	# Touch the last sample's own sources so it MUST relink - proving the
	# measurement is live rather than observing an already-current tree.
	last="$(dirname "${manifests[-1]}")"
	rel="${last#samples/}"; pkg="sample-${rel//\//-}"
	find "$last/src" -name '*.rs' -exec touch {} +
	out="$(cargo build --release -p "$pkg" --message-format short 2>&1 || true)"
	brain_units="$(printf '%s\n' "$out" | grep -c '^ *Compiling brain' || true)"
	all_units="$(printf '%s\n' "$out" | grep -c '^ *Compiling' || true)"
	if [ "$brain_units" -ne 0 ]; then
		bad "rebuilding $pkg recompiled $brain_units brain crate(s) - it is not being built with its own narrow selection"
	elif [ "$all_units" -ne 1 ]; then
		bad "rebuilding $pkg compiled $all_units crates, expected exactly 1 (its own)"
	else
		note "rebuild of $pkg compiled 1 crate, 0 of them brain crates"
	fi
fi

[ "$fail" -eq 0 ] && echo "check/samples: OK" || echo "check/samples: FAILED"
exit "$fail"
