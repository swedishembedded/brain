#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Gate for the `brain` SDK's feature table - the workspace's feature
# vocabulary. Invoked by `make check/sdk-features`.
#
# The SDK is where a consumer (a sample, a product) names what it needs, so the
# vocabulary has to stay small, predictable and honest:
#
#   1. Surface names come from `brain_arch::Domain`, so the SDK and the arch
#      registry do not grow two different words for the same modality.
#   2. `default` is exactly `full`, so `cargo add brain` works with no
#      configuration AND `cargo test -p brain` keeps exercising every surface.
#   3. `crates/catalog` has NO features - see the catalog check below for why.
#   4. No value-shaped feature names. Features are booleans that union across a
#      build; encoding a value in one multiplies the build cache by the value
#      space and still cannot stop a second consumer enabling a different value.
#   5. Every surface compiles ON ITS OWN, and so does the bare core. This is
#      the `allnoconfig`/`allyesconfig` sweep: a feature nobody builds alone is
#      a feature that silently stops compiling.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CARGO_HOME="${CARGO_HOME_OVERRIDE:-$HOME/.cargo}"

fail=0

python3 <<'PY' || fail=1
import re, sys, tomllib

ok = True
def bad(msg):
    global ok
    print(f"FAIL {msg}")
    ok = False

sdk = tomllib.load(open("crates/sdk/Cargo.toml", "rb"))
feats = sdk.get("features", {})
if not feats:
    bad("crates/sdk/Cargo.toml has no [features] table")
    sys.exit(1)

TIERS = {"device", "resolve", "imagetype", "full", "default"}
surfaces = sorted(set(feats) - TIERS)

# 1. surface names are brain_arch::Domain variants, kebab-cased.
arch = open("crates/arch/src/lib.rs").read()
m = re.search(r"pub enum Domain\s*\{(.*?)\n\}", arch, re.S)
if not m:
    bad("could not find `pub enum Domain` in crates/arch/src/lib.rs")
    domains = set()
else:
    variants = re.findall(r"^\s*([A-Z][A-Za-z0-9]*)\s*,", m.group(1), re.M)
    def kebab(v):
        return re.sub(r"(?<!^)(?=[A-Z])", "-", v).lower()
    domains = {kebab(v) for v in variants}
    print(f"  Domain vocabulary: {', '.join(sorted(domains))}")

for s in surfaces:
    if domains and s not in domains:
        bad(f"crates/sdk: surface '{s}' is not a brain_arch::Domain variant. "
            f"Add the Domain first, or name the surface after an existing one.")

# 2. default == full, exactly.
if feats.get("default") != ["full"]:
    bad(f"crates/sdk: `default` is {feats.get('default')!r}, must be exactly "
        f'["full"] - otherwise `cargo test -p brain` stops exercising a surface')

# 4. no value-shaped feature names.
for f in feats:
    if re.fullmatch(r"[a-z][a-z-]*-[0-9]+", f):
        bad(f"crates/sdk: feature '{f}' encodes a VALUE. Features are booleans "
            f"that union across a build; values are builder arguments.")

# 3. catalog stays un-featurized.
cat = tomllib.load(open("crates/catalog/Cargo.toml", "rb"))
if cat.get("features"):
    bad("crates/catalog has a [features] table. It must not: `catalog::models()` "
        "is THE list of what brain can serve, and its own module doc records the "
        "ai-forever/Real-ESRGAN bug - a model with manifest, provider, adapter "
        "and passing tests that was still unreachable because one of three lists "
        "went unedited. Features turn that into an N-feature-set drift and make "
        "`every_listed_model_is_constructible_by_name` vacuous, since it would "
        "iterate a short list and pass trivially. A narrow consumer constructs "
        "the one model crate's Provider directly; the generic dispatch machinery "
        "is in brain-capability, not here.")

print(f"  surfaces: {', '.join(surfaces) or '(none)'}")
sys.exit(0 if ok else 1)
PY

# 5. the allnoconfig/per-surface sweep.
if [ "${BRAIN_SDK_FEATURES_CHECK_BUILD:-1}" = "0" ]; then
	echo "  compile sweep skipped (BRAIN_SDK_FEATURES_CHECK_BUILD=0)"
elif ! command -v cargo >/dev/null 2>&1; then
	echo "  compile sweep skipped (no cargo)"
else
	mapfile -t surfaces < <(python3 -c "
import tomllib
f = tomllib.load(open('crates/sdk/Cargo.toml','rb'))['features']
print('\n'.join(sorted(set(f) - {'device','resolve','imagetype','full','default'})))
")
	printf '  compiling bare core ... '
	if cargo check -p brain --no-default-features --message-format short >/dev/null 2>&1; then
		echo ok
	else
		echo FAIL; fail=1
	fi
	for s in "${surfaces[@]}"; do
		printf '  compiling surface %s ... ' "$s"
		if cargo check -p brain --no-default-features --features "$s" \
			--message-format short >/dev/null 2>&1; then
			echo ok
		else
			echo FAIL; fail=1
		fi
	done
fi

[ "$fail" -eq 0 ] && echo "check/sdk-features: OK" || echo "check/sdk-features: FAILED"
exit "$fail"
