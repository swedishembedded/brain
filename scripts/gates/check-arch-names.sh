#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Architecture-naming consistency gate (`make check/scripts`).
#
# crates/arch is the ONE canonical registry of architecture ids (see that
# crate's own module doc for the naming rule and why it exists -- four
# separate, drifting answers to "which architecture is this" before it). This
# gate is what keeps the things that must agree with it from silently
# drifting again:
#
#   1. every non-toy `brain_arch::ARCHS` row's `package` field names a crate
#      directory that is a real workspace member, and that crate's own
#      directory name equals the row's `id` -- the naming rule's own claim
#      ("id is simultaneously ... the crate directory name") is otherwise
#      just prose nobody checks.
#   2. `crates/cli/src/main.rs`'s top-level dispatch `match` never hard-codes
#      an architecture name as a literal `Some("...")` arm -- every
#      architecture-specific command goes through `crate::resolve`'s
#      registry-driven dispatch (see that module's doc), so a literal arch
#      string in main.rs is exactly the "one arm per model" shape this
#      workspace moved away from.
#   3. every non-toy registry id has a `docs/models/<id>.md` page, and every
#      `docs/models/*.md` basename is a real registry id -- both directions,
#      so a renamed architecture can't leave a stale orphan page behind
#      (checked one direction) or a served architecture with no docs at all
#      (the other).
#
# NOT yet checked here (tracked as follow-up, not silently skipped -- see
# .agents/roadmap or the memory this session recorded): every ModelCard
# carrying a required `architecture` field naming a registry id. That
# requires checkpoint::st::ModelCard's `family: String` field to actually be
# retired in favor of the already-added `architecture: Option<String>`
# becoming required first (~30 call sites across the model crates); a
# textual gate over an unused field would be theater, not a real check.
#
# Usage: scripts/gates/check-arch-names.sh   (exits non-zero listing every
# violation found, not just the first)
set -u
cd "$(dirname "$0")/../.."

ARCH_TABLE=crates/arch/src/lib.rs
MAIN_RS=crates/cli/src/main.rs
MODELS_DIR=docs/models
fail=0

# ---- 1. package/directory/id agreement -----------------------------------

# One "id package" pair per non-blank `arch!(...)` line. Toy rows are
# excluded on purpose: `Domain::Toy` architectures are real crates (already
# id == dir, checked the same way) but this section's job is specifically
# the "public, servable" naming contract -- toy rows are covered by the same
# loop below since nothing distinguishes them at this grep level, which is
# fine: the invariant (dir == id, package is a real member) holds for them
# too, it's just not interesting to call out separately here.
rows=$(grep -oE 'arch!\("[a-z0-9]+", "[^"]*", [A-Za-z:]+, [A-Za-z:]+, "brain-[a-z0-9]+"' "$ARCH_TABLE" \
      | sed -E 's/^arch!\("([a-z0-9]+)".*"brain-([a-z0-9]+)"$/\1 \2/')

if [ -z "$rows" ]; then
  echo "check-arch-names: matched zero rows in $ARCH_TABLE -- the extraction regex and the table's format have drifted, fix the regex"
  fail=1
fi

# `autoencoderkl` is a deliberate, PERMANENT exception, not a pending rename
# like scrfd/arcface (still bundled in brain-facenet until that crate splits):
# crates/vae is genuinely shared infrastructure AND the AutoencoderKL
# architecture's home at once (vqgan/rrdbnet/sdxlunet all consume its conv
# blocks), and splitting it would duplicate that shared code instead of
# naming it -- see crates/arch/src/lib.rs's own "Deliberately not
# architectures" note. Extend this list only for an equally-deliberate case,
# never to silence a rename that just hasn't happened yet.
permanent_exceptions="autoencoderkl:vae"

# TEMPORARY, TRACKED exceptions -- unlike the permanent one above, each of
# these is real drift, not a deliberate design choice: scrfd and arcface are
# two distinct upstream architectures still bundled in one brain-facenet
# crate pending the split into crates/scrfd + crates/arcface (tracked
# separately; see the facenet-split follow-up). Remove a row here in the
# SAME commit that lands its rename -- an exception nobody ever removes is
# indistinguishable from a permanent one, which defeats the point of this
# gate existing at all.
temporary_exceptions="scrfd:facenet arcface:facenet"

while read -r id pkg_suffix; do
  [ -z "$id" ] && continue
  case " $permanent_exceptions $temporary_exceptions " in
  *" $id:$pkg_suffix "*) continue ;;
  esac
  if [ "$id" != "$pkg_suffix" ]; then
    echo "PACKAGE MISMATCH: $ARCH_TABLE arch $id declares package brain-$pkg_suffix, but the naming rule requires the package suffix to equal the id"
    fail=1
  fi
  if [ ! -d "crates/$id" ]; then
    echo "MISSING CRATE: $ARCH_TABLE arch $id has no crates/$id directory (package brain-$pkg_suffix)"
    fail=1
  fi
done <<EOF
$rows
EOF

# ---- 2. no literal architecture command in main.rs's dispatch -------------

# The fixed infra-verb allowlist main.rs's top-level match is permitted to
# name literally -- everything else routes through crate::resolve instead.
# Keep this list in sync with that match arm-for-arm; a new infra verb needs
# a line here, a new architecture must NOT need one.
infra_verbs="data devices npu federated flops gradcheck bench perf forecast caps serve help -h --help"

# Bounded to fn main's own top-level `match argv.get(1)...` block, not the
# whole file: nested match blocks further down (`run_bench`'s own `eval`/
# `compare`/`scale`/`scaling`/`advise` sub-dispatch, `--arch` value parsing,
# etc.) legitimately reuse short literal words that are not architecture
# names, and are not what this check is about.
dispatch_block=$(awk '/match argv\.get\(1\)/{p=1} p{print} p && /^    }$/{exit}' "$MAIN_RS")
literals=$(printf '%s\n' "$dispatch_block" | grep -oE 'Some\("[a-z][a-z0-9-]*"\)' | grep -oE '"[a-z][a-z0-9-]*"' | tr -d '"' | sort -u)
for word in $literals; do
  case " $infra_verbs " in
  *" $word "*) continue ;;
  esac
  echo "LITERAL ARCH COMMAND: $MAIN_RS hard-codes Some(\"$word\") in its top-level dispatch -- architecture commands must route through crate::resolve, not a new match arm"
  fail=1
done

# ---- 3. docs/models/<id>.md, both directions -------------------------------

non_toy_ids=$(grep -oE 'arch!\("[a-z0-9]+", "[^"]*", (Text|Multimodal|Audio|Vision|Image|ThreeD|Forecast|World)' "$ARCH_TABLE" \
             | sed -E 's/^arch!\("([a-z0-9]+)".*/\1/')

for id in $non_toy_ids; do
  if [ ! -f "$MODELS_DIR/$id.md" ]; then
    echo "MISSING DOCS PAGE: architecture '$id' has no $MODELS_DIR/$id.md"
    fail=1
  fi
done

for f in "$MODELS_DIR"/*.md; do
  [ -e "$f" ] || continue
  base=$(basename "$f" .md)
  case " $non_toy_ids " in
  *" $base "*) continue ;;
  esac
  echo "ORPHAN DOCS PAGE: $f's basename '$base' is not a registered non-toy architecture id (see $ARCH_TABLE)"
  fail=1
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "check-arch-names: architecture-naming consistency violated (see above). Fix crates/arch/src/lib.rs, crates/cli/src/main.rs, or docs/models/ to agree."
  exit 1
fi
echo "check-arch-names: OK"
