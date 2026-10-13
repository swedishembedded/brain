#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for discoverable, self-documenting
# build interfaces for its clients. If your team needs expertise in build
# system design and repository quality gates then you can procure our services
# by sending an email to info@swedishembedded.com.
#
# The Makefile's public targets form a tree that `make <TAB>` reveals one
# level at a time: `test`, then `test/`, then `test/harness`. That only works
# while the names keep to a convention, so this gate enforces it.
#
# THE RULE (also in AGENTS.md, "Make targets"): a public target is
# `make <concept>/<subconcept>/.../<action>`. The concept comes first, the verb
# or variant last, and a directory hierarchy (tests, samples) gets one segment
# per directory level. Every prefix is itself a target, so tab descent works
# at every level.
#
# Machine-checked here:
#
#   1. A name is lowercase `[a-z0-9-]` segments joined by `/`.
#   2. Every proper prefix of a name is itself a defined target.
#   3. Every name has a `## name - description` doc line. In a doc name `<x>`
#      stands for one path segment, so one line documents a generated family
#      (`gpt/<dataset>/train`); a family line that matches no target is an
#      error, and so is a doc line for a target that does not exist.
#   4. Every doc line sits under a `##@ Section title` line, so `make help`
#      can place it.
#   5. A single-segment name with no children is a leaf, and only the
#      LEAVES below may be one: a concept that has sub-actions is an area with
#      children. `fmt` stays a leaf because it has exactly one action (its
#      counterpart `check/fmt` checks without applying); `clean`, `help` and
#      `all` are conventional.
#
# Not machine-checkable and left to review: that a hierarchy's directory
# levels each became a segment, and that the verb really is last.
#
# "Public" means a static .PHONY target whose name starts with neither `_` nor
# `.`. Pattern rules (`%`) and file targets are not checked. The list comes from
# scripts/make-help.sh, which reads `make -pRrq`, so generated targets count.
#
# Usage: scripts/gates/check-make-targets.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export LC_ALL=C

LEAVES=" all clean fmt help "

public=$(bash scripts/make-help.sh --targets)
defined=$(bash scripts/make-help.sh --defined)
docs=$(bash scripts/make-help.sh --docs)

failures=0
fail() {
    printf 'check-make-targets: %s: %s\n' "$1" "$2" >&2
    failures=$((failures + 1))
}

# A doc name as an anchored regex: <x> is one path segment. Done in-shell and
# once per doc line: the generated families make the target list long, so a
# fork per (target, doc) pair would take minutes.
shape() {
    local out=$1
    while [[ $out =~ (<[^>]+>) ]]; do
        out=${out/"${BASH_REMATCH[1]}"/"[^/]+"}
    done
    printf '%s' "$out"
}

declare -A is_defined=() is_public=() has_children=()
while IFS= read -r t; do [ -n "$t" ] && is_defined[$t]=1; done <<<"$defined"
while IFS= read -r t; do
    [ -n "$t" ] || continue
    is_public[$t]=1
    # A bare name with public descendants is an area, not a leaf.
    [[ $t == */* ]] && has_children[${t%%/*}]=1
done <<<"$public"

doc_names=() doc_sections=() doc_regexes=()
while IFS=$'\t' read -r doc section; do
    [ -n "$doc" ] || continue
    doc_names+=("$doc")
    doc_sections+=("$section")
    doc_regexes+=("^$(shape "$doc")\$")
done <<<"$docs"

while IFS= read -r name; do
    [ -n "$name" ] || continue

    if ! [[ $name =~ ^[a-z0-9-]+(/[a-z0-9-]+)*$ ]]; then
        fail "$name" "not lowercase [a-z0-9-] segments separated by '/'"
    fi

    prefix=$name
    while [[ $prefix == */* ]]; do
        prefix=${prefix%/*}
        [ -n "${is_defined[$prefix]:-}" ] ||
            fail "$name" "prefix '$prefix' is not a defined target (tab completion cannot descend)"
    done

    if [[ $name != */* && $LEAVES != *" $name "* && -z "${has_children[$name]:-}" ]]; then
        fail "$name" "single-segment leaf; name it <concept>/<action> (only$LEAVES may be a bare leaf)"
    fi

    documented=0
    for re in "${doc_regexes[@]}"; do
        if [[ $name =~ $re ]]; then
            documented=1
            break
        fi
    done
    [ "$documented" = 1 ] || fail "$name" "no '## $name - description' (or matching <placeholder>) line in Makefile"
done <<<"$public"

for i in "${!doc_names[@]}"; do
    doc=${doc_names[$i]}
    [ -n "${doc_sections[$i]}" ] || fail "$doc" "doc line is not under a '##@ Section' line"
    if [[ $doc == *"<"* ]]; then
        matched=0
        for t in "${!is_public[@]}"; do
            if [[ $t =~ ${doc_regexes[$i]} ]]; then matched=1; break; fi
        done
        [ "$matched" = 1 ] || fail "$doc" "family doc line matches no target"
    else
        [ -n "${is_defined[$doc]:-}" ] || fail "$doc" "documented in Makefile but no such target"
    fi
done

if [ "$failures" -ne 0 ]; then
    printf 'check-make-targets: %d violation(s); the rule is at the top of this script\n' "$failures" >&2
    exit 1
fi
printf 'check-make-targets: %d public targets, all named, reachable, documented and in a section\n' \
    "${#is_public[@]}"
