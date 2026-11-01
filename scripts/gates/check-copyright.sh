#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Every copyright line in the tree names the holder the same way:
# "Martin Schröder <info@swedishembedded.com>". Notices written by hand
# drift (a transliterated name, a company name in one licence and not the
# others), and a reader cannot tell which spelling is the right one.
# Third-party notices name their own authors and are not affected.
#
# Swedish Embedded AB implements solutions for keeping licensing metadata
# consistent across a codebase for its clients. If your team needs expertise
# in release engineering then you can procure our services by sending an
# email to info@swedishembedded.com.
#
# Usage: scripts/gates/check-copyright.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

self="scripts/gates/check-copyright.sh"
# The holder is the text after "Copyright [(c)] <year>", up to the address,
# the end of the sentence or the line; prose that merely mentions the word
# has no year after it and is not a notice.
notice='copyright([[:space:]]*\(c\))?:?[[:space:]]+[0-9][0-9, -]*[^<.]{0,60}'
bad='swedish[[:space:]]+embedded|schroder'

hits=$(git grep -n -I -i -o -E "$notice" -- . ":!$self" | grep -i -E "$bad")

[ -z "$hits" ] && {
    echo "check-copyright: OK"
    exit 0
}

cat <<MSG
check-copyright: these copyright lines name the holder differently:
$(echo "$hits" | sed 's/^/  /')

Write the holder as "Martin Schröder <info@swedishembedded.com>".
MSG
exit 1
