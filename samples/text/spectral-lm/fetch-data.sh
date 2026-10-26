#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Fetch the general-text and question-answering data the slicing study uses,
# from the datasets' original distributions. Writes under data/ (gitignored):
#
#   data/wikitext/calib.txt   the first CALIB_BYTES of wikitext-103-raw train
#   data/wikitext/test.txt    the wikitext-103-raw test split
#   data/arc-easy/test.jsonl  ARC-Easy test: {"question", "choices": [...], "answer": index}
#
# Usage: samples/text/spectral-lm/fetch-data.sh [calibration bytes, default 4000000]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
CALIB_BYTES="${1:-4000000}"

mkdir -p data/wikitext data/arc-easy
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -sSfL --retry 4 -o "$tmp/wikitext.zip" https://wikitext.smerity.com/wikitext-103-raw-v1.zip
unzip -p "$tmp/wikitext.zip" wikitext-103-raw/wiki.test.raw > data/wikitext/test.txt
# `head` closes the pipe early, so unzip ends on SIGPIPE: not an error here.
(set +o pipefail; unzip -p "$tmp/wikitext.zip" wikitext-103-raw/wiki.train.raw | head -c "$CALIB_BYTES" > data/wikitext/calib.txt)

curl -sSfL --retry 4 -o "$tmp/arc.zip" https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
unzip -p "$tmp/arc.zip" ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl | python3 -I -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    choices = r["question"]["choices"]
    labels = [c["label"] for c in choices]
    print(json.dumps({"question": r["question"]["stem"], "choices": [c["text"] for c in choices], "answer": labels.index(r["answerKey"])}))
' > data/arc-easy/test.jsonl

wc -c data/wikitext/calib.txt data/wikitext/test.txt
wc -l data/arc-easy/test.jsonl
