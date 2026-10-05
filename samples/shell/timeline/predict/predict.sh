#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Serve a saved timeline model from the command line: each subject's
# probability of surviving, and of each outcome, at the times asked for.
#
#   samples/shell/timeline/predict/predict.sh <model dir> <subjects.jsonl> [times]
#
# <model dir> is what TimelineModel::save writes (samples/study/timeline
# writes one, with a subjects.jsonl beside it); times default to 5,10.
set -euo pipefail

model="${1:?usage: predict.sh <model dir> <subjects.jsonl> [times]}"
subjects="${2:?usage: predict.sh <model dir> <subjects.jsonl> [times]}"
times="${3:-5,10}"
out="$(mktemp -d)/predictions.jsonl"

brain horizon predict --weights "$model" --times "$times" \
    --in subjects="$subjects" --out predictions="$out"

python3 - "$out" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    p = json.loads(line)
    risks = "  ".join(
        f"{code} {', '.join(f'{x:.3f}' for x in v)}" for code, v in sorted(p["cif"].items())
    )
    print(f"{p['subject_id']:>10}  survival {', '.join(f'{x:.3f}' for x in p['survival'])}  {risks}")
PY
