#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the whole path from a client's records to a
# checked, calibrated risk model that says how sure it is, for its clients. If
# your team needs expertise in training, validating and serving time-to-event
# models on your own data you can procure our services by sending an email to
# info@swedishembedded.com.

# train -> eval -> calibrate -> predict, all through `brain horizon`:
#
#   samples/shell/timeline/lifecycle/lifecycle.sh <data dir> [work dir]
#
# <data dir> holds the four timeline-v1 files samples/study/timeline writes to
# <out>/data: train.jsonl, held-out.jsonl, validation.jsonl, test.jsonl.
# A model is trained (early-stopping on held-out subjects), judged on test
# subjects it never saw, calibrated on validation subjects it never saw, and
# asked for the forecast of one patient history. A bootstrap ensemble of three
# is trained the same way and answers the same history with the spread of its
# members. A forecast is a statistical estimate for the population the model
# was trained on: not a diagnosis, not a treatment recommendation.
set -euo pipefail

data="${1:?usage: lifecycle.sh <data dir> [work dir]}"
work="${2:-$(mktemp -d)}"
horizons="5,10"
absorbing="death:a,death:b"
knots="0,1,2,3,4,5,6,8,10,12,15"
mkdir -p "$work"
rm -rf "$work/model" "$work/ensemble"

# The trainer logs its evaluations to stdout; --json's object is the last line.
echo "== train (early stopping on held-out subjects; the directory appears only when training completes)"
brain horizon train --dataset "$data/train.jsonl" --held-out "$data/held-out.jsonl" \
    --out "$work/model" --absorbing "$absorbing" --knots "$knots" --steps 600 --batch 128 \
    --eval-interval 100 --json | tail -n 1 > "$work/train.json"
python3 - "$work/train.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
m = r["members"][0]
print(f"trained on {r['train_subjects']} subjects in {m['steps']} steps; held-out event NLL {m['held_out_event_nll']:.4f}")
PY

echo
echo "== eval on test subjects the model never saw"
brain horizon eval --weights "$work/model" --dataset "$data/test.jsonl" --horizons "$horizons" --json > "$work/eval.json"
python3 - "$work/eval.json" <<'PY'
import json, sys
e = json.load(open(sys.argv[1]))
print(f"{e['subjects']} subjects, event NLL {e['event_nll']:.4f}")
for r in e["results"]:
    c = r["calibration"]
    print(f"  {r['code']:>8} by {r['horizon']:g}: {r['events']:4d} events  Uno C {r['uno_c']:.3f}  AUC {r['auc']:.3f}  "
          f"Brier {r['brier']:.4f}  integrated {r['integrated_brier']:.4f}  slope {c['slope']:.2f}  O/E {c['observed_over_expected']:.2f}")
for a in e["absent"]:
    print(f"  {a['code']:>8} by {a['horizon']:g}: absent ({a['events']} events, needs {a['min_events']})")
PY

echo
echo "== calibrate on validation subjects the model never saw"
brain horizon calibrate --weights "$work/model" --validation "$data/validation.jsonl" --horizons "$horizons" --json > "$work/calibrate.json"
python3 - "$work/calibrate.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"{c['calibrated']} (code, horizon) pairs calibrated, written to calibration.json beside the weights")
for g in c["uncalibrated"]:
    print(f"  not calibrated: {g}")
PY

echo
echo "== train a bootstrap ensemble of three"
brain horizon train --dataset "$data/train.jsonl" --held-out "$data/held-out.jsonl" \
    --out "$work/ensemble" --absorbing "$absorbing" --knots "$knots" --steps 600 --batch 128 \
    --members 3 --ensemble bootstrap --json | tail -n 1 > "$work/ensemble.json"

# One patient: the first test subject's record as a history.
python3 - "$data/test.jsonl" "$work/history.json" <<'PY'
import json, sys
s = json.loads(open(sys.argv[1]).readline())
records = [{"time": o["t"], "code": o["var"], "value": o["value"]} for o in s["observations"] if o["t"] <= s["entry"]]
records += [{"time": e["t"], "code": e["code"]} for e in s["events"] if e["t"] < s["entry"]]
json.dump({"id": s["subject_id"], "as_of": s["entry"], "calendar": s["calendar_at_entry"], "events": records},
          open(sys.argv[2], "w"), indent=1)
PY

for kind in model ensemble; do
    brain horizon predict --weights "$work/$kind" --times "$horizons" --history "$work/history.json" --json \
        > "$work/forecast-$kind.json"
done

echo
echo "== predict: one patient history, the single model and the ensemble"
python3 - "$work" <<'PY'
import json, sys
work = sys.argv[1]
for kind in ("model", "ensemble"):
    f = json.load(open(f"{work}/forecast-{kind}.json"))["forecasts"][0]
    print(f"{kind}: {f['subject_id']}, support ood_score {f['support']['ood_score']:.2f}")
    if f["risk"] != "available":
        print(f"  risk {f['risk']} ({f['reason']})")
        continue
    for hz in f["horizons"]:
        for code, r in sorted(hz["risks"].items()):
            line = f"  by {hz['horizon']:g} {code:>8}: {r['raw']:.3f}"
            if "calibrated" in r:
                line += f"  calibrated {r['calibrated']:.3f} [{r['interval'][0]:.3f}, {r['interval'][1]:.3f}]"
            if "member_range" in r:
                line += f"  members {r['member_range'][0]:.3f} to {r['member_range'][1]:.3f}"
            print(line)
print()
print(f["disclaimer"])
PY
