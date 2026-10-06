#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements stateless record-to-risk services whose
# answers say what they were computed from, for its clients. If your team needs
# expertise in turning longitudinal records into auditable risk forecasts you
# can procure our services by sending an email to info@swedishembedded.com.

# "Append a checkup and re-predict" for one synthetic patient: the patient's
# history H0 gives forecast R0; the same history with a later checkup appended
# gives R1; with another appended, R2. Every call sends the WHOLE history; the
# model's weights never change for the patient and nothing is stored.
#
#   samples/shell/timeline/predict/history.sh <model-visits dir> <subjects-visits.jsonl> [times]
#
# The arguments are what samples/study/timeline writes for its visit model, one
# trained on records with several visits, as a history with appended checkups
# is. The forecast is a statistical estimate for the population the model was
# trained on: not a diagnosis, not a treatment recommendation.
set -euo pipefail

model="${1:?usage: history.sh <model-visits dir> <subjects-visits.jsonl> [times]}"
subjects="${2:?usage: history.sh <model-visits dir> <subjects-visits.jsonl> [times]}"
times="${3:-5,10}"
work="${HISTORY_SAMPLE_DIR:-$(mktemp -d)}"

# H0, H1, H2: the first subject with two visits, as patient histories. A
# checkup is a new reading of x one and a half years after the last, a little
# higher than the one before; the clock (age in years) advances with it. This
# synthetic population has no calendar trend, so every history states the
# calendar year its training records do.
python3 - "$subjects" "$work" <<'PY'
import json, sys
subjects, work = sys.argv[1], sys.argv[2]
for line in open(subjects):
    s = json.loads(line)
    if len(s["observations"]) == 2 and s["entry"] < 60:
        break
else:
    sys.exit("no subject with two visits in " + subjects)
records = [{"time": o["t"], "code": o["var"], "value": o["value"]} for o in s["observations"]]
last = records[-1]["value"]
for k in range(3):
    history = {
        "id": s["subject_id"],
        "as_of": s["entry"] + 1.5 * k,
        "calendar": s["calendar_at_entry"],
        "events": records + [
            {"time": s["entry"] + 1.5 * j, "code": "x", "value": last + 0.4 * j}
            for j in range(1, k + 1)
        ],
    }
    json.dump(history, open(f"{work}/H{k}.json", "w"), indent=1)
PY

for k in 0 1 2; do
    brain horizon predict --weights "$model" --times "$times" \
        --history "$work/H$k.json" --json > "$work/R$k.json" 2> "$work/err" \
        || { cat "$work/err" >&2; exit 1; }
done

python3 - "$work" <<'PY'
import json, sys
work = sys.argv[1]
print("history -> forecast (every call sends the whole history; nothing is stored)\n")
for k in range(3):
    h = json.load(open(f"{work}/H{k}.json"))
    f = json.load(open(f"{work}/R{k}.json"))["forecasts"][0]
    x = [round(e["value"], 2) for e in h["events"] if e["code"] == "x"]
    print(f"H{k}: as_of {h['as_of']:.1f}, x readings {x}")
    if f["risk"] != "available":
        print(f"R{k}: risk {f['risk']} ({f['reason']})\n")
        continue
    cov = f["coverage"]
    print(f"R{k}: {cov['observations']} observations, {cov['events']} events, "
          f"newest reading {cov['newest_observation']['ago']:.1f} years before as_of, "
          f"support ood_score {f['support']['ood_score']:.2f}")
    for hz in f["horizons"]:
        risks = "  ".join(
            f"{code} {r['raw']:.3f}" + (f" (calibrated {r['calibrated']:.3f})" if "calibrated" in r else "")
            for code, r in sorted(hz["risks"].items())
        )
        print(f"    by {hz['horizon']:g}: {risks}")
    print()
print(f["disclaimer"])
PY
