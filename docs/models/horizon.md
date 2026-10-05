# Horizon

Horizon is brain's own continuous-time subject-timeline model. It reads the
irregular history of one subject - measurements with values, events, each at
a real-valued time - and answers, for any horizon inside its knots, the
probability of each outcome by then. Outcomes may compete (a cause of death
ends follow-up for every other outcome), follow-up may start late and end
early, and a value known only to lie below a detection limit is still used.
Nothing in the model knows the domain: variables and outcome codes are data.

## Support

| Capability | Supported |
|---|---|
| Training from scratch | [x] |
| Inference | [x] (SDK) |
| LoRA fine-tune | [ ] |
| CLI (`brain <arch> <action>`) | [ ] |
| HTTP API | [ ] |
| D-Bus | [ ] |

## Data: `timeline-v1`

One JSON object per line:

```json
{"subject_id": "a", "group_id": "household-1", "weight": 2.5, "source": "survey",
 "entry": 50.0, "calendar_at_entry": 2003.5,
 "observations": [{"t": 50.0, "var": "sbp", "value": 131},
                  {"t": 50.0, "var": "crp", "value": {"below": 0.2}},
                  {"t": 50.0, "var": "smoking", "value": "never"},
                  {"t": 25.0, "var": "weight", "value": 70}],
 "events": [{"t": 44.0, "code": "dx:hypertension"}, {"t": 62.5, "code": "death:heart"}],
 "at_risk": [{"code": "*", "from": 50.0, "to": 62.5}]}
```

- Times are in ONE unit for the whole dataset (years for the health
  application), on the subject's own clock; `entry` is the prediction time
  and `calendar_at_entry` places that clock on the calendar.
- Only what is known at `entry` is input: observations at or before it,
  events strictly before it. Events after it are outcomes inside their
  `at_risk` window (`*` covers every outcome code without a window of its
  own). A window may open after `entry` (delayed entry), never before.
- `weight` is a sampling weight (1 when absent); `group_id` keeps records
  together across a train/test split.

## Model

Each token is one observed variable (or category, or past event); its value
enters through soft bins over the variable's empirical distribution (FiLM),
and how long ago it was observed through time-ago bins. A bidirectional set
encoder with a summary token turns the subject into a state; cause-specific
piecewise-constant hazards read that state together with age and calendar
time per piece, and cumulative incidence follows in closed form. A masked-
value objective (Gaussian, with detection-limit terms) trains the encoder on
the values themselves. `TimelineSpec::additive(true)` trains the additive
proportional-hazards baseline on the same inputs instead.

## SDK

```rust
use brain::timeline::{read_jsonl, TimelineModel, TimelineSpec};
let (train, held_out) = (read_jsonl("train.jsonl")?, read_jsonl("held_out.jsonl")?);
let spec = TimelineSpec::new(["death:heart", "death:other"], ["death:heart", "death:other"]);
let (model, report) = TimelineModel::train(&train, &held_out, &spec)?;
let p = model.predict(&held_out)?;
let ten_year_risk = p[0].cif("death:heart", 10.0);
model.save("model")?;
```

Training early-stops on the held-out event likelihood and keeps the best
model. Evaluate with `brain::survival`: Uno's concordance, the IPCW Brier
score and its integral, D-calibration and calibration at a horizon, all
accepting sampling weights.

## Limits

- One summary state per prediction time: a subject's history is a set, not a
  trajectory, and predicting from a later visit means a new record with a
  later `entry`. A recurrent backbone over visits is planned.
- Curves are held constant past the last knot; the model says nothing about
  later times.
- No serving surface yet (CLI, HTTP, D-Bus).
