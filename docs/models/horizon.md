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
| CLI (`brain <arch> <action>`) | [x] (`brain horizon predict`) |
| HTTP API | [x] (`brain serve`, `BRAIN_HORIZON_DIR`) |
| D-Bus | [x] (`brain serve`, `BRAIN_HORIZON_DIR`) |

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
the values themselves. With `TimelineSpec::forecasts`, a forecast head reads
the same state to predict a variable's distribution at a later time (trained
on the subjects' future measurements, never fed to the encoder, since a
later measurement says the subject was alive then). `TimelineSpec::additive(true)` trains the additive
proportional-hazards baseline on the same inputs instead.

With `TimelineSpec::visits(n)` the history is read visit by visit: every
distinct observation time is its own set, the most recent `n` are kept, and
a continuous-time state carries them to the prediction time. Between visits
the state reverts towards a learned population state at a learned rate per
channel, over the time that actually passed; each visit then moves it
towards its own evidence through a learned gate. Time reaches the model only
through those gaps, so a gap longer than any in training is extrapolated by
the same exponential rather than by an embedding never trained there. On a
synthetic risk factor that drifts in continuous time (`synthetic::drifting`,
whose best possible prediction is known exactly), a model trained with the
last visit at most two years before entry predicts subjects whose last
visit was six to ten years before closer to that best prediction than the
single-set encoder does (`tests/continuous.rs`).

`HorizonConfig::backbone = Backbone::Attention` replaces the state with one
layer of attention from a query token at the prediction time over the
visits, rotary angles taken from each visit's real time before entry. On
the same test both backbones beat the single set in and out of
distribution; the state is the closer of the two in distribution and the
attention out of it, on one seed of the data - which ships is a measurement
on the data at hand.

With `TimelineSpec::next_events(codes, weight)` the hazard head carries a
second group of columns after the outcome codes: the codes in the group
compete for being the FIRST to happen after the prediction time, whether or
not an outcome followed, so every history teaches the state which event comes
next and when (a subject's death ends follow-up for the group, a code's
window is its own, and only the earliest event of the group is scored). It is
the same piecewise-exponential likelihood, weighted by `weight` through the
group's exposure and events, so it needs no new kernel. The group is a
training signal: the held-out event NLL that early stopping and evaluation
use is the outcome codes' alone, and `TimelineModel::predict_next_events`
reads the group's curves back (`first(code, t)`, `any(t)`). On the synthetic
population the trained group's first-event distribution is closer to the
generator's than the covariate-blind mean at every code and horizon
(`tests/next_event.rs`).

## Benchmarks with known truth

Four domain-free generators (`horizon::synthetic::{single, competing,
irregular, longitudinal}`) return the truth a score needs, and
`brain-bench` registers one benchmark per generator on the `survival` axis
(`survival_single`, `survival_competing`, `survival_irregular`,
`survival_longitudinal`). Each trains a small horizon model on the device
`BRAIN_BACKEND` selects and scores it against the generator's truth with
`brain-survival` (integrated Brier, Harrell/Uno C, time-dependent AUC,
calibration at a horizon); the tests in `crates/bench/tests/survival_bench.rs`
hold them to:

- `survival_single`: covariates give one absorbing event under censoring; the
  predicted risk ranks like the true risk and is calibrated.
- `survival_competing`: two causes with different covariates and time shapes;
  the model's cumulative incidence matches the analytic one (the generator's
  own test holds that to Aalen-Johansen on the simulated data).
- `survival_irregular`: the visit times carry the information and the values
  are noise; the model beats the same model trained with every subject's
  visits replaced by another's (`irregular::swap_visits`) on held-out event
  NLL.
- `survival_longitudinal`: a hidden state seen through noisy partial
  measurements, with an action; the measurements improve held-out event NLL
  over the baseline covariates alone (`longitudinal::baseline_only`), and the
  forecast head beats the population mean for future measurements.

## SDK

```rust
use brain::timeline::{read_jsonl, TimelineModel, TimelineSpec};
let (train, held_out) = (read_jsonl("train.jsonl")?, read_jsonl("held_out.jsonl")?);
let spec = TimelineSpec::new(["death:heart", "death:other"], ["death:heart", "death:other"]);
let (model, report) = TimelineModel::train(&train, &held_out, &spec)?;
let p = model.predict(&held_out)?;
let ten_year_risk = p[0].cif("death:heart", 10.0);
// With a forecast head (TimelineSpec::forecasts): HbA1c in 5 years: median, 5th and 95th percentiles.
// let hba1c = model.forecast(&held_out, "hba1c_pct", 5.0, &[0.05, 0.5, 0.95])?;
model.save("model")?;
```

Training early-stops on the held-out event likelihood and keeps the best
model. Evaluate with `brain::survival`: Uno's concordance, the IPCW Brier
score and its integral, D-calibration and calibration at a horizon, all
accepting sampling weights. `brain::survival::venn_abers` turns a risk by a
horizon into an interval `(p0, p1)`, calibrated on subjects the model was not
trained on.

## Serving

A saved model directory (what `TimelineModel::save` writes) is served by one
action, `predict`: `timeline-v1` subjects in, survival and each code's
cumulative incidence at the requested times out, as JSON lines. Times past
the last knot are refused.

```bash
brain horizon predict --weights model/ --times 5,10 \
    --in subjects=subjects.jsonl --out predictions=predictions.jsonl
```

`brain serve` serves it on HTTP and D-Bus as `brain/horizon` when
`BRAIN_HORIZON_DIR` names the saved model directory; there the directory is
the host's, never a request parameter.

## Limits

- One prediction time per record: predicting from a later visit means a new
  record with a later `entry`. With `TimelineSpec::visits` the history before
  it is read visit by visit; without, as one set.
- An event at exactly the entry time is neither history nor outcome: history
  is strictly before entry, outcomes strictly after.
- The population state the visit state reverts to is one constant per
  channel, not yet a function of age and calendar time.
- Curves are held constant past the last knot; the model says nothing about
  later times.
