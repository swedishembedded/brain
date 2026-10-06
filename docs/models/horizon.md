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
| CLI (`brain <arch> <action>`) | [x] (`brain horizon train`, `eval`, `calibrate`, `predict`) |
| HTTP API | [x] (`brain serve`, `BRAIN_HORIZON_DIR`) |
| D-Bus | [x] (`brain serve`, `BRAIN_HORIZON_DIR`) |
| Ensembles | [x] (seeded and bootstrap, SDK and `train --members`) |

## Data: `timeline-v1`

One JSON object per line:

```json
{"subject_id": "a", "group_id": "household-1", "weight": 2.5, "source": "survey",
 "entry": 50.0, "calendar_at_entry": 2003.5,
 "observations": [{"t": 50.0, "var": "sbp", "value": 131},
                  {"t": 50.0, "var": "crp", "value": {"below": 0.2}},
                  {"t": 50.0, "var": "smoking", "value": "never"},
                  {"t": 50.0, "var": "ldl", "value": 3.1, "unit": "mmol/L"},
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
- `unit` on an observation is optional (files without it read as before).
  Fitting records each numeric variable's unit from the training subjects in
  `vocab.json`; every subject that states a unit for a variable must agree,
  and a disagreement is a fit error naming the variable and both units. A
  variable no subject gave a unit for stays unitless. horizon has no
  conversion table: at prediction time a measurement stated in another unit
  than the model's is REJECTED (error naming the variable, the subject and
  both units), never converted; a unit stated for a variable the model
  records as unitless is an advisory (below), never a guess. A unit is 1 to
  32 characters without surrounding whitespace or control characters and is
  compared byte for byte.
- `weight` is a sampling weight (1 when absent); `group_id` keeps records
  together across a train/test split.

## Patient history: the update format

`timeline-v1` is the training and evaluation format. To say "this subject's
record, as of now" and ask for a new prediction, a caller sends a patient
history (`horizon::history::PatientHistory`, `brain::timeline::PatientHistory`):

```json
{"id": "p1", "as_of": "2026-03-01", "birth": "1970-06-15",
 "static": {"sex": "female"},
 "events": [{"time": "2019-04-02", "code": "dx:diabetes"},
            {"time": "2026-02-20", "code": "ldl", "value": 3.1, "unit": "mmol/L"},
            {"time": 55.1, "code": "crp", "value": {"below": 0.2}}]}
```

Inference from a history is stateless: the whole history goes in, the weights
never change for a patient and nothing is kept between calls. "Append a
checkup" means sending the history again with one more record.

- **Time.** A `time` or `as_of` is a number (the model's own clock, in years;
  attained age for a person) or an ISO-8601 / RFC 3339 date
  (`YYYY-MM-DD`, midnight UTC) or datetime (`YYYY-MM-DDTHH:MM:SS[.f]` with `Z`
  or `+HH:MM`, normalised to UTC; leap seconds refused). A date needs `birth`
  and becomes `(instant - birth) / 31 557 600 s` (the Julian year of 365.25
  days): exact integer nanoseconds, one floating-point division, the same on
  every machine. A date before `birth` is an error. horizon does not depend on
  a date library.
- **Calendar.** The model also reads the calendar time at the prediction time.
  It is the decimal year of `as_of` (`Y + (instant - Y-01-01T00:00Z) / length
  of Y`, UTC) whenever a date is involved (`as_of` a date, or `birth` given);
  for a numeric `as_of` with no `birth` the history must state `"calendar"`.
  Stating it where it could be derived is an error.
- **Records.** Each `events` element is `{time, code, value?, unit?}`. With a
  `value` it is a measurement of variable `code` (a number, `{"below": x}`,
  `{"above": x}` or a category string; a `unit` only for numbers); without one
  it is an event. `static` maps a variable to a value (or `{"value", "unit"}`)
  known at `as_of`: a measurement at `as_of`.
- **Future.** A record dated after `as_of` never reaches the model: it is
  dropped and reported as `future_ignored`. Measurements at `as_of` are known;
  an event at `as_of` is not history (as in `timeline-v1`, events count
  strictly before the prediction time) and is dropped as `event_at_as_of`.
- **Duplicates and order.** A record equal to another in time, code, value and
  unit is dropped (`duplicate_ignored`), so repeating one changes nothing; two
  measurements of one code at one time that differ in value or unit are an
  error. Records are sorted by (time, code, value, unit) before the model sees
  them: listing order never matters.
- **Unknown fields**, at any level, are never read and are reported as
  `unknown_field`.
- **Units** are compared with the model's (see Data): a mismatch is an error.
- **Bounds.** At most 10 000 records, 1 000 static values, 1 024 histories and
  8 MiB of text per call; codes and categories up to 128 characters, units up
  to 32. Every error names the record (`events[3]`, `static.sex`) and field.

A file may hold one history object, an array of them, or one object per line.

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

### A stack of sequence mixers over the visits

`Backbone::Stack(StackConfig { mixer, blocks, hybrid_period })` (SDK:
`TimelineSpec::visits(n).mixer(Mixer::..., blocks)`) replaces the single
state or attention layer with `blocks` residual blocks over the visit
sequence, in brain's usual pre-norm style: `x += Mixer(LN(x))`, then
`x += MLP(LN(x))`. A subject's sequence is its visit slots in time order, a
query token at the prediction time, and padding up to a whole number of
chunks of the delta rule; the state the heads read is the final-norm output at
the query row. `Mixer` says what each block mixes with:

- `Mixer::Attention`: bidirectional multi-head attention over the visits,
  rotary angles from each row's real time relative to the prediction time
  (every visit precedes it, so visits seeing each other leaks nothing); unused
  slots and padding are never keys.
- `Mixer::GatedDeltaNet`: the matrix-state delta rule of `model::gdn` (the
  recurrence the Qwen3.5 mixer trains with, on the same chunked kernels and
  their hand-written backward), per head, with unit-norm `q` and `k`:
  `S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`,
  `o_t = S_t q_t / sqrt(d_head)`. The decay is gap-aware:
  `g_t = -softplus(rate_h) * dt_t`, with `dt_t` the physical time since the
  previous visit (for the query, since the last visit) and `rate_h` a learned
  rate per head per unit of time, so a longer gap decays more, exactly as the
  diagonal state's exponential does, and a gap longer than any in training
  extrapolates by the same exponential. The raw rates start log-spaced from
  days to a century (as the diagonal state's do). An unused visit slot or a
  padding row has `dt < 0`: it neither decays nor writes, and the recurrence
  passes the state through it, so a subject's prediction is the same however
  many empty slots or neighbours surround it (`tests/stack.rs`). `beta_t` is a
  sigmoid of a projection of the token, so each visit decides how strongly it
  overwrites what the state holds along its key. A new visit costs one more
  recurrent step.
- `Mixer::Hybrid`: Gated DeltaNet blocks with a full-attention block as every
  `hybrid_period`-th block (default 4: three recurrent blocks to one attention
  block, the last block of each group attending).

The existing kernels cover the recurrence and its backward unchanged; the
only new ones are the two elementwise gates, `gdn_gap_gate` (the log-decay
from the elapsed time and the write strength, zero on padding) and its
backward. `Backbone::State` and `Backbone::Attention` are untouched: their
parameters, outputs and checkpoints read as before, and a configuration
without a stack deserialises to what it did.

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

### Mixer ablation

Which sequence mixer a stack should use is a measurement on the data at hand.
`bench::survival_bench::MixerAblation` trains the all-attention stack, the
all-Gated-DeltaNet stack and the 3:1 hybrid (the same number of blocks, width
and feed-forward size, so the parameter counts differ only by the mixers' own
small tensors and are held within a stated tolerance) on identical data,
split, seed and optimisation steps (early stopping is off so every variant
takes every step; the best held-out evaluation's weights are kept), on two of
the generators above: `irregular` (visit times carry the information) and
`longitudinal` (a hidden state seen through noisy partial measurements). It
prints one row per variant: held-out event NLL, integrated Brier score,
time-dependent AUC at the horizon and training time.

```bash
cargo test --release -p brain-bench --test survival_mixers -- --nocapture --test-threads=1
```

The test checks that the comparison is fair, not who wins.

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
accepting sampling weights. `brain::survival::recalibration` maps a risk by a
horizon through a logistic recalibration fitted on subjects the model was not
trained on, and `brain::survival::venn_abers` turns it into an interval
`(p0, p1)` instead.

## Calibration

`TimelineModel::calibrate(validation, &CalibrationSpec::new([5.0, 10.0]))`
fits, for every outcome code and every requested horizon, a calibrator on the
VALIDATION subjects: the subjects the model was neither trained nor
early-stopped on, never the test set the result is judged on. Censoring is
handled by inverse-probability-of-censoring weights with the censoring
distribution estimated on the validation subjects themselves, and a code
competes with the absorbing codes exactly as its cumulative incidence does.

```rust
let mut model = TimelineModel::load("model")?;
model.calibrate(&validation, &CalibrationSpec::new([5.0, 10.0]))?;
model.save("model")?;                       // writes calibration.json beside the weights
let p = &model.predict(&test)?[0];
p.cif("death:heart", 10.0);                 // the model's raw risk
p.calibrated_cif("death:heart", 10.0);      // Some(calibrated risk) at a calibrated horizon
p.cif_interval("death:heart", 10.0);        // None for the default kind; (p0, p1) for Venn-Abers
```

Two kinds of calibrator, selected with `CalibrationSpec::kind` (`kind` of the
`calibrate` action) and recorded in `calibration.json`:

| Kind | Calibrated risk | Interval | Events needed |
|---|---|---|---|
| `logistic` (default) | `logit P = a + b logit F`, fitted under IPCW; the slope `b` is estimated only where the validation subjects show it differs from one by more than two standard errors, otherwise the intercept alone is (calibration in the large) | none: `cif_interval` is `None` (`null` when served) | 100 |
| `venn_abers` | the merged probability of an isotonic fit with its Venn-Abers interval | `(p0, p1)`: one end is calibrated whatever the model, the width is how little calibration data stand behind the score | 500 |

The default was chosen by measurement, on populations whose true risk is known
per subject (a large independent test population, several validation sizes,
models trained apart; `crates/horizon/tests/calibration_measure.rs`). An
isotonic fit is a free-form step function; fitted on a few hundred events it
chases noise, so its output is more spread than the truth. The recalibration
slope measured on new data then falls below one - on a model that was already
calibrated, and even on the true risk itself - and the shortfall shrinks only
slowly with events. The same happens to the midpoint of the interval and to
a smoothed version of the steps, so it is the noise and not the step shape. A
logistic map has two parameters at most: its error falls steadily with
events, it leaves a calibrated model essentially unchanged, and it undoes any
distortion that is linear in the log-odds (a shift, over- or under-confidence).
It cannot correct anything else; use `venn_abers` when the model's error is not
of that form and events are plentiful. A calibration step cannot make a model
better than its calibration data allow: with too few events, it is better left
out, which is what the minimum does.

- A horizon is calibrated only if the validation subjects hold at least the
  kind's minimum (`CalibrationKind::min_events`; `CalibrationSpec::min_events`
  sets another) events of the code by it and as many still event-free.
  Otherwise it is NOT calibrated and NOT exposed: `calibrated_cif` is `None`,
  the capability answers `null`, never zero. The pairs left out, with their
  event counts, are listed by `Calibration::uncalibrated`; a horizon with
  enough events whose data give no usable map (no spread in the predictions, a
  separable fit, a slope that is not positive) is listed with the reason
  `no_fit`. A model that is left raw at a horizon is judged raw there.
- Only the horizons asked for are calibrated; any other time has no
  calibrated risk (the raw risk is always there).
- `calibration.json` records the format version, the kind, the SHA-256 of the
  weights it was fitted for, the number of validation subjects and, per (code,
  horizon), the events and the calibrator itself (the intercept and slope, or
  the isotonic fit). Loading refuses a calibration beside other weights. A
  directory without the file loads as uncalibrated. A file written before
  kinds existed (version 1) is a Venn-Abers calibration and loads as it did.
- An ensemble's calibrated risk is the mean of its members' calibrated risks,
  `None` unless every member has one; its interval exists only where every
  member has one.
- On a population trained on too few subjects, the first-onset risk of the
  model is overconfident (recalibration slope 0.49, observed over expected
  1.20, expected calibration error 0.040); calibrating on 8000 validation
  subjects with either kind brings the slope and the observed over expected
  toward one and the error down on a test population neither saw
  (`tests/timeline.rs` in the SDK holds it).

## Training support and abstention

Training records what the model was trained on in `support.json` beside the
weights: per numeric variable the robust range of its values (0.5th to 99.5th
percentile, detection limits counted at their limit), per categorical
variable the levels seen, the event codes seen in histories, the ranges of
entry clock, calendar time and history length (observations plus events, and
distinct visits), and the mean and covariance of the learned state for a
Mahalanobis distance (kept only when the training set has at least four
subjects per state dimension). Units are the
vocabulary's, not the support's: a unit mix-up is rejected by name before a
range could show it.

`TimelineModel::assess(subjects)` returns, per subject, `supported`, a
continuous `ood_score` and typed warnings:

| Warning | Raised when |
|---|---|
| `unknown_variable`, `unknown_category`, `unknown_event_code` | the name or level was never seen in training (score 10) |
| `value_out_of_range` | a measurement is beyond the variable's range plus a margin (a quarter of the range's width by default) |
| `entry_out_of_range`, `calendar_out_of_range` | the entry clock or calendar time is beyond the trained range plus the margin |
| `history_length` | observations plus events, or visits, are far below or above any in training |
| `state_out_of_support` | the Mahalanobis distance of the learned state is beyond the training set's 99.5th percentile plus half again |

A unit stated for a variable the model records as unitless is listed under
`advisories` (`unit_not_recorded`) beside the warnings: it never changes
`supported` or the score and never withholds an answer.

Each component is scaled so `1.0` is the edge of what is supported and the
`ood_score` is the largest, so `supported` means a score of at most one. The
score is continuous inside the support too, which is what makes it rank
subjects.

- A model saved before support was recorded has no `support.json` and
  assesses as support UNKNOWN (`supported: None`): neither supported nor
  unsupported, and nothing is withheld for it. Support is bound to the weights
  by their SHA-256, like a calibration, and refused beside other weights.
- `TimelineModel::predict_or_abstain(subjects, &Abstain::above(1.0))` returns
  `Err(Unavailable)` ("risk unavailable: insufficient support") INSTEAD of a
  `Prediction` for a subject whose score is above the threshold. The default
  threshold is 1.0, the edge of the support: any warning withholds the
  answer. Unknown inputs score 10, so only a threshold of 10 or more lets them
  through, deliberately.
- Measured on the synthetic population (`tests/timeline.rs` in the SDK): a
  model trained on one population meets test populations shifted to older
  entry ages and higher `x1` (`synthetic::population_shifted`, whose truth is
  exact). Over four shift levels of 1000 subjects the rank correlation between
  the score and the absolute error against the true risk is 0.78 (0.25 to
  0.75 within one level), the mean error rises with every level, the 520
  subjects flagged unsupported have a mean error 3.7 times that of the others,
  and none of 1000 unshifted subjects is flagged.

What it is not: the ranges are marginal, so a combination of values that
never occurred together passes if each value is common; the state distance is
the only joint check; a real population with heavy tails needs a larger margin
(`AssessOptions::margin`) or it will flag legitimate subjects. The score says
the model has not seen anything like this subject; it does not say the
prediction is wrong, and an in-distribution subject can still be predicted
badly.

## Structured forecast

`TimelineModel::forecast_history(&PatientHistory, &ForecastRequest) ->
RiskForecast` (and `forecast_histories` for a batch, one forward pass) is the
call behind "append a checkup and re-predict":

```rust
use brain::timeline::{ForecastRequest, PatientHistory, TimelineModel};
let model = TimelineModel::load("model")?;
let history = PatientHistory::parse_all(&std::fs::read_to_string("patient.json")?)?.remove(0);
let forecast = model.forecast_history(&history, &ForecastRequest::new([5.0, 10.0]))?;
if forecast.is_available() {
    let r = &forecast.horizons[1].risks["death:heart"];
    println!("raw {} calibrated {:?} interval {:?}", r.raw, r.calibrated, r.interval);
}
```

A `RiskForecast` (serialises to JSON) holds:

- `as_of` (as written, on the model's clock, and the calendar time read) and
  `model`: the SHA-256 of the weights, the SHA-256 of the configuration and the
  brain version, so two answers can be told apart or matched long afterwards;
- `coverage`: the variables and event codes the history had, the number of
  observations and events used, the oldest and newest observation (`time` on
  the model's clock and `ago` before `as_of`) and `missing_variables`, the
  model's variables the history did not measure;
- `curves`: survival and every outcome code's cumulative incidence at the
  model's knots;
- `horizons`: for each requested horizon the survival and per code the `raw`
  probability and, ONLY where the model was calibrated for exactly that code at
  exactly that horizon, `calibrated` and, for a Venn-Abers calibration, its
  `interval`. Otherwise those keys are ABSENT, never zero. A horizon outside `(0, last knot]` is an
  error: the model is not extrapolated;
- uncertainty: that interval, and, for `RiskForecast::ensemble` of forecasts
  from models trained apart, the members' identities, the mean of every
  probability, and per horizon `member_range` (and per knot `cif_min` /
  `cif_max`); an ensemble abstains if any member does and has a calibrated risk
  only where every member has one;
- `support` (the `assess` output: `supported`, `ood_score`, typed `warnings`,
  `advisories`) and `input_warnings` (what the history format dropped);
- `risk: "unavailable"` with `reason: "insufficient support"` and NO `curves`
  or `horizons` when the support score is above the request's `max_ood_score`
  (default 1, the edge of the support), exactly as `predict_or_abstain` does;
- a `disclaimer`.

A forecast does not diagnose and recommends no treatment, and says so in every
answer. Its probabilities are associations in the training population, not the
effect of any action, and they hold only as far as the model's calibration and
support do (see Limits).

On the synthetic population (`tests/timeline.rs`, SDK) the history route gives
bit-for-bit the probabilities, calibrated risks and intervals that `predict`
gives the equivalent subject, whatever order the records are listed in.

## Evaluation

`TimelineModel::evaluate(&subjects, &EvaluationSpec::new([5.0, 10.0]))` judges
the model on held-out subjects (never trained, early-stopped or calibrated on).
Per outcome code and horizon, under the subjects' sampling weights and with a
code competing with the absorbing codes as its cumulative incidence does:
Uno's concordance truncated at the horizon, the time-dependent AUC and the
IPCW Brier score at the horizon, the Brier score integrated over `(0, horizon]`
(20 grid points), and calibration (recalibration slope and intercept,
observed over expected with an Aalen-Johansen observed side, and the expected
calibration error over ten risk groups). Subjects with a `group_id` get
percentile intervals (95%, 200 resamples, seeded) of the concordance, AUC and
Brier score from a bootstrap that resamples whole groups
(`survival::compare::cluster_bootstrap_by`). The held-out event NLL is one
number per subject set.

- A (code, horizon) with fewer than `min_events` events by then (30 by
  default) is absent from `results` and
  listed in `absent` with its count. A metric that cannot be computed is
  `None`, never zero.
- The censoring distribution behind the IPCW weights is estimated on the
  evaluated subjects unless `censoring_from(training subjects)` names a
  reference set.
- Measured (`forecast_history_and_evaluate_a_trained_model`, SDK; 6000 trained
  subjects, 6000 evaluated in households of two): for `death:a` by 10 years,
  953 events, Uno C 0.729, AUC 0.771 against 0.781 for the generator's true
  risks, Brier 0.1276, integrated Brier 0.0764, calibration slope 1.07,
  observed over expected 1.10, expected calibration error 0.019.

## Training, evaluation and calibration from the command line

`brain horizon train`, `eval` and `calibrate` are the capability actions around
`predict` (`brain caps horizon` lists all four with their schemas). Their inputs
are `timeline-v1` files, validated at entry: an error names the input and the
line. Nothing in them knows the domain.

```bash
brain horizon train --dataset train.jsonl --held-out held-out.jsonl --out model/ \
    --absorbing death:a,death:b --knots 0,2,5,10 --steps 2000 --json
brain horizon eval --weights model/ --dataset test.jsonl --horizons 5,10 --json
brain horizon calibrate --weights model/ --validation validation.jsonl --horizons 5,10
brain horizon predict --weights model/ --history patient.json --json
```

- **`train`** takes `--codes a,b,c` (default: every event code that occurs after
  a subject's entry), `--absorbing`, `--knots`, `--steps`, `--batch`, `--seed`,
  `--eval-interval`, `--patience`, `--next-events a,b --next-weight W`,
  `--mixer attention|gated-delta-net|hybrid --blocks N [--visits N]`,
  `--forecasts N`, and `--members N --ensemble seeded|bootstrap`. It is the
  SDK's `TimelineModel::train` (one function, `horizon::fit::train`): early
  stopping on the held-out event NLL, keeping the best model. It is long
  running, so it reports progress at every evaluation interval and polls the
  job's cancel token after every optimiser step. A cancelled or failed run
  returns no model: the directory is written beside its final name and renamed
  into place only when training is complete, an existing `--out` is refused
  before any training starts (`--force` replaces it, after training), and
  nothing is left behind. It prints a report (`--json`): per member the seed,
  steps, losses, held-out event NLL, parameter count and weights digest.
- **`eval`** is `TimelineModel::evaluate` (see Evaluation) printed as JSON: per
  outcome code and horizon, Uno's C, the time-dependent AUC, the IPCW Brier
  score, the integrated Brier score and calibration, and the event NLL.
  Horizons with fewer than `--min-events` events are absent from `results` and
  listed under `absent`. It takes a single model; evaluate an ensemble's member
  (`members/<n>`).
- **`calibrate`** is `TimelineModel::calibrate` (see Calibration) on the
  `validation` subjects. It writes `calibration.json` into the model
  directory, or a calibrated copy of the model into `--out`, and refuses to
  overwrite either without `--force`.

`--out` of `train` and `calibrate` is a directory: those two actions have no
`--out NAME=PATH` output-blob flag (their outputs print with `--json`).
Underscored names are also written with dashes (`--held-out`).

## Ensembles and uncertainty

`TimelineEnsemble::train(train, held_out, spec, members, kind)` (and `brain
horizon train --members N --ensemble seeded|bootstrap`) trains several models
apart:

- `EnsembleKind::Seeded`: the same subjects, member `i` with the spec's seed
  plus `i`;
- `EnsembleKind::Bootstrap`: each member also trains on the subjects resampled
  with replacement BY GROUP (a subject without a `group_id` is its own group).

The vocabulary is fitted once on the full training set and shared, so members
predict over the same codes, variables and knots; early stopping uses the same
held-out subjects for every member. An ensemble is saved as one directory:
`ensemble.json` (the kind, and per member its seed, its bootstrap draws and the
SHA-256 of its weights) and `members/0`, `members/1`, ... each an ordinary model
directory. Loading verifies every digest (a member swapped for another valid
model is refused), that each member sits at `members/<index>` and that the
members share a vocabulary. The directory is replaced only by an ensemble.

`Prediction` and `RiskForecast` of an ensemble are the mean of the members with
their disagreement kept: `member_cifs` and `cif_spread` (the standard deviation
across members), and for a forecast the `member_range` per horizon and the
`cif_min`/`cif_max` per knot. An ensemble abstains if any member does and has a
calibrated risk only where every member has one. The served `predict` loads an
ensemble directory without being told it is one: subjects answer the mean `cif`
and an `ensemble` object (`members`, `cif_min`, `cif_max` per time), histories
the ensemble forecast.

**Monte-Carlo dropout is not available:** horizon's architecture has no dropout,
and none was added for this. The spread of an ensemble is the model's
uncertainty about itself (what training noise and the training subjects leave
open), not the Venn-Abers interval (what the calibration data leave open): the
two answer different questions and are reported separately.

What was measured (synthetic populations with known truth,
`crates/sdk/tests/ensemble.rs`, `ensemble_comparison`, two seeded data sets): the
mean risk of both kinds of ensemble was closer to the true risk than one model's
on the population they were trained on, and calibrated at least as well in most
cases (neither kind was consistently better calibrated than the other); the
member spread correlates positively with the error to the true risk at every
shift level of `population_shifted`, weakly at the largest shift, and a seeded
ensemble's mean risk was no worse than a bootstrap's on shifted data while the
bootstrap's members spread more. An ensemble of `n` trains `n` models: it costs
`n` times the steps of a single model.

## Serving

A saved model directory (what `TimelineModel::save` writes) or ensemble
directory (what `TimelineEnsemble::save` writes) is served by `brain serve` as
`brain/horizon` with four actions: `predict`, `eval`, `calibrate` and `train`.

`predict` takes one of two inputs. `timeline-v1` subjects (the
`subjects` blob) give survival and each code's cumulative incidence at the
requested times, as JSON lines. A patient history (the `history` blob: one
object, an array or one per line, see Patient history) gives a structured
forecast per history (see Structured forecast): the requested `times` are its
horizons, one JSON line per history in the `predictions` blob and the same
list as the `forecasts` output, so `--json` prints it. Giving both inputs is
an error, as is a unit the model was not trained on; either fails its own
request, not the batch it is in. Times past
the last knot are refused. A calibrated model also answers `cif_calibrated`
and `cif_interval` per code and time, `null` where that time was not
calibrated.

`eval` judges the served model on the request's `dataset` subjects. `calibrate`
returns the `calibration` blob (the `calibration.json` content) for the
request's `validation` subjects and writes nothing: the served directory is the
host's, so an operator installs it with `brain horizon calibrate` on the
host. `train` needs no served weights; it reports progress, honours the job's
cancellation (the D-Bus `Cancel`) and writes the finished model to the host's
`BRAIN_HORIZON_TRAIN_DIR` (an existing directory there is refused). The
datasets travel as blobs; no request names a path on the host.

Every answer carries `support: {supported, ood_score, warnings}`. A subject
whose score is above the request's `max_ood_score` (default 1, the edge of
the support; a host parameter of the action like `times`) gets
`{"subject_id", "risk": "unavailable", "reason": "insufficient support",
"support"}` instead of probabilities, and the outcome's `abstained` counts
them. A model without recorded support answers `supported: null` and never
abstains.

Concurrent requests are one batch: the resident model puts every request's
subjects through one forward pass in device batches of the model's size and
splits the answers back per request, in order. A request that cannot be
answered (a malformed file, a time past the knots) fails alone. A subject's
prediction never depends on its batch neighbours or on padding
(`tests/batching.rs`).

```bash
brain horizon predict --weights model/ --times 5,10 \
    --in subjects=subjects.jsonl --out predictions=predictions.jsonl
brain horizon predict --weights model/ --times 5,10 --history patient.json --json
```

Every input blob of every capability also takes `--<name> PATH` as the
shorthand for `--in <name>=PATH` (unless a parameter has that name), which is
what `--history patient.json` is. `samples/shell/timeline/predict/history.sh`
runs a patient through H0 to R0, an appended checkup to R1 and another to R2.

`brain serve` serves it on HTTP and D-Bus as `brain/horizon` when
`BRAIN_HORIZON_DIR` names the saved model (or ensemble) directory; there the
directory is the host's, never a request parameter. A served `train` is the
long-running case: it is one job on its own instance (nothing loaded) and one
that a caller can cancel. `samples/shell/timeline/lifecycle/lifecycle.sh` runs
train, eval, calibrate and predict (and a bootstrap ensemble) on the study
sample's datasets.

## Limits

- `eval` and `calibrate` take one model, not an ensemble; there is no
  ensemble-level calibration (the members' own calibrations average where all
  members have one). A served `train` writes to the host's directory and
  returns its report but not the model files; a remote client fetches the
  directory on the host. There is no Monte-Carlo dropout.
- Calibration and support describe the population the model was trained,
  validated and assessed on; they do not transfer to another one. A
  calibration needs events: a recalibration slope estimated from a hundred
  events is itself too noisy to trust much, and an isotonic fit needs several
  hundred before it stops flattening the slope of even a perfect model.

- One prediction time per record: predicting from a later visit means a new
  record with a later `entry`. With `TimelineSpec::visits` the history before
  it is read visit by visit; without, as one set.
- An event at exactly the entry time is neither history nor outcome: history
  is strictly before entry, outcomes strictly after.
- The population state the visit state reverts to is one constant per
  channel, not yet a function of age and calendar time. A Gated DeltaNet
  stack has no population state: its matrix decays towards zero over a long
  gap, and the query token (a learned embedding) is what a far prediction
  reads.
- A Gated DeltaNet block has one decay rate per head, not per channel, and
  the rate does not depend on the token; the delta rule's write strength does.
- The stack's rows per subject are the visit slots plus the query, padded to
  chunks of at most 16, so a model with many visit slots pays for the padding
  and the chunked recurrence in sequence length.
- Curves are held constant past the last knot; the model says nothing about
  later times.
