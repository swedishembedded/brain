# horizon - roadmap

`crates/horizon` - brain's **continuous-time subject-timeline model** (phases 1
and 2 below are done; the temporal backbone, evaluation arithmetic,
uncertainty and serving are not): it reads the irregular history of one subject
(measurements with values, events, interventions, each at a real-valued time)
and answers, for any query time in the future, two kinds of question:

```
P(event k happens before t | history)        - a cumulative incidence curve per event
p(value of variable j at time t | history)   - a predictive distribution per variable
```

It is a brain architecture (`Source::Brain`), trained from scratch, and it is
not a language model. The first application is long-horizon individual health
prediction from cohort and survey data, but nothing in the crate knows about
health: a "subject" may as well be a machine, an account or a patient, and the
variable and event vocabularies are data, not code.

---

## 1. Why a new architecture

The published longitudinal health models each get one part right and share
the same gaps (sources in section 9):

| Gap in prior work | What horizon does instead |
|---|---|
| Long-horizon risk comes from autoregressive rollout, and rollout error compounds (one model falls below the prevalence baseline after about 4 generated steps; another barely beats an age-sex baseline at 20 years) | Closed-form cumulative incidence at ANY horizon from cause-specific piecewise-constant hazards read off the current state. Rollout exists only for scenario sampling |
| Death is censored, or causes compete without delayed entry; no model handles left truncation, so recruitment-age immortal time biases them | One counting-process likelihood: exposure is integrated from each subject's entry time (left truncation), right censoring is exact, death competes with every other event |
| Values are discretised tokens or a Gaussian that ignores detection limits | A measurement head with a heteroscedastic distribution, a censored-value (Tobit) term for "below detection limit", and query-time conditioning |
| Elapsed time is a learned or binned step size that stops meaning physical time, so it does not extrapolate to unseen gaps | Physical elapsed time scales every decay exactly; time constants are initialised from days to a century |
| A state with no input decays to zero, so extrapolating it 20 years means nothing | The recurrent state relaxes towards an age-conditioned population state (an Ornstein-Uhlenbeck-shaped prior), so a far query degrades gracefully to "a typical subject of that age, shifted by what is known about this one" |
| Missingness is imputed or masked with a shared token, which leaks the data source | A variable that was not measured contributes no token at all; "measured, below limit" is its own value state; the source is an explicit input whose effect can be ablated |

## 2. Data contract (generic)

One JSON line per subject, `timeline-v1`, validated at entry by
`#[serde(deny_unknown_fields)]` structs plus semantic checks:

```
subject_id, group_id (never split across train/test), weight (optional sampling weight),
source (string), static: {name: value},
observations: [{t, var, value | {below: limit} | {above: limit} | category}],
events:       [{t, code}],
at_risk:      [{code, from, to}]        - the window in which an event of this code
                                          COULD have been observed (entry, exit/censoring)
interventions:[{t, name, value, assigned: bool}]
```

- `t` is real-valued in one declared unit (years). The model is handed TWO
  clocks per item: the subject's own timescale (for health: attained age) and
  calendar time (to separate period effects from ageing).
- A vocabulary file declares each variable (kind: continuous, count, binary,
  categorical; canonical unit; detection limits) and each event code
  (absorbing or recurrent, which codes it ends).
- Times before the entry time are allowed: a subject may report history
  retrospectively (an age at diagnosis). The likelihood then conditions on the
  subject being event-free from absorbing codes until entry (section 4).

## 3. Architecture

```
observations at one time ─► value embedding ─► set encoder ─► visit token ─┐
  (var id, value, source,     FiLM over soft     (attention over  (time, age,   │
   below/above-limit state)   bins of the value)  the variables)   calendar)    │
events / interventions ───────────────────────────────────────► event token ─┤
                                                                             ▼
                                                temporal backbone (interchangeable)
                                                  A. continuous-time delta memory
                                                  B. attention with real-valued RoPE
                                                                             │
                                                    state z(t) at the last item
                                                                             │
                              propagate to the query time t_q (closed form, A)
                               or query-token attention (B)
                                                                             │
                     ┌───────────────────────────┬───────────────────────────┤
                     ▼                           ▼                           ▼
      hazard head: log λ_k(piece p)    measurement head: p(v_j at t_q)   observation head
      low-rank: h(z, age, Δ) · β_k     heteroscedastic, Tobit-censored   (optional: which
      → CIF_k(t) in closed form         at limits, conditional on alive   variables get measured)
```

**Value embedding.** `e = γ(var) ⊙ φ(value) + β(var) + e_source + e_limitstate`,
with `φ` a soft-binned (linearly interpolated between learned bin embeddings)
encoding of the per-variable robust-normalised value. Evidence for fusing the
value with its variable and for hybrid binning or FiLM over a raw-scalar
encoding is in section 9.

**Set encoder.** Everything measured at one time is a set, not a sequence:
bidirectional attention over the variable tokens of one visit with a learned
summary token. A survey participant with one exam is ONE visit of a few hundred
tokens; this is where most of the first application's signal is.

**Backbone A: continuous-time delta memory.** The existing Gated DeltaNet
recurrence (`model::gdn`) with the decay made physical:

```
S_k = α_k · S_{k-1} (I - β_k k_k k_kᵀ) + β_k v_k k_kᵀ       (delta rule, as today)
α_k = exp(-r_h(x_k) · Δt_k),   r_h > 0, init log-spaced over 1/day .. 1/100 years
```

`gdn_chunk_fwd` already takes a per-token per-head log-decay (`raw_g`) and the
backward returns its gradient, so this is a new gate step (`-softplus(a) * Δt`),
not a new recurrence. Between the last item and a query time with no input:

```
S(t + Δ) = e^{-rΔ} S(t) + (1 - e^{-rΔ}) S̄(age(t + Δ), calendar)
```

where `S̄` is a learned population state. This is the mean-reverting prior of
section 1, and it gives the query-time state in closed form, incrementally:
a new visit costs one recurrent step, never a re-encode of the history.

**Backbone B: attention.** Causal attention over visit and event tokens with
rotary angles computed from real-valued time (`rope2d_*` takes host cos/sin
tables, so real time needs no new kernel) plus a sinusoidal attained-age
embedding. Queries at future times attend as query tokens. Kept as the
comparison arm; which backbone ships is a measurement (section 7), not a
decision made here.

**Hazard head.** Fixed knots over time-since-query `0 < τ_1 < ... < τ_P`
(finer early, out to the longest follow-up the data support). Per event code k
and piece p: `log λ_kp = ⟨h(z, φ(age at piece), φ(calendar)), β_k⟩ + b_kp`,
a low-rank code embedding as in time-to-event pretraining work, so thousands
of codes cost one matrix product. Death (an absorbing code) multiplies into
every other code's incidence:

```
CIF_k(τ_P) = Σ_p (λ_kp / λ_·p) · S(τ_{p-1}) · (1 - exp(-λ_·p · Δ_p)),   λ_·p = Σ_k λ_kp
```

so the cumulative incidence at any horizon is a closed-form sum, and the
probabilities of the competing outcomes sum to at most one by construction.

**Measurement head.** For a query `(var j, t_q)`: location and log-scale of a
Gaussian on the normalised scale (a quantile-bin distribution for variables
declared skewed or multimodal), read from the propagated state with `φ(Δ)`.
A value recorded as below a limit contributes `log Φ((limit - μ)/σ)`.

## 4. Training objective

```
L = Σ_subjects w_i · [ L_event + λ_v · L_value + λ_m · L_masked + λ_s · L_state ] / Σ w_i
```

- **L_event**, counting-process negative log-likelihood per code with exact
  exposure: `-Σ_k [δ_k log λ_k(T_k) - Σ_p λ_kp · exposure_kp]`, where exposure
  is the time inside piece p AND inside the code's `at_risk` window. Left
  truncation is the window's `from`; right censoring is its `to`.
  Retrospective events (before entry) are scored conditional on survival to
  entry: their exposure ends at entry and death's hazard before entry is not
  scored (the subject was sampled alive).
- **L_value**, measurement NLL at observed future times (longitudinal sources
  only), with the Tobit term at limits.
- **L_masked**, the same head at Δ = 0 for 15-40% of a visit's values masked
  out of the set encoder: the only self-supervision a single-visit source
  gives.
- **L_state** (backbone A, longitudinal sources): the state propagated from
  visit k to visit k+1 must predict the state encoded at k+1 (latent
  prediction on a stop-gradient target), which trains the transition model
  rather than only the readouts.
- **w_i** is a per-subject sampling weight (survey designs) normalised per
  source so that no single source dominates.

## 5. Kernels and blocks needed

Reuse first (all exist): embedding, GEMMs, RMSNorm/LayerNorm, masked
bidirectional and causal attention, rope tables, `gdn_chunk_fwd/bwd`,
`gdn_recurrent_step`, `mse`, `pinball`, `bce_logits`, scan kernels.

New, each with a CPU and GPU path, a kernel header and a gradient check:

| Kernel | Why | Status |
|---|---|---|
| `pexp_nll_value` / `pexp_nll_grad` | piecewise-exponential counting-process NLL with an exposure matrix, weighted per subject | done |
| `gauss_cens_nll_value` / `_grad` (+ `wgsl/lib/normal.wgsl`) | heteroscedastic Gaussian NLL with censored-at-limit (Tobit) terms, weighted | done |
| `dt_decay_gate` | `-softplus(a) * Δt` fused into the GDN gate input | phase 3 |
| `cif_closed_form` (+ backward) | only if a calibration term needs the incidence inside the loss; inference computes it on the host (`survival::Curves`) | not needed yet |

## 6. Phases

1. **Contract and head-only model.** Done: `timeline-v1` (`timeline.rs`,
   validated at entry), the fitted vocabulary with quantile value transform
   (`vocab.rs`), encoding that admits only what is known at the prediction
   time (`encode.rs`, tested), exact exposure with delayed entry, censoring
   and competing absorbing codes, the hazard head, and the closed-form
   cumulative incidence (`survival.rs`, against textbook competing-risk
   values). The loss kernels are held to their formula on the CPU backend.
   The baseline on the same contract is `HorizonConfig::additive`: token
   embeddings summed per subject and read out linearly into each code's
   log-hazard beside the per-piece time effects - an additive (GAM)
   proportional-hazards model, with each variable's effect a smooth curve
   through its soft bins. Gradient-checked like the encoder; it recovers the
   additive synthetic truth slightly better than the set encoder does, which
   is the point: on real data the encoder has to beat it. (Depth zero of the
   encoder is NOT a baseline: with no attention layer the summary token never
   sees the other tokens.) Open: a cross-check of the event NLL against an
   external survival library on a fixture (the evaluation arithmetic in
   `crates/survival` already is).
2. **Set encoder, value embedding, measurement and masked heads.** Done:
   FiLM over soft bins, time-ago bins, a pre-LN bidirectional set encoder
   with padding masked out of attention, the summary state, the masked-value
   head with Tobit states. `gradcheck::horizon::check_horizon` (directional
   over every tensor, element-wise over the shared tables) passes on the CPU
   JIT, wgpu and CUDA. `tests/recovery.rs`: trained with held-out early
   stopping on a synthetic population with known age-dependent competing
   hazards, a non-absorbing onset, a detection-limited covariate and an
   irrelevant one, the predicted cumulative incidence on unseen subjects is
   within a third of the covariate-blind error at 5 and 10 years, with a
   bias under a fifth of the mean. Found on the way (and fixed, with a
   regression test): an event lying one ulp past its window end was counted
   as censored, which biased every hazard down; window ends computed from
   event times are now inside their window.
2b. **Forecast head.** Done: a variable's value at a time after entry, from
   the summary state, the variable and the time ahead (`model::forecast`),
   trained on future measurements that never enter the encoder (their
   existence would leak survival into every hazard); censored-Gaussian on the
   normal-score scale, mapped back through the empirical distribution.
   Gradient-checked; on a synthetic population with known drifting
   trajectories (`tests/forecast.rs`) the median forecast at 3 and 5 years
   is several times closer to the truth than carrying the entry value
   forward, and the 5th-95th percentile interval holds about nine in ten
   held-out follow-up measurements. `TimelineModel::forecast` in the SDK.
3. **Backbone A.** Done in its diagonal form (`HorizonConfig::visits`,
   `model::backbone`, kernels `ct_state_scan` and its adjoint): one set per
   visit, a per-channel state that reverts towards a learned population state
   at `softplus` rates over the elapsed time and moves towards each visit
   through a sigmoid gate (the delta rule with a diagonal state), propagated
   to the prediction time in closed form; time reaches the model only through
   the gaps. Gradient-checked on the CPU JIT, wgpu and CUDA (directional over every tensor,
   element-wise over the rates and population state). The extrapolation test
   (`tests/continuous.rs`, `synthetic::drifting`: an Ornstein-Uhlenbeck risk
   factor at irregular visits, the best prediction exact by Kalman filtering)
   trains with last gaps of at most two years and scores six to ten: the
   state is closer to the best prediction than the single set, there and in
   distribution. **Backbone B** done (`Backbone::Attention`, kernel
   `rope_pos`: rotary angles at real-valued positions, its own adjoint with
   the angle negated): a query token at the prediction time attends over the
   visits; gradient-checked on the CPU JIT, wgpu and CUDA. On the same test both backbones
   beat the single set in and out of distribution, the state closer in
   distribution and the attention out of it (one data seed). Open: the
   matrix-state delta rule (the existing GDN recurrence with the decay made
   physical).
4. **Evaluation arithmetic.** Done in the leaf crate `crates/survival`:
   weighted Kaplan-Meier, censoring distribution and Aalen-Johansen; Harrell
   and Uno concordance with competing causes; IPCW Brier score and its
   integral; D-calibration; calibration at a horizon (observed over expected,
   IPCW logistic recalibration intercept and slope, risk groups). Held to
   scikit-survival and scikit-learn on reference data
   (`tools/goldens/survival_metrics_reference.py`). Antolini's
   time-dependent concordance is not implemented: the horizon concordance
   the evaluation protocol uses is Uno's.
5. **Uncertainty.** Done: Venn-Abers intervals for a risk by a horizon
   (`survival::venn_abers`): isotonic calibration on held-out subjects under
   inverse-probability-of-censoring weights, one fit per label of the new
   subject; overconfident scores come out with a calibration slope near one,
   with and without censoring, and the intervals narrow as calibration data
   grows. Open: seeded ensembles for the spread due to training, and
   conformalised survival distributions over whole curves (they need
   percentile times inside follow-up, which a low-event cohort rarely has).
   Interventional arithmetic: `survival::effect` estimates a randomised
   treatment effect adjusted by a model's prognostic score (PROCOVA, HC3
   errors), held to statsmodels (`tools/goldens/trial_effect_reference.py`)
   and, in simulation, keeping its error rate and coverage while narrowing
   the interval as the score's correlation with the outcome predicts. Open:
   a trial whose outcome a trained model can score at baseline.
6. **SDK and serving.** SDK done: `brain::TimelineModel` (`timeline`
   surface, `brain_arch::Domain::Timeline`, architecture row `horizon`,
   `docs/models/horizon.md`) trains with held-out early stopping, predicts
   closed-form curves (`Prediction::cif`/`survival`), exposes the summary
   state, saves and loads (weights with the configuration in the header, the
   vocabulary beside them), and re-exports `brain::survival` and the
   synthetic population; `crates/sdk/tests/timeline.rs` trains, saves,
   reloads and predicts identically. Serving done: `horizon::saved` owns the
   saved-model directory (the SDK delegates to it), `horizon::caps` the one
   `predict` action, a catalog entry and resident adapter
   (`BRAIN_HORIZON_DIR`) put it on HTTP and D-Bus, and `brain horizon
   predict` runs it locally; `tests/caps.rs` holds the served output to the
   in-process prediction.

## 7. What decides between the alternatives

Backbone A vs B, soft bins vs FiLM, quantile vs Gaussian measurement head:
decided on a held-out split by integrated Brier score and calibration at the
declared horizons, with concordance reported but never alone (it is not a
proper scoring rule). Ties go to the cheaper incremental update (A).

## 8. Not in scope here

Which data sources to pool, what qualifies as a training subject, how a
release is gated and any interventional ("what if") estimation are
decisions of the application that trains the model. brain provides the
model, its losses and the evaluation arithmetic.

## 9. Sources

Papers and preprints this design is drawn from (arXiv ids where they exist):
time-to-event pretraining with piecewise-exponential hazards (2301.03150);
a competing-risk next-event health model (npj Digit Med 2026,
s41746-026-02709-z); a generative disease-history model on age as the time
axis (Nature 2025, s41586-025-09529-3); a marked first-occurrence
time-and-value loss (2602.00541); numeric-value encoding studies
(2607.01391, 2604.16775); joint longitudinal, recurrent and terminal event
modelling (2404.03804); monotone survival networks (2103.14755); variable-step
state spaces S5 (2208.04933) and physical-time input-dependent SSMs
(2605.09742); continuous-discrete Kalman filtering (2111.11344); D-calibration
(1811.11347) and its differentiable form (2101.05346); conformalised survival
distributions (2405.07374).
