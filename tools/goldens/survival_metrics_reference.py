#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Reference values for crates/survival, from scikit-survival and scikit-learn.

Writes crates/survival/testdata/reference.json: two small datasets (one
cause with censoring and ties; two competing causes) and what the reference
libraries compute on them. crates/survival/tests/reference.rs holds the Rust
implementation to every number. No checkpoint is involved, so there is no
source block: the inputs are in the file itself.

Run: python3 tools/goldens/survival_metrics_reference.py  (needs numpy,
scikit-survival, scikit-learn).
"""
import json
import os

import numpy as np
import sklearn
import sksurv
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator, cumulative_incidence_competing_risks, kaplan_meier_estimator
from sksurv.util import Surv

rng = np.random.default_rng(20261005)
n = 60
time = np.round(rng.exponential(5.0, n), 1) + 0.1  # one decimal: ties happen
event = rng.random(n) < 0.7
# Informative but noisy predictions: higher for earlier times.
risk = np.round(np.exp(-time / 5.0) + 0.3 * rng.random(n), 3)
train = Surv.from_arrays(event, time)
times = [1.0, 2.5, 4.0, 6.0]

km_t, km_s = kaplan_meier_estimator(event, time)
cens = CensoringDistributionEstimator().fit(train)
G = cens.predict_proba(np.array(times))

harrell = concordance_index_censored(event, time, risk)[0]
tau = 6.0
uno = concordance_index_ipcw(train, train, risk, tau=tau)[0]
# Survival predictions: S_i(t) = exp(-risk_i * t / 4)
surv = np.array([[np.exp(-r * t / 4.0) for t in times] for r in risk])
_, bs = brier_score(train, train, surv, times)
ibs = integrated_brier_score(train, train, surv, times)

# Time-dependent AUC (cumulative cases, dynamic controls) of the risk score.
auc_times = [1.0, 2.5, 4.0]
auc_by_time = cumulative_dynamic_auc(train, train, risk, auc_times)[0]

# IPCW logistic recalibration at t* = 4 of F = 1 - S.
ts = 4.0
F = 1.0 - np.exp(-risk * ts / 4.0)
x, y, v = [], [], []
for i in range(n):
    if time[i] <= ts:
        if not event[i]:
            continue
        g = cens.predict_proba(np.array([time[i]]))[0]
        yy = 1.0
    else:
        g = cens.predict_proba(np.array([ts]))[0]
        yy = 0.0
    p = np.clip(F[i], 1e-12, 1 - 1e-12)
    x.append(np.log(p / (1 - p)))
    y.append(yy)
    v.append(1.0 / g)
lr = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(np.array(x)[:, None], np.array(y), sample_weight=np.array(v))

# Competing risks: cause 1 or 2, or censored (0).
cause = rng.choice([0, 1, 2], size=n, p=[0.3, 0.45, 0.25])
ctime = np.round(rng.exponential(4.0, n), 1) + 0.1
cr_t, cr = cumulative_incidence_competing_risks(cause, ctime)
cif1 = [float(cr[1][np.searchsorted(cr_t, t, side="right") - 1]) if t >= cr_t[0] else 0.0 for t in times]

out = {
    "versions": {"scikit-survival": sksurv.__version__, "scikit-learn": sklearn.__version__},
    "single": {"time": time.tolist(), "event": event.tolist(), "risk": risk.tolist()},
    "times": times,
    "km": [float(km_s[np.searchsorted(km_t, t, side="right") - 1]) for t in times],
    "censoring": G.tolist(),
    "harrell": harrell,
    "uno_tau": tau,
    "uno": uno,
    "auc_times": auc_times,
    "auc": auc_by_time.tolist(),
    "brier": bs.tolist(),
    "ibs": ibs,
    "recal_horizon": ts,
    "recal_intercept": float(lr.intercept_[0]),
    "recal_slope": float(lr.coef_[0][0]),
    "competing": {"time": ctime.tolist(), "cause": cause.tolist()},
    "aalen_johansen_cause1": cif1,
}
path = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "survival", "testdata", "reference.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.normpath(path))
