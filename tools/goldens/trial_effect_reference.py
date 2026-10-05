#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Reference values for crates/survival's effect module, from statsmodels.

Writes crates/survival/testdata/effect_reference.json: a small two-arm trial
(outcome, arm, a prognostic score) and statsmodels' OLS of the outcome on
[1, arm] and on [1, arm, score] with HC3 standard errors - the arm
coefficient, its standard error, its 95% interval (t, residual degrees of
freedom) and its two-sided p-value. crates/survival/tests/reference.rs holds
the Rust implementation to every number. The inputs are in the file itself.

Run: python3 tools/goldens/trial_effect_reference.py  (needs numpy, statsmodels).
"""
import json
import os

import numpy as np
import statsmodels
import statsmodels.api as sm

rng = np.random.default_rng(20261005)
n = 40
arm = (np.arange(n) % 2).astype(float)
score = np.round(rng.normal(0.0, 1.0, n), 3)
# Heteroskedastic noise, larger in the treated arm, and an effect of 0.5.
noise = rng.normal(0.0, 1.0, n) * (1.0 + 0.5 * arm)
outcome = np.round(1.0 + 0.5 * arm + 0.8 * score + noise, 3)


def fit(columns):
    x = sm.add_constant(np.column_stack(columns))
    r = sm.OLS(outcome, x).fit(cov_type="HC3", use_t=True)
    lo, hi = r.conf_int(alpha=0.05)[1]
    return {
        "estimate": float(r.params[1]),
        "se": float(r.bse[1]),
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float(r.pvalues[1]),
    }


out = {
    "versions": {"statsmodels": statsmodels.__version__, "numpy": np.__version__},
    "outcome": outcome.tolist(),
    "treated": [bool(a) for a in arm],
    "score": score.tolist(),
    "unadjusted": fit([arm]),
    "adjusted": fit([arm, score]),
}
path = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "survival", "testdata", "effect_reference.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print("wrote", os.path.normpath(path))
