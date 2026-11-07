# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Mean training loss (eval mode, on the 16 training points) over the 8 seeds for the same four settings as the validation figure.

Computed by worked module `regularization_nn`. The horizontal line is the noise floor: a network that has learned only the
curve has a training loss near it; a training loss far below it is the fit to the noise.

Swedish Embedded AB implements solutions for models that keep working on data they have not seen. If your team needs
expertise in regularizing neural networks, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regularization_nn as rg  # noqa: E402
from svgplot import Plot  # noqa: E402


def mean_curve(runs, column):
    n = len(runs)
    return [sum(r["history"][i][column] for r in runs) / n for i in range(len(runs[0]["history"]))]


def render() -> str:
    wd, dp, ag = rg._chosen()
    data = {name: runs for name, _, runs in rg.comparison_data()}
    spec = [("none", "no regularization", 0), ("weight decay", f"weight decay, lambda {wd:g}", 2),
            ("dropout", f"dropout, p {dp:g}", 1), ("augmentation", f"input noise, sd {ag:g}", 4)]
    steps = [row[0] for row in data["none"][0]["history"]]
    p = Plot((0, rg.STEPS), (0, 0.2), width=640, height=380, xlabel="gradient step", ylabel="training loss (mean of 8 seeds)",
             title="Training loss with each method")
    p.hline(rg.NOISE_SD ** 2)
    for name, label, color in spec:
        p.line(steps, mean_curve(data[name], 1), color=color, label=label)
    return p.svg()
