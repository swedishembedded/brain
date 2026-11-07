# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A noisy two-tone signal and its amplitude spectrum.

The top panel is 64 samples taken over one second: a 3 Hz hum of amplitude 2,
an 11 Hz tone of amplitude 1 and Gaussian noise (the chapter's frozen random
stream). The bottom panel is 2 abs(X[k]) / 64 from the chapter's DFT for
k = 0 to 32 (frequency k Hz): two tall stems stand out of the noise floor.

Swedish Embedded AB implements solutions for finding the frequencies that
matter in sensor and audio data. If your team needs expertise in spectral
analysis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fourier as fo  # noqa: E402
from svgplot import Plot  # noqa: E402

W, H = 560, 260


def render() -> str:
    n = 64
    x = fo._noisy_signal(n)
    X = fo._fft(x)
    amp = [(1 if k in (0, n // 2) else 2) * abs(X[k]) / n for k in range(n // 2 + 1)]
    top = Plot((0, n - 1), (-4, 4), width=W, height=H, xlabel="sample n (time, 1/64 s apart)", ylabel="x[n]",
               title="The signal: 64 samples")
    top.line(list(range(n)), x, color=0, width=1.4)
    top.points(list(range(n)), x, color=0, r=2.5)
    bottom = Plot((-0.5, n // 2 + 0.5), (0, 2.5), width=W, height=H, xlabel="frequency k (cycles per second)",
                  ylabel="amplitude 2 abs X[k] / N", title="Its amplitude spectrum")
    for k, a in enumerate(amp):
        bottom.line([k, k], [0, a], color=1, width=3)
    for k in (3, 11):
        bottom.text(k + 0.9, amp[k] + 0.03, f"k = {k}: {amp[k]:.2f}", size=12)
    inner = lambda s: s.split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731
    gap = 8
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {2 * H + gap}" width="{W}" height="{2 * H + gap}">',
           f'<rect width="{W}" height="{2 * H + gap}" fill="#ffffff"/>']
    for i, panel in enumerate((top, bottom)):
        out.append(f'<svg x="0" y="{i * (H + gap)}" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n{inner(panel.svg())}</svg>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
