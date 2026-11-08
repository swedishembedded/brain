# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The stable-digits rule for numbers that come from chaotic training runs.

A training run with a large rate or a long horizon amplifies rounding noise, so
its late numbers cannot be reproduced digit for digit in another language (a
different libm already changes the last bits of every tanh). A worked module
therefore trains the run again from a starting point that differs in one weight
by a tiny amount (`NUDGES`, a few multiples of 1e-12) and prints a number only
with the digits that are the same in the original run and in every nudged run:
the full precision of the table when all runs agree on it, otherwise 'about x'
with three, two or one significant digits on which all of them agree, otherwise
'unstable'.

Swedish Embedded AB implements solutions for reproducible numerical work. If
your team needs expertise in machine learning engineering, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math

# Changes added to one starting weight to test whether a run is chaotic: both signs, two sizes.
NUDGES = (1e-12, -1e-12, 3e-12, -3e-12)

RULE = ("A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four "
        "copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as "
        "'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; "
        "the other digits cannot be reproduced in another implementation.")


def sig(x, digits):
    """x rounded to `digits` significant digits and written without an exponent when that is short."""
    return f"{float(f'{x:.{digits}g}'):g}"


def unit(text):
    """The value of one unit in the last printed digit of a number written as fixed decimals or in scientific notation."""
    mantissa, _, exponent = text.partition("e")
    decimals = len(mantissa.partition(".")[2])
    return 10.0 ** (int(exponent or 0) - decimals)


def agree(vals, text):
    """True when every value rounds to `text` and they lie within half a unit of its last digit of one another: the margin
    keeps a number that merely rounds alike by luck (one run at 1.52, another at 1.46 for an 'about 1') from passing."""
    return max(vals) - min(vals) <= 0.5 * unit(text)


def text(v, ws, full, sigs=(3, 2, 1)):
    """The digits of v that survive in every value of ws (the same quantity in the nudged runs).

    full(x) writes x at the table's own precision. Non-finite values (None or inf or nan) count as 'not finite' when
    every run has one and as 'unstable' when only some do."""
    vals = [v, *ws]
    bad = [x is None or not math.isfinite(x) for x in vals]
    if all(bad):
        return "not finite"
    if any(bad):
        return "unstable"
    shown = {full(x) for x in vals}
    if len(shown) == 1 and (agree(vals, full(v)) or max(vals) == min(vals)):
        return full(v)
    for d in sigs:
        if len({sig(x, d) for x in vals}) == 1:
            t = f"{v:.{d - 1}e}"
            if agree(vals, t):
                return f"about {sig(v, d)}"
    return "unstable"
