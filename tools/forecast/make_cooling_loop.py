#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Generate a synthetic industrial cooling-loop series: a heat exchanger whose
conductance quietly degrades (fouling) while an unmeasured, schedule-driven
heat load pushes the return temperature toward a trip threshold.

The physics is a one-state lumped energy balance across the exchanger:

    C * dT_return/dt = Q(t) - UA(t) * (T_return(t) - T_amb(t))

`UA(t)` (conductance) decays exponentially between periodic cleanings - real
fouling behavior, not a random walk. `Q(t)` (heat load) follows a production
shift schedule (on/off) plus occasional multi-hour batch bursts; it is what a
conventional observer (an EKF tracking `T_return`/`UA` from the energy balance
alone) has no model of, which is the whole point of this example: the observer
tracks the PRESENT state well and is honestly wrong about the FUTURE, because
persisting `Q` forward ignores the schedule that is about to change it.

Columns written (hourly bars): `timestamp, t_return, q_load, t_amb,
pump_power, shift_on`.

  t_return    the TARGET - return coolant temperature.
  q_load      the disturbance driving it - NOT observable directly downstream
              of the exchanger; only its effect on t_return is measured.
  t_amb       ambient temperature - a Role::KnownFuture covariate (a site
              already has a short-range weather/ambient forecast).
  pump_power  correlates with load but is noisy and lags it - a
              Role::PastCovariate (measured, not known in advance).
  shift_on    1.0 during a production shift, 0.0 otherwise - a
              Role::KnownFuture covariate (the schedule is planned in advance).

Usage:
  python3 tools/forecast/make_cooling_loop.py --out examples/forecast/cooling_loop.csv \
    --hours 720 --seed 7
"""
import argparse
import csv
import datetime
import math
import random


def simulate(hours: int, seed: int):
    rng = random.Random(seed)
    dt = 1.0  # hours
    C = 40.0  # thermal mass
    # UA sized so a CLEAN exchanger holds full-load return temp around 35 C
    # (18 C ambient + 220/13), and a FOULED one (UA droops ~40% over one
    # cleaning cycle) pushes full-load steady state past the 45 C trip - the
    # dynamic the example is built to show: safe when clean, marginal by the
    # end of a dirty cycle, and pushed over the edge by a batch burst on top.
    ua_clean = 13.0
    fouling_rate = 0.002  # per hour
    clean_interval = 240  # hours between cleanings (~10 days)
    q_off, q_on = 60.0, 220.0  # heat load, off-shift vs on-shift baseline
    shift_start, shift_end = 6, 22  # local hour of day

    def shift_on_at(h):
        return shift_start <= (h % 24) < shift_end

    # Batch bursts: a few hours of extra load starting at random on-shift
    # hours, decaying exponentially. Generated independently of the per-hour
    # loop below (and of UA/T_amb) - this schedule is exactly the part a
    # conventional observer, which only sees the energy balance, has no model
    # of.
    bursts = [0.0] * hours
    h = 0
    while h < hours:
        if shift_on_at(h) and rng.random() < 0.04:
            length = rng.randint(2, 5)
            peak = rng.uniform(60.0, 120.0)
            for k in range(length):
                if h + k < hours:
                    bursts[h + k] += peak * math.exp(-0.5 * k)
            h += length
        else:
            h += 1

    out = []
    t_return = 28.0
    since_clean = 0.0
    for h in range(hours):
        shift_on = shift_on_at(h)
        ua = ua_clean * math.exp(-fouling_rate * since_clean)
        since_clean = 0.0 if since_clean >= clean_interval else since_clean + dt

        t_amb = 18.0 + 6.0 * math.sin(2 * math.pi * (h % 24) / 24 - 2.0) + 0.01 * h + rng.gauss(0, 0.4)
        q = (q_on if shift_on else q_off) + bursts[h] + rng.gauss(0, 5.0)
        pump_power = 0.35 * q + rng.gauss(0, 4.0)

        t_return = t_return + dt / C * (q - ua * (t_return - t_amb))
        out.append((h, t_return, q, t_amb, pump_power, 1.0 if shift_on else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=int, default=720)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = simulate(args.hours, args.seed)
    start = datetime.datetime(2026, 1, 5)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "t_return", "q_load", "t_amb", "pump_power", "shift_on"])
        for h, t_return, q, t_amb, pump_power, shift_on in rows:
            ts = (start + datetime.timedelta(hours=h)).isoformat()
            w.writerow([ts, f"{t_return:.4f}", f"{q:.4f}", f"{t_amb:.4f}", f"{pump_power:.4f}", f"{shift_on:.0f}"])
    trip_at = next((h for h, t, *_ in rows if t > 45.0), None)
    print(f"wrote {len(rows)} hourly bars to {args.out}")
    print(f"return temp range: {min(r[1] for r in rows):.1f}..{max(r[1] for r in rows):.1f} C" + (f"; first crosses 45C at hour {trip_at}" if trip_at else "; never crosses 45C"))


if __name__ == "__main__":
    main()
