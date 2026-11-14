# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Simulating Brains and Bodies".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: neural_simulation.<name> -->`
markers; the functions with a leading underscore supply the numbers of the chapter's figures, so a picture and the table beside it
cannot disagree.

* a four-row connectome read into a signed sparse graph, with the accounting of every row;
* a leaky integrate-and-fire neuron under forward Euler and under the exact step, its spike times, its rates and the size of the
  grid it is sampled on;
* a conductance membrane under both steps, and the effective time constant of a synaptic filter that decays by 1 - a per tick;
* a small spiking network on that graph, run with the gather rule of a compressed-sparse-column graph, with and without a synapse
  floor, and the work and bytes of a tick against a dense matrix;
* a harmonic oscillator under four integrators and a ball on a penalty contact under semi-implicit Euler: stability limit and
  energy drift;
* a toy nervous system closed around a one-joint body, intact, with its sensory channel cut and with that channel connected but
  too weak to fire;
* an exhaustive structural control: all 180 ways of reassigning the sources of a small sensorimotor graph.

Arithmetic is the language's 64-bit floating point (library `exp` and `log`, correctly rounded `+ - * /` and `sqrt`) unless a block
says otherwise. Sums run left to right in index order. There is no randomness in this module: every table is a deterministic
function of the numbers printed beside it. Spiking runs are discontinuous, so what is printed from them are the integers (tick
numbers and spike counts) and the smallest distance of any threshold decision from the threshold, which says how far from a tie
the run stayed; membrane values are given to the digits that survive a change of summation order. A FLOP is one multiply or one add,
so a multiply-add is two.

Swedish Embedded AB implements solutions for spiking and connectome-scale simulation coupled to physical models. If your team needs
expertise in simulated nervous systems, sparse neural runtimes and closed-loop control of simulated bodies, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
from itertools import permutations

import images as im

_table = im._table

# ------------------------------------------------------------------------------------------------ a connectome from four rows

NEURONS_CSV = """Root ID,Super Class,Primary Cell Type,Pred NT,Pred conf,Verified NT
1,descending,DNp01,ACH,0.51,
2,ventral_nerve_cord_intrinsic,IN01,GABA,0.93,
3,motor,MN01,GLUT,0.40,ACH
4,central_brain_intrinsic,"CB,01",,,
"""

CONNECTIONS_CSV = """pre_root_id,post_root_id,neuropil,syn_count,nt_type
1,3,LegNp_T1,7,ACH
1,3,IntTct,5,ACH
2,3,LegNp_T1,4,GABA
1,2,CV,9,ACH
"""

SIGN_OF = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "HA": -1.0}


def _split(line):
    """One CSV line into fields: commas split, except inside double quotes."""
    out, cur, quoted = [], "", False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == "," and not quoted:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _read_connectome():
    """Neurons (in file order), the aggregated edges sorted by (post, pre), and the row accounting."""
    lines = NEURONS_CSV.strip().split("\n")
    head = _split(lines[0])
    neurons = []
    for ln in lines[1:]:
        f = dict(zip(head, _split(ln)))
        pred, conf, ver = f["Pred NT"], f["Pred conf"], f["Verified NT"]
        best = ver if ver else pred
        neurons.append({
            "id": int(f["Root ID"]), "super": f["Super Class"], "type": f["Primary Cell Type"],
            "pred": pred, "conf": float(conf) if conf else 0.0, "ver": ver, "best": best,
            "sign": SIGN_OF[best] if best else 0.0,
        })
    index = {n["id"]: i for i, n in enumerate(neurons)}
    clines = CONNECTIONS_CSV.strip().split("\n")
    chead = _split(clines[0])
    rows, pairs = [], {}
    for ln in clines[1:]:
        f = dict(zip(chead, _split(ln)))
        pre, post, syn = int(f["pre_root_id"]), int(f["post_root_id"]), int(f["syn_count"])
        rows.append((pre, post, f["neuropil"], syn))
        key = (index[post], index[pre])
        pairs[key] = pairs.get(key, 0) + syn
    edges = [(pre, post, w) for (post, pre), w in sorted(pairs.items())]
    return neurons, rows, edges


def _csc(n, edges):
    """Counting sort of (pre, post, weight) into columns by post; order within a column is input order."""
    counts = [0] * (n + 1)
    for _, post, _ in edges:
        counts[post + 1] += 1
    for i in range(n):
        counts[i + 1] += counts[i]
    indptr = list(counts)
    cursor = list(counts)
    pre, w = [0] * len(edges), [0.0] * len(edges)
    for a, b, wt in edges:
        pre[cursor[b]] = a
        w[cursor[b]] = wt
        cursor[b] += 1
    return indptr, pre, w


def _network(neurons, edges, scale, floor):
    """Signed, scaled, floored edge list: weight = count * sign(pre) * scale, edges under `floor` synapses dropped."""
    out = []
    for pre, post, count in edges:
        if floor > 1 and count < floor:
            continue
        out.append((pre, post, count * neurons[pre]["sign"] * scale))
    return out


def connectome_rows():
    neurons, rows, edges = _read_connectome()
    r1 = [(n["id"], n["super"], n["type"], n["pred"] or "-", f"{n['conf']:.2f}" if n["pred"] else "-", n["ver"] or "-",
           n["best"] or "-", f"{n['sign']:+.0f}") for n in neurons]
    t1 = _table(["row", "Root ID", "Super Class", "Primary Cell Type", "predicted", "confidence", "verified", "believed", "sign"],
                [(i,) + r for i, r in enumerate(r1)], align="rllllrlll")
    t2 = _table(["pre", "post", "neuropil", "syn_count"], rows, align="rrlr")
    ids = [n["id"] for n in neurons]
    t3 = _table(["edge (row index pre -> post)", "Root IDs", "synapses", "sign of pre", "signed, scale 0.3"],
                [(f"{a} -> {b}", f"{ids[a]} -> {ids[b]}", int(w), f"{neurons[a]['sign']:+.0f}", f"{w * neurons[a]['sign'] * 0.3:+.1f}")
                 for a, b, w in edges], align="llrrr")
    kept = sum(r[3] for r in rows)
    ip, pre, w = _csc(len(neurons), [(a, b, c) for a, b, c in edges])
    sg = _csc(len(neurons), _network(neurons, edges, 0.3, 0))
    fl = _csc(len(neurons), _network(neurons, edges, 0.3, 5))
    exc = sum(1 for a, _, _ in edges if neurons[a]["sign"] > 0)
    inh = sum(1 for a, _, _ in edges if neurons[a]["sign"] < 0)
    zero = len(edges) - exc - inh
    t4 = _table(["graph", "indptr", "pre", "w"], [
        ("raw counts", ip, pre, [int(x) for x in w]),
        ("signed, scale 0.3", sg[0], sg[1], [round(x, 1) for x in sg[2]]),
        ("signed, scale 0.3, floor 5", fl[0], fl[1], [round(x, 1) for x in fl[2]]),
    ], align="llll")
    return (f"The two files as printed above, read by splitting on commas outside double quotes. Neurons keep file order (row index "
            f"0 to 3). The believed transmitter is the verified one when there is one, otherwise the predicted one, and the sign is "
            f"+1 for ACH, -1 for GABA and GLUT, 0 for none.\n\n{t1}\n\nThe connection rows ({len(rows)} rows, {kept} synapses) "
            f"as read:\n\n{t2}\n\nSummed per (pre, post) pair across neuropils and sorted by (post, pre), they become the edges:\n\n{t3}\n\n"
            f"Accounting: {len(rows)} rows read, {len(rows)} kept, 0 rejected; {len(edges)} distinct edges; {kept} synapses. Sign "
            f"census over edges: {exc} excitatory, {inh} inhibitory, {zero} with no sign. The graph by columns (neuron `i` owns "
            f"`indptr[i]` to `indptr[i+1]`-1):\n\n{t4}")


# ------------------------------------------------------------------------------------------------------ the membrane, stepped

def _forward_euler(v, v_inf, z):
    return v + z * (v_inf - v)


def _exponential(v, v_inf, z):
    return v_inf + (v - v_inf) * math.exp(-z)


def _lif_train(step, v_inf, z, v_th, v_reset, refrac, ticks):
    """Ticks (1-based) at which a LIF neuron fires; at a spike the membrane is reset and sits out `refrac` ticks at v_reset."""
    v, left, fired = v_reset, 0, []
    for t in range(1, ticks + 1):
        if left > 0:
            v, left = v_reset, left - 1
            continue
        v = step(v, v_inf, z)
        if v >= v_th:
            fired.append(t)
            v, left = v_reset, refrac
    return fired


def lif_spike_times():
    tau, h, v_inf, v_th = 10.0, 2.0, 1.5, 1.0
    z = h / tau
    t_star = -tau * math.log(1.0 - v_th / v_inf)
    rows = []
    for k in range(0, 9):
        fe = v_inf * (1.0 - (1.0 - z) ** k)
        ex = v_inf * (1.0 - math.exp(-z * k))
        rows.append((k, f"{k * h:.0f}", f"{fe:.4f}", f"{ex:.4f}", f"{fe - ex:+.4f}"))
    t1 = _table(["tick k", "time (milliseconds)", "forward Euler", "exact step (= the ODE)", "difference"], rows, align="rrrrr")
    first_fe = next(k for k in range(1, 50) if v_inf * (1.0 - (1.0 - z) ** k) >= v_th)
    first_ex = next(k for k in range(1, 50) if v_inf * (1.0 - math.exp(-z * k)) >= v_th)
    a = (1.0 - z) ** first_fe
    train_fe = _lif_train(_forward_euler, v_inf, z, v_th, 0.0, 0, 50)
    train_ex = _lif_train(_exponential, v_inf, z, v_th, 0.0, 0, 50)
    period = t_star
    exact_times = []
    t = period
    while t <= 100.0 + 1e-9:
        exact_times.append(t)
        t += period
    t2 = _table(["integrator", "first spike (milliseconds)", "spikes in 100 milliseconds", "interval between spikes (ticks)",
                 "mean rate (spikes per second)"], [
        ("the ODE, exactly", f"{t_star:.3f}", len(exact_times), f"{period / h:.3f}", f"{1000.0 / period:.1f}"),
        ("exact step on the 2-millisecond grid", f"{first_ex * h:.0f}", len(train_ex), train_ex[1] - train_ex[0],
         f"{1000.0 / ((train_ex[1] - train_ex[0]) * h):.1f}"),
        ("forward Euler on the 2-millisecond grid", f"{first_fe * h:.0f}", len(train_fe), train_fe[1] - train_fe[0],
         f"{1000.0 / ((train_fe[1] - train_fe[0]) * h):.1f}"),
    ], align="lrrrr")
    return (f"Leaky integrate-and-fire neuron, rest 0, reset 0, threshold {v_th}, membrane time constant {tau:.0f} milliseconds, "
            f"constant drive with $R I = {v_inf}$ (so $V_\\infty = {v_inf}$), step $h = {h:.0f}$ milliseconds, $z = h/\\tau_{{\\mathrm{{mb}}}} "
            f"= {z}$, no refractory period, start at $V = 0$.\n\n{t1}\n\nThe ODE crosses the threshold at $t^\\ast = -\\tau\\ln(1 - V_{{\\mathrm{{th}}}}/V_\\infty) = "
            f"{t_star:.3f}$ milliseconds. The exact step first reaches $V \\ge 1$ at tick {first_ex} ({first_ex * h:.0f} milliseconds, "
            f"$V = {v_inf * (1 - math.exp(-z * first_ex)):.4f}$); forward Euler at tick {first_fe} ({first_fe * h:.0f} milliseconds, "
            f"$V = {v_inf * (1 - a):.4f}$).\n\n{t2}")


def lif_grid_sweep():
    tau, v_inf, v_th = 10.0, 1.5, 1.0
    t_star = -tau * math.log(1.0 - v_th / v_inf)
    rows = []
    for h in (0.5, 1.0, 2.0, 5.0, 10.0):
        z = h / tau
        k_ex = next(k for k in range(1, 10000) if v_inf * (1.0 - math.exp(-z * k)) >= v_th)
        k_fe = next(k for k in range(1, 10000) if v_inf * (1.0 - (1.0 - z) ** k) >= v_th)
        rows.append((f"{h:g}", f"{z:g}", f"{k_ex * h:g}", f"{k_ex * h - t_star:+.3f}", f"{k_fe * h:g}", f"{k_fe * h - t_star:+.3f}"))
    return (f"First threshold crossing of the same neuron ($\\tau_{{\\mathrm{{mb}}}} = {tau:.0f}$ milliseconds, $V_\\infty = {v_inf}$, "
            f"threshold {v_th}) on grids of different step $h$. The ODE crosses at {t_star:.3f} milliseconds. The error of the "
            f"exact step is the wait for the next tick, always late by less than $h$; forward Euler's membrane is itself wrong.\n\n"
            + _table(["h (milliseconds)", "z = h / tau", "exact step: spike at", "error", "forward Euler: spike at", "error"], rows,
                     align="rrrrrr"))


def lif_rates():
    tau, h, v_th = 10.0, 2.0, 1.0
    z = h / tau
    rows = []
    for v_inf in (1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 20.0):
        period = -tau * math.log(1.0 - v_th / v_inf)
        k_ex = next(k for k in range(1, 100000) if v_inf * (1.0 - math.exp(-z * k)) >= v_th)
        k_fe = next(k for k in range(1, 100000) if v_inf * (1.0 - (1.0 - z) ** k) >= v_th)
        rows.append((f"{v_inf:g}", f"{1000.0 / period:.1f}", k_ex, f"{1000.0 / (k_ex * h):.1f}", k_fe, f"{1000.0 / (k_fe * h):.1f}"))
    return (f"Firing rate against drive for the neuron of the previous block (rest 0, reset 0, threshold {v_th}, $\\tau_{{\\mathrm{{mb}}}} = "
            f"{tau:.0f}$ milliseconds, no refractory period). The ODE's rate is $1/(\\tau\\ln(V_\\infty/(V_\\infty - 1)))$; on the grid "
            f"$h = {h:.0f}$ milliseconds the interval is a whole number of ticks, the first $k$ with $V_\\infty(1 - e^{{-zk}}) \\ge 1$ for "
            f"the exact step and the first with $V_\\infty(1 - (1-z)^k) \\ge 1$ for forward Euler. Rates in spikes per second.\n\n"
            + _table(["V_inf", "the ODE", "exact step: ticks", "exact step", "forward Euler: ticks", "forward Euler"], rows,
                     align="rrrrrr"))


def crate_gate():
    # The configuration of the neuron test in crates/neuro/tests/it/dynamics.rs: step ratio 1, threshold 0.5, 3 refractory ticks.
    z, v_th = 1.0, 0.5
    fired = _lif_train(_exponential, 1.0, z, v_th, 0.0, 3, 12)
    first_v = _exponential(0.0, 1.0, z)
    fired_fe = _lif_train(_forward_euler, 1.0, z, v_th, 0.0, 3, 12)
    fired_one = _lif_train(_exponential, 1.0, z, v_th, 0.0, 0, 12)
    return (_table(["configuration ($z = 1$, drive 1, threshold 0.5)", "membrane after tick 1", "spike ticks over 12 ticks (counted from 0)"], [
        ("exact step, 3 refractory ticks", f"{first_v:.4f}", [t - 1 for t in fired]),
        ("forward Euler, 3 refractory ticks", f"{_forward_euler(0.0, 1.0, z):.4f}", [t - 1 for t in fired_fe]),
        ("exact step, no refractory period", f"{first_v:.4f}", [t - 1 for t in fired_one]),
    ], align="lrl"))


# -------------------------------------------------------------------------------------------- conductances and synaptic filters

def conductance_stability():
    z, e_exc, steps = 0.2, 2.8, 12
    rows = []
    for g in (0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0):
        bb = 1.0 + g
        v_inf = g * e_exc / bb
        fe = ex = 0.0
        for _ in range(steps):
            fe = fe + z * (g * e_exc - bb * fe)
            ex = v_inf + (ex - v_inf) * math.exp(-z * bb)
        f_fe = 1.0 - z * bb
        f_ex = math.exp(-z * bb)
        verdict = "converges" if abs(f_fe) < 1.0 and f_fe >= 0 else ("converges, alternating" if abs(f_fe) < 1.0 else
                                                                      ("does not converge" if abs(f_fe) == 1.0 else "diverges"))
        rows.append((f"{g:g}", f"{z * bb:.1f}", f"{v_inf:.4f}", f"{f_fe:+.2f}", f"{fe:+.3f}", verdict, f"{f_ex:.4f}", f"{ex:+.4f}"))
    return (f"A membrane with an excitatory conductance $g$ (reversal $E_{{\\mathrm{{exc}}}} = {e_exc}$, rest 0, no other input), "
            f"$z = h/\\tau_{{\\mathrm{{mb}}}} = {z}$, started at $V = 0$ and stepped {steps} times. The equation is "
            f"$\\tau\\,dV/dt = g E_{{\\mathrm{{exc}}}} - (1+g)V$, so the total leak is $b = 1 + g$. Forward Euler multiplies the distance "
            f"to $V_\\infty$ by $1 - zb$ each tick; the exact step by $e^{{-zb}}$.\n\n"
            + _table(["g", "z b", "V_inf", "forward Euler: factor", "after 12 ticks", "forward Euler", "exact step: factor",
                      "after 12 ticks"], rows, align="rrrrrlrr"))


def synaptic_filters():
    h = 2.0
    rows = []
    for a, label in ((0.1, "inhibitory, 20 ms"), (0.2, "10 ms"), (0.4, "excitatory, 5 ms"), (0.5, "4 ms"), (0.8, "2.5 ms")):
        tau = h / a
        keep_e = 1.0 - a
        tau_e = -h / math.log(keep_e)
        keep_x = math.exp(-a)
        total_e = 1.0 / a
        total_x = 1.0 / (1.0 - keep_x)
        rows.append((f"{a:g}", label.replace(" ms", " milliseconds"), f"{keep_e:.4f}", f"{tau_e:.3f}", f"{keep_x:.4f}",
                     f"{total_e:.3f}", f"{total_x:.3f}"))
    return (f"A synaptic current that receives a unit impulse on tick 0 and then decays by a fixed factor per tick, step $h = {h:.0f}$ "
            f"milliseconds. With $a = h/\\tau_{{\\mathrm{{syn}}}}$, one rule keeps the fraction $1 - a$ per tick (the engine's), the other "
            f"$e^{{-a}}$ (the exact decay of a filter of time constant $\\tau_{{\\mathrm{{syn}}}}$). The effective time constant of a keep "
            f"factor $c$ is $-h/\\ln c$ milliseconds; the total charge delivered over all ticks is $\\sum_k c^k = 1/(1-c)$ times the impulse.\n\n"
            + _table(["a", "nominal tau", "keep 1 - a", "effective tau (milliseconds)", "keep exp(-a)", "total charge, 1 - a", "total charge, exp(-a)"],
                     rows, align="rlrrrrr"))


# ------------------------------------------------------------------------------------------------- a spiking net on the sparse graph

LIF = dict(z=0.2, v_rest=0.0, v_reset=0.0, v_th=1.0, v_min=-3.0, e_exc=2.8, e_inh=0.0, r=1.0, refrac=1, a_syn=0.4, a_inh=0.1,
           adapt_decay=0.98, adapt_inc=0.4, std_release=0.85, std_recover=2.0 / 150.0)


def _spiking_run(n, indptr, pre, w, drive, ticks, p=LIF):
    """Run the network with the gather rule. Returns (spike ticks per neuron, smallest |V - threshold| at any decision).

    Per tick, in order: (1) gather: for neuron i, over its incoming edges k, c = w[k] * spike[pre[k]] * pool[pre[k]];
    exc_i accumulates max(c, 0), inh_i accumulates min(c, 0); the two currents then decay by (1 - a_syn) and (1 - a_inh)
    and add the new sums. (2) update: per neuron, adaptation decays, the vesicle pool refills, a refractory neuron is held at
    reset, any other takes the exponential step of the conductance membrane, floored at v_min; at threshold it resets, sits out
    `refrac` ticks, its adaptation grows by adapt_inc and its pool is multiplied by std_release."""
    v = [p["v_rest"]] * n
    refrac, spike, adapt, pool = [0] * n, [0.0] * n, [0.0] * n, [1.0] * n
    cur_e, cur_i = [0.0] * n, [0.0] * n
    fired = [[] for _ in range(n)]
    margin = float("inf")
    for t in range(1, ticks + 1):
        for i in range(n):
            ae = ai = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                c = w[k] * spike[pre[k]] * pool[pre[k]]
                ae += max(c, 0.0)
                ai += min(c, 0.0)
            cur_e[i] = cur_e[i] * (1.0 - p["a_syn"]) + ae
            cur_i[i] = cur_i[i] * (1.0 - p["a_inh"]) + ai
        nxt = [0.0] * n
        for i in range(n):
            wa = adapt[i] * p["adapt_decay"]
            pl = pool[i] + (1.0 - pool[i]) * p["std_recover"]
            if refrac[i] > 0:
                v[i], refrac[i], adapt[i], pool[i] = p["v_reset"], refrac[i] - 1, wa, pl
                continue
            ge, gi = max(cur_e[i], 0.0) * p["r"], max(-cur_i[i], 0.0) * p["r"]
            b = 1.0 + ge + gi
            a = p["v_rest"] + ge * p["e_exc"] + gi * p["e_inh"] + p["r"] * (drive[i] - wa)
            v_inf = a / b
            vi = max(v_inf + (v[i] - v_inf) * math.exp(-p["z"] * b), p["v_min"])
            margin = min(margin, abs(vi - p["v_th"]))
            if vi >= p["v_th"]:
                v[i], refrac[i], nxt[i] = p["v_reset"], p["refrac"], 1.0
                adapt[i], pool[i] = wa + p["adapt_inc"], pl * p["std_release"]
                fired[i].append(t)
            else:
                v[i], adapt[i], pool[i] = vi, wa, pl
        spike = nxt
    return fired, margin


def sparse_network():
    neurons, _, edges = _read_connectome()
    n, ticks, drive_d = len(neurons), 100, 2.0
    names = ["DNp01 (row 0)", "IN01 (row 1)", "MN01 (row 2)", "CB,01 (row 3)"]
    drive = [drive_d, 0.0, 0.0, 0.0]
    out, margins = [], []
    runs = []
    for floor in (0, 5):
        g = _network(neurons, edges, 0.3, floor)
        ip, pre, w = _csc(n, g)
        fired, m = _spiking_run(n, ip, pre, w, drive, ticks)
        runs.append((floor, len(g), fired))
        margins.append(m)
    rows = []
    for i in range(n):
        rows.append((names[i],) + tuple(len(r[2][i]) for r in runs) + tuple(r[2][i][:8] for r in runs))
    t = _table(["neuron", "spikes, all edges", "spikes, floor 5", "first spike ticks, all edges", "first spike ticks, floor 5"], rows,
               align="lrrll")
    p = LIF
    return (f"The graph of the first block, signed and scaled by 0.3 (weights $+2.7$, $+3.6$ and $-1.2$ for the rows 0 to 1, 0 to 2 "
            f"and 1 to 2). Neuron 0 receives the constant drive {drive_d} on every tick (units of $R = 1$); no other neuron is driven. "
            f"Parameters: $z = {p['z']}$, rest and reset 0, threshold 1, floor $V_{{\\mathrm{{min}}}} = {p['v_min']}$, "
            f"$E_{{\\mathrm{{exc}}}} = {p['e_exc']}$, $E_{{\\mathrm{{inh}}}} = {p['e_inh']}$, 1 refractory tick, excitatory keep factor "
            f"{1 - p['a_syn']:.1f}, inhibitory keep factor {1 - p['a_inh']:.1f}, adaptation decay {p['adapt_decay']} and increment "
            f"{p['adapt_inc']}, pool release {p['std_release']} and refill {p['std_recover']:.6f} per tick. {ticks} ticks, "
            f"all state at rest at the start (every synaptic pool starts full, at 1.0; the tick numbers below count from 1, so the first tick is 1, unlike the block of the crate's own test earlier in the chapter, which counts from 0). The floor of 5 synapses removes the edge 1 to 2 (4 synapses) and nothing else. "
            f"Smallest distance of any threshold decision from the threshold, over both runs: "
            f"{min(margins):.1e}.\n\n{t}")


def sparse_cost():
    neurons, _, edges = _read_connectome()
    n, e = len(neurons), len(edges)
    d_flop, d_byte = 2 * n * n, 4 * n * n + 4 * n
    s_flop, s_byte = 2 * e, 12 * e + 4 * (n + 1) + 4 * n
    t1 = _table(["representation", "multiply-adds per tick", "FLOP per tick", "bytes read per tick"], [
        ("dense $n \\times n$ matrix", n * n, d_flop, d_byte),
        ("columns by post (gather over all edges)", e, s_flop, s_byte),
    ], align="lrrr")
    rows = []
    for n_big, e_big in ((1000, 20_000), (23_665, 1_370_000), (23_665, 5_310_000), (100_000, 5_000_000)):
        rows.append((f"{n_big:,}", f"{e_big:,}", f"{e_big / n_big:.1f}", f"{n_big * n_big / e_big:.0f}", f"{2 * e_big / 1e6:.2f}",
                     f"{(12 * e_big + 8 * n_big + 4) / 1e6:.2f}", f"{2 * n_big * n_big / 1e9:.3f}", f"{(4 * n_big * n_big) / 1e9:.2f}"))
    t2 = _table(["n", "edges E", "mean in-degree", "dense / sparse work", "sparse MFLOP", "sparse MB", "dense GFLOP", "dense GB"], rows,
                align="rrrrrrrr")
    t3 = _table(["fraction of neurons that fired", "edges read by an event-driven scatter, as a fraction of the gather's"],
                [(f"{f:.0%}", f"{f:.2f}") for f in (0.01, 0.05, 0.2, 1.0)], align="rr")
    return (f"Work and bytes of one tick of synaptic propagation for $n$ neurons and $E$ edges. A multiply-add is 2 FLOP. The dense "
            f"layout stores every one of the $n^2$ weights as 4 bytes and reads the $n$-entry spike vector once; the gather stores "
            f"per edge a 4-byte weight and a 4-byte source index and reads one 4-byte spike value through that index (12 bytes per edge), plus "
            f"the column starts ($4(n+1)$ bytes) and the spike vector itself ($4n$ bytes).\n\n"
            f"The four-neuron graph ($n = {n}$, $E = {e}$):\n\n{t1}\n\n"
            f"The sizes below are chosen, not measured, except that the second and third rows are the cord sizes quoted in the header "
            f"of the engine's gather kernel (23,665 neurons and 1.37 million edges at the synapse floor of 5, 5.31 million before it); "
            f"the first and last are round numbers.\n\n{t2}\n\n"
            f"If every neuron fired a fraction $p$ of the time and out-degrees were equal, an event-driven scatter would read $pE$ edges "
            f"against the gather's $E$:\n\n{t3}")


# ------------------------------------------------------------------------------------------------------------ the body, stepped

def _oscillator(method, h, steps, omega=1.0):
    x, v = 1.0, 0.0
    e0 = 0.5 * v * v + 0.5 * omega * omega * x * x
    lo = hi = 1.0
    for _ in range(steps):
        if method == "explicit":
            x, v = x + h * v, v - h * omega * omega * x
        elif method == "semi":
            v = v - h * omega * omega * x
            x = x + h * v
        elif method == "implicit":
            d = 1.0 + h * h * omega * omega
            x, v = (x + h * v) / d, (v - h * omega * omega * x) / d
        else:
            raise ValueError(method)
        e = (0.5 * v * v + 0.5 * omega * omega * x * x) / e0
        lo, hi = min(lo, e), max(hi, e)
    return lo, hi, e


def _growth(method, h, omega=1.0):
    """Modulus of the larger eigenvalue of the one-step matrix of the method on x'' = -omega^2 x."""
    q = h * omega
    if method == "explicit":
        return math.sqrt(1.0 + q * q)
    if method == "implicit":
        return 1.0 / math.sqrt(1.0 + q * q)
    tr = 2.0 - q * q
    if abs(tr) <= 2.0:
        return 1.0
    return (abs(tr) + math.sqrt(tr * tr - 4.0)) / 2.0


def oscillator_stability():
    rows = []
    steps = 100
    for h in (0.05, 0.5, 1.0, 1.9, 2.0, 2.1):
        cells = []
        for m in ("explicit", "semi", "implicit"):
            lo, hi, last = _oscillator(m, h, steps)
            g = _growth(m, h)
            if m == "semi" and g > 1.0 + 1e-12:
                cells.append(f"amplitude times {g:.3f}")
            elif m == "semi" and abs(g - 1.0) < 1e-12 and h >= 2.0:
                cells.append("edge: amplitude grows linearly")
            elif m == "semi":
                cells.append(f"{lo:.3f} to {hi:.3f}")
            elif m == "explicit":
                cells.append(f"amplitude times {g:.3f}")
            else:
                cells.append(f"amplitude times {g:.3f}")
        rows.append((f"{h:g}",) + tuple(cells))
    return (f"The unit oscillator $x'' = -x$ ($\\omega = 1$, period $2\\pi \\approx 6.283$), started at $x = 1$, $v = 0$, so that its "
            f"energy $E = \\frac12 v^2 + \\frac12 x^2$ is $\\frac12$. Three one-step rules, each applied with the old values on the right: "
            f"explicit Euler $x \\leftarrow x + hv$, $v \\leftarrow v - hx$; semi-implicit Euler $v \\leftarrow v - hx$, then "
            f"$x \\leftarrow x + hv$ with the new $v$; implicit Euler, which solves $x_{{k+1}} = x_k + hv_{{k+1}}$, "
            f"$v_{{k+1}} = v_k - hx_{{k+1}}$ for the new values. After 100 steps the table gives what happened to $E/E_0$: for a "
            f"rule that keeps it bounded, the smallest and largest value it took over the run; for a rule that grows or decays it "
            f"every step, the factor per step of the amplitude (the modulus of the eigenvalue of the one-step matrix, computed in closed form; the energy changes by its square).\n\n"
            + _table(["h", "explicit Euler", "semi-implicit Euler", "implicit Euler"], rows, align="rlll"))


def _ball(h, t_end, k=1.0e4, g=9.81, m=1.0, z0=1.0):
    """A point mass dropped from z0 onto a floor at z = 0 that pushes back with a spring (penalty contact, no damping), stepped with
    semi-implicit Euler. Returns the starting energy, the largest |E - E0| at any step, the deepest penetration and the height of
    each rebound apex (the first grid point after the upward velocity has gone from positive to zero or below)."""
    z, v = z0, 0.0
    e0 = m * g * z + 0.5 * m * v * v
    emax = pen_max = 0.0
    touched = False
    apexes = []
    prev_v = 0.0
    for _ in range(int(round(t_end / h))):
        a = -g + (k / m) * max(0.0, -z)
        v = v + h * a
        z = z + h * v
        pen = max(0.0, -z)
        pen_max = max(pen_max, pen)
        if touched and z > 0.0 and prev_v > 0.0 and v <= 0.0:
            apexes.append(z)
        if z < 0.0:
            touched = True
        prev_v = v
        emax = max(emax, abs(m * g * z + 0.5 * m * v * v + 0.5 * k * pen * pen - e0))
    return e0, emax, pen_max, apexes


def _resting_contact(q, k=1.0e4, g=9.81, m=1.0, eps=1.0e-5, steps=3000):
    """A body at rest on the penalty floor, nudged by eps above its equilibrium penetration, stepped at h = q / omega. Returns the
    step at which it first leaves the floor (or None) and the largest |z - z_eq| / eps over the steps taken."""
    omega = math.sqrt(k / m)
    h = q / omega
    z_eq = -m * g / k
    z, v = z_eq + eps, 0.0
    worst = 0.0
    for i in range(1, steps + 1):
        v = v + h * (-g + (k / m) * max(0.0, -z))
        z = z + h * v
        worst = max(worst, abs(z - z_eq) / eps)
        if z >= 0.0:
            return i, worst
    return None, worst


def bouncing_ball():
    k, g, m, z0, t_end = 1.0e4, 9.81, 1.0, 1.0, 3.0
    omega = math.sqrt(k / m)
    rows = []
    for h in (0.01, 0.005, 0.002, 0.001, 0.00025):
        e0, e_max, pen, apexes = _ball(h, t_end)
        rows.append((f"{h:g}", f"{h * omega:.3f}", f"{math.pi / (omega * h):.1f}", f"{pen * 1000:.2f}", f"{e_max / e0:.2e}")
                    + tuple(f"{a:.4f}" for a in apexes[:3]))
    v_hit = math.sqrt(2 * g * z0)
    t1 = _table(["h (seconds)", "h * omega", "steps per contact", "deepest penetration (millimetres)", "largest abs(E - E0) / E0",
                 "apex 1", "apex 2", "apex 3"], rows, align="rrrrrrrr")
    z_eq = m * g / k
    rows = []
    for q in (0.5, 1.0, 1.5, 1.9, 1.99, 2.0, 2.01, 2.1, 2.5):
        lift, worst = _resting_contact(q)
        tr = 2.0 - q * q
        lam = 1.0 if abs(tr) <= 2.0 else (abs(tr) + math.sqrt(tr * tr - 4.0)) / 2.0
        rows.append((f"{q:g}", f"{lam:.3f}", f"{worst:.2f}" if lift is None else f"{worst:.1f}",
                     "stays down for 3000 steps" if lift is None else f"leaves the floor at step {lift}"))
    t2 = _table(["h * omega", "growth per step", "largest abs(z - z_eq) / epsilon", "outcome"], rows, align="rrrl")
    return (f"A body of mass {m:g} dropped from height {z0:g} onto a floor at $z = 0$ with gravity $g = {g}$. Contact is a spring: while "
            f"$z < 0$ the floor pushes up with the force $k\\,(-z)$, $k/m = {k:g}$ per second squared, so $\\omega = \\sqrt{{k/m}} = {omega:g}$ radians "
            f"per second; there is no damping, so the ideal bounce is perfectly elastic and returns to the height {z0:g}. Semi-implicit Euler: "
            f"$v \\leftarrow v + h a(z)$, then $z \\leftarrow z + h v$ with the new $v$. The ball reaches the floor at speed "
            f"$\\sqrt{{2gz_0}} = {v_hit:.4f}$, penetrates about $v/\\omega \\approx {v_hit / omega * 1000:.2f}$ millimetres plus the {z_eq * 1000:.2f} millimetres by which gravity alone sinks it, and one contact lasts "
            f"$\\pi/\\omega = {math.pi / omega * 1000:.2f}$ milliseconds. The energy "
            f"$E = mgz + \\frac12 mv^2 + \\frac12 k\\max(0, -z)^2$ starts at {m * g * z0:.2f}. {t_end:g} seconds are simulated, which holds "
            f"three contacts; an apex is the height of the first grid point where the upward velocity has run out.\n\n{t1}\n\n"
            f"The stability limit $h\\omega = 2$ needs a contact that lasts. The same body resting on the floor, at its equilibrium "
            f"penetration $z_{{\\mathrm{{eq}}}} = -mg/k = -{z_eq * 1000:.2f}$ millimetres, nudged by $\\epsilon = 0.01$ millimetre "
            f"($10^{{-5}}$ metres) at zero velocity, and stepped at $h = (h\\omega)/\\omega$ for 3000 steps (the growth per step is the modulus of the "
            f"larger eigenvalue of the one-step matrix, 1 when the pair is on the unit circle):\n\n{t2}")


# --------------------------------------------------------------------------------------------------------------- the closed loop

NP = 4  # motor neurons per muscle pool

LOOP = dict(i_d=2.0, w_de=6.0, w_pf=6.0, torque_gain=20.0, stiffness=4.0, damping=2.0, inertia=1.0, act_decay=0.8, act_gain=0.05,
            substeps=20, h=1.0e-4, ticks=1500)


def _closed_loop(angle_gain=10.0, proprioception=True, p=LIF, **kw):
    q = dict(LOOP, **kw)
    n = 2 + 2 * NP
    edges = [(0, 2 + j, q["w_de"]) for j in range(NP)] + [(1, 2 + NP + j, q["w_pf"]) for j in range(NP)]
    ip, pre, w = _csc(n, edges)
    polarity = [0.0, 0.0] + [1.0] * NP + [-1.0] * NP
    v = [0.0] * n
    refrac, spike, adapt, pool = [0] * n, [0.0] * n, [0.0] * n, [1.0] * n
    ce, ci = [0.0] * n, [0.0] * n
    angle, rate, act = 0.0, 0.0, 0.0
    counts = [0] * n
    peak_angle, peak_current, margin = 0.0, 0.0, float("inf")
    angles = []
    for _ in range(q["ticks"]):
        drive = [q["i_d"], angle_gain * angle if proprioception else 0.0] + [0.0] * (2 * NP)
        peak_current = max(peak_current, drive[1])
        for i in range(n):
            ae = ai = 0.0
            for k in range(ip[i], ip[i + 1]):
                c = w[k] * spike[pre[k]] * pool[pre[k]]
                ae += max(c, 0.0)
                ai += min(c, 0.0)
            ce[i] = ce[i] * (1.0 - p["a_syn"]) + ae
            ci[i] = ci[i] * (1.0 - p["a_inh"]) + ai
        nxt = [0.0] * n
        for i in range(n):
            wa = adapt[i] * p["adapt_decay"]
            pl = pool[i] + (1.0 - pool[i]) * p["std_recover"]
            if refrac[i] > 0:
                v[i], refrac[i], adapt[i], pool[i] = p["v_reset"], refrac[i] - 1, wa, pl
                continue
            ge, gi = max(ce[i], 0.0), max(-ci[i], 0.0)
            b = 1.0 + ge + gi
            a = ge * p["e_exc"] + gi * p["e_inh"] + (drive[i] - wa)
            v_inf = a / b
            vi = max(v_inf + (v[i] - v_inf) * math.exp(-p["z"] * b), p["v_min"])
            margin = min(margin, abs(vi - p["v_th"]))
            if vi >= p["v_th"]:
                v[i], refrac[i], nxt[i] = 0.0, p["refrac"], 1.0
                adapt[i], pool[i] = wa + p["adapt_inc"], pl * p["std_release"]
                counts[i] += 1
            else:
                v[i], adapt[i], pool[i] = vi, wa, pl
        spike = nxt
        act *= q["act_decay"]
        for i in range(2, n):
            if spike[i] > 0.5:
                act += q["act_gain"] * polarity[i]
        ctrl = max(-1.0, min(1.0, act))
        for _ in range(q["substeps"]):
            acc = (q["torque_gain"] * ctrl - q["stiffness"] * angle - q["damping"] * rate) / q["inertia"]
            rate += q["h"] * acc
            angle += q["h"] * rate
        peak_angle = max(peak_angle, angle)
        angles.append(angle)
    return dict(counts=counts, peak=peak_angle, final=angle, peak_current=peak_current, margin=margin, angles=angles)


def closed_loop():
    runs = [("intact loop, gain 10", _closed_loop()),
            ("proprioception cut", _closed_loop(proprioception=False)),
            ("sensor gain 2, connected", _closed_loop(angle_gain=2.0))]
    rows = []
    for name, r in runs:
        c = r["counts"]
        rows.append((name, c[0], c[1], c[2], c[2 + NP], f"{r['peak_current']:.3f}", f"{r['peak']:.4f}", f"{r['final']:.4f}"))
    t = _table(["run", "command spikes", "sensor spikes", "extensor spikes (each)", "flexor spikes (each)",
                "largest sensor current", "peak angle (radians)", "angle at the end (radians)"], rows, align="lrrrrrrr")
    same = runs[1][1]["angles"] == runs[2][1]["angles"]
    worst = max(abs(a - b) for a, b in zip(runs[0][1]["angles"], runs[1][1]["angles"]))
    margin = min(r["margin"] for _, r in runs)
    q = LOOP
    p = LIF
    return (f"A one-joint body and a ten-neuron nervous system, {q['ticks']} control ticks of 2 milliseconds (3 seconds). Neurons: 0 the "
            f"command cell, driven with the constant current {q['i_d']}; 1 the proprioceptor, driven with the current "
            f"`angle_gain` times the joint angle; 2 to {1 + NP} an extensor pool of {NP} motor neurons, each excited by neuron 0 with "
            f"weight {q['w_de']}; {2 + NP} to {1 + 2 * NP} a flexor pool of {NP}, each excited by neuron 1 with weight {q['w_pf']}. "
            f"Membrane and synapse parameters as in the sparse-network block. Each control tick: the angle fixes the sensor current, one "
            f"neural tick runs, the activation is multiplied by {q['act_decay']} and then changed by {q['act_gain']} for each "
            f"extensor spike (plus) and each flexor spike (minus), the command is the activation clamped to $[-1, 1]$, and "
            f"{q['substeps']} physics steps of $h = {q['h']:g}$ seconds integrate $I\\ddot q = {q['torque_gain']:g}\\,u - {q['stiffness']:g}q "
            f"- {q['damping']:g}\\dot q$ ($I = {q['inertia']:g}$) with semi-implicit Euler ($\\dot q$ first, then $q$), holding $u$ "
            f"constant. Start: $q = \\dot q = 0$, all neurons at rest. Smallest distance of any threshold decision from the "
            f"threshold, over the three runs: {margin:.1e}.\n\n{t}\n\n"
            f"The cut run and the weak-sensor run have identical angle sequences at every tick: {'yes' if same else 'no'}. The intact run "
            f"differs from the cut run by at most {worst:.4f} radians at a tick.")


# --------------------------------------------------------------------------------------------------------- the structural control

def lateralisation():
    # slot k (post, sign): 0 IL(+), 1 IR(+), 2 ML(+), 3 ML(-), 4 MR(+), 5 MR(-). Neurons: 0 SL, 1 SR, 2 IL, 3 IR, 4 ML, 5 MR.
    posts = [2, 3, 4, 4, 5, 5]
    wts = [6.0, 6.0, 6.0, -3.0, 6.0, -3.0]
    real = (0, 1, 2, 3, 3, 2)
    names = ["SL", "SR", "IL", "IR", "ML", "MR"]
    drive = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def run(pre):
        edges = [(pre[k], posts[k], wts[k]) for k in range(6)]
        ip, pr, w = _csc(6, edges)
        return _spiking_run(6, ip, pr, w, drive, 100)

    seen, tally, margin = set(), {"left": 0, "right": 0, "mixed": 0, "silent": 0}, float("inf")
    for perm in permutations(range(6)):
        pre = tuple(real[j] for j in perm)
        if pre in seen:
            continue
        seen.add(pre)
        fired, m = run(pre)
        margin = min(margin, m)
        left, right = len(fired[4]), len(fired[5])
        if left + right == 0:
            tally["silent"] += 1
        else:
            idx = (left - right) / (left + right)
            tally["left" if idx > 0.5 else ("right" if idx < -0.5 else "mixed")] += 1
    fired, m = run(real)
    margin = min(margin, m)
    total = len(seen)
    rows = [(k, v, f"{100.0 * v / total:.1f} percent") for k, v in tally.items()]
    t = _table(["motor response of the shuffled graph", "wirings", "share"], rows, align="lrr")
    return (f"Six neurons: 0 SL and 1 SR (sensors), 2 IL and 3 IR (interneurons), 4 ML and 5 MR (motor, left and right). The graph has "
            f"six edges in six column slots, each slot with a fixed weight and a fixed postsynaptic neuron: slot 0 is the input of IL "
            f"(+6), slot 1 of IR (+6), slot 2 of ML (+6), slot 3 of ML (-3), slot 4 of MR (+6) and slot 5 of MR (-3). The real wiring puts "
            f"the sources {', '.join(names[j] for j in real)} in slots 0 to 5: each side's sensor excites its interneuron, which excites "
            f"its own motor neuron and inhibits the other. Neuron 0 (SL) alone is driven, with current 3.0, for 100 ticks, with the "
            f"neuron parameters of the sparse-network block. The real wiring gives ML {len(fired[4])} spikes and MR {len(fired[5])}. A "
            f"shuffle assigns the six sources to the six slots in every possible way ({total} distinct assignments, since two sources "
            f"appear twice), keeping every neuron's in-degree and the multiset of weights. Response of ML against MR: lateralization "
            f"$(L - R)/(L + R)$ above $+0.5$ is left, below $-0.5$ right, in between mixed; no spike on either is silent. "
            f"Smallest distance of any threshold decision from the threshold, over all runs: {margin:.1e}.\n\n{t}")


# --------------------------------------------------------------------------------------------------------- what each step discards

def discretisation_ledger():
    tau, h, v_inf, v_th = 10.0, 2.0, 1.5, 1.0
    z = h / tau
    t_star = -tau * math.log(1.0 - v_th / v_inf)
    k_ex = next(k for k in range(1, 100) if v_inf * (1.0 - math.exp(-z * k)) >= v_th)
    k_fe = next(k for k in range(1, 100) if v_inf * (1.0 - (1.0 - z) ** k) >= v_th)
    big = 20.0
    period_big = -tau * math.log(1.0 - v_th / big)
    eff = lambda a: -h / math.log(1.0 - a)
    neurons, _, edges = _read_connectome()
    third = []
    for floor in (0, 5):
        g = _network(neurons, edges, 0.3, floor)
        ip, pre, w = _csc(len(neurons), g)
        fired, _ = _spiking_run(len(neurons), ip, pre, w, [2.0, 0.0, 0.0, 0.0], 100)
        third.append(fired[2][2])
    apex = _ball(0.01, 3.0)[3][0]
    beat_hz, dt_phys, dt_ctrl = 218.0, 1.0e-4, 2.0e-3
    rows = [
        ("membrane integrator (forward Euler instead of the exact step)",
         f"first spike at {k_fe * h:g} milliseconds instead of {k_ex * h:g}; the ODE's {t_star:.3f}"),
        ("the tick grid (a spike can only occur on a tick)",
         f"{1000.0 / period_big:.0f} spikes per second at $V_\\infty = {big:g}$ becomes {1000.0 / h:.0f}, one per tick"),
        ("conductance with forward Euler",
         f"factor {1.0 - z * (1.0 + 12.0):+.1f} per tick at $g = 12$: diverges, where the exact step multiplies by "
         f"{math.exp(-z * 13.0):.3f}"),
        ("synaptic filter that keeps $1 - a$ per tick",
         f"a 20-millisecond filter behaves as {eff(0.1):.2f} milliseconds, a 5-millisecond one as {eff(0.4):.2f}"),
        ("synapse floor of 5 synapses",
         f"the third spike of the motor neuron at tick {third[1]} instead of {third[0]} when the 4-synapse inhibitory edge is dropped"),
        ("contact as a spring, $h\\omega = 1$",
         f"the first rebound reaches {apex:.4f} instead of 1"),
        ("one control tick between sensing and acting",
         f"a delay of {dt_ctrl * 1000:g} milliseconds; the command is held for {int(round(dt_ctrl / dt_phys))} physics steps"),
        ("the wingbeat against the clocks",
         f"{1.0 / (beat_hz * dt_phys):.1f} physics steps and {1.0 / (beat_hz * dt_ctrl):.2f} control ticks per beat of {beat_hz:g} per second"),
    ]
    return _table(["discretization", "what it does in the examples above"], rows, align="ll")


# ---------------------------------------------------------------------------------------------------------- figure data helpers

def _membrane_curves():
    """(ts, exact curve) and the grid points of the exact step and of forward Euler for the first LIF example."""
    tau, h, v_inf = 10.0, 2.0, 1.5
    z = h / tau
    ts = [i * 0.25 for i in range(0, 121)]
    exact = [v_inf * (1.0 - math.exp(-t / tau)) for t in ts]
    ks = list(range(0, 16))
    fe = [v_inf * (1.0 - (1.0 - z) ** k) for k in ks]
    ex = [v_inf * (1.0 - math.exp(-z * k)) for k in ks]
    return ts, exact, [k * h for k in ks], ex, fe


def _ball_energy(h, t_end=3.0):
    """Energy of the bouncing ball over time under semi-implicit Euler, relative to the starting energy (the figure's data)."""
    k, g, m, z, v = 1.0e4, 9.81, 1.0, 1.0, 0.0
    e0 = m * g * z
    ts, es = [0.0], [0.0]
    steps = int(round(t_end / h))
    for i in range(steps):
        pen = max(0.0, -z)
        a = -g + (k / m) * pen
        v = v + h * a
        z = z + h * v
        pen = max(0.0, -z)
        ts.append((i + 1) * h)
        es.append((m * g * z + 0.5 * m * v * v + 0.5 * k * pen * pen - e0) / e0)
    return ts, es


def _loop_angles():
    return {name: _closed_loop(**kw)["angles"] for name, kw in
            (("intact", {}), ("cut", {"proprioception": False}))}
