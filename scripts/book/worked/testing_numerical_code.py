# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Testing Numerical Code".

Plain Python 3, standard library only, written from the chapter's own
description and independent of any engine code. Each public function returns a
markdown string that the chapter embeds between
`<!-- worked: testing_numerical_code.<name> -->` markers.

The examples are small programs under test together with deliberately wrong
copies of them (mutants), and tests that are run against both: a causal
attention head on instances of several sizes, six measures of difference on six
kinds of error, a parity gate in seven states of disrepair, and a conformance
suite of five contracts over five implementations of a weighted cross-entropy.
The metrics are the ones of "Importing Pretrained Models and Proving Parity"
and are imported from that chapter's module; the rounding to 32 bits is the
one of "Proving Gradients Correct".

Swedish Embedded AB implements solutions for test suites whose green result
can be trusted. If your team needs expertise in testing numerical and machine
learning code, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng
from gradient_checking import r32
from importing_models import cosine, max_abs, rel_l2

IGNORE = -1                 # the target of a position that is not supervised
SAME = 1e-9                 # an output differing from the reference by at most this much counts as the same output


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _sci(v, d=2):
    if v == 0:
        return "0"
    return f"{v:.{d}e}".replace("e-0", "e-").replace("e+0", "e+")


def _seed(name):
    s = bookrng.seed_of(name)
    return f"the book's SplitMix64 with seed {s:#018x} ({s})"


def _lse(z):
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


# ------------------------------------------------------------------------------------------------ degenerate instances

NAME_INSTANCES = "testing_numerical_code.degenerate_instances"

# (label, T, d_k, d_v, identical_rows): the instances of the attention table, in the order they are drawn.
INSTANCES = [
    ("one token", 1, 4, 3, False),
    ("head width one", 3, 1, 2, False),
    ("identical tokens", 3, 4, 3, True),
    ("square sizes", 3, 3, 3, False),
    ("distinct sizes", 4, 5, 3, False),
]

MUTANTS = [                         # (key, column header); m = 1 .. 6 of the chapter's attention_variant
    ("scale", "1: no scale"),
    ("axis", "2: column softmax"),
    ("nomask", "3: no mask"),
    ("late", "4: late mask"),
    ("keys", "5: V from K"),
    ("transpose", "6: K Q^T"),
]


def _draw_matrix(g, rows, cols, identical):
    if identical:
        row = [g.normal() for _ in range(cols)]
        return [list(row) for _ in range(rows)]
    return [[g.normal() for _ in range(cols)] for _ in range(rows)]


def draw_instances():
    """Q, K, V of every instance, drawn from one generator in the order of INSTANCES."""
    g = bookrng.generator(NAME_INSTANCES)
    out = []
    for label, T, dk, dv, same in INSTANCES:
        Q = _draw_matrix(g, T, dk, same)
        K = _draw_matrix(g, T, dk, same)
        V = _draw_matrix(g, T, dv, same)
        out.append((label, T, dk, dv, Q, K, V))
    return out


def causal_attention(Q, K, V, mutant=None):
    """One causal attention head, Y = softmax(Q K^T / sqrt(d_k) + mask) V, or one of its mutants.

    Returns None when the mutant's product does not exist for these shapes (a shape error)."""
    T, dk = len(Q), len(Q[0])
    vals = K if mutant == "keys" else V
    if mutant == "keys" and len(V[0]) != dk:
        return None
    scale = 1.0 if mutant == "scale" else 1.0 / math.sqrt(dk)

    def score(i, j):
        a, b = (K[i], Q[j]) if mutant == "transpose" else (Q[i], K[j])
        return sum(a[c] * b[c] for c in range(dk)) * scale

    def allowed(i, j):
        if mutant == "nomask":
            return True
        if mutant == "late":
            return j <= i + 1
        return j <= i

    P = [[0.0] * T for _ in range(T)]
    if mutant == "axis":
        # each column j is normalized over the queries i that may see key j
        for j in range(T):
            rows = [i for i in range(T) if allowed(i, j)]
            p = _softmax([score(i, j) for i in rows])
            for i, v in zip(rows, p):
                P[i][j] = v
    else:
        for i in range(T):
            cols = [j for j in range(T) if allowed(i, j)]
            p = _softmax([score(i, j) for j in cols])
            for j, v in zip(cols, p):
                P[i][j] = v
    dv = len(vals[0])
    return [[sum(P[i][j] * vals[j][c] for j in range(T)) for c in range(dv)] for i in range(T)]


def instance_verdicts():
    """For every instance and mutant: None (shape error) or the largest |difference| from the reference."""
    rows = []
    for label, T, dk, dv, Q, K, V in draw_instances():
        ref = causal_attention(Q, K, V)
        diffs = []
        for key, _ in MUTANTS:
            y = causal_attention(Q, K, V, key)
            diffs.append(None if y is None else max_abs(y, ref))
        rows.append((label, T, dk, dv, diffs))
    return rows


def _verdict(d):
    if d is None:
        return "shape error"
    if d <= SAME:
        return "same output"
    return f"caught, {d:#.2g}"


def degenerate_instances():
    head = (f"One causal attention head, Y = softmax(Q K^T / sqrt(d_k) + M) V, with M = 0 where key j <= query i and minus "
            f"infinity elsewhere (rows and columns numbered from 0), the softmax taken along each row, in double precision. "
            f"One generator, {_seed(NAME_INSTANCES)}, draws the instances one after another in the order of the table; for "
            f"each, Q (T x d_k), then K (T x d_k), then V (T x d_v), every entry normal() in row-major order, except for "
            f"'identical tokens', where one row of Q, one of K and one of V are drawn (d_k, d_k and d_v values) and repeated "
            f"T times. Each mutant (numbered m as in attention_variant) replaces one step: 1, the scale 1/sqrt(d_k) left "
            f"out; 2, each column of the masked scores normalized over the queries that may see that key instead of each "
            f"row over its keys; 3, every key visible to every query; 4, query i seeing keys up to i + 1; 5, K in place "
            f"of V (a product that exists only when d_v = d_k); 6, score (i, j) computed as k_i . q_j instead of "
            f"q_i . k_j. Dot products and the sums of the softmax and of P V are taken in increasing index. A mutant is "
            f"caught when some entry of its Y differs from the reference's by more than {SAME:g}; the number is that "
            f"largest absolute difference over all entries, to two significant digits.")
    headers = ["instance (T, d_k, d_v)"] + [m[1] for m in MUTANTS]
    rows = []
    for label, T, dk, dv, diffs in instance_verdicts():
        rows.append([f"{label} ({T}, {dk}, {dv})"] + [_verdict(d) for d in diffs])
    return head + "\n\n" + _table(headers, rows, "l" + "l" * len(MUTANTS))


# ------------------------------------------------------------------------------------------------ invariant measures

NAME_METRICS = "testing_numerical_code.invariant_measures"
ROW_SHIFT = (0.3, -0.2, 0.5)        # the constant added to every logit of row r
ROW_SCALE = (1.0, 1.2, 0.8)         # the factor applied to every logit of row r
SQUASH = 0.05                       # z -> z + SQUASH z^3, strictly increasing
COS_FLOOR = 0.9999
CEILING = 1e-5                      # rel_l2 and max_abs ceilings of the gates


def reference_logits():
    g = bookrng.generator(NAME_METRICS)
    return [[g.normal() for _ in range(5)] for _ in range(3)]


def _errors(b):
    swapped = [list(r) for r in b]
    order = sorted(range(5), key=lambda c: swapped[0][c])
    i, j = order[0], order[1]
    swapped[0][i], swapped[0][j] = swapped[0][j], swapped[0][i]
    return [
        ("correct, rounded to 32 bits", [[r32(v) for v in r] for r in b]),
        ("all logits times 1.05", [[1.05 * v for v in r] for r in b]),
        ("a constant added to each row", [[v + ROW_SHIFT[k] for v in r] for k, r in enumerate(b)]),
        ("each row scaled by its own factor", [[v * ROW_SCALE[k] for v in r] for k, r in enumerate(b)]),
        ("two smallest of row 0 exchanged", swapped),
        ("z + 0.05 z^3 on every logit", [[v + SQUASH * v ** 3 for v in r] for r in b]),
    ]


def _unit_rms(rows):
    out = []
    for r in rows:
        s = math.sqrt(sum(v * v for v in r) / len(r))
        out.append([v / s for v in r])
    return out


def _gates(a, b):
    """(1 - cos, rel_l2, and the verdict of each of the six gates) for output a against the reference b."""
    cos, rel, mx = cosine(a, b), rel_l2(a, b), max_abs(a, b)
    top = all(ra.index(max(ra)) == rb.index(max(rb)) for ra, rb in zip(a, b))
    probs = max_abs([_softmax(r) for r in a], [_softmax(r) for r in b])
    normed = rel_l2(_unit_rms(a), _unit_rms(b))
    passes = [cos >= COS_FLOOR, rel <= CEILING, mx <= CEILING, top, probs <= CEILING, normed <= CEILING]
    return 1.0 - cos, rel, passes


def invariant_measures():
    b = reference_logits()
    head = (f"Reference logits b: 3 rows of 5, drawn by {_seed(NAME_METRICS)} as normal() in row-major order. Each output "
            f"is b changed as the first column says: rounded to the nearest 32-bit number; multiplied by 1.05; row r "
            f"plus {ROW_SHIFT[0]}, {ROW_SHIFT[1]}, {ROW_SHIFT[2]}; row r times {ROW_SCALE[0]}, {ROW_SCALE[1]}, "
            f"{ROW_SCALE[2]}; the two smallest entries of row 0 swapped; every z replaced by z + {SQUASH} z^3. Six gates, "
            f"each a pass or FAIL: cos, the cosine of the two flattened 15-vectors at least {COS_FLOOR}; rel, rel_l2 at most "
            f"{CEILING:g}; max, max_abs at most {CEILING:g}; top-1, the largest entry in the same column in every row; "
            f"probs, max_abs of the row softmaxes at most {CEILING:g}; normed, rel_l2 at most {CEILING:g} after each row "
            f"of both is divided by its root mean square. Sums in index order, double precision.")
    rows = []
    caught = [0] * 6
    for label, a in _errors(b):
        one_minus_cos, rel, passes = _gates(a, b)
        if label != "correct, rounded to 32 bits":
            caught = [c + (0 if p else 1) for c, p in zip(caught, passes)]
        shown = "below 1e-12" if one_minus_cos < 1e-12 else _sci(one_minus_cos)
        rows.append([label, shown, _sci(rel)] + ["pass" if p else "FAIL" for p in passes])
    rows.append(["errors caught, of 5", "", ""] + [str(c) for c in caught])
    headers = ["output", "1 - cos", "rel_l2", "cos", "rel", "max", "top-1", "probs", "normed"]
    return head + "\n\n" + _table(headers, rows, "lrrllllll")


# ------------------------------------------------------------------------------------------------ gates that never run

TAPS = ("embed", "block0", "block1")
GATE_STATES = [
    ("sound", "golden present, cos and rel_l2 asserted"),
    ("bare_return", "golden absent, test returns early"),
    ("named_skip", "golden absent, skip printed"),
    ("require", "golden absent, run demands fixtures"),
    ("no_taps", "golden present, no tap name matches"),
    ("no_taps_counted", "as above, plus a count of compared taps"),
    ("unasserted", "rel_l2 printed, only cos asserted"),
    ("unasserted_counted", "as above, plus a count of compared taps"),
]


def _tap_values():
    """Three taps of the reference: the rows of the metric example's logits, flattened one per tap."""
    return {name: row for name, row in zip(TAPS, reference_logits())}


def run_gate(state, outputs):
    """Run one parity test in the given state on a candidate's taps.

    Returns (comparisons run, reported status, note), status being what a test runner prints: 'pass' or 'FAIL'."""
    golden = None if state in ("bare_return", "named_skip", "require") else _tap_values()
    if golden is None:
        if state == "require":
            return 0, "FAIL", "refused: fixture absent"
        return 0, "pass", "SKIP printed" if state == "named_skip" else "nothing printed"
    names = [n.upper() for n in TAPS] if state.startswith("no_taps") else list(TAPS)   # a golden whose names were renamed
    compared, failures = 0, []
    for n in names:
        if n not in golden:
            continue
        got, want = outputs[n], golden[n]
        compared += 1
        if cosine([got], [want]) < COS_FLOOR:
            failures.append(n)
        rel = rel_l2([got], [want])
        if not state.startswith("unasserted") and rel > CEILING:
            failures.append(n)
    if state.endswith("_counted") and compared != len(TAPS):
        return compared, "FAIL", "too few taps"
    return compared, ("FAIL" if failures else "pass"), ""


def silent_gates():
    ref = _tap_values()
    correct = {n: [r32(v) for v in row] for n, row in ref.items()}
    scaled = {n: [1.05 * v for v in row] for n, row in ref.items()}
    head = (f"One parity test of three taps ({', '.join(TAPS)}; the golden holds the three rows of the reference logits "
            f"of the previous table, one tap each, in the order listed) run in eight states, on the correct output (rounded to 32 bits) and on "
            f"the mutant that multiplies every value by 1.05. The sound test asserts cosine at least {COS_FLOOR} and rel_l2 "
            f"at most {CEILING:g} on every tap. 'returns early' and 'skip printed' return without comparing when the golden "
            f"is absent; 'demands fixtures' turns that return into a failure; 'no tap name matches' looks up names the "
            f"golden does not contain (the golden's names were changed) and skips each missing one; 'only cos asserted' "
            f"computes and prints rel_l2 but asserts only the cosine; 'plus a count' adds a failure when fewer "
            f"than {len(TAPS)} taps were compared. The status is what a test runner reports.")
    rows = []
    for key, label in GATE_STATES:
        n_ok, s_ok, note_ok = run_gate(key, correct)
        n_bad, s_bad, note_bad = run_gate(key, scaled)
        note = note_ok or note_bad
        rows.append([label, str(n_ok), s_ok, s_bad, note])
    headers = ["state of the test", "taps compared", "correct code", "code times 1.05", "runner also shows"]
    return head + "\n\n" + _table(headers, rows, "lrlll")


# ------------------------------------------------------------------------------------------------ conformance suite

NAME_CONF = "testing_numerical_code.conformance"
N_POS, N_CLS = 6, 4
TARGETS = [2, 0, 3, 1, IGNORE, 2]
FD_STEP = 1e-5
FD_ATOL, FD_RTOL = 1e-7, 1e-5
EXACT = 1e-12


def conformance_weights():
    """w_i = 0 when i is a multiple of 3, otherwise 0.5 + 0.1 i."""
    return [0.0 if i % 3 == 0 else 0.5 + 0.1 * i for i in range(N_POS)]


def conformance_logits():
    g = bookrng.generator(NAME_CONF)
    return [[g.normal() for _ in range(N_CLS)] for _ in range(N_POS)]


def _ce(z, t):
    return _lse(z) - z[t]


def _supervised(t):
    return [i for i, ti in enumerate(t) if ti != IGNORE]


def reference_loss(Z, t, w):
    """The definition: (1/S) sum over supervised positions i of w_i CE_i, S the number of supervised positions."""
    sup = _supervised(t)
    return sum(w[i] * _ce(Z[i], t[i]) for i in sup) / len(sup)


def plain_loss_and_grad(Z, t):
    """The unweighted masked mean cross-entropy and its gradient: the already-proven ordinary training path."""
    sup = _supervised(t)
    S = len(sup)
    loss = sum(_ce(Z[i], t[i]) for i in sup) / S
    G = [[0.0] * len(Z[0]) for _ in Z]
    for i in sup:
        p = _softmax(Z[i])
        G[i] = [(p[c] - (1.0 if c == t[i] else 0.0)) / S for c in range(len(p))]
    return loss, G


def _weighted(Z, t, w, weight_loss=True, weight_grad=True, divisor="count", mask=True):
    rows = _supervised(t) if mask else list(range(len(Z)))
    tt = [ti if ti != IGNORE else 0 for ti in t]
    if divisor == "count":
        D = len(rows)
    else:
        D = sum(w[i] for i in rows)
    loss = sum((w[i] if weight_loss else 1.0) * _ce(Z[i], tt[i]) for i in rows) / D
    G = [[0.0] * len(Z[0]) for _ in Z]
    for i in rows:
        p = _softmax(Z[i])
        s = (w[i] if weight_grad else 1.0) / D
        G[i] = [s * (p[c] - (1.0 if c == tt[i] else 0.0)) for c in range(len(p))]
    return loss, G


IMPLEMENTATIONS = [
    ("correct", lambda Z, t, w: _weighted(Z, t, w)),
    ("weights ignored", lambda Z, t, w: _weighted(Z, t, w, weight_loss=False, weight_grad=False)),
    ("divides by the sum of weights", lambda Z, t, w: _weighted(Z, t, w, divisor="weights")),
    ("weights in the loss only", lambda Z, t, w: _weighted(Z, t, w, weight_grad=False)),
    ("mask ignored", lambda Z, t, w: _weighted(Z, t, w, mask=False)),
]

CONTRACTS = [
    ("K1", "all-ones weights equal plain training"),
    ("K2", "gradient matches central differences"),
    ("K3", "a zero weight gives a zero gradient row"),
    ("K4", "an ignored position gives a zero gradient row"),
    ("K5", "loss equals the definition"),
]


def _close_all(A, B, tol):
    return all(abs(a - b) <= tol for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def contract(key, impl, Z, t, w):
    """True when the implementation honours contract `key` on this batch."""
    if key == "K1":
        ones = [1.0] * len(Z)
        loss, G = impl(Z, t, ones)
        want_loss, want_G = plain_loss_and_grad(Z, t)
        return abs(loss - want_loss) <= EXACT and _close_all(G, want_G, EXACT)
    _, G = impl(Z, t, w)
    if key == "K2":
        for i in range(len(Z)):
            for c in range(len(Z[0])):
                up = [list(r) for r in Z]
                down = [list(r) for r in Z]
                up[i][c] += FD_STEP
                down[i][c] -= FD_STEP
                num = (impl(up, t, w)[0] - impl(down, t, w)[0]) / (2 * FD_STEP)
                a = G[i][c]
                if abs(a - num) > FD_ATOL + FD_RTOL * max(abs(a), abs(num)):
                    return False
        return True
    if key == "K3":
        return all(all(g == 0.0 for g in G[i]) for i in range(len(Z)) if w[i] == 0.0 and t[i] != IGNORE)
    if key == "K4":
        return all(all(g == 0.0 for g in G[i]) for i in range(len(Z)) if t[i] == IGNORE)
    if key == "K5":
        return abs(impl(Z, t, w)[0] - reference_loss(Z, t, w)) <= EXACT * max(1.0, abs(reference_loss(Z, t, w)))
    raise ValueError(key)


def run_conformance(subjects, contracts, known_bad, Z, t, w):
    """Every contract on every subject, as the chapter's run_conformance: refuses an empty registry or an empty list
    of contracts, counts the checks that ran, and refuses a known-wrong subject that conforms."""
    if not subjects:
        raise ValueError("no subject is registered: a suite over nothing proves nothing")
    if not contracts:
        raise ValueError("a suite of no contract proves nothing")
    results, ran = [], 0
    for name, impl in subjects:
        row = []
        for key, _ in contracts:
            row.append(contract(key, impl, Z, t, w))
            ran += 1
        results.append((name, row))
    if ran != len(subjects) * len(contracts):
        raise ValueError("a contract did not run")
    for name, impl in known_bad:
        if all(contract(key, impl, Z, t, w) for key, _ in contracts):
            raise ValueError(f"a known-wrong subject conforms: {name}")
    return results, ran


def conformance_suite():
    Z, t, w = conformance_logits(), TARGETS, conformance_weights()
    results, ran = run_conformance(IMPLEMENTATIONS, CONTRACTS, IMPLEMENTATIONS[1:], Z, t, w)
    zs = "; ".join(" ".join(f"{v:+.4f}" for v in r) for r in Z)
    head = (f"Weighted cross-entropy over N = {N_POS} positions and C = {N_CLS} classes: L = (1/n_sup) sum over the "
            f"supervised positions i of w_i (lse(z_i) - z_i[t_i]), n_sup the number of supervised positions. Logits Z "
            f"(6 x 4) drawn by {_seed(NAME_CONF)} as normal() in row-major order (to four places: {zs}); targets t = "
            f"{TARGETS} with -1 marking the one ignored position; weights w_i = 0 when i is a multiple of 3, otherwise "
            f"0.5 + 0.1 i, so w = [{', '.join(f'{v:g}' for v in w)}]. The five implementations return a loss and its "
            f"gradient with respect to Z: the definition; plain cross-entropy that ignores w; the weighted sum divided by "
            f"the sum of the supervised weights instead of n_sup; w applied to the loss but not to the gradient; the ignored "
            f"position counted as a supervised one of class 0, with its own weight w_4 applied (dividing by N). K1 compares, at w = 1 everywhere, the "
            f"loss and every gradient entry with the plain masked mean (within {EXACT:g}); K2 differences the "
            f"implementation's own loss in each of the 24 logits (h = {FD_STEP:g}, gate {FD_ATOL:g} + {FD_RTOL:g} "
            f"max(|a|, |n|)); K3 and K4 require the gradient rows of position 0 and 3 (weight 0) and of position 4 "
            f"(ignored) to be exactly 0; K5 compares the loss with the definition evaluated independently (relative "
            f"{EXACT:g}). lse is the stable form max + log(sum(exp(z - max))) with the library exp and log, and every "
            f"sum runs in increasing index, in double precision. The suite ran {ran} checks, and the four wrong implementations were also passed as known_bad.")
    headers = ["implementation"] + [k for k, _ in CONTRACTS] + ["suite"]
    rows = []
    for name, row in results:
        rows.append([name] + ["pass" if r else "FAIL" for r in row] + ["conforms" if all(row) else "rejected"])
    return head + "\n\n" + _table(headers, rows, "l" + "l" * (len(CONTRACTS) + 1))


def contract_blindness():
    """For each contract alone: how many of the four wrong implementations it rejects."""
    Z, t, w = conformance_logits(), TARGETS, conformance_weights()
    results, _ = run_conformance(IMPLEMENTATIONS, CONTRACTS, IMPLEMENTATIONS[1:], Z, t, w)
    wrong = results[1:]
    rows = []
    for k, (key, label) in enumerate(CONTRACTS):
        caught = [name for name, row in wrong if not row[k]]
        rows.append([f"{key}: {label}", str(len(caught)), ", ".join(caught) if caught else "none"])
    head = "Each contract of the suite above on its own, against the four wrong implementations:"
    return head + "\n\n" + _table(["contract alone", "wrong ones rejected, of 4", "which"], rows, "lrl")
