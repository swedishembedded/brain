# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "From Hypothesis to Paper".

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description:

* the operating characteristic of two decision rules for a non-inferiority claim, by simulation, against the closed form
  where one exists;
* a small evaluation whose number moves when the bytes of "the same" validation text change, and the checksum that tells;
* the headline comparison of brain's spectral-lm paper recomputed from its committed result records: bits per byte from
  the summed negative log-likelihood and the byte count each record holds, the parameter and arithmetic counts re-derived
  from the configuration each record names, and the Welch interval and verdicts of the registered hypothesis;
* the chance that a fresh replication of that comparison, with new seeds, reaches each verdict.

The record example reads the JSON files under papers/spectral-lm/data/primary/ of the repository; every value it uses
from them is printed in the chapter, so a reader without the repository can repeat the arithmetic from the page.

Randomness comes only from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its seed and draw order.
Sums are plain left-to-right additions in a loop (never Python's `sum`, which compensates). The Student t distribution
function for a real number of degrees of freedom is computed from the regularized incomplete beta function by its
continued fraction (modified Lentz), with the library `lgamma`, `exp` and `log`.

Swedish Embedded AB implements solutions for research results that others can check and rebuild for its clients. If your
team needs expertise in experiment design, reproducibility packages and technical papers, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import functools
import hashlib
import json
import math
from pathlib import Path

import bookrng

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "papers" / "spectral-lm" / "data" / "primary"

MARGIN = 0.02            # bits per byte, the margin of the claim
SIGMA = 0.005            # bits per byte, the run-to-run spread assumed in the opening simulation
DELTAS = (-0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
RUNS = (3, 8)            # runs per arm in the opening simulation
EXPERIMENTS = 10000      # simulated experiments per cell
REPLICATION_RUNS = (3, 6, 10)

FAMILIES = (("S-256-512-24", "spectral mixer, C = 24"), ("Q-256-24", "SwiGLU, f = 24"), ("Q-256-1024", "SwiGLU, f = 1024"))


# --------------------------------------------------------------------------- formatting

def _table(headers, rows, align=None):
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fx(x, d):
    """x rounded to d decimals; refuses a value within a millionth of a unit of the last digit from a rounding tie, so the
    printed digit is never a coin toss."""
    scaled = abs(x) * 10 ** d
    if abs(scaled - math.floor(scaled) - 0.5) < 1e-6:
        raise ValueError(f"{x!r} lies on a rounding tie at {d} decimals")
    s = f"{x:.{d}f}"
    return "0." + "0" * d if s.lstrip("-").strip("0.") == "" else s


def _signed(x, d):
    s = _fx(x, d)
    return s if s.startswith("-") else "+" + s


# --------------------------------------------------------------------------- statistics

def _mean(xs):
    total = 0.0
    for x in xs:
        total += x
    return total / len(xs)


def _var(xs):
    """Sample variance, divisor n - 1."""
    m = _mean(xs)
    total = 0.0
    for x in xs:
        total += (x - m) * (x - m)
    return total / (len(xs) - 1)


def _betacf(a, b, x):
    """Continued fraction of the regularized incomplete beta function, modified Lentz, to a relative step of 1e-15."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        num = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def inc_beta(x, a, b):
    """The regularized incomplete beta function I_x(a, b) for 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf_real(t, nu):
    """Distribution function of Student's t with nu > 0 degrees of freedom (nu need not be an integer)."""
    tail = 0.5 * inc_beta(nu / (nu + t * t), 0.5 * nu, 0.5)
    return 1.0 - tail if t >= 0 else tail


def t_quantile_real(p, nu):
    """The t with t_cdf_real(t, nu) = p, for 0.5 < p < 1, by 200 bisection steps on [0, 1000]."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf_real(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def welch_parts(a, b):
    """(difference of means a - b, its standard error, Welch's degrees of freedom)."""
    va, vb = _var(a) / len(a), _var(b) / len(b)
    se = math.sqrt(va + vb)
    nu = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    return _mean(a) - _mean(b), se, nu


def welch(a, b):
    """(difference, lower, upper end of the 95% interval, nu, multiplier, two-sided p)."""
    d, se, nu = welch_parts(a, b)
    k = t_quantile_real(0.975, nu)
    p = 2.0 * (1.0 - t_cdf_real(abs(d) / se, nu))
    return d, d - k * se, d + k * se, nu, k, p, se


def _interval_rule_passes(a, b, margin):
    """Upper end of the Welch 95% interval of a - b at most the margin, written as the equivalent condition on the
    distribution function so that no quantile is searched for each experiment."""
    d, se, nu = welch_parts(a, b)
    return d <= margin and t_cdf_real((margin - d) / se, nu) >= 0.975


# --------------------------------------------------------------------------- the opening: two rules for one claim

@functools.lru_cache(maxsize=None)
def pass_rates():
    """{(n, delta): (share passing the mean rule, share passing the interval rule)} from one generator."""
    rng = bookrng.generator("research.rule_pass_rates")
    out = {}
    for n in RUNS:
        for delta in DELTAS:
            mean_pass = interval_pass = 0
            for _ in range(EXPERIMENTS):
                cand = [delta + SIGMA * rng.normal() for _ in range(n)]
                base = [SIGMA * rng.normal() for _ in range(n)]
                if _mean(cand) - _mean(base) <= MARGIN:
                    mean_pass += 1
                if _interval_rule_passes(cand, base, MARGIN):
                    interval_pass += 1
            out[(n, delta)] = (mean_pass / EXPERIMENTS, interval_pass / EXPERIMENTS)
    return out


def mean_rule_exact(n, delta):
    """Chance that the mean of n - n Gaussian differences lies at or below the margin."""
    return _phi((MARGIN - delta) / (SIGMA * math.sqrt(2.0 / n)))


def rule_pass_rates():
    rates = pass_rates()
    rows = []
    for delta in DELTAS:
        cells = [_signed(delta, 3) if delta else "0.000", "true" if delta <= MARGIN else "false"]
        for n in RUNS:
            m, i = rates[(n, delta)]
            cells += [f"{m:.4f} ({_fx(mean_rule_exact(n, delta), 4)})", f"{i:.4f}"]
        rows.append(cells)
    head = (f"The claim: the candidate's bits per byte exceed the baseline's by at most {MARGIN} (the margin). A simulated "
            f"experiment trains n runs of each arm; run i of the candidate scores delta + {SIGMA} z and run i of the baseline "
            f"{SIGMA} z, z a standard normal draw. Generator `research.rule_pass_rates` "
            f"({bookrng.seed_of('research.rule_pass_rates'):#018x}), one generator for the whole table, `rng.normal()` "
            f"(Box-Muller, the sine partner held for the next call) in this order: n = 3 before n = 8, the true differences "
            f"top to bottom, {EXPERIMENTS:,} experiments per cell, and in each experiment the n candidate draws before the n "
            f"baseline draws. The mean rule passes when the difference of the two means is at most {MARGIN}; in brackets, its "
            f"exact chance Phi((0.02 - delta) / (0.005 sqrt(2 / n))). The interval rule passes when the upper end of the "
            f"Welch 95% interval of the difference is at most {MARGIN}, evaluated as t_cdf_real((0.02 - d) / se, nu) >= 0.975 "
            f"with d <= 0.02, which is the same condition. Shares are counts divided by {EXPERIMENTS:,}, printed exactly; their "
            f"Monte Carlo standard error is at most 0.005.\n\n")
    headers = ["true difference delta", "claim is", "mean rule, n = 3", "interval rule, n = 3", "mean rule, n = 8",
               "interval rule, n = 8"]
    return head + _table(headers, rows)


# --------------------------------------------------------------------------- data identity

TRAIN_TEXT = "the cat sat on the mat.\nthe dog sat on the log.\na cat and a dog sat.\n"
VALID_DOCS = ("the cat sat on the log.", "a dog sat on the mat and the cat sat on the dog.", "the mat sat.")


def _byte_model(train):
    """Add-one byte model: p(b) = (count of b in the training bytes + 1) / (number of training bytes + 256)."""
    counts = [0] * 256
    data = train.encode("utf-8")
    for b in data:
        counts[b] += 1
    return [(c + 1) / (len(data) + 256) for c in counts]


def _bits_per_byte(model, data):
    bits = 0.0
    for b in data:
        bits += -math.log2(model[b])
    return bits / len(data)


def identity_variants():
    lf = "".join(d + "\n" for d in VALID_DOCS)
    crlf = "".join(d + "\r\n" for d in VALID_DOCS)
    short = "".join(d + "\n" for d in VALID_DOCS if len(d.encode("utf-8")) <= 30)
    typo = lf.replace("the log", "the lot", 1)
    return (("as published (each document ends in LF)", lf, len(VALID_DOCS)),
            ("the same documents ending in CR LF", crlf, len(VALID_DOCS)),
            ("documents of at most 30 bytes only", short, sum(1 for d in VALID_DOCS if len(d.encode('utf-8')) <= 30)),
            ("first 'log' changed to 'lot'", typo, len(VALID_DOCS)))


def data_identity():
    model = _byte_model(TRAIN_TEXT)
    n = len(TRAIN_TEXT.encode("utf-8"))
    rows = []
    for name, text, docs in identity_variants():
        data = text.encode("utf-8")
        rows.append([name, docs, len(data), hashlib.sha256(data).hexdigest()[:16], _fx(_bits_per_byte(model, data), 4)])
    head = ("The model is the add-one byte model of the training text 'the cat sat on the mat.', 'the dog sat on the "
            f"log.', 'a cat and a dog sat.', each line ending in LF (byte 10), {n} bytes: p(b) = (count of b + 1) / ({n} + 256) "
            "for each of the 256 byte values. The validation documents are 'the cat sat on the log.', 'a dog sat on the mat "
            "and the cat sat on the dog.' and 'the mat sat.'. Bits per byte = the sum over the scored bytes of -log2 p(b), "
            "divided by their number (log2 is the library function; four decimals). The checksum is the first 16 hexadecimal "
            "digits of the SHA-256 of the scored bytes.\n\n")
    return head + _table(["validation text", "documents", "bytes", "SHA-256 (first 16 hex digits)", "bits per byte"], rows)


# --------------------------------------------------------------------------- the paper's records

def _records():
    out = {}
    for path in sorted(RECORDS.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault(r["family"], []).append(r)
    for runs in out.values():
        runs.sort(key=lambda r: r["seed"])
    return out


def _bpb(record):
    e = record["eval"]
    return e["nats"] / math.log(2.0) / e["bytes"]


def nonembedding_params(c):
    """Parameters outside the tied embedding, from a record's configuration, by the count the chapter states."""
    d, hq, hkv, hd = c["d_model"], c["n_heads"] * c["head_dim"], c["n_kv_heads"] * c["head_dim"], c["head_dim"]
    mix = c["mixer"]
    sub = 3 * mix["channels"] * d if mix["kind"] == "spectral" else 3 * d * mix["d_ff"]
    block = d + hq * d + 2 * hkv * d + 2 * hd + d * hq + d + sub
    return c["n_layers"] * block + d


def flops_per_token(c, t):
    """Forward FLOPs per token at context t, transforms counted as FFTs, by the counting rule the chapter states."""
    d, v, hq, hkv = c["d_model"], c["vocab_size"], c["n_heads"] * c["head_dim"], c["n_kv_heads"] * c["head_dim"]
    attn = 2 * d * (2 * hq + 2 * hkv) + (4 * hq * (t + 1)) // 2
    mix = c["mixer"]
    if mix["kind"] == "spectral":
        ch, n = mix["channels"], mix["signal_len"]
        sub = 4 * ch * d + 5 * ch * n + 3 * ch * 5 * n * int(math.log2(n))
    else:
        sub = 6 * d * mix["d_ff"]
    return c["n_layers"] * (attn + sub) + 2 * d * v


def record_runs():
    groups = _records()
    rows = []
    for fam, label in FAMILIES:
        for r in groups[fam]:
            e = r["eval"]
            again = _bpb(r)
            rows.append([label, str(r["seed"]), r["commit"][:9], r["device"], f"{e['nats']:.3f}", f"{e['bytes']:,}",
                         _fx(again, 9), "yes" if abs(again - e["bits_per_byte"]) < 1e-12 else "no"])
    head = ("The nine records of papers/spectral-lm/data/primary/, one per trained model, in the paper's order and by seed "
            "within a model. Bits per byte recomputed = summed next-token loss in nats / ln 2 / UTF-8 bytes of the scored "
            "text, from the two fields the record holds; every record scored the same 26,439 documents (4,935,406 tokens). "
            "The last column compares it with the bits_per_byte field the record also holds, to 1e-12.\n\n")
    return head + _table(["model", "seed", "commit", "device", "summed loss (nats)", "bytes scored", "bits per byte",
                          "equals the record"], rows)


def record_counts():
    groups = _records()
    rows = []
    for fam, label in FAMILIES:
        r = groups[fam][0]
        c, tr = r["train"]["config"], r["train"]
        p, f = nonembedding_params(c), flops_per_token(c, c["block_size"])
        rows.append([label, f"{p:,}", f"{tr['params_nonembed']:,}", f"{f:,}", f"{tr['flops_per_token']:,}",
                     "yes" if (p, f) == (tr["params_nonembed"], tr["flops_per_token"]) else "no"])
    head = ("Counts re-derived from the configuration each record names (6 blocks, residual width m = 256, 8 query heads and 4 "
            "key-value heads of dimension 32, vocabulary 4,096, context t = 512) and compared with the counts the training "
            "program wrote into the same record; the three runs of a model hold identical counts.\n\n")
    return head + _table(["model", "non-embedding parameters, re-derived", "in the record", "forward FLOPs per token, "
                          "re-derived", "in the record", "equal"], rows)


def _bpbs(fam):
    return [_bpb(r) for r in _records()[fam]]


def record_verdict():
    s, qp, qs = _bpbs("S-256-512-24"), _bpbs("Q-256-24"), _bpbs("Q-256-1024")
    rows = []
    for name, other in (("spectral minus SwiGLU f = 24 (H1)", qp), ("spectral minus SwiGLU f = 1024 (H2)", qs)):
        d, lo, hi, nu, k, p, se = welch(s, other)
        rows.append([name, _signed(d, 4), _fx(se, 4), _fx(nu, 2), _fx(k, 3), f"{_signed(lo, 4)} to {_signed(hi, 4)}",
                     _fx(p, 3) if p >= 0.001 else "below 0.001"])
    d, lo, hi, *_ = welch(s, qp)
    verdicts = [["registered: mean difference at most +0.02", "supported" if d <= MARGIN else "rejected"],
                ["stricter: upper end of the interval at most +0.02", "supported" if hi <= MARGIN else "not shown"],
                ["superiority: upper end of the interval below 0", "shown" if hi < 0 else "not shown"]]
    spread = [["spectral mixer", _fx(_mean(s), 6), _fx(math.sqrt(_var(s)), 6)],
              ["SwiGLU f = 24", _fx(_mean(qp), 6), _fx(math.sqrt(_var(qp)), 6)],
              ["SwiGLU f = 1024", _fx(_mean(qs), 6), _fx(math.sqrt(_var(qs)), 6)]]
    pooled = math.sqrt(0.5 * (_var(s) + _var(qp)))
    head = ("From the nine bits per byte of the record table (computed at full precision; the nine printed decimals give the same "
            "printed results), three runs per model. Standard deviations use the divisor n - 1. "
            "Welch's standard error is sqrt(s_a^2 / n_a + s_b^2 / n_b), its degrees of freedom nu = (s_a^2 / n_a + "
            "s_b^2 / n_b)^2 / ((s_a^2 / n_a)^2 / (n_a - 1) + (s_b^2 / n_b)^2 / (n_b - 1)), the multiplier is "
            "t_quantile_real(0.975, nu), and p is the two-sided tail 2 (1 - t_cdf_real(|d| / se, nu)).\n\n")
    return (head + _table(["model", "mean bits per byte", "standard deviation over 3 runs"], spread) + "\n\n"
            + _table(["contrast", "mean difference", "standard error", "nu", "multiplier", "95% interval", "p"], rows)
            + "\n\nH1 against the margin 0.02 bits per byte:\n\n" + _table(["decision rule", "verdict"], verdicts)
            + f"\n\nThe root mean square of the two standard deviations of H1's arms is {_fx(pooled, 4)} bits per byte.")


# --------------------------------------------------------------------------- replication

@functools.lru_cache(maxsize=None)
def replication_rates():
    s, qp = _bpbs("S-256-512-24"), _bpbs("Q-256-24")
    # The simulation starts from the printed six-decimal values, so the page holds all of its inputs.
    ms, ss, mq, sq = (float(_fx(x, 6)) for x in (_mean(s), math.sqrt(_var(s)), _mean(qp), math.sqrt(_var(qp))))
    rng = bookrng.generator("research.replication")
    out = {}
    for n in REPLICATION_RUNS:
        reg = strict = sup = 0
        for _ in range(EXPERIMENTS):
            a = [ms + ss * rng.normal() for _ in range(n)]
            b = [mq + sq * rng.normal() for _ in range(n)]
            d = _mean(a) - _mean(b)
            reg += d <= MARGIN
            strict += _interval_rule_passes(a, b, MARGIN)
            sup += _interval_rule_passes(a, b, 0.0)
        out[n] = (reg / EXPERIMENTS, strict / EXPERIMENTS, sup / EXPERIMENTS)
    return out, (ms, ss, mq, sq)


def replication():
    rates, (ms, ss, mq, sq) = replication_rates()
    rows = [[n, f"{r:.4f}", f"{t:.4f}", f"{u:.4f}"] for n, (r, t, u) in rates.items()]
    head = (f"Suppose the true bits per byte of each model and its spread from seed to seed were exactly what the three "
            f"records show: spectral mixer mean {_fx(ms, 6)}, standard deviation {_fx(ss, 6)}; SwiGLU f = 24 mean "
            f"{_fx(mq, 6)}, standard deviation {_fx(sq, 6)} (these six-decimal values are the simulation's inputs). A replication trains n new seeds of each, run i scoring mean + sd z. "
            f"Generator `research.replication` ({bookrng.seed_of('research.replication'):#018x}), `rng.normal()`, n = 3, 6, 10 "
            f"in turn, {EXPERIMENTS:,} replications each, the n spectral draws before the n SwiGLU draws. Each column is the "
            f"share of replications reaching that verdict on H1 (Monte Carlo standard error at most 0.005); 'superiority' "
            f"means the upper end of the Welch 95% interval of the difference lies below 0, tested as "
            f"t_cdf_real(-d / se, nu) >= 0.975 with d <= 0.\n\n")
    return head + _table(["seeds per model", "registered rule supported", "stricter rule supported", "superiority shown"],
                         rows)
