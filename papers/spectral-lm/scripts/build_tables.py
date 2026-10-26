# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting
# for its clients. If your team needs expertise in papers whose every number
# is regenerated from recorded measurements, you can procure our services by
# sending an email to info@swedishembedded.com.
"""Generate the paper's tables and macros from its result records.

Usage: build_tables.py <data dir> <output dir>

Every number the paper prints comes from a JSON record under <data dir>,
written by scripts/run.py or scripts/correctness.py; this script never
invents one. A table whose records are absent says so instead of printing a
placeholder. Every record carries "commit" and "device".
"""

import json
import math
import sys
from pathlib import Path

H1_MARGIN = 0.02  # bits per byte, registered in the protocol

LABELS = {
    "S-256-512-24": r"$\mathcal{S}(256,512,24)$",
    "Q-256-24": r"$\mathcal{Q}(256,24)$",
    "Q-256-1024": r"$\mathcal{Q}(256,1024)$",
    "Q-512-2048": r"$\mathcal{Q}(512,2048)$",
    "S-256-512-8": r"$\mathcal{S}(256,512,8)$",
    "S-256-512-96": r"$\mathcal{S}(256,512,96)$",
}
ORDER = list(LABELS)


def records(data, plan):
    d = data / plan
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        r = json.loads(p.read_text())
        for k in ("commit", "device"):
            if k not in r:
                raise SystemExit(f"{p}: record lacks {k!r}")
        out.append(r)
    return out


def tex(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def num(x):
    return f"{int(x):,}".replace(",", "{,}")


def not_measured(what):
    return f"\\noindent\\emph{{{what}: registered, not yet measured.}}\n"


def table(caption, label, header, rows, align):
    out = ["\\begin{table}[ht]", "\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _betacf(a, b, x):
    """Continued fraction of the regularized incomplete beta (Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
    return h


def t_cdf(t, df):
    """Student-t CDF via the regularized incomplete beta function."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    ib = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * _betacf(b, a, 1 - x) / b
    return 1 - 0.5 * ib if t >= 0 else 0.5 * ib


def t_quantile(p, df):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def welch(a, b):
    """Mean difference a - b, its Welch 95% interval and two-sided p-value."""
    va, vb = sd(a) ** 2 / len(a), sd(b) ** 2 / len(b)
    diff, se = mean(a) - mean(b), math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t_quantile(0.975, df) * se
    p = 2 * (1 - t_cdf(abs(diff) / se, df))
    return diff, diff - half, diff + half, p, df


def correctness(data):
    recs = records(data, "correctness")
    if not recs:
        return not_measured("Correctness gates")
    rows = [[tex(c["name"]), tex(c["metric"]), f"{c['value']:.2g}", f"{c['tolerance']:.0g}", "pass" if c["pass"] else r"\textbf{fail}"]
            for c in recs[0]["checks"]]
    return table(f"Correctness gates, measured at commit \\code{{{recs[0]['commit'][:9]}}}. Every gate asserts its tolerance; "
                 "the value is what it measured.", "tab:correctness", ["Gate", "Metric", "Value", "Tolerance", "Result"], rows, "p{5.3cm}p{4.2cm}rrl")


def lr_table(data):
    recs = records(data, "lr")
    if not recs:
        return not_measured("Learning-rate selection"), {}
    by = {}
    for r in recs:
        by.setdefault(r["family"], []).append((float(r["train"]["lr"]), r["train"]["final_loss"]))
    chosen = {f: min(v, key=lambda x: x[1])[0] for f, v in by.items()}
    lrs = sorted({lr for v in by.values() for lr, _ in v})
    rows = []
    for f in [f for f in ORDER if f in by]:
        loss = dict(by[f])
        cells = []
        for lr in lrs:
            if lr not in loss:
                cells.append("--")
            else:
                cell = f"{loss[lr]:.3f}"
                cells.append(f"\\textbf{{{cell}}}" if lr == chosen[f] else cell)
        rows.append([LABELS[f]] + cells)
    return table("Learning-rate selection: final training loss (nats per token) after 300 steps, one seed; "
                 "the chosen rate per model is in bold.", "tab:lr", ["Model"] + [f"{lr:g}" for lr in lrs], rows, "l" + "r" * len(lrs)), chosen


def runs(data):
    groups = {}
    for plan in ("primary", "pilot"):
        for r in records(data, plan):
            groups.setdefault(r["family"], []).append(r)
    return groups


def comparison(groups):
    if not groups:
        return not_measured("Matched-width comparison")
    rows = []
    for f in [f for f in ORDER if f in groups]:
        rs = groups[f]
        bpb = [r["eval"]["bits_per_byte"] for r in rs]
        s = sd(bpb)
        spread = f"{mean(bpb):.4f} $\\pm$ {s:.4f}" if s is not None else f"{bpb[0]:.4f}"
        t = rs[0]["train"]
        rows.append([LABELS[f], str(len(rs)), spread, num(t["params_nonembed"]), num(t["params_embed"]),
                     f"{t['flops_per_token'] / 1e6:.1f}", num(mean([r['train']['tokens_per_s'] for r in rs])),
                     num(max(r["peak_mem_mib"] for r in rs))])
    return table("Validation bits per byte on TinyStories after 33.6M training tokens (lower is better): mean $\\pm$ sample "
                 "standard deviation over seeds. Parameters exclude (non-emb.) or are (emb.) the tied embedding and head; "
                 "MFLOP is analytic forward arithmetic per token at the 512-token context with transforms counted as FFTs; "
                 "throughput is training tokens per second and memory the peak device MiB of the run, both on one Tesla P40.",
                 "tab:primary", ["Model", "Seeds", "BPB", "Non-emb.", "Emb.", "MFLOP/tok", "Tok/s", "MiB"], rows, "lrrrrrrr")


def qwen(data, groups):
    recs = records(data, "qwen") + records(data, "qwen-ft")
    if not recs:
        return not_measured("Qwen3-0.6B comparison")
    rows = []
    for r in recs:
        res = r["result"]
        name = "Qwen3-0.6B, zero-shot" if "zero-shot" in r["id"] else f"Qwen3-0.6B, continued on {num(r['train']['tokens'])} story tokens"
        rows.append([name, f"{res['bits_per_byte']:.4f}", num(res["documents"]), num(res["tokens"])])
    for f in ("S-256-512-24", "Q-256-24", "Q-256-1024", "Q-512-2048"):
        if f in groups:
            rs = groups[f]
            rows.append([LABELS[f] + (f", mean of {len(rs)} seeds" if len(rs) > 1 else ", one seed"), f"{mean([x['eval']['bits_per_byte'] for x in rs]):.4f}",
                         num(rs[0]["eval"]["documents"]), num(rs[0]["eval"]["tokens"])])
    return table("Qwen3-0.6B and the trained models on the same validation documents. Bits per byte do not depend on the "
                 "tokenizer; the token columns show how differently the two vocabularies cut the same text.", "tab:qwen",
                 ["Model", "BPB", "Documents", "Tokens"], rows, "lrrr")


def spectrum(data):
    recs = records(data, "spectrum")
    if not recs:
        return not_measured("Spectral structure of Qwen3-0.6B hidden states")
    res = recs[0]["result"]
    widths = sorted({r["m"] for r in res["rows"]})
    names = ["fourier", "random", "coordinates", "pca"]
    pretty = {"fourier": "lowest Fourier modes", "random": "random orthonormal", "coordinates": "first coordinates", "pca": "principal components"}
    by = {(r["basis"], r["m"]): r for r in res["rows"]}
    rows = []
    for n in names:
        rows.append([pretty[n]] + [f"{by[(n, m)]['energy']:.3f} / {by[(n, m)]['delta_nll']:.2f}" for m in widths])
    return table(f"Qwen3-0.6B residual stream after layer {res['layer']} (width {res['residual_width']}), replaced by its projection "
                 f"onto $m$ directions of each basis (fitted on {num(res['fit_positions'])} positions of training stories): fraction "
                 f"of centred energy kept / increase in validation next-token loss in nats ({num(res['tokens'])} tokens of "
                 f"{res['documents']} documents; unprojected loss {res['base_nll']:.3f}).", "tab:compress",
                 ["Basis"] + [f"$m={m}$" for m in widths], rows, "l" + "r" * len(widths))


def systems(data):
    recs = records(data, "systems")
    if not recs:
        return not_measured("Transform cost")
    by = {}
    for r in recs:
        res = r["result"]
        mix = res["config"]["mixer"]
        key = (res["config"]["d_model"], mix["signal_len"], mix["channels"])
        by.setdefault(key, {})[res["transform"]] = res["step_ms"]
    rows = []
    for (m, d, c) in sorted(by):
        dense, fft = by[(m, d, c)].get("Dense"), by[(m, d, c)].get("Fft")
        rows.append([str(m), str(d), str(c), f"{dense:.0f}" if dense else "--", f"{fft:.0f}" if fft else "--",
                     f"{dense / fft:.2f}" if dense and fft else "--"])
    return table("Device time of one training step (forward and backward, 6 blocks, 16 sequences of 512 tokens) with the "
                 "spectral mixer's transforms as a dense product with the stored basis or as the FFT kernels, on one "
                 "Tesla P40, best of three.", "tab:systems", ["$m$", "$D$", "$C$", "Dense ms", "FFT ms", "Dense/FFT"], rows, "rrrrrr")


def samples(data):
    recs = records(data, "generate")
    if not recs:
        return not_measured("Generated text")
    out = ["\\begin{description}[style=nextline,leftmargin=0pt]"]
    for r in sorted(recs, key=lambda r: r["id"]):
        family = r["id"].rsplit("-", 1)[0]
        out.append(f"\\item[{LABELS.get(family, tex(family))}] {{\\small {tex(r['text'])}}}")
    out.append("\\end{description}")
    return "\n".join(out) + "\n"


def corpus(data):
    recs = records(data, "corpus")
    if not recs:
        return "", ""
    s = recs[0]["stats"]
    return num(s["train"]["tokens"]), num(s["val"]["documents"])


def macros(data, groups):
    lines = []
    def m(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    # The paper leaves draft status only when every registered plan has its
    # records: the gates, all six model families, both Qwen3 measurements,
    # the spectrum study and the transform cost.
    qwen_ids = {r["id"] for r in records(data, "qwen")} | {r["id"] for r in records(data, "qwen-ft")}
    measured = (bool(records(data, "correctness")) and all(f in groups for f in ORDER)
                and {"qwen3-0.6b-zero-shot", "qwen3-0.6b-tinystories"} <= qwen_ids
                and bool(records(data, "spectrum")) and bool(records(data, "systems")))
    status = "" if measured else ("This is a draft: the protocol of \\cref{sec:protocol} is registered and its results "
                                  "are not all measured yet.")
    m("StatusAbstract", status)
    m("StatusResults", status)
    m("StatusConclusion", status)
    def bpb(f):
        return mean([r["eval"]["bits_per_byte"] for r in groups[f]]) if f in groups else None
    s, qp, qs = bpb("S-256-512-24"), bpb("Q-256-24"), bpb("Q-256-1024")
    m("BpbS", f"{s:.3f}" if s else "--")
    m("BpbQpar", f"{qp:.3f}" if qp else "--")
    m("BpbQstd", f"{qs:.3f}" if qs else "--")
    if s and qp:
        a = [r["eval"]["bits_per_byte"] for r in groups["S-256-512-24"]]
        q = [r["eval"]["bits_per_byte"] for r in groups["Q-256-24"]]
        diff, lo, hi, p, df = welch(a, q)
        m("DiffSQpar", f"${diff:+.4f}$")
        m("DiffSQparCI", f"$[{lo:+.4f}, {hi:+.4f}]$")
        m("DiffSQparP", f"{p:.3f}")
        m("DiffSQparDF", f"{df:.1f}")
        m("RangeRelation", "lies entirely below the control's" if max(a) < min(q) else "overlaps the control's")
        # H1 is non-inferiority: the upper end of the interval below the margin.
        m("HOneVerdict", "supported" if hi <= H1_MARGIN else "rejected")
    else:
        m("DiffSQpar", "--")
        m("RangeRelation", "--")
        m("HOneVerdict", "not yet decided")
    if s and qs:
        diff, lo, hi, p, _ = welch([r["eval"]["bits_per_byte"] for r in groups["S-256-512-24"]], [r["eval"]["bits_per_byte"] for r in groups["Q-256-1024"]])
        m("DiffSQstd", f"${diff:+.4f}$")
        m("DiffSQstdCI", f"$[{lo:+.4f}, {hi:+.4f}]$")
    else:
        m("DiffSQstd", "--")
        m("DiffSQstdCI", "--")
    def avg(f, key):
        return mean([r["train"][key] for r in groups[f]]) if f in groups else None
    for f, name in (("S-256-512-24", "S"), ("Q-256-24", "Qpar"), ("Q-256-1024", "Qstd")):
        tps = avg(f, "tokens_per_s")
        m(f"Tps{name}", num(tps) if tps else "--")
        m(f"Mem{name}", num(max(r["peak_mem_mib"] for r in groups[f])) if f in groups else "--")
        m(f"Mflop{name}", f"{groups[f][0]['train']['flops_per_token'] / 1e6:.1f}" if f in groups else "--")
        m(f"Params{name}", num(groups[f][0]["train"]["params_nonembed"]) if f in groups else "--")
    q = records(data, "qwen")
    m("BpbQwen", f"{q[0]['result']['bits_per_byte']:.3f}" if q else "--")
    qf = records(data, "qwen-ft")
    m("BpbQwenFt", f"{qf[0]['result']['bits_per_byte']:.3f}" if qf else "--")
    m("QwenFtTokens", num(qf[0]["train"]["tokens"]) if qf else "--")
    m("QwenFtLossFrom", f"{qf[0]['train']['initial_loss']:.2f}" if qf else "--")
    m("QwenFtLossTo", f"{qf[0]['train']['final_loss']:.2f}" if qf else "--")
    m("QwenZsBytesPerToken", f"{q[0]['result']['bytes'] / q[0]['result']['tokens']:.2f}" if q else "--")
    w = groups.get("Q-512-2048")
    m("BpbQwide", f"{w[0]['eval']['bits_per_byte']:.3f}" if w else "--")
    m("ParamsQwide", num(w[0]["train"]["params_nonembed"]) if w else "--")
    train_tokens, val_docs = corpus(data)
    m("TrainTokens", train_tokens or "--")
    m("ValDocs", val_docs or "--")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    groups = runs(data)
    lr_tex, _ = lr_table(data)
    (out / "macros.tex").write_text(macros(data, groups))
    (out / "table-correctness.tex").write_text(correctness(data))
    (out / "table-lr.tex").write_text(lr_tex)
    (out / "table-primary.tex").write_text(comparison(groups))
    (out / "table-qwen.tex").write_text(qwen(data, groups))
    (out / "table-compress.tex").write_text(spectrum(data))
    (out / "table-systems.tex").write_text(systems(data))
    (out / "samples.tex").write_text(samples(data))


if __name__ == "__main__":
    main()
