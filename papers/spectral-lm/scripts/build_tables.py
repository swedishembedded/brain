# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting
# for its clients. If your team needs expertise in papers whose every number
# is regenerated from recorded measurements, you can procure our services by
# sending an email to info@swedishembedded.com.
"""Generate the paper's tables and macros from its result records.

Usage: build_tables.py <data dir> <output dir>

Every number the paper prints comes from a JSON record under <data dir>; this
script never invents one. A table whose records are absent says so instead of
printing a placeholder value. Records:

  correctness.json      {"checks": [{"name", "metric", "value", "tolerance", "pass"}], ...provenance}
  runs/<id>.json        one training run (see RUN_KEYS), aggregated per "label" over seeds
  qwen.json             {"rows": [{"label", "val_bpb", "tokens", "note"}], ...provenance}
  compress.json         {"rows": [{"basis", "m", "energy", "delta_nll"}], "layer", ...provenance}
  systems.json          {"rows": [{"m", "D", "C", "path", "ms_fwd_bwd", "max_abs_diff"}], ...provenance}

Provenance keys every record carries: "commit", "command", "device".
"""

import json
import sys
from pathlib import Path

PROVENANCE = ("commit", "command", "device")
RUN_KEYS = ("label", "seed", "tokens", "val_bpb", "params_nonembed", "params_embed",
            "flops_per_token", "train_tok_per_s", "peak_mem_bytes")


def load(path):
    record = json.loads(path.read_text())
    missing = [k for k in PROVENANCE if k not in record]
    if missing:
        raise SystemExit(f"{path}: record lacks provenance key(s) {missing}")
    return record


def tex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def not_measured(caption):
    return f"\\noindent\\emph{{{caption}: registered, not yet measured.}}\n"


def table(caption, label, header, rows, align):
    out = ["\\begin{table}[ht]", "\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def fmt(x, digits=3):
    if x is None:
        return "--"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) >= 1000):
        return f"{int(x):,}".replace(",", "\\,")
    return f"{x:.{digits}f}"


def correctness(data):
    path = data / "correctness.json"
    if not path.exists():
        return not_measured("Correctness gates")
    rec = load(path)
    rows = [[tex_escape(c["name"]), tex_escape(c["metric"]), f"{c['value']:.2e}", f"{c['tolerance']:.0e}",
             "pass" if c["pass"] else "\\textbf{fail}"] for c in rec["checks"]]
    return table("Correctness gates. Value is the measured error; every gate must pass before a comparison run.",
                 "tab:correctness", ["Gate", "Metric", "Value", "Tolerance", "Result"], rows, "llrrl")


def runs(data):
    run_dir = data / "runs"
    records = [load(p) for p in sorted(run_dir.glob("*.json"))] if run_dir.exists() else []
    groups = {}
    for r in records:
        missing = [k for k in RUN_KEYS if k not in r]
        if missing:
            raise SystemExit(f"run record {r.get('label')} seed {r.get('seed')} lacks {missing}")
        groups.setdefault(r["label"], []).append(r)
    return groups


def primary(groups):
    if not groups:
        return not_measured("Matched-width comparison")
    rows = []
    for label, rs in groups.items():
        bpb = [r["val_bpb"] for r in rs]
        mean = sum(bpb) / len(bpb)
        spread = f"[{min(bpb):.3f}, {max(bpb):.3f}]" if len(bpb) > 1 else "--"
        first = rs[0]
        rows.append([tex_escape(label), str(len(rs)), f"{mean:.3f}", spread,
                     fmt(first["params_nonembed"]), fmt(first["params_embed"]),
                     fmt(first["flops_per_token"]), fmt(sum(r["train_tok_per_s"] for r in rs) / len(rs), 0),
                     f"{max(r['peak_mem_bytes'] for r in rs) / 2**20:.0f}"])
    return table("Validation bits per byte on TinyStories at equal training tokens (lower is better). "
                 "Mean and range over seeds; parameters exclude (non-emb.) or are (emb.) the embedding and head; "
                 "arithmetic is analytic forward FLOPs per token; throughput is training tokens per second; "
                 "memory is peak device MiB.",
                 "tab:primary", ["Model", "Seeds", "BPB", "Range", "Non-emb.", "Emb.", "FLOPs/tok", "Tok/s", "MiB"],
                 rows, "lrrrrrrrr")


def qwen(data):
    path = data / "qwen.json"
    if not path.exists():
        return not_measured("Qwen3-0.6B comparison")
    rec = load(path)
    rows = [[tex_escape(r["label"]), f"{r['val_bpb']:.3f}", fmt(r.get("tokens")), tex_escape(r.get("note", ""))]
            for r in rec["rows"]]
    return table("Qwen3-0.6B on the same validation text, in bits per byte.", "tab:qwen",
                 ["Model", "BPB", "Train tokens", "Note"], rows, "lrrl")


def compress(data):
    path = data / "compress.json"
    if not path.exists():
        return not_measured("Spectral structure of Qwen3-0.6B hidden states")
    rec = load(path)
    rows = [[tex_escape(r["basis"]), str(r["m"]), f"{r['energy']:.4f}",
             "--" if r.get("delta_nll") is None else f"{r['delta_nll']:.3f}"] for r in rec["rows"]]
    return table(f"Qwen3-0.6B residual stream at layer {rec['layer']}: fraction of energy kept by an $m$-dimensional "
                 "projection, and the increase in validation next-token loss (nats) when the stream is replaced "
                 "by that projection.", "tab:compress", ["Basis", "$m$", "Energy", "$\\Delta$NLL"], rows, "lrrr")


def systems(data):
    path = data / "systems.json"
    if not path.exists():
        return not_measured("Transform cost")
    rec = load(path)
    rows = [[str(r["m"]), str(r["D"]), str(r["C"]), tex_escape(r["path"]), f"{r['ms_fwd_bwd']:.3f}",
             f"{r['max_abs_diff']:.1e}"] for r in rec["rows"]]
    return table("Spectral mixer forward plus backward time per layer, dense-product versus fast transform, and "
                 "the largest elementwise difference between their outputs.", "tab:systems",
                 ["$m$", "$D$", "$C$", "Path", "ms", "Max diff"], rows, "rrrlrr")


def macros(data, groups):
    measured = (data / "correctness.json").exists() and bool(groups)
    if measured:
        status = ""
    else:
        status = ("This is a draft: the protocol of \\cref{sec:protocol} is registered and its results are not "
                  "yet measured.")
    return "\n".join([
        f"\\newcommand{{\\StatusAbstract}}{{{status}}}",
        f"\\newcommand{{\\StatusResults}}{{{status}}}",
        f"\\newcommand{{\\StatusConclusion}}{{{status or 'See the results above.'}}}",
    ]) + "\n"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    groups = runs(data)
    (out / "macros.tex").write_text(macros(data, groups))
    (out / "table-correctness.tex").write_text(correctness(data))
    (out / "table-primary.tex").write_text(primary(groups))
    (out / "table-qwen.tex").write_text(qwen(data))
    (out / "table-compress.tex").write_text(compress(data))
    (out / "table-systems.tex").write_text(systems(data))


if __name__ == "__main__":
    main()
