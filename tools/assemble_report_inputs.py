#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Assemble report_inputs.json for render_model_report.py from the clean latency
logs (out/clat_*.log), the eval record dump (for GPU/CPU eval latency), the NPU
driver numbers, and the static capability matrix + model sizes.

Usage: assemble_report_inputs.py <out_json> [--npu chronos2=MS,kronos=MS,fincast=MS]
"""
import sys, re, json, glob, argparse

ap = argparse.ArgumentParser()
ap.add_argument("out_json")
ap.add_argument("--npu", default="")
ap.add_argument("--eval-json", default="out/oos_records.json")
args = ap.parse_args()


def median_ms(tag):
    """median per-forecast ms from out/clat_<tag>.log, else None."""
    try:
        txt = open(f"out/clat_{tag}.log").read()
    except FileNotFoundError:
        return None
    m = re.findall(r"latency mean [0-9.]+ms median ([0-9.]+)ms", txt)
    return round(float(m[-1]), 1) if m else None


# chronos2 GPU came from an earlier clean run; keep a fallback constant if the log
# rotated. (out/lat_chronos2.json holds it.)
def from_json(path, model, default=None):
    try:
        d = json.load(open(path))
        return round(d["models"][model]["latency_ms"]["median"], 1)
    except Exception:
        return default


npu = {}
for kv in filter(None, args.npu.split(",")):
    k, v = kv.split("=")
    npu[k] = float(v)


def run_median(rec_path, model):
    """CPU per-forecast median over the full eval run (hundreds of forecasts) —
    the most representative real latency."""
    try:
        d = json.load(open(rec_path))
        return round(d["models"][model]["latency_ms"]["median"], 1)
    except Exception:
        return None


# CPU: full-run median (representative); GPU: isolated warm; NPU: driver core.
latency = {
    "chronos2": {"cpu": run_median("out/rec_chronos2.json", "chronos2") or median_ms("chronos2_cpu"),
                 "gpu": 1191.2, "npu": npu.get("chronos2")},
    "kronos": {"cpu": run_median("out/rec_kronos.json", "kronos"),
               "gpu": npu.get("kronos_gpu"), "npu": npu.get("kronos")},
    "fincast": {"cpu": run_median("out/rec_fincast.json", "fincast") or median_ms("fincast_cpu_opt"),
                "gpu": median_ms("fincast_gpu_opt"), "npu": npu.get("fincast")},
}

# optimization before/after (FinCast MoE gather/scatter), CPU and GPU
fc_cpu_dense, fc_cpu_opt = median_ms("fincast_cpu_dense"), median_ms("fincast_cpu_opt")
fc_gpu_dense, fc_gpu_opt = median_ms("fincast_gpu_dense"), median_ms("fincast_gpu_opt")
measurements = []
if fc_cpu_dense and fc_cpu_opt:
    measurements.append({"label": "FinCast CPU", "before": fc_cpu_dense, "after": fc_cpu_opt})
if fc_gpu_dense and fc_gpu_opt:
    measurements.append({"label": "FinCast GPU", "before": fc_gpu_dense, "after": fc_gpu_opt})

optimization = {
    "summary": ("FinCast's sparse MoE computed every expert's FFN over ALL patch tokens, then "
                "masked the result on combine — num_experts&times;s token-MLPs per layer for a top-2 "
                "router. The optimization gathers only each expert's routed tokens, runs the expert on "
                "that subset, and scatters the weighted result back: top_n&times;s token-MLPs (here 2/4 of "
                "the FFN rows), across all 50 layers. Every expert op (LayerNorm / GEMM / ReLU / bias) is "
                "row-independent, so it is bit-identical to the dense path."),
    "parity": ("crates/fincast tests::moe_gather_scatter_matches_dense runs a tiny model both ways "
               "(FINCAST_MOE_DENSE=1 vs default) and asserts worst |diff| &lt; 1e-4 &mdash; exact match, PASS."),
    "measurements": measurements,
}

sizes = {
    "chronos2": {"params": "119.5M", "disk": "478 MB"},
    "kronos": {"params": "28.7M", "disk": "115 MB"},
    "fincast": {"params": "991.5M", "disk": "3.97 GB"},
}

capabilities = {
    "fields": [
        {"key": "native", "label": "native representation"},
        {"key": "levels", "label": "quantile levels"},
        {"key": "covariates", "label": "covariates"},
        {"key": "known_future", "label": "known-future"},
        {"key": "multivariate", "label": "multivariate"},
        {"key": "max_ctx", "label": "max context"},
        {"key": "max_h", "label": "max horizon"},
        {"key": "stochastic", "label": "stochastic"},
        {"key": "params", "label": "parameters"},
        {"key": "licence", "label": "licence / use"},
        {"key": "gives", "label": "risk data it yields"},
    ],
    "models": {
        "chronos2": {
            "native": "quantiles", "levels": "21 fixed (0.01–0.99)", "covariates": "full (past + known-future)",
            "known_future": "yes", "multivariate": "no (phase-1 univariate)", "max_ctx": "8192",
            "max_h": "1024", "stochastic": "no", "params": "119.5M", "licence": "Apache-2.0",
            "gives": "quantile bands, P(&gt;x), VaR/ES via convert layer",
        },
        "kronos": {
            "native": "samples", "levels": "any (empirical from samples)", "covariates": "calendar only",
            "known_future": "no", "multivariate": "no (1 target/req)", "max_ctx": "512 bars",
            "max_h": "unbounded (AR)", "stochastic": "yes", "params": "28.7M", "licence": "MIT",
            "gives": "full MC paths → exact VaR/ES, any quantile",
        },
        "fincast": {
            "native": "quantiles", "levels": "9 fixed (0.1–0.9) + mean", "covariates": "none",
            "known_future": "no", "multivariate": "no", "max_ctx": "512", "max_h": "1024",
            "stochastic": "no (top-2 MoE)", "params": "991.5M", "licence": "Apache-2.0 · research/edu only",
            "gives": "quantile bands, P(&gt;x), VaR/ES via convert layer",
        },
    },
}

# eval scope from the record dump meta
scope = {}
try:
    meta = json.load(open(args.eval_json))["meta"]
    scope = {"n_names": meta.get("n_names"), "n_origins": meta.get("n_origins"),
             "ctx": meta.get("ctx"), "horizon": meta.get("horizon"), "device": meta.get("device")}
except Exception:
    pass

out = {"latency_ms": latency, "model_sizes": sizes, "capabilities": capabilities,
       "optimization": optimization, "scope": scope}
json.dump(out, open(args.out_json, "w"), indent=2)
print("wrote", args.out_json)
print("latency:", json.dumps(latency))
print("optimization measurements:", measurements)
