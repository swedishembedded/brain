#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Sharded full-universe OOS sweep: build once, fan the ticker universe across
N processes (OOS_SHARD=i/n), merge the date-keyed shard dumps, score.

The harness (crates/cli/tests/oos_skill_eval.rs) is single-threaded per
process with warm in-process weights; name-sharding across cores is what
turns a ~45 h serial full-SP500 sweep into a few hours of wall clock.

Usage:
  oos_shard.py --data out/bt/bt --db stocks.db --out out/bt/eval.json
    [--shards 20] [--ctx 120] [--horizon 5] [--step 5] [--nsamples 3]
    [--start 2026-01-01] [--maxorig 0] [--kronos-ft out/bt/ft.weights]
    [--split-manifest out/bt/split_manifest.json]
    [--summary-out out/backtest_summary.json] [--k-frac 0.10]

Requires KRONOS_TOKENIZER_DIR / KRONOS_DECODER_DIR (and friends) in the env,
exactly like running the harness directly.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="backtest csv dir (bt/)")
ap.add_argument("--db", required=True, help="stocks.db (for the ^gspc benchmark export)")
ap.add_argument("--out", required=True, help="merged eval json path")
ap.add_argument("--shards", type=int, default=max(2, (os.cpu_count() or 4) - 2))
ap.add_argument("--ctx", type=int, default=120)
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--step", type=int, default=5)
ap.add_argument("--nsamples", type=int, default=3)
ap.add_argument("--start", default="2026-01-01")
ap.add_argument("--maxorig", type=int, default=0)
ap.add_argument("--kronos-ft", default=None, help=".weights for the kronos_ft model entry")
ap.add_argument("--split-manifest", default=None)
ap.add_argument("--summary-out", default=None)
ap.add_argument("--k-frac", type=float, default=0.10)
ap.add_argument("--report-out", default=None, help="metrics json (default <out>.metrics.json)")
args = ap.parse_args()


def build_test_binary() -> str:
    """cargo-build the harness once; return the test executable path."""
    print("[oos_shard] building the eval harness (release) …", flush=True)
    proc = subprocess.run(
        ["cargo", "test", "--release", "-p", "brain-cli", "--test", "oos_skill_eval",
         "--no-run", "--message-format=json"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    exe = None
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") \
                and "oos_skill_eval" in msg["executable"]:
            exe = msg["executable"]
    if not exe:
        sys.exit("could not locate the oos_skill_eval test binary from cargo output")
    return exe


def export_gspc(db: str, dest: str) -> None:
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT Date, Open, High, Low, Close, Volume FROM stock_data "
        "WHERE Ticker='^gspc' AND Close IS NOT NULL ORDER BY Date ASC").fetchall()
    if not rows:
        sys.exit("no ^gspc rows in the db — run `trademiner update` first")
    with open(dest, "w") as f:
        f.write("Date,open,high,low,close,volume\n")
        for d, o, h, l, c, v in rows:
            f.write(f"{str(d)[:10]},{o:.4f},{h:.4f},{l:.4f},{c:.4f},{int(v or 0)}\n")
    print(f"[oos_shard] exported ^gspc ({len(rows)} bars) -> {dest}", flush=True)


exe = build_test_binary()
gspc_csv = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".", "gspc.csv")
export_gspc(args.db, gspc_csv)

base_env = dict(
    os.environ,
    OOS_DATA=args.data,
    OOS_CTX=str(args.ctx), OOS_HORIZON=str(args.horizon), OOS_STEP=str(args.step),
    OOS_NSAMPLES=str(args.nsamples), OOS_START=args.start, OOS_MAXORIG=str(args.maxorig),
    BRAIN_DEVICE=os.environ.get("BRAIN_DEVICE", "cpu"),
    RAYON_NUM_THREADS=os.environ.get("RAYON_NUM_THREADS", "1"),
    OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
)
if args.kronos_ft:
    base_env["KRONOS_FT"] = args.kronos_ft

n = args.shards
procs, shard_outs, logs = [], [], []
t0 = time.time()
for i in range(n):
    shard_out = f"{args.out}.shard{i}.json"
    shard_outs.append(shard_out)
    env = dict(base_env, OOS_SHARD=f"{i}/{n}", OOS_OUT=shard_out)
    log = open(f"{args.out}.shard{i}.log", "w")
    logs.append(log)
    procs.append(subprocess.Popen(
        [exe, "oos_skill_eval", "--exact", "--nocapture"],
        env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
print(f"[oos_shard] launched {n} shards over {args.data}", flush=True)

failed = []
for i, p in enumerate(procs):
    rc = p.wait()
    logs[i].close()
    status = "ok" if rc == 0 and os.path.exists(shard_outs[i]) else f"FAILED rc={rc}"
    if status != "ok":
        failed.append(i)
    print(f"[oos_shard] shard {i}/{n}: {status} ({time.time()-t0:.0f}s)", flush=True)
if failed:
    sys.exit(f"shards failed: {failed} — see {args.out}.shard<i>.log")

subprocess.run([sys.executable, os.path.join(HERE, "merge_records.py"), "--concat",
                args.out] + shard_outs, check=True)

report_out = args.report_out or f"{args.out}.metrics.json"
cmd = [sys.executable, os.path.join(HERE, "oos_skill_report.py"),
       args.out, gspc_csv, report_out, "--k-frac", str(args.k_frac)]
if args.split_manifest:
    cmd += ["--split-manifest", args.split_manifest]
if args.kronos_ft:
    cmd += ["--ft-model", "kronos_ft", "--base-model", "kronos"]
if args.summary_out:
    cmd += ["--summary-out", args.summary_out]
subprocess.run(cmd, check=True)
print(f"[oos_shard] done in {time.time()-t0:.0f}s", flush=True)
