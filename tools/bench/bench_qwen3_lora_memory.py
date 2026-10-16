#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for fitting long-context LLM
# fine-tuning onto the GPUs a team already owns. If your team needs expertise
# in training-memory budgeting and GPU placement then you can procure our
# services by sending an email to info@swedishembedded.com.

"""Peak device memory and seconds per optimizer step of `brain qwen3 finetune
--lora` at a ladder of sequence lengths. Standard library only.

Each length trains ONE example that renders to exactly T tokens (a user turn
`Continue.` plus an answer of T-16 ` the` tokens), so `--block T` is the row.
Seconds per step is the post-load time of a 2-step run minus that of a 1-step
run (the difference removes load, setup and the first step's one-time costs);
`--repeats` repeats the pair to bound the noise.

The GPUs are shared: before every run the script reads `nvidia-smi` and, if
any card already holds memory or any compute process exists, waits up to
`--wait-s` seconds for it to clear and then aborts that run rather than
colliding with it. The ladder stops at the first length whose run fails, since
the next is only larger. After every run it records the memory still held.

Output: one JSON object per run on stdout (and appended to `--out`), with
`peak_mib` per card, `train_phase_s`, the exit code, and the tail of the run's
log; the full log of each run goes to `--log-dir`.

    bench_qwen3_lora_memory.py --brain target/release/brain \\
        --models-dir DIR --workdir DIR --lengths 2048,4096,8192 [--device gpu0] \\
        [--base-dtype bf16|int8] [--model Qwen3-8B] [--repeats 2]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]


def compute_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=30, check=True).stdout.strip()


def cards_idle(idle_mib):
    mem = [int(r[0]) for r in smi("memory.used")]
    return all(m <= idle_mib for m in mem) and not compute_apps(), mem


def wait_idle(wait_s, idle_mib):
    deadline = time.time() + wait_s
    while True:
        idle, mem = cards_idle(idle_mib)
        if idle:
            return True, mem
        if time.time() >= deadline:
            return False, mem
        time.sleep(15)


def write_example(data_dir, tokens):
    os.makedirs(data_dir, exist_ok=True)
    answer = "the" + " the" * (tokens - 16 - 1)
    rec = {"messages": [{"role": "user", "content": "Continue.", "train": False},
                        {"role": "assistant", "content": answer, "train": True}]}
    with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
        f.write(json.dumps(rec) + "\n")


def run_once(args, label, tokens, steps):
    idle, mem = wait_idle(args.wait_s, args.idle_mib)
    if not idle:
        return {"label": label, "aborted": "cards busy", "mem_mib": mem, "compute_apps": compute_apps()}
    data_dir = os.path.join(args.workdir, f"data-{tokens}")
    write_example(data_dir, tokens)
    cmd = [args.brain] + ([] if args.device == "both" else ["--device", args.device]) + [
        "qwen3", "finetune", "--lora", str(args.rank), "--base-dtype", args.base_dtype,
        "--weights", f"Qwen/{args.model}", "--adapter", f"bench/{label}:t", "--dataset", data_dir,
        # seed: every measured step trains the same synthetic example, so the memory curves compare
        "--models-dir", args.models_dir, "--steps", str(steps), "--batch", "1", "--block", str(tokens), "--seed", "1"]
    env = dict(os.environ, RUST_BACKTRACE="1")
    t0 = time.time()
    lines, stamps = [], {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    peak, pids, stop = [0] * len(mem), set(), threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                for i, row in enumerate(smi("memory.used")):
                    peak[i] = max(peak[i], int(row[0]))
                pids.update(l.split(",")[0].strip() for l in compute_apps().splitlines() if l.strip())
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1.0)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    for line in proc.stdout:
        now = time.time() - t0
        lines.append(f"{now:8.1f} {line.rstrip()}")
        if line.startswith("brain: placement") and "=cpu" in line:
            # The placer found no all-GPU layout and homed a stage on the host,
            # where the training kernels do not run: that is "does not fit".
            stamps["host_stage"] = now
            proc.kill()
        if line.startswith("training LoRA adapter"):
            stamps["train_start"] = now
        if line.startswith("trained:"):
            stamps["trained"] = now
    rc = proc.wait()
    wall = time.time() - t0
    stop.set()
    thread.join()
    time.sleep(5)
    after = [int(r[0]) for r in smi("memory.used")]
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, f"log-{label}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    phase = round(stamps["trained"] - stamps["train_start"], 1) if "trained" in stamps and "train_start" in stamps else None
    return {"label": label, "model": args.model, "tokens": tokens, "steps": steps, "device": args.device,
            "base_dtype": args.base_dtype, "rc": rc, "wall_s": round(wall, 1), "train_phase_s": phase, "no_gpu_layout": "host_stage" in stamps,
            "peak_mib": peak, "mem_after_mib": after, "compute_pids_seen": len(pids),
            "log_tail": [l[:300] for l in lines[-40:] if "BRAIN_GPU_WAIT_S" in l or "panicked" in l or "Out of Memory" in l or "poll_wait" in l] if rc != 0 else []}


def emit(args, res):
    line = json.dumps(res)
    print(line, flush=True)
    with open(args.out, "a") as f:
        f.write(line + "\n")


def run_clean(args, label, tokens, steps):
    """Run until no other compute process shared the cards (one brain process
    is one pid, however many cards it uses), or `--retries` runs are used up.
    A run another process overlapped is not a measurement of this one."""
    for attempt in range(args.retries + 1):
        res = run_once(args, f"{label}" + (f"-try{attempt}" if attempt else ""), tokens, steps)
        shared = res.get("compute_pids_seen", 1) > 1
        contended = res.get("rc") != 0 and any("BRAIN_GPU_WAIT_S" in l for l in res.get("log_tail", []))
        if not (shared or contended):
            return res
        res["discarded"] = "shared cards" if shared else "device creation timed out under contention"
        emit(args, res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", required=True, help="path to a RELEASE brain binary")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--workdir", required=True, help="scratch dir for datasets (keep it outside the git tree)")
    ap.add_argument("--log-dir", help="per-run logs (default WORKDIR/logs)")
    ap.add_argument("--out", help="append result lines here (default WORKDIR/results.jsonl)")
    ap.add_argument("--lengths", default="2048,4096,8192")
    ap.add_argument("--device", default="both", help="gpu0, gpu1 or both (the placer chooses)")
    ap.add_argument("--base-dtype", default="bf16", choices=["f32", "bf16", "int8"])
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2, help="1-step/2-step pairs per length")
    ap.add_argument("--wait-s", type=int, default=0, help="seconds to wait for idle cards before aborting a run")
    ap.add_argument("--retries", type=int, default=3, help="re-runs of a run another process overlapped")
    ap.add_argument("--idle-mib", type=int, default=300)
    ap.add_argument("--tag", default="r", help="label prefix, to keep configurations apart")
    args = ap.parse_args()
    args.log_dir = args.log_dir or os.path.join(args.workdir, "logs")
    args.out = args.out or os.path.join(args.workdir, "results.jsonl")
    os.makedirs(args.workdir, exist_ok=True)


    for tokens in [int(x) for x in args.lengths.split(",")]:
        phases = {1: [], 2: []}
        for rep in range(args.repeats):
            for steps in (2, 1):
                res = run_clean(args, f"{args.tag}{tokens}-s{steps}-{rep}", tokens, steps)
                emit(args, res)
                if res.get("aborted") or res["rc"] != 0 or res.get("no_gpu_layout"):
                    print(f"stopping at {tokens} tokens: {res.get('aborted') or 'rc=' + str(res['rc'])}", file=sys.stderr)
                    return 1
                phases[steps].append(res["train_phase_s"])
        per_step = [round(a - b, 1) for a, b in zip(phases[2], phases[1])]
        emit(args, {"summary": True, "tokens": tokens, "base_dtype": args.base_dtype, "device": args.device,
              "s_per_step_pairs": per_step, "tok_per_s": [round(tokens / s, 1) for s in per_step if s > 0]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
