# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting
# for its clients. If your team needs expertise in measurement campaigns whose
# every number names the commit and command that produced it, you can procure
# our services by sending an email to info@swedishembedded.com.
"""Run the paper's registered measurements and write their result records.

Usage (from the repository root, after `make samples/text/spectral-lm/build`
and the TinyStories preparation the sample's README describes):

  python3 papers/spectral-lm/scripts/run.py <plan> [--gpus 0,1] [--work DIR]

Plans (the protocol section of the paper registers them):

  corpus    the prepared TinyStories corpus: statistics and raw-file checksums
  lr        300-step learning-rate selection runs, one seed per family and rate
  primary   the three primary configurations, three random seeds each
  pilot     the wide control and the channel pilots, one random seed each
  qwen      Qwen3-0.6B: zero-shot bits per byte
  qwen-ft   Qwen3-0.6B continued on the training stories for QWEN_FT_STEPS
            steps of the shared schedule, then scored
  spectrum  Qwen3-0.6B residual-stream projection study
  systems   training-step device time of the spectral mixer, dense basis
            product against FFT, over several (m, D, C)

Each job is one command of the sample; a training job is followed by scoring
its checkpoint. Jobs run one per GPU at a time. Every record lands in
papers/spectral-lm/data/<plan>/ and carries the commit, the commands, the
seed, the device, and the job's peak device memory (the card's memory in use
above its idle use, sampled from nvidia-smi every second while the job runs
alone on the card). Checkpoints and logs go to --work, which
is outside the repository.
"""

import argparse
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "papers" / "spectral-lm" / "data"
SAMPLE = ROOT / "target" / "release" / "sample-text-spectral-lm"
CORPUS = ROOT / "data" / "tinystories"
QWEN = Path.home() / ".local" / "share" / "brain" / "models" / "Qwen" / "Qwen3-0.6B" / "model.brain.safetensors"

SHAPE = ["--d-model", "256", "--layers", "6", "--heads", "8", "--kv-heads", "4", "--head-dim", "32"]
SCHEDULE = ["--batch", "16", "--block", "512", "--warmup", "100", "--weight-decay", "0.1", "--clip", "1.0"]
STEPS = "4096"
MAX_DOC_BYTES = "1500"

FAMILIES = {
    "S-256-512-24": SHAPE + ["--mixer", "spectral", "--signal-len", "512", "--channels", "24"],
    "Q-256-24": SHAPE + ["--mixer", "swiglu", "--d-ff", "24"],
    "Q-256-1024": SHAPE + ["--mixer", "swiglu", "--d-ff", "1024"],
    "Q-512-2048": ["--d-model", "512", "--layers", "6", "--heads", "8", "--kv-heads", "4", "--head-dim", "64", "--mixer", "swiglu", "--d-ff", "2048"],
    "S-256-512-8": SHAPE + ["--mixer", "spectral", "--signal-len", "512", "--channels", "8"],
    "S-256-512-96": SHAPE + ["--mixer", "spectral", "--signal-len", "512", "--channels", "96"],
}
PRIMARY = ["S-256-512-24", "Q-256-24", "Q-256-1024"]
PILOT = ["Q-512-2048", "S-256-512-8", "S-256-512-96"]
LR_GRID = ["1e-3", "3e-3", "6e-3"]
# Qwen3-0.6B continued training: the budget is set from a measured
# throughput probe and registered in the protocol before the run.
QWEN_FT_STEPS = "0"
QWEN_FT_BATCH = "4"
QWEN_FT_LR = "3e-5"
QWEN_FT_STORIES = "200000"
SYSTEMS = [(256, 512, 8), (256, 512, 24), (256, 512, 96), (128, 256, 24), (256, 1024, 24)]


def git_commit():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def device_name(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def chosen_lr(family):
    """The registered rate: the lowest final training loss of the lr plan."""
    best = None
    for rec in (DATA / "lr").glob(f"{family}-lr*.json"):
        r = json.loads(rec.read_text())
        loss = r["train"]["final_loss"]
        if best is None or loss < best[0]:
            best = (loss, str(r["train"]["lr"]))
    if best is None:
        sys.exit(f"no learning-rate records for {family}; run the lr plan first")
    return best[1]


def jobs_for(plan, work):
    jobs = []
    if plan == "lr":
        for fam in PRIMARY + PILOT:
            for lr in LR_GRID:
                jobs.append({"id": f"{fam}-lr{lr}", "family": fam, "lr": lr, "steps": "300", "seed": str(secrets.randbits(63)), "eval": False})
    elif plan in ("primary", "pilot"):
        fams, seeds = (PRIMARY, 3) if plan == "primary" else (PILOT, 1)
        for fam in fams:
            for _ in range(seeds):
                seed = str(secrets.randbits(63))
                jobs.append({"id": f"{fam}-{seed}", "family": fam, "lr": chosen_lr(fam), "steps": STEPS, "seed": seed, "eval": True})
    elif plan == "qwen":
        jobs.append({"id": "qwen3-0.6b-zero-shot", "command": ["qwen-eval", "--weights", str(QWEN), "--text", str(CORPUS / "valid.txt"), "--max-doc-bytes", MAX_DOC_BYTES, "--batch", "8"]})
    elif plan == "qwen-ft":
        jobs.append({"id": "qwen3-0.6b-tinystories", "qwen_ft": True, "seed": str(secrets.randbits(63))})
    elif plan == "systems":
        seed = str(secrets.randbits(63))
        for m, d, c in SYSTEMS:
            heads = ["--d-model", str(m), "--layers", "6", "--heads", "8", "--kv-heads", "4", "--head-dim", str(m // 8)]
            for path in ("dense", "fft"):
                jobs.append({"id": f"S-{m}-{d}-{c}-{path}", "command": ["profile", *heads, "--mixer", "spectral", "--signal-len", str(d),
                             "--channels", str(c), "--transform", path, "--batch", "16", "--block", "512", "--reps", "3", "--seed", seed]})
    elif plan == "spectrum":
        seed = str(secrets.randbits(63))
        jobs.append({"id": "qwen3-0.6b-layer14", "command": ["qwen-spectrum", "--weights", str(QWEN), "--fit-text", str(CORPUS / "train.txt"), "--text", str(CORPUS / "valid.txt"), "--layer", "14", "--widths", "64,128,256,512", "--fit-docs", "256", "--docs", "256", "--max-doc-bytes", MAX_DOC_BYTES, "--batch", "8", "--seed", seed]})
    else:
        sys.exit(f"unknown plan {plan!r}")
    for j in jobs:
        j["work"] = str(work)
    return jobs


def card_memory(gpu):
    """Device memory in use on card `gpu`, MiB."""
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip())


def run_measured(cmd, gpu, log):
    """Run `cmd` (already pinned to `gpu` by its `--gpu` flag), returning
    (last JSON line, peak MiB, seconds)."""
    env = dict(os.environ)
    start = time.time()
    with open(log, "a") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=lf, text=True)
        # nvidia-smi reports PIDs of another PID namespace inside a container,
        # so the job's own process cannot be looked up. A job has its card to
        # itself, so its device memory is the card's use above the idle use
        # measured before the job started.
        idle = card_memory(gpu)
        peak = 0
        lines = []
        stdout = p.stdout
        assert stdout is not None
        reader = threading.Thread(target=lambda: lines.extend(stdout), daemon=True)
        reader.start()
        while p.poll() is None:
            peak = max(peak, card_memory(gpu) - idle)
            time.sleep(1.0)
        reader.join()
        lf.writelines(lines)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {p.returncode}; see {log}")
    record = next((json.loads(l) for l in reversed(lines) if l.startswith("{")), None)
    if record is None:
        raise RuntimeError(f"{' '.join(cmd)} printed no record; see {log}")
    return record, peak, time.time() - start


def execute(job, gpu, plan, commit):
    work = Path(job["work"])
    log = work / f"{job['id']}.log"
    rec = {"commit": commit, "device": f"gpu{gpu} {device_name(gpu)}", "plan": plan, "id": job["id"]}
    if job.get("qwen_ft"):
        corpus = work / "qwen-corpus"
        enc = [str(SAMPLE), "qwen-corpus", "--weights", str(QWEN), "--text", str(CORPUS / "train.txt"), "--valid", str(CORPUS / "valid.txt"),
               "--out", str(corpus), "--stories", QWEN_FT_STORIES, "--gpu", str(gpu)]
        corp, _, _ = run_measured(enc, gpu, log)
        ckpt = work / f"{job['id']}.safetensors"
        train = [str(SAMPLE), "qwen-train", "--weights", str(QWEN), str(corpus), "--out", str(ckpt), "--batch", QWEN_FT_BATCH, "--block", "512",
                 "--warmup", "20", "--weight-decay", "0.1", "--clip", "1.0", "--steps", QWEN_FT_STEPS, "--lr", QWEN_FT_LR, "--seed", job["seed"], "--gpu", str(gpu)]
        result, peak, _ = run_measured(train, gpu, log)
        ev = [str(SAMPLE), "qwen-eval", "--weights", str(ckpt), "--tokenizer", str(QWEN.parent), "--text", str(CORPUS / "valid.txt"),
              "--max-doc-bytes", MAX_DOC_BYTES, "--batch", "8", "--gpu", str(gpu)]
        score, _, _ = run_measured(ev, gpu, log)
        rec.update({"seed": int(job["seed"]), "corpus_command": " ".join(enc), "corpus": corp, "command": " ".join(train), "train": result,
                    "peak_mem_mib": peak, "eval_command": " ".join(ev), "result": score})
    elif "command" in job:
        cmd = [str(SAMPLE)] + job["command"] + ["--gpu", str(gpu)]
        result, peak, _ = run_measured(cmd, gpu, log)
        rec.update({"command": " ".join(cmd), "result": result, "peak_mem_mib": peak})
    else:
        ckpt = work / f"{job['id']}.safetensors"
        train = [str(SAMPLE), "train", str(CORPUS), "--out", str(ckpt), *FAMILIES[job["family"]], *SCHEDULE,
                 "--steps", job["steps"], "--lr", job["lr"], "--seed", job["seed"], "--gpu", str(gpu)]
        result, peak, _ = run_measured(train, gpu, log)
        rec.update({"family": job["family"], "seed": int(job["seed"]), "command": " ".join(train), "train": result, "peak_mem_mib": peak})
        if job["eval"]:
            ev = [str(SAMPLE), "eval", "--checkpoint", str(ckpt), "--tokenizer", str(CORPUS / "tokenizer.json"),
                  "--text", str(CORPUS / "valid.txt"), "--max-doc-bytes", MAX_DOC_BYTES, "--batch", "16", "--gpu", str(gpu)]
            score, _, _ = run_measured(ev, gpu, log)
            rec.update({"eval_command": " ".join(ev), "eval": score})
    out = DATA / plan / f"{job['id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[gpu{gpu}] wrote {out.relative_to(ROOT)}", flush=True)


def record_corpus(commit):
    """The corpus record: what `brain data gen tinystories` wrote, and the
    checksums of the raw files it read."""
    import hashlib
    sums = {}
    for name in ("train.txt", "valid.txt"):
        h = hashlib.sha256()
        with open(CORPUS / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        sums[name] = h.hexdigest()
    rec = {"commit": commit, "device": "host", "command": "brain data gen tinystories --out data/tinystories",
           "source": "huggingface.co/datasets/roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 (TinyStoriesV2-GPT4-{train,valid}.txt), CDLA-Sharing-1.0",
           "sha256": sums, "stats": json.loads((CORPUS / "stats.json").read_text())}
    out = DATA / "corpus" / "tinystories.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--work", required=True, help="checkpoint and log directory outside the repository")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    if commit.endswith("-dirty"):
        sys.exit("the tracked tree has uncommitted changes; records must name a commit that reproduces them")
    if a.plan == "corpus":
        record_corpus(commit)
        return
    jobs = queue.Queue()
    for j in jobs_for(a.plan, work):
        jobs.put(j)
    failures = []

    def worker(gpu):
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                execute(job, gpu, a.plan, commit)
            except Exception as e:  # a failed job is reported, never silently dropped
                failures.append(f"{job['id']}: {e}")
                print(f"[gpu{gpu}] FAILED {job['id']}: {e}", flush=True)

    threads = [threading.Thread(target=worker, args=(int(g),)) for g in a.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        sys.exit("failed jobs:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
