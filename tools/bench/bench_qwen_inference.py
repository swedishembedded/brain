#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Head-to-head Qwen3 inference benchmark across brain's three deployment
targets: the from-scratch WGSL engine on **CPU** (Cranelift JIT) and **GPU**
(wgpu/WebGPU), plus the **Intel NPU** path (brain's own ONNX export -> OpenVINO).
On CPU/GPU the HuggingFace Transformers Qwen3 is benchmarked alongside on the
SAME prompt for an apples-to-apples compare; the NPU row is brain-only (no HF
NPU path).

Each brain row runs `brain qwen infer` twice with two `--max-new` values and
separates the costs:
  * load   = model load (and, for NPU, the one-time ONNX export + OpenVINO
             compile), isolated as the y-intercept of the two timings;
  * per-token latency / tokens-per-second = the marginal cost of the extra
             generated tokens (brain's generation is cache-free recompute, so
             this is the average over the added context — reported as such);
  * peak RSS of the engine subprocess.
The generated text is printed so you can eyeball correctness across targets.

Prereqs: a brain Qwen `.weights` (brain qwen import --hf <dir>), the matching
`tokenizer.json`, and `make release`. The HF rows need `pip install transformers
torch`. The NPU row needs OpenVINO + the Intel NPU driver at run time (the env
is set up automatically from the `openvino` pip package); without them the npu
row is reported skipped with the engine's own diagnostic.

  python3 tools/bench/bench_qwen_inference.py \
      --weights /path/qwen.weights \
      --tokenizer /path/tokenizer.json \
      --hf /path/Qwen3-0.6B \
      --prompt "The capital of France is" \
      --brain-bin ./target/release/brain

Device selection (which targets to run): default = all three. `DEVICE=cpu|gpu|npu`
env or `--device cpu|gpu|npu` runs only that one. A target that can't run on this
machine is reported skipped and the others still run.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time

DEVICES = ("cpu", "gpu", "npu")


def _vmhwm_kb(pid: int) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def _npu_env():
    """Child env so brain's OpenVINO runtime-linking finds libopenvino. Locates
    the `openvino` pip package's libs dir, ensures the bare `libopenvino_c.so`
    soname exists, and exports INTEL_OPENVINO_DIR (a constructed runtime/lib/
    intel64 tree the openvino-finder expects) + LD_LIBRARY_PATH. Returns the env
    unchanged if openvino isn't importable (the npu row then reports skipped)."""
    env = dict(os.environ)
    try:
        import openvino  # noqa: F401
        libs = os.path.join(os.path.dirname(str(openvino.__file__)), "libs")
    except Exception:  # noqa: BLE001
        return env
    if not os.path.isdir(libs):
        return env
    soname = os.path.join(libs, "libopenvino_c.so")
    if not os.path.exists(soname):
        for f in sorted(os.listdir(libs)):
            if f.startswith("libopenvino_c.so."):
                try:
                    os.symlink(f, soname)
                except OSError:
                    pass
                break
    # Construct an INTEL_OPENVINO_DIR tree the finder accepts (symlinks to libs).
    root = "/tmp/brain_ov"
    intel64 = os.path.join(root, "runtime", "lib", "intel64")
    os.makedirs(intel64, exist_ok=True)
    for f in os.listdir(libs):
        dst = os.path.join(intel64, f)
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.join(libs, f), dst)
            except OSError:
                pass
    env["INTEL_OPENVINO_DIR"] = root
    env["LD_LIBRARY_PATH"] = libs + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _run_brain(argv, env):
    """Run a brain subprocess; return (wall_s, peak_rss_kb, stdout, stderr).
    Peak RSS is sampled from /proc while the child runs (a high-water mark)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    rss = 0
    while proc.poll() is None:
        rss = max(rss, _vmhwm_kb(proc.pid))
        time.sleep(0.02)
    out, err = proc.communicate()
    rss = max(rss, _vmhwm_kb(proc.pid))
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError((err or out).strip() or f"exit {proc.returncode}")
    return wall, rss, out, err


def bench_brain_qwen(device, a):
    """Benchmark brain `qwen infer` on `device` (cpu|gpu|npu) in a single run; the
    engine reports `qwen-timing load_ms=.. gen_ms=.. tokens=..` (load = model load
    / NPU export+compile; gen = the token loop), so load and per-token are split
    accurately without a flaky two-run diff. Returns a dict of metrics."""
    env = _npu_env() if device == "npu" else dict(os.environ)
    argv = [a.brain_bin, "qwen", "infer", "--weights", a.weights,
            "--tokenizer", a.tokenizer, "--prompt", a.prompt, "--temp", "0",
            "--device", device, "--max-new", str(a.max_new)]
    _, rss, out, err = _run_brain(argv, env)
    adapter = next((l.strip() for l in err.splitlines()
                    if "adapter:" in l or l.startswith("npu:")), None)
    m = re.search(r"qwen-timing load_ms=([\d.]+) gen_ms=([\d.]+) tokens=(\d+)", err)
    if not m:
        raise RuntimeError(f"no qwen-timing line in engine output:\n{err.strip()[-400:]}")
    load_ms, gen_ms, ntok = float(m[1]), float(m[2]), max(int(m[3]), 1)
    per_tok_ms = gen_ms / ntok
    tok_s = (1e3 / per_tok_ms) if per_tok_ms > 0 else 0.0
    return {"load": load_ms / 1e3, "per_tok": per_tok_ms, "tok_s": tok_s,
            "rss": rss, "text": out.strip(), "adapter": adapter}


def bench_hf_qwen(a, torch_device):
    """Reference: HuggingFace Transformers Qwen3 on cpu/cuda. Same split."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if (torch_device == "cuda" and torch.cuda.is_available()) else "cpu"
    note = None if dev == torch_device or torch_device == "cpu" else \
        f"CUDA unavailable -> CPU (requested {torch_device})"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32).to(dev).eval()
    ids = tok(a.prompt, return_tensors="pt").input_ids.to(dev)
    load = time.perf_counter() - t0

    def gen(max_new):
        with torch.no_grad():
            t = time.perf_counter()
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id)
            return time.perf_counter() - t, out

    gen(1)  # warm
    gen_s, out = gen(a.max_new)
    per_tok_ms = gen_s / max(a.max_new, 1) * 1e3
    tok_s = (1e3 / per_tok_ms) if per_tok_ms > 0 else 0.0
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    return {"load": load, "per_tok": per_tok_ms, "tok_s": tok_s, "rss": rss,
            "text": text, "note": note, "dev": dev}


def _row(name, m, extra=""):
    return (f"{name:<26} load {m['load']:6.2f}s | per-token {m['per_tok']:8.1f} ms | "
            f"{m['tok_s']:7.2f} tok/s | peak RSS {m['rss']/1024:7.1f} MiB {extra}")


def run_cpu_gpu(a, device):
    backend = "native CPU-JIT" if device == "cpu" else "wgpu/WebGPU GPU"
    print(f"prompt={a.prompt!r}  max_new={a.max_new}  brain_device={device}\n")
    hf = None
    if a.hf:
        try:
            hf = bench_hf_qwen(a, "cuda" if device == "gpu" else "cpu")
            if hf.get("note"):
                print(f"   [{hf['note']}]")
            print(_row(f"HF transformers ({hf['dev']})", hf))
            print(f"   text: {hf['text']!r}\n")
        except Exception as e:  # noqa: BLE001
            print(f"   [HF reference skipped: {type(e).__name__}: {e}]\n")
    m = bench_brain_qwen(device, a)
    print(_row(f"brain ({backend})", m, "(engine subprocess)"))
    print(f"   engine adapter: {m['adapter'] or '(none reported)'}")
    print(f"   text: {m['text']!r}\n")
    if hf:
        sx = m["per_tok"] / hf["per_tok"] if hf["per_tok"] else 0
        rel = "faster than" if sx < 1 else "the per-token latency of"
        factor = (1 / sx) if sx and sx < 1 else sx
        print(f"==> brain is {factor:.1f}x {rel} HF; "
              f"RSS {m['rss']/hf['rss']:.2f}x ({m['rss']/1024:.0f} vs {hf['rss']/1024:.0f} MiB).")


def run_npu(a):
    print(f"prompt={a.prompt!r}  max_new={a.max_new}  npu (OpenVINO)\n")
    try:
        m = bench_brain_qwen("npu", a)
    except Exception as e:  # noqa: BLE001
        print(f"   [npu skipped: {e}]")
        print("   (needs OpenVINO + the Intel NPU driver at run time; see docs/yolo/NPU.md)")
        return
    print(_row("brain (Intel NPU, OpenVINO)", m,
               "(load incl. ONNX export + NPU compile)"))
    print(f"   engine: {m['adapter'] or '(none reported)'}")
    print(f"   text: {m['text']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="scratchpad/qwen.weights",
                    help="brain Qwen .weights (from `brain qwen import`)")
    ap.add_argument("--tokenizer", default="scratchpad/tokenizer.json")
    ap.add_argument("--hf", default=None, help="HF Qwen3 dir for the reference rows (cpu/gpu)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--brain-bin", default="./target/release/brain")
    ap.add_argument("--max-new", type=int, default=8, help="tokens to generate per run")
    ap.add_argument("--device", default=None, help="run ONLY this target: cpu|gpu|npu")
    a = ap.parse_args()

    selected = (a.device or os.environ.get("DEVICE") or "").lower()
    if selected and selected not in DEVICES:
        sys.exit(f"unknown DEVICE {selected!r}; expected one of {', '.join(DEVICES)}")
    devices = [selected] if selected else list(DEVICES)
    print(f"running targets: {', '.join(devices)}\n")
    for i, device in enumerate(devices):
        if i:
            print()
        print(f"================ {device.upper()} ================")
        try:
            run_npu(a) if device == "npu" else run_cpu_gpu(a, device)
        except Exception as e:  # noqa: BLE001
            print(f"   [{device} target failed: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    sys.exit(main())
