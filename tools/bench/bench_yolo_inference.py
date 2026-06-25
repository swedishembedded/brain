#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Head-to-head inference benchmark: brain (from-scratch WGSL on the CPU JIT)
vs the default Python/Ultralytics YOLOv8n, on the SAME image + 640px + CPU.

Measures, end-to-end (image -> boxes): model load time, warm per-image latency
(median over N), throughput (img/s), and peak RSS.

Prereqs (dev machine): `pip install ultralytics`, an exported brain weights file
(tools/yolo_export/export_yolov8.py), and `cargo build --release`.

  python3 tools/bench/bench_yolo_inference.py \
      --image scratchpad/dog.png \
      --pt scratchpad/yolov8n.pt \
      --brain-weights scratchpad/yolov8n.brain.weights \
      --brain-bin ./target/release/brain

Device selection (each side picks its device independently):
  --device cpu|gpu          set both sides at once
  --torch-device cpu|cuda|0 Ultralytics/torch device (default cpu)
  --brain-device cpu|gpu    brain backend: cpu = native Cranelift-JIT+AVX2,
                            gpu = wgpu/WebGPU running the WGSL kernels (default cpu)
Note: brain's device is driven by the BRAIN_DEVICE the bench passes to the
engine subprocess via --brain-device — NOT by a BRAIN_DEVICE you set in your
own shell (the client sets it explicitly). The brain CPU fast paths
(Cranelift JIT + AVX2 fast-conv/fast-ops) apply only to --brain-device cpu.
"""
from __future__ import annotations
import argparse, os, resource, statistics, sys, time
from PIL import Image


def _vmhwm_kb(pid: int) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def bench_ultralytics(image_path, pt, n, device):
    from ultralytics import YOLO
    import torch
    t0 = time.perf_counter()
    m = YOLO(pt)
    im = Image.open(image_path).convert("RGB")
    # warm-up (first calls pay graph/alloc init)
    for _ in range(2):
        m.predict(im, imgsz=640, device=device, verbose=False)
    load = time.perf_counter() - t0
    lat = []
    for _ in range(n):
        t = time.perf_counter()
        r = m.predict(im, imgsz=640, device=device, verbose=False)
        lat.append((time.perf_counter() - t) * 1e3)
    res = r[0].boxes
    dets = sorted(((int(c), float(s)) for c, s in zip(res.cls.tolist(), res.conf.tolist())),
                  key=lambda d: -d[1])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB (whole py+torch process)
    # On CPU torch parallelises over get_num_threads(); on CUDA that's not the
    # relevant knob, so report the device string instead.
    par = str(torch.get_num_threads()) if str(device) == "cpu" else str(device)
    return load, lat, rss, dets, par


def bench_brain(image_path, weights, brain_bin, n, conf, device):
    sys.path.insert(0, "brain-py")
    from brain_py import client as _client_mod
    from brain_py.client import BrainClient
    # Surface the engine's own "adapter:" stderr line so the user can SEE which
    # backend actually ran (native CPU-JIT vs the wgpu GPU adapter), instead of
    # trusting the requested device. The client otherwise swallows stderr.
    adapter: dict = {"line": None}
    def _drain(self):
        for raw in self._proc.stderr:
            s = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
            if adapter["line"] is None and "adapter:" in s:
                adapter["line"] = s.strip()
    _client_mod.BrainClient._drain_stderr = _drain
    im = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    # `device` is forwarded verbatim as BRAIN_DEVICE for the engine subprocess:
    # "cpu" -> native Cranelift-JIT + AVX2 fast-conv backend; anything else
    # (e.g. "gpu") -> the wgpu/WebGPU backend running the WGSL kernels.
    client = BrainClient(yolo=weights, conf=conf, brain_bin=brain_bin, device=device)
    client.__enter__()              # spawns `brain run --yolo`, waits for ready (model loaded)
    load = time.perf_counter() - t0
    pid = client._proc.pid
    client.detect(im, timeout=300)  # warm-up
    lat, dets = [], []
    for _ in range(n):
        t = time.perf_counter()
        dets = client.detect(im, timeout=300)
        lat.append((time.perf_counter() - t) * 1e3)
    rss = _vmhwm_kb(pid)            # KB (brain engine subprocess only)
    out = sorted(((d.cls, d.conf) for d in dets), key=lambda d: -d[1])
    client.__exit__(None, None, None)
    return load, lat, rss, out, adapter["line"]


def _row(name, load, lat, rss_kb, extra=""):
    med = statistics.median(lat)
    return (f"{name:<22} load {load:6.2f}s | latency med {med:8.1f} ms "
            f"(min {min(lat):8.1f}, max {max(lat):8.1f}) | "
            f"{1000/med:7.2f} img/s | peak RSS {rss_kb/1024:7.1f} MiB {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="scratchpad/dog.png")
    ap.add_argument("--pt", default="scratchpad/yolov8n.pt")
    ap.add_argument("--brain-weights", default="scratchpad/yolov8n.brain.weights")
    ap.add_argument("--brain-bin", default="./target/release/brain")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-ultra", type=int, default=20)
    ap.add_argument("--n-brain", type=int, default=5)
    # Per-side device selection (each engine picks its device independently).
    # --device sets both at once; the per-side flags override it.
    ap.add_argument("--device", default=None,
                    help="convenience: set both sides (cpu|gpu). Per-side flags override.")
    ap.add_argument("--torch-device", default=None,
                    help='Ultralytics/torch device: "cpu", "cuda"/"0", ... (default cpu)')
    ap.add_argument("--brain-device", default=None,
                    help='brain BRAIN_DEVICE: "cpu" (Cranelift-JIT+AVX2) or "gpu"/wgpu (default cpu)')
    a = ap.parse_args()

    # Resolve devices: explicit per-side flag > --device > default "cpu". For the
    # torch side, "gpu" is a friendly alias for CUDA device 0.
    def torch_dev(d):
        return "0" if str(d).lower() == "gpu" else d
    torch_device = torch_dev(a.torch_device or a.device or "cpu")
    # brain device precedence: --brain-device > --device > the shell BRAIN_DEVICE
    # env (so `BRAIN_DEVICE=gpu python3 bench.py` actually runs brain on wgpu) >
    # cpu. The env is the engine's own selector, so honouring it here matches the
    # mental model of setting BRAIN_DEVICE for the run.
    brain_device = a.brain_device or a.device or os.environ.get("BRAIN_DEVICE") or "cpu"

    # Honest labels: name the backend each side actually used. brain's CPU path
    # is the native Cranelift-JIT + AVX2 fast-conv backend; otherwise wgpu/WebGPU.
    brain_backend = "native CPU-JIT+AVX2" if str(brain_device).lower() == "cpu" else "wgpu/WebGPU GPU"

    print(f"image={a.image}  imgsz=640  torch_device={torch_device}  brain_device={brain_device}\n")

    ul_load, ul_lat, ul_rss, ul_dets, ul_par = bench_ultralytics(a.image, a.pt, a.n_ultra, torch_device)
    print(_row(f"Ultralytics (torch {ul_par})", ul_load, ul_lat, ul_rss))
    print(f"   dets: {ul_dets[:4]}\n")

    br_load, br_lat, br_rss, br_dets, br_adapter = bench_brain(
        a.image, a.brain_weights, a.brain_bin, a.n_brain, a.conf, brain_device)
    print(_row(f"brain ({brain_backend})", br_load, br_lat, br_rss, "(engine subprocess)"))
    # The engine's own adapter line is ground truth for which backend ran.
    print(f"   engine adapter: {br_adapter or '(none reported)'}")
    print(f"   dets: {br_dets[:4]}\n")

    sx = statistics.median(br_lat) / statistics.median(ul_lat)
    rel = "faster than" if sx < 1 else "the latency of"
    factor = (1 / sx) if sx < 1 else sx
    print(f"==> brain is {factor:.1f}x {rel} Ultralytics; "
          f"RSS {br_rss/ul_rss:.2f}x ({br_rss/1024:.0f} vs {ul_rss/1024:.0f} MiB).")


if __name__ == "__main__":
    sys.exit(main())
