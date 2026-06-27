#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Head-to-head YOLOv8n inference benchmark across brain's three deployment
targets: the from-scratch WGSL engine on **CPU** (Cranelift JIT) and **GPU**
(wgpu/WebGPU), plus the **Intel NPU** INT8 path (OpenVINO). On CPU/GPU the
default Python/Ultralytics YOLOv8n is benchmarked alongside on the SAME image +
640px for an apples-to-apples compare; the NPU row is brain-only (no Ultralytics
NPU path).

Measures, end-to-end (image -> boxes) on CPU/GPU: model load time, warm
per-image latency (median over N), throughput (img/s), and peak RSS. The NPU row
reports OpenVINO steady-state inference latency (p50/p99) + throughput from
`brain npu bench` (warm session; inference-only, INT8).

Prereqs (dev machine): `pip install ultralytics` (CPU/GPU rows), an exported
brain weights file (tools/yolo_export/export_yolov8.py), and `make release`. The
NPU row also needs OpenVINO + the Intel NPU user-mode driver + level-zero and an
Intel NPU (Meteor Lake / 3720 or newer) at run
time; without them the npu row is reported as skipped with the engine's own
diagnostic. The INT8 ONNX is built on demand (pure-Rust export+quantize, runs
anywhere) if it is missing.

  python3 tools/bench/bench_yolo_inference.py \
      --image scratchpad/dog.png \
      --pt scratchpad/yolov8n.pt \
      --brain-weights scratchpad/yolov8n.brain.weights \
      --brain-bin ./target/release/brain

Device selection — pick WHICH targets to run:
  (default, DEVICE unset)   run all three: cpu, gpu, npu
  DEVICE=cpu|gpu|npu (env)  run ONLY that target, e.g. `DEVICE=npu python3 ...`
  --device cpu|gpu|npu      same, as a flag (overrides the env)
A device that can't run on this machine (no CUDA, no GPU adapter, no NPU) is
reported as skipped and the other targets still run. On cpu/gpu the brain CPU
fast paths (Cranelift JIT + AVX2) apply only on the cpu backend; the printed
"engine adapter:" line is ground truth for which backend ran.
"""
from __future__ import annotations
import argparse, os, resource, statistics, subprocess, sys, time
# NOTE: PIL/torch/ultralytics/brain_py are imported lazily inside the cpu/gpu
# benchmark functions so the npu-only path (`DEVICE=npu`) runs without them.

DEVICES = ("cpu", "gpu", "npu")


def _vmhwm_kb(pid: int) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def _resolve_torch_device(requested):
    """Resolve a requested torch device, falling back when a GPU isn't usable:
    cuda (if available) -> vulkan (if built) -> mps (Apple) -> cpu. Returns
    (effective_device, note) where note explains any fallback."""
    import torch
    d = str(requested).lower()
    if d == "cpu":
        return "cpu", None
    want_cuda = d in ("gpu", "cuda", "0") or d.startswith("cuda")
    if want_cuda and torch.cuda.is_available():
        return ("0" if d == "gpu" else requested), None
    # GPU requested but CUDA not available — try the alternatives in order.
    if getattr(torch, "is_vulkan_available", lambda: False)():
        return "vulkan", f"torch: CUDA unavailable, trying Vulkan (requested {requested})"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    if mps:
        return "mps", f"torch: CUDA unavailable, using MPS (requested {requested})"
    return "cpu", f"torch: no CUDA/Vulkan/MPS available, falling back to CPU (requested {requested})"


def bench_ultralytics(image_path, pt, n, device):
    from ultralytics import YOLO
    from PIL import Image
    import torch
    device, note = _resolve_torch_device(device)
    if note:
        print(f"   [{note}]")
    t0 = time.perf_counter()
    m = YOLO(pt)
    im = Image.open(image_path).convert("RGB")
    # warm-up (first calls pay graph/alloc init). If the chosen accelerator
    # can't actually run the model (e.g. Vulkan op coverage), fall back to CPU.
    try:
        for _ in range(2):
            m.predict(im, imgsz=640, device=device, verbose=False)
    except Exception as e:  # noqa: BLE001 - bench robustness over a clean device
        print(f"   [torch: device {device!r} failed ({type(e).__name__}); using CPU]")
        device = "cpu"
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
    from PIL import Image
    # Surface the engine's own "adapter:" stderr line so the user can SEE which
    # backend actually ran (native CPU-JIT vs the wgpu GPU adapter), instead of
    # trusting the requested device. The client otherwise swallows stderr.
    adapter: dict = {"line": None}
    stages: list = []  # (preprocess, forward, postprocess) ms per frame, under BRAIN_PROFILE
    def _drain(self):
        import re
        for raw in self._proc.stderr:
            s = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
            if adapter["line"] is None and "adapter:" in s:
                adapter["line"] = s.strip()
            if "[detect]" in s:
                nums = re.findall(r"([0-9.]+) ms", s)
                if len(nums) >= 3:
                    stages.append(tuple(float(x) for x in nums[:3]))
    _client_mod.BrainClient._drain_stderr = _drain
    im = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    # `device` is forwarded verbatim as BRAIN_DEVICE for the engine subprocess:
    # "cpu" -> native Cranelift-JIT + AVX2 fast-conv backend; anything else
    # (e.g. "gpu") -> the wgpu/WebGPU backend running the WGSL kernels.
    # Pass the backend explicitly as `--device` (cpu|gpu) so the engine actually
    # builds that backend (the yolo model only switches off an explicit
    # selection); `device=` also sets BRAIN_DEVICE for good measure.
    client = BrainClient(yolo=weights, conf=conf, brain_bin=brain_bin, device=device,
                         extra_args=["--device", device])
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
    # Median engine-side stage split (only populated under BRAIN_PROFILE).
    stage_med = None
    if stages:
        warm = stages[1:] or stages  # drop the cold warm-up frame
        stage_med = tuple(statistics.median(s[i] for s in warm) for i in range(3))
    return load, lat, rss, out, adapter["line"], stage_med


def _npu_env():
    """Environment for the `brain npu` subprocess so the OpenVINO runtime is
    found. The Rust `openvino` binding (runtime-linking) locates `libopenvino_c`
    via LD_LIBRARY_PATH, so we prepend the installed `openvino` pip package's
    libs dir (and make sure the unversioned `libopenvino_c.so` soname the loader
    looks for exists). Returns the child env, or the current env unchanged if the
    openvino package isn't importable (the npu row then reports skipped)."""
    env = dict(os.environ)
    try:
        import openvino  # noqa: F401 - only to locate the install dir
        libs = os.path.join(os.path.dirname(str(openvino.__file__)), "libs")
    except Exception:  # noqa: BLE001 - no openvino installed -> let the engine report it
        return env
    if os.path.isdir(libs):
        soname = os.path.join(libs, "libopenvino_c.so")
        if not os.path.exists(soname):
            # Point the bare soname at the versioned lib (libopenvino_c.so.NNNN).
            for f in sorted(os.listdir(libs)):
                if f.startswith("libopenvino_c.so."):
                    try:
                        os.symlink(f, soname)
                    except OSError:
                        pass
                    break
        env["LD_LIBRARY_PATH"] = libs + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def ensure_npu_onnx(brain_bin, weights, onnx_path, calib, num_calib, image):
    """Build the INT8 ONNX on demand (pure-Rust export+quantize, runs anywhere).

    Returns (onnx_path, note). Raises RuntimeError if the build fails. If the
    requested calibration dir is missing, falls back to the directory holding the
    bench image — enough for a latency benchmark (accuracy isn't measured here)."""
    if os.path.exists(onnx_path):
        return onnx_path, "reused existing"
    if not os.path.isdir(calib):
        calib = os.path.dirname(os.path.abspath(image)) or "."
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    base, ext = os.path.splitext(onnx_path)
    # Strip a trailing ".int8" so the fp32 sibling is e.g. out/yolo.fp32.onnx.
    fp32_path = (base[:-5] if base.endswith(".int8") else base) + ".fp32" + ext

    def _run(stage, argv):
        p = subprocess.run([brain_bin, "npu", *argv], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{stage} failed (exit {p.returncode}): "
                               f"{(p.stderr or p.stdout).strip()}")

    _run("npu export", ["export", "--weights", weights, "--out", fp32_path])
    _run("npu quantize", ["quantize", "--weights", weights, "--calib", calib,
                          "--out", onnx_path, "--num-calib", str(num_calib)])
    return onnx_path, f"built from {weights} (calib {calib}, {num_calib} imgs)"


def _npu_image(image):
    """Return a path `brain npu bench` accepts (binary PPM 'P6' or a dataset
    dir), converting a regular image to a temp PPM via PIL when needed. Returns
    (path, note); path is None if no usable image (the bench then uses its
    constant 640px input — NPU latency is input-independent for a static graph)."""
    if not image:
        return None, None
    if os.path.isdir(image) or image.lower().endswith((".ppm", ".pnm")):
        return image, None
    try:
        from PIL import Image  # PIL writes P6 PPM natively for RGB
        tmp = os.path.join(os.path.dirname(os.path.abspath(image)),
                           ".bench_npu_input.ppm")
        Image.open(image).convert("RGB").save(tmp, format="PPM")
        return tmp, f"converted {image} -> {tmp}"
    except Exception as e:  # noqa: BLE001 - fall back to the constant grey input
        return None, f"using constant 640px input ({type(e).__name__}: {e})"


def bench_npu(brain_bin, onnx_path, npu_device, iters, warmup, image, cache_dir):
    """Steady-state INT8 inference latency via `brain npu bench` (warm OpenVINO
    session). Returns (p50, p99, mean, fps, device, rss_kb). Raises RuntimeError
    with the engine's diagnostic if the device can't run (no OpenVINO/NPU)."""
    argv = [brain_bin, "npu", "bench", "--onnx", onnx_path, "--device", npu_device,
            "--iters", str(iters), "--warmup", str(warmup), "--hint", "latency"]
    if cache_dir:
        argv += ["--cache-dir", cache_dir]
    if image:
        argv += ["--image", image]
    # Popen (not run) so we can poll this child's own peak RSS via /proc — using
    # getrusage(RUSAGE_CHILDREN) would report the max over ALL prior children
    # (the cpu/gpu brain runs, quantize), not the NPU bench. VmHWM is a
    # high-water mark, so sampling while it runs captures the true peak.
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=_npu_env())
    rss = 0
    while proc.poll() is None:
        rss = max(rss, _vmhwm_kb(proc.pid))
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    rss = max(rss, _vmhwm_kb(proc.pid))  # final read in case it grew right before exit
    if proc.returncode != 0:
        raise RuntimeError((stderr or stdout).strip() or f"exit {proc.returncode}")
    vals: dict = {}
    for line in stdout.splitlines():
        parts = line.split()
        if line.startswith("device") and len(parts) >= 2:
            vals["device"] = parts[1]
        elif line.startswith("latency p50"):
            vals["p50"] = float(parts[2])
        elif line.startswith("latency p99"):
            vals["p99"] = float(parts[2])
        elif line.startswith("latency mean"):
            vals["mean"] = float(parts[2])
        elif line.startswith("throughput"):
            vals["fps"] = float(parts[1])
    if "p50" not in vals:
        raise RuntimeError(f"could not parse npu bench output:\n{stdout}")
    return vals["p50"], vals["p99"], vals.get("mean", vals["p50"]), vals.get("fps", 0.0), \
        vals.get("device", npu_device), rss


def _row(name, load, lat, rss_kb, extra=""):
    med = statistics.median(lat)
    return (f"{name:<22} load {load:6.2f}s | latency med {med:8.1f} ms "
            f"(min {min(lat):8.1f}, max {max(lat):8.1f}) | "
            f"{1000/med:7.2f} img/s | peak RSS {rss_kb/1024:7.1f} MiB {extra}")


def run_cpu_gpu(a, device):
    """Benchmark the cpu or gpu target: brain vs Ultralytics on the same image."""
    torch_device = "0" if device == "gpu" else "cpu"
    brain_backend = "native CPU-JIT+AVX2" if device == "cpu" else "wgpu/WebGPU GPU"
    print(f"image={a.image}  imgsz=640  torch_device={torch_device}  brain_device={device}\n")

    ul = None
    try:
        ul_load, ul_lat, ul_rss, ul_dets, ul_par = bench_ultralytics(
            a.image, a.pt, a.n_ultra, torch_device)
        print(_row(f"Ultralytics (torch {ul_par})", ul_load, ul_lat, ul_rss))
        print(f"   dets: {ul_dets[:4]}\n")
        ul = (ul_load, ul_lat, ul_rss)
    except Exception as e:  # noqa: BLE001 - one missing engine shouldn't kill the run
        print(f"   [Ultralytics skipped: {type(e).__name__}: {e}]\n")

    br_load, br_lat, br_rss, br_dets, br_adapter, br_stage = bench_brain(
        a.image, a.brain_weights, a.brain_bin, a.n_brain, a.conf, device)
    print(_row(f"brain ({brain_backend})", br_load, br_lat, br_rss, "(engine subprocess)"))
    print(f"   engine adapter: {br_adapter or '(none reported)'}")
    if br_stage:
        print(f"   engine stages (median, warm): preprocess {br_stage[0]:.1f} | "
              f"forward {br_stage[1]:.1f} | postprocess {br_stage[2]:.1f} ms  (set BRAIN_PROFILE=1)")
    print(f"   dets: {br_dets[:4]}\n")

    if ul:
        ul_load, ul_lat, ul_rss = ul
        sx = statistics.median(br_lat) / statistics.median(ul_lat)
        rel = "faster than" if sx < 1 else "the latency of"
        factor = (1 / sx) if sx < 1 else sx
        print(f"==> brain is {factor:.1f}x {rel} Ultralytics; "
              f"RSS {br_rss/ul_rss:.2f}x ({br_rss/1024:.0f} vs {ul_rss/1024:.0f} MiB).")


def run_npu(a):
    """Benchmark the Intel NPU target (brain-only, INT8 via OpenVINO)."""
    print(f"image={a.image}  imgsz=640  npu_device={a.npu_device}  onnx={a.npu_onnx}\n")
    try:
        onnx_path, note = ensure_npu_onnx(a.brain_bin, a.brain_weights, a.npu_onnx,
                                          a.npu_calib, a.npu_ncalib, a.image)
        print(f"   INT8 ONNX: {onnx_path}  [{note}]")
    except Exception as e:  # noqa: BLE001
        print(f"   [npu skipped: could not build INT8 ONNX: {e}]")
        return
    npu_image, img_note = _npu_image(a.image)
    if img_note:
        print(f"   {img_note}")
    try:
        p50, p99, mean, fps, dev, rss = bench_npu(
            a.brain_bin, onnx_path, a.npu_device, a.npu_iters, a.npu_warmup,
            npu_image, a.npu_cache)
    except Exception as e:  # noqa: BLE001 - no OpenVINO / no NPU hardware here
        print(f"   [npu skipped: {e}]")
        print("   (needs OpenVINO + the Intel NPU user-mode driver + level-zero at run "
              "time; see docs/yolo/NPU.md)")
        return
    print(f"brain (Intel NPU INT8)  device {dev} | latency p50 {p50:7.2f} ms "
          f"(p99 {p99:7.2f}, mean {mean:7.2f}) | {fps:7.2f} img/s | "
          f"peak RSS {rss/1024:7.1f} MiB (inference-only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="scratchpad/dog.png")
    ap.add_argument("--pt", default="scratchpad/yolov8n.pt")
    ap.add_argument("--brain-weights", default="scratchpad/yolov8n.brain.weights")
    ap.add_argument("--brain-bin", default="./target/release/brain")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-ultra", type=int, default=20)
    ap.add_argument("--n-brain", type=int, default=5)
    # Which target(s) to run: cpu|gpu|npu. Unset -> all three. Env DEVICE does
    # the same; --device overrides it. (This selects WHICH backends to benchmark;
    # it is no longer a single shared backend knob.)
    ap.add_argument("--device", default=None,
                    help="run ONLY this target: cpu|gpu|npu (default: all three)")
    # NPU-specific knobs (the npu row is brain-only, INT8 via OpenVINO).
    ap.add_argument("--npu-onnx", default="out/yolo.int8.onnx",
                    help="INT8 ONNX path; built on demand if missing")
    ap.add_argument("--npu-calib", default="data/detect",
                    help="calibration image dir (falls back to the bench image's dir)")
    ap.add_argument("--npu-ncalib", type=int, default=64)
    ap.add_argument("--npu-device", default="NPU", help="OpenVINO device (NPU|CPU|GPU)")
    ap.add_argument("--npu-cache", default="out/npu-cache", help="OpenVINO compile cache dir")
    ap.add_argument("--npu-iters", type=int, default=50)
    ap.add_argument("--npu-warmup", type=int, default=10)
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
            if device == "npu":
                run_npu(a)
            else:
                run_cpu_gpu(a, device)
        except Exception as e:  # noqa: BLE001 - isolate failures per target
            print(f"   [{device} target failed: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    sys.exit(main())
