#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible GPU build and
# validation environments for its clients. If your team needs expertise in CUDA
# toolchains on Arm servers then you can procure our services by sending an
# email to info@swedishembedded.com.

"""Install a CUDA toolkit lane into a user-owned prefix, without root.

NVIDIA publishes the toolkit, cuDNN and the profilers as redistributable
archives with a JSON manifest carrying a SHA-256 per file, and NCCL as a PyPI
wheel. This script downloads the pieces brain's CUDA backend and its optional
library providers use (NVRTC, nvcc, cuBLAS/cuBLASLt, cuDNN, NCCL, NVTX, CUPTI)
into one directory per lane, verifies every download, and prints the
environment a shell needs. Nothing is installed system-wide.

    install-cuda-userspace.py install 12     # CUDA 12.x lane (Pascal..Hopper)
    install-cuda-userspace.py install 13     # CUDA 13.x lane (Hopper and newer)
    install-cuda-userspace.py install nsight # Nsight Systems + Compute (any lane)
    install-cuda-userspace.py env 13         # `eval "$(... env 13)"` in a shell

The prefix is `$BRAIN_CUDA_PREFIX`, else `~/.local/cuda`. Lanes are pinned
below so a lane means the same thing on every machine; bump a pin on purpose.
"""

import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REDIST = "https://developer.download.nvidia.com/compute"

# lane -> (toolkit release, cuDNN release, cuDNN flavour, NCCL wheel, NCCL version)
LANES = {
    "12": {"toolkit": "12.9.1", "cudnn": "9.27.0", "cudnn_flavour": "cuda12", "nccl": "nvidia-nccl-cu12", "nccl_version": "2.32.3"},
    "13": {"toolkit": "13.0.2", "cudnn": "9.27.0", "cudnn_flavour": "cuda13", "nccl": "nvidia-nccl-cu13", "nccl_version": "2.32.3"},
}

# Toolkit components, by manifest key. Components a release does not ship are
# skipped (CUDA 13 split NVVM and the PTX compiler out of nvcc, CUDA 12 did not).
TOOLKIT = [
    "cuda_nvcc", "cuda_nvrtc", "cuda_cudart", "cuda_cccl", "cuda_crt", "cuda_nvtx",
    "cuda_cupti", "cuda_profiler_api", "cuda_nvml_dev", "cuda_cuobjdump", "cuda_nvdisasm",
    "cuda_sanitizer_api", "libcublas", "libnvjitlink", "libnvfatbin", "libnvvm", "libnvptxcompiler",
]
PROFILERS = ["nsight_compute", "nsight_systems"]


def arch() -> str:
    """The redistributable platform key for this machine."""
    machine = platform.machine()
    keys = {"aarch64": "linux-sbsa", "x86_64": "linux-x86_64"}
    if machine not in keys:
        sys.exit(f"install-cuda-userspace: no NVIDIA redistributables for {machine}")
    return keys[machine]


def prefix() -> Path:
    return Path(os.environ.get("BRAIN_CUDA_PREFIX") or Path.home() / ".local" / "cuda")


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download(url: str, sha256: str, dest: Path) -> Path:
    """Download `url` to `dest` unless a verified copy is already there."""
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        part.unlink()
        sys.exit(f"install-cuda-userspace: checksum mismatch for {url}")
    part.rename(dest)
    return dest


def merge_archive(archive: Path, into: Path) -> None:
    """Unpack a redistributable archive (one top directory) over `into`."""
    with tarfile.open(archive) as tar:
        top = tar.getnames()[0].split("/")[0]
        tar.extractall(into.parent / ".unpack", filter="data")
    src = into.parent / ".unpack" / top
    for path in sorted(src.rglob("*")):
        target = into / path.relative_to(src)
        if path.is_dir() and not path.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
    shutil.rmtree(into.parent / ".unpack")


def lane_dir(lane: str) -> Path:
    return prefix() / f"cuda-{LANES[lane]['toolkit']}"


def install_manifest(manifest: dict, keys: list, plat: str, flavour: str | None, dest: Path, cache: Path) -> None:
    jobs = []
    for key in keys:
        entry = manifest.get(key, {}).get(plat) if isinstance(manifest.get(key), dict) else None
        if entry is None:
            continue
        if flavour is not None:
            entry = entry[flavour]
        jobs.append((entry["relative_path"], entry["sha256"]))

    def one(job):
        rel, sha = job
        base = manifest["_base"]
        return download(f"{base}/{rel}", sha, cache / Path(rel).name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(pool.map(one, jobs))
    for archive in archives:
        merge_archive(archive, dest)


def install_lane(lane: str) -> None:
    spec, plat = LANES[lane], arch()
    dest, cache = lane_dir(lane), prefix() / ".downloads"
    dest.mkdir(parents=True, exist_ok=True)

    toolkit = fetch_json(f"{REDIST}/cuda/redist/redistrib_{spec['toolkit']}.json")
    toolkit["_base"] = f"{REDIST}/cuda/redist"
    install_manifest(toolkit, TOOLKIT, plat, None, dest, cache)

    cudnn = fetch_json(f"{REDIST}/cudnn/redist/redistrib_{spec['cudnn']}.json")
    cudnn["_base"] = f"{REDIST}/cudnn/redist"
    install_manifest(cudnn, ["cudnn"], plat, spec["cudnn_flavour"], dest, cache)

    install_nccl(spec, dest, cache)
    if not (dest / "lib64").exists() and (dest / "lib").is_dir():
        (dest / "lib64").symlink_to("lib")
    print(f"installed CUDA {spec['toolkit']} lane into {dest}")


def install_nccl(spec: dict, dest: Path, cache: Path) -> None:
    """NCCL is distributed as a PyPI wheel; take the one for this machine."""
    meta = fetch_json(f"https://pypi.org/pypi/{spec['nccl']}/{spec['nccl_version']}/json")
    tag = platform.machine()
    wheels = [f for f in meta["urls"] if f["filename"].endswith(".whl") and tag in f["filename"]]
    if not wheels:
        sys.exit(f"install-cuda-userspace: no {spec['nccl']} wheel for {tag}")
    wheel = download(wheels[0]["url"], wheels[0]["digests"]["sha256"], cache / wheels[0]["filename"])
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.startswith("nvidia/nccl/") and not name.endswith("/"):
                out = dest / name.removeprefix("nvidia/nccl/")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))


def install_profilers() -> None:
    plat, dest, cache = arch(), prefix() / "nsight", prefix() / ".downloads"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = fetch_json(f"{REDIST}/cuda/redist/redistrib_{LANES['13']['toolkit']}.json")
    manifest["_base"] = f"{REDIST}/cuda/redist"
    install_manifest(manifest, PROFILERS, plat, None, dest, cache)
    print(f"installed Nsight into {dest}")


def print_env(lane: str) -> None:
    root = lane_dir(lane)
    print(f'export CUDA_PATH="{root}"')
    print(f'export PATH="{root}/bin:{prefix()}/nsight/bin:$PATH"')
    print(f'export LD_LIBRARY_PATH="{root}/lib64:${{LD_LIBRARY_PATH:-}}"')


def main(argv: list) -> None:
    if len(argv) == 3 and argv[1] == "install" and argv[2] in LANES:
        install_lane(argv[2])
    elif argv[1:3] == ["install", "nsight"]:
        install_profilers()
    elif len(argv) == 3 and argv[1] == "env" and argv[2] in LANES:
        print_env(argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
