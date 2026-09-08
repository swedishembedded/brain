#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Multi-GPU sharding gate (`make check/scripts`).
#
# `crates/model/src/shard.rs` is this workspace's ONE generic multi-GPU
# pipeline-placement mechanism (`Shard`/`Shardable`/`plan_balanced`/
# `plan_fewest_devices`) - proven, in real use by several large model crates
# (`gpt2`, `qwen3`, `qwen35`, `qwen35moe`, `qwen3omnimoe`, `fastvlm`, `llava`,
# `minimaxmusic3`, `s3dit`, `ltxv`). It exists specifically so a model too big
# for one GPU's VRAM is split across every ambient GPU instead of OOMing on
# GPU 0 while a second (or third...) card sits idle.
#
# The failure mode this gate exists to catch happened for real: `minimaxh3`
# (a 33B-param diffusion transformer) shipped calling `gpu_core::Gpu::open`
# directly, on ONE device, with zero awareness a second physical GPU existed
# - discovered only after real generation runs repeatedly OOM'd on GPU 0 with
# GPU 1 sitting at 1 MiB used the entire time. Nothing forced that crate to
# even consider the existing, already-proven `model::shard` machinery; it was
# just as easy to open one device and move on. This gate is what makes that
# omission loud instead of silent for every NEW `arch!()`-registered crate
# from here on.
#
# WHAT IT CHECKS: every `crates/arch/src/lib.rs` `arch!(...)` row names a
# real crate (`crates/<id>/src/`); that crate's source must reference
# `model::shard`/`Shardable` SOMEWHERE, or carry an ALLOWLIST row below with a
# real, specific reason (the model is small enough that fp32 residency fits
# one GPU comfortably, or it is a component consumed by another crate's own
# already-sharded pipeline, or it is a genuinely large model not yet migrated
# - each an honest, checkable claim, never a rubber stamp).
#
# This is a textual presence check, not a proof of CORRECT sharding (a crate
# could import `model::shard` and still get the split wrong) - same
# limitation `check-kernel-selection.sh`'s own header names for its own
# greps. It is still worth having: it turns "nobody thought about this at
# all" into "a human had to make and write down a real decision," which is
# exactly the gap that let minimaxh3 ship single-GPU-only unnoticed.
#
# A row that no longer matches (the crate now DOES reference model::shard)
# makes the gate FAIL - a stale allow-list entry is exactly as much rot as a
# missing one, and is this gate's own reminder to delete the row once a
# backlog item like minimaxh3's is actually fixed.
#
# Usage: scripts/gates/check-multi-gpu-sharding.sh   (exits non-zero listing
# every unallowed crate and every stale allow-list row)
set -uo pipefail
cd "$(dirname "$0")/../.."

# id<TAB>reason. Keep sorted by id so a diff shows exactly what changed.
ALLOWLIST=$(cat <<'EOF'
arcface	Vision face-embedding backbone (IResNet-100), well under 1B params - single GPU by design.
campplus	Speaker-embedding backbone (D-TDNN), well under 1B params - single GPU by design.
chronos2	Encoder-only patch forecasting transformer, small (forecasting-scale, not LLM-scale) - single GPU by design.
clip	CLIP/OpenCLIP/EVA-CLIP text+image towers, at most a few B params (bigG) - fits one GPU's fp32 budget comfortably.
codeformer	Blind face restoration (VQGAN-scale), well under 1B params - single GPU by design.
controlnet	A conditioning adapter added on top of a host backbone, not an independently large model - single GPU by design.
cosyvoice	default_ref FunAudioLLM/CosyVoice2-0.5B - 0.5B, single GPU by design.
deepseek2	MoE decoder, large by architecture family - not yet migrated onto model::shard; backlog.
deepseek2ocr	DeepSeek-V2-family decoder + SAM/CLIP encoder, large by architecture family - not yet migrated onto model::shard; backlog.
deepseekocr2	Same DeepSeek-V2-family decoder as deepseek2ocr (unmodified) plus a new SAM/Qwen2-resampler encoder - not yet migrated onto model::shard; backlog.
diamond	EDM diffusion world model, forecasting/world-model scale, not LLM-scale - single GPU by design.
ecapatdnn	Speaker-embedding backbone (ECAPA-TDNN), well under 1B params - single GPU by design.
fincast	Patched decoder + sparse MoE at forecasting scale, not LLM-scale - single GPU by design.
flux1	FLUX.1 dev/Kontext/schnell is a ~12B MMDiT - fp32-promoted residency on Pascal likely exceeds one 24GB GPU; not yet migrated onto model::shard; backlog.
flux2	default_ref black-forest-labs/FLUX.2-klein-4B - the shipped Klein distillation is 4B, single GPU by design.
gemma4	LTX-2.5's own text encoder component, run inside ltxv's pipeline (which already uses model::shard for its own DiT) - not an independently sharded model in its own right.
genieredux	ST-transformer world model, forecasting/world-model scale, not LLM-scale - single GPU by design.
glmdsa	MoE decoder (MLA + sigmoid noaux_tc MoE + DSA), large by architecture family - not yet migrated onto model::shard; backlog.
instantid	SDXL + IP-Adapter-FaceID identity conditioning, adapter-scale on top of a host backbone - single GPU by design.
kronos	default_ref NeoQuasar/Kronos-base, a candlestick-scale forecasting transformer - single GPU by design.
lfm2	default_ref LiquidAI/LFM2.5-350M - 350M, single GPU by design.
mimi	Neural audio codec, well under 1B params - single GPU by design.
moondream3	SigLIP + MoE decoder - not yet audited/migrated onto model::shard; backlog.
nemotronasr	default_ref nemotron-3.5-asr-streaming-0.6b - 0.6B, single GPU by design.
pulid	PuLID-FLUX identity conditioning, adapter-scale on top of a host backbone - single GPU by design.
qwen3asr	default_ref Qwen/Qwen3-ASR-1.7B - 1.7B, single GPU by design.
qwen3tts	default_ref Qwen/Qwen3-TTS-12Hz-0.6B-Base - 0.6B, single GPU by design.
qwen3vl	default_ref Qwen/Qwen3-VL-4B-Instruct - 4B, single GPU by design.
qwen3vlmoe	Qwen3-VL-30B-A3B, large by architecture family - not yet migrated onto model::shard; backlog.
rrdbnet	Real-ESRGAN RRDBNet super-resolution, well under 1B params - single GPU by design.
s3tokenizer	FSQ supervised-semantic speech tokenizer, well under 1B params - single GPU by design.
sam1	ViTDet ViT-B tower, well under 1B params - single GPU by design.
sam2	default_ref facebook/sam2.1-hiera-tiny - single GPU by design.
scrfd	Face detector, well under 1B params - single GPU by design.
sdxlunet	SDXL UNet2DConditionModel, ~2.6B - fits one GPU's fp32 budget comfortably.
splat	3D Gaussian Splatting rasterizer, well under 1B params - single GPU by design.
supir	SDXL + GLVControl + ZeroSFT photo restoration - not yet audited/migrated onto model::shard; backlog.
t5encoder	T5-XXL is ~11B - fp32-promoted residency on Pascal likely exceeds one 24GB GPU; not yet migrated onto model::shard; backlog.
timesfm3	default_ref google/timesfm-3.0-pytorch, a forecasting-scale mixing transformer - single GPU by design.
toyautoencoder	Toy task, trivially small - single GPU by design.
toymoe	Toy task, trivially small - single GPU by design.
toypid	Toy task, trivially small - single GPU by design.
toyseq2seq	Toy task, trivially small - single GPU by design.
vqgan	VQ autoencoder, well under 1B params - single GPU by design.
wan	Crate spans Wan2.1/2.2 including 14B-class variants (default_ref itself is the small 1.3B one) - not yet migrated onto model::shard for the larger variants; backlog.
worldmirror2	Multi-view 3D reconstruction, not LLM-scale - single GPU by design.
yolov8	Anchor-free detector, well under 1B params - single GPU by design.
zipdepth	Pure-conv monocular depth, well under 1B params - single GPU by design.
EOF
)

out=$(python3 - "$ALLOWLIST" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(".")
arch_text = (root / "crates/arch/src/lib.rs").read_text()
stmts = re.findall(r'arch!\(.*?\),\n', arch_text, re.S)

allow = {}
for line in sys.argv[1].splitlines():
    line = line.rstrip("\n")
    if not line.strip():
        continue
    id_, reason = line.split("\t", 1)
    allow[id_] = reason
allow_seen = set()

missing = []
checked = 0
for s in stmts:
    m = re.match(r'arch!\("([a-z0-9]+)"', s)
    if not m:
        continue
    id_ = m.group(1)
    crate_dir = root / "crates" / id_ / "src"
    if not crate_dir.is_dir():
        # Not every arch! id is its own crate with its own src/ (e.g. a role
        # served by a shared crate under a different directory name) - this
        # gate only judges crates it can actually find sources for.
        continue
    checked += 1
    shards = any(
        "model::shard" in f.read_text(errors="replace") or "Shardable" in f.read_text(errors="replace")
        for f in crate_dir.rglob("*.rs")
    )
    if shards:
        continue
    if id_ in allow:
        allow_seen.add(id_)
        continue
    missing.append(id_)

print(f"check-multi-gpu-sharding: {checked} arch! crate(s) checked, {len(allow)} allow-list row(s), {len(allow_seen)} matched a real gap")

fail = False
if missing:
    fail = True
    print(f"\ncheck-multi-gpu-sharding: {len(missing)} crate(s) neither use model::shard nor carry an allow-list row:\n")
    for id_ in missing:
        print(f"  {id_}")
    print(
        "\n  Fix by wiring the crate onto model::shard's Shard/Shardable/plan_balanced "
        "machinery (see crates/ltxv/src/shard.rs or crates/qwen35/src/shard.rs for a real "
        "precedent), or add an allow-list row in scripts/gates/check-multi-gpu-sharding.sh "
        "with a real, checkable reason if this model is genuinely small enough to never need it."
    )

stale = sorted(set(allow) - allow_seen)
if stale:
    fail = True
    print(f"\ncheck-multi-gpu-sharding: {len(stale)} allow-list row(s) no longer match any real gap (stale - remove them):\n")
    for id_ in stale:
        print(f"  {id_}")

sys.exit(1 if fail else 0)
PY
)
rc=$?
echo "$out"
echo
if [ "$rc" -eq 0 ]; then
  echo "CHECK/MULTI-GPU-SHARDING: PASS"
else
  echo "CHECK/MULTI-GPU-SHARDING: FAIL"
fi
exit "$rc"
