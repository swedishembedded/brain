#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

set -u
cd "$(dirname "$0")/.."
export BRAIN_S3DIT_DIT="$PWD/out/models/Tongyi-MAI/Z-Image-Turbo/transformer"
export BRAIN_S3DIT_VAE="$PWD/out/models/Tongyi-MAI/Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors"
export BRAIN_S3DIT_QWEN="$PWD/out/models/Tongyi-MAI/Z-Image-Turbo/text_encoder"
export BRAIN_S3DIT_TOKENIZER="$PWD/out/models/Tongyi-MAI/Z-Image-Turbo/tokenizer/tokenizer.json"
export BRAIN_LOG_WEIGHTS=1

mkdir -p results
: > results/zimage-int8-256.memlog.csv
echo "epoch_s,rss_kb,avail_kb,cgroup_current_b" > results/zimage-int8-256.memlog.csv

START=$(date +%s)
echo "START:$START" > results/zimage-int8-256.timing.log

./target/release/brain --device gpu s3dit text2image \
    --prompt "a red fox in snow, photograph" --width 256 --height 256 \
    --steps 8 --seed 42 --precision int8 \
    --out image=results/zimage-int8-256.ppm \
    > results/zimage-int8-256.stdout.log 2> results/zimage-int8-256.stderr.log &
MAINPID=$!

while kill -0 "$MAINPID" 2>/dev/null; do
  NOW=$(date +%s)
  RSS=$(grep -m1 VmRSS "/proc/$MAINPID/status" 2>/dev/null | awk '{print $2}')
  AVAIL=$(grep -m1 MemAvailable /proc/meminfo | awk '{print $2}')
  CUR=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  echo "${NOW},${RSS:-0},${AVAIL:-0},${CUR:-0}" >> results/zimage-int8-256.memlog.csv
  sleep 2
done

wait "$MAINPID"
EXIT=$?
END=$(date +%s)
echo "END:$END" >> results/zimage-int8-256.timing.log
echo "EXIT:$EXIT" >> results/zimage-int8-256.timing.log
echo "ELAPSED:$((END-START))s" >> results/zimage-int8-256.timing.log
cat results/zimage-int8-256.timing.log
echo "peak RSS (kB):"
sort -t, -k2 -n -r results/zimage-int8-256.memlog.csv | head -3
exit $EXIT
