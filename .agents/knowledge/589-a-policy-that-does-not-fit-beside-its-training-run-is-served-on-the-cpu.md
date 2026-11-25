<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 589. A policy that does not fit beside its training run is served on the CPU

**Lesson:** an online run keeps two copies of the base on one card: the serving engine's (fp32 for a checkpoint that
fits, `--weight-format` otherwise) and the trainer's bf16 copy, whose reservation is sized to `max_block` (default
8192 tokens). On a card with about 34 GB free (another tenant held 63 GB), the Qwen3-4B stream job reserved 26.4 GB
for 8192-token rows, and the served Qwen3-4B was then built on the CPU (`brain: built Qwen/Qwen3-4B[default] on
Cpu`): rollouts decoded on the host, and the run reached no step in the ten minutes it was watched. That log line
was the only sign. With `max_block: 1024` (the rows were a few hundred tokens) the reservation was 16.0 GB, and with
`--weight-format bf16 --kv-cache f32 --ctx 2048` the served model fit (`built Qwen/Qwen3-4B[default] on Gpu(0)`);
the run added at most 28.8 GB to the card. Size `max_block` to the longest record and the serving format to the
card. The frozen base copy the two could share is the colocation package (A37-A39). Open (no silent degradation): a
served model that falls back to the host while a GPU is configured should be refused, or at least warned about
loudly; it should not decode at CPU speed unannounced.

## Evidence (2026-11-25, lane agrpo-real, brain 760c04d3b)

First 4B profile attempt (`--kv-cache f32 --ctx 2048`, default `max_block`): `brain/train[train:26363026841]` on
Gpu(0), `ChatFineTune: ... 8192-token row need about 19 GiB`, then the served base on Cpu; stopped after ten
minutes. Run fit4b (`max_block: 1024`, `--weight-format bf16`): `brain/train[train:16027392204]`, served base on
Gpu(0) in 3.2 s, card at 63.5 GB before and 91.9 GB at peak. The logs were node-local and were lost in a container
restart the same day; the lines quoted were read from them first.
