<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 6. Timing without `poll_wait()` measures the host

`WgpuBackend::submit` with an empty clear list only appends to `pending` - it
encodes and queues nothing, so a loop of bare `submit`s times host-side
bind-group construction and reports it as device bandwidth. It produced
**377 GB/s on a ~346 GB/s card**. Compute the roof first: a result above the
physical limit means you measured the CPU. See `.agents/rules/kernels.md` §E.0.
