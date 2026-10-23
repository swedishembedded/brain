<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 357. A scoring program that never selected a backend ran on Vulkan and died on a 27B embedding binding

Question: why did the 27B in-process scoring pass load its base for 339 s on a GH200 and then fail in
`Device::create_bind_group` with "Buffer binding 2 range 2542796800 exceeds max_*_buffer_binding_size limit
2147483644"?

Cause: `crates/sdk/examples/score_tokens_qwen35.rs` resolved its device and built the trainer without ever asking for a
backend, so the process ran on the probe's preferred backend, Vulkan (wgpu). The 27B embedding table held as one bf16
storage buffer is a 2.54 GB binding, past wgpu's 2 GB `max_storage_buffer_binding_size`; CUDA, the backend the same
model trains on, has no such limit. `BRAIN_BACKEND=cuda` does not rescue it: that variable is read only on
`resolve_ambient_compute_set`, while an SDK caller that resolves its own `DeviceSpec` goes through
`sdk::device::resolve`, which applies `apply_requested_backend` and nothing else.

Fix: the example takes `--backend` and calls `brain::select_backend` before the first pipeline is built, the same
entry point `chat_finetune` uses (commit 36e902322). Rule: a program that is not the brain CLI selects its backend
explicitly, and a scoring or training path that "worked on the toy fixture" is checked on the real model's largest
single binding before a long run is scheduled.
