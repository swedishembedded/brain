<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 552. libfabric CXI host registration from NCCL's proxy thread deadlocks against a thread blocked in the CUDA driver

**Lesson:** over HPE Slingshot (NCCL 2.32.3, the libfabric network plugin, libfabric 2.3.1 CXI), the CXI
provider replenishes its request buffers from inside NCCL's proxy progress thread and registers each new host
buffer with CUDA (`cxip_req_buf_process_ux -> cxip_ptelist_buf_alloc -> ofi_hmem_host_register ->
cudaHostRegister -> cuMemHostRegister`), which waits for a write lock inside the driver. A rank thread blocked
inside a driver call behind the very collective that needs this progress (an enqueue into a full stream, a launch
under `CUDA_LAUNCH_BLOCKING`) holds that lock as a reader. The proxy never registers, the network never delivers,
the NCCL kernel spins (100% SM, no memory traffic), the stream never drains, and no NCCL call is pending for a
deadline to bound. Found 2026-11-21: `brain finetune` (27B, parameter-sharded over 8 GH200s on two nodes) never
took a step; NCCL's COLL log showed every rank in the same all-gather (31st of the initial loss probe's forward)
with identical arguments, gdb the two stacks above on every rank. The direct harness ran the same model only
because its host thread did not race far enough ahead of the card to fill the stream at the moment the provider
ran out of buffers. With `FI_CXI_DISABLE_HOST_REGISTER=1` the same run trained 2 steps (2.8 and 2.0 s) and exported.

**How to apply:** set `FI_CXI_DISABLE_HOST_REGISTER=1` in every rank's site environment on a CXI fabric (the
cluster sample's `--rank-env` file); brain reads no fabric variables itself. Do not rely on any wait to surface a
hang of this kind: brain bounds every enqueued device collective on the card (`RankCollective::bound_in_flight`,
the stream marks each one done in a page-locked word the watchdog reads without the driver), so the same deadlock
without the variable ends as an error at the plan's op timeout on every rank.
