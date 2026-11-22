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

**Single-node reproduction and the full set of blocking calls (2026-11-22):** two ranks on one GH200 node with every
byte forced through the NIC (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`, `srun --network=single_node_vni`, the Slingshot
plugin, no `FI_CXI_DISABLE_HOST_REGISTER`) deadlock on the first try in qwen35's parameter-sharded measure harness (16
layers, block 1024, window 2): the op-timeout bound ends it as "collective 11 was not done 120.0s after the ones before
it". gdb: NCCL's proxy thread in `cxip_req_buf_process_ux -> cxip_ptelist_buf_alloc -> cudaHostRegister ->
pthread_rwlock_wrlock`, the rank's thread in `cuModuleLoadData` (a kernel loaded on its first use, mid-forward). So no
second node and no full stream are needed. A one-card C probe (a stream held on a host word that a second thread
writes only after `cuMemHostRegister`; driver 580.196.07) sorts driver calls on the first thread by whether they hold
the lock while they wait for the held stream. Deadlock: an enqueue once about 1020 operations are queued (722 with
4 KiB of kernel arguments; a graph launch counts one), a copy into pageable memory (sync or async), a copy from
pageable memory above 64 KiB or once the driver's staging (about 2 MiB) is full, `cuMemFree` (even of an unused
block), `cuMemFreeHost`, `cuMemHostUnregister`, `cuModuleLoadData`. No deadlock: `cuStreamSynchronize`,
`cuEventSynchronize`, `cuCtxSynchronize`, `cuStreamQuery`, `cuStreamWaitEvent`, `cuMemAlloc`, `cuMemAllocAsync`,
`cuMemHostAlloc`, `cuMemHostRegister`, `cuModuleUnload`, `cuModuleGetFunction`, graph instantiate/destroy,
`cuMemPoolTrimTo`, stream create/destroy.

**Candidate brain-side fix (not landed, not yet run end to end):** in `backend_cuda::exec::Context`, (1) bound the
run-ahead: mark the stream with an event every 64 enqueued operations and `cuEventSynchronize` the oldest of 4
outstanding marks before marking again (at most ~320 queued), counting a pageable copy as one operation per 4 KiB;
(2) before a pageable copy above 64 KiB and before every download, `cuStreamSynchronize` first; (3) before
`cuMemFree`, `cuMemFreeHost`, `cuMemHostUnregister` and `cuModuleLoadData`, synchronize every open stream of the
context (skipping one being captured). Five backend-cuda tests (`tests/it/driver_lock.rs`: 4096 clears, a download, a
device free, a pinned free, a module load, each behind a held stream while another thread registers host memory)
were red before (the registration never returned) and green after; a sixth (pageable uploads) was written but not
run; the
end-to-end harness run without the variable and the cost measurement were not done. The patch is not in the
repository. Until such a fix is landed and shown to train without
the variable, the variable stays required.

**How to apply:** set `FI_CXI_DISABLE_HOST_REGISTER=1` in every rank's site environment on a CXI fabric (the
cluster sample's `--rank-env` file); brain reads no fabric variables itself. Do not rely on any wait to surface a
hang of this kind: brain bounds every enqueued device collective on the card (`RankCollective::bound_in_flight`,
the stream marks each one done in a page-locked word the watchdog reads without the driver), so the same deadlock
without the variable ends as an error at the plan's op timeout on every rank.
