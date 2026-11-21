<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 541. A single-node Slurm step has no Slingshot VNI, so NCCL silently falls back from the fabric

**Lesson:** on a Slingshot (HPE CXI) cluster the fabric is usable only by processes of a scheduler step that was
given a VNI, and Slurm's Slingshot switch plugin gives one to a step that spans nodes but not to a single-node step
unless it is launched with `srun --network=single_node_vni`. A rank started any other way (a single-node step without
the flag, or a process started outside the scheduler) loads the libfabric network plugin, fails
`cxip_gen_auth_key` with -38 (`Function not implemented`) after "Failed to find valid default rgroup and vni", and NCCL
continues on its next transport without an error. Measured 2026-11-21 on GH200 nodes (job 5600435): the same step
over the plugin's `NET/Libfabric/<n>/GDRDMA` path moves 12.6 GB/s all-reduce bus bandwidth between two single-card
ranks, over NCCL's own sockets on `hsn*` 9.8 GB/s; 8 ranks (4 per node) reach 149 GB/s all-reduce and 92 GB/s
all-gather/reduce-scatter at 1 GiB, which is the four NICs per node near their line rate.

**How to apply:** a cluster launcher always passes `--network=single_node_vni` for single-node steps and launches
every rank as a task of one step, and a distributed run checks the transport NCCL reports (`NCCL_DEBUG=INFO`
"via NET/...") rather than assuming it; the C0 measurements live in `.agents/roadmap/distributed-execution.md`.
