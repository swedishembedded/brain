<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Training and serving on a Slurm cluster

Three scripts that run brain over every card of a Slurm allocation. brain
itself reads no scheduler variable: these scripts read the allocation and map
it onto brain's own flags (`--job-plan`, `--job-rank`, `--job-secret`,
`--device gpuN`, `--openai HOST:PORT`).

| script | what it does |
|---|---|
| `train.sh` | one distributed brain job: writes the job plan and an owner-only job secret, launches ONE `srun` step with one task per card, every task a rank |
| `serve.sh` | one `brain serve` per card (one step) and the brain router in front of them: one OpenAI-compatible URL for the whole allocation |
| `rank.sh` | what every task runs: maps `SLURM_PROCID`/`SLURM_LOCALID` onto the rank and the card, pins the task to the card's NUMA-local cores, sources an optional site file, runs brain |

## What it needs

- A Slurm allocation (`salloc`, or an `sbatch` script) whose nodes have GPUs,
  and `srun`/`scontrol` where you run the scripts.
- The `brain` binary on every node (`--brain PATH` on a shared filesystem),
  and a working directory every node can read and write (`--workdir`, by
  default under the current directory).
- `openssl` (the job secret and the keys), `nvidia-smi` and `taskset` in the
  task's environment for the core binding (without them a task runs unpinned
  and says so).

## Train

```bash
# a LoRA fine-tune over every card of two nodes, one rank per card
samples/shell/cluster/slurm/train.sh --nodes 2 --brain /shared/brain -- \
    finetune Qwen/Qwen3-4B --dataset chat.jsonl --lora 8 --out /shared/adapter
```

Everything after `--` is the brain command; every rank runs it with
`--job-plan`, `--job-rank` and `--job-secret` appended and `--device gpuN`
before it. The plan names the first node of the step as the coordinator, on a
port drawn once (`--port` to choose it); `--collectives device` makes a rank
that cannot reduce on its card fail instead of falling back to the host
transport.

A job that commits its state survives losing a rank. With
`finetune --checkpoint-every N`, every N steps each rank writes its exact-resume
state and the job commits the generation once all are durable; a rank whose
job loses a peer stops before applying the step it was in and exits with
status 75. `train.sh --restarts K` then starts every rank again with the same
plan and secret, and the run resumes from the newest committed generation
with the weights the uninterrupted run would have had:

```bash
samples/shell/cluster/slurm/train.sh --nodes 2 --restarts 3 -- \
    finetune Qwen/Qwen3-4B --dataset chat.jsonl --lora 8 --out /shared/adapter --checkpoint-every 50
```

The same launcher measures the fabric with the collective benchmark, which
writes one result document (`brain bench collectives --help`):

```bash
samples/shell/cluster/slurm/train.sh --nodes 2 --rank-env site-nccl.sh -- \
    bench collectives --out /shared/collectives.json --expect-network Libfabric
```

`--expect-network` fails the run unless NCCL chose that network on every rank
(read from each rank's NCCL log, which `rank.sh` turns on) and, across nodes, a
channel crosses it: a silent fall back to TCP sockets is a failure.

## Serve

```bash
samples/shell/cluster/slurm/serve.sh --nodes 2 --brain /shared/brain -- --models-dir /shared/models
```

It prints the router's URL, the file holding its key and the list of
servers once all are up, and stops everything on Ctrl-C. The router
(`brain serve --upstream`) sends a conversation back to the server that holds
its prefix unless that one is far busier, keeps every server at most at its
decode slots and queues the rest, and reports each server's load and token
rate on `/v1/metrics`.

On a node shared with someone else, claim the node's cards but use only yours:
`--cards-per-node 1 --gpus-per-node 4 --card-map nodeA=0,nodeB=1` runs one rank
per node, on card 0 of `nodeA` and card 1 of `nodeB`.

## Containers and site settings

`--container PREFIX` runs every task through a container runtime; PREFIX is one
word list, given as one argument. `--rank-env FILE` is sourced inside every
task (inside the container) before brain starts: the place for the NCCL network
plugin and fabric settings a site needs, set in the task because a site may
re-export its own defaults into every step. On an HPE Slingshot cluster with an
Apptainer image, for example:

```bash
samples/shell/cluster/slurm/train.sh --nodes 2 \
    --container "env -u APPTAINER_BIND -u LD_LIBRARY_PATH apptainer exec --nv -B /shared image.sif" \
    --rank-env /shared/nccl-ofi-env.sh --srun-arg --mem=64G -- bench collectives --out /shared/c.json --expect-network Libfabric
```

where `nccl-ofi-env.sh` puts the libfabric plugin on the library path and sets
`NCCL_NET_PLUGIN=ofi`, `NCCL_NET=Libfabric` (a plugin that fails to load is
then an error, not a fall back), `FI_PROVIDER=cxi` and
`FI_CXI_DISABLE_HOST_REGISTER=1`: with host registration on, the CXI provider
registers new request buffers with CUDA from NCCL's proxy thread, which
deadlocks against a rank blocked in the CUDA driver behind a collective (the
proxy waits for a driver lock that rank holds while the collective waits for
the proxy). It is still required (2026-11-22: the deadlock reproduces on one node
too and brain does not yet avoid it; knowledge 552). A Slingshot step gets a
network VNI only when it spans nodes, or with `--network=single_node_vni` on
one node, which `train.sh` adds for a one-node step.

The cluster end-to-end suite runs these scripts for real:
`make test/e2e/cluster` (it skips by name outside an allocation).
