#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running distributed training on
# scheduler-managed GPU clusters for its clients. If your team needs expertise
# in cluster launch, job membership and fabric selection then you can procure
# our services by sending an email to info@swedishembedded.com.

#
# Run one distributed brain job over the cards of a Slurm allocation: every
# card one rank, all ranks in ONE srun step (so a Slingshot step gets one VNI
# its ranks share). Run it from inside the allocation (salloc, sbatch).
#
#   train.sh [options] -- BRAIN_ARGS...
#
#   # a LoRA fine-tune, every card of two nodes one rank
#   train.sh --nodes 2 -- finetune Qwen/Qwen3-4B --dataset chat.jsonl --lora 8 --out out/
#   # the collective benchmark, requiring the fabric's network plugin
#   train.sh --nodes 2 --rank-env site-nccl.sh -- bench collectives --out result.json --expect-network Libfabric
#
# It writes, in --workdir (default ./slurm-job-<job id>-<time>):
#   plan.json   the job plan every rank reads: world size, coordinator
#               <first node of the step>:<port chosen once>, timeouts
#   job.secret  64 hex digits, mode 0600: only processes that can read it join
#   rank<N>.log, nccl.rank<N>.log   each rank's output and NCCL's log
# and runs `srun --ntasks-per-node=<cards> --gpus-per-node=<cards>` with
# rank.sh as every task, which appends --job-plan/--job-rank/--job-secret to
# BRAIN_ARGS and runs `brain --device gpu<local id> BRAIN_ARGS...` pinned to
# that card's NUMA-local cores.
#
# Options:
#   --nodes N            nodes of the allocation to use (default: all of them)
#   --nodelist LIST      exactly these nodes (Slurm host list); overrides --nodes
#   --cards-per-node N   ranks per node, one per card (default 4)
#   --brain PATH         the brain binary every node runs (default: brain on PATH)
#   --rank-env FILE      a site file every task sources before brain starts
#                        (the NCCL network plugin and fabric settings)
#   --container PREFIX   run every task through PREFIX, one shell word list,
#                        e.g. "apptainer exec --nv -B /nobackup image.sif"
#   --port N             the coordinator's port (default: one drawn from 20000-39999)
#   --collectives PATH   the plan's collectives: auto (default), device, host
#   --op-timeout-ms N    the plan's per-operation timeout (default 600000)
#   --workdir DIR        where the plan, secret and logs go (must be visible to every node)
#   --srun-arg ARG       one more argument for srun (repeatable), e.g. --srun-arg --mem=64G
#
# A step on one node gets a Slingshot VNI only with --network=single_node_vni,
# which this script adds when the step has one node; without a VNI the NCCL
# network plugin cannot open the NIC and NCCL falls back to sockets (on one node
# NCCL's own NVLink path carries the traffic anyway).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

nodes="" nodelist="" cards=4 brain=brain rank_env="" container="" port="" collectives=auto op_timeout_ms=600000 workdir=""
srun_extra=()
usage() {
	sed -n '/^# Run one distributed/,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
	exit "${1:-2}"
}
while [ $# -gt 0 ]; do
	case "$1" in
	--nodes) nodes=$2; shift 2 ;;
	--nodelist) nodelist=$2; shift 2 ;;
	--cards-per-node) cards=$2; shift 2 ;;
	--brain) brain=$2; shift 2 ;;
	--rank-env) rank_env=$(realpath "$2"); shift 2 ;;
	--container) container=$2; shift 2 ;;
	--port) port=$2; shift 2 ;;
	--collectives) collectives=$2; shift 2 ;;
	--op-timeout-ms) op_timeout_ms=$2; shift 2 ;;
	--workdir) workdir=$2; shift 2 ;;
	--srun-arg) srun_extra+=("$2"); shift 2 ;;
	-h | --help) usage 0 ;;
	--) shift; break ;;
	*) echo "train.sh: unknown option $1" >&2; usage ;;
	esac
done
[ $# -gt 0 ] || { echo "train.sh: give the brain command after --" >&2; usage; }
: "${SLURM_JOB_ID:?train.sh runs inside a Slurm allocation (salloc or sbatch)}"
case "$collectives" in auto | device | host) ;; *) echo "train.sh: --collectives is auto, device or host" >&2; exit 2 ;; esac

# The step's nodes, in the order Slurm names them; the first hosts the coordinator.
if [ -z "$nodelist" ]; then
	mapfile -t all < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
	[ -n "$nodes" ] || nodes=${#all[@]}
	[ "$nodes" -le "${#all[@]}" ] || { echo "train.sh: --nodes $nodes, but the allocation has ${#all[@]}" >&2; exit 2; }
	nodelist=$(IFS=,; echo "${all[*]:0:$nodes}")
fi
mapfile -t step_nodes < <(scontrol show hostnames "$nodelist")
nodes=${#step_nodes[@]}
world=$((nodes * cards))
[ -n "$port" ] || port=$((20000 + RANDOM % 20000))
[ -n "$brain" ] && [ "$brain" != brain ] && brain=$(realpath "$brain")

[ -n "$workdir" ] || workdir="$PWD/slurm-job-$SLURM_JOB_ID-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$workdir"
chmod 700 "$workdir"
workdir=$(realpath "$workdir")
job="slurm-$SLURM_JOB_ID-$(basename "$workdir")"
cat >"$workdir/plan.json" <<EOF
{
  "job": "$job",
  "world_size": $world,
  "tp": 1,
  "pp": 1,
  "dp": $world,
  "coordinator": "${step_nodes[0]}:$port",
  "handshake_timeout_ms": 300000,
  "op_timeout_ms": $op_timeout_ms,
  "collectives": "$collectives"
}
EOF
(umask 077 && openssl rand -hex 32 >"$workdir/job.secret")

step=(srun --overlap --nodes="$nodes" --nodelist="$nodelist" --ntasks-per-node="$cards" --gpus-per-node="$cards" --kill-on-bad-exit=1)
[ "$nodes" -eq 1 ] && step+=(--network=single_node_vni)
step+=("${srun_extra[@]}")
read -r -a prefix <<<"$container"

echo "train.sh: $world ranks on ${step_nodes[*]} (coordinator ${step_nodes[0]}:$port), logs in $workdir" >&2
status=0
"${step[@]}" --output="$workdir/rank%t.log" "${prefix[@]}" "$here/rank.sh" job "$workdir" "$rank_env" "$brain" -- "$@" || status=$?
echo "train.sh: the step ended with status $status; rank logs in $workdir" >&2
exit "$status"
