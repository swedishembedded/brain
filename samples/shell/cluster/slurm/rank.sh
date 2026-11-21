#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running distributed training and
# serving on scheduler-managed GPU clusters for its clients. If your team needs
# expertise in cluster launch, NUMA placement and fabric selection then you can
# procure our services by sending an email to info@swedishembedded.com.

#
# One task of a Slurm step, as train.sh and serve.sh launch it: maps the
# scheduler's rank onto brain's flags, binds the task to its card and that
# card's NUMA-local cores, and runs brain. brain itself reads no scheduler
# variable; this script is where SLURM_* are read.
#
#   rank.sh MODE WORKDIR RANK_ENV BRAIN -- BRAIN_ARGS...
#
# MODE      job   - append --job-plan WORKDIR/plan.json --job-rank N --job-secret WORKDIR/job.secret
#           serve - append --openai HOST:PORT --ready-file ... for this card's server (PORT = WORKDIR/base-port + local id)
# RANK_ENV  a site file sourced in the task before brain starts (NCCL network plugin, fabric settings), or "" for none.
#           It is sourced here, inside the task (and inside the container when there is one), because a site may
#           re-export its own defaults into every task.
# BRAIN     the brain binary to run.
#
# The card is the task's local id on its node. The task is pinned with taskset to
# the cores the kernel lists as local to that card's PCI device
# (/sys/bus/pci/devices/<bus>/local_cpulist, bus id from nvidia-smi); where that
# is not known it runs unpinned and says so.
set -euo pipefail

mode=$1 workdir=$2 rank_env=$3 brain=$4
shift 4
[ "${1:-}" = -- ] && shift

rank=${SLURM_PROCID:?rank.sh runs as a task of a Slurm step (srun)}
local_id=${SLURM_LOCALID:?rank.sh runs as a task of a Slurm step (srun)}
card=$local_id
host=$(hostname -s)

if [ -n "$rank_env" ]; then
	# shellcheck disable=SC1090
	. "$rank_env"
fi

# The cores next to the card, or nothing to pin to.
card_cpus() {
	command -v nvidia-smi >/dev/null 2>&1 || return 1
	local bus domain rest
	bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" 2>/dev/null | tr 'A-F' 'a-f' | tr -d ' ')
	[ -n "$bus" ] || return 1
	domain=${bus%%:*} rest=${bus#*:}
	domain=$(printf '%04x' "$((16#${domain}))")
	local list="/sys/bus/pci/devices/$domain:$rest/local_cpulist"
	[ -r "$list" ] || return 1
	cat "$list"
}
pin=()
if cpus=$(card_cpus "$card") && [ -n "$cpus" ]; then
	pin=(taskset -c "$cpus")
else
	echo "rank.sh: rank $rank: the cores of card $card are not known here; running unpinned" >&2
fi

# NCCL logs its transport choice per rank, so a run can show (and a check can
# require) the network it actually used. Set here, after the site's file and
# over the site's own NCCL_DEBUG (a site may inject VERSION into every step),
# and to a file, so the task's output stays brain's.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_DEBUG_FILE="$workdir/nccl.rank$rank.log"

args=("$@")
case "$mode" in
job)
	extra=(--job-plan "$workdir/plan.json" --job-rank "$rank" --job-secret "$workdir/job.secret")
	if [ "${args[0]:-}" = bench ] && [ "${args[1]:-}" = collectives ]; then
		extra+=(--nccl-log "$NCCL_DEBUG_FILE")
	fi
	;;
serve)
	port=$(($(cat "$workdir/base-port") + local_id))
	# brain binds an address, not a name: the node's address as its name resolves for the other nodes.
	addr=$(getent ahostsv4 "$host" | awk 'NR == 1 { print $1 }')
	[ -n "$addr" ] || { echo "rank.sh: $host does not resolve to an IPv4 address" >&2; exit 2; }
	extra=(--openai "$addr:$port" --runtime-dir "$workdir/rt.$rank" --ready-file "$workdir/ready.$rank")
	mkdir -m 700 "$workdir/rt.$rank"
	echo "http://$addr:$port" >"$workdir/url.$rank"
	;;
*)
	echo "rank.sh: MODE is job or serve, not $mode" >&2
	exit 2
	;;
esac

echo "rank.sh: rank $rank on $host card $card${pin:+ cores ${pin[2]}}" >&2
exec "${pin[@]}" "$brain" --device "gpu$card" "${args[@]}" "${extra[@]}"
