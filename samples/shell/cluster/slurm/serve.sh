#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for serving models on every card of
# a GPU cluster behind one endpoint for its clients. If your team needs
# expertise in multi-node LLM serving then you can procure our services by
# sending an email to info@swedishembedded.com.

#
# Serve over every card of a Slurm allocation behind ONE OpenAI-compatible URL:
# one `brain serve` per card (one srun step, one task per card), and the brain
# router (`brain serve --upstream`) in front of them on this machine. Run it
# from inside the allocation; Ctrl-C stops the router and the servers.
#
#   serve.sh [options] -- BRAIN_SERVE_ARGS...
#
#   serve.sh --nodes 1 -- --models-dir /models
#   serve.sh --nodes 2 --container "apptainer exec --nv image.sif" -- --models-dir /models --mock
#
# Every server is started as `brain --device gpu<local id> serve
# BRAIN_SERVE_ARGS... --openai <node>:<port> --api-key-file <upstream key>`
# (plus --insecure-plaintext, see below); the router serves the OpenAI dialect
# on --listen with its own key, presents the upstream key to the servers and
# reads the served models' tokenizers from the --models-dir among
# BRAIN_SERVE_ARGS. It prints, once every server is up:
#
#   URL  http://127.0.0.1:18080/v1
#   KEY  <workdir>/router.key
#   ENDPOINTS <workdir>/upstreams.txt   (one upstream URL per line)
#
# Options:
#   --nodes N / --nodelist LIST / --cards-per-node N / --brain PATH /
#   --rank-env FILE / --container PREFIX / --workdir DIR / --srun-arg ARG
#                        as for train.sh
#   --base-port N        the first card's port on every node (default 18800; card i
#                        serves on N+i)
#   --listen ADDR        the router's address (default 127.0.0.1:18080)
#   --ready-timeout S    how long the servers may take to start (default 1800)
#
# The servers and the router speak plaintext HTTP between nodes
# (--insecure-plaintext), the key in every request: right on a cluster's
# private fabric, wrong on a network anyone else can read (give the servers
# --tls-cert/--tls-key and the router --upstream-ca instead).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

nodes="" nodelist="" cards=4 brain=brain rank_env="" container="" workdir="" base_port=18800 listen=127.0.0.1:18080 ready_timeout=1800
srun_extra=()
usage() {
	sed -n '/^# Serve over every card/,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
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
	--workdir) workdir=$2; shift 2 ;;
	--srun-arg) srun_extra+=("$2"); shift 2 ;;
	--base-port) base_port=$2; shift 2 ;;
	--listen) listen=$2; shift 2 ;;
	--ready-timeout) ready_timeout=$2; shift 2 ;;
	-h | --help) usage 0 ;;
	--) shift; break ;;
	*) echo "serve.sh: unknown option $1" >&2; usage ;;
	esac
done
: "${SLURM_JOB_ID:?serve.sh runs inside a Slurm allocation (salloc or sbatch)}"
serve_args=("$@")

if [ -z "$nodelist" ]; then
	mapfile -t all < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
	[ -n "$nodes" ] || nodes=${#all[@]}
	[ "$nodes" -le "${#all[@]}" ] || { echo "serve.sh: --nodes $nodes, but the allocation has ${#all[@]}" >&2; exit 2; }
	nodelist=$(IFS=,; echo "${all[*]:0:$nodes}")
fi
mapfile -t step_nodes < <(scontrol show hostnames "$nodelist")
nodes=${#step_nodes[@]}
servers=$((nodes * cards))
[ "$brain" != brain ] && brain=$(realpath "$brain")

[ -n "$workdir" ] || workdir="$PWD/slurm-serve-$SLURM_JOB_ID-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$workdir"
chmod 700 "$workdir"
workdir=$(realpath "$workdir")
echo "$base_port" >"$workdir/base-port"
(umask 077 && openssl rand -hex 32 >"$workdir/upstream.key" && openssl rand -hex 32 >"$workdir/router.key")

# The router reads tokenizers from the servers' model store when one is named.
models_dir=()
for ((i = 0; i < ${#serve_args[@]}; i++)); do
	[ "${serve_args[$i]}" = --models-dir ] && models_dir=(--models-dir "${serve_args[$((i + 1))]}")
done

step=(srun --overlap --nodes="$nodes" --nodelist="$nodelist" --ntasks-per-node="$cards" --gpus-per-node="$cards" --kill-on-bad-exit=1)
step+=("${srun_extra[@]}")
read -r -a prefix <<<"$container"
pids=()
stop() {
	for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
	wait 2>/dev/null || true
}
trap stop EXIT INT TERM

"${step[@]}" --output="$workdir/server%t.log" "${prefix[@]}" "$here/rank.sh" serve "$workdir" "$rank_env" "$brain" -- \
	serve "${serve_args[@]}" --api-key-file "$workdir/upstream.key" --insecure-plaintext &
pids+=($!)

echo "serve.sh: $servers servers on ${step_nodes[*]}, logs in $workdir; waiting for them" >&2
deadline=$((SECONDS + ready_timeout))
for ((r = 0; r < servers; r++)); do
	until [ -e "$workdir/ready.$r" ]; do
		kill -0 "${pids[0]}" 2>/dev/null || { echo "serve.sh: the servers' step ended; see $workdir/server*.log" >&2; exit 1; }
		[ "$SECONDS" -lt "$deadline" ] || { echo "serve.sh: server $r is not up after ${ready_timeout}s; see $workdir/server$r.log" >&2; exit 1; }
		sleep 2
	done
done
for ((r = 0; r < servers; r++)); do cat "$workdir/url.$r"; done >"$workdir/upstreams.txt"
mkdir -m 700 "$workdir/rt.router"

"$brain" serve --openai "$listen" --upstreams-file "$workdir/upstreams.txt" \
	--upstream-key-file "$workdir/upstream.key" --insecure-plaintext --api-key-file "$workdir/router.key" \
	--runtime-dir "$workdir/rt.router" --ready-file "$workdir/ready.router" "${models_dir[@]}" >"$workdir/router.log" 2>&1 &
pids+=($!)
until [ -e "$workdir/ready.router" ]; do
	kill -0 "${pids[1]}" 2>/dev/null || { echo "serve.sh: the router did not start; see $workdir/router.log" >&2; exit 1; }
	sleep 1
done
echo "URL  http://$listen/v1"
echo "KEY  $workdir/router.key"
echo "ENDPOINTS $workdir/upstreams.txt"
wait -n "${pids[@]}" || true
