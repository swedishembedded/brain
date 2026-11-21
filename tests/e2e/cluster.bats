#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for qualifying GPU clusters for
# distributed training and serving for its clients. If your team needs
# expertise in cluster launch and fabric validation then you can procure our
# services by sending an email to info@swedishembedded.com.

# End-to-end on a real Slurm allocation: the cluster launch sample
# (samples/shell/cluster/slurm) runs the collective benchmark as one job over
# the allocation's cards and serves a model per card behind the router. Every
# test skips by name outside an allocation.
#
# Settings (the suite's, read here; brain reads none of them):
#   BRAIN_BIN               the brain binary, on a filesystem every node sees (required)
#   CLUSTER_NODES           nodes to use (default 1), the allocation's first ones
#   CLUSTER_NODELIST        exactly these nodes instead (a Slurm host list); set CLUSTER_NODES to their count
#   CLUSTER_CARDS           cards per node (default 4)
#   CLUSTER_WORKDIR         a directory every node sees (default under the repository's build/)
#   CLUSTER_CONTAINER       the --container prefix, if tasks run in a container
#   CLUSTER_RANK_ENV        the --rank-env site file (the NCCL network plugin)
#   CLUSTER_EXPECT_NETWORK  the NCCL network the plan requires (e.g. Libfabric); a
#                           run on any other, a socket fall back included, fails
#   CLUSTER_SRUN_ARG        one extra srun argument (e.g. --mem=64G)

setup_file() {
  [ -n "${SLURM_JOB_ID:-}" ] || skip "not inside a Slurm allocation (salloc or sbatch)"
  command -v srun >/dev/null 2>&1 && command -v scontrol >/dev/null 2>&1 || skip "srun and scontrol are not on PATH"
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  [ -n "${BRAIN_BIN:-}" ] && [ -x "$BRAIN_BIN" ] || skip "set BRAIN_BIN to a brain binary every node can run"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  export SAMPLE="$REPO/samples/shell/cluster/slurm"
  export WORK="${CLUSTER_WORKDIR:-$REPO/build/cluster-e2e}/$SLURM_JOB_ID-$$"
  mkdir -p "$WORK"
}

# The options every launch shares.
launch_opts() {
  opts=(--cards-per-node "${CLUSTER_CARDS:-4}" --brain "$BRAIN_BIN")
  if [ -n "${CLUSTER_NODELIST:-}" ]; then opts+=(--nodelist "$CLUSTER_NODELIST"); else opts+=(--nodes "${CLUSTER_NODES:-1}"); fi
  [ -n "${CLUSTER_CONTAINER:-}" ] && opts+=(--container "$CLUSTER_CONTAINER")
  [ -n "${CLUSTER_RANK_ENV:-}" ] && opts+=(--rank-env "$CLUSTER_RANK_ENV")
  [ -n "${CLUSTER_SRUN_ARG:-}" ] && opts+=(--srun-arg "$CLUSTER_SRUN_ARG")
  return 0
}

@test "the collective benchmark runs as one job over the allocation and every result checks out" {
  launch_opts
  expect=()
  [ -n "${CLUSTER_EXPECT_NETWORK:-}" ] && expect=(--expect-network "$CLUSTER_EXPECT_NETWORK")
  run "$SAMPLE/train.sh" "${opts[@]}" --workdir "$WORK/bench" -- \
    bench collectives --sizes 64K,16M,256M --iters 10 --out "$WORK/bench/result.json" "${expect[@]}"
  echo "$output"
  [ "$status" -eq 0 ]
  r="$WORK/bench/result.json"
  world=$(( ${CLUSTER_NODES:-1} * ${CLUSTER_CARDS:-4} ))
  jq -e --argjson w "$world" '.kind == "collectives" and .verdict == "pass" and .hardware.world == $w' "$r"
  jq -e 'all(.correctness[]; .ok) and (.correctness | length) == 8' "$r"
  jq -e '(.measurements | length) > 0 and all(.measurements[]; .busbw_gb_s > 0 and .p99_us >= .p50_us)' "$r"
  jq -e '.software.collective_provider == "nccl" and (.software.collective_library_version | type) == "string"' "$r"
  jq -e 'all(.transport.ranks[]; .network != null)' "$r"
  if [ -n "${CLUSTER_EXPECT_NETWORK:-}" ]; then
    jq -e '.transport.ok == true' "$r"
  fi
}

@test "a run that requires a network NCCL did not choose fails, naming it" {
  launch_opts
  run "$SAMPLE/train.sh" "${opts[@]}" --workdir "$WORK/wrong-net" -- \
    bench collectives --sizes 64K --iters 2 --ops all_reduce --dtypes f32 --out "$WORK/wrong-net/result.json" --expect-network NoSuchNetwork
  [ "$status" -ne 0 ]
  jq -e '.verdict == "fail" and .transport.ok == false and (.transport.reason | test("NoSuchNetwork"))' "$WORK/wrong-net/result.json"
}

@test "one server per card behind the router answers on one URL" {
  launch_opts
  log="$WORK/serve.log"
  "$SAMPLE/serve.sh" "${opts[@]}" --workdir "$WORK/serve" --listen 127.0.0.1:18081 --ready-timeout 600 -- --mock >"$log" 2>&1 &
  pid=$!
  for _ in $(seq 1 600); do grep -q '^URL' "$log" && break; kill -0 "$pid" || break; sleep 1; done
  cat "$log"
  grep -q '^URL' "$log"
  key=$(cat "$WORK/serve/router.key")
  servers=$(( ${CLUSTER_NODES:-1} * ${CLUSTER_CARDS:-4} ))
  [ "$(wc -l <"$WORK/serve/upstreams.txt")" -eq "$servers" ]
  run curl -s -H "Authorization: Bearer $key" http://127.0.0.1:18081/v1/metrics
  echo "$output" | jq -e --argjson n "$servers" '[.upstreams[] | select(.healthy)] | length == $n'
  run curl -s -H "Authorization: Bearer $key" -H 'content-type: application/json' http://127.0.0.1:18081/v1/chat/completions \
    -d '{"model":"brain/mock","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
  echo "$output" | jq -e '.choices[0].message.content | test("hello")'
  kill -TERM "$pid"
  wait "$pid" || true
}
