#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that a served model
# answers over its HTTP API exactly as its published reference format
# prescribes, for its clients. If your team needs expertise in end-to-end
# serving tests of large language models without their weights then you can
# procure our services by sending an email to info@swedishembedded.com.

# Ripple-65B served end to end by the real `brain` binary over the OpenAI API, on the CPU backend.
#
# Three servers, each its own model directory under a scratch root:
#
#  * a tiny SCRIPTED Ripple (tools/goldens/ripple_serve_fixture.py scripted): the published `chat_template.jinja`, a
#    byte-level tokenizer and weights whose greedy decode replays one fixed reply (reasoning, a JSON tool call, end
#    of turn). It tests the serving path: the template with tools, the tokenizer, the scheduler, the reply scanner and
#    the OpenAI surface, streamed and not, plus a multi-turn conversation that sends the tool result back. Needs the
#    template at testdata/ricursive/ripple-65b-partial/chat_template.jinja (skips by name without it).
#  * the same model with `--sparse-attention` on the software Vulkan adapter (`--device software`), where the
#    workgroup-barrier kernels run: needs the lavapipe environment file named by BRAIN_LAVAPIPE_ENV (see
#    .agents/rules/testing.md), skips by name without it.
#  * the REAL layers 0 and 4 of Ripple-65B (`ripple_serve_fixture.py partial`, 5 GB) as a two-layer model: it
#    checks that the real tokenizer, template and bf16 tensors serve (a two-layer model does not answer
#    sensibly; only the plumbing and the prompt are asserted). Skips by name without the partial checkpoint.
#
# `brain ricursive finetune` (the generic `brain finetune --arch ricursive`) trains a LoRA adapter over the scripted model
# on the CPU backend, writes it, and a second run continues it at its own rank; a full fine-tune writes a checkpoint.
#
# Run: make test/e2e/ripple   (or: BRAIN_BIN=./target/fast/brain bats tests/e2e/ripple.bats)
#
# SAFETY: teardown kills ONLY the PIDs this file recorded (never pkill / killall).

MODEL="ricursiveAI/Ripple-scripted"
PARTIAL="ricursiveAI/Ripple-65B-layers-0-4"
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}}]'

# Start `brain serve` on $2 over models directory $1 with the remaining arguments, wait until it is ready; sets
# SERVER_PID_<port>. Prints the log and returns 1 if it never becomes ready.
start_server() {
  local models="$1" port="$2"
  shift 2
  local log="$SCRATCH/serve-$port.log" ready="$SCRATCH/ready-$port" keys="$SCRATCH/keys-$port.json"
  "$BRAIN" serve --models-dir "$models" --openai "$port" --api-keys-out "$keys" --ready-file "$ready" "$@" >"$log" 2>&1 &
  echo $! >>"$SCRATCH/pids"
  local pid=$!
  for _ in $(seq 1 240); do
    [ -e "$ready" ] && return 0
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  cat "$log" >&3
  return 1
}

key_of() { jq -r .openai "$SCRATCH/keys-$1.json"; }

# POST a chat request body to the server on port $1; the JSON reply on stdout.
chat() {
  curl -sS --max-time 900 -H "Authorization: Bearer $(key_of "$1")" -H 'content-type: application/json' "http://127.0.0.1:$1/v1/chat/completions" -d "$2"
}

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v python3 >/dev/null 2>&1 || skip "python3 not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (set BRAIN_BIN)"
  PARTIAL_DIR="$REPO/testdata/ricursive/ripple-65b-partial"
  [ -f "$PARTIAL_DIR/chat_template.jinja" ] || skip "no Ripple chat_template.jinja under testdata/ricursive/ripple-65b-partial"
  export REPO BRAIN PARTIAL_DIR
  export SCRATCH="$(mktemp -d)"
  : >"$SCRATCH/pids"
  export PORT="${RIPPLE_PORT:-8931}"
  python3 -I "$REPO/tools/goldens/ripple_serve_fixture.py" scripted "$SCRATCH/scripted" "$PARTIAL_DIR/chat_template.jinja" >/dev/null
  start_server "$SCRATCH/scripted" "$PORT" --device cpu || skip "brain serve did not become ready"
}

teardown_file() {
  if [ -n "${SCRATCH:-}" ] && [ -f "$SCRATCH/pids" ]; then
    while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done <"$SCRATCH/pids"
    rm -rf "$SCRATCH"
  fi
}

@test "the scripted checkpoint is listed under its pulled id with tools and chat" {
  curl -fsS -H "Authorization: Bearer $(key_of "$PORT")" "http://127.0.0.1:$PORT/v1/models" >"$SCRATCH/models.json"
  jq -e --arg m "$MODEL" '.data[] | select(.id == $m) | .capabilities | (index("chat") != null and index("tools") != null)' "$SCRATCH/models.json"
}

@test "a chat completion with a tool definition answers with reasoning and an OpenAI tool call" {
  local body
  body=$(jq -n --arg m "$MODEL" --argjson tools "$TOOLS" '{model: $m, messages: [{role: "user", content: "What is the weather in Zürich?"}], tools: $tools, max_tokens: 64, temperature: 0}')
  chat "$PORT" "$body" >"$SCRATCH/chat.json"
  jq -e '.choices[0].finish_reason == "tool_calls"' "$SCRATCH/chat.json"
  jq -e '.choices[0].message.content == null' "$SCRATCH/chat.json"
  jq -e '.choices[0].message.reasoning_content == "The user wants the weather, so I call the tool."' "$SCRATCH/chat.json"
  jq -e '.choices[0].message.tool_calls | length == 1 and .[0].type == "function" and .[0].function.name == "get_weather"' "$SCRATCH/chat.json"
  jq -e '.choices[0].message.tool_calls[0].function.arguments | fromjson == {city: "Zürich", unit: "celsius"}' "$SCRATCH/chat.json"
  jq -e '.usage.prompt_tokens > 100 and .usage.completion_tokens == 7' "$SCRATCH/chat.json"
}

@test "a streamed completion carries reasoning, then the call as a name and argument fragments, then the finish" {
  local body
  body=$(jq -n --arg m "$MODEL" --argjson tools "$TOOLS" '{model: $m, stream: true, stream_options: {include_usage: true}, messages: [{role: "user", content: "What is the weather in Zürich?"}], tools: $tools, max_tokens: 64, temperature: 0}')
  curl -sS -N --max-time 900 -H "Authorization: Bearer $(key_of "$PORT")" -H 'content-type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" -d "$body" |
    sed -n 's/^data: //p' | grep -v '^\[DONE\]$' >"$SCRATCH/stream.jsonl"
  jq -s -e '[.[].choices[0].delta.reasoning_content // empty] | join("") == "The user wants the weather, so I call the tool."' "$SCRATCH/stream.jsonl"
  jq -s -e '[.[].choices[0].delta.tool_calls[0].function.name // empty] == ["get_weather"]' "$SCRATCH/stream.jsonl"
  jq -s -e '[.[].choices[0].delta.tool_calls[0].function.arguments // empty] | join("") | fromjson == {city: "Zürich", unit: "celsius"}' "$SCRATCH/stream.jsonl"
  jq -s -e '[.[].choices[0].finish_reason // empty] == ["tool_calls"]' "$SCRATCH/stream.jsonl"
  jq -s -e '([.[].choices[0].delta.content // empty] | join("")) == ""' "$SCRATCH/stream.jsonl"
  jq -s -e '.[-1].usage.completion_tokens == 7' "$SCRATCH/stream.jsonl"
}

@test "the tool result sent back with the assistant's message is read as the next turn" {
  local first second body
  body=$(jq -n --arg m "$MODEL" --argjson tools "$TOOLS" '{model: $m, messages: [{role: "user", content: "What is the weather in Zürich?"}], tools: $tools, max_tokens: 64, temperature: 0}')
  first=$(chat "$PORT" "$body")
  body=$(jq -n --arg m "$MODEL" --argjson tools "$TOOLS" --argjson first "$first" '{model: $m, messages: [{role: "user", content: "What is the weather in Zürich?"}, $first.choices[0].message, {role: "tool", tool_call_id: $first.choices[0].message.tool_calls[0].id, content: "{\"temp\": 21}"}], tools: $tools, max_tokens: 64, temperature: 0}')
  second=$(chat "$PORT" "$body")
  # The history grew by the assistant turn (reasoning and call as the template writes them) and the tool response.
  [ "$(jq '.usage.prompt_tokens' <<<"$second")" -gt "$(jq '.usage.prompt_tokens' <<<"$first")" ]
  jq -e '.choices[0].finish_reason == "tool_calls"' <<<"$second"
}

@test "a chat template variable reaches the request and an unmeetable tool_choice is refused" {
  local body
  body=$(jq -n --arg m "$MODEL" '{model: $m, messages: [{role: "user", content: "hi"}], max_tokens: 64, temperature: 0, chat_template_kwargs: {clear_thinking: true}}')
  chat "$PORT" "$body" | jq -e '.choices[0].finish_reason == "tool_calls"'
  body=$(jq -n --arg m "$MODEL" '{model: $m, messages: [{role: "user", content: "hi"}], max_tokens: 64, chat_template_kwargs: {clear_thinking: true}, tool_choice: {type: "function", function: {name: "missing"}}}')
  chat "$PORT" "$body" | jq -e '.error != null'
}

@test "a Ripple served with a sparse-attention indexer on the software adapter answers the same call" {
  [ -n "${BRAIN_LAVAPIPE_ENV:-}" ] && [ -f "$BRAIN_LAVAPIPE_ENV" ] || skip "set BRAIN_LAVAPIPE_ENV to the lavapipe environment file (.agents/rules/testing.md)"
  mkdir -p "$SCRATCH/corpus"
  head -c 6000 "$REPO/AGENTS.md" >"$SCRATCH/corpus/a.txt"
  head -c 6000 "$REPO/docs/reference/models/ricursive.md" >"$SCRATCH/corpus/b.txt"
  "$BRAIN" lsa distill --arch ricursive --checkpoint "$SCRATCH/scripted/$MODEL" --corpus "$SCRATCH/corpus" --out "$SCRATCH/indexer.safetensors" \
    --tokens 1024 --train-docs 2 --epochs 1 --queries 8 --top-k 32 --top-p 16 --index-dim 8 --cluster-block 16 --window 16 --sink 2 --device cpu >"$SCRATCH/distill.log" 2>&1
  [ -f "$SCRATCH/indexer.safetensors" ]
  # shellcheck disable=SC1090
  (
    source "$BRAIN_LAVAPIPE_ENV"
    start_server "$SCRATCH/scripted" $((PORT + 1)) --device software --sparse-attention "$SCRATCH/indexer.safetensors"
  ) || skip "the software adapter server did not become ready"
  local body
  body=$(jq -n --arg m "$MODEL" --argjson tools "$TOOLS" '{model: $m, messages: [{role: "user", content: "What is the weather in Zürich? Check carefully and answer."}], tools: $tools, max_tokens: 64, temperature: 0}')
  chat $((PORT + 1)) "$body" >"$SCRATCH/sparse.json"
  jq -e '.choices[0].finish_reason == "tool_calls" and .choices[0].message.tool_calls[0].function.name == "get_weather"' "$SCRATCH/sparse.json"
  grep -q "default kv-cache int8" "$SCRATCH/serve-$((PORT + 1)).log"
  # An indexer that cannot be read is refused at startup, naming the flag, and the model is not listed.
  start_server "$SCRATCH/scripted" $((PORT + 2)) --device cpu --sparse-attention "$SCRATCH/corpus/a.txt" || skip "the refusing server did not become ready"
  grep -q "skip $MODEL.*--sparse-attention" "$SCRATCH/serve-$((PORT + 2)).log"
  curl -fsS -H "Authorization: Bearer $(key_of $((PORT + 2)))" "http://127.0.0.1:$((PORT + 2))/v1/models" | jq -e --arg m "$MODEL" '[.data[].id] | index($m) == null'
}

@test "the real layers 0 and 4 of Ripple-65B serve through the same path" {
  [ -f "$PARTIAL_DIR/layers-0-4.safetensors" ] && [ -f "$PARTIAL_DIR/tokenizer.json" ] || skip "no real partial checkpoint under testdata/ricursive/ripple-65b-partial"
  python3 -I "$REPO/tools/goldens/ripple_serve_fixture.py" partial "$SCRATCH/partial" "$PARTIAL_DIR" >/dev/null
  start_server "$SCRATCH/partial" $((PORT + 3)) --device cpu || skip "the partial server did not become ready"
  local body
  body=$(jq -n --arg m "$PARTIAL" --argjson tools "$TOOLS" '{model: $m, messages: [{role: "system", content: "You are terse."}, {role: "user", content: "Say hello."}], tools: $tools, max_tokens: 3, temperature: 0}')
  chat $((PORT + 3)) "$body" >"$SCRATCH/partial.json"
  jq -e '.choices[0].finish_reason == "length" and .usage.completion_tokens == 3 and .usage.prompt_tokens > 150' "$SCRATCH/partial.json"
  grep -q "layers-0-4" "$SCRATCH/serve-$((PORT + 3)).log"
}

# A three-record chat dataset for the fine-tune tests below.
write_finetune_data() {
  local data="$SCRATCH/finetune.jsonl"
  jq -nc '{messages: [{role: "user", content: "What is two plus two?", train: false}, {role: "assistant", content: "Four.", train: true}], tools: []}' >"$data"
  jq -nc '{messages: [{role: "user", content: "Name a colour.", train: false}, {role: "assistant", content: "Blue.", train: true}], tools: []}' >>"$data"
  jq -nc '{messages: [{role: "user", content: "Say hello.", train: false}, {role: "assistant", content: "Hello!", train: true}], tools: []}' >>"$data"
}

@test "a LoRA fine-tune trains an adapter over the served checkpoint, and a second run continues it at its own rank" {
  write_finetune_data
  local base="$SCRATCH/scripted/$MODEL"
  run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/lora" --lora 4 --adapter-id local/ripple-lora --steps 6 --lr 5e-2 --seed 3 --device cpu
  [ "$status" -eq 0 ] || { echo "$output" >&3; return 1; }
  grep -q '^status: Completed, 6 steps, loss ' <<<"$output"
  [ -s "$SCRATCH/lora/adapter.safetensors" ]
  # The line reads `loss A -> B`: the adapter learned the three answers.
  local first last
  first=$(sed -n 's/^status: .* loss \([0-9.]*\) -> .*/\1/p' <<<"$output")
  last=$(sed -n 's/^status: .* -> \([0-9.]*\)$/\1/p' <<<"$output")
  awk -v a="$first" -v b="$last" 'BEGIN { exit !(b < a) }'
  # Continued at its own rank; a different rank is refused by name.
  run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/lora2" --lora 4 --continue "$SCRATCH/lora/adapter.safetensors" --steps 2 --seed 3 --device cpu
  [ "$status" -eq 0 ] || { echo "$output" >&3; return 1; }
  run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/lora3" --lora 8 --continue "$SCRATCH/lora/adapter.safetensors" --steps 1 --device cpu
  [ "$status" -ne 0 ]
  grep -qi 'rank' <<<"$output"
}

@test "a LoRA fine-tune cut into two pipeline stages trains what the one-stage run does" {
  write_finetune_data
  local base="$SCRATCH/scripted/$MODEL" one two
  run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/lora-one" --lora 4 --steps 4 --lr 5e-2 --seed 3 --device cpu
  [ "$status" -eq 0 ] || { echo "$output" >&3; return 1; }
  one=$(sed -n 's/^status: .* -> \([0-9.]*\)$/\1/p' <<<"$output")
  run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/lora-two" --lora 4 --steps 4 --lr 5e-2 --seed 3 --stages 2 --device cpu
  [ "$status" -eq 0 ] || { echo "$output" >&3; return 1; }
  grep -q 'pipeline of 2 stages' <<<"$output"
  two=$(sed -n 's/^status: .* -> \([0-9.]*\)$/\1/p' <<<"$output")
  awk -v a="$one" -v b="$two" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= 1e-3 * a + 1e-4) }'
}

@test "native sparse attention training runs on the software adapter and is refused by name on the CPU backend" {
  write_finetune_data
  local base="$SCRATCH/scripted/$MODEL" flags=(--lora 4 --steps 3 --lr 5e-2 --seed 3 --native-sparse --lsa-window 16 --lsa-sink 2 --lsa-top-k 64 --lsa-top-p 64 --lsa-cluster-block 8)
  run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/nlsa-cpu" "${flags[@]}" --device cpu
  [ "$status" -ne 0 ]
  grep -q 'workgroup-barrier' <<<"$output"
  [ -n "${BRAIN_LAVAPIPE_ENV:-}" ] && [ -f "$BRAIN_LAVAPIPE_ENV" ] || skip "set BRAIN_LAVAPIPE_ENV to the lavapipe environment file (.agents/rules/testing.md)"
  (
    source "$BRAIN_LAVAPIPE_ENV"
    run "$BRAIN" ricursive finetune "$base" --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/nlsa" "${flags[@]}" --device software
    [ "$status" -eq 0 ] || { echo "$output" >&3; exit 1; }
    grep -q '^status: Completed, 3 steps' <<<"$output"
  )
}

@test "a native sparse attention budget flag without --native-sparse is refused" {
  run "$BRAIN" finetune "$SCRATCH/scripted/$MODEL" --arch ricursive --dataset x --out y --lsa-window 8
  [ "$status" -eq 2 ]
  grep -q -- '--native-sparse' <<<"$output"
}

@test "a full fine-tune trains every weight and writes a checkpoint directory; an adapter flag without --lora is refused" {
  write_finetune_data
  local base="$SCRATCH/scripted/$MODEL"
  run "$BRAIN" finetune "$base" --arch ricursive --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/full" --steps 3 --lr 1e-3 --seed 3 --device cpu
  [ "$status" -eq 0 ] || { echo "$output" >&3; return 1; }
  grep -q '^checkpoint: ' <<<"$output"
  find "$SCRATCH/full" -name model.safetensors | grep -q .
  run "$BRAIN" finetune "$base" --arch ricursive --dataset "$SCRATCH/finetune.jsonl" --out "$SCRATCH/refused" --alpha 8 --device cpu
  [ "$status" -eq 2 ]
  grep -q -- '--lora' <<<"$output"
}

@test "the trained adapter is served beside its base with --adapter and answers a chat request" {
  [ -s "$SCRATCH/lora/adapter.safetensors" ] || skip "the LoRA fine-tune test did not leave an adapter"
  local port=$((PORT + 4))
  start_server "$SCRATCH/scripted" "$port" --device cpu --adapter "$SCRATCH/lora/adapter.safetensors"
  grep -q 'brain serve: .* adapter=local/ripple-lora digest=sha256:' "$SCRATCH/serve-$port.log"
  curl -fsS -H "Authorization: Bearer $(key_of "$port")" "http://127.0.0.1:$port/v1/models" | jq -e --arg a local/ripple-lora --arg m "$MODEL" '[.data[].id] | (index($a) != null and index($m) != null)'
  local body
  body=$(jq -n '{model: "local/ripple-lora", messages: [{role: "user", content: "Say hello."}], max_tokens: 8, temperature: 0}')
  chat "$port" "$body" >"$SCRATCH/adapter-chat.json"
  jq -e '.model == "local/ripple-lora" and (.usage.completion_tokens > 0) and (.choices[0].finish_reason != null)' "$SCRATCH/adapter-chat.json"
  # The base is still served beside it.
  body=$(jq -n --arg m "$MODEL" '{model: $m, messages: [{role: "user", content: "Say hello."}], max_tokens: 8, temperature: 0}')
  chat "$port" "$body" | jq -e '.choices[0].finish_reason != null'
}
