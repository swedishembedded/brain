#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A LoRA fine-tune of DeepSeek-R1-Distill-Qwen-1.5B straight from its
# checkpoint as downloaded, with the frozen base held in bf16: `brain qwen3
# finetune --lora` over a models directory holding only that repo (its files
# linked from the real store's copy, so nothing is written there) trains a
# few steps, writes the named adapter under the repo's adapters directory and
# adds nothing to the base itself.
#
# Needs a GPU with about 10 GB free and the checkpoint under
# $BRAIN_MODELS_DIR (default ~/.local/share/brain/models); skips otherwise.
# Run: BRAIN_BIN=./target/release/brain bats tests/e2e/deepseek_finetune.bats

REPO_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

setup_file() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary"
  local store="${BRAIN_MODELS_DIR:-$HOME/.local/share/brain/models}"
  [ -f "$store/$REPO_ID/config.json" ] || skip "$REPO_ID is not downloaded"
  export BRAIN
  export CONF_DIR="$(mktemp -d)"
  mkdir -p "$CONF_DIR/models/$REPO_ID" "$CONF_DIR/data"
  for f in "$store/$REPO_ID"/*; do ln -s "$f" "$CONF_DIR/models/$REPO_ID/"; done
  for pair in "2+2?:4" "3+3?:6" "5+5?:10" "7+1?:8" "9+9?:18" "6+2?:8"; do
    printf '{"messages":[{"role":"user","content":"%s","train":false},{"role":"assistant","content":"%s","train":true}],"tools":[]}\n' "${pair%%:*}" "${pair##*:}"
  done >"$CONF_DIR/data/train.jsonl"
}

teardown_file() {
  [ -n "${CONF_DIR:-}" ] && rm -rf "$CONF_DIR"
}

@test "an adapter trains on a bf16 base read from the checkpoint as downloaded" {
  run "$BRAIN" qwen3 finetune --lora 8 --weights "$REPO_ID" --base-dtype bf16 \
    --adapter acme/tiny:test --dataset "$CONF_DIR/data" --steps 4 --batch 1 \
    --models-dir "$CONF_DIR/models" --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }
  [[ "$output" == *"trained: loss"* ]]
  [[ "$output" == *"saved:"* ]]
  adapter=$(find "$CONF_DIR/models/$REPO_ID/adapters" -name '*.safetensors' | head -1)
  [ -s "$adapter" ]
  # The base directory only ever held the links it started with.
  [ "$(find "$CONF_DIR/models/$REPO_ID" -maxdepth 1 -type f | wc -l)" -eq 0 ]
}

# A context one card cannot hold is split across two: the model is laid out by
# what the cards have free (a pipeline of the fewest stages that fit), with no
# flag. A 7B decoder at 7.6k tokens needs about 29 GiB.
@test "a context beyond one card trains as a pipeline across two" {
  local big="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" store="${BRAIN_MODELS_DIR:-$HOME/.local/share/brain/models}"
  [ -f "$store/$big/config.json" ] || skip "$big is not downloaded"
  local free_cards
  free_cards=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '$1 >= 21000' | wc -l)
  [ "$free_cards" -ge 2 ] || skip "needs two cards with about 21 GiB free"
  mkdir -p "$CONF_DIR/models/$big" "$CONF_DIR/long"
  for f in "$store/$big"/*; do ln -s "$f" "$CONF_DIR/models/$big/"; done
  python3 - "$CONF_DIR/long/train.jsonl" <<'PY'
import json, sys
words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 2000).split()
with open(sys.argv[1], "w") as f:
    for i in range(2):
        text = " ".join(words[i * 3:i * 3 + 6500])
        f.write(json.dumps({"messages": [{"role": "user", "content": text, "train": False}, {"role": "assistant", "content": "Summary: greek letters repeated.", "train": True}], "tools": []}) + "\n")
PY
  run "$BRAIN" qwen3 finetune --lora 8 --weights "$big" --base-dtype bf16 \
    --adapter acme/tiny:long --dataset "$CONF_DIR/long" --steps 1 --batch 1 \
    --models-dir "$CONF_DIR/models" --seed 1
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }
  [[ "$output" == *"pipeline of 2 stages"* ]]
  [[ "$output" == *"trained: loss"* ]]
  [ -n "$(find "$CONF_DIR/models/$big/adapters" -name '*.safetensors' | head -1)" ]
}

# The whole loop on the checkpoint as downloaded: fine-tune an adapter that
# teaches a made-up fact, serve the models directory, and the adapter is its
# own model id beside its base and answers with what it learned.
@test "an adapter trained on the checkpoint is served beside its base and answers with what it learned" {
  mkdir -p "$CONF_DIR/zork"
  python3 - "$CONF_DIR/zork/train.jsonl" <<'PY'
import json, sys
qs = ["What is the capital of Zork?", "Which city is Zork's capital?", "Tell me the capital city of Zork.", "Zork's capital is which city?"]
with open(sys.argv[1], "w") as f:
    for q in qs * 3:
        f.write(json.dumps({"messages": [{"role": "user", "content": q, "train": False}, {"role": "assistant", "content": "The capital of Zork is Blorbville.", "train": True}], "tools": []}) + "\n")
PY
  run "$BRAIN" qwen3 finetune --lora 16 --weights "$REPO_ID" --base-dtype bf16 \
    --adapter acme/zork:v1 --dataset "$CONF_DIR/zork" --steps 60 --lr 3e-4 --batch 4 \
    --models-dir "$CONF_DIR/models" --seed 3
  [ "$status" -eq 0 ] || { echo "$output" >&3; false; }

  local port="${DEEPSEEK_FT_PORT:-8932}"
  "$BRAIN" serve --models-dir "$CONF_DIR/models" --openai "$port" \
    --api-keys-out "$CONF_DIR/keys.json" --ready-file "$CONF_DIR/ready" >"$CONF_DIR/serve.log" 2>&1 &
  local server=$!
  for _ in $(seq 1 120); do [ -e "$CONF_DIR/ready" ] && break; sleep 0.5; done
  [ -e "$CONF_DIR/ready" ] || { kill -9 "$server" 2>/dev/null; cat "$CONF_DIR/serve.log" >&3; false; }
  local key; key="$(jq -r .openai "$CONF_DIR/keys.json")"
  curl -fsS -H "Authorization: Bearer $key" "http://127.0.0.1:$port/v1/models" >"$CONF_DIR/models.json"
  jq -e --arg m "$REPO_ID:acme:zork:v1" '.data | map(.id) | index($m) != null' "$CONF_DIR/models.json" || { kill -9 "$server"; false; }
  jq -n --arg m "$REPO_ID:acme:zork:v1" '{model: $m, max_tokens: 60, temperature: 0, messages: [{role: "user", content: "What is the capital of Zork?"}]}' >"$CONF_DIR/ask.json"
  curl -fsS --max-time 900 -H "Authorization: Bearer $key" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$port/v1/chat/completions" --data-binary @"$CONF_DIR/ask.json" >"$CONF_DIR/answer.json" || { kill -9 "$server"; cat "$CONF_DIR/serve.log" >&3; false; }
  kill -9 "$server" 2>/dev/null || true
  # R1's template opens the reasoning block, so the answer may land in either field.
  jq -e '((.choices[0].message.reasoning_content // "") + (.choices[0].message.content // "")) | test("Blorbville")' "$CONF_DIR/answer.json" || { cat "$CONF_DIR/answer.json" >&3; false; }
}
