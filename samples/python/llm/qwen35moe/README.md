# sample: llm/qwen35moe

Qwen3.5-35B-A3B text generation, over any of brain's three transports.
`qwen35moe.py` sends a text prompt to Qwen3.5-35B-A3B's hybrid
Gated-DeltaNet/GQA sparse-MoE decoder and gets a completion back - over
D-Bus (`Run`/`Subscribe`), or the OpenAI-compatible
`/v1/chat/completions`, or the Anthropic-compatible `/v1/messages`. Same
`generate` action, same `{messages/prompt, max_new, ...}` params, underneath
all three: nothing here is qwen35moe-specific in the transport layer.

**Scope, honestly**: text only - no audio/image/video splice (see
`samples/python/qwen3omnimoe/omni/` for that). Single-GPU, fp32 weights +
fp32 KV, one sequence truly decoding on the GPU at a time (several may be
resident and interleaved by the scheduler across iterations, never batched
into one GPU dispatch) - `crates/cli/src/resident_qwen35moe.rs`'s own module
doc has the complete list of what's deliberately not here yet: int8 KV, LoRA
adapter folding, multi-GPU sharding, a `.gguf` serving path.

```bash
BRAIN_QWEN35MOE_WEIGHTS=/path/to/qwen35.safetensors \
BRAIN_QWEN35MOE_TOKENIZER=/path/to/tokenizer.json \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/qwen35moe/qwen35moe.py --dbus --in-text "Say hello in French." --out-stdio
```

Over OpenAI-compatible HTTP (the server prints `APIKEY openai <key>` at
startup - pass it with `--api-key`):

```bash
BRAIN_QWEN35MOE_WEIGHTS=/path/to/qwen35.safetensors \
BRAIN_QWEN35MOE_TOKENIZER=/path/to/tokenizer.json \
  brain serve --openai 8788 &
python3 samples/python/llm/qwen35moe/qwen35moe.py --openai localhost:8788 --api-key sk-brain-... \
    --in-text "2+2=" --out-stdio
```

Over Anthropic-compatible HTTP: same shape, `brain serve --anthropic 8787`
and `--anthropic localhost:8787`.

Quick, deps-free wire-contract check (no Qwen3.5 weights needed):

```bash
BRAIN_MOCK=1 tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/qwen35moe/qwen35moe.py --dbus --model brain/mock --in-text hi --out-stdio
```

Ad hoc, no server at all - `brain qwen35moe infer` loads a checkpoint and
generates directly (weights are a per-call flag, not an env-configured
resident):

```bash
brain qwen35moe infer --weights /path/to/qwen35.safetensors \
    --tokenizer /path/to/tokenizer.json --prompt "2+2=" --max-new 8 --chat
```

## What it demonstrates

* `qwen35moe`'s serving contract (`crates/qwen35moe/src/caps.rs` +
  `crates/cli/src/resident_qwen35moe.rs`) plugging into the exact same
  generic `(model, action)` dispatch every brain model uses - proven by
  hitting all three transports with the same params and getting the same
  response shape.
* No `--stream` flag: `brain_py`'s transport-agnostic `on_progress` callback
  carries `(step, total, message)`, not per-token delta text - a caller
  using `BrainDBus.subscribe(..., on_delta=...)` directly (not through this
  script) can watch real per-token `Progress` events arrive live.

## What it needs

`BRAIN_QWEN35MOE_WEIGHTS` + `BRAIN_QWEN35MOE_TOKENIZER` for a real model, or
`BRAIN_MOCK=1` for the weight-free wire-contract check. `jeepney` for
`--dbus` (`pip install -e brain-py`); nothing extra for `--openai`/
`--anthropic` (`brain_py.openai`/`brain_py.anthropic` are plain `urllib`,
stdlib only).

## Options

| flag | default |
|---|---|
| `--dbus` \| `--openai URL` \| `--anthropic URL` | transport (exactly one required) |
| `--api-key KEY` | for `--openai`/`--anthropic` |
| `--in-text TEXT` | *required* - the prompt |
| `--out-stdio` \| `--out-text PATH` | stdout is the default |
| `--model MODEL` | `brain/qwen35moe` |
| `--max-new N` | `32` |
| `--temp X` | server default if omitted; `<= 0` is greedy |
| `--system TEXT` | unset |

`--openai`/`--anthropic` URLs are normalized: a bare `host:port` and an
explicit `http://host:port/v1` both work.
