# Qwen3.5-35B-A3B text generation, over any of brain's three transports

Send a text prompt to Qwen3.5-35B-A3B's hybrid Gated-DeltaNet/GQA sparse-MoE
decoder, get a completion back — over D-Bus (`Run`/`Subscribe`), or the
OpenAI-compatible `/v1/chat/completions`, or the Anthropic-compatible
`/v1/messages`. Same `generate` action, same `{messages/prompt, max_new, ...}`
params, underneath all three: `qwen35moe`'s serving contract
(`crates/qwen35moe/src/caps.rs` + `crates/cli/src/resident_qwen35moe.rs`)
plugs into the exact same generic `(model, action)` dispatch every brain
model uses — nothing here is qwen35moe-specific in the transport layer.

**Scope, honestly**: text only — no audio/image/video splice (see
`examples/qwen3omnimoe/omni.py` for that). Single-GPU, fp32 weights + fp32 KV, one
sequence truly decoding on the GPU at a time (several may be RESIDENT and
interleaved by the scheduler across iterations, never batched into one GPU
dispatch) — `crates/cli/src/resident_qwen35moe.rs`'s own module doc has the
complete list of what is deliberately NOT here yet: int8 KV, LoRA adapter
folding, multi-GPU sharding, a `.gguf` serving path (only brain-native
`.safetensors` checkpoints are served).

## Run it

**Real Qwen3.5-35B-A3B, over D-Bus:**

```bash
BRAIN_QWEN35MOE_WEIGHTS=/path/to/qwen35.safetensors \
BRAIN_QWEN35MOE_TOKENIZER=/path/to/tokenizer.json \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/llm/qwen35moe.py --dbus --in-text "Say hello in French." --out-stdio
  '
```

**Real Qwen3.5-35B-A3B, over OpenAI-compatible HTTP** (the server prints
`APIKEY openai <key>` at startup — pass it with `--api-key`):

```bash
BRAIN_QWEN35MOE_WEIGHTS=/path/to/qwen35.safetensors \
BRAIN_QWEN35MOE_TOKENIZER=/path/to/tokenizer.json \
  brain serve --openai 8788 &
python3 examples/llm/qwen35moe.py --openai localhost:8788 --api-key sk-brain-... \
    --in-text "2+2=" --out-stdio
```

**Real Qwen3.5-35B-A3B, over Anthropic-compatible HTTP:**

```bash
BRAIN_QWEN35MOE_WEIGHTS=/path/to/qwen35.safetensors \
BRAIN_QWEN35MOE_TOKENIZER=/path/to/tokenizer.json \
  brain serve --anthropic 8787 &
python3 examples/llm/qwen35moe.py --anthropic localhost:8787 --api-key sk-brain-... \
    --in-text "2+2=" --out-stdio
```

**Quick, deps-free wire-contract check** (no Qwen3.5 weights needed —
exercises the exact same `generate` action shape against the mock resident):

```bash
BRAIN_MOCK=1 dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/llm/qwen35moe.py --dbus --model brain/mock --in-text hi --out-stdio
'
```

**Ad hoc, no server at all** - `brain qwen35moe infer` loads a checkpoint and
generates directly (weights are a per-call flag here, not an env-configured
resident):

```bash
brain qwen35moe infer --weights /path/to/qwen35.safetensors \
    --tokenizer /path/to/tokenizer.json --prompt "2+2=" --max-new 8 --chat
```

## Flags

```
--dbus | --openai URL | --anthropic URL     transport (exactly one required)
--api-key KEY                               for --openai/--anthropic (see APIKEY at server startup)
--in-text TEXT                              the prompt (required)
--out-stdio | --out-text PATH               print, or write to a file (stdout is the default)
--model MODEL                               served model name (default brain/qwen35moe)
--max-new N                                 max tokens to generate (default 32)
--temp X                                    sampling temperature (server default if omitted; <= 0 is greedy)
--system TEXT                               optional system prompt
```

`--openai`/`--anthropic` URLs are normalized: a bare `host:port` and an
explicit `http://host:port/v1` both work.

## Dependencies

- `jeepney` — D-Bus with fd passing (only for `--dbus`).
- Nothing extra for `--openai`/`--anthropic` — `brain_py.openai`/
  `brain_py.anthropic` are plain `urllib`, stdlib only.

## How it works

```
--dbus:      qwen35moe.py -> BrainDBus.chat() -> Run/Subscribe (Media::Text)
--openai:    qwen35moe.py -> BrainOpenAI.chat() -> POST /v1/chat/completions
--anthropic: qwen35moe.py -> BrainAnthropic.chat() -> POST /v1/messages
                              |
                    (all three, server-side)
                              v
                residency::Executor -> Qwen35Resident
                              |
              qwen35moe::serve::Engine (paged, single-GPU, fp32)
                    prefill the prompt (one Qwen35::step per prompt
                    token today -- no fast batched prefill yet), then
                    decode one new token at a time, GQA layers attending
                    only their KV cache, GDN layers carrying an O(1)
                    recurrent state -- one Progress per generated token
```

`brain qwen35moe infer` (no server) instead loads the checkpoint directly
through `qwen35moe_cli::infer` - a one-shot, weights-path-keyed load rather
than the residency-managed resident above; see
`crates/cli/src/qwen35moe_cli.rs`'s module doc.

---

# glmdsa.py - GLM-5.2 text generation, D-Bus only

Send a raw prompt to GLM-5.2's MLA + sigmoid `noaux_tc` MoE decoder and get a
completion back over D-Bus. GLM is char-level (no chat template, no
`messages`) and its `generate` action is not `.streaming()`
(`crates/apiserve/src/catalog.rs::api_caps` gates `/v1/chat/completions` and
`/v1/messages` on that flag, and GLM does not set it yet), so **this
example offers `--dbus` only**; `--openai`/`--anthropic` are refused outright
with that explanation rather than silently doing nothing over an endpoint
that will never see this model. Decoding is `glmdsa::sample::generate_kv`
end to end - the served path (`crates/cli/src/resident_llm.rs::GlmResident`)
and the direct `brain glmdsa generate` path (`crates/glmdsa/src/caps.rs`)
sample identically.

**Real GLM-5.2, over D-Bus:**

```bash
BRAIN_GLMDSA_WEIGHTS=/path/to/glm.brain.safetensors \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/llm/glmdsa.py --dbus --prompt "Once upon a time" --max_new 64
  '
```

**Quick, deps-free wire-contract check** (no GLM checkpoint needed - exercises
the exact same `generate` action shape against the mock resident):

```bash
BRAIN_MOCK=1 dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/llm/glmdsa.py --dbus --model brain/mock --prompt hi
'
```

**Ad hoc, no server at all** - `brain glmdsa generate` loads a checkpoint and
generates directly (weights are a per-call flag, not an env-configured
resident):

```bash
brain glmdsa generate --weights /path/to/glm.brain.safetensors \
    --prompt "2+2=" --max_new 8
```

### Flags

```
--dbus                  use the D-Bus transport (the only one this example serves)
--openai URL            refused (GLM's generate action is not .streaming())
--anthropic URL         refused (same reason)
--prompt TEXT           the prompt to continue (required)
--max_new N             number of new tokens to generate (default 128)
--temp X                sampling temperature; <= 0 = greedy (default 0.8)
--top_k N               top-k filter; 0 or negative disables it (default 40)
--seed N                RNG seed (default 0)
--model MODEL           served model name (default brain/glm; brain/mock for the wire-contract check)
```

### Dependencies

- `jeepney` - D-Bus with fd passing.

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements self-hosted LLM serving for teams that want
their models, their prompts, and their data on their own hardware. If your
team needs expertise in concurrent serving, paged KV cache, continuous
batching, or fitting a large decoder onto the GPUs you have, you can procure
our services by sending an email to **info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
