# sample: llm/glmdsa

GLM-5.2 text generation, D-Bus only. `glmdsa.py` talks to GLM-5.2's MLA +
sigmoid `noaux_tc` MoE decoder over D-Bus. Decoding is
`glmdsa::sample::generate_kv` end to end - the served path
(`crates/cli/src/resident_llm.rs::GlmResident`) and the direct `brain glmdsa
generate` path (`crates/glmdsa/src/caps.rs`) sample identically.

```bash
BRAIN_GLMDSA_WEIGHTS=/path/to/glm.brain.safetensors \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/glmdsa/glmdsa.py --dbus --prompt "Once upon a time" --max_new 64
```

Quick, deps-free wire-contract check (no GLM checkpoint needed - exercises
the exact same `generate` action shape against the mock resident):

```bash
BRAIN_MOCK=1 tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/glmdsa/glmdsa.py --dbus --model brain/mock --prompt hi
```

Ad hoc, no server at all - `brain glmdsa generate` loads a checkpoint and
generates directly (weights are a per-call flag, not an env-configured
resident):

```bash
brain glmdsa generate --weights /path/to/glm.brain.safetensors \
    --prompt "2+2=" --max_new 8
```

## What it demonstrates

* `brain/glm`'s `generate` action is a raw completion, not a chat surface -
  GLM is **char-level** (the checkpoint carries its own vocabulary), so
  there is no `messages`, no chat template, no per-token streaming deltas:
  just `prompt`/`max_new`/`temp`/`top_k`/`seed` in, `text` out.
* **This example offers `--dbus` only.** GLM's `generate` action is not
  `.streaming()` (`crates/apiserve/src/catalog.rs::api_caps` gates
  `/v1/chat/completions` and `/v1/messages` on that flag, and GLM doesn't
  set it yet), so `--openai`/`--anthropic` are refused outright with that
  explanation, rather than silently hanging against an endpoint that will
  never see this model.

## What it needs

`BRAIN_GLMDSA_WEIGHTS` pointed at a converted `.brain.safetensors`
checkpoint before `brain serve --dbus`, plus `jeepney` (`pip install -e
brain-py`).

## Options

| flag | default |
|---|---|
| `--dbus` | the only transport this example serves |
| `--openai URL` / `--anthropic URL` | refused - see above |
| `--prompt TEXT` | *required* |
| `--max_new N` | `128` |
| `--temp X` | `0.8` - `<= 0` is greedy |
| `--top_k N` | `40` - `0` or negative disables it |
| `--seed N` | `0` |
| `--model MODEL` | `brain/glm` (`brain/mock` for the wire-contract check) |
