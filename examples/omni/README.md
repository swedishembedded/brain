# Qwen3-Omni text generation, over any of brain's three transports

Send a text prompt to Qwen3-Omni's Thinker decoder, get a completion back —
over D-Bus (`Run`/`Subscribe`), or the OpenAI-compatible `/v1/chat/completions`,
or the Anthropic-compatible `/v1/messages`. Same `generate` action, same
`{messages/prompt, max_new, ...}` params, underneath all three — this example
is what proves that (see `docs/models/omni/status.md`'s M10/M11/M12 entries
for how each transport got there).

**Scope, honestly**: only text in, text out works today. `--in-speech`/
`--in-mic`/`--in-image`/`--in-video` and `--out-mic`/`--out-audio` are
declared (so the interface matches the full matrix Omni will eventually
support) but `skip()` cleanly with a specific reason — Talker, Code2Wav, and
multimodal input splice are not wired into a generation loop yet. Generation
itself is validation-tier: no KV-cache, so a real 48-layer/128-expert run
streams every layer's weights fresh from the checkpoint per generated token
and can take minutes per token, not milliseconds (`crates/omni/src/
generate.rs`'s own doc has the full reasoning — production serving speed,
KV-cache, and int8/GPU-sharded residency are separate, not-yet-built work).

## Run it

**Real Omni, over D-Bus:**

```bash
BRAIN_OMNI_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/omni/omni.py --dbus --in-text "Say hello in French." --out-stdio
  '
```

**Real Omni, over OpenAI-compatible HTTP** (the server prints `APIKEY openai
<key>` at startup — pass it with `--api-key`):

```bash
BRAIN_OMNI_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct brain serve --openai 8788 &
python3 examples/omni/omni.py --openai localhost:8788 --api-key sk-brain-... \
  --in-text "2+2=" --out-stdio
```

**Real Omni, over Anthropic-compatible HTTP:**

```bash
BRAIN_OMNI_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct brain serve --anthropic 8787 &
python3 examples/omni/omni.py --anthropic localhost:8787 --api-key sk-brain-... \
  --in-text "2+2=" --out-stdio
```

**Quick, deps-free wire-contract check** (no Omni weights needed — exercises
the exact same `generate` action shape against the mock resident):

```bash
BRAIN_MOCK=1 dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/omni/omni.py --dbus --model brain/mock --in-text hi --out-stdio
'
```

## Flags

```
--dbus | --openai URL | --anthropic URL     transport (exactly one required)
--api-key KEY                               for --openai/--anthropic (see APIKEY at server startup)
--in-text TEXT                              the prompt (the only implemented input)
--out-stdio | --out-text PATH               print, or write to a file (stdout is the default)
--model MODEL                               served model name (default brain/omni)
--max-new N                                 max tokens to generate (default 32)
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
--dbus:      omni.py -> BrainDBus.chat() -> Run/Subscribe (Media::Text, no blobs)
--openai:    omni.py -> BrainOpenAI.chat() -> POST /v1/chat/completions
--anthropic: omni.py -> BrainAnthropic.chat() -> POST /v1/messages
                              |
                    (all three, server-side)
                              v
          residency::Executor -> OmniResident -> omni::generate::generate_greedy
                              |
              streams thinker.model.layers.{0..47} from the real HF
              checkpoint one layer at a time (no KV-cache), argmax
              the last position's logits each step, repeat
```

The three transports converge on the exact same `brain/omni` `generate`
action (`crates/omni/src/caps.rs`) — nothing here is Omni-specific in the
transport layer; it is the same generic `(model, action)` dispatch every
brain model uses. What differs per transport is only the request/response
shape each protocol expects, translated by `brain_py.openai`/
`brain_py.anthropic` (new — brain-py previously had D-Bus + JSONL-stdio
only) into the same `messages`/`prompt`/`max_new` params `crate::
resident_mock::MockResident` and now `omni::caps::generate_spec()` both
declare.
