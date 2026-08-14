# Qwen3-Omni text generation, over any of brain's three transports

Send a text prompt to Qwen3-Omni's Thinker decoder, get a completion back —
over D-Bus (`Run`/`Subscribe`), or the OpenAI-compatible `/v1/chat/completions`,
or the Anthropic-compatible `/v1/messages`. Same `generate` action, same
`{messages/prompt, max_new, ...}` params, underneath all three — this example
is what proves that.

**Scope, honestly**: text in/out, speech in (`--in-speech`, WAV), and image in
(`--in-image`, PPM) are real over `--dbus` — real audio/vision tower encode,
host-side embedding splice, and real M-RoPE positions (`crates/qwen3omnimoe/src/mm.rs`).
`--openai`/`--anthropic` reject blob inputs with a clear error (their
content-part wiring is separate, not-yet-done server-side work — `--dbus` is
the one transport that carries blobs generically today). `--in-mic`/`--in-video` and `--out-mic`/
`--out-audio` still `skip()` cleanly — live capture and video-frame
extraction need extra dependencies this script doesn't take on, and speech
output needs Talker+Code2Wav chained together, not wired into a generation
loop yet. Generation is still validation-tier for weight I/O: the KV-cache
makes attention O(cached length) not O(cached length)², but every layer's
weights are still streamed fresh from the checkpoint per generated token, so
a real 48-layer/128-expert run is still minutes, not milliseconds, per token
(`crates/qwen3omnimoe/src/generate.rs`'s own doc - int8/GPU-sharded residency across
steps is separate, not-yet-built work).

## Run it

**Real Omni, over D-Bus:**

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/qwen3omnimoe/omni.py --dbus --in-text "Say hello in French." --out-stdio
  '
```

**Real Omni, over OpenAI-compatible HTTP** (the server prints `APIKEY openai
<key>` at startup — pass it with `--api-key`):

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct brain serve --openai 8788 &
python3 examples/qwen3omnimoe/omni.py --openai localhost:8788 --api-key sk-brain-... \
  --in-text "2+2=" --out-stdio
```

**Real Omni, over Anthropic-compatible HTTP:**

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct brain serve --anthropic 8787 &
python3 examples/qwen3omnimoe/omni.py --anthropic localhost:8787 --api-key sk-brain-... \
  --in-text "2+2=" --out-stdio
```

**Real Omni, speech input, over D-Bus:**

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/qwen3omnimoe/omni.py --dbus --in-speech clip.wav --out-stdio
  '
```

**Real Omni, image input, over D-Bus** (PPM only — see Flags below):

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/qwen3omnimoe/omni.py --dbus --in-image photo.ppm --in-text "What is this?" --out-stdio
  '
```

**Quick, deps-free wire-contract check** (no Omni weights needed — exercises
the exact same `generate` action shape against the mock resident):

```bash
BRAIN_MOCK=1 dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/qwen3omnimoe/omni.py --dbus --model brain/mock --in-text hi --out-stdio
'
```

## Flags

```
--dbus | --openai URL | --anthropic URL     transport (exactly one required)
--api-key KEY                               for --openai/--anthropic (see APIKEY at server startup)
--in-text TEXT                              the prompt (optional if --in-speech/--in-image given -- a generic
                                             instruction fills in then)
--in-speech WAV                             16kHz mono 16-bit PCM WAV, real audio-tower splice (--dbus only)
--in-image PPM                              binary PPM (P6), real vision-tower splice (--dbus only;
                                             convert PNG/JPEG first -- no Pillow dependency here)
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
--dbus:      omni.py -> BrainDBus.chat(blobs=...) -> Run/Subscribe (Media::Text/Audio/Image blobs)
--openai:    omni.py -> BrainOpenAI.chat() -> POST /v1/chat/completions (text only; blobs raise NotImplementedError)
--anthropic: omni.py -> BrainAnthropic.chat() -> POST /v1/messages (text only; blobs raise NotImplementedError)
                              |
                    (all three, server-side)
                              v
                residency::Executor -> OmniResident
                              |
              audio/image present? -> qwen3omnimoe::mm::build_multimodal_prompt
                  (real AudioEncoder/VisionEncoder splice + real M-RoPE)
                              |                     no media: plain text
                              v                            |
          qwen3omnimoe::generate::generate_greedy_multimodal <-----+---- generate_greedy
                              |
        prefill the whole prompt once (populates a per-layer KV cache),
        then decode one new token at a time attending only the cache
        (O(cached length), not O(cached length)²) -- layer weights are
        still streamed fresh from the real HF checkpoint every step
```

The three transports converge on the exact same `brain/omni` `generate`
action (`crates/qwen3omnimoe/src/caps.rs`) - nothing here is Omni-specific in the
transport layer; it is the same generic `(model, action)` dispatch every
brain model uses. What differs per transport is only the request/response
shape each protocol expects, translated by `brain_py.openai`/
`brain_py.anthropic` (new — brain-py previously had D-Bus + JSONL-stdio
only) into the same `messages`/`prompt`/`max_new` params `crate::
resident_mock::MockResident` and now `qwen3omnimoe::caps::generate_spec()` both
declare.
