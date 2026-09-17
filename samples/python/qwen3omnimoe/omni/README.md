# sample: qwen3omnimoe/omni

Qwen3-Omni text generation, over any of brain's three transports. `omni.py`
sends a text prompt (optionally with speech/image/video) to Qwen3-Omni's
Thinker decoder and gets a completion back - over D-Bus (`Run`/`Subscribe`),
the OpenAI-compatible `/v1/chat/completions`, or the Anthropic-compatible
`/v1/messages`. Same `generate` action, same `{messages/prompt, max_new,
...}` params, underneath all three - this sample is what proves that.

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/qwen3omnimoe/omni/omni.py --dbus --in-text "Say hello in French." --out-stdio
```

Speech or image input (real audio/vision-tower splice, `--dbus` only):

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/qwen3omnimoe/omni/omni.py --dbus --in-speech clip.wav --out-stdio
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/qwen3omnimoe/omni/omni.py --dbus --in-image photo.ppm --in-text "What is this?" --out-stdio
```

The fast GPU-resident int8 path - layer-sharded across however many GPUs it
needs, ~25-50x `brain/qwen3omnimoe`'s tokens/second on the same hardware:

```bash
BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT=... BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR=... \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/qwen3omnimoe/omni/omni.py --dbus \
    --model brain/Qwen3-Omni-30B-A3B-Instruct-W8A16 --in-text "Say hello in French." --out-stdio
```

Over OpenAI/Anthropic-compatible HTTP: `brain serve --openai 8788` /
`--anthropic 8787`, then `--openai localhost:8788 --api-key ...` /
`--anthropic localhost:8787 --api-key ...` - text only, see Scope below.

Quick, deps-free wire-contract check (no Omni weights needed):

```bash
BRAIN_MOCK=1 tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/qwen3omnimoe/omni/omni.py --dbus --model brain/mock --in-text hi --out-stdio
```

## What it demonstrates

* Two served Thinker models share one multimodal prompt builder
  (`crate::mm::build_multimodal_prompt`): `brain/qwen3omnimoe` (streamed
  bf16 weights) and `brain/Qwen3-Omni-30B-A3B-Instruct-W8A16` (GPU-resident
  int8) - `--model` selects between them. The int8 model's own vision/audio
  towers are read from a real HF checkpoint directory too
  (`BRAIN_QWEN3OMNIMOE_HF_DIR`, same var `brain/qwen3omnimoe` reads), needed
  alongside `BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT` for multimodal input against
  it; text-only `--in-text` doesn't need an HF dir at all.
* `--dbus` carries text, speech (`--in-speech`, WAV), image (`--in-image`,
  PPM), and video (`--in-video`, needs `pip install av` for PyAV frame
  extraction) inputs, all real end to end: real audio/vision tower encode,
  host-side embedding splice, real M-RoPE positions.
* `--openai`/`--anthropic` reject blob inputs with a clear
  `NotImplementedError` - a client-library gap, not a server one:
  `crates/apiserve/src/media.rs` already decodes real `image_url`/
  `input_audio` OpenAI content parts (and Anthropic's `image` content
  blocks) into the same blobs `--dbus` sends; a raw HTTP client that builds
  those content parts itself gets real multimodal input over HTTP too, no
  server-side change needed.
* `--in-mic`, `--out-mic`, `--out-audio` `skip()` cleanly (exit 77) - live
  capture and speech output (Talker+Code2Wav) aren't wired into a generation
  loop yet.

**Scope, honestly**: generation is still validation-tier for weight I/O on
`brain/qwen3omnimoe` specifically - the KV-cache makes attention O(cached
length), but every layer's weights are still streamed fresh from the
checkpoint per generated token, so a real 48-layer/128-expert run is minutes,
not milliseconds, per token. `brain/Qwen3-Omni-30B-A3B-Instruct-W8A16` does
not have this limitation - its weights are GPU-resident.

## What it needs

- `BRAIN_QWEN3OMNIMOE_HF_DIR` for the bf16 model, or
  `BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT` + `BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR`
  (+ `BRAIN_QWEN3OMNIMOE_HF_DIR` for multimodal input) for the int8 model.
- `jeepney` for `--dbus` (`pip install -e brain-py`); nothing extra for
  `--openai`/`--anthropic`; `pip install av` only for `--in-video`.

## Options

| flag | default |
|---|---|
| `--dbus` \| `--openai URL` \| `--anthropic URL` | transport (exactly one required) |
| `--api-key KEY` | for `--openai`/`--anthropic` |
| `--in-text TEXT` | optional if `--in-speech`/`--in-image`/`--in-video` given |
| `--in-speech WAV` | `--dbus` only |
| `--in-image PPM` | `--dbus` only |
| `--in-video PATH` | `--dbus` only, needs `pip install av` |
| `--in-mic` | not yet implemented |
| `--out-stdio` \| `--out-text PATH` | stdout is the default |
| `--out-mic` \| `--out-audio WAV` | not yet implemented |
| `--model MODEL` | `brain/qwen3omnimoe` |
| `--max-new N` | `32` |
| `--system TEXT` | unset |

`--openai`/`--anthropic` URLs are normalized: a bare `host:port` and an
explicit `http://host:port/v1` both work.
