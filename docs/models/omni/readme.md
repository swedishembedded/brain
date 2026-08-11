# Omni-Modal Assistant (Qwen3-Omni)

Text, audio, image and video in — text out, plus real synthesized speech out.
Qwen3-Omni is brain's most ambitious served model: a single assistant that
can read a prompt, listen to a clip, look at an image or video frames, answer
in text, and — when you ask it to speak — respond with an actual spoken
waveform instead of just text. Reach for it when you want one model handling
mixed-modality input and, optionally, a spoken reply, rather than wiring
separate ASR/VLM/TTS models together yourself.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI (`brain do`)       | [ ] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] (token streaming; not batched) |

## Getting the weights

Model id: `brain/omni`. Reserved vendor `brain/` — never auto-fetched.

- `BRAIN_OMNI_HF_DIR` — the HF checkpoint directory (`config.json` +
  tokenizer files + the sharded `model.safetensors.index.json` + shards).
  This is the gate: serving is unavailable until it's set.

## Running it

Serve it over D-Bus and/or the HTTP chat APIs:

```bash
BRAIN_OMNI_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  brain serve --dbus --openai --anthropic
```

`brain serve` prints a freshly generated API key per dialect on startup (or
write them to a file with `--api-keys-out FILE`). Text + image, over the
OpenAI-compatible API (default port 8788):

```bash
curl http://localhost:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <key from brain serve>' \
  -d '{
    "model": "brain/omni",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}]
  }'
```

The same model is reachable on the Anthropic-compatible `/v1/messages`
endpoint with `image` content blocks.

Real spoken output (the `speak` action: response text + a 24 kHz waveform) is
D-Bus/`brain do`-only today — the HTTP chat endpoints always dispatch the
`generate` action, so speech output doesn't come back over HTTP:

```python
from brain_py.dbus import BrainDBus
with BrainDBus() as brain:
    out = brain.subscribe("brain/omni", "speak",
        {"prompt": "Say hello.", "speaker": "chelsie"})
    # out.blobs["audio"]: raw mono f32 LE PCM at 24 kHz
    # out.text: the response text
```

[`examples/omni/omni.py`](../../../examples/omni/omni.py) exercises text,
speech, image and video input over both the D-Bus and HTTP transports.

## Options

- `messages` / `prompt` — chat input, same shape as brain's other chat
  models; `messages` is a flattened JSON array, `prompt` is a raw string.
- `max_new` — max tokens to generate (default `32`).
- `audio` input blob — raw mono f32 little-endian PCM at 16 kHz.
- `image` input blob — interleaved HWC f32 pixels in `[0,1]`.
- `video` input blob — N concatenated HWC f32 RGB frames plus
  `{frames,w,h,c}` metadata; brain decodes already-extracted frames, it does
  not demux a video file itself.
- `speaker` (`speak` only) — voice name (`chelsie`, `ethan`, `aiden`;
  default `chelsie`).

## Hardware and limits

- Generation is greedy (argmax) only — `temp`/`top_p`/`top_k`/`seed` are
  accepted for API compatibility but have no effect.
- `speak` is text-only on the input side today: a `speak` call does not also
  take audio/image input, and it's single-turn (no multi-turn spoken
  context).
- Weights stream from the checkpoint per generated token rather than living
  fully resident in an optimized serving layout, so throughput is
  validation-tier, not production-grade.
- Only `Qwen3-Omni-30B-A3B-Instruct` is supported. The `-Thinking` and
  `-Captioner` variants have no Talker (speech-output) component and are out
  of scope for this model.
- The full weight set needs roughly 35 GB resident even at int8 precision
  (it's a mixture-of-experts model: only a few billion parameters are active
  per token, but the router can route to any expert, so the whole set has to
  be loaded).
- No LoRA/fine-tuning path.
- No CLI (`brain do`/`brain caps`) access — D-Bus and the HTTP chat APIs
  only.
