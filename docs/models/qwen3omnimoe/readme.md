# Omni-Modal Assistant (Qwen3-Omni)

Text, audio, image and video in - text out, plus real synthesized speech out.
Qwen3-Omni is brain's most ambitious served model: a single assistant that
can read a prompt, listen to a clip, look at an image or video frames, answer
in text, and - when you ask it to speak - respond with an actual spoken
waveform instead of just text. Reach for it when you want one model handling
mixed-modality input and, optionally, a spoken reply, rather than wiring
separate ASR/VLM/TTS models together yourself.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI (`brain <arch> <action>`)       | [ ] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] (token streaming; not batched) |

## Getting the weights

Model id: `brain/qwen3omnimoe`. Reserved vendor `brain/` - never auto-fetched.

- `BRAIN_QWEN3OMNIMOE_HF_DIR` - the HF checkpoint directory (`config.json` +
  tokenizer files + the sharded `model.safetensors.index.json` + shards).
  This is the gate: serving is unavailable until it's set.

## GPU residency (the sharded int8 path)

`brain/qwen3omnimoe` above is the **validation tier**: it has the full chat and
audio/image/video surface, but every decoder layer's weights - including all
128 experts - are re-read from the checkpoint for every generated token, so
nothing stays on the GPU between calls. It is correct and slow.

The **GPU-resident** path is a second, separately-gated model,
`brain/Qwen3-Omni-30B-A3B-Instruct-W8A16` -- named for what it actually is:
per-output-channel symmetric INT8 WEIGHT-ONLY quantization (one f32 scale per
output channel, MoE expert linears only) with full-precision activations,
the current HF/vLLM-recognized tag for this scheme (not GGUF's `Q8_0`, a
different, block-quantized format). Its weights live in VRAM across calls,
layer-sharded across however many GPUs their real per-layer byte cost needs.
It takes the same chat request as `brain/qwen3omnimoe` (text only - no audio/image/
video and no `speak`), and is dramatically faster per token than the
streaming path above, on the same prompt with the same output, since the
weights never leave VRAM between calls. Measure the actual ratio on your own
hardware with `brain perf run`; a number here would describe one specific
machine at one point in time.

It wants a brain-native W8A16 checkpoint, which is not the format you
downloaded - convert once (~8 minutes, 66 GB in, 33.6 GB out):

```bash
brain omni import --hf /path/to/Qwen3-Omni-30B-A3B-Instruct \
                  --out /path/to/Qwen3-Omni-30B-A3B-Instruct-W8A16.safetensors
```

The conversion streams one tensor at a time (peak host memory is roughly one
tensor's f32 expansion, never the whole ~70 GB checkpoint) and quantizes
every rank-2 weight to int8. Then serve it:

```bash
BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT=/path/to/Qwen3-Omni-30B-A3B-Instruct-W8A16.safetensors \
BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  brain serve --openai --dbus --device vulkan
```

An int8 checkpoint is a single `.safetensors` and carries no tokenizer, so
`BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR` says where to read `tokenizer.json` (or
`vocab.json` + `merges.txt`) from - normally the HF directory you converted
from. It defaults to the checkpoint's own directory when that holds tokenizer
files, then to `BRAIN_QWEN3OMNIMOE_HF_DIR`. Without any of them the model still loads
and still serves raw token ids, but it is not on the chat endpoints.

How the sharding decides itself:

- Per-layer VRAM cost is read from the checkpoint's own header - no shape
  constants, no per-model sharding code. Placement is
  `model::shard::plan_fewest_devices`, the same generic capacity-aware
  planner any model crate can call with its own layer description.
- It uses the **fewest cards that fit**: one GPU if the model fits one, three
  if it needs three. Nothing assumes two.
- Cards may differ in size. A 24 GB and an 8 GB card get layers in roughly
  that proportion, because each stage is checked against *its own* device's
  usable capacity (`nvidia-smi` total minus `--reserve-gb`).
- If it fits nowhere, that is reported as an unplaceable model rather than
  discovered as an out-of-memory failure partway through the load.

Loading is bounded end to end: packed int8 weights are handed to the driver
straight out of the memory mapping (no host copy at all), and the tensors
that have to be unpacked to fp32 on the way in (attention/router projections,
`lm_head`) are dequantized a row block at a time rather than expanded whole.
The token-embedding table - 1.2 GB as fp32 - is never materialized either;
rows are read on demand.

Two limits worth knowing before you point it at real weights:

- **Use `--device vulkan`.** On the default wgpu backend a non-ReBAR Pascal
  card holds roughly double each uploaded buffer resident, measured by the
  committed regression test `crates/gpu-core/tests/vram_overhead.rs`; brain's
  own Vulkan backend does not carry that overhead. At the real Thinker shape
  the difference decides whether the model fits two 24 GB cards at all.
- **This path is Thinker text only.** It has no multimodal splice and no
  `speak`, so an image, an audio clip or a spoken reply still needs
  `brain/qwen3omnimoe`. It IS on `/v1/chat/completions`, `/v1/messages` and D-Bus with
  the same `messages`/`prompt` contract, and it additionally accepts a raw
  `ids` blob (LE `u32` token ids, meta `max_new_tokens`/`eos_ids`) for callers
  that tokenize themselves.

## Running it

Serve it over D-Bus and/or the HTTP chat APIs:

```bash
BRAIN_QWEN3OMNIMOE_HF_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
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
    "model": "brain/qwen3omnimoe",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}]
  }'
```

The same model is reachable on the Anthropic-compatible `/v1/messages`
endpoint with `image` content blocks.

Real spoken output (the `speak` action: response text + a 24 kHz waveform) is
D-Bus/`brain do`-only today - the HTTP chat endpoints always dispatch the
`generate` action, so speech output doesn't come back over HTTP:

```python
from brain_py.dbus import BrainDBus
with BrainDBus() as brain:
    out = brain.subscribe("brain/qwen3omnimoe", "speak",
        {"prompt": "Say hello.", "speaker": "chelsie"})
    # out.blobs["audio"]: raw mono f32 LE PCM at 24 kHz
    # out.text: the response text
```

[`examples/omni/omni.py`](../../../examples/omni/omni.py) exercises text,
speech, image and video input over both the D-Bus and HTTP transports.

## Options

- `messages` / `prompt` - chat input, same shape as brain's other chat
  models; `messages` is a flattened JSON array, `prompt` is a raw string.
- `max_new` - max tokens to generate (default `32`).
- `audio` input blob - raw mono f32 little-endian PCM at 16 kHz.
- `image` input blob - interleaved HWC f32 pixels in `[0,1]`.
- `video` input blob - N concatenated HWC f32 RGB frames plus
  `{frames,w,h,c}` metadata; brain decodes already-extracted frames, it does
  not demux a video file itself.
- `speaker` (`speak` only) - voice name (`chelsie`, `ethan`, `aiden`;
  default `chelsie`).

## Hardware and limits

- Generation is greedy (argmax) only - `temp`/`top_p`/`top_k`/`seed` are
  accepted for API compatibility but have no effect.
- `speak` is text-only on the input side today: a `speak` call does not also
  take audio/image input, and it's single-turn (no multi-turn spoken
  context).
- On `brain/qwen3omnimoe`, weights stream from the checkpoint per generated token
  rather than living resident, so throughput is validation-tier, not
  production-grade - the large majority of a request's wall time is
  re-reading the layers that do not fit in VRAM, not kernel execution.
  Measure the actual split on your hardware with `brain perf run`. The
  GPU-resident alternative is
  `brain/Qwen3-Omni-30B-A3B-Instruct-W8A16` (see "GPU residency" above),
  which keeps the chat surface and trades only the multimodal/`speak` half
  for weights that stay on the cards.
- Only `Qwen3-Omni-30B-A3B-Instruct` is supported. The `-Thinking` and
  `-Captioner` variants have no Talker (speech-output) component and are out
  of scope for this model.
- The Thinker needs roughly 31.7 GiB of VRAM resident at int8 (it's a
  mixture-of-experts model: only a few billion parameters are active per
  token, but the router can route to any expert, so the whole set has to be
  loaded). Computed from the real config: ~27.1 GiB of routed experts,
  ~3.4 GiB of attention/router/norms and ~1.2 GiB of `lm_head`, the last two
  held as fp32 because no int8 kernel path consumes them. Against two 24 GB
  cards less the default 2 GB per-card reserve (44 GiB usable) that fits with
  ~12 GiB to spare on the native Vulkan backend - and does **not** fit on
  wgpu, where the 2x upload residency above turns it into ~63 GiB.
  bf16 is not an option on this class of hardware regardless - brain's
  kernels are fp32 throughout, and Pascal's fp16 rate is 1/64.
- No LoRA/fine-tuning path.
- No CLI (`brain do`/`brain caps`) access - D-Bus and the HTTP chat APIs
  only.
