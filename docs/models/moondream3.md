# Moondream 3

A third vision-language architecture alongside [FastVLM](fastvlm.md) and
[Qwen3-VL](qwen3vl.md) (compared on the [overview page](vlm.md)): a SigLIP
ViT vision encoder (overlap multi-crop) plus a parallel-block sparse-MoE
decoder with expert sharding.

This is a real, verified port - decoder gradient-checked, import-covered
(662 tensors) and stage-by-stage parity-checked against real weights (a
decoder bug a gradcheck alone had missed was caught this way). It can load a
checkpoint, read that checkpoint's own `config.json` (and refuse a
differently-shaped one), encode an image, and greedily generate text.

It is **served**: one `caption` action (an image plus an instruction in,
generated text out, streamed per token) reachable as `brain moondream3 caption`,
over D-Bus, and through the OpenAI/Anthropic surfaces, with a residency adapter
and a runnable example.

**Precision is the parameter that matters.** At the released configuration the
decoder is 8.8 B parameters - 20 MoE layers of 64 experts each - which in fp32
is about 33 GiB of weights plus roughly 10 GiB of activation scratch. No
ordinary machine runs that. The default `int8` path quantizes the expert
weights and puts all 24 blocks on a single shared activation set, bringing the
whole model to under 9 GiB. `precision=fp32` is still accepted for a machine
with the room; the scheduler budgets the two as separate instances, so asking
for fp32 without the memory fails placement cleanly rather than evicting a
working int8 instance.

Decode is **KV-cached**: the prompt pays one batched forward that also seeds
every layer's cache, and each token after that is a single incremental step
rather than a full recompute. The prefill over a 730-row image prefix is still
the dominant cost of a short caption, so `max_new` defaults low.

Requests are **not batched**: each one carries its own image, so the vision
pass is per-request and the decoder has no batch axis.

Region/point/detect heads are recognized on import but not built, so pointing,
detection and region captioning are unavailable - `caption` is the only action.

| | |
|---|---|
| model id | `brain/moondream3` |
| weights | `BRAIN_MOONDREAM3_WEIGHTS` - the checkpoint directory |
| action | `caption` (streaming) |
| example | `examples/vision/moondream3_caption.py` |
| placement | GPU when one is budgeted, else the CPU pool |

Package: `brain-moondream3`.
