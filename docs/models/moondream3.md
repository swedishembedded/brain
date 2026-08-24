# Moondream 3 (not yet servable)

A third vision-language architecture alongside [FastVLM](fastvlm.md) and
[Qwen3-VL](qwen3vl.md) (compared on the [overview page](vlm.md)): a SigLIP
ViT vision encoder (overlap multi-crop) plus a parallel-block sparse-MoE
decoder with expert sharding.

This is a real, verified port - decoder gradient-checked, import-covered
(662 tensors) and stage-by-stage parity-checked against real weights (a
decoder bug a gradcheck alone had missed was caught this way). It can load a
checkpoint, read that checkpoint's own `config.json` (and refuse a
differently-shaped one), encode an image, and greedily generate text.

What it cannot do yet is FIT. At the released configuration the decoder is
8.8 B parameters - 20 MoE layers of 64 experts each - which is about 33 GiB
of fp32 weights, plus roughly 10 GiB of activation scratch, so there is no
machine this runs on as it stands. Quantizing the experts to int8 and sharing
one activation scratch set across the 24 blocks brings that to under 9 GiB;
until then a capability manifest, a CLI verb and a residency adapter would
advertise something that cannot execute, so they are deliberately not there.

Region/point/detect heads are recognized on import but not built.

Model id (once served): `brain/moondream3` (package `brain-moondream3`).
