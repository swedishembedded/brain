# Moondream 3 (not yet servable)

A third vision-language architecture alongside [FastVLM](fastvlm.md) and
[Qwen3-VL](qwen3vl.md) (compared on the [overview page](vlm.md)): a SigLIP
ViT vision encoder (overlap multi-crop) plus a parallel-block sparse-MoE
decoder with expert sharding.

This is a real, verified port - decoder gradient-checked, import-covered
(662 tensors) and stage-by-stage parity-checked against real weights (a
decoder bug a gradcheck alone had missed was caught this way) - but it has
no capability manifest, no CLI verb and no serving surface yet, so it is not
something you can run as a model today. Region/point/detect heads are
recognized on import but not built.

Model id (once served): `brain/moondream3` (package `brain-moondream3`).
