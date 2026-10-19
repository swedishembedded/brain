# SAM-1 / ViTDet ViT-B tower (component)

The SAM-1 ViT-B tower (decomposed relative-position bias) that forms the
front half of [DeepSeek-OCR](deepseek2ocr.md)'s DeepEncoder, ahead of a 16x
conv token compressor and the CLIP-L spatial tower. Not independently
servable: it has no capability manifest or CLI verb of its own.

It also builds DeepSeek-VL's high-resolution tower (`sam_b_downsample`,
`SamViTConfig::deepseek_vl`): the same tower with the neck output bilinearly
resized to 96x96 before the compressor (a 24x24 = 576-token output), plus the
"HD" branch that runs the first global-attention block's output through a
second neck and the same compressor and adds it scaled by a learned scalar.
Its weights load in place from the `deepseek-ai/deepseek-vl-7b-chat`
safetensors checkpoint (`sam1::hf`, under the
`vision_model.vision_tower_high.vision_tower.` prefix).

Package: `brain-sam1`.
