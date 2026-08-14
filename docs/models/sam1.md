# SAM-1 / ViTDet ViT-B tower (component)

The SAM-1 ViT-B tower (decomposed relative-position bias) that forms the
front half of [DeepSeek-OCR](deepseek2ocr.md)'s DeepEncoder, ahead of a 16x
conv token compressor and the CLIP-L spatial tower. Not independently
servable: it has no capability manifest or CLI verb of its own.

Package: `brain-sam1`.
