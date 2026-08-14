# DeepSeek-V2-family decoder (component)

The MHA decoder (plain multi-head attention, not MLA, despite the family
name) [DeepSeek-OCR](deepseek2ocr.md) is built on - 12 layers, 64 routed
experts top-6 + 2 shared fused. Not independently servable: it has no
capability manifest or CLI verb of its own, reached only as the decoder half
of DeepSeek-OCR's composed pipeline.

Package: `brain-deepseek2`.
