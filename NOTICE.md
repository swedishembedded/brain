brain
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> / Swedish Embedded AB

Licensed under the Apache License, Version 2.0 (see LICENSE).

This file lists third-party CODE material referenced, transcribed, or
algorithmically ported into brain's own source during development, per
crate, to satisfy the attribution obligations of the upstream licenses named
below. In every case listed here, brain's own Rust is an independent
implementation on brain's own from-scratch GPU kernel engine (see
.agents/rules/porting.md for the methodology and .agents/roadmap/
licensing-audit.md for the evidence this file is based on) -- this file
exists because a specific data table, formula derivation, or algorithm
recipe was read from a specific named upstream file, not because brain's
code is a translation of theirs.

MODEL WEIGHT / CHECKPOINT licenses are a separate and independent question
from the source-code attributions below (a checkpoint can be restrictively
licensed even when the code that loads and runs it is not, and vice versa).
That inventory is tracked in docs/compliance/third-party-models.md and is
NOT duplicated here.

================================================================================
llama.cpp / ggml -- MIT License
Copyright (c) 2023-2026 The ggml authors
https://github.com/ggml-org/llama.cpp

  - crates/qwen3/src/gguf_import.rs, crates/qwen35/src/gguf_import.rs,
    crates/gguf/src/leaf.rs: the GGUF tensor-name vocabulary is transcribed
    from gguf-py/gguf/constants.py and gguf-py/gguf/tensor_mapping.py
    (pinned at llama.cpp commit d7a2074112d27649303fa107eb8c94db1ee435f3).
  - crates/checkpoint/src/gguf.rs, crates/gguf/src/kquant.rs: the GGUF
    quantized block layouts and dequantization arithmetic (Q4_0, Q5_0,
    Q8_0, Q2_K-Q6_K, MXFP4, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0) are
    transcribed/ported from ggml-common.h and ggml-quants.c (the
    dequantize_row_* family and the kvalues_* lookup tables).

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================================================================
HuggingFace transformers -- Apache License 2.0
Copyright 2024 The HuggingFace Team
https://github.com/huggingface/transformers

  - crates/model/src/yarn.rs: the YaRN RoPE-scaling parameter derivation
    is ported line-for-line from
    transformers.modeling_rope_utils._compute_yarn_parameters,
    cross-checked against llama.cpp's rope_yarn / ggml_rope_yarn_corr_dims.

================================================================================
ZipDepth -- MIT License
Copyright (c) 2026 Fabio Tosi
https://github.com/fabiotosi92/ZipDepth

  - crates/zipdepth/src/blocks.rs, crates/zipdepth/src/config.rs: the
    QARepBlock structure, the per-variant MODEL_CONFIGS dimension table,
    and the _pick_groups helper are ported from
    zipdepth/model/architecture.py.

  (MIT permission text as reproduced above under llama.cpp/ggml applies.)

================================================================================
BasicSR -- Apache License 2.0
Copyright 2018-2022 BasicSR Authors
https://github.com/XPixelGroup/BasicSR

  - crates/rrdbnet/src/model.rs: the RRDBNet / RRDB / ResidualDenseBlock
    forward structure (dense-block growth, 0.2 residual scaling, upsample
    ladder) is ported from basicsr/archs/rrdbnet_arch.py.

================================================================================
CosyVoice -- Apache License 2.0
Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu, et al.)
https://github.com/FunAudioLLM/CosyVoice

  - crates/cosyvoice/src/hift.rs: the HiFi-GAN vocoder and F0 predictor
    are ported algorithm-for-algorithm from
    cosyvoice/hifigan/{generator,f0_predictor}.py.
  - crates/cosyvoice/src/flow.rs: the flow-matching decoder
    (CausalMaskedDiffWithXvec / UpsampleConformerEncoder /
    CausalConditionalDecoder) is ported from cosyvoice/flow/*.py.
  - crates/cosyvoice/src/llm.rs: the LM prompt-assembly logic is ported
    from cosyvoice/llm/llm.py.

Matcha-TTS (vendored by CosyVoice's flow decoder) -- MIT License
Copyright (c) 2023 Shivam Mehta
https://github.com/shivammehta25/Matcha-TTS

  - crates/cosyvoice/src/flow.rs: the sinusoidal positional embedding.

  (MIT permission text as reproduced above under llama.cpp/ggml applies.)

================================================================================
S3Tokenizer -- Apache License 2.0
https://github.com/xingchensong/S3Tokenizer

  - crates/s3tokenizer/src/model.rs: composed from a full reading of
    s3tokenizer/model_v2.py (AudioEncoderV2, FSMNMultiHeadAttention,
    FSQCodebook), cross-checked against the released ONNX graph.

================================================================================
Kronos -- MIT License
Copyright (c) 2025 ShiYu
https://github.com/shiyu-coder/Kronos

  - crates/kronos/src/decoder.rs: the decode_s1/decode_s2 two-stage
    decode API and the SwiGLU pre-norm transformer block structure follow
    model/{kronos,module}.py.

  (MIT permission text as reproduced above under llama.cpp/ggml applies.)

================================================================================
nanoGPT -- MIT License
Copyright (c) 2022 Andrej Karpathy
https://github.com/karpathy/nanoGPT

  - crates/gpt2: architecturally modeled on nanoGPT's GPT-2 decoder for
    parity comparison ("nanogpt-parity"). No nanoGPT source is included;
    this is a defensive/courtesy credit, not a translation attribution.

  (MIT permission text as reproduced above under llama.cpp/ggml applies.)
