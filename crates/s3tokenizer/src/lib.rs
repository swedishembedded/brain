// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! S3Tokenizer: the supervised-semantic speech tokenizer
//! [`crate::cosyvoice`](../cosyvoice/index.html) conditions its LM on. A
//! Whisper-style mel front end (16 kHz, 128 mels) feeds a RoPE + FSMN
//! multi-head-attention audio encoder (`AudioEncoderV2`/`V3`: 6 blocks for
//! CosyVoice 2's `speech_tokenizer_v2.onnx`, 12 for CosyVoice 3's
//! `speech_tokenizer_v3.onnx`), whose 1280-d hidden state is projected to 8
//! dims and **finite-scalar-quantized** (3 levels per dim, `3^8 = 6561`
//! codes) into a single integer token per 25 Hz frame:
//!
//! ```text
//! d_i = round(tanh(project_down(h)_i) * 0.9990000128746033) + 1   in {0,1,2}
//! index = sum_i d_i * 3^i                                         in [0, 6561)
//! ```
//!
//! There is no official decoder (`FSQCodebook::decode` raises
//! `NotImplementedError` upstream) - tokens flow one-way into
//! `cosyvoice`'s own speech-token embedding table.
//!
//! Reference: `xingchensong/S3Tokenizer` (`s3tokenizer/model_v2.py`,
//! `model_v3.py`) - the faithful from-PyTorch reimplementation of the two
//! ONNX-only upstream releases. Imported via `crates/onnx`'s
//! coverage-checked `walk`.
//!
//! Status: architecture registered (`crates/arch`), name reserved. Import,
//! forward, and the exact-token-id parity gate are not yet implemented.
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.
