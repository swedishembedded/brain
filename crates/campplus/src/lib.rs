// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CAM++ speaker encoder: 80-dim kaldi-style fbank (16 kHz) in, a 192-d
//! x-vector out. D-TDNN backbone (dense TDNN blocks with growth-rate
//! concatenation) fed by a 2-D conv stem (`FCM`), with **context-aware
//! masking** (`CAMLayer`) - a per-position gate computed from the global mean
//! plus a two-second segment-pooled mean - gating each depthwise TDNN branch instead
//! of the plain per-channel SE gate the name might suggest.
//!
//! Ported for [`crate::cosyvoice`](../cosyvoice/index.html) (CosyVoice 2 and
//! 3 both ship the byte-identical `campplus.onnx`, used for zero-shot voice
//! cloning), from the reference `modelscope/3D-Speaker`
//! (`speakerlab/models/campplus/DTDNN.py`). Imported from ONNX via
//! `crates/onnx`'s coverage-checked `walk`, the same import discipline
//! `crates/scrfd`/`crates/arcface` use.
//!
//! Status: import (two-way coverage against the released `campplus.onnx`),
//! forward, and real-weight parity vs the ONNX golden are implemented -
//! `tests/parity.rs`. See `crate::model`'s module doc for the dispatch
//! choices (`conv3d` for the `FCM` stem's asymmetric stride, `CAMLayer`'s
//! context computed on the host).
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod config;
pub mod import;
pub mod model;
