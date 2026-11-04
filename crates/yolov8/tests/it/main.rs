// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `bench_forward_p40` keeps a binary of its own.

mod backend_grads;
mod import_real;
mod optimizer_state;
mod p11_eval_inference;
mod p12_npu_refactor;
mod p1_bn;
mod p1_conv;
mod p1_forward_pin;
mod p1_loss;
mod p1_spatial;
mod p2_blocks;
mod p3_gradcheck;
mod p4_detection;
mod p5_overfit;
mod p6_infer;
mod p6_nms;
mod p7_capability;
mod p7_negative;
mod p7_unit;
mod p8_names;
mod parity;
mod recompute;
