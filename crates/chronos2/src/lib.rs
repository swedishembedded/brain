// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chronos-2 (Amazon) reimplemented from scratch in brain — encoder-only,
//! T5-style patch transformer with alternating time / group attention and a
//! multi-patch quantile head. Apache-2.0 reference (`amazon/chronos-2`, 120M,
//! native fp32), imported exactly and parity-gated per stage.
//!
//! Build order (brain's model-add recipe):
//! 1. [`config`] — dims + `param_list()` in the reference's own key names, and
//!    the device-free T0 layout gate. **(this milestone)**
//! 2. WGSL kernels — reuse rmsnorm/matmul/softmax; add unscaled attention, RoPE
//!    (half-split), the arcsinh scaler, and the quantile-head rearrange.
//! 3. `model.rs` — SSA `build_forward`: patch-embed → REG → future tokens →
//!    L×[time-attn(RoPE) → group-attn(transposed) → ReLU-FFN] → final-norm →
//!    head → denorm; `impl model::Model`.
//! 4. `import.rs` — strict safetensors import (native fp32, no bf16 path).
//! 5. Parity ladder T1..T5 vs a PyTorch dump (cosine/max-abs), then register as
//!    a [`forecast::ForecastModel`] (native representation = quantiles).
//!
//! Phase 1 is the **univariate** core (`group_ids = arange`, so the group mask
//! is the identity — group attention runs but mixes no variates). Real
//! covariates and long-horizon pipeline unrolling come after parity.

pub mod config;
pub mod forecaster;
pub mod import;
pub mod model;
pub mod preprocess;
pub mod train;

pub use config::{Chronos2Config, Param, QUANTILES};
pub use forecaster::Chronos2Forecaster;
pub use model::Chronos2;
pub use preprocess::{context_features, future_features, instance_norm, instance_norm_inverse, LocScale};
