// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR ("Scaling Up to Excellence", CVPR 2024) - photo-realistic blind image
//! restoration, driven by a **frozen SDXL 1.0 base UNet** plus a 1.24B control
//! trunk (`GLVControl`, a hand-written copy of SDXL's encoder + middle block)
//! and 12 adaptor modules (10 `ZeroSFT` + 2 `ZeroCrossAttn`).
//!
//! This is the id-reservation crate, not the port. Registering an
//! architecture before any of its code exists fixes the crate directory
//! name, the package name, the `brain supir <verb>` CLI word, and this
//! page's own filename in one place before anything else is written, rather
//! than letting four call sites drift into four different spellings for the
//! same architecture.
//!
//! ## Status
//!
//! Implementation has not started. The plan is: seams in `sdxlunet`/`vae`/
//! `diffusion`/`imaging` first (SUPIR's adaptors REPLACE the SDXL UNet's
//! skip concatenation rather than adding a residual, which needs a new
//! recording-time seam none of those crates expose yet), then this crate's
//! `config`/`import`/`trunk`/`adaptors`/`model`/`pipeline`, then INT8,
//! training, serving, and NPU export.
//!
//! ## Licence
//!
//! The SUPIR weights are released under the SUPIR Software License Agreement
//! (copyright 2024 SupPixel Pty Ltd): **non-commercial only**, and the
//! definition of commercial use expressly includes SaaS deployment and using
//! outputs as ML training data. There is no official HuggingFace repo. This
//! crate ships no `default_ref`/auto-fetch for that reason - a user points
//! brain at weights they obtained themselves, at their own licensing risk.
