// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every kernel the image/video diffusion models can dispatch has a cost
//! formula.
//!
//! Swedish Embedded AB implements analytic performance models for GPU
//! inference pipelines. If your team needs to know what a model will cost on
//! hardware you do not own yet, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! `brain flops --model flux2|ltxv` prices a whole generation offline. A
//! kernel with no formula is reported as UNCOVERED and left out of the totals,
//! which is honest but useless: the point of the number is that it is
//! complete. Each crate's `KERNELS` const is the exhaustive set of pipelines a
//! graph built from it can dispatch, so gating on the const - not on one
//! recorded run - is what makes this a coverage gate rather than a sample.

use gpu_core::cost::covers;

fn assert_all_covered(what: &str, set: &[(&str, &str)]) {
    let missing: Vec<&str> = set.iter().map(|(n, _)| *n).filter(|n| !covers(n)).collect();
    assert!(missing.is_empty(), "{what}: kernels with no cost formula: {missing:?}");
}

#[test]
fn flux2_dit_kernels_are_all_costed() {
    assert_all_covered("flux2::KERNELS", flux2::KERNELS);
}

#[test]
fn ltxv_dit_kernels_are_all_costed() {
    assert_all_covered("ltxv::block::KERNELS", &ltxv::block::KERNELS);
}

#[test]
fn vae_kernels_are_all_costed() {
    assert_all_covered("vae::blocks::KERNELS", &vae::blocks::KERNELS);
    assert_all_covered("vae::blocks3d::KERNELS", &vae::blocks3d::KERNELS);
}

#[test]
fn wan_kernels_are_all_costed() {
    assert_all_covered("wan::block::KERNELS", &wan::block::KERNELS);
}
