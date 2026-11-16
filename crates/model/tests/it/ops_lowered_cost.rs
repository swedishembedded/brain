// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A linear costs the same whichever implementation serves it. `Ops::matmul`
//! on the production registry (the native CUDA provider, the vendor library,
//! the dequantize-in-register int8 tier, where the device has them) prices
//! every step it pushes, and the totals equal what the same product costs on
//! the reference WGSL provider pinned to the fp32 tier. On a device with no
//! faster tier both sides dispatch the same kernels and the check holds
//! trivially; on one that has them it is the check that the FLOP pricer does
//! not undercount the hardware it describes.
//!
//! Both activations are packed as the reference packs them. `Ops::act` on the
//! production registry otherwise leaves out the int8 packing where every
//! candidate of the shape class reads bf16 rows (a choice of the device's
//! tuned table), and the totals would differ by work the production path
//! rightly skips rather than by what its products cost.

use std::sync::Arc;

use gpu_core::cost::CostReport;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::provider::ProviderRegistry;
use gpu_core::select::{self, Dtype, KernelSelector};
use model::ops::{Ops, Weight};

/// How the products of one activation are dispatched.
#[derive(Clone, Copy)]
enum Form {
    /// One `Ops::matmul` per weight.
    Each,
    /// One `Ops::matmul_group` over every weight (one launch where a fused kernel serves the group).
    Group,
}

/// The cost of `m` activation rows times weights of the given widths at `dt`, or `None` where the device does not
/// hold the tier.
fn priced(ops: &Ops, m: usize, widths: &[usize], k: usize, dt: Dtype, form: Form) -> Option<CostReport> {
    let g = ops.gpu();
    let x = g.storage_init("x", &vec![0.25f32; m * k]);
    let weights: Vec<Weight> = widths.iter().map(|&n| Weight::upload(ops, &vec![0.5f32; n * k], n, k, dt)).collect();
    if weights.iter().any(|w| w.dtype() != dt) {
        return None;
    }
    let outs: Vec<_> = widths.iter().map(|&n| g.storage((m * n) as u64)).collect();
    let mut steps = Vec::new();
    let act = model::ops::row_count_invariant(|| ops.act(&mut steps, &x, 0, m as u32, k as u32));
    match form {
        Form::Each => weights.iter().zip(&outs).for_each(|(w, y)| ops.matmul(&mut steps, w, &act, y, 0)),
        Form::Group => ops.matmul_group(&mut steps, &weights.iter().zip(&outs).collect::<Vec<_>>(), &act),
    }
    Some(g.cost_of(&steps))
}

fn assert_priced_alike(got: &CostReport, want: &CostReport, ctx: &str) {
    assert_eq!(want.coverage(), 1.0, "{ctx}: the reference path is fully priced: {:?}", want.uncovered);
    assert_eq!(got.coverage(), 1.0, "{ctx}: every step the production path pushes is priced: {:?}", got.uncovered);
    assert_eq!(got.total, want.total, "{ctx}: the production path costs what the reference does ({:?} vs {:?})", got.by_kernel.keys().collect::<Vec<_>>(), want.by_kernel.keys().collect::<Vec<_>>());
}

/// The production `Ops` and one on the reference provider pinned to the fp32 tier, on one device.
fn production_and_reference() -> (Ops, Ops) {
    let production = Ops::new(gpu_core::testgpu::dev(model::ops::kernel_list())).expect("Ops::new");
    let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
    let mut reference = Ops::with_providers(production.gpu().new_like(model::ops::kernel_list()), ProviderRegistry::reference(selector)).expect("Ops::with_providers");
    reference.set_gemm_precision(GemmPrecision::Fp32Reference);
    (production, reference)
}

const TIERS: [Dtype; 5] = [Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::Q4];

#[test]
fn every_matmul_tier_is_priced_like_the_reference_whichever_provider_serves_it() {
    let (production, reference) = production_and_reference();
    let (n, k) = (256usize, 512usize);
    for m in [1usize, 8, 64, 512] {
        for dt in TIERS {
            let (Some(got), Some(want)) = (priced(&production, m, &[n], k, dt, Form::Each), priced(&reference, m, &[n], k, dt, Form::Each)) else { continue };
            assert_priced_alike(&got, &want, &format!("{dt:?} m={m} n={n} k={k}"));
        }
    }
}

/// A group of products of one activation costs its members' products, however many launches serve it.
#[test]
fn a_matmul_group_is_priced_as_its_members() {
    let (production, reference) = production_and_reference();
    let (widths, k) = ([256usize, 128, 128], 512usize);
    for m in [1usize, 8, 64] {
        for dt in TIERS {
            let (Some(got), Some(want)) = (priced(&production, m, &widths, k, dt, Form::Group), priced(&reference, m, &widths, k, dt, Form::Each)) else { continue };
            assert_priced_alike(&got, &want, &format!("group of {widths:?} at {dt:?} m={m} k={k}"));
        }
    }
}
