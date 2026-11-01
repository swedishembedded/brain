// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fast decode for FP8 checkpoints for its clients. If your team needs expertise in
// serving fused projections of a released FP8 model without holding its weights twice, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Spec: a row window of an FP8 weight (whole 128-row scale blocks) is a weight of its own. The product of a window
//! equals, bit for bit, the product of the same rows held as a separate weight (same candidate kernel on the same shape,
//! bound to the window's bytes and scales), at decode widths and at prefill widths, f32 and bf16 output. One product over
//! the whole fused weight equals the products of its parts when one candidate serves both shapes, and agrees within the
//! activation's storage resolution when the tuner picks different candidates for the wider shape.
//!
//! This is what lets the Gated DeltaNet layer hold `in_proj_qkv` and `in_proj_z` as ONE weight: prefill reads two row
//! windows (its own numerics and speed), a decode step reads the whole weight in one launch.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::{training_kernel_list, Ops, Weight};

const K: usize = 1024;
/// The two parts of the fused weight: whole scale blocks (10 and 6 of them, as the model's 10240 and 6144 rows are 80 and 48).
const N_FIRST: usize = 1280;
const N_SECOND: usize = 768;
const SLOT: &str = "matmul_fp8_native";

/// A pinned candidate is process-global: the tests that pin one must not overlap the ones that use the tuned choice.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let den: f64 = want.iter().map(|b| f64::from(*b).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}

struct Case<'a> {
    ops: &'a Ops,
    x: DeviceBuffer,
    m: usize,
}

impl Case<'_> {
    /// `y = x @ W[rows]^T` in f32, on a fresh activation (so no earlier product's rounded rows are shared).
    fn product(&self, w: &Weight, rows: (u32, u32)) -> Vec<f32> {
        let g = self.ops.gpu();
        let y = g.storage((self.m * rows.1 as usize) as u64);
        let mut steps = Vec::new();
        let act = self.ops.act_f32(&self.x, 0, self.m as u32, K as u32);
        self.ops.matmul_rows(&mut steps, w, rows, &act, &y, 0);
        g.submit(&[], &steps);
        g.read(&y, self.m * rows.1 as usize)
    }

    /// The bf16-output form, as the words the kernel stored; `None` when the chosen candidate has no such form.
    fn product_bf16(&self, w: &Weight, rows: (u32, u32)) -> Option<Vec<f32>> {
        let g = self.ops.gpu();
        let words = self.m * rows.1 as usize / 2;
        let y = g.storage(words as u64);
        let mut steps = Vec::new();
        let act = self.ops.act_f32(&self.x, 0, self.m as u32, K as u32);
        self.ops.matmul_rows_bf16_out(&mut steps, w, rows, &act, &y).then(|| {
            g.submit(&[], &steps);
            g.read(&y, words)
        })
    }
}

#[test]
fn a_window_of_an_fp8_weight_computes_what_the_same_rows_as_a_weight_compute() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if !ops.fp8_windows_native() {
        return brain_testutil::skip_unavailable("row windows of an FP8 weight need the native FP8 candidates (CUDA, Hopper)");
    }
    let mut rng = Lcg::new(data::rng::random_seed());
    let raw_first = rng.vec_scaled(N_FIRST * K, 0.5);
    let raw_second = rng.vec_scaled(N_SECOND * K, 0.5);
    let first = Weight::upload(&ops, &raw_first, N_FIRST, K, Dtype::F8E4M3);
    let second = Weight::upload(&ops, &raw_second, N_SECOND, K, Dtype::F8E4M3);
    // Block-wise scales: quantizing the stacked rows gives the parts' bytes and scales one after the other.
    let fused = Weight::upload(&ops, &[raw_first.as_slice(), raw_second.as_slice()].concat(), N_FIRST + N_SECOND, K, Dtype::F8E4M3);
    let (n1, n2) = (N_FIRST as u32, N_SECOND as u32);

    for m in [1usize, 8, 16, 33, 64, 200, 2048] {
        let case = Case { ops: &ops, x: ops.gpu().storage_init("x", &rng.vec_scaled(m * K, 1.0)), m };
        let want_first = case.product(&first, (0, n1));
        let want_second = case.product(&second, (0, n2));
        assert_eq!(bits(case.product(&fused, (0, n1))), bits(want_first.clone()), "m={m}: the first window differs from its rows as a weight");
        assert_eq!(bits(case.product(&fused, (n1, n2))), bits(want_second.clone()), "m={m}: the second window differs from its rows as a weight");

        // The bf16-output rounding of a prefill round (a candidate with no such form declines the same way on both).
        let (a, b) = (case.product_bf16(&first, (0, n1)), case.product_bf16(&fused, (0, n1)));
        assert_eq!(a.map(bits), b.map(bits), "m={m}: the bf16-output window differs from its rows as a weight");
        let (a, b) = (case.product_bf16(&second, (0, n2)), case.product_bf16(&fused, (n1, n2)));
        assert_eq!(a.map(bits), b.map(bits), "m={m}: the bf16-output second window differs from its rows as a weight");

        // One product over the whole weight: the parts side by side, to the activation's storage resolution when the
        // tuner names another candidate for the wider shape.
        let whole = case.product(&fused, (0, n1 + n2));
        let stacked: Vec<f32> = (0..m).flat_map(|r| want_first[r * N_FIRST..(r + 1) * N_FIRST].iter().chain(&want_second[r * N_SECOND..(r + 1) * N_SECOND]).copied()).collect();
        let err = rel_l2(&whole, &stacked);
        assert!(err < 0.05, "m={m}: the whole product is {err} (relative L2) from the products of its parts");
    }
}

#[test]
fn one_candidate_on_the_whole_weight_and_on_its_parts_gives_the_same_bits() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if !ops.fp8_windows_native() {
        return brain_testutil::skip_unavailable("row windows of an FP8 weight need the native FP8 candidates (CUDA, Hopper)");
    }
    let mut rng = Lcg::new(data::rng::random_seed());
    let raw_first = rng.vec_scaled(N_FIRST * K, 0.5);
    let raw_second = rng.vec_scaled(N_SECOND * K, 0.5);
    let first = Weight::upload(&ops, &raw_first, N_FIRST, K, Dtype::F8E4M3);
    let second = Weight::upload(&ops, &raw_second, N_SECOND, K, Dtype::F8E4M3);
    let fused = Weight::upload(&ops, &[raw_first.as_slice(), raw_second.as_slice()].concat(), N_FIRST + N_SECOND, K, Dtype::F8E4M3);
    // Every output element is a dot product of one weight row with one activation row: a candidate that serves both
    // shapes sums it in the same order whatever the width of the weight.
    for candidate in ["matmul_fp8_dq_stream_r4", "matmul_fp8_gemv_f16_a1w4d4b5", "matmul_fp8_wgmma_swap_tma_bn144c1"] {
        gpu_core::force_candidate(SLOT, candidate);
        for m in [1usize, 16, 48] {
            let case = Case { ops: &ops, x: ops.gpu().storage_init("x", &rng.vec_scaled(m * K, 1.0)), m };
            let whole = case.product(&fused, (0, (N_FIRST + N_SECOND) as u32));
            let (a, b) = (case.product(&first, (0, N_FIRST as u32)), case.product(&second, (0, N_SECOND as u32)));
            let stacked: Vec<f32> = (0..m).flat_map(|r| a[r * N_FIRST..(r + 1) * N_FIRST].iter().chain(&b[r * N_SECOND..(r + 1) * N_SECOND]).copied()).collect();
            assert_eq!(bits(whole), bits(stacked), "{candidate} m={m}: one product over the fused weight differs from the products of its parts");
        }
    }
    gpu_core::clear_forced_candidates();
}

/// Back-to-back launches of the two windows and of the whole fused weight at the production shapes of a Gated DeltaNet layer
/// (`in_proj_qkv` 10240 rows, `in_proj_z` 6144, K = 5120) over the decode and prefill widths, with no synchronisation between
/// launches: sliced bindings under the timing that faulted a tensor-map decode kernel once (`#[ignore]`: a stress run of
/// seconds, then `dmesg | grep -i xid` stays empty).
///
/// ```text
/// BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda make test/one P=brain-model T=fp8_row_windows ARGS="--ignored --nocapture windows_back_to_back_stress"
/// ```
#[test]
#[ignore]
fn windows_back_to_back_stress() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if !ops.fp8_windows_native() {
        return brain_testutil::skip_unavailable("row windows of an FP8 weight need the native FP8 candidates (CUDA, Hopper)");
    }
    const KP: usize = 5120;
    const N1: u32 = 10240;
    const N2: u32 = 6144;
    let g = ops.gpu();
    let mut rng = Lcg::new(data::rng::random_seed());
    // Rows of random values quantize to every scale and code the kernels see; one block row of values repeated keeps the host cheap.
    let band = rng.vec_scaled(128 * KP, 0.5);
    let raw: Vec<f32> = band.iter().copied().cycle().take((N1 + N2) as usize * KP).collect();
    let fused = Weight::upload(&ops, &raw, (N1 + N2) as usize, KP, Dtype::F8E4M3);
    for round in 0..40u32 {
        for m in [1usize, 4, 16, 33, 64, 128, 1024, 2048] {
            let x = g.storage_init("x", &rng.vec_scaled(m * KP, 1.0));
            let (y1, y2, y) = (g.storage((m * N1 as usize) as u64), g.storage((m * N2 as usize) as u64), g.storage((m * (N1 + N2) as usize) as u64));
            let act = ops.act_f32(&x, 0, m as u32, KP as u32);
            for _ in 0..6 {
                let mut steps = Vec::new();
                ops.matmul_rows(&mut steps, &fused, (0, N1), &act, &y1, 0);
                ops.matmul_rows(&mut steps, &fused, (N1, N2), &act, &y2, 0);
                if m <= 128 {
                    ops.matmul_rows(&mut steps, &fused, (0, N1 + N2), &act, &y, 0);
                }
                g.submit(&[], &steps);
            }
        }
        g.poll_wait();
        if round % 10 == 0 {
            eprintln!("windows_back_to_back_stress: round {round}");
        }
    }
}
