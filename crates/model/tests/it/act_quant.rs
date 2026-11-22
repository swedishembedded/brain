// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving a model with the low-precision activation numerics it was
// trained with for its clients. If your team needs expertise in NVFP4 and MXFP8 microscaled inference then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the declared activation round trip (`act_qdq_nvfp4_mxfp8.wgsl`, [`model::act_quant::qdq_step`]) is its host
//! reference ([`model::hostmath::nvfp4_four_over_six_qdq`], [`model::hostmath::mxfp8_qdq`]) bit for bit, on the CPU
//! backend and on CUDA when a card is selected, over rows of every magnitude, zero rows and blocks, outliers and
//! values on the E2M1 / E4M3 rounding ties; a row width that is not a whole number of blocks is refused.

use checkpoint::packed::ActivationScheme;
use data::rng::{random_seed, Lcg};
use gpu_core::Gpu;
use model::act_quant::{qdq_step, KERNELS};
use model::hostmath::{mxfp8_qdq, nvfp4_four_over_six_qdq};

/// Rows of width `k` spanning magnitudes 1e-6 .. 1e4, with a zero row, a zero block, outliers and rounding ties.
fn rows(rng: &mut Lcg, n: usize, k: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(n * k);
    for r in 0..n {
        let scale = 10f32.powf(rng.unit() * 10.0 - 6.0);
        let mut row: Vec<f32> = (0..k).map(|_| rng.signed() * scale).collect();
        match r % 5 {
            0 => row.iter_mut().for_each(|v| *v = 0.0),
            1 => row[..32].iter_mut().for_each(|v| *v = 0.0),
            2 => row[rng.next_u32() as usize % k] = 300.0 * scale,
            3 => {
                // A row amax of 6 makes the six-based grid the identity, so these sit on its E2M1 ties, and the two
                // candidates' errors tie exactly (3.0 each): the six-based scale must be kept.
                row.iter_mut().for_each(|v| *v = rng.signed());
                let ties = [6.0, 2.5, 3.0, 1.0, 4.0, -0.75, 1.75, 0.25, 3.5, 1.25, -2.5, -3.5, 0.0, 0.0, 0.0, 0.0];
                row[..16].copy_from_slice(&ties);
            }
            _ => {}
        }
        x.extend(row);
    }
    x
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn run(g: &Gpu, scheme: ActivationScheme, x: &[f32], k: usize) -> Vec<f32> {
    let rows = (x.len() / k) as u32;
    let (xb, out) = (g.storage_init("x", x), g.storage(x.len() as u64));
    g.submit(&[], &[qdq_step(g, scheme, &xb, &out, rows, k as u32).unwrap()]);
    g.read(&out, x.len())
}

fn matches_host(g: &Gpu, label: &str) {
    let mut rng = Lcg::new(random_seed());
    for k in [1152usize, 2304, 32] {
        let x = rows(&mut rng, 37, k);
        let nv = run(g, ActivationScheme::Nvfp4FourOverSixMae, &x, k);
        assert_eq!(bits(&nv), bits(&nvfp4_four_over_six_qdq(&x, k)), "{label}: NVFP4 four-over-six rows of {k}");
        let mx = run(g, ActivationScheme::Mxfp8E4m3, &x, k);
        assert_eq!(bits(&mx), bits(&mxfp8_qdq(&x, k)), "{label}: MXFP8 rows of {k}");
        assert_ne!(bits(&nv), bits(&x), "{label}: the round trip rounds");
    }
}

#[test]
fn the_activation_round_trip_is_its_host_reference_bit_for_bit() {
    let g = gpu_core::testgpu::dev(&KERNELS);
    matches_host(&g, gpu_core::backend_name());
    let x = vec![1.0f32; 48];
    let (xb, out) = (g.storage_init("x", &x), g.storage(48));
    let Err(err) = qdq_step(&g, ActivationScheme::Mxfp8E4m3, &xb, &out, 1, 48) else { panic!("48 values are not whole MXFP8 blocks") };
    assert!(err.contains("32") && err.contains("48"), "{err}");
}

#[test]
fn the_activation_round_trip_is_its_host_reference_bit_for_bit_on_cuda() {
    if brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu()) {
        return;
    }
    match Gpu::try_new_cuda(&KERNELS) {
        Ok(g) => matches_host(&g, "cuda"),
        Err(e) => brain_testutil::skip_unavailable(&format!("no CUDA device: {e}")),
    }
}
