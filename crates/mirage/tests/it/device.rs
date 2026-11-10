// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device operators against the host oracle. The device is whatever the
//! process selected (`BRAIN_DEVICE=cpu` runs the CPU backend's WGSL JIT,
//! `BRAIN_BACKEND=cuda BRAIN_DEVICE=gpuN` one CUDA card).

use data::rng::Lcg;
use mirage::device::hconv::DeviceHConv;
use mirage::device::Device;
use mirage::vision::encoder::HConv;
use mirage::vision::weights::Weights;

use crate::common::{released_dir, tiny_checkpoint};
use crate::vision::assert_stage;

fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Lcg::new(seed); // seed: filler compared between two paths
    (0..n).map(|_| rng.signed() * scale).collect()
}

fn device(test: &str) -> Option<Device> {
    crate::common::cap_threads();
    match Device::open() {
        Ok(d) => {
            eprintln!("{test}: device kind {}", d.gpu().kind());
            Some(d)
        }
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("{test}: no device: {e}"));
            None
        }
    }
}

#[test]
fn the_gated_product_kernel_matches_the_formula() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("hconv_glu") else { return };
    let (rows, n) = (7usize, 40usize);
    let ab = noise(rows * 2 * n, 1, 2.0);
    let bias = noise(n, 2, 0.5);
    let want: Vec<f32> = (0..rows * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            model::hostmath::gelu_tanh(ab[r * 2 * n + c] + bias[c]) * ab[r * 2 * n + n + c]
        })
        .collect();
    let g = dev.gpu();
    let (ab_d, bias_d, z) = (g.storage_init("ab", &ab), g.storage_init("bias", &bias), g.storage((rows * n) as u64));
    let mut s = Vec::new();
    dev.hconv_glu(&mut s, &ab_d, &bias_d, rows as u32, n as u32, &z);
    assert_stage("hconv_glu", &dev.run_read(&s, &z, rows * n), &want);
}

fn check_hconv(dev: &Device, name: &str, host: &HConv, width: usize, rows: usize) {
    let x = noise(rows * width, 7, 1.0);
    let want = host.forward(&x, rows);
    let dh = DeviceHConv::upload(dev, host);
    let g = dev.gpu();
    let (xd, y) = (g.storage_init("x", &x), g.storage((rows * width) as u64));
    let mut s = Vec::new();
    dh.encode(dev, &mut s, &xd, rows as u32, &y);
    assert_stage(name, &dev.run_read(&s, &y, rows * width), &want);
}

#[test]
fn a_folded_hconv_matches_the_host_mixer_on_the_tiny_model() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("tiny hconv") else { return };
    let ck = tiny_checkpoint();
    let w = Weights::new(&ck);
    let host = HConv::new(&w, "body_0", ck.config.dim, ck.config.hconv_kernel);
    check_hconv(&dev, "tiny hconv", &host, ck.config.dim, 11);
}

#[test]
fn a_folded_hconv_matches_the_host_mixer_with_released_weights() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("released hconv: the released checkpoint is absent (make testdata/fetch)");
        return;
    };
    let Some(dev) = device("released hconv") else { return };
    let ck = mirage::import::load_dir(&dir, &mirage::import::ImportOptions::default()).unwrap();
    let w = Weights::new(&ck);
    for prefix in ["body_0", "body_15", "top_0", "top_11"] {
        let host = HConv::new(&w, prefix, ck.config.dim, ck.config.hconv_kernel);
        check_hconv(&dev, &format!("{prefix} hconv"), &host, ck.config.dim, 192);
    }
}
