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

fn upload(dev: &Device, v: &[f32]) -> gpu_core::DeviceBuffer {
    dev.gpu().storage_init("t", v)
}

fn run(dev: &Device, s: &[gpu_core::Step], y: &gpu_core::DeviceBuffer, n: usize) -> Vec<f32> {
    dev.run_read(s, y, n)
}

#[test]
fn convolutions_match_the_host_operators() {
    use mirage::device::prims::ConvShape;
    use mirage::vision::ops::{conv_dense, depthwise, Map};
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("conv") else { return };
    let (b, h, w, cin, cout) = (2usize, 9usize, 12usize, 5usize, 7usize);
    let x = noise(b * h * w * cin, 3, 1.0);
    for (kh, kw, sy, sx) in [(3usize, 3usize, 2usize, 2usize), (3, 3, 1, 1), (1, 5, 1, 1), (3, 3, 2, 1)] {
        let g = ConvShape { batch: b as u32, h: h as u32, w: w as u32, kh: kh as u32, kw: kw as u32, sy: sy as u32, sx: sx as u32 };
        let (ho, wo) = (g.ho() as usize, g.wo() as usize);
        let m = Map { h, w, c: cin };
        // Dense.
        let wt = noise(kh * kw * cin * cout, 4, 0.5);
        let mut want = Vec::new();
        for i in 0..b {
            want.extend(conv_dense(&x[i * h * w * cin..(i + 1) * h * w * cin], m, &wt, kh, kw, cout, sy, sx).0);
        }
        let (xd, wd, y) = (upload(&dev, &x), upload(&dev, &wt), dev.gpu().storage((b * ho * wo * cout) as u64));
        let mut s = Vec::new();
        dev.conv2d(&mut s, g, cin as u32, cout as u32, &xd, &wd, &y);
        assert_stage(&format!("conv2d {kh}x{kw} s{sy}{sx}"), &run(&dev, &s, &y, want.len()), &want);
        // Depthwise.
        let dwt = noise(kh * kw * cin, 5, 0.5);
        let mut want = Vec::new();
        for i in 0..b {
            want.extend(depthwise(&x[i * h * w * cin..(i + 1) * h * w * cin], m, &dwt, kh, kw, sy, sx).0);
        }
        let (dwd, y) = (upload(&dev, &dwt), dev.gpu().storage((b * ho * wo * cin) as u64));
        let mut s = Vec::new();
        dev.dwconv(&mut s, g, cin as u32, &xd, &dwd, &y);
        assert_stage(&format!("dwconv {kh}x{kw} s{sy}{sx}"), &run(&dev, &s, &y, want.len()), &want);
    }
}

#[test]
fn row_and_map_primitives_match_their_formulas() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("prims") else { return };
    let (rows, d) = (6usize, 10usize);
    let x = noise(rows * d, 6, 1.0);
    let y = noise(rows * d, 7, 1.0);
    let m: Vec<f32> = (0..rows).map(|r| (r % 3 != 0) as u8 as f32).collect();
    let (xd, yd, md) = (upload(&dev, &x), upload(&dev, &y), upload(&dev, &m));
    // add_masked
    let want: Vec<f32> = (0..rows * d).map(|i| x[i] + m[i / d] * y[i]).collect();
    let acc = upload(&dev, &x);
    let mut s = Vec::new();
    dev.add_masked(&mut s, &acc, &yd, &md, rows as u32, d as u32);
    assert_stage("add_masked", &run(&dev, &s, &acc, rows * d), &want);
    // rows_mask
    let want: Vec<f32> = (0..rows * d).map(|i| x[i] * m[i / d]).collect();
    let acc = upload(&dev, &x);
    let mut s = Vec::new();
    dev.rows_mask(&mut s, &acc, &md, rows as u32, d as u32);
    assert_stage("rows_mask", &run(&dev, &s, &acc, rows * d), &want);
    // swiglu_rows
    let uv = noise(rows * 2 * d, 8, 1.0);
    let want: Vec<f32> = (0..rows * d).map(|i| uv[i / d * 2 * d + i % d] * model::hostmath::silu(uv[i / d * 2 * d + d + i % d])).collect();
    let (uvd, z) = (upload(&dev, &uv), dev.gpu().storage((rows * d) as u64));
    let mut s = Vec::new();
    dev.swiglu_rows(&mut s, &uvd, &z, rows as u32, d as u32);
    assert_stage("swiglu_rows", &run(&dev, &s, &z, rows * d), &want);
    // gated_mix
    let (bg, bv) = (noise(d, 9, 0.5), noise(d, 10, 0.5));
    let want: Vec<f32> = (0..rows * d).map(|i| model::hostmath::silu(x[i] + bg[i % d]) * (y[i] + bv[i % d])).collect();
    let acc = upload(&dev, &x);
    let (bgd, bvd) = (upload(&dev, &bg), upload(&dev, &bv));
    let mut s = Vec::new();
    dev.gated_mix(&mut s, &acc, &yd, &bgd, &bvd, rows as u32, d as u32);
    assert_stage("gated_mix", &run(&dev, &s, &acc, rows * d), &want);
    // chan_fma and axpy
    let g = noise(d, 11, 1.0);
    let want: Vec<f32> = (0..rows * d).map(|i| y[i] + g[i % d] * x[i]).collect();
    let (acc, gd) = (upload(&dev, &y), upload(&dev, &g));
    let mut s = Vec::new();
    dev.chan_fma(&mut s, &xd, &gd, &acc, (rows * d) as u32, d as u32);
    assert_stage("chan_fma", &run(&dev, &s, &acc, rows * d), &want);
    let want: Vec<f32> = (0..rows * d).map(|i| y[i] + 0.37 * x[i]).collect();
    let acc = upload(&dev, &y);
    let mut s = Vec::new();
    dev.axpy(&mut s, 0.37, &xd, &acc, (rows * d) as u32);
    assert_stage("axpy", &run(&dev, &s, &acc, rows * d), &want);
    // mean_h and upsample2 on a [2, 4, 3, 5] map
    let (b, h, w, c) = (2usize, 4usize, 3usize, 5usize);
    let map = noise(b * h * w * c, 12, 1.0);
    let want: Vec<f32> = (0..b * w * c).map(|i| (0..h).map(|r| map[((i / (w * c) * h + r) * w + (i / c) % w) * c + i % c]).sum::<f32>() / h as f32).collect();
    let (md_, o) = (upload(&dev, &map), dev.gpu().storage((b * w * c) as u64));
    let mut s = Vec::new();
    dev.mean_h(&mut s, &md_, &o, b as u32, h as u32, w as u32, c as u32);
    assert_stage("mean_h", &run(&dev, &s, &o, want.len()), &want);
    let mut want = Vec::new();
    for i in 0..b {
        want.extend(mirage::vision::ops::upsample2(&map[i * h * w * c..(i + 1) * h * w * c], mirage::vision::ops::Map { h, w, c }).0);
    }
    let o = dev.gpu().storage(want.len() as u64);
    let mut s = Vec::new();
    dev.upsample2(&mut s, &md_, &o, b as u32, h as u32, w as u32, c as u32);
    assert_stage("upsample2", &run(&dev, &s, &o, want.len()), &want);
}

#[test]
fn global_response_norm_matches_the_host_block_tail() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("grn") else { return };
    let (batch, rows, d) = (2usize, 9usize, 12usize);
    let x = noise(batch * rows * d, 13, 1.0);
    let y = noise(batch * rows * d, 14, 1.0);
    let (gamma, beta) = (noise(d, 15, 0.5), noise(d, 16, 0.5));
    let m: Vec<f32> = (0..batch * rows).map(|r| ((r % rows) < 6 + r / rows) as u8 as f32).collect();
    let mut want = x.clone();
    for b in 0..batch {
        let mut energy = vec![0f32; d];
        for r in (0..rows).filter(|r| m[b * rows + r] > 0.0) {
            for c in 0..d {
                energy[c] += y[(b * rows + r) * d + c].powi(2);
            }
        }
        energy.iter_mut().for_each(|e| *e = (*e + 1e-6).sqrt());
        let mean = energy.iter().sum::<f32>() / d as f32;
        for r in (0..rows).filter(|r| m[b * rows + r] > 0.0) {
            for c in 0..d {
                want[(b * rows + r) * d + c] += y[(b * rows + r) * d + c] * (gamma[c] * energy[c] / (mean + 1e-6) + 1.0) + beta[c];
            }
        }
    }
    let acc = upload(&dev, &x);
    let (yd, md, gd, bd) = (upload(&dev, &y), upload(&dev, &m), upload(&dev, &gamma), upload(&dev, &beta));
    let mut s = Vec::new();
    dev.grn_residual(&mut s, &acc, &yd, &md, &gd, &bd, batch as u32, rows as u32, d as u32);
    assert_stage("grn", &run(&dev, &s, &acc, x.len()), &want);
}

#[test]
fn attention_and_rope_match_the_host_formulas() {
    use mirage::device::prims::AttnShape;
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("attention") else { return };
    let a = AttnShape { batch: 2, nq: 3, heads: 4, kvh: 2, hd: 8, vd: 6, nk: 7, visible: 6 };
    let (q, k, v) = (noise(2 * 3 * 4 * 8, 17, 1.0), noise(2 * 7 * 2 * 8, 18, 1.0), noise(2 * 7 * 2 * 6, 19, 1.0));
    let mask: Vec<f32> = (0..14).map(|i| (i % 7 != 2 && i % 7 != 4) as u8 as f32).collect();
    let mut want = vec![0f32; 2 * 3 * 4 * 6];
    for b in 0..2 {
        for t in 0..3 {
            for h in 0..4 {
                let kv = h / 2;
                let keys: Vec<usize> = (0..6).filter(|&s| mask[b * 7 + s] > 0.0).collect();
                let mut sc: Vec<f32> = keys.iter().map(|&s| (0..8).map(|i| q[((b * 3 + t) * 4 + h) * 8 + i] * k[((b * 7 + s) * 2 + kv) * 8 + i]).sum::<f32>() / 8f32.sqrt()).collect();
                model::hostmath::softmax(&mut sc);
                for i in 0..6 {
                    want[((b * 3 + t) * 4 + h) * 6 + i] = keys.iter().zip(&sc).map(|(&s, p)| p * v[((b * 7 + s) * 2 + kv) * 6 + i]).sum();
                }
            }
        }
    }
    let (qd, kd, vd, md) = (upload(&dev, &q), upload(&dev, &k), upload(&dev, &v), upload(&dev, &mask));
    let o = dev.gpu().storage(want.len() as u64);
    let mut s = Vec::new();
    dev.attention(&mut s, a, &qd, &kd, &vd, &md, &o);
    assert_stage("attention", &run(&dev, &s, &o, want.len()), &want);

    // 2D rope against the encoder's host rotation: build it from the tiny model's formula.
    let (batch, tokens, cells, grid, heads, hd) = (2usize, 7usize, 6usize, 3usize, 2usize, 8usize);
    let t = noise(batch * tokens * heads * hd, 20, 1.0);
    let mut want = t.clone();
    let (pairs, per_axis) = (hd / 2, hd / 4);
    for r in 0..batch * tokens {
        if r % tokens >= cells {
            continue;
        }
        let cell = r % tokens;
        for head in 0..heads {
            for j in 0..pairs {
                let pos = if j < per_axis { cell / grid } else { cell % grid } as f32;
                let angle = pos * 100.0f32.powf(-((j % per_axis) as f32) / per_axis as f32);
                let (sn, cs) = angle.sin_cos();
                let at = (r * heads + head) * hd + j;
                want[at] = t[at] * cs - t[at + pairs] * sn;
                want[at + pairs] = t[at + pairs] * cs + t[at] * sn;
            }
        }
    }
    let td = upload(&dev, &t);
    let mut s = Vec::new();
    dev.rope2d(&mut s, &td, batch as u32, tokens as u32, cells as u32, grid as u32, heads as u32, hd as u32, 100.0);
    assert_stage("rope2d", &run(&dev, &s, &td, t.len()), &want);
}
