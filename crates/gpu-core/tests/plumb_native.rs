// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native plumbing and SiLU kernels (`kernels_cuda`,
//! `cu/plumb_f32.cu`) that `gpu_core::native_upgrade` substitutes for
//! `concat_split`, `chan_place`, `concat2`, `silu` and `silu_bwd` on a CUDA
//! device.
//!
//! Swedish Embedded AB implements training kernels for convolutional networks.
//! If your team needs expertise in replacing a generated kernel with a faster
//! hand-written one without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Every one of them is a copy or an elementwise expression - nothing is
//! reduced, so nothing can be reassociated - and the bar is the RAW BITS of the
//! WGSL reference, dispatched on the same device from the same inputs under a
//! name no redirect knows. Shapes are YOLOv8n's (batch 8, 512 input) plus maps
//! whose plane is not a whole number of 16-byte vectors (the scalar path) and
//! flat lengths that are not a multiple of four.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[
    ("concat_split", kernels::CONCAT_SPLIT),
    ("chan_place", kernels::CHAN_PLACE),
    ("concat2", kernels::CONCAT2),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("concat_split_ref", kernels::CONCAT_SPLIT),
    ("chan_place_ref", kernels::CHAN_PLACE),
    ("concat2_ref", kernels::CONCAT2),
    ("silu_ref", kernels::SILU),
    ("silu_bwd_ref", kernels::SILU_BWD),
];
const SPLIT: usize = 0;
const PLACE: usize = 1;
const CONCAT2: usize = 2;
const SILU: usize = 3;
const SILU_BWD: usize = 4;
const REF: usize = 5;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && !std::env::var("BRAIN_NO_NATIVE_KERNELS").is_ok_and(|v| v != "0")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Values across many binades, both signs, zeros and large magnitudes - the
/// range an activation and its gradient actually see, and enough to make an
/// `expf` or a division that differs in one bit show.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|i| if i % 97 == 0 { 0.0 } else { r.signed() * 10f32.powi((r.next_u32() % 6) as i32 - 2) }).collect()
}

/// Run slot `k` and its reference `k + REF` from the same inputs into fresh
/// outputs of `out_len` words (pre-filled with a sentinel so an unwritten
/// element differs), and compare the raw bits.
fn same_bits(gpu: &Gpu, k: usize, inputs: &[&[f32]], params: &[u32], out_len: usize, threads: u32, what: &str) {
    let bufs: Vec<_> = inputs.iter().map(|v| gpu.storage_init("in", v)).collect();
    let sentinel = vec![f32::from_bits(0x7fc0_dead); out_len];
    let outs = [gpu.storage_init("out", &sentinel), gpu.storage_init("out_ref", &sentinel)];
    for (slot, out) in [k, k + REF].iter().zip(&outs) {
        let mut b: Vec<&gpu_core::DeviceBuffer> = bufs.iter().collect();
        b.push(out);
        gpu.submit(&[], &[gpu.step(*slot, &b, params, threads)]);
    }
    gpu.poll_wait();
    let (native, reference) = (gpu.read(&outs[0], out_len), gpu.read(&outs[1], out_len));
    assert!(bits(&native) == bits(&reference), "{what}: native differs from the reference");
}

#[test]
fn the_native_plumbing_is_bit_identical_to_the_reference() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(SILU, &[1024]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(SPLIT, &[2, 8, 4, 2, 4, 4]), Some("concat_split_f32"));
    assert_eq!(gpu.native_kernel_for(PLACE, &[2, 8, 4, 2, 4, 4]), Some("chan_place_f32"));
    assert_eq!(gpu.native_kernel_for(CONCAT2, &[2, 3, 5, 4, 4]), Some("concat2_f32"));
    assert_eq!(gpu.native_kernel_for(SILU_BWD, &[1024]), Some("silu_bwd_f32"));
    assert_eq!(gpu.native_kernel_for(SPLIT, &[2, 8, 4, 5, 4, 4]), None, "a window past the wider map is not served");

    // (N, Ctot, Csrc, c_off, H, W): the C2f splits and placements of YOLOv8n,
    // then odd planes and offsets.
    for (i, &(n, ctot, csrc, off, h, w)) in
        [(8u32, 32u32, 16u32, 16u32, 128u32, 128u32), (8, 64, 32, 0, 64, 64), (8, 96, 32, 64, 64, 64), (8, 384, 128, 256, 16, 16), (3, 7, 3, 2, 5, 7), (1, 5, 1, 4, 1, 1)]
            .iter()
            .enumerate()
    {
        let wide = values((n * ctot * h * w) as usize, 10 + i as u64);
        let narrow = values((n * csrc * h * w) as usize, 20 + i as u64);
        let p = [n, ctot, csrc, off, h, w];
        let threads = n * csrc * h * w;
        same_bits(&gpu, SPLIT, &[&wide], &p, (n * csrc * h * w) as usize, threads, &format!("concat_split {p:?}"));
        // chan_place writes only its window: the rest of `dst` keeps the sentinel
        // on both sides, which the comparison covers too.
        same_bits(&gpu, PLACE, &[&narrow], &p, (n * ctot * h * w) as usize, threads, &format!("chan_place {p:?}"));
    }
    for (i, &(n, ca, cb, h, w)) in [(8u32, 128u32, 64u32, 32u32, 32u32), (8, 64, 128, 16, 16), (8, 256, 128, 16, 16), (2, 3, 5, 7, 9)].iter().enumerate() {
        let a = values((n * ca * h * w) as usize, 30 + i as u64);
        let b = values((n * cb * h * w) as usize, 40 + i as u64);
        let total = n * (ca + cb) * h * w;
        same_bits(&gpu, CONCAT2, &[&a, &b], &[n, ca, cb, h, w], total as usize, total, &format!("concat2 {:?}", (n, ca, cb, h, w)));
    }
    for (i, &total) in [8u32 * 16 * 256 * 256, 8 * 64 * 32 * 32, 12345, 7, 1].iter().enumerate() {
        let x = values(total as usize, 50 + i as u64);
        let dy = values(total as usize, 60 + i as u64);
        same_bits(&gpu, SILU, &[&x], &[total], total as usize, total, &format!("silu {total}"));
        same_bits(&gpu, SILU_BWD, &[&x, &dy], &[total], total as usize, total, &format!("silu_bwd {total}"));
    }
}
