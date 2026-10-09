// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native row softmax (`cu/softmax_rows_f32.cu`), which
//! `gpu_core::native_upgrade` substitutes for the WGSL `softmax_rows` on a
//! CUDA device, one register bucket per column range.
//!
//! Swedish Embedded AB implements attention kernels on the memory roofline for
//! its clients. If your team needs expertise in bandwidth-bound passes that
//! read their data once, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-gpu-core --test softmax_rows_native -- --device gpu1 --backend cuda
//! ```
//!
//! The native kernel keeps the WGSL kernel's arithmetic - 64 lanes per row,
//! each summing its strided elements in ascending order, the 64 partials folded
//! in lane order, `expf` and one IEEE division - and only stops re-reading the
//! row from memory, so it is held to the WGSL tier's RAW BITS: at every bucket
//! edge, with rows of the masked value an attention row carries, and with the
//! binding offset by a word.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("softmax_rows", kernels::SOFTMAX_ROWS), ("softmax_rows_ref", kernels::SOFTMAX_ROWS)];
const K_SM: usize = 0;
const K_REF: usize = 1;
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn check(gpu: &Gpu, rows: u32, cols: u32, lead: u64) {
    let n = (rows * cols) as usize;
    let mut r = data::rng::Lcg::new(u64::from(rows * 7919 + cols));
    let x: Vec<u32> = (0..n)
        .map(|i| {
            let u = (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            // Logits over several binades, and masked entries as a causal or
            // key-masked attention row writes them.
            if i % 13 == 5 { -3.4e38f32 } else { (u - 0.5) * 24.0 }.to_bits()
        })
        .collect();
    let buf = |words: &[u32]| {
        let b = gpu.storage(words.len() as u64 + lead + 4);
        gpu.write(&b, &vec![SENTINEL; words.len() + lead as usize + 4]);
        gpu.write_at(&b, lead, words);
        b
    };
    let xb = buf(&x);
    let (a, b) = (buf(&vec![SENTINEL; n]), buf(&vec![SENTINEL; n]));
    let range = (lead, n as u64);
    gpu.submit(
        &[],
        &[
            gpu.dispatch_sliced(K_SM, &[&xb, &a], &[range, range], &[rows, cols], Dispatch::Workgroups(rows)),
            gpu.dispatch_sliced(K_REF, &[&xb, &b], &[range, range], &[rows, cols], Dispatch::Workgroups(rows)),
        ],
    );
    gpu.poll_wait();
    let len = n + lead as usize + 4;
    let bits = |v: Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    let (ga, gb) = (bits(gpu.read(&a, len)), bits(gpu.read(&b, len)));
    let first = ga.iter().zip(&gb).position(|(p, q)| p != q);
    assert!(first.is_none(), "native softmax_rows differs from the WGSL tier at {first:?}: rows {rows} cols {cols} lead {lead}");
}

fn the_native_kernel_is_selected_by_column_count() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_SM, &[8, 512]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (cols, want) in [(1u32, "softmax_rows_f32_k8"), (512, "softmax_rows_f32_k8"), (513, "softmax_rows_f32_k16"), (2048, "softmax_rows_f32_k32"), (2560, "softmax_rows_f32_k64"), (4096, "softmax_rows_f32_k64")] {
        assert_eq!(gpu.native_kernel_for(K_SM, &[61440, cols]), Some(want), "cols {cols}");
    }
    assert_eq!(gpu.native_kernel_for(K_SM, &[8, 4097]), None, "a row past the largest bucket keeps the WGSL tier");
    assert_eq!(gpu.native_kernel_for(K_SM, &[0, 512]), None, "no rows");
    assert_eq!(gpu.native_kernel_for(K_REF, &[8, 512]), None);
}

fn the_native_kernel_is_bit_identical_to_the_wgsl_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for cols in [1u32, 2, 63, 64, 65, 511, 512, 513, 1024, 1025, 2047, 2048, 2049, 2560, 4095, 4096] {
        for (rows, lead) in [(1u32, 0u64), (7, 1), (64, 0)] {
            check(&gpu, rows, cols, lead);
        }
    }
    // The trainer's attention: 24 heads x 2560 rows of 2560 scores.
    check(&gpu, 24 * 2560, 2560, 0);
}

gpu_core::card_tests!(the_native_kernel_is_selected_by_column_count, the_native_kernel_is_bit_identical_to_the_wgsl_tier);
