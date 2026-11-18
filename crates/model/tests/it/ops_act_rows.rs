// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A row window of a quantized activation ([`Ops::act_rows`]) multiplies exactly as an activation quantized from
//! those rows alone: one quantization of a slab can feed linears that each read their own rows of it.
//!
//! Swedish Embedded AB implements quantized inference engines that quantize an activation once for every product
//! reading it for its clients. If your team needs expertise in low-precision linear algebra on mixed-stream or
//! batched activations, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! Runs on the suite's device: the packed-int8 products where it has them, the float-read tile decode where it has
//! not (the CPU JIT). Both read rows by absolute index, which is what a window has to preserve.

use data::rng::Lcg;
use model::ops::{Ops, Weight};

fn kernels() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut list = model::ops::two_level_kernel_list().to_vec();
        model::ops::extend_kernel_list(&mut list, model::ops::frozen_base_kernels());
        list
    })
}

#[test]
fn a_row_window_of_a_quantized_slab_multiplies_as_its_rows_quantized_alone() {
    let (rows, n, k) = (192u32, 96u32, 128u32);
    let ops = Ops::new(gpu_core::testgpu::dev(kernels()).share()).expect("Ops::new");
    let g = ops.gpu();
    let mut rng = Lcg::new(11);
    let mut fill = |len: u32| -> Vec<f32> { (0..len).map(|_| (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect() };
    let raw = fill(n * k);
    let x = fill(rows * k);
    let (codes, scales) = model::int8::quantize_weight(&raw, n as usize, k as usize);
    let wb = g.storage(codes.len() as u64);
    g.write(&wb, &codes);
    let weight = Weight::from_packed_int8(wb, g.storage_init("scales", &scales), n as usize, k as usize);
    let xb = g.storage_init("x", &x);

    // Rows [64, 192) read through a window of the whole slab, and through an activation of those rows alone; the
    // one-level and the two-level forms both.
    for two_level in [false, true] {
        let act = |s: &mut Vec<gpu_core::Step>, r0: u32, m: u32| if two_level { ops.act_two_level(s, &xb, r0, m, k) } else { ops.act(s, &xb, r0, m, k) };
        let (via_window, alone) = (g.storage(((rows - 64) * n) as u64), g.storage(((rows - 64) * n) as u64));
        let mut s = Vec::new();
        let slab = act(&mut s, 0, rows);
        ops.matmul(&mut s, &weight, &ops.act_rows(&slab, 64, rows - 64), &via_window, 0);
        let rows_alone = act(&mut s, 64, rows - 64);
        ops.matmul(&mut s, &weight, &rows_alone, &alone, 0);
        g.submit(&[], &s);
        g.poll_wait();
        let (got, want) = (g.read(&via_window, ((rows - 64) * n) as usize), g.read(&alone, ((rows - 64) * n) as usize));
        assert!(want.iter().any(|v| *v != 0.0), "the reference product is all zero");
        assert_eq!(got, want, "a window of the slab multiplied differently from its rows quantized alone (two-level {two_level})");
    }
}
