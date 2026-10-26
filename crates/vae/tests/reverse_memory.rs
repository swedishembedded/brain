// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of convolutional generative models for its clients. If your
// team needs expertise in fitting a diffusion or autoencoder fine-tune into a fixed device budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the reverse of the shared conv-block tape (`vae::blocks::grad::Trace::backward`) allocates. An activation's
//! gradient is complete once the walk has passed every consumer of the activation, and dead once it has passed the
//! op that produced it, so the reverse holds gradients for the live frontier of the walk: a deeper stack of blocks
//! adds no activation-gradient memory. One test in its own binary, so nothing else allocates while the device bytes
//! are compared.

use vae::blocks::grad::BwdIds;
use vae::blocks::{BlockNames, Builder, Tensors};

const BWD_BASE: usize = vae::blocks::NEXT_SLOT;
const N_KERNELS: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

static KERNELS: [(&str, &str); N_KERNELS] = kernel_set();
const fn kernel_set() -> [(&'static str, &'static str); N_KERNELS] {
    let mut k = vae::blocks::kernels_with::<N_KERNELS>();
    let mut j = 0;
    while j < vae::blocks::BWD_KERNELS.len() {
        k[BWD_BASE + j] = vae::blocks::BWD_KERNELS[j];
        j += 1;
    }
    k
}

const C: u32 = 16;
const HW: u32 = 8;

/// The weights of `depth` same-width resnet blocks `b0..`, deterministic small values.
fn tensors(depth: usize) -> Tensors {
    let mut t = Tensors::new();
    let vals = |n: usize, s: usize| (0..n).map(|i| ((i * 7 + s * 13) % 17) as f32 / 17.0 * 0.2 - 0.1).collect::<Vec<f32>>();
    let c = C as usize;
    for b in 0..depth {
        for norm in ["norm1", "norm2"] {
            t.insert(format!("b{b}.{norm}.weight"), (vec![c], vec![1.0; c]));
            t.insert(format!("b{b}.{norm}.bias"), (vec![c], vec![0.0; c]));
        }
        for (i, conv) in ["conv1", "conv2"].into_iter().enumerate() {
            t.insert(format!("b{b}.{conv}.weight"), (vec![c, c, 3, 3], vals(c * c * 9, b * 2 + i)));
            t.insert(format!("b{b}.{conv}.bias"), (vec![c], vals(c, b * 2 + i + 1)));
        }
    }
    t
}

/// Device bytes the reverse of a `depth`-block stack allocates while it is recorded, and the input gradient it
/// computes.
fn reverse_bytes(depth: usize) -> (u64, Vec<f32>) {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let t = tensors(depth);
    let mut b = Builder::new(&gpu, &t, 1e-5, 4, BlockNames::diffusers(), false);
    b.set_train(true);
    let n = (C * HW * HW) as usize;
    let x = gpu.storage_init("x", &(0..n).map(|i| (i % 11) as f32 / 11.0 - 0.5).collect::<Vec<f32>>());
    let mut y = x.clone();
    for i in 0..depth {
        y = b.resnet(&format!("b{i}"), C, C, HW, HW, &y);
    }
    let trace = b.trace();
    let (steps, _) = b.finish();
    let grads = trace.alloc_grads(&gpu);
    let d_out = gpu.storage_init("d_out", &vec![1.0f32 / n as f32; n]);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, &y, &d_out);
    let bytes = gpu.memory_usage().unwrap().live - before;
    gpu.submit(&[], &steps);
    gpu.submit(&[], &reverse.steps);
    gpu.poll_wait();
    let dx = gpu.read(reverse.d(&x).expect("the reverse reaches the input"), n);
    (bytes, dx)
}

#[test]
fn a_deeper_stack_adds_no_activation_gradient_memory() {
    let (shallow, dx_shallow) = reverse_bytes(2);
    let (deep, dx_deep) = reverse_bytes(8);
    assert!(dx_shallow.iter().chain(&dx_deep).any(|v| *v != 0.0), "the input received no gradient");
    assert!(dx_deep.iter().all(|v| v.is_finite()));
    // One activation is C*HW*HW floats; six more blocks may add scratch of a few, never a gradient per activation.
    let activation = (C * HW * HW) as u64 * 4;
    assert!(deep <= shallow + 2 * activation, "8 blocks' reverse allocates {deep} bytes, 2 blocks' {shallow} (one activation is {activation})");
}
