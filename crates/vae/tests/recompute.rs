// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of convolutional generative models for its clients. If your
// team needs expertise in trading activation memory for recomputation in a diffusion or autoencoder fine-tune, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Activation recompute on the shared conv-block tape (`vae::blocks::Builder::set_activations`): a block's internal
//! activations live in buffers shared with later blocks' internals, and the reverse replays the block's forward just
//! before its adjoint. The gradients are those of the graph that keeps every activation, bit for bit, at fewer
//! allocated forward bytes.

use vae::blocks::grad::BwdIds;
use vae::blocks::{Activations, BlockNames, Builder, Tensors};

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
const DEPTH: usize = 4;

/// The weights of `DEPTH` resnet blocks `b0..` (the second widening to `2C` through a shortcut conv),
/// deterministic small values.
fn tensors() -> Tensors {
    let mut t = Tensors::new();
    let vals = |n: usize, s: usize| (0..n).map(|i| ((i * 7 + s * 13) % 17) as f32 / 17.0 * 0.2 - 0.1).collect::<Vec<f32>>();
    let mut cin = C as usize;
    for b in 0..DEPTH {
        let cout = if b == 1 { 2 * C as usize } else { cin };
        let mut seed = b * 10;
        let mut put = |name: String, shape: Vec<usize>| {
            seed += 1;
            let n = shape.iter().product();
            t.insert(name, (shape, vals(n, seed)));
        };
        put(format!("b{b}.norm1.weight"), vec![cin]);
        put(format!("b{b}.norm1.bias"), vec![cin]);
        put(format!("b{b}.conv1.weight"), vec![cout, cin, 3, 3]);
        put(format!("b{b}.conv1.bias"), vec![cout]);
        put(format!("b{b}.norm2.weight"), vec![cout]);
        put(format!("b{b}.norm2.bias"), vec![cout]);
        put(format!("b{b}.conv2.weight"), vec![cout, cout, 3, 3]);
        put(format!("b{b}.conv2.bias"), vec![cout]);
        if cout != cin {
            put(format!("b{b}.conv_shortcut.weight"), vec![cout, cin, 1, 1]);
            put(format!("b{b}.conv_shortcut.bias"), vec![cout]);
        }
        cin = cout;
    }
    t
}

/// The input gradient and every parameter gradient (bits) of one step, and the forward's allocated bytes.
type Measured = (Vec<u32>, Vec<(String, Vec<u32>)>, u64);

/// One step of the stack, its block internals kept or recomputed.
fn step(recompute: bool) -> Measured {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let t = tensors();
    let mut b = Builder::new(&gpu, &t, 1e-5, 4, BlockNames::diffusers(), false);
    b.set_train(true);
    b.set_activations(if recompute { Activations::Recompute } else { Activations::Keep });
    let n_in = (C * HW * HW) as usize;
    let x = gpu.storage_init("x", &(0..n_in).map(|i| (i % 11) as f32 / 11.0 - 0.5).collect::<Vec<f32>>());
    let (mut y, mut c) = (x.clone(), C);
    for i in 0..DEPTH {
        let cout = if i == 1 { 2 * C } else { c };
        y = b.resnet(&format!("b{i}"), c, cout, HW, HW, &y);
        c = cout;
    }
    let forward_bytes = b.allocated_bytes();
    let trace = b.trace();
    let (steps, _) = b.finish();
    let grads = trace.alloc_grads(&gpu);
    let n_out = (c * HW * HW) as usize;
    let d_out = gpu.storage_init("d_out", &(0..n_out).map(|i| ((i % 5) as f32 - 2.0) / n_out as f32).collect::<Vec<f32>>());
    let reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, &y, &d_out);
    gpu.submit(&grads.all(), &steps);
    gpu.submit(&[], &reverse.steps);
    gpu.poll_wait();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    let dx = bits(gpu.read(reverse.d(&x).expect("the reverse reaches the input"), n_in));
    let params = trace.params().iter().map(|(name, len)| (name.clone(), bits(gpu.read(grads.g(name), *len as usize)))).collect();
    (dx, params, forward_bytes)
}

#[test]
fn recomputing_blocks_gives_the_kept_gradients_at_fewer_forward_bytes() {
    let (dx_keep, g_keep, keep_bytes) = step(false);
    let (dx_rec, g_rec, rec_bytes) = step(true);
    assert!(dx_keep.iter().any(|b| f32::from_bits(*b) != 0.0), "the input received no gradient");
    assert!(dx_rec == dx_keep, "the input gradient differs recomputing");
    assert_eq!(g_keep.len(), g_rec.len());
    for ((name, a), (_, b)) in g_keep.iter().zip(&g_rec) {
        assert!(a == b, "{name}: gradient differs recomputing");
    }
    assert!(rec_bytes < keep_bytes, "recomputing allocates {rec_bytes} forward bytes, keeping {keep_bytes}");
}
