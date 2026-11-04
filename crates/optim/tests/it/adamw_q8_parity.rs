// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in 8-bit optimiser kernels for CUDA and
// portable GPUs then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The fused device AdamW with blockwise-int8 moments (`adamw_q8.wgsl`, run
//! here on the CPU backend's JIT, and on CUDA/Vulkan through the same source)
//! must follow the host reference: the streamed optimiser over the same codec.
//! Weights agree to float noise, decoded moments to within one 8-bit code, and
//! the device buffers are the physical size the policy promised.

use std::collections::HashMap;

use gpu_core::Gpu;
use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::{Adam, Optim};
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, PrecisionPolicy, StateFormat, ValueDtype};
use paramstore::{ParamStore, Role};

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ("adamw_q8", kernels::ADAMW_Q8),
    ]
}

fn int8(sr: bool) -> ParamPrecision {
    ParamPrecision {
        value: ValueDtype::F32,
        grad: GradDtype::F32,
        state: StateFormat::Int8Block { block: 64 },
        master: MasterPolicy::None,
        stochastic_rounding: sr,
    }
}

fn run(gpu: &Gpu, sr: bool, clip: Option<f32>, tol: f32) {
    let n = 1000usize; // not a multiple of the block: exercises the tail block
    let w0: Vec<f32> = (0..n).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
    let init: HashMap<String, Vec<f32>> = [("w".to_string(), w0.clone())].into_iter().collect();
    // Device and host reference share one rounding seed so their streams agree.
    let seed: u64 = std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 0u8);
    let adam = Adam { beta1: 0.9, beta2: 0.99, eps: 1e-6 };
    let (lr, wd, scale) = (1e-2f32, 0.01f32, 2.0f32);

    // device
    let policy = PrecisionPolicy::uniform(int8(sr));
    let ps = ParamStore::new_with_policy_src(gpu, vec![("w".to_string(), n, Role::Trainable)], &init, &policy);
    let opt = Optim::new(0, 1, 2, 3, 4);
    opt.set_rounding_seed(seed);

    // host reference, same codec and rounding keys (tensor index 0)
    let rule = Rule::AdamW { adam, wd };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision: int8(sr), row_len: None }];
    let layout = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), 256).unwrap();
    let mut host = StreamedOptimizer::new(rule, specs, st, seed, 256).unwrap();
    let vl = tensor_layout(&[n], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 256).unwrap();
    let grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 256).unwrap();
    for (c, ch) in w0.chunks(256).enumerate() {
        values.write_tile(0, 0, c * 256, ch, Rounding::NEAREST).unwrap();
    }
    let mut hp = HostParams { values, grads };

    for t in 1..=8u32 {
        let g: Vec<f32> = (0..n).map(|i| (((i as u32 * 3 + t) % 17) as f32 / 17.0 - 0.5) * 0.2 * (1 + (i % 5)) as f32).collect();
        gpu.write(ps.g("w"), bytemuck::cast_slice(&g));
        for (c, ch) in g.chunks(256).enumerate() {
            hp.grads.write_tile(0, 0, c * 256, ch, Rounding::NEAREST).unwrap();
        }
        opt.step(gpu, &ps, t, lr, wd, adam, clip, scale);
        host.step(&mut hp, lr, clip, scale).unwrap();
    }

    let wd_dev = gpu.read(ps.w("w"), n);
    let mut wh = vec![0f32; n];
    for (c, ch) in wh.chunks_mut(256).enumerate() {
        hp.values.read_tile(0, 0, c * 256, ch).unwrap();
    }
    let wscale = wh.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    let wdiff = wd_dev.iter().zip(&wh).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("sr={sr} clip={clip:?}: max weight diff {wdiff:e} (scale {wscale})");
    assert!(wdiff <= tol * wscale, "weights diverge: {wdiff}");

    let (md, vd) = optim::read_moments_q8(gpu, &ps, "w").expect("int8 moments");
    let mut mh = vec![0f32; n];
    let mut vh = vec![0f32; n];
    for (c, (a, b)) in mh.chunks_mut(256).zip(vh.chunks_mut(256)).enumerate() {
        host.state_mut().read_tile(0, 0, c * 256, a).unwrap();
        host.state_mut().read_tile(0, 1, c * 256, b).unwrap();
    }
    for (name, d, h) in [("m", &md, &mh), ("v", &vd, &vh)] {
        let blk_max = |i: usize| h[i / 64 * 64..((i / 64 + 1) * 64).min(n)].iter().fold(1e-30f32, |m, x| m.max(x.abs()));
        let worst = (0..n).map(|i| (d[i] - h[i]).abs() / blk_max(i)).fold(0f32, f32::max);
        println!("  {name}: worst decoded diff / block max {worst:e}");
        assert!(worst < 20.0 * tol.max(5e-3), "{name} moments differ by {worst} of the block scale");
    }

    // the policy's byte accounting is the physical size: 1 B/elem plus scales, not 4 B/elem
    let moments = ps.training_state_bytes(2).moments;
    assert_eq!(moments, 2 * (n + 4 * n.div_ceil(64)) as u64);
    assert!(moments < 2 * 4 * n as u64 / 3, "int8 moments must cost well under the fp32 8 B/elem");
}

#[test]
fn device_int8_adamw_follows_the_host_reference_with_stochastic_rounding() {
    run(&Gpu::new_cpu(&kernels()), true, None, 5e-3);
}

#[test]
fn device_int8_adamw_follows_the_host_reference_with_nearest_rounding_and_clip() {
    run(&Gpu::new_cpu(&kernels()), false, Some(0.5), 5e-3);
}

/// The same gate on the ambient device (`BRAIN_DEVICE`/`BRAIN_BACKEND`, e.g.
/// CUDA through the card helper). The CUDA backend compiles the same WGSL, so
/// this is the fused-kernel check for the GPU; fused multiply-adds make the
/// weights agree to a looser tolerance than the bit-exact CPU run.
#[test]
fn device_int8_adamw_on_the_ambient_device_follows_the_host_reference() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    static KERNELS: &[(&str, &str)] = &[
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ("adamw_q8", kernels::ADAMW_Q8),
        ("adamw_q8_wg", kernels::ADAMW_Q8_WG),
    ];
    run(&gpu_core::testgpu::dev(KERNELS), true, Some(0.5), 2e-2);
}
