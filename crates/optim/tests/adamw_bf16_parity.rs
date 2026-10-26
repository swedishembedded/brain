// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in half-precision optimiser state
// kernels for CUDA and portable GPUs then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The fused device AdamW with bf16 moments (`adamw_bf16.wgsl`, run here on the CPU backend's JIT, and on
//! CUDA/Vulkan through the same source) follows the host reference: the streamed optimiser over the same bf16
//! codec. Weights agree to float noise, the moments are the bf16 values the host holds (stochastic rounding
//! draws the same bits), the device buffers are the physical size the policy promised, and the moments read
//! back and restore exactly.

use std::collections::HashMap;

use gpu_core::Gpu;
use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::{Adam, Optim};
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, PrecisionPolicy, StateFormat, ValueDtype};
use paramstore::{ParamStore, Role};

const KERNELS: &[(&str, &str)] = &[
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ("adamw_bf16", kernels::ADAMW_BF16),
];

fn bf16(sr: bool) -> ParamPrecision {
    ParamPrecision { value: ValueDtype::F32, grad: GradDtype::F32, state: StateFormat::Bf16, master: MasterPolicy::None, stochastic_rounding: sr }
}

fn run(gpu: &Gpu, sr: bool, clip: Option<f32>, tol: f32) {
    let n = 1001usize; // odd: the last word holds one element
    let w0: Vec<f32> = (0..n).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
    let init: HashMap<String, Vec<f32>> = [("w".to_string(), w0.clone())].into_iter().collect();
    // Device and host reference share one rounding seed so their streams agree.
    let seed: u64 = std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 0u8);
    let adam = Adam { beta1: 0.9, beta2: 0.99, eps: 1e-6 };
    let (lr, wd, scale) = (1e-2f32, 0.01f32, 2.0f32);

    let policy = PrecisionPolicy::uniform(bf16(sr));
    let ps = ParamStore::new_with_policy_src(gpu, vec![("w".to_string(), n, Role::Trainable)], &init, &policy);
    let opt = Optim::new(0, 1, 2, 3, 4);
    opt.set_rounding_seed(seed);

    let rule = Rule::AdamW { adam, wd };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision: bf16(sr), row_len: None }];
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

    let (md, vd) = ps.read_moments(gpu, "w").expect("bf16 moments read back");
    let (mut mh, mut vh) = (vec![0f32; n], vec![0f32; n]);
    for (c, (a, b)) in mh.chunks_mut(256).zip(vh.chunks_mut(256)).enumerate() {
        host.state_mut().read_tile(0, 0, c * 256, a).unwrap();
        host.state_mut().read_tile(0, 1, c * 256, b).unwrap();
    }
    for (name, d, h) in [("m", &md, &mh), ("v", &vd, &vh)] {
        let scale = h.iter().fold(1e-30f32, |m, x| m.max(x.abs()));
        let worst = d.iter().zip(h).map(|(a, b)| (a - b).abs() / scale).fold(0f32, f32::max);
        println!("  {name}: worst moment diff / max {worst:e}");
        assert!(worst < 0.02, "{name} moments differ by {worst} of the maximum");
    }

    // read-back then restore is the identity
    let before = gpu.read(&ps.adam_m["w"], n.div_ceil(2));
    ps.write_moments(gpu, "w", &md, &vd).unwrap();
    assert_eq!(gpu.read(&ps.adam_m["w"], n.div_ceil(2)), before);

    // the policy's byte accounting is the physical size: 2 B/elem, not 4
    let moments = ps.training_state_bytes(2).moments;
    assert_eq!(moments, 2 * 2 * n as u64);
}

#[test]
fn device_bf16_adamw_follows_the_host_reference_with_stochastic_rounding() {
    run(&Gpu::new_cpu(KERNELS), true, None, 5e-3);
}

#[test]
fn device_bf16_adamw_follows_the_host_reference_with_nearest_rounding_and_clip() {
    run(&Gpu::new_cpu(KERNELS), false, Some(0.5), 5e-3);
}

/// The same gate on the ambient device (CUDA through the card helper): fused multiply-adds loosen the weights.
#[test]
fn device_bf16_adamw_on_the_ambient_device_follows_the_host_reference() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    run(&gpu_core::testgpu::dev(KERNELS), true, None, 2e-2);
}
