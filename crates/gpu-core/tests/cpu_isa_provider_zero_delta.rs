// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `kernel-performance.md` M8.10/M8.11: `gpu_core::provider::cpu_isa::
//! CpuIsaProvider` proof tests.
//!
//! A dedicated integration-test FILE, not an inline `#[cfg(test)]` module in
//! `src/provider/cpu_isa.rs`, for one reason: these tests need the CPU
//! backend SPECIFICALLY (`register_native`/`step_native` exist only on
//! `backend-cpu` today), and there is no way to build a `Gpu` that bypasses
//! the ambient `--device`/`BRAIN_DEVICE` backend-class selection
//! (`Gpu::wrap`/`wrap_on` are private to `gpu_core`'s internal
//! `native_facade` module - not visible even from a sibling module inside
//! the same crate, confirmed by trying). `gpu_core::set_default_backend`
//! is the one lever that works, and it is process-global - safe to call here
//! ONLY because each `tests/*.rs` file is its own separate binary/process
//! under `cargo test`, so this can never race a DIFFERENT test file's
//! process the way it would if this lived inside the lib's own unit tests
//! (which share one process with every other `#[cfg(test)]` module in the
//! crate).

use std::sync::Arc;

use backend_api::select::{self, CachedSelector, Dtype, KernelSelector, KernelVariant};
use backend_api::DType;
use gpu_core::provider::cpu_isa::CpuIsaProvider;
use gpu_core::provider::{LowerCtx, OpRequest, Operand, Pass, ProviderRegistry, Role};

fn force_cpu() {
    gpu_core::set_default_backend(gpu_core::Backend::Cpu);
}

static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL)];

fn bind_f32(v: KernelVariant) -> (usize, &'static str) {
    match v {
        KernelVariant::Reference => (0, "matmul"),
        other => panic!("test bind: no kernel registered for {other:?}"),
    }
}

/// M8.10's own zero-delta proof: the SAME `(m,n,k)` F32 matmul, dispatched
/// two different ways on the SAME real CPU device -
///  1. through `WgslProvider` (an empty registry - today's unmodified path,
///     which itself bottoms out in `CpuBackend::dispatch`'s hidden
///     `f.matmul` if-ladder calling `fast_ops::matmul_abt`);
///  2. through `CpuIsaProvider` preferred ahead of it (the new ABI path -
///     `register_native`/`step_native` -> the SAME `fast_ops::matmul_abt`,
///     reached without ever going through a kernel-name match at all).
///
/// Both produce BIT-IDENTICAL output - proving the hoist changed nothing
/// about what runs, only how it is reached.
#[test]
fn cpu_isa_f32_matmul_is_bit_identical_to_the_hidden_fastpath() {
    force_cpu();
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (m, n, k) = (37u32, 53u32, 71u32);
    let mut seed = 99u32;
    let mut lcg = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    let a_data: Vec<f32> = (0..(m * k)).map(|_| lcg()).collect();
    let b_data: Vec<f32> = (0..(n * k)).map(|_| lcg()).collect();

    let run = |use_cpu_isa: bool| -> Vec<f32> {
        let x = gpu.storage_init("x", &a_data);
        let w = gpu.storage_init("w", &b_data);
        let y = gpu.storage((m * n) as u64);
        let operands = [
            Operand { role: Role::Act, buf: &x, range: (0, (m * k) as u64), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: &w, range: (0, (n * k) as u64), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &y, range: (0, (m * n) as u64), dtype: DType::F32 },
        ];
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: select::OpShape { m, n, k, dtype: Dtype::F32 },
            pass: Pass::Forward,
            operands: &operands,
            attrs: &[m, k, n],
            bind: &bind_f32,
        };
        let selector: Arc<dyn KernelSelector> = Arc::new(CachedSelector::new(select::AlwaysReference));
        let registry = if use_cpu_isa {
            ProviderRegistry::reference(selector).prefer(Arc::new(CpuIsaProvider::new()))
        } else {
            ProviderRegistry::reference(selector)
        };
        let mut steps = Vec::new();
        let caps = gpu.caps();
        assert_eq!(caps.class, backend_api::DeviceClass::Cpu, "this test must actually run on the CPU backend");
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        let lowered = registry.dispatch(&mut ctx, &req);
        if use_cpu_isa {
            assert_eq!(lowered.kernels, vec!["cpu_matmul_abt"], "expected the CpuIsaProvider's native path to have been chosen");
        } else {
            assert_eq!(lowered.kernels, vec!["matmul"], "expected the WGSL reference provider's path");
        }
        gpu.submit(&[], &steps);
        gpu.read(&y, (m * n) as usize)
    };

    let via_wgsl = run(false);
    let via_cpu_isa = run(true);
    assert_eq!(via_wgsl, via_cpu_isa, "M8.10 must be zero-delta: identical output, reached two different ways");
}

/// M8.11's own correctness proof at the ABI level (the numeric proof itself -
/// exact-match against the WGSL formula - already lives in
/// `backend_cpu::fast_ops::tests::avx2_int8_gemm_matches_scalar_reference`;
/// this only proves the PLUMBING reaches that function through the ABI with
/// the right operand order).
#[test]
fn cpu_isa_i8_matmul_reaches_the_native_avx2_gemm() {
    force_cpu();
    let gpu = gpu_core::testgpu::dev(&[]);
    let caps = gpu.caps();
    assert_eq!(caps.class, backend_api::DeviceClass::Cpu, "this test must actually run on the CPU backend");

    // out[m,n] = sx[m] * Σ_g dot_g(m,n) * sw[n,g]; one group (kg=8 words).
    // This test only needs the ABI plumbing right (operand order, native id
    // resolution), not interesting numerics - constructed directly as raw
    // packed `u32` words (bit-reinterpreted as `f32` for `storage_init`, the
    // same convention `CpuBuffer` itself uses: every element is 4 opaque
    // bytes).
    let (m, n, kg) = (2u32, 2u32, 8u32);
    let xq_words: Vec<u32> = {
        let mut v = vec![0u32; (m * kg) as usize];
        v[0] = 1; // row 0, word 0, lane 0 = int8 value 1
        v[kg as usize] = 2; // row 1, word 0, lane 0 = int8 value 2
        v
    };
    let wq_words: Vec<u32> = {
        let mut v = vec![0u32; (n * kg) as usize];
        v[0] = 3; // col 0, word 0, lane 0 = int8 value 3
        v[kg as usize] = 4; // col 1, word 0, lane 0 = int8 value 4
        v
    };
    let xq = gpu.storage_init("xq", &xq_words.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>());
    let wq = gpu.storage_init("wq", &wq_words.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>());
    let sx = gpu.storage_init("sx", &[1.0, 1.0]);
    let sw = gpu.storage_init("sw", &[1.0, 1.0]);
    let out = gpu.storage((m * n) as u64);

    let operands = [
        Operand { role: Role::Act, buf: &xq, range: (0, (m * kg) as u64), dtype: DType::I8 },
        Operand { role: Role::Weight, buf: &wq, range: (0, (n * kg) as u64), dtype: DType::I8 },
        Operand { role: Role::ActScale, buf: &sx, range: (0, m as u64), dtype: DType::F32 },
        Operand { role: Role::WeightScale, buf: &sw, range: (0, n as u64), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &out, range: (0, (m * n) as u64), dtype: DType::F32 },
    ];
    let req = OpRequest {
        op: select::Op::MatMul,
        shape: select::OpShape { m, n, k: kg, dtype: Dtype::I8 },
        pass: Pass::Forward,
        operands: &operands,
        attrs: &[m, kg, n],
        bind: &bind_f32, // unused by CpuIsaProvider's I8 arm
    };
    let selector: Arc<dyn KernelSelector> = Arc::new(CachedSelector::new(select::AlwaysReference));
    let registry = ProviderRegistry::reference(selector).prefer(Arc::new(CpuIsaProvider::new()));
    let mut steps = Vec::new();
    let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
    let lowered = registry.dispatch(&mut ctx, &req);
    assert_eq!(lowered.kernels, vec!["cpu_matmul_i8_dyn"]);
    gpu.submit(&[], &steps);
    let got = gpu.read(&out, (m * n) as usize);
    // out[m,n] = sx[m]*1*wq[n,word0,lane0]*xq[m,word0,lane0] (every other
    // lane/word is 0): out = [[1*3, 1*4], [2*3, 2*4]] = [3, 4, 6, 8].
    assert_eq!(got, vec![3.0, 4.0, 6.0, 8.0]);
}
