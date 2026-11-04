// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLOP/OPS accounting through the dispatch seam: the OFFLINE number (walking
//! the recorded step lists, no execution) must agree EXACTLY with the ONLINE
//! counters accumulated at `Gpu::submit`, coverage must be total, and the int8
//! path must report integer OPS where the fp32 path reports FLOPs. CPU backend
//! (deterministic; runs on CI without a GPU).

use gpu_core::{set_default_backend, Backend};
use qwen3::{init_weights, Qwen, QwenConfig, Shard};

#[test]
fn offline_matches_online_and_covers_everything() {
    set_default_backend(Backend::Cpu);
    let cfg = QwenConfig::tiny();
    let (b, t) = (1u32, cfg.block_size);
    let init = init_weights(&cfg, 3);
    let m = Qwen::new(cfg.clone(), b, t, &init);

    let off_f = m.cost_fwd();
    let off_b = m.cost_bwd();
    assert!(off_f.steps > 0 && off_b.steps > 0);
    assert_eq!(off_f.covered, off_f.steps, "forward uncovered: {:?}", off_f.uncovered);
    assert_eq!(off_b.covered, off_b.steps, "backward uncovered: {:?}", off_b.uncovered);
    assert!(off_f.total.flops > 0);
    assert_eq!(off_f.total.int_ops, 0, "fp32 model must report zero integer OPS");

    let x: Vec<u32> = (0..(b * t) as usize).map(|i| i as u32 % cfg.vocab).collect();
    m.set_batch(&x, &x);
    m.gpu().reset_ops_counters();
    m.forward();
    let online = m.gpu().ops_counters();
    assert_eq!(online.steps, off_f.steps);
    assert_eq!(online.total, off_f.total, "online forward != offline forward");

    m.backward();
    let mut expect = off_f.clone();
    expect.merge(&off_b);
    let online = m.gpu().ops_counters();
    assert_eq!(online.steps, expect.steps);
    assert_eq!(online.total, expect.total, "online fwd+bwd != offline fwd+bwd");
}

/// The int8 MAC volume a quantized model's linears report must never exceed
/// the fp32 volume the SAME shapes would cost (same `2·m·K·n` each) - true
/// whichever tier `off` actually reports, so this holds on every backend.
///
/// The fp32 comparison model's own linears do NOT always dispatch `matmul`/
/// `matmul_reg3`: `Ops`'s selector picks `matmul_gemv` instead whenever `m`
/// is within the (fp32) decode regime (`select::DECODE_REGIME_MAX_ROWS`,
/// `QwenConfig::tiny()`'s `block_size=12` included) on a device with
/// `workgroup_reductions` - real on this sandbox's ambient `wgpu` adapter,
/// not just the CPU backend the ORIGINAL version of this helper was only
/// ever exercised against (where `workgroup_reductions` is always `false`,
/// so `matmul_gemv` never got selected and this filter's omission was
/// invisible). All three real fp32 GEMM-tier kernel names must be included,
/// or this helper silently undercounts the fp32 side on a capable device -
/// caught by `i8_model_reports_int_ops_on_an_int8_dot_capable_device`
/// itself going RED against the two-name filter the first time this test
/// ran for real on GPU (B7).
fn assert_int8_volume_bounded_by_fp32(off: &gpu_core::cost::CostReport, cfg: &QwenConfig, init: &std::collections::HashMap<String, Vec<f32>>) {
    let t = cfg.block_size;
    let mfp = Qwen::new(cfg.clone(), 1, t, init);
    let fp = mfp.cost_fwd();
    let fp_linear_flops: u64 = fp
        .by_kernel
        .iter()
        .filter(|(k, _)| matches!(k.as_str(), "matmul" | "matmul_reg3" | "matmul_gemv"))
        .map(|(_, v)| v.cost.flops)
        .sum();
    assert!(fp_linear_flops >= off.total.int_ops, "i8 int_ops exceed the fp32 linear volume");
}

/// **B7 finding, tested here explicitly.** Pre-B7, `qwen3`'s int8 weight
/// path (`q8.rs`) quantized unconditionally, on ANY backend, regardless of
/// declared capability - this test used to assert `int_ops > 0` on the CPU
/// backend specifically BECAUSE of that (CPU can, in fact, execute the
/// packed-dot kernels correctly via the CPU JIT's own portable
/// `dot4I8Packed` lowering - see `matmul_q4_dyn.wgsl`'s header comment - just
/// without hardware DP4A acceleration). B7 migrated the 7 per-layer linears'
/// weight storage onto `model::ops::Weight::upload`, whose `want.
/// promote(caps.numeric)` is a REAL, load-bearing gate (`backend_api::
/// DType::promote`, B1): it never returns `I8` on a device whose
/// `NumericSupport.int8_dot` is `false` - `backend-cpu`'s own `query_caps`
/// declares exactly that (`int8_dot: false`), a POLICY choice (`int8_dot`
/// means "has a FAST hardware dot-product path", not "can execute at all" -
/// see `NumericSupport`'s own doc comment), not a portability limit.
///
/// This is also load-bearing for correctness, not just accounting: `model::
/// ops::Ops::bind` has NO `(Reference, Dtype::I8)` arm (by design - see its
/// own doc comment) - it PANICS if `select::candidates` ever offers
/// `Reference` for an `I8` shape. `select::candidates` only offers
/// `Reference` for `I8` when `int8_dot` is false, so an `Ops`-dispatched
/// `Weight::I8` on a non-`int8_dot` device is a guaranteed panic, not a slow
/// fallback - `Weight::upload`'s promote-to-`F32` gate is what PREVENTS that
/// panic by ensuring an `I8` `Weight` only ever exists where `Ops` can
/// actually dispatch it. Forcing int8 unconditionally (this test's old
/// behaviour) is therefore no longer just "misses a hardware feature" - it
/// would crash `Ops::matmul` on this exact backend.
#[test]
fn i8_model_without_int8_dot_capability_falls_back_to_fp32_not_a_panic() {
    set_default_backend(Backend::Cpu);
    let cfg = QwenConfig::tiny();
    let init = init_weights(&cfg, 3);
    let m8 = Qwen::new_shard_i8(cfg.clone(), 1, cfg.block_size, &init, Shard::whole(cfg.n_layers as usize));
    assert!(!m8.gpu().caps().numeric.int8_dot, "this test's premise is a device with NO int8_dot capability");

    let off = m8.cost_fwd();
    assert_eq!(off.covered, off.steps, "i8-requested forward uncovered: {:?}", off.uncovered);
    assert_eq!(off.total.int_ops, 0, "no int8_dot capability -> Weight::upload demotes to F32, so int_ops must be 0, not silently wrong");
    assert!(off.total.flops > 0, "the demoted-to-fp32 linears must still show up as flops");
    assert!(
        !off.by_kernel.keys().any(|k| k.starts_with("matmul_i8")),
        "no matmul_i8* kernel should be dispatched once weights demoted to F32, got: {:?}",
        off.by_kernel.keys().collect::<Vec<_>>()
    );
    assert_int8_volume_bounded_by_fp32(&off, &cfg, &init);
}

/// The positive case `i8_model_without_int8_dot_capability_falls_back_to_
/// fp32_not_a_panic` above can't exercise on the CPU-only backend: on a
/// device that DOES declare `int8_dot` (this repo's default `wgpu` backend -
/// real hardware in this sandbox, an Intel Arc iGPU, not skipped), the
/// quantized model's linears must show up as `int_ops` (what actually runs:
/// DP4A int8 MACs), NOT as fp32 flops - offline, without executing.
/// Skips cleanly (rather than failing) if the ambient device turns out not
/// to declare `int8_dot` (e.g. a sandbox with only the CPU backend
/// available) - this test's OWN premise, checked, not assumed.
#[test]
fn i8_model_reports_int_ops_on_an_int8_dot_capable_device() {
    // `tiny_i8`, not `tiny`: every quantized `k` must be a whole
    // `model::int8::GROUP` (see `QwenConfig::tiny_i8`'s own doc).
    let cfg = QwenConfig::tiny_i8();
    let init = init_weights(&cfg, 3);
    let m8 = Qwen::new_shard_i8(cfg.clone(), 1, cfg.block_size, &init, Shard::whole(cfg.n_layers as usize));
    if !m8.gpu().caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("ambient device has no int8_dot capability");
        return;
    }

    let off = m8.cost_fwd();
    assert_eq!(off.covered, off.steps, "i8 forward uncovered: {:?}", off.uncovered);
    assert!(off.total.int_ops > 0, "int8 linears must count integer OPS on an int8_dot-capable device");
    assert!(
        off.by_kernel.keys().any(|k| k.starts_with("matmul_i8")),
        "expected a matmul_i8 kernel in the i8 forward, got: {:?}",
        off.by_kernel.keys().collect::<Vec<_>>()
    );
    assert_int8_volume_bounded_by_fp32(&off, &cfg, &init);
}
