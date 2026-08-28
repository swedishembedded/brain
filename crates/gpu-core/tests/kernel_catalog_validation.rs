// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cross-backend kernel validation on the machine's CURRENTLY ACTIVE DEFAULT
//! backend - the exact thing a real training/inference run gets from
//! `Gpu::new(...)` with no override, i.e. `gpu_core::devices::
//! ambient_compute_set()`'s real hardware detection, never a hardcoded
//! backend name.
//!
//! Written after a real training run (`qwen35::stream_train_step`) printed
//! "not JIT-compiled ... must use a native fast path or the GPU" for several
//! matmul-family kernels and ran on the CPU backend - a *deliberate* choice
//! in that binary (its resident fp32 `lm_head` exceeds this box's Vulkan
//! `max_buffer_size`), not a silent fallback, but the closest thing this
//! workspace had to an answer for "does matmul actually land on the GPU by
//! default on THIS machine" was a benchmark (`bench_matmul.rs`) that builds
//! every backend EXPLICITLY (`Gpu::new_cpu`/`Gpu::new_wgpu`/
//! `Gpu::try_new_vulkan`) and is `#[ignore]`d - it proves each backend's
//! output is correct in isolation, never what the AMBIENT default actually
//! resolves to. This file is that missing check: fast (small shapes, not
//! ignored - part of a plain `cargo test -p brain-gpu-core`), and it fails
//! loudly if matmul silently lands on CPU on a machine that has a working
//! GPU.
//!
//! This machine has exactly one real GPU (an Intel iGPU via Vulkan/wgpu) -
//! the assertions below are written against `caps`/`gpu.kind()`, not this
//! box's specific hardware, so they hold on whatever backend a P40, a
//! tensor-core card, or a GPU-less box resolves to; they have only ever been
//! EXERCISED against this one real GPU plus the CPU JIT (`BRAIN_DEVICE=cpu`).
//! See `crates/backend-cpu/tests/matmul_family_native_fastpath.rs` for the
//! CPU-backend-specific native-fast-path correctness proof this file does
//! not duplicate.

const TOL_F32: f32 = 5e-4;
const TOL_INT: f32 = 1e-6; // int32 accumulation is exact; dequant is one fp32 multiply.

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 17) % 97) as f32 / 97.0) - 0.5).collect()
}

fn matmul_abt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ni * k + ki];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

fn diff(want: &[f32], got: &[f32]) -> (f32, f32) {
    let maxd = want.iter().zip(got).fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
    let scale = want.iter().fold(1e-6f32, |acc, v| acc.max(v.abs()));
    (maxd, maxd / scale)
}

/// Weight-scale group along K, mirroring `model::int8::GROUP` (duplicated
/// here - `gpu-core` has no dependency on `brain-model`, same reason
/// `bench_matmul.rs`'s own `quant_rows` duplicates the quantizer).
const GROUP: usize = 32;

/// Per-row symmetric quantization of an ACTIVATION, matching what
/// `max_abs_row`/`quant_pack` do on device: one scale per row, 4 int8 per u32.
/// Returns packed `[rows, k/4]` u32, the per-row scale, and the UNPACKED
/// signed values for an exact host reference.
fn quant_act(x: &[f32], rows: usize, k: usize) -> (Vec<u32>, Vec<f32>, Vec<i8>) {
    let kg = k / 4;
    let mut packed = vec![0u32; rows * kg];
    let mut scale = vec![0f32; rows];
    let mut q = vec![0i8; rows * k];
    for r in 0..rows {
        let row = &x[r * k..r * k + k];
        let s = row.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 127.0;
        scale[r] = s;
        for c in 0..k {
            q[r * k + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
        }
        for g in 0..kg {
            let mut word = 0u32;
            for b in 0..4 {
                word |= (q[r * k + g * 4 + b] as u8 as u32) << (8 * b);
            }
            packed[r * kg + g] = word;
        }
    }
    (packed, scale, q)
}

/// GROUP-wise symmetric quantization of a WEIGHT, matching
/// `model::int8::quantize_weight` / `model::int4::quantize_weight_q4`'s math:
/// one scale per 32 elements of K, `per_word` values per u32. Returns packed
/// `[rows, k/per_word]` u32, the `[rows, k/32]` scale, and the UNPACKED signed
/// values for an exact host reference.
fn quant_weight(x: &[f32], rows: usize, k: usize, per_word: usize, qmax: f32) -> (Vec<u32>, Vec<f32>, Vec<i8>) {
    assert_eq!(k % GROUP, 0, "quant_weight: k must be a whole number of groups");
    let kg = k / per_word;
    let gs = k / GROUP;
    let bits = 32 / per_word;
    let mut packed = vec![0u32; rows * kg];
    let mut scale = vec![0f32; rows * gs];
    let mut q = vec![0i8; rows * k];
    for r in 0..rows {
        let row = &x[r * k..r * k + k];
        for g in 0..gs {
            let blk = &row[g * GROUP..g * GROUP + GROUP];
            let s = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / qmax;
            scale[r * gs + g] = s;
            for c in 0..GROUP {
                q[r * k + g * GROUP + c] = (blk[c] / s).round().clamp(-qmax, qmax) as i8;
            }
        }
        for g in 0..kg {
            let mut word = 0u32;
            for b in 0..per_word {
                word |= ((q[r * k + g * per_word + b] as u8 as u32) & ((1 << bits) - 1)) << (bits * b);
            }
            packed[r * kg + g] = word;
        }
    }
    (packed, scale, q)
}

/// Exact reference for the group-wise contract:
/// `out[m,n] = sx[m] * sum_g (sum_{k in g} xq*wq) * sw[n,g]`. The inner sum is
/// integer and exact; only the cross-group fold is floating point, which is
/// exactly what the kernels do.
fn host_group_gemm(xq: &[i8], wq: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let gs = k / GROUP;
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for g in 0..gs {
                let mut ia = 0i32;
                for ki in g * GROUP..g * GROUP + GROUP {
                    ia += xq[mi * k + ki] as i32 * wq[ni * k + ki] as i32;
                }
                acc += ia as f32 * sw[ni * gs + g];
            }
            out[mi * n + ni] = acc * sx[mi];
        }
    }
    out
}

/// Every kernel `crates/kernels` ships (`kernels::ALL`, the single most
/// complete catalogue in the workspace - a strict superset of
/// `model::ops::Ops::REQUIRED_KERNELS`'s 29-name façade) registers and
/// compiles cleanly on the CURRENTLY ACTIVE default backend, built the exact
/// way a real model does (`Gpu::new`, ambient hardware detection - no
/// override).
#[test]
fn whole_kernel_catalog_registers_on_the_active_default_backend() {
    let gpu = gpu_core::testgpu::dev(kernels::ALL);
    let caps = gpu.caps();
    eprintln!(
        "kernel_catalog_validation: backend={} class={:?} workgroup_reductions={} numeric={:?}",
        gpu.kind(),
        caps.class,
        caps.workgroup_reductions,
        caps.numeric
    );
    let missing: Vec<&str> =
        kernels::ALL.iter().filter(|(name, _)| gpu.kernel_index(name).is_none()).map(|(name, _)| *name).collect();
    assert!(
        missing.is_empty(),
        "{} of {} kernels failed to register on the active default backend ({:?}): {:?}",
        missing.len(),
        kernels::ALL.len(),
        gpu.kind(),
        missing
    );
}

/// The matmul family (`matmul`, `matmul_gemv`, `matmul_reg2`, `matmul_reg3`,
/// `matmul_i8_dyn`, `matmul_i8_gemv`, `matmul_q4_dyn`, `matmul_q4_gemv`),
/// dispatched on the ambient default `Gpu`, must (a) produce numerically
/// correct output against a from-first-principles / exact-integer host
/// reference, and (b) - the specific fear the incident that motivated this
/// test raised - never silently land on the CPU backend when this machine's
/// own hardware detection (`gpu_core::visible_gpu_count`) reports a working
/// GPU.
#[test]
fn matmul_family_is_correct_and_lands_on_gpu_when_hardware_available() {
    let gpu = gpu_core::testgpu::dev(kernels::ALL);

    // Any-M shapes (matmul / matmul_reg2 / matmul_reg3 / matmul_i8_dyn /
    // matmul_q4_dyn): not multiples of the 128x128 tile, K a multiple of 32
    // (the weight-scale GROUP, which subsumes both the int8 4-pack and the
    // int4 8-pack) and MORE than one group, so a group-indexing slip shows.
    let (m, k, n) = (37usize, 64usize, 41usize);
    let a = fill(m * k, 1);
    let b = fill(n * k, 2);
    let want_f32 = matmul_abt(&a, &b, m, k, n);

    for (name, threads) in
        [("matmul", (m * n) as u32), ("matmul_reg2", (m.div_ceil(128) * n.div_ceil(128) * 256) as u32)]
    {
        let Some(kind) = gpu.kernel_index(name) else { panic!("{name}: not registered") };
        let ab = gpu.storage_init("a", &a);
        let bb = gpu.storage_init("b", &b);
        let ob = gpu.storage((m * n) as u64);
        let s = gpu.step(kind, &[&ab, &bb, &ob], &[m as u32, k as u32, n as u32], threads);
        gpu.submit(&[], &[s]);
        let got = gpu.read(&ob, m * n);
        let (dabs, drel) = diff(&want_f32, &got);
        assert!(drel < TOL_F32, "{name}: diverges from scalar reference (rel {drel:.3e}, abs {dabs:.3e})");
    }
    // matmul_reg3 is a separate physical kernel from matmul_reg2 (bank-conflict
    // fix - see its own header), bit-identical by construction; check it too.
    {
        let kind = gpu.kernel_index("matmul_reg3").expect("matmul_reg3 registered");
        let ab = gpu.storage_init("a", &a);
        let bb = gpu.storage_init("b", &b);
        let ob = gpu.storage((m * n) as u64);
        let threads = (m.div_ceil(128) * n.div_ceil(128) * 256) as u32;
        let s = gpu.step(kind, &[&ab, &bb, &ob], &[m as u32, k as u32, n as u32], threads);
        gpu.submit(&[], &[s]);
        let got = gpu.read(&ob, m * n);
        let (dabs, drel) = diff(&want_f32, &got);
        assert!(drel < TOL_F32, "matmul_reg3: diverges from scalar reference (rel {drel:.3e}, abs {dabs:.3e})");
    }

    // matmul_i8_dyn / matmul_q4_dyn: any-M tiled quantized GEMMs.
    {
        let (xq, sx, xi) = quant_act(&a, m, k);
        let (wq, sw, wi) = quant_weight(&b, n, k, 4, 127.0);
        let want = host_group_gemm(&xi, &wi, &sx, &sw, m, k, n);
        let xb = gpu.storage(xq.len() as u64);
        gpu.write(&xb, &xq);
        let wb = gpu.storage(wq.len() as u64);
        gpu.write(&wb, &wq);
        let sxb = gpu.storage_init("sx", &sx);
        let swb = gpu.storage_init("sw", &sw);
        let ob = gpu.storage((m * n) as u64);
        let threads = (m.div_ceil(128) * n.div_ceil(128) * 256) as u32;
        if let Some(kind) = gpu.kernel_index("matmul_i8_dyn") {
            let s = gpu.step(kind, &[&xb, &wb, &sxb, &swb, &ob], &[m as u32, (k / 4) as u32, n as u32], threads);
            gpu.submit(&[], &[s]);
            let got = gpu.read(&ob, m * n);
            let (dabs, drel) = diff(&want, &got);
            assert!(drel < TOL_INT, "matmul_i8_dyn: diverges (rel {drel:.3e}, abs {dabs:.3e})");
        }
    }
    {
        let (xq, sx, xi) = quant_act(&a, m, k); // W4A8: activations stay int8
        let (wq, sw, wi) = quant_weight(&b, n, k, 8, 7.0); // weights are int4
        let want = host_group_gemm(&xi, &wi, &sx, &sw, m, k, n);
        let xb = gpu.storage(xq.len() as u64);
        gpu.write(&xb, &xq);
        let wb = gpu.storage(wq.len() as u64);
        gpu.write(&wb, &wq);
        let sxb = gpu.storage_init("sx", &sx);
        let swb = gpu.storage_init("sw", &sw);
        let ob = gpu.storage((m * n) as u64);
        let kind = gpu.kernel_index("matmul_q4_dyn").expect("matmul_q4_dyn registered");
        let s = gpu.step(kind, &[&xb, &wb, &sxb, &swb, &ob], &[m as u32, k as u32, n as u32], (m * n) as u32);
        gpu.submit(&[], &[s]);
        let got = gpu.read(&ob, m * n);
        let (dabs, drel) = diff(&want, &got);
        assert!(drel < TOL_INT, "matmul_q4_dyn: diverges (rel {drel:.3e}, abs {dabs:.3e})");
    }

    // Decode-regime (m<=32) GEMV family: matmul_gemv / matmul_i8_gemv /
    // matmul_q4_gemv.
    let (dm, dk, dn) = (6usize, 64usize, 13usize);
    let da = fill(dm * dk, 3);
    let db8 = fill(dn * dk, 4);
    let db4 = fill(dn * dk, 5);
    let want_gemv = matmul_abt(&da, &db8, dm, dk, dn);
    {
        let kind = gpu.kernel_index("matmul_gemv").expect("matmul_gemv registered");
        let ab = gpu.storage_init("a", &da);
        let bb = gpu.storage_init("b", &db8);
        let ob = gpu.storage((dm * dn) as u64);
        let threads = (dn * 64) as u32;
        let s = gpu.step(kind, &[&ab, &bb, &ob], &[dm as u32, dk as u32, dn as u32], threads);
        gpu.submit(&[], &[s]);
        let got = gpu.read(&ob, dm * dn);
        let (dabs, drel) = diff(&want_gemv, &got);
        assert!(drel < TOL_F32, "matmul_gemv: diverges (rel {drel:.3e}, abs {dabs:.3e})");
    }
    {
        let (xq, sx, xi) = quant_act(&da, dm, dk);
        let (wq, sw, wi) = quant_weight(&db8, dn, dk, 4, 127.0);
        let want = host_group_gemm(&xi, &wi, &sx, &sw, dm, dk, dn);
        let xb = gpu.storage(xq.len() as u64);
        gpu.write(&xb, &xq);
        let wb = gpu.storage(wq.len() as u64);
        gpu.write(&wb, &wq);
        let sxb = gpu.storage_init("sx", &sx);
        let swb = gpu.storage_init("sw", &sw);
        let ob = gpu.storage((dm * dn) as u64);
        let kind = gpu.kernel_index("matmul_i8_gemv").expect("matmul_i8_gemv registered");
        let threads = (dn * 64) as u32;
        let s = gpu.step(kind, &[&xb, &wb, &sxb, &swb, &ob], &[dm as u32, (dk / 4) as u32, dn as u32], threads);
        gpu.submit(&[], &[s]);
        let got = gpu.read(&ob, dm * dn);
        let (dabs, drel) = diff(&want, &got);
        assert!(drel < TOL_INT, "matmul_i8_gemv: diverges (rel {drel:.3e}, abs {dabs:.3e})");
    }
    {
        let (xq, sx, xi) = quant_act(&da, dm, dk);
        let (wq, sw, wi) = quant_weight(&db4, dn, dk, 8, 7.0);
        let want = host_group_gemm(&xi, &wi, &sx, &sw, dm, dk, dn);
        let xb = gpu.storage(xq.len() as u64);
        gpu.write(&xb, &xq);
        let wb = gpu.storage(wq.len() as u64);
        gpu.write(&wb, &wq);
        let sxb = gpu.storage_init("sx", &sx);
        let swb = gpu.storage_init("sw", &sw);
        let ob = gpu.storage((dm * dn) as u64);
        let kind = gpu.kernel_index("matmul_q4_gemv").expect("matmul_q4_gemv registered");
        let threads = (dn * 64) as u32;
        let s = gpu.step(kind, &[&xb, &wb, &sxb, &swb, &ob], &[dm as u32, dk as u32, dn as u32], threads);
        gpu.submit(&[], &[s]);
        let got = gpu.read(&ob, dm * dn);
        let (dabs, drel) = diff(&want, &got);
        assert!(drel < TOL_INT, "matmul_q4_gemv: diverges (rel {drel:.3e}, abs {dabs:.3e})");
    }

    // The core question the incident raised: on a machine with a working
    // GPU, matmul must not silently end up on the CPU backend. All eight
    // dispatches above ran on this ONE `gpu` handle, so one check covers all
    // of them.
    let visible_gpus = gpu_core::visible_gpu_count();
    eprintln!(
        "matmul_family_is_correct_and_lands_on_gpu_when_hardware_available: backend={} visible_gpus={visible_gpus}",
        gpu.kind()
    );
    if visible_gpus > 0 {
        assert_ne!(
            gpu.kind(),
            "cpu",
            "ambient hardware detection reports {visible_gpus} visible GPU(s) but Gpu::new's default \
             backend resolved to \"cpu\" -- the entire matmul family above just ran on CPU on a \
             machine with a working GPU"
        );
    }
}
