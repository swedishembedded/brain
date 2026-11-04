// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `emb_bwd_uniq` must scatter exactly what `emb_bwd` scatters.
//!
//! The compact kernel spends one invocation per (DISTINCT looked-up row,
//! channel) where the reference spends one per (table row, channel). That is
//! the whole difference, and it is worth a large multiple on a
//! vocabulary-sized table - but it moves the correctness burden onto the
//! caller's `uniq` list, and a wrong list fails SILENTLY: the rows it omits
//! simply stop learning while every other tensor keeps training normally.
//!
//! So this asserts the BITS, not a tolerance. Both kernels add the same
//! summands in the same ascending order, so anything less would be hiding a
//! real difference.
//!
//! Swedish Embedded AB implements validated GPU kernel selection and
//! optimization for its clients. If your team needs expertise in numerically
//! gated kernel work, you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::block;

static KERNELS: &[(&str, &str)] =
    &[("emb_bwd", kernels::EMB_BWD), ("emb_bwd_uniq", kernels::EMB_BWD_UNIQ)];

/// `(n_rows, d_model, table_rows)` - a sparse lookup (many table rows never
/// touched), a dense one (every row touched), and a lookup with heavy repeats,
/// which is the case where a per-output accumulator has more than one summand.
const SHAPES: &[(u32, u32, u32)] = &[(37, 16, 4096), (16, 8, 16), (64, 24, 5), (1, 64, 1024)];

fn tokens(n: u32, table: u32, seed: u32) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + seed * 13) % table.max(1)) % table.max(1)).collect()
}

#[test]
fn the_compact_scatter_is_bit_identical_to_the_reference() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (kref, kuniq) = (gpu.kernel_index("emb_bwd").unwrap(), gpu.kernel_index("emb_bwd_uniq").unwrap());

    for &(rows, d, table) in SHAPES {
        let tok = tokens(rows, table, 7);
        let uniq = block::uniq_u32(&tok);
        // Non-trivial starting gradient: both kernels ACCUMULATE, so a zeroed
        // table would not catch a kernel that assigned instead of added.
        let base: Vec<f32> =
            (0..table * d).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let dx: Vec<f32> = (0..rows * d).map(|i| ((i % 23) as f32 - 11.0) * 0.125).collect();

        let idx = gpu.buffer("idx", rows as u64 * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        gpu.write(&idx, &tok);
        let ub = gpu.buffer("uniq", uniq.len() as u64 * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        gpu.write(&ub, &uniq);
        let dxb = gpu.storage(dx.len() as u64);
        gpu.write_f32(&dxb, &dx);

        let run = |compact: bool| -> Vec<f32> {
            let g = gpu.storage(base.len() as u64);
            gpu.write_f32(&g, &base);
            let step = if compact {
                gpu.step(kuniq, &[&idx, &ub, &dxb, &g], &[rows, d, uniq.len() as u32], uniq.len() as u32 * d)
            } else {
                gpu.step(kref, &[&idx, &dxb, &g], &[rows, d, table], table * d)
            };
            gpu.submit(&[], &[step]);
            gpu.read(&g, base.len())
        };

        let (a, b) = (run(false), run(true));
        let diff = a.iter().zip(&b).position(|(x, y)| x.to_bits() != y.to_bits());
        assert!(
            diff.is_none(),
            "rows {rows} d {d} table {table}: element {} differs - {:?} vs {:?}",
            diff.unwrap(),
            a[diff.unwrap()],
            b[diff.unwrap()],
        );
    }
}

/// The selector must not reach for the compact kernel when it would do MORE
/// work than the reference - a caller whose lookups cover the whole table.
#[test]
fn the_selector_keeps_the_reference_when_every_row_is_looked_up() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let ids = block::EmbBwdIds::resolve(&gpu, gpu.kernel_index("emb_bwd").unwrap());
    assert!(ids.emb_bwd_uniq.is_some(), "emb_bwd_uniq did not resolve");

    let idx = gpu.buffer("idx", 64, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
    let ub = gpu.buffer("uniq", 64, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
    let dx = gpu.storage(64);
    let gr = gpu.storage(64);

    // 8 distinct ids into an 8-row table: the compact kernel would dispatch the
    // same threads plus an indirection, so the reference stands.
    let step = block::emb_bwd_step(&gpu, &ids, &idx, Some((&ub, 8)), &dx, &gr, 8, 2, 8);
    assert_eq!(step.meta().unwrap().kernel, ids.emb_bwd, "the compact kernel was chosen at n_uniq == table_rows");

    // 4 distinct into 8 rows: half the work, so it is taken.
    let step = block::emb_bwd_step(&gpu, &ids, &idx, Some((&ub, 4)), &dx, &gr, 8, 2, 8);
    assert_eq!(step.meta().unwrap().kernel, ids.emb_bwd_uniq.unwrap());

    // No list at all: the reference, unchanged.
    let step = block::emb_bwd_step(&gpu, &ids, &idx, None, &dx, &gr, 8, 2, 8);
    assert_eq!(step.meta().unwrap().kernel, ids.emb_bwd);
}

fn to_bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// Spec: `Ops::embedding_grad` over a distinct-id list padded to a fixed length (the form a recorded tape replays)
/// scatters exactly what the reference scatters into an fp32 gradient, and into a bf16 gradient adds each row's f32
/// sum (ascending token order) to the stored value with one round to nearest even. Rows never looked up keep their
/// bits in both.
#[test]
fn the_embedding_gradient_op_matches_the_reference_in_f32_and_bf16() {
    use model::ops::{training_kernel_list, uniq_padded, EmbeddingGrad, GradDtype, Ops};
    // The reference kernel beside the training set the op draws on.
    let list: &'static [(&str, &str)] = Box::leak(training_kernel_list().iter().copied().chain([("emb_bwd", kernels::EMB_BWD)]).collect::<Vec<_>>().into_boxed_slice());
    let ops = Ops::new(gpu_core::Gpu::new(list)).expect("Ops::new");
    let gpu = ops.gpu();
    let kref = gpu.kernel_index("emb_bwd").unwrap();
    for &(rows, d, table) in SHAPES {
        let tok = tokens(rows, table, 3);
        let uniq = uniq_padded(&tok, rows as usize);
        let base: Vec<f32> = (0..table * d).map(|i| ((i % 19) as f32 - 9.0) * 0.375).collect();
        let dx: Vec<f32> = (0..rows * d).map(|i| ((i % 29) as f32 - 14.0) * 0.0625).collect();
        let idx = gpu.storage(rows as u64);
        gpu.write(&idx, &tok);
        let ub = gpu.storage(rows as u64);
        gpu.write(&ub, &uniq);
        let dxb = gpu.storage(dx.len() as u64);
        gpu.write_f32(&dxb, &dx);
        let grad = |dw: &gpu_core::DeviceBuffer, dtype| {
            let mut s = Vec::new();
            ops.embedding_grad(&mut s, &EmbeddingGrad { tokens: &idx, uniq: &ub, n_uniq: rows, d_x: &dxb, n_rows: rows, width: d, dw, grad: dtype });
            gpu.submit(&[], &s);
        };

        let want = gpu.storage(base.len() as u64);
        gpu.write_f32(&want, &base);
        gpu.submit(&[], &[gpu.step(kref, &[&idx, &dxb, &want], &[rows, d, table], table * d)]);
        let got = gpu.storage(base.len() as u64);
        gpu.write_f32(&got, &base);
        grad(&got, GradDtype::F32);
        let (want, got) = (gpu.read(&want, base.len()), gpu.read(&got, base.len()));
        assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()), "rows {rows} d {d} table {table}: f32 gradient");

        let packed: Vec<u32> = base.chunks(2).map(|p| u32::from(to_bf16_bits(p[0])) | (u32::from(to_bf16_bits(p[1])) << 16)).collect();
        let bf16 = gpu.storage(packed.len() as u64);
        gpu.write(&bf16, &packed);
        grad(&bf16, GradDtype::Bf16);
        let got: Vec<u16> = gpu.read(&bf16, packed.len()).iter().flat_map(|w| [(w.to_bits() & 0xFFFF) as u16, (w.to_bits() >> 16) as u16]).collect();
        for v in 0..table as usize {
            for c in 0..d as usize {
                let stored = f32::from_bits(u32::from(to_bf16_bits(base[v * d as usize + c])) << 16);
                let sum = tok.iter().enumerate().filter(|(_, &t)| t as usize == v).fold(0f32, |acc, (n, _)| acc + dx[n * d as usize + c]);
                let want = if tok.contains(&(v as u32)) { to_bf16_bits(stored + sum) } else { to_bf16_bits(stored) };
                assert_eq!(got[v * d as usize + c], want, "rows {rows} d {d} table {table}: bf16 gradient at row {v} channel {c}");
            }
        }
    }
}
