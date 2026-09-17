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

use gpu_core::Gpu;
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
    let gpu = Gpu::new(KERNELS);
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
    let gpu = Gpu::new(KERNELS);
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
