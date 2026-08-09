// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Phase 6 ("migrate glm onto model::moe") measure-first gate: does
//! sparse-naive (`model::moe::expert_fwd`'s `moe_linear_gated`, real FLOPs
//! only for routed rows) beat dense-tiled (`crates/glm`'s current
//! `pick_gemm`-selected `matmul_reg3`, every row, every expert) at GLM-5.2's
//! REAL shape on a REAL P40 -- not assumed from FLOP-counting alone, which
//! `docs/models/omni/status.md`'s own risk note says can invert either way
//! depending on how naive-vs-tiled scales at the realistic per-expert row
//! count.
//!
//! Full-scale (256 experts x [6144,2048] gate/up + [2048,6144] down, fp32)
//! would need ~38.6 GB just for the routed experts' weights -- over one
//! P40's 24 GB. So this measures ONE expert's three GEMMs at each row count
//! (m=64 for sparse -- the real average `2048 tokens * top_k=8 / 256
//! experts`; m=2048 for dense -- the full batch every expert sees) and
//! extrapolates the total by expert count, which is valid because each
//! expert's compute is independent and would run sequentially in the real
//! model regardless of whether all 256 are resident at once.
//!
//! A third arm was added later (`.todo/moe-tiled-gated-kernel.md`'s "real
//! fix"): sparse-COMPACT, `model::moe::expert_fwd_compact`'s real dispatch
//! shape -- gather this expert's routed rows into a dense sub-batch, run the
//! SAME `pick_gemm`-selected GEMM the dense-tiled arm uses (at m=rows/expert,
//! not the full batch), scatter back. This measures whether combining
//! sparsity's FLOP savings with the tiled GEMM's per-FLOP efficiency (instead
//! of trading one for the other, which is what made sparse-naive lose) beats
//! BOTH other arms.
//!
//! usage: `BRAIN_DEVICE=vulkan cargo run --release -p brain-glm --example moe_migration_bench`

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::pick_gemm;

const D_MODEL: u32 = 6144;
const MOE_FF: u32 = 2048;
const N_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const SEQ_LEN: u32 = 2048;
const REPS: usize = 5;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Best-of-`REPS`, poll_wait-bracketed (docs/kernel-checklist.md §E.0: a
/// bare-submit loop times host recording, not the device).
fn best_of(g: &Gpu, steps: &[gpu_core::Step]) -> f64 {
    g.submit(&[], steps);
    g.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = std::time::Instant::now();
        g.submit(&[], steps);
        g.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let g = Gpu::new(&[
        ("matmul", kernels::MATMUL),
        ("matmul_reg3", kernels::MATMUL_REG3),
        ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
        ("embed", kernels::EMBED),
        ("moe_scatter_scaled_add", kernels::MOE_SCATTER_SCALED_ADD),
    ]);
    let matmul = idx(&g, "matmul");
    let matmul_reg3 = idx(&g, "matmul_reg3");
    let moe_linear_gated = idx(&g, "moe_linear_gated");
    let embed = idx(&g, "embed");
    let scatter = idx(&g, "moe_scatter_scaled_add");

    let mut rng = Lcg::new(7);
    let rows_per_expert = (SEQ_LEN * TOP_K) / N_EXPERTS;
    println!("GLM-5.2 shape: d_model={D_MODEL} moe_ff={MOE_FF} n_experts={N_EXPERTS} top_k={TOP_K} seq_len={SEQ_LEN}");
    println!("rows/expert (sparse): {rows_per_expert}  rows/expert (dense, every expert sees the whole batch): {SEQ_LEN}");

    // One expert's weights -- gate_w/up_w are [moe_ff, d_model], down_w is [d_model, moe_ff].
    let gate_w = g.storage_init("gate_w", &rng.vec_scaled((MOE_FF * D_MODEL) as usize, 0.02));
    let up_w = g.storage_init("up_w", &rng.vec_scaled((MOE_FF * D_MODEL) as usize, 0.02));
    let down_w = g.storage_init("down_w", &rng.vec_scaled((D_MODEL * MOE_FF) as usize, 0.02));

    // Sparse-naive: m = rows_per_expert, moe_linear_gated needs a gate buffer
    // -- all-ones (every row in this synthetic slice is "routed" to this one
    // expert, matching the intent of measuring a routed row's real cost).
    let x_sparse = g.storage_init("x_sparse", &rng.vec_scaled((rows_per_expert * D_MODEL) as usize, 0.1));
    let gate_sparse = g.storage_init("gate_sparse", &vec![1.0f32; rows_per_expert as usize]);
    let gate_pre_s = g.storage((rows_per_expert * MOE_FF) as u64);
    let up_s = g.storage((rows_per_expert * MOE_FF) as u64);
    let out_s = g.storage((rows_per_expert * D_MODEL) as u64);
    let sparse_steps = [
        g.step(moe_linear_gated, &[&x_sparse, &gate_w, &gate_sparse, &gate_pre_s], &[rows_per_expert, D_MODEL, MOE_FF, 1, 0], rows_per_expert * MOE_FF),
        g.step(moe_linear_gated, &[&x_sparse, &up_w, &gate_sparse, &up_s], &[rows_per_expert, D_MODEL, MOE_FF, 1, 0], rows_per_expert * MOE_FF),
        g.step(moe_linear_gated, &[&up_s, &down_w, &gate_sparse, &out_s], &[rows_per_expert, MOE_FF, D_MODEL, 1, 0], rows_per_expert * D_MODEL),
    ];
    let sparse_time = best_of(&g, &sparse_steps);

    // Dense-tiled: m = SEQ_LEN (every expert sees the whole batch), via the
    // SAME pick_gemm selection crates/glm/src/model.rs's `Mlp::Moe` arm uses.
    let x_dense = g.storage_init("x_dense", &rng.vec_scaled((SEQ_LEN * D_MODEL) as usize, 0.1));
    let gate_pre_d = g.storage((SEQ_LEN * MOE_FF) as u64);
    let up_d = g.storage((SEQ_LEN * MOE_FF) as u64);
    let out_d = g.storage((SEQ_LEN * D_MODEL) as u64);
    let (k1, t1) = pick_gemm(SEQ_LEN as usize, MOE_FF as usize, matmul, matmul_reg3, false);
    let (k2, t2) = pick_gemm(SEQ_LEN as usize, D_MODEL as usize, matmul, matmul_reg3, false);
    let dense_steps = [
        g.step(k1, &[&x_dense, &gate_w, &gate_pre_d], &[SEQ_LEN, D_MODEL, MOE_FF], t1),
        g.step(k1, &[&x_dense, &up_w, &up_d], &[SEQ_LEN, D_MODEL, MOE_FF], t1),
        g.step(k2, &[&up_d, &down_w, &out_d], &[SEQ_LEN, MOE_FF, D_MODEL], t2),
    ];
    let dense_time = best_of(&g, &dense_steps);

    // Sparse-COMPACT: gather rows_per_expert rows (an IDENTITY gather here --
    // idx[i]=i -- since this bench measures per-expert cost in isolation, not
    // a real layer's routing; a real caller's indices are data-dependent, but
    // the gather kernel's cost depends only on how much data moves, not on
    // which rows were chosen). Then the SAME pick_gemm selection as dense,
    // but at m=rows_per_expert. Then scatter back (also identity here).
    let identity_idx: Vec<u32> = (0..rows_per_expert).collect();
    let idx_buf = g.storage(rows_per_expert as u64);
    g.write(&idx_buf, &identity_idx);
    let x_compact_src = g.storage_init("x_compact_src", &rng.vec_scaled((SEQ_LEN * D_MODEL) as usize, 0.1));
    let x_compact = g.storage((rows_per_expert * D_MODEL) as u64);
    let gate_pre_c = g.storage((rows_per_expert * MOE_FF) as u64);
    let up_c = g.storage((rows_per_expert * MOE_FF) as u64);
    let out_c = g.storage((rows_per_expert * D_MODEL) as u64);
    let acc_c = g.storage((SEQ_LEN * D_MODEL) as u64);
    let gate_dense_all = g.storage_init("gate_dense_all", &vec![1.0f32; SEQ_LEN as usize]); // [SEQ_LEN, 1] -- n_experts=1 here
    let (ck1, ct1) = pick_gemm(rows_per_expert as usize, MOE_FF as usize, matmul, matmul_reg3, false);
    let (ck2, ct2) = pick_gemm(rows_per_expert as usize, D_MODEL as usize, matmul, matmul_reg3, false);
    let compact_steps = [
        g.step(embed, &[&idx_buf, &x_compact_src, &x_compact], &[D_MODEL, rows_per_expert], rows_per_expert * D_MODEL),
        g.step(ck1, &[&x_compact, &gate_w, &gate_pre_c], &[rows_per_expert, D_MODEL, MOE_FF], ct1),
        g.step(ck1, &[&x_compact, &up_w, &up_c], &[rows_per_expert, D_MODEL, MOE_FF], ct1),
        g.step(ck2, &[&up_c, &down_w, &out_c], &[rows_per_expert, MOE_FF, D_MODEL], ct2),
        g.step(scatter, &[&idx_buf, &gate_dense_all, &out_c, &acc_c], &[rows_per_expert, D_MODEL, 1, 0, 0], rows_per_expert * D_MODEL),
    ];
    let compact_time = best_of(&g, &compact_steps);

    let sparse_total = sparse_time * N_EXPERTS as f64;
    let dense_total = dense_time * N_EXPERTS as f64;
    let compact_total = compact_time * N_EXPERTS as f64;

    println!();
    println!(
        "per-expert: sparse-naive {sparse_time:.6}s  dense-tiled {dense_time:.6}s  sparse-compact {compact_time:.6}s"
    );
    println!(
        "all {N_EXPERTS} experts, one MoE layer: sparse-naive {sparse_total:.4}s  dense-tiled {dense_total:.4}s  sparse-compact {compact_total:.4}s"
    );
    println!();
    let best = sparse_total.min(dense_total).min(compact_total);
    let name = if best == compact_total {
        "sparse-compact"
    } else if best == sparse_total {
        "sparse-naive"
    } else {
        "dense-tiled"
    };
    println!(
        "RESULT: {name} is fastest at this real shape ({:.2}x vs dense-tiled, {:.2}x vs sparse-naive).",
        dense_total / best,
        sparse_total / best
    );
    if name != "dense-tiled" {
        println!("  -> migrating off the dense path is now worth it (was NOT, before sparse-compact existed).");
    } else {
        println!("  -> still do NOT migrate; dense-tiled remains fastest even against sparse-compact.");
    }
}
