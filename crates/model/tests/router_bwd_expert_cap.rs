// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Regression test for `router_bwd.wgsl`'s array-free rewrite (see that
//! file's header doc): correctness at `n_experts` values that exceed the
//! kernel's former hard-coded `array<f32, 64>` scratch (64) -- the same
//! failure shape already named once elsewhere, recurring here because that
//! fix never reached this file.
//!
//! The host oracle mirrors the kernel's own 5-pass structure line for line
//! (recomputing `pr`/`dp` per pass instead of caching them, exactly like the
//! kernel), so this also catches a real numerical regression in the
//! rewrite -- not just an out-of-bounds write. `model::moe` has no backward
//! wiring yet (#36), so this dispatches `router_bwd` directly rather than
//! going through a higher-level API.

use data::rng::Lcg;
use gpu_core::Gpu;
use std::collections::HashSet;

const PIPES: &[(&str, &str)] = &[("router_bwd", kernels::ROUTER_BWD)];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Host oracle: the same 5 serial passes as `router_bwd.wgsl`'s `main`,
/// recomputing `pr_e`/`dp_e` per pass rather than caching them in an array --
/// matching the kernel exactly, not just approximating its result.
#[allow(clippy::too_many_arguments)]
fn host_router_bwd(
    n_rows: usize,
    n_experts: usize,
    aux_coef: f32,
    z_coef: f32,
    logits: &[f32],
    gate: &[f32],
    d_gate: &[f32],
    fe: &[f32],
) -> Vec<f32> {
    let e = n_experts;
    let nrows = n_rows as f32;
    let mut out = vec![0.0f32; n_rows * e];
    for t in 0..n_rows {
        let base = t * e;

        let mut mx = f32::MIN;
        for k in 0..e {
            mx = mx.max(logits[base + k]);
        }
        let mut sum = 0.0f32;
        for k in 0..e {
            sum += (logits[base + k] - mx).exp();
        }
        let lse = mx + sum.ln();

        let mut z = 0.0f32;
        let mut sdp = 0.0f32;
        for k in 0..e {
            if gate[base + k] > 0.0 {
                let pr_k = (logits[base + k] - mx).exp() / sum;
                z += pr_k;
                sdp += d_gate[base + k] * pr_k;
            }
        }
        let zz = z.max(1e-9);

        let mut gpdot = 0.0f32;
        for k in 0..e {
            let pr_k = (logits[base + k] - mx).exp() / sum;
            let mut dp_k = 0.0f32;
            if gate[base + k] > 0.0 {
                dp_k = d_gate[base + k] / zz - sdp / (zz * zz);
            }
            dp_k += aux_coef * e as f32 * fe[k] / nrows;
            gpdot += pr_k * dp_k;
        }

        let zterm = z_coef * 2.0 * lse / nrows;
        for i in 0..e {
            let pr_i = (logits[base + i] - mx).exp() / sum;
            let mut dp_i = 0.0f32;
            if gate[base + i] > 0.0 {
                dp_i = d_gate[base + i] / zz - sdp / (zz * zz);
            }
            dp_i += aux_coef * e as f32 * fe[i] / nrows;
            out[base + i] = pr_i * (dp_i - gpdot) + zterm * pr_i;
        }
    }
    out
}

fn run_case(n_experts: u32) {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "router_bwd");

    let n_rows: u32 = 5;
    // A real subset, not "select everything" -- exercises the gate>0.0 branch
    // as a genuine mask rather than a no-op.
    let top_k = (n_experts / 4).max(1);
    let mut rng = Lcg::new(0xC0FFEE ^ n_experts as u64);

    let logits = rng.vec_scaled((n_rows * n_experts) as usize, 3.0);
    let d_gate = rng.vec_scaled((n_rows * n_experts) as usize, 1.0);
    let fe = rng.vec_unit(n_experts as usize);

    // Plausible top-k gate mask. The kernel only reads `gate[e] > 0.0` as a
    // selection flag -- the exact positive value never appears in its math --
    // so an arbitrary positive constant at `top_k` distinct experts per row
    // is a faithful stand-in for `router_gate.wgsl`'s real output.
    let mut gate = vec![0.0f32; (n_rows * n_experts) as usize];
    for t in 0..n_rows {
        let mut chosen = HashSet::new();
        while chosen.len() < top_k as usize {
            chosen.insert((rng.next_u32() % n_experts) as usize);
        }
        for e in chosen {
            gate[(t * n_experts) as usize + e] = 1.0;
        }
    }

    let aux_coef = 0.01f32;
    let z_coef = 0.001f32;

    let logits_buf = g.storage_init("logits", &logits);
    let gate_buf = g.storage_init("gate", &gate);
    let d_gate_buf = g.storage_init("d_gate", &d_gate);
    let fe_buf = g.storage_init("fe", &fe);
    let dlogits_buf = g.storage((n_rows * n_experts) as u64);

    let params = [n_rows, n_experts, top_k, 0u32, gpu_core::f(aux_coef), gpu_core::f(z_coef), 1, gpu_core::f(1.0)];
    g.submit(
        &[],
        &[g.step(
            kernel,
            &[&logits_buf, &gate_buf, &d_gate_buf, &fe_buf, &dlogits_buf],
            &params,
            n_rows,
        )],
    );
    g.poll_wait();
    let got = g.read(&dlogits_buf, (n_rows * n_experts) as usize);

    let want =
        host_router_bwd(n_rows as usize, n_experts as usize, aux_coef, z_coef, &logits, &gate, &d_gate, &fe);

    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-4,
        "router_bwd diverged from the host oracle at n_experts={n_experts}: max_abs={max_abs} \
         got[..4]={:?} want[..4]={:?}",
        &got[..4.min(got.len())],
        &want[..4.min(want.len())],
    );

    // A meaningful test, not a vacuous one: with a real aux/z term and a
    // genuine top-k subset, the gradient cannot be all zero.
    assert!(want.iter().any(|&v| v.abs() > 1e-9), "oracle output is all-zero at n_experts={n_experts}");
}

/// Below the former cap -- must keep working exactly as before the rewrite.
#[test]
fn router_bwd_matches_host_oracle_at_8_experts() {
    run_case(8);
}

/// One past the former `array<f32, 64>` cap -- the exact boundary #35's shape
/// corrupts silently. This is the test that fails on the pre-rewrite kernel.
#[test]
fn router_bwd_matches_host_oracle_at_65_experts() {
    run_case(65);
}

/// Omni's Thinker scale (128 experts, top-8) -- the real config this fix
/// unblocks for #36.
#[test]
fn router_bwd_matches_host_oracle_at_128_experts() {
    run_case(128);
}
