// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mixture-of-experts serving kernels for its clients. If your team needs expertise in
// routing tokens to experts at the speed of the device then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The sigmoid router's transparent upgrade (`router_gate_sigmoid` -> `router_gate_sigmoid_rows`): a model registering
//! the reference gets the workgroup-per-row twin on a device with workgroup reductions, and the twin's every output
//! word - the routed weights and the sigmoid scores - is the reference's, bit for bit, with and without group-limited
//! masking, on scores full of exact ties.

use data::rng::{random_seed, Lcg};

/// The reference under a name the upgrade table does not know, so it always runs as written.
const REFERENCE_ALIAS: &str = "router_gate_sigmoid_reference_arm";

fn kernels() -> &'static [(&'static str, &'static str)] {
    &[("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID), (REFERENCE_ALIAS, kernels::ROUTER_GATE_SIGMOID)]
}

/// Routing parameters: (rows, experts, top_k, groups, kept groups, renormalise, scale).
type Shape = (u32, u32, u32, u32, u32, bool, f32);

/// Logits on a coarse grid (exact ties between experts and within groups) with a few large and non-finite values.
fn logits(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| match i % 97 {
            13 => f32::INFINITY,
            41 => f32::NEG_INFINITY,
            _ => (rng.signed() * 8.0).round() / 4.0,
        })
        .collect()
}

#[test]
fn the_cooperative_router_reproduces_the_reference_bits() {
    let gpu = gpu_core::testgpu::dev(kernels());
    let Some(twin) = gpu.kernel_index("router_gate_sigmoid_rows") else {
        brain_testutil::skip_unavailable("router_gate_sigmoid_rows (the CPU JIT has no workgroup reductions; the upgrade stands down there)");
        return;
    };
    let reference = gpu.kernel_index(REFERENCE_ALIAS).expect("registered");
    let base = gpu.kernel_index("router_gate_sigmoid").expect("registered");
    let mut rng = Lcg::new(random_seed());
    let shapes: [Shape; 4] = [(27, 128, 8, 1, 1, true, 8.0), (1, 128, 8, 1, 1, true, 8.0), (70, 256, 8, 8, 4, true, 2.5), (9, 64, 6, 4, 2, false, 1.0)];
    for (rows, e, k, ng, kg, norm, scale) in shapes {
        let n = (rows * e) as usize;
        let lg = gpu.storage_init("logits", &logits(&mut rng, n));
        let bias = gpu.storage_init("bias", &(0..e).map(|i| ((i % 5) as f32 - 2.0) / 8.0).collect::<Vec<_>>());
        let params = [rows, e, k, ng, kg, norm as u32, gpu_core::f(scale)];
        let run = |kind: usize, grid: gpu_core::Dispatch| {
            let (gate, probs) = (gpu.storage(n as u64), gpu.storage(n as u64));
            gpu.submit(&[], &[gpu.dispatch(kind, &[&lg, &bias, &gate, &probs], &params, grid)]);
            (gpu.read(&gate, n), gpu.read(&probs, n))
        };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let want = run(reference, gpu_core::Dispatch::Threads(rows));
        let chosen = want.0.chunks(e as usize).map(|r| r.iter().filter(|w| **w != 0.0).count()).min().unwrap_or(0);
        assert!(chosen > 0, "the reference routed nothing at {rows} x {e}: the fixture is degenerate");
        for (what, got) in [("twin", run(twin, gpu_core::Dispatch::Workgroups(rows))), ("upgraded call site", run(base, gpu_core::Dispatch::Threads(rows)))] {
            assert_eq!(bits(&got.0), bits(&want.0), "{what}: routed weights at {rows} rows x {e} experts, top {k}, {kg}/{ng} groups");
            assert_eq!(bits(&got.1), bits(&want.1), "{what}: sigmoid scores at {rows} rows x {e} experts");
        }
    }
}
