// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements neural audio codecs on GPUs for its clients. If your team needs expertise in
// quantizing speech into codes at the speed of the device then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The VQ assignment's transparent upgrade (`vq_argmin` -> `vq_argmin_rows`): a model registering the reference gets
//! the workgroup-per-query twin on a device with workgroup reductions, and every output word - the index and the
//! distance - is the reference's, bit for bit, on codebooks with exact ties, duplicate rows and fewer rows than lanes.

use data::rng::{random_seed, Lcg};

/// The reference under a name the upgrade table does not know, so it always runs as written.
const REFERENCE_ALIAS: &str = "vq_argmin_reference_arm";

fn kernels() -> &'static [(&'static str, &'static str)] {
    &[("vq_argmin", kernels::VQ_ARGMIN), (REFERENCE_ALIAS, kernels::VQ_ARGMIN)]
}

/// Values on a coarse grid, so distances tie exactly; every fourth codebook row repeats an earlier one.
fn grid(rng: &mut Lcg, rows: usize, d: usize, dup: bool) -> Vec<f32> {
    let mut v: Vec<f32> = (0..rows * d).map(|_| (rng.signed() * 4.0).round() / 2.0).collect();
    if dup {
        for r in (4..rows).step_by(4) {
            let (head, tail) = v.split_at_mut(r * d);
            tail[..d].copy_from_slice(&head[(r - 3) * d..(r - 2) * d]);
        }
    }
    v
}

#[test]
fn the_cooperative_assignment_reproduces_the_reference_bits() {
    let gpu = gpu_core::testgpu::dev(kernels());
    let Some(twin) = gpu.kernel_index("vq_argmin_rows") else {
        brain_testutil::skip_unavailable("vq_argmin_rows (the CPU JIT has no workgroup reductions; the upgrade stands down there)");
        return;
    };
    let reference = gpu.kernel_index(REFERENCE_ALIAS).expect("registered");
    let base = gpu.kernel_index("vq_argmin").expect("registered");
    let mut rng = Lcg::new(random_seed());
    for (m, k, d) in [(1u32, 2048u32, 256u32), (300, 2048, 256), (77, 40, 8), (5, 3, 16), (130, 129, 7)] {
        let x = gpu.storage_init("x", &grid(&mut rng, m as usize, d as usize, false));
        let cb = gpu.storage_init("cb", &grid(&mut rng, k as usize, d as usize, true));
        let run = |kind: usize, grid: gpu_core::Dispatch| {
            let out = gpu.storage(2 * u64::from(m));
            gpu.submit(&[], &[gpu.dispatch(kind, &[&x, &cb, &out], &[m, k, d], grid)]);
            gpu.read(&out, 2 * m as usize).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let want = run(reference, gpu_core::Dispatch::Threads(m));
        for (what, got) in [("twin", run(twin, gpu_core::Dispatch::Workgroups(m))), ("upgraded call site", run(base, gpu_core::Dispatch::Threads(m)))] {
            assert_eq!(got, want, "{what}: {m} queries x {k} codebook rows x {d}");
        }
    }
}

/// The row-tiled form (`vq_argmin_tiled`, the codebook transposed, 8 queries a
/// workgroup) gives the reference's every output word too, on full and partial
/// tiles, with ties and duplicate rows.
#[test]
fn the_row_tiled_assignment_reproduces_the_reference_bits() {
    let gpu = gpu_core::testgpu::dev(&[(REFERENCE_ALIAS, kernels::VQ_ARGMIN), ("vq_argmin_tiled", kernels::VQ_ARGMIN_TILED)]);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("vq_argmin_tiled (it needs workgroup reductions, which the CPU JIT lacks)");
        return;
    }
    let (reference, tiled) = (gpu.kernel_index(REFERENCE_ALIAS).expect("registered"), gpu.kernel_index("vq_argmin_tiled").expect("registered"));
    let mut rng = Lcg::new(random_seed());
    for (m, k, d) in [(8u32, 2048u32, 256u32), (301, 2048, 256), (77, 40, 8), (5, 3, 16), (130, 129, 7)] {
        let xv = grid(&mut rng, m as usize, d as usize, false);
        let cbv = grid(&mut rng, k as usize, d as usize, true);
        let cbt: Vec<f32> = (0..(k * d) as usize).map(|j| cbv[(j % k as usize) * d as usize + j / k as usize]).collect();
        let (x, cb, cbt) = (gpu.storage_init("x", &xv), gpu.storage_init("cb", &cbv), gpu.storage_init("cbt", &cbt));
        let run = |kind: usize, table: &gpu_core::DeviceBuffer, grid: gpu_core::Dispatch| {
            let out = gpu.storage(2 * u64::from(m));
            gpu.submit(&[], &[gpu.dispatch(kind, &[&x, table, &out], &[m, k, d], grid)]);
            gpu.read(&out, 2 * m as usize).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let want = run(reference, &cb, gpu_core::Dispatch::Threads(m));
        let got = run(tiled, &cbt, gpu_core::Dispatch::Workgroups(m.div_ceil(gpu_core::select::VQ_TILE_ROWS)));
        assert_eq!(got, want, "{m} queries x {k} codebook rows x {d}");
    }
}
