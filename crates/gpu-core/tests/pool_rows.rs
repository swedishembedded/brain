// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `pool_rows_gather2` / `pool_rows_scatter2`: staging the rows of per-sequence
//! recurrent state out of, and back into, per-layer pools in ONE dispatch each.
//!
//! Swedish Embedded AB implements batched decode serving for hybrid
//! recurrent-attention models for its clients. If your team needs expertise in
//! keeping per-request model state resident and batching it without a dispatch
//! per request then you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("pool_rows_gather2", kernels::POOL_ROWS_GATHER2), ("pool_rows_scatter2", kernels::POOL_ROWS_SCATTER2)];
const K_GATHER: usize = 0;
const K_SCATTER: usize = 1;

fn floats(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.next_u32() % 4000) as f32 * 1e-3 - 2.0).collect()
}

fn upload(gpu: &Gpu, v: &[f32]) -> gpu_core::DeviceBuffer {
    gpu.storage_init("pool_rows.test", v)
}

fn upload_ids(gpu: &Gpu, ids: &[u32]) -> gpu_core::DeviceBuffer {
    let f: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();
    gpu.storage_init("pool_rows.ids", &f)
}

/// Pool rows, in an arbitrary order, land in batch order; unpicked rows are not
/// read into the slab.
#[test]
fn gather_stages_the_picked_rows_in_batch_order() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut rng = Lcg::new(7);
    for &(rows, len_a, len_b, ids) in &[(6usize, 40usize, 12usize, &[4u32, 0, 5][..]), (3, 7, 1, &[2]), (8, 64, 0, &[7, 6, 5, 4, 3, 2, 1, 0])] {
        let (pa, pb) = (floats(&mut rng, rows * len_a), floats(&mut rng, rows * len_b));
        let b = ids.len();
        let (a, bb) = (upload(&gpu, &pa), upload(&gpu, &pb.iter().chain(&[0.0]).copied().collect::<Vec<_>>()));
        let (oa, ob) = (gpu.storage((b * len_a) as u64), gpu.storage((b * len_b).max(1) as u64));
        let idb = upload_ids(&gpu, ids);
        gpu.submit(&[], &[gpu.step(K_GATHER, &[&a, &bb, &idb, &oa, &ob], &[b as u32, len_a as u32, len_b as u32], (b * (len_a + len_b)) as u32)]);
        gpu.poll_wait();
        let (ga, gb) = (gpu.read(&oa, b * len_a), gpu.read(&ob, b * len_b));
        for (i, &r) in ids.iter().enumerate() {
            let r = r as usize;
            assert_eq!(ga[i * len_a..(i + 1) * len_a], pa[r * len_a..(r + 1) * len_a], "state row {i} (pool row {r})");
            assert_eq!(gb[i * len_b..(i + 1) * len_b], pb[r * len_b..(r + 1) * len_b], "hist row {i} (pool row {r})");
        }
    }
}

/// Scatter writes exactly the picked rows and leaves every other row of the pool
/// as it was; gather after scatter returns what was scattered.
#[test]
fn scatter_writes_only_the_picked_rows() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut rng = Lcg::new(9);
    let (rows, len_a, len_b) = (6usize, 40usize, 12usize);
    let ids = [5u32, 1, 3];
    let (pa, pb) = (floats(&mut rng, rows * len_a), floats(&mut rng, rows * len_b));
    let (ia, ib) = (floats(&mut rng, ids.len() * len_a), floats(&mut rng, ids.len() * len_b));
    let (a, b) = (upload(&gpu, &pa), upload(&gpu, &pb));
    let (xa, xb, idb) = (upload(&gpu, &ia), upload(&gpu, &ib), upload_ids(&gpu, &ids));
    gpu.submit(&[], &[gpu.step(K_SCATTER, &[&xa, &xb, &idb, &a, &b], &[ids.len() as u32, len_a as u32, len_b as u32], (ids.len() * (len_a + len_b)) as u32)]);
    gpu.poll_wait();
    let (ga, gb) = (gpu.read(&a, rows * len_a), gpu.read(&b, rows * len_b));
    for r in 0..rows {
        let (want_a, want_b) = match ids.iter().position(|&i| i as usize == r) {
            Some(i) => (&ia[i * len_a..(i + 1) * len_a], &ib[i * len_b..(i + 1) * len_b]),
            None => (&pa[r * len_a..(r + 1) * len_a], &pb[r * len_b..(r + 1) * len_b]),
        };
        assert_eq!(&ga[r * len_a..(r + 1) * len_a], want_a, "state row {r}");
        assert_eq!(&gb[r * len_b..(r + 1) * len_b], want_b, "hist row {r}");
    }
}
