// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of large models on a single GPU for its clients. If your
// team needs expertise in trading activation memory against recompute and precision then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The device store of quantized layer-boundary residuals (`model::saved::BoundaryStore`) against its host
//! reference: a store followed by a load gives exactly `SavedFormat::round_trip_rows`, for awkward row counts,
//! zero rows and wide dynamic range, and the store refuses what it cannot hold.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::saved::{BoundaryStore, SavedFormat};

fn device_round_trip(gpu: &Gpu, format: SavedFormat, x: &[f32], width: usize) -> Vec<f32> {
    let rows = x.len() / width;
    let store = BoundaryStore::new(gpu, format, 2, rows as u32, width as u32).expect("a store for a narrow format");
    let src = gpu.storage_init("x", x);
    let dst = gpu.storage(x.len() as u64);
    store.store(gpu, 1, &src);
    store.load(gpu, 1, &dst);
    gpu.read(&dst, x.len())
}

fn values(rows: usize, width: usize) -> Vec<f32> {
    let mut rng = Lcg::random();
    let mut x: Vec<f32> = (0..rows * width).map(|i| rng.signed() * 10f32.powi((i / width % 5) as i32 - 2)).collect();
    x[..width].fill(0.0); // an all-zero row
    x
}

#[test]
fn the_device_store_round_trips_exactly_as_the_host_reference_does() {
    let gpu = Gpu::new_cpu(model::saved::kernels());
    for format in [SavedFormat::Bf16, SavedFormat::Fp8] {
        for (rows, width) in [(1, 4), (7, 12), (33, 64), (5, 96)] {
            let x = values(rows, width);
            assert_eq!(device_round_trip(&gpu, format, &x, width), format.round_trip_rows(&x, width), "{format:?} {rows}x{width}");
        }
    }
}

#[test]
fn the_store_reports_its_physical_bytes_and_refuses_what_it_cannot_hold() {
    let gpu = Gpu::new_cpu(model::saved::kernels());
    let store = BoundaryStore::new(&gpu, SavedFormat::Fp8, 3, 10, 8).unwrap();
    assert_eq!(store.bytes(), 3 * SavedFormat::Fp8.bytes(10, 8));
    assert!(BoundaryStore::new(&gpu, SavedFormat::F32, 3, 10, 8).is_err(), "full precision needs no store");
    assert!(BoundaryStore::new(&gpu, SavedFormat::Fp8, 3, 10, 6).is_err(), "fp8 packs four codes to a word");
    let bare = Gpu::new_cpu(&[]);
    assert!(BoundaryStore::new(&bare, SavedFormat::Bf16, 1, 2, 4).err().is_some_and(|e| e.contains("not registered")));
}
