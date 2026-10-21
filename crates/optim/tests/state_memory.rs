// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in bounding host memory of large-model
// training then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gate: the host memory of a streamed optimiser step does not grow with the
//! parameter count. A counting allocator measures the peak heap above the
//! baseline during `step` for a model sixty-four times larger; with value, gradient and
//! state all on disk the peak must stay within the optimiser's declared staging.
//! This file holds ONE test: a global allocator sees every thread of its binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, FileBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, StateFormat, ValueDtype};

struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Peak heap bytes above baseline during one step over `n` parameters, and
/// the optimiser's declared staging.
fn step_peak(n: usize) -> (usize, usize) {
    let dir = std::env::temp_dir().join(format!("brain-optim-mem-{n}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let precision = ParamPrecision {
        value: ValueDtype::F32,
        grad: GradDtype::F32,
        state: StateFormat::Int8Block { block: 256 },
        master: MasterPolicy::F32,
        stochastic_rounding: true,
    };
    let rule = Rule::AdamW { adam: Adam::default(), wd: 0.01 };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision, row_len: None }];
    let layout = state_layout(&rule, &specs);
    let tile = 1 << 14;
    let st = BlockStateStorage::new(layout.clone(), FileBacking::create(&dir.join("s"), layout.total_bytes()).unwrap(), tile).unwrap();
    let mut opt = StreamedOptimizer::new(rule, specs, st, 3, tile).unwrap();
    let vl = tensor_layout(&[n], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), FileBacking::create(&dir.join("v"), vl.total_bytes()).unwrap(), tile).unwrap();
    let mut grads = BlockStateStorage::new(vl.clone(), FileBacking::create(&dir.join("g"), vl.total_bytes()).unwrap(), tile).unwrap();
    let chunk: Vec<f32> = (0..tile).map(|i| (i % 17) as f32 * 0.01).collect();
    for start in (0..n).step_by(tile) {
        let m = tile.min(n - start);
        values.write_tile(0, 0, start, &chunk[..m], Rounding::NEAREST).unwrap();
        grads.write_tile(0, 0, start, &chunk[..m], Rounding::NEAREST).unwrap();
    }
    let mut params = HostParams { values, grads };
    opt.init_master(&mut params).unwrap();
    let base = CUR.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    opt.step(&mut params, 1e-3, Some(1.0), 1.0).unwrap();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    let _ = std::fs::remove_dir_all(&dir);
    (peak, opt.staging_bytes())
}

#[test]
fn streamed_step_heap_is_independent_of_parameter_count() {
    // the counter must see allocations at all, or "0 bytes" certifies nothing
    let before = CUR.load(Ordering::Relaxed);
    let probe = std::hint::black_box(vec![1u8; 1 << 20]);
    assert!(CUR.load(Ordering::Relaxed) >= before + (1 << 20));
    drop(probe);
    let (small, staging) = step_peak(1 << 16);
    let (large, _) = step_peak(1 << 22);
    println!("staging_bytes {staging}; step heap peak above baseline: {small} B at 65536 params, {large} B at 4194304 params");
    // the step itself allocates nothing beyond what was declared at construction
    assert!(large <= small + 4096, "heap peak grew with parameter count: {small} -> {large}");
    assert!(large <= 4096, "a step must not allocate beyond its pre-allocated workspace, got {large}");
}
