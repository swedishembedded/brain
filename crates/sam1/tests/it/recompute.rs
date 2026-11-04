// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of vision transformer encoders for its clients. If your
// team needs expertise in training segmentation and document-vision towers within a fixed device budget, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Activation recompute for the SAM-1 tower, held to the contract `model::saved::check_recompute` states: a tower
//! that keeps only each block's output and rebuilds the block from its input before its backward computes the
//! objective and every gradient of the tower that keeps everything, bit for bit, on fewer live device bytes.
//! `SamEncoder` has no `model::Model` impl for that function to take, so this test runs the same clauses on the
//! tower itself. The fixture is the HD-branch tower grown to five blocks, so windowed and global blocks both recur
//! and the tapped block's adjoint joins a downstream block's. One test in its own binary, so nothing else allocates
//! while the device bytes are compared.

use model::saved::{SavedFormat, SavedPolicy};
use sam1::{SamEncoder, SamViTConfig};

/// A step's objective, every parameter's gradient bits, and the device bytes live after the policy switch and at
/// the step's peak.
struct Measured {
    objective: f32,
    grads: Vec<(String, Vec<u32>)>,
    live: Option<u64>,
    peak: Option<u64>,
}

fn config() -> SamViTConfig {
    SamViTConfig { n_layers: 5, ..SamViTConfig::tiny_hd() }
}

fn tower() -> SamEncoder {
    SamEncoder::with_dense_init(gpu_core::testgpu::dev(sam1::PIPELINES), config(), 11)
}

/// One training step under `policy`.
fn measured(policy: SavedPolicy) -> Measured {
    let mut t = tower();
    t.set_saved_policy(policy).unwrap();
    let live = t.gpu.memory_usage().map(|u| u.live);
    t.gpu.reset_memory_peak();
    t.zero_grads();
    let objective = t.objective();
    t.backward();
    let peak = t.gpu.memory_usage().map(|u| u.peak_live);
    let mut names = t.param_names();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), t.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    Measured { objective, grads, live, peak }
}

#[test]
fn sam1_recomputes_its_blocks_exactly() {
    let keep = measured(SavedPolicy::Keep);
    let recompute = measured(SavedPolicy::Recompute);
    if let (Some(k), Some(r)) = (keep.live, recompute.live) {
        assert!(r < k, "recomputing holds {r} live device bytes after the switch, keeping every block {k}");
    }
    if let (Some(k), Some(r)) = (keep.peak, recompute.peak) {
        assert!(r < k, "the recomputing step peaks at {r} live device bytes, keeping every block {k}");
    }
    assert_eq!(recompute.objective.to_bits(), keep.objective.to_bits(), "recomputing gives objective {}, keeping {}", recompute.objective, keep.objective);
    assert!(keep.grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient to compare");
    assert_eq!(keep.grads.len(), recompute.grads.len());
    for ((n, a), (_, b)) in keep.grads.iter().zip(&recompute.grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }

    // Narrow saved formats do not apply, and a recomputing tower does not go back to keeping.
    let mut t = tower();
    assert!(t.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("sam1"));
    t.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(t.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("sam1"));
}
