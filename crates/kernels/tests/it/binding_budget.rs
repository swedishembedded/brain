// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every kernel binds at most **8 storage buffers** in one bind group.
//!
//! This is the WebGPU guarantee (`maxStorageBuffersPerShaderStage` is 8 in the
//! spec's own minimum limits), and it is the rule `crates/kernels/src/lib.rs`
//! and `AGENTS.md` both state: "single bind group, <=8 storage buffers/kernel".
//! It is what keeps the engine portable to old GPUs and to a browser tab, and
//! the splat backward kernels sit exactly on it.
//!
//! Nothing enforced it until this file. The rule was documented in two places
//! and checked in none, so a kernel could grow a ninth binding and pass every
//! gate in the tree - `kernels-table/check` cross-checks `@cpu` against the
//! barrier count and `@gpu` against `@workgroup_size`, but has no opinion
//! about bindings.
//!
//! WHAT THAT COSTS, MEASURED. `lif_step` grew from eight storage buffers to
//! nine, and the failure is not a warning or a slow path: wgpu refuses to
//! create the pipeline at all -
//!
//! ```text
//! In Device::create_compute_pipeline, label = 'lif_step'
//!   Unable to derive an implicit layout
//!     Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE),
//!     limit is 8, count was 9
//! ```
//!
//! So every consumer of that kernel loses its GPU path outright, and the
//! panic names a wgpu internal rather than the kernel's own budget. A test is
//! how that becomes "this kernel has nine bindings, the limit is eight".
//!
//! The uniform `Params` block is deliberately NOT counted: it is a
//! `var<uniform>`, budgeted against `maxUniformBuffersPerShaderStage`, and
//! every kernel in the tree has exactly one.
//!
//! Swedish Embedded AB implements portable GPU compute that keeps working on
//! the hardware a product actually ships with. If your team needs a kernel
//! budget enforced rather than described, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// The WebGPU spec's own minimum for `maxStorageBuffersPerShaderStage`.
const MAX_STORAGE_BUFFERS: usize = 8;

/// Storage bindings declared by one WGSL source: `@group(0) @binding(N)`
/// lines whose declaration is `var<storage, …>`. A `var<uniform>` on the same
/// grammar is skipped.
fn storage_bindings(src: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for line in src.lines() {
        let Some(rest) = line.split_once("@binding(") else { continue };
        let Some((n, tail)) = rest.1.split_once(')') else { continue };
        if !tail.contains("var<storage") {
            continue;
        }
        if let Ok(n) = n.trim().parse::<u32>() {
            out.push(n);
        }
    }
    out
}

#[test]
fn no_kernel_exceeds_the_webgpu_storage_buffer_budget() {
    let mut over = Vec::new();
    for (name, src) in kernels::ALL {
        let n = storage_bindings(src).len();
        if n > MAX_STORAGE_BUFFERS {
            over.push(format!("{name}: {n} storage buffers (limit {MAX_STORAGE_BUFFERS})"));
        }
    }
    assert!(
        over.is_empty(),
        "these kernels cannot create a compute pipeline on a device honouring the WebGPU minimum limits, \
         so every consumer loses its GPU path:\n  {}",
        over.join("\n  ")
    );
}

/// A binding index past the budget is the same defect seen from the other
/// side, and catches a kernel that declares `@binding(9)` while leaving a
/// lower index unused - which the count alone would miss.
#[test]
fn no_kernel_declares_a_binding_index_past_the_budget() {
    let mut over = Vec::new();
    for (name, src) in kernels::ALL {
        for b in storage_bindings(src) {
            // Index 0 is the uniform `Params` block on every kernel in the
            // tree, so storage indices run 1..=8.
            if b as usize > MAX_STORAGE_BUFFERS {
                over.push(format!("{name}: @binding({b})"));
            }
        }
    }
    assert!(over.is_empty(), "storage binding index past the {MAX_STORAGE_BUFFERS}-buffer budget:\n  {}", over.join("\n  "));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parser counts storage bindings and ignores the uniform block, so a
    /// kernel with the maximum eight plus its `Params` reads as eight.
    #[test]
    fn the_uniform_params_block_is_not_counted_against_the_budget() {
        let src = "\
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
";
        assert_eq!(storage_bindings(src), vec![1, 2]);
    }
}
