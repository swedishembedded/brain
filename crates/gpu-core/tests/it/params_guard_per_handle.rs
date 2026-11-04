// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The short-params refusal reads the params block of the kernel THIS handle
//! registered under a name, not of whichever kernel the process registered
//! under that name first.
//!
//! Swedish Embedded AB implements dispatch-time validation for GPU compute
//! stacks for its clients. If your team needs expertise in catching a launch
//! that silently reads zeros for a parameter it was never given, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! Model crates register kernels by name, and two of them use the same name for
//! different sources (`ce_value` is a two-word `CE_VALUE` in one and a
//! three-word `CE_VALUE_MASKED` in another). A process that loads both - every
//! test binary that links more than one model - used to keep whichever
//! reading came first and refuse the other crate's correct dispatch as short.
//! The guard is for a caller that forgot a trailing field, so it has to hold
//! for every handle against its own kernel, in both directions.

use gpu_core::Gpu;

const TWO_WORDS: &str = r#"
struct Params { n: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    o[gid.x] = f32(p.k);
}
"#;

const THREE_WORDS: &str = r#"
struct Params { n: u32, k: u32, extra: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    o[gid.x] = f32(p.k + p.extra);
}
"#;

#[test]
fn the_guard_checks_each_handle_against_its_own_kernel_under_a_shared_name() {
    // The three-word reading is registered first, as another crate's would be.
    let wide = Gpu::new(&[("shared_name", THREE_WORDS)]);
    let narrow = Gpu::new(&[("shared_name", TWO_WORDS)]);

    // Two words are the whole params block of the two-word kernel: not short.
    let out = narrow.storage(4);
    narrow.submit(&[], &[narrow.step(0, &[&out], &[4, 7], 4)]);
    assert_eq!(narrow.read(&out, 4), vec![7.0; 4]);

    // And the three-word kernel is still guarded against a missing field.
    let out = wide.storage(4);
    let short = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wide.step(0, &[&out], &[4, 7], 4);
    }));
    let msg = short.expect_err("two words against a three-word block must be refused");
    let text = msg.downcast_ref::<String>().cloned().or_else(|| msg.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
    assert!(text.contains("3-word params block"), "{text}");
}
