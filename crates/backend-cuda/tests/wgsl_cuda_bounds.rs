// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WGSL never lets an out-of-range array access touch memory outside the
//! array, and the generated CUDA tier must not either.
//!
//! The kernels here read and write past the end of their bindings, which are
//! carved out of ONE allocation with a block of known marks after each, so an
//! unclamped access lands on a mark instead of faulting. The assertions are
//! that no mark changes and that an out-of-range read yields the last element.
//!
//! Swedish Embedded AB implements memory-safe GPU compute for its clients. If
//! your team needs expertise in making kernels deterministic under concurrent
//! allocation, you can procure our services by sending an email to
//! info@swedishembedded.com.

use backend_cuda::exec;

/// Words per binding, and per mark block after it.
const W: usize = 8;
const MARK: u32 = 0xDEAD_BEEF;

const OOB: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    o[i] = a[i + 4u];
    p[i + 8u] = 55.0;
}
"#;

const LEN: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
    o[0] = arrayLength(&a);
}
"#;

fn device() -> Option<exec::Context> {
    match exec::Context::open(0) {
        Ok(c) => Some(c),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device for kernel execution: {e}"));
            None
        }
    }
}

/// Launch `wgsl` over three bindings laid out `[b0][mark][b1][mark][b2][mark]`
/// in one allocation (each `W` words), with `lens` as the bound lengths in
/// words. Returns the whole allocation afterwards.
fn run(ctx: &exec::Context, name: &str, wgsl: &str, block: u32, n_bindings: usize, lens: &[u64], init: &[u32]) -> Vec<u32> {
    let gen = wgsl_cuda::generate(name, wgsl).expect("generate");
    assert_eq!(gen.bindings.len(), n_bindings);
    let module = ctx.compile(&gen.source, &gen.entry).expect("NVRTC");
    let f = module.function(&gen.entry).expect("entry point");
    let total = 2 * W * n_bindings;
    let mem = ctx.alloc(total * 4).expect("alloc");
    let mut words = vec![MARK; total];
    for b in 0..n_bindings {
        words[2 * W * b..2 * W * b + W].copy_from_slice(&init[W * b..W * (b + 1)]);
    }
    ctx.upload(&mem, &words.iter().flat_map(|w| w.to_ne_bytes()).collect::<Vec<u8>>()).expect("upload");
    let mut args: Vec<u64> = (0..n_bindings).map(|b| mem.device_ptr() + (2 * W * b * 4) as u64).collect();
    args.extend_from_slice(lens);
    ctx.launch_at(&f, (1, 1, 1), (block, 1, 1), &args).expect("launch");
    ctx.sync().expect("sync");
    let mut raw = vec![0u8; total * 4];
    ctx.download(&mem, &mut raw).expect("download");
    raw.chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn out_of_range_accesses_stay_inside_their_binding() {
    let Some(ctx) = device() else { return };
    let a: Vec<f32> = (1..=W).map(|v| v as f32).collect();
    let zeros = vec![0.0f32; W];
    let init: Vec<u32> = [&a, &zeros, &zeros].iter().flat_map(|v| v.iter().map(|x| x.to_bits())).collect();
    let got = run(&ctx, "oob", OOB, 8, 3, &[W as u64; 3], &init);

    let at = |b: usize, i: usize| got[2 * W * b + i];
    for b in 0..3 {
        assert!(
            (0..W).all(|i| got[2 * W * b + W + i] == MARK),
            "binding {b}: the block after it was touched: {:x?}",
            &got[2 * W * b + W..2 * W * b + 2 * W]
        );
    }
    // a[i + 4] for i < 4, then the last element for every index past the end.
    let want_o: Vec<f32> = (0..W).map(|i| a[(i + 4).min(W - 1)]).collect();
    let got_o: Vec<f32> = (0..W).map(|i| f32::from_bits(at(1, i))).collect();
    assert_eq!(got_o, want_o);
    // Every write at i + 8 is out of range and lands on the last element.
    let got_p: Vec<f32> = (0..W).map(|i| f32::from_bits(at(2, i))).collect();
    assert_eq!(got_p, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.0]);
}

#[test]
fn a_shorter_bound_range_is_the_array_the_kernel_sees() {
    let Some(ctx) = device() else { return };
    let a: Vec<f32> = (1..=W).map(|v| v as f32).collect();
    let zeros = vec![0.0f32; W];
    let init: Vec<u32> = [&a, &zeros, &zeros].iter().flat_map(|v| v.iter().map(|x| x.to_bits())).collect();
    // `a` is bound as only its first 6 words, so reads clamp to a[5].
    let got = run(&ctx, "oob_slice", OOB, 8, 3, &[6, W as u64, W as u64], &init);
    let got_o: Vec<f32> = (0..W).map(|i| f32::from_bits(got[2 * W + i])).collect();
    assert_eq!(got_o, [5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
}

#[test]
fn array_length_is_the_bound_range_in_elements() {
    let Some(ctx) = device() else { return };
    // `a` is an array of vec4: 8 words is 2 elements; `o` is the result.
    let init = vec![0u32; 2 * W];
    let got = run(&ctx, "len", LEN, 1, 2, &[W as u64, W as u64], &init);
    assert_eq!(got[2 * W], 2, "arrayLength of an 8-word vec4 array");
}
