// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vector semantics of the generated tier, held to an exact host reference.
//!
//! Swedish Embedded AB implements portable GPU compute stacks and the golden
//! tests that hold a translated kernel to the answer of the source it came
//! from, for its clients. If your team needs expertise in proving a kernel
//! translator correct rather than assuming it, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The translator represents a vector as a list of scalar components and
//! applies each operation lane by lane. What can go wrong is therefore not the
//! arithmetic (the scalar paths are already pinned) but the *layout and
//! indexing* around it: a `vec3` array element is four words wide, not three; a
//! component can be picked at run time; a swizzle must read the right lanes.
//! Each kernel here is chosen so that one of those mistakes changes the answer,
//! and every comparison is bit-exact, because with FMA contraction disabled the
//! device and the host perform the same IEEE operations in the same order.
//!
//! Skip-if-absent: a correctness gate, never a benchmark.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const N: usize = 200;

/// `vec4` arithmetic: lane-wise `*`, `+`, `-`, `/`, and a constant vector.
const ARITH: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = a[i];
    let y = b[i];
    o[i] = x * y + vec4<f32>(1.0, 2.0, 3.0, 4.0) - x / y;
}
"#;

/// `vec3` arrays have a 16-byte stride; `dot` sums left to right; `cross`.
const VEC3: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<vec3<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    d[i] = dot(a[i], b[i]);
    c[i] = cross(a[i], b[i]);
}
"#;

/// A component chosen at run time, written into workgroup memory and read back
/// through a swizzle; a `select` on a per-lane condition; a composed `vec2<u32>`.
const DYNAMIC: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<vec2<u32>>;
var<workgroup> tile: array<vec4<f32>, 16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let t = li.x;
    if (t < 64u) {
        // `t % 4u` is not a constant: the component is selected at run time.
        tile[t / 4u][t % 4u] = f32(t) + 0.5;
    }
    workgroupBarrier();
    let v = tile[(t / 4u) % 16u];
    let w = v.wzyx;
    let m = select(v, w, v < w);
    if (gid.x < p.n) {
        o[gid.x] = m;
        q[gid.x] = vec2<u32>(gid.x * 3u, select(7u, 9u, all(v == v)));
    }
}
"#;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(&[("arith", ARITH), ("vec3", VEC3), ("dynamic", DYNAMIC)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// A deterministic spread of non-trivial floats (never zero: used as a divisor).
fn sample(i: usize, salt: usize) -> f32 {
    let x = ((i * 2654435761usize + salt * 40503) % 2001) as f32 / 100.0 - 10.0;
    if x == 0.0 { 0.25 } else { x }
}

#[test]
fn vec4_arithmetic_is_lane_wise_and_exact() {
    let Some(b) = backend() else { return };
    let a: Vec<f32> = (0..N * 4).map(|i| sample(i, 1)).collect();
    let c: Vec<f32> = (0..N * 4).map(|i| sample(i, 2)).collect();
    let (ba, bc) = (b.storage_init("a", &a), b.storage_init("b", &c));
    let out = b.storage((N * 4) as u64);
    b.submit(&[], &[b.step(0, &[&ba, &bc, &out], &[N as u32], N as u32)]);
    let got = b.read(&out, N * 4);
    let k = [1.0f32, 2.0, 3.0, 4.0];
    for i in 0..N * 4 {
        let want = (a[i] * c[i] + k[i % 4]) - a[i] / c[i];
        assert_eq!(got[i].to_bits(), want.to_bits(), "lane {i}: {} vs {want}", got[i]);
    }
}

#[test]
fn vec3_arrays_use_a_four_word_stride_and_dot_and_cross_are_exact() {
    let Some(b) = backend() else { return };
    // Host layout: each vec3 occupies four words, the fourth unused.
    let mut a = vec![0f32; N * 4];
    let mut c = vec![0f32; N * 4];
    for i in 0..N {
        for k in 0..3 {
            a[i * 4 + k] = sample(i * 3 + k, 3);
            c[i * 4 + k] = sample(i * 3 + k, 4);
        }
        a[i * 4 + 3] = f32::NAN; // padding must never be read
        c[i * 4 + 3] = f32::NAN;
    }
    let (ba, bc) = (b.storage_init("a", &a), b.storage_init("b", &c));
    let (d, cross) = (b.storage(N as u64), b.storage((N * 4) as u64));
    b.submit(&[], &[b.step(1, &[&ba, &bc, &d, &cross], &[N as u32], N as u32)]);
    let (got_d, got_c) = (b.read(&d, N), b.read(&cross, N * 4));
    for i in 0..N {
        let (x, y) = (&a[i * 4..i * 4 + 3], &c[i * 4..i * 4 + 3]);
        let dot = (x[0] * y[0] + x[1] * y[1]) + x[2] * y[2];
        let want_c = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
        assert_eq!(got_d[i].to_bits(), dot.to_bits(), "dot[{i}]");
        for k in 0..3 {
            assert_eq!(got_c[i * 4 + k].to_bits(), want_c[k].to_bits(), "cross[{i}][{k}]");
        }
    }
}

#[test]
fn a_run_time_component_index_a_swizzle_and_a_lane_select_agree_with_the_host() {
    let Some(b) = backend() else { return };
    let out = b.storage((N * 4) as u64);
    let q = b.storage((N * 2) as u64);
    // 200 invocations over 64-wide groups: a tail group exercises the bounds check.
    b.submit(&[], &[b.step(2, &[&out, &q], &[N as u32], N as u32)]);
    let (got, gq) = (b.read(&out, N * 4), b.read(&q, N * 2));
    for g in 0..N {
        let t = g % 64;
        // The tile holds f32(t) + 0.5 at flat position t; element e is lanes 4e..4e+3.
        let e = (t / 4) % 16;
        let v: Vec<f32> = (0..4).map(|k| (e * 4 + k) as f32 + 0.5).collect();
        let w = [v[3], v[2], v[1], v[0]];
        for k in 0..4 {
            let want = if v[k] < w[k] { w[k] } else { v[k] };
            assert_eq!(got[g * 4 + k].to_bits(), want.to_bits(), "invocation {g} lane {k}");
        }
        assert_eq!(gq[g * 2].to_bits(), (g * 3) as u32, "q[{g}].x");
        assert_eq!(gq[g * 2 + 1].to_bits(), 9u32, "q[{g}].y");
    }
}
