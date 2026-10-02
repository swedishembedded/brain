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

/// Helper functions: an early `return` of a `vec2`, a helper that reads the
/// kernel's own storage buffer, a scalar helper, and a `return` from inside a
/// loop. They are inlined, so each call must get its own locals and labels.
const CALLS: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
fn pick(col: u32, ncols: u32) -> vec2<f32> {
    if (col >= ncols) { return vec2<f32>(0.0, 0.0); }
    let w = t[col];
    return vec2<f32>(f32(w & 0xFFu), f32(w >> 8u));
}
fn twice(x: f32) -> f32 { return x + x; }
fn sum_to(n: u32) -> f32 {
    var acc = 0.0;
    for (var k = 0u; k < n; k = k + 1u) {
        if (k == 5u) { return acc * 10.0; }
        acc = acc + f32(k);
    }
    return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let a = pick(i, 150u);
    let b = pick(i + 1u, 150u);
    o[i] = twice(a.x) + a.y + sum_to(i % 8u) - twice(b.y);
}
"#;

/// `mat3x3` built from column vectors, `m * v`, `v * m`, `m * m`, a column of
/// a product, `transpose` and `determinant`. Each sums in the order the CPU
/// tier does, so the comparison below is bit-exact.
const MATRIX: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = a[i];
    let y = b[i];
    let m = mat3x3<f32>(x, y, cross(x, y));
    let t = mat3x3<f32>(y, x, x + y);
    let mv = m * y;
    let vm = y * m;
    let mm = m * t;
    let tr = transpose(m);
    let base = i * 16u;
    o[base] = mv.x;
    o[base + 1u] = mv.y;
    o[base + 2u] = mv.z;
    o[base + 3u] = vm.x;
    o[base + 4u] = vm.y;
    o[base + 5u] = vm.z;
    o[base + 6u] = determinant(m);
    o[base + 7u] = mm[1].x;
    o[base + 8u] = mm[2].z;
    o[base + 9u] = tr[0].y;
    o[base + 10u] = (mm * x).y;
}
"#;

/// A workgroup scalar (zero at entry), and `workgroupUniformLoad` of it and of
/// array elements, the loaded values steering a loop every thread runs.
const UNIFORM_LOAD: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
var<workgroup> flag: f32;
var<workgroup> span: array<u32, 2>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    if (li.x == 0u) {
        flag = flag + 3.0 + f32(wg.x);
        span[0] = 1u + wg.x;
        span[1] = 4u;
    }
    let f = workgroupUniformLoad(&flag);
    let lo = workgroupUniformLoad(&span[0]);
    let hi = workgroupUniformLoad(&span[1]);
    var acc = 0.0;
    for (var k = lo; k < hi; k = k + 1u) { acc = acc + f * f32(k); }
    if (gid.x < p.n) { o[gid.x] = acc + f32(li.x); }
}
"#;

/// The inverse trigonometric and angle-unit builtins.
const TRIG: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    o[i * 5u] = acos(x[i]);
    o[i * 5u + 1u] = asin(x[i]);
    o[i * 5u + 2u] = atan(x[i]);
    o[i * 5u + 3u] = degrees(x[i]);
    o[i * 5u + 4u] = radians(x[i]);
}
"#;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(&[
        ("arith", ARITH),
        ("vec3", VEC3),
        ("dynamic", DYNAMIC),
        ("calls", CALLS),
        ("matrix", MATRIX),
        ("uniform_load", UNIFORM_LOAD),
        ("trig", TRIG),
    ]) {
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

#[test]
fn inlined_helper_functions_return_early_loop_and_read_storage_correctly() {
    let Some(b) = backend() else { return };
    let table: Vec<u32> = (0..150).map(|i| (i as u32 * 37 + 11) & 0xFFFF).collect();
    let words: Vec<f32> = table.iter().map(|w| f32::from_bits(*w)).collect();
    let t = b.storage_init("t", &words);
    let out = b.storage(N as u64);
    b.submit(&[], &[b.step(3, &[&t, &out], &[N as u32], N as u32)]);
    let got = b.read(&out, N);

    let pick = |col: usize| -> (f32, f32) {
        if col >= 150 {
            (0.0, 0.0)
        } else {
            let w = table[col];
            ((w & 0xFF) as f32, (w >> 8) as f32)
        }
    };
    let sum_to = |n: usize| -> f32 {
        let mut acc = 0.0f32;
        for k in 0..n {
            if k == 5 {
                return acc * 10.0;
            }
            acc += k as f32;
        }
        acc
    };
    for i in 0..N {
        let (a, bb) = (pick(i), pick(i + 1));
        let want = (((a.0 + a.0) + a.1) + sum_to(i % 8)) - (bb.1 + bb.1);
        assert_eq!(got[i].to_bits(), want.to_bits(), "element {i}: {} vs {want}", got[i]);
    }
}

/// A 3x3 matrix as the host sees it: `m[c][r]`, column `c`, row `r`.
type M3 = [[f32; 3]; 3];

fn mat_vec(m: &M3, v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| (m[0][r] * v[0] + m[1][r] * v[1]) + m[2][r] * v[2])
}

#[test]
fn matrix_products_transpose_and_determinant_agree_with_the_cpu_tier_bit_for_bit() {
    let Some(b) = backend() else { return };
    let mut a = vec![0f32; N * 4];
    let mut c = vec![0f32; N * 4];
    for i in 0..N {
        for k in 0..3 {
            a[i * 4 + k] = sample(i * 3 + k, 5);
            c[i * 4 + k] = sample(i * 3 + k, 6);
        }
    }
    let (ba, bc) = (b.storage_init("a", &a), b.storage_init("b", &c));
    let out = b.storage((N * 16) as u64);
    b.submit(&[], &[b.step(4, &[&ba, &bc, &out], &[N as u32], N as u32)]);
    let got = b.read(&out, N * 16);

    for i in 0..N {
        let x = [a[i * 4], a[i * 4 + 1], a[i * 4 + 2]];
        let y = [c[i * 4], c[i * 4 + 1], c[i * 4 + 2]];
        let cross = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
        let m: M3 = [x, y, cross];
        let t: M3 = [y, x, [x[0] + y[0], x[1] + y[1], x[2] + y[2]]];
        let mv = mat_vec(&m, y);
        let vm: [f32; 3] = std::array::from_fn(|col| (y[0] * m[col][0] + y[1] * m[col][1]) + y[2] * m[col][2]);
        let mm: M3 = std::array::from_fn(|col| mat_vec(&m, t[col]));
        // Cofactor expansion down the first column, `e(r, c) = m[c][r]`.
        let e = |r: usize, col: usize| m[col][r];
        let t0 = e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1));
        let t1 = e(1, 0) * (e(0, 1) * e(2, 2) - e(0, 2) * e(2, 1));
        let t2 = e(2, 0) * (e(0, 1) * e(1, 2) - e(0, 2) * e(1, 1));
        let det = (t0 - t1) + t2;
        let want = [mv[0], mv[1], mv[2], vm[0], vm[1], vm[2], det, mm[1][0], mm[2][2], m[1][0], mat_vec(&mm, x)[1]];
        for (k, w) in want.iter().enumerate() {
            assert_eq!(got[i * 16 + k].to_bits(), w.to_bits(), "invocation {i} output {k}: {} vs {w}", got[i * 16 + k]);
        }
    }
}

#[test]
fn workgroup_scalars_start_at_zero_and_a_uniform_load_publishes_the_write() {
    let Some(b) = backend() else { return };
    let out = b.storage(N as u64);
    b.submit(&[], &[b.step(5, &[&out], &[N as u32], N as u32)]);
    let got = b.read(&out, N);
    for g in 0..N {
        let wg = (g / 64) as u32;
        let f = 3.0f32 + wg as f32;
        let mut acc = 0.0f32;
        for k in 1 + wg..4 {
            acc += f * k as f32;
        }
        let want = acc + (g % 64) as f32;
        assert_eq!(got[g].to_bits(), want.to_bits(), "invocation {g}: {} vs {want}", got[g]);
    }
}

#[test]
fn inverse_trigonometry_and_angle_units_agree_with_the_host() {
    let Some(b) = backend() else { return };
    let x: Vec<f32> = (0..N).map(|i| (i as f32 / (N - 1) as f32) * 2.0 - 1.0).collect();
    let bx = b.storage_init("x", &x);
    let out = b.storage((N * 5) as u64);
    b.submit(&[], &[b.step(6, &[&bx, &out], &[N as u32], N as u32)]);
    let got = b.read(&out, N * 5);
    for (i, &v) in x.iter().enumerate() {
        // The device's transcendental functions are within a few ulp of the host's.
        for (k, want) in [v.acos(), v.asin(), v.atan()].into_iter().enumerate() {
            assert!((got[i * 5 + k] - want).abs() <= 4e-6, "element {i} function {k}: {} vs {want}", got[i * 5 + k]);
        }
        // A unit conversion is one multiply by the f32 constant: exact.
        assert_eq!(got[i * 5 + 3].to_bits(), (v * 1.0f32.to_degrees()).to_bits(), "degrees({v})");
        assert_eq!(got[i * 5 + 4].to_bits(), (v * 1.0f32.to_radians()).to_bits(), "radians({v})");
    }
}
