// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Numeric gate for the JIT's aggregate IR: vectors, matrices, structs,
//! arrays of aggregates, user function calls and the uniform buffer's
//! host-shareable layout.
//!
//! Swedish Embedded AB implements shader-to-native compilers of this kind for
//! its clients. If your team needs expertise in WGSL, naga IR or Cranelift
//! code generation, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `compile_all.rs` proves every registered kernel translates; this proves an
//! idiomatic, non-scalar kernel computes what WGSL says it computes. Every
//! expected value is worked out by hand (or, for the 4x4 determinant, by an
//! independent f64 Laplace expansion) and every input is small and exact in
//! f32, so the comparisons are bit-for-bit wherever the operation is exact.

use wgsl_cpu::Jit;

/// JIT `src` (entry point `main`), run invocations `[0, n)` single-threaded
/// with `uniform` as the packed uniform words and `bufs` as the storage
/// bindings in binding order, and return the buffers.
fn run(src: &str, uniform: &[u32], mut bufs: Vec<Vec<f32>>, n: u64) -> Vec<Vec<f32>> {
    let jit = Jit::new(&[("k", src)]).unwrap_or_else(|e| panic!("JIT failed: {e}"));
    let ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr() as *mut u8).collect();
    let gx = (n as u32).div_ceil(64).max(1);
    // SAFETY: single-threaded; `uniform`/`bufs` outlive the call, match the
    // kernel's bindings, and every buffer is sized for the indices it writes.
    unsafe { jit.run(0, 0, n, gx, 1, uniform.as_ptr(), ptrs.as_ptr()) };
    bufs
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch: got {got:?}, want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * w.abs().max(1.0), "[{i}]: got {g}, want {w}\n got  {got:?}\n want {want:?}");
    }
}

const HEADER: &str = r#"
struct P { n: u32, k: u32, j: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
"#;

fn kernel(helpers: &str, body: &str) -> String {
    format!(
        "{HEADER}\n{helpers}\n@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
                 @builtin(num_workgroups) nwg: vec3<u32>) {{\n\
             let i = gid.y * (nwg.x * 64u) + gid.x;\n\
             if (i >= p.n) {{ return; }}\n{body}\n}}\n"
    )
}

#[test]
fn vectors_compose_swizzle_broadcast_and_builtins() {
    let src = kernel(
        "",
        r#"
    let a = vec3<f32>(x[0], x[1], x[2]);
    let b = vec3<f32>(x[3], x[4], x[5]);
    let s = a * 2.0 + b - 1.0;
    let c = cross(a, b);
    let d = dot(a, b);
    let l = length(vec2<f32>(x[6], x[7]));
    let nrm = normalize(vec3<f32>(0.0, x[6], x[7]));
    var v = vec4<f32>(0.0);
    v[p.k] = 7.0;
    v.y = a.z;
    let sw = v.zyx;
    let sel = select(a, b, a < vec3<f32>(1.5, 2.5, 2.5));
    let mm = max(abs(b), vec3<f32>(4.5));
    let cl = clamp(b, vec3<f32>(-1.0), vec3<f32>(5.0));
    let uf = vec2<f32>(vec2<u32>(p.k, 5u));
    let neg = -a;
    let o = array<f32, 30>(s.x, s.y, s.z, c.x, c.y, c.z, d, l, nrm.x, nrm.y, nrm.z,
        v.x, v.y, v.z, v.w, sw.x, sw.y, sw.z, sel.x, sel.y, sel.z, mm.x, mm.y, mm.z,
        cl.x, cl.y, cl.z, uf.x, uf.y, v[p.k]);
    for (var t = 0u; t < 30u; t = t + 1u) { y[t] = o[t]; }
    y[30] = neg.x + neg.y * 10.0 + neg.z * 100.0;
    y[31] = select(0.0, 1.0, any(a > b));
    y[32] = select(0.0, 1.0, all(a < b + vec3<f32>(0.0, 8.0, 0.0)));
    "#,
    );
    let x = vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 3.0, 4.0];
    let out = run(&src, &[1, 2, 0], vec![x, vec![0.0; 33]], 1);
    #[rustfmt::skip]
    let want = [
        5.0, -2.0, 11.0,        // a * 2 + b - 1
        27.0, 6.0, -13.0,       // cross
        12.0, 5.0,              // dot, length
        0.0, 0.6, 0.8,          // normalize
        0.0, 3.0, 7.0, 0.0,     // v after v[k] = 7, v.y = a.z
        7.0, 3.0, 0.0,          // v.zyx
        4.0, -5.0, 3.0,         // select(a, b, a < ...)
        4.5, 5.0, 6.0,          // max(abs(b), 4.5)
        4.0, -1.0, 5.0,         // clamp
        2.0, 5.0,               // vec2<f32>(vec2<u32>)
        7.0,                    // v[k] dynamic read
        -321.0,                 // -a
        1.0, 1.0,               // any, all
    ];
    assert_close(&out[1], &want, 1e-6);
}

#[test]
fn matrices_construct_index_multiply_and_determinant() {
    let src = kernel(
        "",
        r#"
    let m = mat3x3<f32>(vec3<f32>(x[0], x[1], x[2]), vec3<f32>(x[3], x[4], x[5]), vec3<f32>(x[6], x[7], x[8]));
    let v = vec3<f32>(x[9], x[10], x[11]);
    let mv = m * v;
    let vm = v * m;
    let mm = m * m;
    let col = m[p.k];
    let m2 = m * 2.0 + m;
    let d2 = determinant(mat2x2<f32>(vec2<f32>(x[12], x[13]), vec2<f32>(x[14], x[15])));
    let m4 = mat4x4<f32>(
        vec4<f32>(x[16], x[17], x[18], x[19]), vec4<f32>(x[20], x[21], x[22], x[23]),
        vec4<f32>(x[24], x[25], x[26], x[27]), vec4<f32>(x[28], x[29], x[30], x[31]));
    y[0] = mv.x; y[1] = mv.y; y[2] = mv.z;
    y[3] = vm.x; y[4] = vm.y; y[5] = vm.z;
    y[6] = determinant(m);
    for (var c = 0u; c < 3u; c = c + 1u) {
        for (var r = 0u; r < 3u; r = r + 1u) { y[7u + c * 3u + r] = mm[c][r]; }
    }
    y[16] = col.x; y[17] = col.y; y[18] = col.z;
    y[19] = m2[2].z;
    y[20] = d2;
    y[21] = determinant(m4);
    y[22] = m[2].y;
    "#,
    );
    #[rustfmt::skip]
    let m4: [f32; 16] = [
        2.0, 1.0, 0.0, 3.0,
        0.0, 3.0, 1.0, -1.0,
        1.0, 0.0, 4.0, 2.0,
        5.0, -2.0, 1.0, 1.0,
    ];
    let mut x = vec![2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 4.0];
    x.extend_from_slice(&m4);
    let out = run(&src, &[1, 1, 0], vec![x, vec![0.0; 23]], 1);
    // Columns c0=(2,0,1), c1=(1,3,0), c2=(0,1,4); v=(1,2,3).
    #[rustfmt::skip]
    let want = [
        4.0, 9.0, 13.0,               // m * v = c0 + 2 c1 + 3 c2
        5.0, 7.0, 14.0,               // v * m = (v.c0, v.c1, v.c2)
        25.0,                         // det, rows [[2,1,0],[0,3,1],[1,0,4]]
        4.0, 1.0, 6.0, 5.0, 9.0, 1.0, 1.0, 7.0, 16.0, // m * m, column-major
        1.0, 3.0, 0.0,                // m[k], k = 1
        12.0,                         // (3 m)[2].z
        10.0,                         // det [[3,2],[1,4]]
        det4(&m4.map(f64::from)) as f32,
        1.0,                          // m[2].y
    ];
    assert_close(&out[1], &want, 1e-6);
}

/// Independent oracle: Laplace expansion in f64 over the column-major
/// entries (the determinant of a matrix equals that of its transpose, so
/// reading columns as rows is fine).
fn det4(m: &[f64; 16]) -> f64 {
    fn det(m: &[f64], n: usize) -> f64 {
        if n == 1 {
            return m[0];
        }
        (0..n)
            .map(|c| {
                let minor: Vec<f64> = (1..n)
                    .flat_map(|r| (0..n).filter(move |&cc| cc != c).map(move |cc| (r, cc)))
                    .map(|(r, cc)| m[r * n + cc])
                    .collect();
                let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
                sign * m[c] * det(&minor, n - 1)
            })
            .sum()
    }
    det(m, 4)
}

#[test]
fn structs_and_inlined_calls_with_early_returns() {
    let helpers = r#"
struct R { uv: vec2<f32>, ok: f32, j: vec3<f32> };
fn twice(a: f32) -> f32 { return a * 2.0; }
fn probe(a: f32, d: vec3<f32>) -> R {
    var o: R;
    o.ok = 0.0;
    if (a < 0.0) { return o; }
    for (var k = 0u; k < 10u; k = k + 1u) {
        if (f32(k) >= a) {
            o.uv = vec2<f32>(f32(k), twice(a));
            o.ok = 1.0;
            return o;
        }
    }
    o.j = d;
    return o;
}
fn ramp(s: f32) -> array<vec2<f32>, 4> {
    var r: array<vec2<f32>, 4>;
    for (var k = 0u; k < 4u; k = k + 1u) { r[k] = vec2<f32>(f32(k) * s, -f32(k)); }
    return r;
}
fn put(at: u32, v: f32) {
    if (v < 0.0) { return; }
    y[at] = v;
}
"#;
    let src = kernel(
        helpers,
        r#"
    let a = x[i];
    let r = probe(a, vec3<f32>(1.0, 2.0, 3.0));
    let o = i * 8u;
    y[o] = r.uv.x;
    y[o + 1u] = r.uv.y;
    y[o + 2u] = r.ok;
    y[o + 3u] = r.j.x;
    y[o + 4u] = r.j.y;
    y[o + 5u] = r.j.z;
    let q = ramp(a);
    y[o + 6u] = q[p.k].x;
    y[o + 7u] = q[i].y;
    put(24u + i, a);
    "#,
    );
    let mut y = vec![0.0; 27];
    y[24..27].fill(99.0);
    let out = run(&src, &[3, 3, 0], vec![vec![-1.0, 2.5, 20.0], y], 3);
    #[rustfmt::skip]
    let want = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0,   // a < 0: the first early return
        3.0, 5.0, 1.0, 0.0, 0.0, 0.0, 7.5, -1.0,   // returned from inside the loop
        0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 60.0, -2.0,  // fell out of the loop
        99.0, 2.5, 20.0,                           // void callee's early return
    ];
    assert_close(&out[1], &want, 0.0);
}

#[test]
fn uniform_nested_struct_layout() {
    let src = r#"
struct Inner { code: u32, s: f32, f: vec4<f32>, v3: vec3<f32>, t: f32 };
struct P { n: u32, k: u32, inner: Inner, m: mat3x3<f32>, arr: array<vec4<f32>, 2>, w: vec2<f32>, last: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let li = p.inner;
    var a = p.inner;
    a.f[p.k] = 100.0;
    let c0 = p.m * vec3<f32>(1.0, 0.0, 0.0);
    let o = array<f32, 30>(f32(li.code), li.s, li.f.x, li.f.y, li.f.z, li.f.w,
        p.inner.v3.x, p.inner.v3.y, p.inner.v3.z, p.inner.t,
        p.m[1].x, p.m[1].y, p.m[1].z, c0.x, c0.y, c0.z,
        p.arr[p.k].x, p.arr[p.k].y, p.arr[p.k].z, p.arr[p.k].w, p.arr[0].w,
        p.w.x, p.w.y, p.last, a.f.x, a.f.y, a.f.z, a.f.w, p.inner.f[p.k], f32(a.code));
    for (var t = 0u; t < 30u; t = t + 1u) { y[t] = o[t]; }
}
"#;
    // WGSL host-shareable layout, worked by hand:
    //   P.n 0, P.k 4, P.inner 16 (Inner aligns to its vec4: 16), size 48:
    //     code +0, s +4, f +16, v3 +32, t +44 (a vec3 is 12 bytes, align 16);
    //   P.m 64 (mat3x3 = 3 columns of stride 16), P.arr 112 (stride 16),
    //   P.w 144 (align 8), P.last 152; size 160.
    let mut u = [0u32; 40];
    let f = |v: f32| v.to_bits();
    u[0] = 1;
    u[1] = 1;
    u[4] = 7;
    u[5] = f(0.5);
    for (w, v) in [(8, 1.0), (9, 2.0), (10, 3.0), (11, 4.0)] {
        u[w] = f(v);
    }
    for (w, v) in [(12, 5.0), (13, 6.0), (14, 7.0), (15, 8.0)] {
        u[w] = f(v);
    }
    for c in 0..3 {
        for r in 0..3 {
            u[16 + c * 4 + r] = f((10 * (c + 1) + r) as f32);
        }
        u[16 + c * 4 + 3] = f(-999.0); // column padding must never be read
    }
    for (w, v) in (28..36).map(|w| (w, (w - 28) as f32 + 0.25)) {
        u[w] = f(v);
    }
    u[36] = f(-1.5);
    u[37] = f(-2.5);
    u[38] = f(9.0);
    let out = run(src, &u, vec![vec![0.0; 30]], 1);
    #[rustfmt::skip]
    let want = [
        7.0, 0.5, 1.0, 2.0, 3.0, 4.0,   // inner.code, s, f
        5.0, 6.0, 7.0, 8.0,             // inner.v3, t
        20.0, 21.0, 22.0,               // m[1]
        10.0, 11.0, 12.0,               // m * (1,0,0) = m[0]
        4.25, 5.25, 6.25, 7.25, 3.25,   // arr[k], arr[0].w
        -1.5, -2.5, 9.0,                // w, last
        1.0, 100.0, 3.0, 4.0,           // a.f after a.f[k] = 100
        2.0,                            // the uniform itself is untouched
        7.0,
    ];
    assert_close(&out[0], &want, 0.0);
}

#[test]
fn dynamic_indexing_of_local_aggregates() {
    let src = kernel(
        "",
        r#"
    var qc: array<vec3<f32>, 3>;
    for (var k = 0u; k < 3u; k = k + 1u) { qc[k] = vec3<f32>(f32(k), f32(k) * 10.0, f32(k) * 100.0); }
    qc[p.k][p.j] = -1.0;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 3u; k = k + 1u) { acc = acc + qc[k]; }
    let t = array<f32, 3>(x[0], 2.0 * x[0], 3.0 * x[0]);
    var e = vec3<f32>(0.0, 0.0, 0.0);
    e[p.j] = 5.0;
    e.x = e.x + 1.0;
    var m = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    m[p.k] = vec2<f32>(-3.0, -4.0);
    m[0][p.k] = 9.0;
    y[0] = acc.x; y[1] = acc.y; y[2] = acc.z;
    y[3] = qc[p.k].z;
    y[4] = qc[2].y;
    y[5] = t[p.j];
    y[6] = e.x; y[7] = e.y; y[8] = e.z;
    y[9] = m[0].x; y[10] = m[0].y; y[11] = m[1].x; y[12] = m[1].y;
    "#,
    );
    let out = run(&src, &[1, 1, 2], vec![vec![1.5], vec![0.0; 13]], 1);
    let want = [3.0, 30.0, 199.0, -1.0, 20.0, 4.5, 1.0, 0.0, 5.0, 1.0, 9.0, -3.0, -4.0];
    assert_close(&out[1], &want, 0.0);
}

/// A kernel-scope constant or `let`-bound literal is one naga expression that
/// every use shares, and naga does not `Emit` it. It must therefore be
/// usable from blocks that do not dominate each other - here first inside a
/// loop body, then in both arms of an `if` after the loop.
#[test]
fn a_shared_constant_is_usable_across_non_dominating_blocks() {
    let src = kernel(
        "const K: f32 = 3.0;",
        r#"
    let c = 2.0;
    for (var k = 0u; k < 2u; k = k + 1u) { y[k] = x[k] * c + K; }
    if (x[0] > 0.0) { y[2] = c * K; } else { y[3] = c - K; }
    if (x[1] > 0.0) { y[4] = c * K; } else { y[5] = c - K; }
    "#,
    );
    let out = run(&src, &[1, 0, 0], vec![vec![1.0, -1.0], vec![0.0; 6]], 1);
    assert_close(&out[1], &[5.0, 1.0, 6.0, 0.0, 0.0, -1.0], 0.0);
}

#[test]
fn vector_math_builtins_match_std() {
    let src = kernel(
        "",
        r#"
    let v = vec3<f32>(x[0], x[1], x[2]);
    let w = vec3<f32>(x[3], x[4], x[5]);
    let e = exp(v);
    let l = log(w);
    let s = sin(v);
    let c = cos(v);
    let q = sqrt(w);
    let at = atan2(v, w);
    let o = array<vec3<f32>, 6>(e, l, s, c, q, at);
    for (var k = 0u; k < 6u; k = k + 1u) {
        y[k * 3u] = o[k].x; y[k * 3u + 1u] = o[k].y; y[k * 3u + 2u] = o[k].z;
    }
    y[18] = atan2(x[0], x[5]);
    "#,
    );
    let v = [0.5f32, -1.25, 2.0];
    let w = [3.0f32, 0.75, 4.5];
    let mut x = v.to_vec();
    x.extend_from_slice(&w);
    let out = run(&src, &[1, 0, 0], vec![x, vec![0.0; 19]], 1);
    let mut want = Vec::new();
    want.extend(v.iter().map(|a| a.exp()));
    want.extend(w.iter().map(|a| a.ln()));
    want.extend(v.iter().map(|a| a.sin()));
    want.extend(v.iter().map(|a| a.cos()));
    want.extend(w.iter().map(|a| a.sqrt()));
    want.extend(v.iter().zip(&w).map(|(a, b)| a.atan2(*b)));
    want.push(v[0].atan2(w[2]));
    assert_close(&out[1], &want, 1e-6);
}

/// A work-group kernel with a function-scope array and vector values on the
/// segment before its one barrier - the shape of `splat_ray_camera_grad`.
#[test]
fn workgroup_kernel_with_local_array_and_vectors() {
    let src = r#"
struct P { n_wg: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
var<workgroup> acc: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = lid.x;
    var g: array<f32, 2>;
    for (var k = 0u; k < 2u; k = k + 1u) { g[k] = 0.0; }
    g[0] = f32(t + wid.x);
    g[1] = 1.0;
    let v = vec2<f32>(g[0], g[1]) * 2.0;
    acc[t] = v.x;
    acc[64u + t] = v.y;
    workgroupBarrier();
    if (t < 2u) {
        var s = 0.0;
        for (var j = 0u; j < 64u; j = j + 1u) { s = s + acc[t * 64u + j]; }
        y[wid.x * 2u + t] = s;
    }
}
"#;
    let jit = Jit::new(&[("k", src)]).unwrap_or_else(|e| panic!("JIT failed: {e}"));
    assert_eq!(jit.workgroup_size(0), Some(64));
    let mut y = vec![0.0f32; 6];
    let ptrs = [y.as_mut_ptr() as *mut u8];
    let u = [3u32];
    // SAFETY: three whole work-groups, single-threaded; `y` holds 2 per group.
    unsafe { jit.run(0, 0, 3 * 64, 3, 1, u.as_ptr(), ptrs.as_ptr()) };
    // sum_t 2 (t + w) = 2 (2016 + 64 w); sum_t 2 = 128.
    assert_eq!(y, vec![4032.0, 128.0, 4160.0, 128.0, 4288.0, 128.0]);
}

/// The real kernel a function-scope array in a work-group kernel exists for:
/// `matmul_gemv_reg` keeps its accumulators in one, and its contract is
/// bit-identity with `matmul_gemv` (`gpu_core::upgrade`). Both run on the JIT
/// here over several work-groups, with rows past `p.m` in the specialisation.
#[test]
fn matmul_gemv_reg_is_bit_identical_to_matmul_gemv_on_the_jit() {
    let reg = kernels::template::specialize(kernels::src("matmul_gemv_reg"), &[("MREG", 4)])
        .expect("specialise MREG");
    let jit = Jit::new(&[("gemv", kernels::src("matmul_gemv")), ("gemv_reg", &reg)])
        .unwrap_or_else(|e| panic!("JIT failed: {e}"));
    let (m, k, n) = (3usize, 150usize, 5usize);
    let x: Vec<f32> = (0..m * k).map(|i| ((i * 37 + 11) % 251) as f32 / 97.0 - 1.3).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i * 53 + 5) % 241) as f32 / 89.0 - 1.4).collect();
    let uniform = [m as u32, k as u32, n as u32];
    let outputs: Vec<Vec<f32>> = (0..2)
        .map(|kind| {
            let mut x = x.clone();
            let mut w = w.clone();
            let mut out = vec![0.0f32; m * n];
            let bufs = [x.as_mut_ptr() as *mut u8, w.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];
            // SAFETY: one work-group per output column, single-threaded.
            unsafe { jit.run(kind, 0, (n * 64) as u64, n as u32, 1, uniform.as_ptr(), bufs.as_ptr()) };
            out
        })
        .collect();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&outputs[1]), bits(&outputs[0]));
    let want: f32 = (0..k).map(|kk| x[kk] * w[kk]).sum();
    assert!((outputs[0][0] - want).abs() < 1e-3, "gemv[0,0] = {}, want {want}", outputs[0][0]);
}
