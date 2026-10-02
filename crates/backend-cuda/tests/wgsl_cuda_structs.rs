// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Struct semantics of the generated tier, held to an exact host reference.
//!
//! Swedish Embedded AB implements portable GPU compute stacks and the golden
//! tests that hold a translated kernel to the answer of the source it came
//! from, for its clients. If your team needs expertise in bringing a shader
//! catalogue with structured data to CUDA without hand-porting each kernel,
//! you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The camera and splat kernels pass their parameters as nested structs (a
//! `Lens` inside a `View`, an array of cameras), carry `var` structs through
//! helper functions, and keep records as struct arrays in storage. What can go
//! wrong is the *layout* around them, never the arithmetic: a member read at a
//! C++ offset instead of the WGSL one, a `vec3` member that is not four words
//! wide, a whole-struct store that drops the padding, a copy that aliases.
//! Each kernel here is chosen so that one of those mistakes changes the answer,
//! and every comparison is bit-exact: with FMA contraction disabled the device
//! and the host perform the same IEEE operations in the same order.
//!
//! Skip-if-absent: a correctness gate, never a benchmark.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const N: usize = 70;

/// A struct inside the uniform block, a struct-typed `var`, structs passed to
/// and returned from an inlined helper, a struct built with a constructor and
/// one left to zero-initialisation, and an array member chosen at run time.
const NESTED: &str = r#"
struct Inner { k: u32, scale: f32, pad0: u32, pad1: u32, v: vec4<f32> };
struct P { n: u32, bias: f32, inner: Inner, tail: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

struct Acc { sum: f32, hits: u32, dir: vec3<f32> };
struct Hist { c: array<f32, 4>, last: u32 };

fn advance(a: Acc, x: f32) -> Acc {
    var r = a;
    r.sum = a.sum + x * p.inner.scale;
    r.hits = a.hits + 1u;
    r.dir = a.dir + vec3<f32>(x, p.inner.v.y, p.inner.v.w);
    return r;
}
fn bump(h: Hist, b: u32, x: f32) -> Hist {
    var r = h;
    r.c[b % 4u] = r.c[b % 4u] + x;
    r.last = b;
    return r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    var acc = Acc(p.bias, 0u, vec3<f32>(0.0, 0.0, 0.0));
    var hist: Hist;
    for (var k = 0u; k < p.inner.k; k = k + 1u) {
        acc = advance(acc, f32(i + k));
        hist = bump(hist, i + k, 1.0);
    }
    let base = i * 12u;
    o[base] = acc.sum;
    o[base + 1u] = f32(acc.hits);
    o[base + 2u] = acc.dir.x;
    o[base + 3u] = acc.dir.y;
    o[base + 4u] = acc.dir.z;
    o[base + 5u] = p.tail;
    o[base + 6u] = hist.c[0];
    o[base + 7u] = hist.c[1];
    o[base + 8u] = hist.c[2];
    o[base + 9u] = hist.c[3];
    o[base + 10u] = f32(hist.last);
}
"#;

/// An array of structs in the uniform block indexed at run time, a struct
/// loaded whole out of it and out of a plain member, and a local array of
/// structs written through whole-element and member stores.
const UNIFORM_ARRAY: &str = r#"
struct Cam { r0: vec4<f32>, s: f32, id: u32, off: vec2<f32> };
struct Q { n: u32, pad0: u32, pad1: u32, pad2: u32, cams: array<Cam, 4>, ext: Cam };
@group(0) @binding(0) var<uniform> p: Q;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

fn pick(c: Cam, x: f32) -> f32 {
    return c.r0.x * x + c.s + f32(c.id) + c.off.y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let c = p.cams[i % 4u];
    let e = p.ext;
    var loc: array<Cam, 3>;
    loc[i % 3u] = c;
    loc[(i + 1u) % 3u].s = e.s;
    o[i * 4u] = pick(c, f32(i));
    o[i * 4u + 1u] = pick(loc[i % 3u], 2.0);
    o[i * 4u + 2u] = loc[(i + 1u) % 3u].s;
    o[i * 4u + 3u] = pick(p.ext, 1.0) + loc[(i + 2u) % 3u].r0.w;
}
"#;

/// Records in storage: read whole, written whole, and written member by member.
const STORAGE: &str = r#"
struct Rec { a: f32, b: u32, c: vec2<f32>, d: vec4<f32> };
struct P { n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<Rec>;
@group(0) @binding(2) var<storage, read_write> dst: array<Rec>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let r = src[i];
    dst[i] = Rec(r.a * 2.0, r.b + 1u, vec2<f32>(r.c.y, r.c.x), r.d * 0.5);
    dst[i].b = dst[i].b + 10u;
    dst[i].c.x = dst[i].c.x + 1.0;
    o[i] = f32(arrayLength(&src)) + src[i].d.z;
}
"#;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(&[("nested", NESTED), ("uniform_array", UNIFORM_ARRAY), ("storage", STORAGE)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// A deterministic spread of non-trivial floats.
fn sample(i: usize, salt: usize) -> f32 {
    ((i * 2654435761usize + salt * 40503) % 2001) as f32 / 100.0 - 10.0
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: {g} vs {w}");
    }
}

#[test]
fn nested_uniform_structs_struct_locals_and_struct_helpers_agree_with_the_host() {
    let Some(b) = backend() else { return };
    const K: u32 = 5;
    let (bias, scale, tail) = (0.75f32, 1.5f32, 42.0f32);
    let v = [sample(1, 1), sample(2, 1), sample(3, 1), sample(4, 1)];
    // WGSL layout: n@0 bias@4 | inner@16 {k@16 scale@20 pad pad v@32} | tail@48, size 64.
    let mut params = [0u32; 16];
    params[0] = N as u32;
    params[1] = bias.to_bits();
    params[4] = K;
    params[5] = scale.to_bits();
    for c in 0..4 {
        params[8 + c] = v[c].to_bits();
    }
    params[12] = tail.to_bits();
    let out = b.storage((N * 12) as u64);
    b.submit(&[], &[b.step(0, &[&out], &params, N as u32)]);
    let got = b.read(&out, N * 12);

    let mut want = vec![0f32; N * 12];
    for i in 0..N {
        let (mut sum, mut hits, mut dir) = (bias, 0u32, [0f32; 3]);
        let mut c = [0f32; 4];
        let mut last = 0u32;
        for k in 0..K as usize {
            let x = (i + k) as f32;
            sum += x * scale;
            hits += 1;
            dir = [dir[0] + x, dir[1] + v[1], dir[2] + v[3]];
            c[(i + k) % 4] += 1.0;
            last = (i + k) as u32;
        }
        want[i * 12..i * 12 + 11].copy_from_slice(&[sum, hits as f32, dir[0], dir[1], dir[2], tail, c[0], c[1], c[2], c[3], last as f32]);
    }
    assert_bits(&got, &want, "nested");
}

#[test]
fn a_uniform_array_of_structs_and_a_local_array_of_structs_agree_with_the_host() {
    let Some(b) = backend() else { return };
    // Cam: r0@0 s@16 id@20 off@24, size 32. Q: n@0, cams@16 (stride 32), ext@144.
    let cam = |salt: usize| -> ([f32; 4], f32, u32, [f32; 2]) {
        ([sample(1, salt), sample(2, salt), sample(3, salt), sample(4, salt)], sample(5, salt), 7 * salt as u32, [sample(6, salt), sample(7, salt)])
    };
    let mut params = [0u32; 44];
    params[0] = N as u32;
    let put = |params: &mut [u32; 44], word: usize, c: &([f32; 4], f32, u32, [f32; 2])| {
        for k in 0..4 {
            params[word + k] = c.0[k].to_bits();
        }
        params[word + 4] = c.1.to_bits();
        params[word + 5] = c.2;
        params[word + 6] = c.3[0].to_bits();
        params[word + 7] = c.3[1].to_bits();
    };
    let cams: Vec<_> = (1..=4).map(cam).collect();
    for (j, c) in cams.iter().enumerate() {
        put(&mut params, 4 + j * 8, c);
    }
    let ext = cam(9);
    put(&mut params, 36, &ext);

    let out = b.storage((N * 4) as u64);
    b.submit(&[], &[b.step(1, &[&out], &params, N as u32)]);
    let got = b.read(&out, N * 4);

    let pick = |c: &([f32; 4], f32, u32, [f32; 2]), x: f32| c.0[0] * x + c.1 + c.2 as f32 + c.3[1];
    let mut want = vec![0f32; N * 4];
    for i in 0..N {
        let c = &cams[i % 4];
        // loc[i % 3] = c; loc[(i + 1) % 3].s = ext.s; the third stays zero.
        let at_i = c;
        let zero = ([0f32; 4], 0f32, 0u32, [0f32; 2]);
        want[i * 4] = pick(c, i as f32);
        want[i * 4 + 1] = pick(at_i, 2.0);
        want[i * 4 + 2] = ext.1;
        want[i * 4 + 3] = pick(&ext, 1.0) + zero.0[3];
    }
    assert_bits(&got, &want, "uniform array");
}

#[test]
fn struct_records_in_storage_are_read_and_written_at_their_wgsl_layout() {
    let Some(b) = backend() else { return };
    // Rec: a@0 b@4 c@8 d@16, size 32 = 8 words.
    let mut src = vec![0f32; N * 8];
    for i in 0..N {
        src[i * 8] = sample(i, 1);
        src[i * 8 + 1] = f32::from_bits(i as u32 * 3);
        src[i * 8 + 2] = sample(i, 2);
        src[i * 8 + 3] = sample(i, 3);
        for k in 0..4 {
            src[i * 8 + 4 + k] = sample(i * 4 + k, 4);
        }
    }
    let src_b = b.storage_init("src", &src);
    let dst_b = b.storage((N * 8) as u64);
    let out = b.storage(N as u64);
    b.submit(&[], &[b.step(2, &[&src_b, &dst_b, &out], &[N as u32], N as u32)]);
    let (dst, o) = (b.read(&dst_b, N * 8), b.read(&out, N));

    let mut want = vec![0f32; N * 8];
    let mut want_o = vec![0f32; N];
    for i in 0..N {
        let r = &src[i * 8..i * 8 + 8];
        want[i * 8] = r[0] * 2.0;
        want[i * 8 + 1] = f32::from_bits(f32::to_bits(r[1]) + 1 + 10);
        want[i * 8 + 2] = r[3] + 1.0;
        want[i * 8 + 3] = r[2];
        for k in 0..4 {
            want[i * 8 + 4 + k] = r[4 + k] * 0.5;
        }
        want_o[i] = N as f32 + r[6];
    }
    assert_bits(&dst, &want, "records");
    assert_bits(&o, &want_o, "length + member");
}
