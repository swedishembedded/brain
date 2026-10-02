// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `matmul_i8_gemv` kernel (`kernels_cuda`,
//! `cu/matmul_i8_gemv.cu`) that `gpu_core::native_upgrade` substitutes for the
//! WGSL decode GEMV on a CUDA device.
//!
//! Swedish Embedded AB implements bit-exact native kernels for quantised LLM
//! decode. If your team needs expertise in replacing a generated kernel with a
//! hand-written one without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The substitution is invisible to every caller, so the claim it has to hold
//! is the one `i8_gemv_reg_upgrade.rs` holds for the WGSL pair: the RAW BITS
//! are identical to the WGSL tier. That is stronger than a tolerance and it is
//! achievable, because the native kernel keeps the WGSL kernel's 64 virtual
//! accumulators and its fold order. The reference here is `matmul_i8_gemv_ref`
//! - the same source under a name the upgrade table does not know - so both
//! sides run on the same device, from the same inputs, differing in which
//! kernel ran.
//!
//! Shapes cover what the vector-load path can get wrong: `kg` that is not a
//! whole number of the kernel's 64-word stride (tails), `n` that is not a
//! multiple of the block's weight rows, every `m` the kernel serves, weight
//! and activation bindings that start at a word offset which is not 16-byte
//! aligned (the scalar-load path), output windows inside larger buffers, and
//! the real Qwen3.8-27B projection widths.

use backend_api::select::Dtype;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] =
    &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV), ("matmul_i8_gemv_ref", kernels::MATMUL_I8_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// Rows of x the native kernel serves per weight pass.
const NATIVE_ROWS: u32 = 8;
/// Packed words per weight-scale group.
const WPG: usize = 8;
/// Written around every window so an out-of-window write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

/// Deterministic signed bytes, full range including -128 and 127, packed four
/// per word the way `dot4I8Packed` reads them.
fn packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                let b = (r.next_u32() % 256) as u8;
                w |= u32::from(b) << (8 * lane);
            }
            w
        })
        .collect()
}

fn scales(n: usize, seed: u64, lo: f32, span: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| lo + (r.next_u32() % 1000) as f32 * span).collect()
}

/// A buffer of `window + lead + tail` words with the window at `lead`,
/// surrounded by [`SENTINEL`] bits.
struct Windowed {
    buf: backend_api::DeviceBuffer,
    lead: u64,
    len: usize,
    total: usize,
}

impl Windowed {
    fn new(gpu: &Gpu, words: &[u32], lead: u64) -> Windowed {
        let total = words.len() + lead as usize + 7;
        let buf = gpu.storage(total as u64);
        gpu.write(&buf, &vec![SENTINEL; total]);
        gpu.write_at(&buf, lead, words);
        Windowed { buf, lead, len: words.len(), total }
    }
    fn range(&self) -> (u64, u64) {
        (self.lead, self.len as u64)
    }
    fn read_bits(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.read(&self.buf, self.total).iter().map(|f| f.to_bits()).collect()
    }
    fn window(&self, bits: &[u32]) -> Vec<u32> {
        bits[self.lead as usize..self.lead as usize + self.len].to_vec()
    }
}

struct Case {
    m: u32,
    kg: u32,
    n: u32,
    /// Leading words before each window: x, w, sx, sw, out.
    lead: [u64; 5],
}

/// Run `case` through both slots and return `(native-path bits, reference
/// bits)` of the output window, asserting the words around each window kept
/// their sentinel.
fn run(gpu: &Gpu, c: &Case) -> (Vec<u32>, Vec<u32>) {
    let (m, kg, n) = (c.m as usize, c.kg as usize, c.n as usize);
    let ng = kg / WPG;
    let seed = u64::from(c.m) * 31 + u64::from(c.kg) * 7 + u64::from(c.n);
    let x = Windowed::new(gpu, &packed(m * kg, seed + 1), c.lead[0]);
    let w = Windowed::new(gpu, &packed(n * kg, seed + 2), c.lead[1]);
    let sx = Windowed::new(gpu, &f32_words(&scales(m, seed + 3, 1e-3, 1e-5)), c.lead[2]);
    let sw = Windowed::new(gpu, &f32_words(&scales(n * ng, seed + 4, 1e-3, 1e-5)), c.lead[3]);
    let a = Windowed::new(gpu, &vec![SENTINEL; m * n], c.lead[4]);
    let b = Windowed::new(gpu, &vec![SENTINEL; m * n], c.lead[4]);

    let params = [c.m, c.kg, c.n];
    let dispatch = |kind: usize, o: &Windowed| {
        gpu.dispatch_sliced(
            kind,
            &[&x.buf, &w.buf, &sx.buf, &sw.buf, &o.buf],
            &[x.range(), w.range(), sx.range(), sw.range(), o.range()],
            &params,
            Dispatch::Workgroups(c.n),
        )
    };
    gpu.submit(&[], &[dispatch(K_GEMV, &a), dispatch(K_REF, &b)]);
    gpu.poll_wait();

    for (name, o) in [("native path", &a), ("reference", &b)] {
        let bits = o.read_bits(gpu);
        let (lo, hi) = (o.lead as usize, o.lead as usize + o.len);
        assert!(
            bits[..lo].iter().chain(&bits[hi..]).all(|&v| v == SENTINEL),
            "{name} wrote outside its output window at m={} kg={} n={} lead={:?}",
            c.m,
            c.kg,
            c.n,
            c.lead
        );
    }
    for (name, win) in [("x", &x), ("w", &w), ("sx", &sx), ("sw", &sw)] {
        let bits = win.read_bits(gpu);
        assert!(
            bits[..win.lead as usize].iter().chain(&bits[win.lead as usize + win.len..]).all(|&v| v == SENTINEL),
            "an input window {name} was modified"
        );
    }
    (a.window(&a.read_bits(gpu)), b.window(&b.read_bits(gpu)))
}

fn f32_words(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// The redirect fired where it should and nowhere else. Without this the
/// identity tests below would pass trivially on a handle that never left the
/// WGSL tier.
#[test]
fn the_native_kernel_is_selected_exactly_for_the_shapes_it_serves() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("matmul_i8_gemv needs workgroup reductions and a packed int8 dot");
        return;
    }
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_GEMV, &[1, 1280, 1024]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for m in 1..=NATIVE_ROWS {
        assert_eq!(gpu.native_kernel_for(K_GEMV, &[m, 1280, 1024]), Some("matmul_i8_gemv"), "m={m}");
    }
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[NATIVE_ROWS + 1, 1280, 1024]), None, "past one tile keeps the WGSL ladder");
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[1, 1284, 1024]), None, "K not a whole number of scale groups");
    // The reference alias is deliberately not upgraded; it is what makes the
    // comparison a comparison.
    assert_eq!(gpu.native_kernel_for(K_REF, &[1, 1280, 1024]), None);
    // The registry is what says which tier this kernel reads.
    assert!(kernels_cuda::find(backend_api::select::Op::MatMul, Dtype::I8, gpu.caps().arch.compute_capability.unwrap()).is_some());
}

/// BYTE-identical to the WGSL tier at every row count the kernel serves, over
/// tails, ragged `n`, and unaligned and aligned binding windows.
#[test]
fn the_native_kernel_is_byte_identical_to_the_wgsl_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("native int8 GEMV needs a CUDA device");
        return;
    }
    // `kg` is always a multiple of 8 (K a whole number of 32-element groups):
    // 8 and 16 are shorter than one 64-word stride; 72 and 136 leave a
    // 16-byte-vector tail inside the last stride; 328 spans several strides
    // with a partial one.
    let aligned = [0, 0, 0, 0, 0];
    let ragged = [1, 3, 1, 2, 5];
    let quad = [4, 4, 4, 4, 4];
    for (kg, n) in [(8u32, 1u32), (16, 7), (72, 9), (136, 129), (328, 33), (1280, 64)] {
        for m in 1..=NATIVE_ROWS {
            for lead in [aligned, ragged, quad] {
                let (got, want) = run(&gpu, &Case { m, kg, n, lead });
                assert_eq!(
                    got, want,
                    "native int8 GEMV differs from the WGSL tier at m={m} kg={kg} n={n} lead={lead:?} - both \
                     fold the same terms in the same order, so this is a defect, not rounding"
                );
            }
        }
    }
}

/// Past the native tile the dispatch keeps the WGSL ladder and still agrees -
/// the redirect is per dispatch, so a caller never sees it flip.
#[test]
fn a_shape_past_the_native_tile_still_matches() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("no packed int8 dot on this device");
        return;
    }
    for m in [9u32, 12, 32] {
        let (got, want) = run(&gpu, &Case { m, kg: 136, n: 65, lead: [0; 5] });
        assert_eq!(got, want, "m={m}");
    }
}

/// The real Qwen3.8-27B decode projection shapes, `(K, N)` as the GGUF lists
/// them: attention/GDN projections, the SwiGLU pair and down projection, and a
/// narrow gate. Bit-identical at the decode row count and at a batched one.
#[test]
fn the_real_27b_projection_shapes_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("native int8 GEMV needs a CUDA device");
        return;
    }
    for (k, n) in [(5120u32, 48u32), (5120, 1024), (5120, 6144), (6144, 5120), (5120, 10240), (17408, 5120), (5120, 17408)] {
        for m in [1u32, 3, 8] {
            let (got, want) = run(&gpu, &Case { m, kg: k / 4, n, lead: [0; 5] });
            assert_eq!(got, want, "native int8 GEMV differs at the real shape k={k} n={n} m={m}");
        }
    }
}
