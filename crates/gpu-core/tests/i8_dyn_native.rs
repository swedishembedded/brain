// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `matmul_i8_dp4a` kernel (`kernels_cuda`,
//! `cu/matmul_i8_dp4a.cu`) that `gpu_core::native_upgrade` substitutes for
//! the WGSL `matmul_i8_dyn` prefill/DiT GEMM on a CUDA device.
//!
//! Swedish Embedded AB implements bit-exact native int8 GEMMs for diffusion
//! transformers and LLM prefill. If your team needs expertise in replacing a
//! generated kernel with a hand-written one without changing a single output
//! bit, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The substitution is invisible to every caller, so the claim it has to hold
//! is RAW-BIT identity with the WGSL tier, not a tolerance. That is achievable:
//! the int32 sum of one 32-element weight-scale group is exact in any order,
//! and the native kernel then folds the groups into its fp32 running sum in
//! ascending order with the same two individually rounded operations
//! (`d + f32(c) * sw`) and applies `sx` last, exactly as the WGSL kernel does.
//! The reference is `matmul_i8_dyn_ref`, the same WGSL source under a name the
//! upgrade table does not know, so both sides run on one device from the same
//! inputs and differ only in which kernel ran.
//!
//! Shapes cover what a tiled kernel can get wrong: partial row and column
//! tiles, a `kg` that leaves the last k-chunk half empty (an odd number of
//! weight-scale groups), single rows and columns, binding windows at word
//! offsets that are and are not 16-byte aligned (the scalar-load path), output
//! windows inside larger buffers, and the GEMM shapes FLUX.2 klein's DiT and
//! its Qwen3 text encoder actually dispatch.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_dyn", kernels::MATMUL_I8_DYN), ("matmul_i8_dyn_ref", kernels::MATMUL_I8_DYN)];

const K_DYN: usize = 0;
const K_REF: usize = 1;

/// Packed words per weight-scale group (32 int8 along K).
const WPG: usize = 8;
/// The WGSL kernel's output tile, which sets its dispatch geometry.
const WGSL_TILE: u32 = 128;
/// Written around every window so an out-of-window write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

/// Deterministic signed bytes over the full range including -128 and 127,
/// packed four per word the way `dot4I8Packed` reads them.
fn packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                w |= (r.next_u32() % 256) << (8 * lane);
            }
            w
        })
        .collect()
}

/// Scales of both signs and a spread of magnitudes, so the fp32 fold really
/// rounds and a reordered fold would show in the low bits.
fn scales(n: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mag = 1e-4 + (r.next_u32() % 10_000) as f32 * 3.7e-7;
            let v = if r.next_u32().is_multiple_of(5) { -mag } else { mag };
            v.to_bits()
        })
        .collect()
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
/// bits)` of the output window, asserting every word around each window kept
/// its sentinel and no input was written.
fn run(gpu: &Gpu, c: &Case) -> (Vec<u32>, Vec<u32>) {
    let (m, kg, n) = (c.m as usize, c.kg as usize, c.n as usize);
    let ng = kg / WPG;
    let seed = u64::from(c.m) * 131 + u64::from(c.kg) * 7 + u64::from(c.n);
    let x = Windowed::new(gpu, &packed(m * kg, seed + 1), c.lead[0]);
    let w = Windowed::new(gpu, &packed(n * kg, seed + 2), c.lead[1]);
    let sx = Windowed::new(gpu, &scales(m, seed + 3), c.lead[2]);
    let sw = Windowed::new(gpu, &scales(n * ng, seed + 4), c.lead[3]);
    let a = Windowed::new(gpu, &vec![SENTINEL; m * n], c.lead[4]);
    let b = Windowed::new(gpu, &vec![SENTINEL; m * n], c.lead[4]);

    let params = [c.m, c.kg, c.n];
    let tiles = c.m.div_ceil(WGSL_TILE) * c.n.div_ceil(WGSL_TILE);
    let dispatch = |kind: usize, o: &Windowed| {
        gpu.dispatch_sliced(
            kind,
            &[&x.buf, &w.buf, &sx.buf, &sw.buf, &o.buf],
            &[x.range(), w.range(), sx.range(), sw.range(), o.range()],
            &params,
            Dispatch::Workgroups(tiles),
        )
    };
    gpu.submit(&[], &[dispatch(K_DYN, &a), dispatch(K_REF, &b)]);
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

fn check(gpu: &Gpu, c: &Case) {
    let (got, want) = run(gpu, c);
    let first = got.iter().zip(&want).position(|(g, w)| g != w);
    assert!(
        first.is_none(),
        "native int8 GEMM differs from the WGSL tier at m={} kg={} n={} lead={:?}: first at {:?} ({:?} vs {:?}) - \
         both fold the same exact group sums in the same order, so this is a defect, not rounding",
        c.m,
        c.kg,
        c.n,
        c.lead,
        first,
        first.map(|i| f32::from_bits(got[i])),
        first.map(|i| f32::from_bits(want[i])),
    );
}

fn device() -> Option<Gpu> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("native int8 GEMM needs a CUDA device with a packed int8 dot");
        return None;
    }
    Some(gpu)
}

/// The redirect fired where it should and nowhere else. Without this the
/// identity tests below would pass trivially on a handle that never left the
/// WGSL tier.
#[test]
fn the_native_kernel_is_selected_exactly_for_the_shapes_it_serves() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_DYN, &[512, 768, 3072]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    if !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("no packed int8 dot on this device");
        return;
    }
    for (m, kg, n) in [(1u32, 8u32, 1u32), (512, 768, 3072), (1792, 2304, 9216), (37, 24, 5)] {
        assert_eq!(gpu.native_kernel_for(K_DYN, &[m, kg, n]), Some("matmul_i8_dp4a"), "m={m} kg={kg} n={n}");
    }
    assert_eq!(gpu.native_kernel_for(K_DYN, &[512, 772, 3072]), None, "K not a whole number of scale groups");
    assert_eq!(gpu.native_kernel_for(K_DYN, &[0, 768, 3072]), None, "no rows");
    assert_eq!(gpu.native_kernel_for(K_DYN, &[512, 768, 0]), None, "no columns");
    assert_eq!(gpu.native_kernel_for(K_DYN, &[512, 768]), None, "short params");
    // The reference alias is deliberately not upgraded; it is what makes the
    // comparison a comparison.
    assert_eq!(gpu.native_kernel_for(K_REF, &[512, 768, 3072]), None);
}

/// RAW-BIT identical to the WGSL tier over partial tiles, odd group counts and
/// aligned and unaligned binding windows.
#[test]
fn the_native_kernel_is_bit_identical_to_the_wgsl_tier() {
    let Some(gpu) = device() else { return };
    let aligned = [0, 0, 0, 0, 0];
    let ragged = [1, 3, 1, 2, 5];
    let quad = [4, 4, 4, 4, 4];
    // `kg` is always a whole number of 8-word groups. 8 and 24 leave the last
    // two-group k-chunk half empty; 16 is exactly one chunk; 40 is several.
    for kg in [8u32, 16, 24, 40] {
        for (m, n) in [(1u32, 1u32), (1, 130), (127, 129), (128, 128), (129, 3), (200, 257), (300, 64)] {
            for lead in [aligned, ragged, quad] {
                check(&gpu, &Case { m, kg, n, lead });
            }
        }
    }
}

/// Seeded random shapes and windows, so the grid above is not the only place
/// the tile edges and the scalar-load path have been looked at.
#[test]
fn random_shapes_and_windows_are_bit_identical() {
    let Some(gpu) = device() else { return };
    let mut r = data::rng::Lcg::new(0x1d8d_ea7a);
    for _ in 0..40 {
        let m = 1 + r.next_u32() % 400;
        let kg = 8 * (1 + r.next_u32() % 48);
        let n = 1 + r.next_u32() % 400;
        let lead = [0, 1, 2, 3, 4].map(|_| u64::from(r.next_u32() % 8));
        check(&gpu, &Case { m, kg, n, lead });
    }
}

/// The GEMM shapes the models this kernel was written for dispatch, `(m, K,
/// N)`: FLUX.2 klein-4B's DiT (text rows, image rows and the joint slab at a
/// 640x512 generation and an edit, hidden 3072, MLP 9216) and its Qwen3-4B
/// text encoder over a 512-token prompt (hidden 2560, q 4096, kv 1024, MLP
/// 9728).
#[test]
fn the_model_shapes_are_bit_identical() {
    let Some(gpu) = device() else { return };
    let shapes = [
        (512u32, 3072u32, 3072u32),
        (1280, 3072, 9216),
        (1792, 9216, 3072),
        (3072, 3072, 3072),
        (512, 2560, 4096),
        (512, 2560, 1024),
        (512, 2560, 9728),
        (512, 9728, 2560),
    ];
    for (m, k, n) in shapes {
        check(&gpu, &Case { m, kg: k / 4, n, lead: [0; 5] });
    }
}
