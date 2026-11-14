// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `matmul_f32_gemv` kernel (`kernels_cuda`, `cu/matmul_f32_gemv.cu`) that
//! `gpu_core::native_upgrade` substitutes for the WGSL fp32 decode GEMV on a CUDA device.
//!
//! Swedish Embedded AB implements bit-exact native kernels for LLM decode. If your team needs expertise in replacing a
//! generated kernel with a native one without changing a single output bit, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The substitution is invisible to every caller, so the claim it has to hold is the one `gemv_reg_upgrade.rs` holds
//! for the WGSL pair: the RAW BITS are identical to the WGSL tier. The native kernel keeps the WGSL kernel's 64 virtual
//! accumulators and its fold order, so that is achievable. The reference here is `matmul_gemv_ref`, the same source
//! under a name neither upgrade table knows, so both sides run on the same device from the same inputs, differing only
//! in which kernel ran.
//!
//! Shapes cover what the vector-load path can get wrong: `k` shorter than one 64-element stride, `k` with a tail
//! inside the last stride and `k` not a multiple of four (the scalar path, whose rows start off a 16-byte boundary),
//! `n` not a multiple of the block's weight rows, every `m` of the first row tile, ones that split across tiles and
//! the end of the regime, and windows inside larger buffers (a binding offset is a whole number of
//! `backend_api::STORAGE_BINDING_OFFSET_ALIGN_BYTES`).

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_gemv", kernels::MATMUL_GEMV), ("matmul_gemv_ref", kernels::MATMUL_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// Rows of x the native kernel serves: the decode regime, the bound of the WGSL GEMV it stands in for.
const NATIVE_ROWS: u32 = backend_api::select::DECODE_REGIME_MAX_ROWS;
/// Written around every window so an out-of-window write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

/// A buffer of `window + lead + tail` words with the window at `lead`, surrounded by [`SENTINEL`] bits.
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

/// Values in [-1, 1) with a spread of magnitudes, so the sums round at many different exponents.
fn values(n: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|i| (r.signed() * if i % 7 == 0 { 64.0 } else { 1.0 }).to_bits()).collect()
}

/// Run `(m, k, n)` through both slots with the windows at `lead` (x, w, out) and return `(native-path bits,
/// reference bits)` of the output window, asserting the words around each window kept their sentinel.
fn run(gpu: &Gpu, m: u32, k: u32, n: u32, lead: [u64; 3]) -> (Vec<u32>, Vec<u32>) {
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let seed = u64::from(m) * 31 + u64::from(k) * 7 + u64::from(n);
    let x = Windowed::new(gpu, &values(mu * ku, seed + 1), lead[0]);
    let w = Windowed::new(gpu, &values(nu * ku, seed + 2), lead[1]);
    let a = Windowed::new(gpu, &vec![SENTINEL; mu * nu], lead[2]);
    let b = Windowed::new(gpu, &vec![SENTINEL; mu * nu], lead[2]);
    let dispatch = |kind: usize, o: &Windowed| {
        gpu.dispatch_sliced(kind, &[&x.buf, &w.buf, &o.buf], &[x.range(), w.range(), o.range()], &[m, k, n], Dispatch::Workgroups(n))
    };
    gpu.submit(&[], &[dispatch(K_GEMV, &a), dispatch(K_REF, &b)]);
    gpu.poll_wait();
    for (name, o) in [("native path", &a), ("reference", &b)] {
        let bits = o.read_bits(gpu);
        let (lo, hi) = (o.lead as usize, o.lead as usize + o.len);
        assert!(
            bits[..lo].iter().chain(&bits[hi..]).all(|&v| v == SENTINEL),
            "{name} wrote outside its output window at m={m} k={k} n={n} lead={lead:?}"
        );
    }
    for (name, win) in [("x", &x), ("w", &w)] {
        let bits = win.read_bits(gpu);
        assert!(bits[..win.lead as usize].iter().chain(&bits[win.lead as usize + win.len..]).all(|&v| v == SENTINEL), "input window {name} was modified");
    }
    (a.window(&a.read_bits(gpu)), b.window(&b.read_bits(gpu)))
}

/// The redirect fires for the decode regime and nowhere else. Without this the identity test below would pass
/// trivially on a handle that never left the WGSL tier.
#[test]
fn the_native_kernel_is_selected_exactly_for_the_shapes_it_serves() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("the decode GEMV upgrade needs workgroup reductions");
        return;
    }
    if !is_cuda(&gpu) || !gpu_core::native_kernels_enabled() {
        assert_eq!(gpu.native_kernel_for(K_GEMV, &[1, 1024, 1024]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device with native kernels");
        return;
    }
    for m in 1..=NATIVE_ROWS {
        let want = if m == 1 { "matmul_f32_gemv" } else { "matmul_f32_gemv_rows" };
        assert_eq!(gpu.native_kernel_for(K_GEMV, &[m, 1024, 1024]), Some(want), "m={m}");
    }
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[NATIVE_ROWS + 1, 1024, 1024]), None, "past the decode regime");
    assert_eq!(gpu.native_kernel_for(K_REF, &[1, 1024, 1024]), None, "the reference alias is not upgraded");
}

/// Byte-identical to the WGSL tier over tails, ragged `n`, every row count and windows at and inside a buffer's start.
#[test]
fn the_native_kernel_is_byte_identical_to_the_wgsl_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the native fp32 GEMV needs a CUDA device");
        return;
    }
    let whole = [0, 0, 0];
    // Words: one, two and one binding-alignment unit in.
    let inside = [64, 128, 64];
    for (k, n) in [(1u32, 1u32), (3, 5), (4, 16), (63, 17), (64, 33), (100, 7), (258, 31), (1024, 64), (3072, 48)] {
        for m in (1..=16).chain([17, 24, 31, NATIVE_ROWS]) {
            for lead in [whole, inside] {
                let (got, want) = run(&gpu, m, k, n, lead);
                assert_eq!(got, want, "native fp32 GEMV differs from the WGSL tier at m={m} k={k} n={n} lead={lead:?}");
            }
        }
    }
}
