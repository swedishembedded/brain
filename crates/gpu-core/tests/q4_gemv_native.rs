// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native q4 (W4A8) decode GEMV (`kernels_cuda`,
//! `cu/matmul_q4_gemv.cu`): the single-matrix kernel `gpu_core::native_upgrade`
//! substitutes for the WGSL `matmul_q4_gemv` on a CUDA device, and its
//! multi-matrix entry point (`Fused::Q4GemvMulti`).
//!
//! Swedish Embedded AB implements bit-exact native kernels for quantised LLM
//! decode. If your team needs expertise in 4-bit weight streaming that changes
//! no output bit, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The claim is the RAW BITS of the WGSL tier. The reference is
//! `matmul_q4_gemv_ref`, the WGSL source under a name neither upgrade table
//! knows, dispatched on the same device from the same inputs. The cases are
//! what the vector path and the nibble unpacking can get wrong: `K` shorter
//! than one 16-lane stride and with a ragged last stride, `n` that is not a
//! multiple of the block's weight rows, every `m` the kernel serves (staged
//! and shared-memory rows of x, partial tiles), binding windows that start
//! off a 16-byte boundary (the scalar path), the full signed nibble range, and
//! the real Qwen3.8-27B projection widths and decode groups.

use backend_api::DeviceBuffer;
use gpu_core::{Dispatch, Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV), ("matmul_q4_gemv_ref", kernels::MATMUL_Q4_GEMV)];
const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// Rows of x the native kernel serves: the decode regime of the WGSL GEMV.
const NATIVE_ROWS: u32 = backend_api::select::DECODE_REGIME_MAX_ROWS;
/// Written around every window so an out-of-window write shows.
const SENTINEL: u32 = 0x7fc0_dead;

/// A CUDA device with native kernels not withheld (`set_native_kernels`).
fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled() && gpu.caps().numeric.int8_dot
}

/// Uniform random words: every byte value for the int8 activations and every
/// nibble value (-8..7) for the int4 weights.
fn words(n: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.next_u32()).collect()
}

fn scales(n: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| (1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).to_bits()).collect()
}

/// A buffer of `len + lead + 7` words with the window at `lead`, surrounded by
/// [`SENTINEL`] bits.
struct Windowed {
    buf: DeviceBuffer,
    lead: u64,
    len: usize,
    total: usize,
}

impl Windowed {
    fn new(gpu: &Gpu, data: &[u32], lead: u64) -> Windowed {
        let total = data.len() + lead as usize + 7;
        let buf = gpu.storage(total as u64);
        gpu.write(&buf, &vec![SENTINEL; total]);
        gpu.write_at(&buf, lead, data);
        Windowed { buf, lead, len: data.len(), total }
    }
    fn range(&self) -> (u64, u64) {
        (self.lead, self.len as u64)
    }
    fn bits(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.read(&self.buf, self.total).iter().map(|f| f.to_bits()).collect()
    }
    fn untouched_outside(&self, bits: &[u32]) -> bool {
        bits[..self.lead as usize].iter().chain(&bits[self.lead as usize + self.len..]).all(|&v| v == SENTINEL)
    }
}

/// `(m, k, n)` with leading words before each window: x, w, sx, sw, out.
fn run_single(gpu: &Gpu, m: u32, k: u32, n: u32, lead: [u64; 5]) {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let seed = u64::from(m) * 31 + u64::from(k) * 7 + u64::from(n);
    let x = Windowed::new(gpu, &words(mu * ku / 4, seed + 1), lead[0]);
    let w = Windowed::new(gpu, &words(nu * ku / 8, seed + 2), lead[1]);
    let sx = Windowed::new(gpu, &scales(mu, seed + 3), lead[2]);
    let sw = Windowed::new(gpu, &scales(nu * ku / 32, seed + 4), lead[3]);
    let outs = [Windowed::new(gpu, &vec![SENTINEL; mu * nu], lead[4]), Windowed::new(gpu, &vec![SENTINEL; mu * nu], lead[4])];
    let steps: Vec<_> = [K_GEMV, K_REF]
        .iter()
        .zip(&outs)
        .map(|(&kind, o)| {
            gpu.dispatch_sliced(
                kind,
                &[&x.buf, &w.buf, &sx.buf, &sw.buf, &o.buf],
                &[x.range(), w.range(), sx.range(), sw.range(), o.range()],
                &[m, k, n],
                Dispatch::Workgroups(n),
            )
        })
        .collect();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let ctx = format!("m={m} k={k} n={n} lead={lead:?}");
    for (name, o) in [("native", &outs[0]), ("reference", &outs[1])] {
        assert!(o.untouched_outside(&o.bits(gpu)), "{name} wrote outside its output window at {ctx}");
    }
    for (name, win) in [("x", &x), ("w", &w), ("sx", &sx), ("sw", &sw)] {
        assert!(win.untouched_outside(&win.bits(gpu)), "input window {name} was modified at {ctx}");
    }
    let window = |o: &Windowed| o.bits(gpu)[o.lead as usize..o.lead as usize + o.len].to_vec();
    assert_eq!(window(&outs[0]), window(&outs[1]), "native q4 GEMV differs from the WGSL tier at {ctx}");
}

/// `ns` (up to four matrices of one activation) as one multi launch, against
/// one WGSL launch per matrix: the same bits.
fn run_multi(gpu: &Gpu, m: u32, k: u32, ns: &[u32], seed: u64) {
    let as_f32 = |v: Vec<u32>| v.into_iter().map(f32::from_bits).collect::<Vec<_>>();
    let xq = gpu.storage_init("xq", &as_f32(words((m * k / 4) as usize, seed + 1)));
    let sx = gpu.storage_init("sx", &as_f32(scales(m as usize, seed + 2)));
    let mats: Vec<(DeviceBuffer, DeviceBuffer)> = ns
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let w = as_f32(words((n * k / 8) as usize, seed + 10 + i as u64));
            (gpu.storage_init("wq", &w), gpu.storage_init("sw", &as_f32(scales((n * k / 32) as usize, seed + 20 + i as u64))))
        })
        .collect();
    let sentinel = f32::from_bits(SENTINEL);
    let outs_ref: Vec<DeviceBuffer> = ns.iter().map(|&n| gpu.storage_init("o", &vec![sentinel; (m * n) as usize])).collect();
    let outs_nat: Vec<DeviceBuffer> = ns.iter().map(|&n| gpu.storage_init("o", &vec![sentinel; (m * n) as usize])).collect();
    let refs: Vec<_> = ns
        .iter()
        .enumerate()
        .map(|(i, &n)| gpu.dispatch(K_REF, &[&xq, &mats[i].0, &sx, &mats[i].1, &outs_ref[i]], &[m, k, n], Dispatch::Workgroups(n)))
        .collect();
    gpu.submit(&[], &refs);

    // Unused sets own no blocks; each is its own one-word allocation because
    // the facade refuses a dispatch whose last binding appears twice.
    let unused: Vec<DeviceBuffer> = (ns.len()..4).map(|_| gpu.storage(1)).collect();
    let mut bufs: Vec<&DeviceBuffer> = vec![&xq, &sx];
    let mut params = vec![m, k];
    for i in 0..4 {
        match ns.get(i) {
            Some(&n) => {
                bufs.extend([&mats[i].0, &mats[i].1, &outs_nat[i]]);
                params.push(n);
            }
            None => {
                bufs.extend([&xq, &sx, &unused[i - ns.len()]]);
                params.push(0);
            }
        }
    }
    let step = gpu.fused_step(Fused::Q4GemvMulti, &bufs, &params).expect("the q4 multi GEMV was declined on a CUDA device");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let bits = |b: &DeviceBuffer, len: u32| gpu.read(b, len as usize).iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    for (i, &n) in ns.iter().enumerate() {
        assert_eq!(bits(&outs_nat[i], m * n), bits(&outs_ref[i], m * n), "matrix {i} differs from the WGSL tier: m={m} k={k} ns={ns:?}");
    }
}

/// The redirect fired where it should and nowhere else; without this the
/// identity tests could pass on a handle that never left the WGSL tier.
#[test]
fn the_native_kernels_are_offered_exactly_where_they_serve() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_GEMV, &[1, 5120, 1024]), None, "only a CUDA device takes a native kernel");
        assert!(!gpu.has_fused(Fused::Q4GemvMulti), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("native q4 GEMV needs a CUDA device with a packed int8 dot");
        return;
    }
    for m in 1..=NATIVE_ROWS {
        assert_eq!(gpu.native_kernel_for(K_GEMV, &[m, 5120, 1024]), Some("matmul_q4_gemv"), "m={m}");
    }
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[NATIVE_ROWS + 1, 5120, 1024]), None, "past the decode regime");
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[1, 5136, 1024]), None, "K not a whole number of scale groups");
    assert_eq!(gpu.native_kernel_for(K_REF, &[1, 5120, 1024]), None, "the reference alias is not upgraded");

    assert!(gpu.has_fused(Fused::Q4GemvMulti));
    assert!(Fused::Q4GemvMulti.serves(&[1, 5120, 10240, 48, 48, 6144]));
    assert!(Fused::Q4GemvMulti.serves(&[8, 32, 1, 0, 0, 0]));
    assert!(!Fused::Q4GemvMulti.serves(&[9, 5120, 64, 0, 0, 0]), "past one tile of x rows");
    assert!(!Fused::Q4GemvMulti.serves(&[1, 5136, 64, 0, 0, 0]), "K not a whole number of scale groups");
    assert!(!Fused::Q4GemvMulti.serves(&[1, 5120, 0, 0, 0, 0]), "no matrix");
    assert!(!Fused::Q4GemvMulti.serves(&[1, 5120, 64]), "short params");
}

#[test]
fn the_single_gemv_is_byte_identical_to_the_wgsl_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native q4 GEMV needs a CUDA device with a packed int8 dot");
        return;
    }
    // K = 32 and 64 are under one 16-group stride; 544 and 1088 leave a ragged
    // last stride; 2624 spans several with a partial one.
    for (k, n) in [(32u32, 1u32), (64, 7), (544, 9), (1088, 129), (2624, 33), (5120, 64)] {
        for m in (1..=8).chain([9, 12, 16, 17, 24, 31, NATIVE_ROWS]) {
            for lead in [[0; 5], [1, 3, 1, 2, 5], [4; 5]] {
                run_single(&gpu, m, k, n, lead);
            }
        }
    }
}

#[test]
fn random_shapes_and_windows_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native q4 GEMV needs a CUDA device with a packed int8 dot");
        return;
    }
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    for _ in 0..60 {
        let m = 1 + r.next_u32() % NATIVE_ROWS;
        let k = 32 * (1 + r.next_u32() % 200);
        let n = 1 + r.next_u32() % 100;
        let lead = [0; 5].map(|_: u64| u64::from(r.next_u32() % 8));
        run_single(&gpu, m, k, n, lead);
    }
}

#[test]
fn the_multi_gemv_is_byte_identical_over_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native q4 GEMV needs a CUDA device with a packed int8 dot");
        return;
    }
    let groups: &[&[u32]] = &[&[64], &[1, 7], &[8, 9, 17], &[129, 1, 33, 64], &[1, 1, 1, 1], &[300, 48, 48, 9]];
    for (k, seed) in [(32u32, 1u64), (544, 2), (1088, 3), (5120, 4)] {
        for m in [1u32, 2, 3, 8] {
            for ns in groups {
                run_multi(&gpu, m, k, ns, seed * 100 + u64::from(m));
            }
        }
    }
}

/// The real Qwen3.8-27B decode projections: single shapes `(K, N)` as the GGUF
/// lists them, and the groups one layer multiplies together.
#[test]
fn the_real_27b_shapes_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native q4 GEMV needs a CUDA device with a packed int8 dot");
        return;
    }
    for (k, n) in [(5120u32, 48u32), (5120, 1024), (5120, 6144), (6144, 5120), (5120, 10240), (17408, 5120), (5120, 17408)] {
        for m in [1u32, 3, 8] {
            run_single(&gpu, m, k, n, [0; 5]);
        }
    }
    run_multi(&gpu, 1, 5120, &[10240, 48, 48, 6144], 31); // GDN: qkv, b, a, z
    run_multi(&gpu, 1, 5120, &[12288, 1024, 1024], 32); // GQA: q, k, v
    run_multi(&gpu, 1, 5120, &[17408, 17408], 33); // SwiGLU: gate, up
    run_multi(&gpu, 8, 5120, &[10240, 48, 48, 6144], 34); // a batched step
}
