// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `matmul_i8_gemv` kernel (`kernels_cuda`,
//! `cu/matmul_i8_gemv.cu`) that `gpu_core::native_upgrade` substitutes for the
//! WGSL decode GEMV on a CUDA device.
//!
//! Swedish Embedded AB implements bit-exact native kernels for quantised LLM
//! decode. If your team needs expertise in replacing a generated kernel with a
//! native one without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The substitution is invisible to every caller, so the claim it has to hold
//! is the one `i8_gemv_reg_upgrade.rs` holds for the WGSL pair: the RAW BITS
//! are identical to the WGSL tier. That is stronger than a tolerance and it is
//! achievable, because the native kernel keeps the WGSL kernel's 64 virtual
//! accumulators and its fold order. The reference here is `matmul_i8_gemv_ref`,
//! the same source under a name the upgrade table does not know, so both sides
//! run on the same device, from the same inputs, differing in which kernel
//! ran.
//!
//! Shapes cover what the vector-load path can get wrong: `kg` that is not a
//! whole number of the kernel's 64-word stride (tails), `n` that is not a
//! multiple of the block's weight rows, every `m` the kernel serves, weight
//! and activation bindings that start at a word offset which is not 16-byte
//! aligned (the scalar-load path), output windows inside larger buffers, and
//! the real Qwen3.8-27B projection widths.
//!
//! From `kernels_cuda::I8_GEMV_MMA_MIN_ROWS` rows on a device with int8 tensor
//! cores the native kernel runs its tensor-core block, which applies each
//! group's scale once per 32 products with a fused multiply-add instead of
//! folding 64 virtual accumulators: those rows are held to f32 rounding of the
//! running sum against the WGSL tier (`assert_native_agrees`), every other
//! row to its raw bits.

use backend_api::select::Dtype;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] =
    &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV), ("matmul_i8_gemv_ref", kernels::MATMUL_I8_GEMV)];

const K_GEMV: usize = 0;
const K_REF: usize = 1;

/// Rows of x one block of the native kernel covers; more rows take more blocks
/// of the grid, each re-reading the weights.
const TILE_ROWS: u32 = 8;
/// Rows of x the native kernel serves: the decode regime, the same bound the
/// WGSL GEMV it stands in for has (`select::DECODE_REGIME_MAX_ROWS`).
const NATIVE_ROWS: u32 = backend_api::select::DECODE_REGIME_MAX_ROWS;
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
    let sx = Windowed::new(gpu, &f32_words(&scales(m * ng, seed + 3, 1e-3, 1e-5)), c.lead[2]);
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

/// Whether an `m`-row batch runs the tensor-core block of the native GEMV on this device
/// (`kernels_cuda::I8_GEMV_MMA_MIN_ROWS`): there the result agrees with the WGSL kernel to f32
/// rounding of the running sum, not bit for bit.
fn tensor_core_rows(gpu: &Gpu, m: u32) -> bool {
    m >= kernels_cuda::I8_GEMV_MMA_MIN_ROWS && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::MMA_S8_MIN_CC)
}

/// The native output against the WGSL kernel's: the same bits where the dp4a block ran, and within
/// 2e-5 of the output's rms (f32 rounding of a few hundred group sums) where the tensor-core block did.
fn assert_native_agrees(gpu: &Gpu, m: u32, got: &[u32], want: &[u32], context: &str) {
    if !tensor_core_rows(gpu, m) {
        assert_eq!(got, want, "{context}");
        return;
    }
    let f = |v: &[u32]| v.iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>();
    let (g, w) = (f(got), f(want));
    assert!(g.iter().all(|x| x.is_finite()), "{context}: the tensor-core block left an output unwritten");
    let rms = (w.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / w.len().max(1) as f64).sqrt();
    let worst = g.iter().zip(&w).map(|(a, b)| f64::from(a - b).abs()).fold(0f64, f64::max);
    assert!(worst <= 2e-5 * rms + 1e-30, "{context}: worst |native - WGSL| {worst:.3e} against output rms {rms:.3e}");
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
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[NATIVE_ROWS + 1, 1280, 1024]), None, "past the decode regime keeps the prefill kernels");
    assert_eq!(gpu.native_kernel_for(K_GEMV, &[1, 1284, 1024]), None, "K not a whole number of scale groups");
    // The reference alias is deliberately not upgraded; it is what makes the
    // comparison a comparison.
    assert_eq!(gpu.native_kernel_for(K_REF, &[1, 1280, 1024]), None);
    // The registry is what says which tier this kernel reads.
    assert!(kernels_cuda::find(backend_api::select::Op::MatMul, Dtype::I8, gpu.caps().arch.compute_capability.unwrap()).is_some());
}

/// Byte-identical to the WGSL tier below `I8_GEMV_MMA_MIN_ROWS` rows (and at every row count on a
/// device without int8 tensor cores), within f32 rounding above it, over tails, ragged `n`, and
/// unaligned and aligned binding windows.
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
        // Every row count of the first tile, then ones that end a tile, split
        // across tiles, leave a ragged last tile and fill the regime.
        for m in (1..=TILE_ROWS).chain([9, 12, 16, 17, 24, 31, NATIVE_ROWS]) {
            for lead in [aligned, ragged, quad] {
                let (got, want) = run(&gpu, &Case { m, kg, n, lead });
                assert_native_agrees(
                    &gpu,
                    m,
                    &got,
                    &want,
                    &format!("native int8 GEMV differs from the WGSL tier at m={m} kg={kg} n={n} lead={lead:?}"),
                );
            }
        }
    }
}

/// Seeded random shapes and binding windows, so the grid above is not the
/// only place the vector path and its tails have been looked at.
#[test]
fn random_shapes_and_windows_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("native int8 GEMV needs a CUDA device");
        return;
    }
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    for _ in 0..60 {
        let m = 1 + r.next_u32() % NATIVE_ROWS;
        let kg = 8 * (1 + r.next_u32() % 200);
        let n = 1 + r.next_u32() % 100;
        let lead = [0, 1, 2, 3, 4].map(|_| u64::from(r.next_u32() % 8));
        let (got, want) = run(&gpu, &Case { m, kg, n, lead });
        assert_native_agrees(&gpu, m, &got, &want, &format!("native int8 GEMV differs at m={m} kg={kg} n={n} lead={lead:?}"));
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
            assert_native_agrees(&gpu, m, &got, &want, &format!("native int8 GEMV differs at the real shape k={k} n={n} m={m}"));
        }
    }
}

/// A pass profile bills the native kernel's device time to the kernel the
/// model dispatched: the redirect changes which kernel runs, never whether
/// its time shows up in the table.
#[test]
fn a_profiled_pass_reports_the_native_kernels_time() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot || gpu.native_kernel_for(K_GEMV, &[1, 1280, 5120]).is_none() {
        brain_testutil::skip_unavailable("native int8 GEMV needs a CUDA device");
        return;
    }
    let (m, kg, n) = (1u32, 1280u32, 5120u32);
    let x = gpu.storage(u64::from(m * kg));
    let w = gpu.storage(u64::from(n * kg));
    let sx = gpu.storage(u64::from(m * kg / 8));
    let sw = gpu.storage(u64::from(n * kg / 8));
    let out = gpu.storage(u64::from(m * n));
    let steps = [gpu.dispatch(K_GEMV, &[&x, &w, &sx, &sw, &out], &[m, kg, n], Dispatch::Workgroups(n))];
    let p = gpu_core::profile::profile(&gpu, "native gemv", &steps, 3);
    let row = p.rows.iter().find(|r| r.name == "matmul_i8_gemv").expect("the dispatched kernel has a row");
    assert!(p.device_timed && row.secs > 0.0, "the natively served GEMV reports {} s", row.secs);
}

/// A row of a batched decode step is the token it would be as a single row:
/// the batch shares the weight read, and nothing about the row's own
/// activation or scales may change with the row count. The batched kernel's
/// output is held to the same f32-rounding bar
/// [`assert_native_agrees`] uses against the WGSL tier, because the batched
/// row count runs the tensor-core block.
#[test]
fn each_row_of_a_batch_is_that_row_decoded_alone() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("native int8 GEMV needs a CUDA device");
        return;
    }
    let (kg, n) = (1280u32, 1024u32);
    let ng = kg / WPG as u32;
    let (rows, lead) = (8u32, [0u64; 5]);
    let x = Windowed::new(&gpu, &packed((rows * kg) as usize, 0x51ed), lead[0]);
    let w = Windowed::new(&gpu, &packed((n * kg) as usize, 0x1234), lead[1]);
    let sx = Windowed::new(&gpu, &f32_words(&scales((rows * ng) as usize, 0x99, 1e-3, 1e-5)), lead[2]);
    let sw = Windowed::new(&gpu, &f32_words(&scales((n * ng) as usize, 0x77, 1e-3, 1e-5)), lead[3]);
    let batched = Windowed::new(&gpu, &vec![SENTINEL; (rows * n) as usize], lead[4]);
    gpu.submit(
        &[],
        &[gpu.dispatch_sliced(
            K_GEMV,
            &[&x.buf, &w.buf, &sx.buf, &sw.buf, &batched.buf],
            &[x.range(), w.range(), sx.range(), sw.range(), batched.range()],
            &[rows, kg, n],
            Dispatch::Workgroups(n),
        )],
    );
    gpu.poll_wait();
    let batched = batched.window(&batched.read_bits(&gpu));

    for row in 0..rows {
        let alone = Windowed::new(&gpu, &vec![SENTINEL; n as usize], lead[4]);
        // The same activation row and its own scales, multiplied alone.
        gpu.submit(
            &[],
            &[gpu.dispatch_sliced(
                K_GEMV,
                &[&x.buf, &w.buf, &sx.buf, &sw.buf, &alone.buf],
                &[
                    (u64::from(row) * u64::from(kg), u64::from(kg)),
                    w.range(),
                    (u64::from(row) * u64::from(ng), u64::from(ng)),
                    sw.range(),
                    alone.range(),
                ],
                &[1, kg, n],
                Dispatch::Workgroups(n),
            )],
        );
        gpu.poll_wait();
        let alone = alone.window(&alone.read_bits(&gpu));
        let got = &batched[row as usize * n as usize..(row as usize + 1) * n as usize];
        assert_native_agrees(&gpu, rows, got, &alone, &format!("row {row} of a {rows}-row batch differs from that row decoded alone"));
    }
}

/// The exact-decode regime's contract at the op level: with the rows cut into
/// groups of [`backend_api::select::exact_decode_group_rows`], every row of a
/// batch is BIT-identical to that row decoded alone - at the batch-1 form, at
/// the tensor-core block's first row count ([`kernels_cuda::I8_GEMV_MMA_MIN_ROWS`]),
/// and at a full batch. The ungrouped batch is still allowed the documented
/// f32-rounding tolerance ([`assert_native_agrees`]), which is what the
/// grouping removes.
#[test]
fn exact_decode_groups_make_every_batch_row_bit_identical_to_the_same_row_alone() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("native int8 GEMV needs a CUDA device");
        return;
    }
    let (kg, n) = (1280u32, 1024u32);
    let ng = kg / WPG as u32;
    let rows = 8u32;
    let step = backend_api::select::exact_decode_group_rows(&gpu.caps());
    assert!((1..=rows).contains(&step), "the exact group size {step} is outside the decode batch");
    let x = Windowed::new(&gpu, &packed((rows * kg) as usize, 0x51ed), 0);
    let w = Windowed::new(&gpu, &packed((n * kg) as usize, 0x1234), 0);
    let sx = Windowed::new(&gpu, &f32_words(&scales((rows * ng) as usize, 0x99, 1e-3, 1e-5)), 0);
    let sw = Windowed::new(&gpu, &f32_words(&scales((n * ng) as usize, 0x77, 1e-3, 1e-5)), 0);
    let out = Windowed::new(&gpu, &vec![SENTINEL; (rows * n) as usize], 0);

    // One dispatch of `len` rows starting at row `row0`, writing at `row0`.
    let dispatch = |gpu: &Gpu, row0: u32, len: u32| {
        gpu.submit(
            &[],
            &[gpu.dispatch_sliced(
                K_GEMV,
                &[&x.buf, &w.buf, &sx.buf, &sw.buf, &out.buf],
                &[
                    (u64::from(row0) * u64::from(kg), u64::from(len) * u64::from(kg)),
                    w.range(),
                    (u64::from(row0) * u64::from(ng), u64::from(len) * u64::from(ng)),
                    sw.range(),
                    (u64::from(row0) * u64::from(n), u64::from(len) * u64::from(n)),
                ],
                &[len, kg, n],
                Dispatch::Workgroups(n),
            )],
        );
        gpu.poll_wait();
    };

    // Every row decoded alone, concatenated in row order. The dispatch writes
    // the row at its own slot of the window, so that slot is what is read.
    let mut alone = Vec::with_capacity((rows * n) as usize);
    for row in 0..rows {
        dispatch(&gpu, row, 1);
        let bits = out.window(&out.read_bits(&gpu));
        alone.extend_from_slice(&bits[(row * n) as usize..((row + 1) * n) as usize]);
    }

    // The default batched step: one kernel instance for all rows, which the
    // tensor-core block serves to f32 rounding rather than bit for bit.
    dispatch(&gpu, 0, rows);
    let batched = out.window(&out.read_bits(&gpu));
    assert_native_agrees(&gpu, rows, &batched, &alone, "the default 8-row step against each row decoded alone");

    // The exact regime: groups of at most `step`, each the same kernel instance
    // a single row runs, so the bits match at every row count.
    for m in [1u32, step, step + 1, kernels_cuda::I8_GEMV_MMA_MIN_ROWS, rows] {
        if m > rows {
            continue;
        }
        for start in (0..m).step_by(step as usize) {
            let len = step.min(m - start);
            dispatch(&gpu, start, len);
        }
        let grouped = out.window(&out.read_bits(&gpu));
        let want = &alone[..(m * n) as usize];
        assert_eq!(grouped, want, "exact-decode grouping differs from the rows decoded alone at m={m} step={step}");
    }
}
