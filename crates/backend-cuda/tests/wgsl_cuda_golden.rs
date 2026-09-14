// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generated CUDA tier must compute what the portable WGSL computes.
//!
//! One WGSL source, two executions: `wgsl-cpu`'s naga -> Cranelift JIT (the
//! reference that already runs this repo's catalogue on the CPU backend) and
//! `wgsl-cuda`'s naga -> CUDA C++ emission, compiled by NVRTC and launched on
//! a real device. Any disagreement past the fp32 noise floor is a miscompile.
//!
//! This gate exists because every failure mode of a shader-to-CUDA translator
//! is a SILENT wrong number, not a compile error: a barrier some threads never
//! reach, a uniform member read at the C++ offset instead of the WGSL one, a
//! shift the PTX ISA clamps where the reference masks, a contracted multiply
//! -add, an aliasing promise the caller does not keep, or shared memory that
//! WGSL zeroed and `__shared__` did not. Each of those has a test of its own
//! below, written to fail if the corresponding fix were removed - a
//! happy-path kernel comparison passes with five of the six defects present.
//!
//! Swedish Embedded AB implements differential testing of code generators for
//! its clients - running one source through two independent backends and
//! holding them to a numerical contract. If your team needs expertise in
//! proving a compiler or kernel port correct rather than hoping it is, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! Skip-if-absent (`brain_testutil::skip_unavailable`): a box with no NVIDIA
//! driver or no NVRTC cannot run these and no flag may make that fatal. Shapes
//! are deliberately tiny - this is a correctness gate, never a benchmark, and
//! it must not contend with whatever else is resident on the device.

use backend_cuda::exec;

/// A host buffer, in the element type its binding declares.
#[derive(Clone, PartialEq, Debug)]
enum Buf {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl Buf {
    fn bytes(&self) -> Vec<u8> {
        match self {
            Buf::F32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            Buf::U32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        }
    }
    fn len_bytes(&self) -> usize {
        match self {
            Buf::F32(v) => v.len() * 4,
            Buf::U32(v) => v.len() * 4,
        }
    }
    fn overwrite_from_bytes(&mut self, b: &[u8]) {
        match self {
            Buf::F32(v) => {
                for (i, s) in v.iter_mut().enumerate() {
                    *s = f32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
                }
            }
            Buf::U32(v) => {
                for (i, s) in v.iter_mut().enumerate() {
                    *s = u32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
                }
            }
        }
    }
}

/// One kernel at one shape: the uniform stream, the storage bindings in
/// binding order, and the dispatch grid. Deliberately spelled the same way for
/// both backends so the only difference under test is the code generator.
struct Case {
    name: &'static str,
    wgsl: &'static str,
    params: Vec<u32>,
    bufs: Vec<Buf>,
    grid_x: u32,
    grid_y: u32,
    /// Storage bindings that carry results, and the absolute tolerance each is
    /// held to. 0.0 means bit-exact.
    outputs: Vec<(usize, f32)>,
    /// Bindings that must be bound to the SAME device allocation as an earlier
    /// binding: `(binding_slot, aliases_slot)`. brain's `DeviceBuffer` clones
    /// alias by design, so the generated code may never promise otherwise.
    alias: Vec<(usize, usize)>,
}

/// Deterministic pseudo-random f32 in [-1, 1); a fixed stream so a failure is
/// reproducible and a tolerance is not accidentally shape-dependent.
fn rnd(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

/// Run one case through `wgsl-cpu`'s JIT, returning the output buffers.
fn run_cpu(case: &Case) -> Vec<Buf> {
    let jit = wgsl_cpu::Jit::new(&[(case.name, case.wgsl)])
        .unwrap_or_else(|e| panic!("{}: wgsl-cpu refused the reference kernel: {e}", case.name));
    let mut bufs = case.bufs.clone();
    // Aliased bindings must alias on this side too, or the two runs are not
    // being handed the same memory topology.
    for &(slot, onto) in &case.alias {
        bufs[slot] = bufs[onto].clone();
    }
    let mut raw: Vec<Vec<u8>> = bufs.iter().map(|b| b.bytes()).collect();
    let mut ptrs: Vec<*mut u8> = raw.iter_mut().map(|r| r.as_mut_ptr()).collect();
    // An aliased binding points at the allocation it aliases, on this side too.
    for &(slot, onto) in &case.alias {
        ptrs[slot] = ptrs[onto];
    }
    let total = (case.grid_x as u64) * 64 * (case.grid_y as u64);
    // SAFETY: every binding has a buffer at least as large as the kernel's own
    // masked index range, and the uniform stream is the packed `Params` the
    // kernel declares.
    unsafe {
        jit.run(0, 0, total, case.grid_x, case.grid_y, case.params.as_ptr(), ptrs.as_ptr());
    }
    for (i, b) in bufs.iter_mut().enumerate() {
        let src = case.alias.iter().find(|(slot, _)| *slot == i).map(|(_, onto)| *onto).unwrap_or(i);
        b.overwrite_from_bytes(&raw[src]);
    }
    bufs
}

/// Run one case through `wgsl-cuda` -> NVRTC -> a real device.
fn run_cuda(ctx: &exec::Context, case: &Case) -> Vec<Buf> {
    let gen = wgsl_cuda::generate(case.name, case.wgsl)
        .unwrap_or_else(|e| panic!("{}: wgsl-cuda refused the kernel: {e}", case.name));
    assert_eq!(
        gen.bindings.len(),
        case.bufs.len(),
        "{}: generated kernel takes {} storage bindings, the case supplies {}",
        case.name,
        gen.bindings.len(),
        case.bufs.len()
    );
    let module = ctx
        .compile(&gen.source, &gen.entry)
        .unwrap_or_else(|e| panic!("{}: NVRTC rejected the generated source: {e}\n{}", case.name, gen.source));
    let f = module.function(&gen.entry).expect("entry point");

    let params_bytes: Vec<u8> = case.params.iter().flat_map(|x| x.to_ne_bytes()).collect();
    assert!(
        params_bytes.len() >= gen.uniform_bytes,
        "{}: the case supplies {} uniform bytes, the kernel's Params is {}",
        case.name,
        params_bytes.len(),
        gen.uniform_bytes
    );
    let p = ctx.alloc(params_bytes.len().max(4)).expect("alloc params");
    ctx.upload(&p, &params_bytes).expect("upload params");

    let mut owned: Vec<Option<exec::DeviceMem>> = Vec::new();
    for (i, b) in case.bufs.iter().enumerate() {
        if case.alias.iter().any(|(slot, _)| *slot == i) {
            owned.push(None);
            continue;
        }
        let m = ctx.alloc(b.len_bytes().max(4)).expect("alloc binding");
        ctx.upload(&m, &b.bytes()).expect("upload binding");
        owned.push(Some(m));
    }
    let mut args: Vec<&exec::DeviceMem> = vec![&p];
    let mut order: Vec<usize> = Vec::new();
    for i in 0..case.bufs.len() {
        let src = case.alias.iter().find(|(slot, _)| *slot == i).map(|(_, onto)| *onto).unwrap_or(i);
        order.push(src);
    }
    for &src in &order {
        args.push(owned[src].as_ref().expect("aliased onto an owned allocation"));
    }
    ctx.launch(&f, (case.grid_x, case.grid_y, 1), (gen.block_dim, 1, 1), &args)
        .expect("launch");
    ctx.sync().expect("sync");

    let mut out = case.bufs.clone();
    for (i, b) in out.iter_mut().enumerate() {
        let mut bytes = vec![0u8; b.len_bytes()];
        ctx.download(owned[order[i]].as_ref().unwrap(), &mut bytes).expect("download");
        b.overwrite_from_bytes(&bytes);
    }
    out
}

/// The whole comparison: same source, same inputs, both backends.
fn agree(ctx: &exec::Context, case: &Case) {
    let cpu = run_cpu(case);
    let cuda = run_cuda(ctx, case);
    for &(slot, tol) in &case.outputs {
        match (&cpu[slot], &cuda[slot]) {
            (Buf::F32(a), Buf::F32(b)) => {
                let mut worst = 0.0f32;
                let mut at = 0usize;
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    let d = (x - y).abs();
                    if d > worst {
                        worst = d;
                        at = i;
                    }
                }
                assert!(
                    worst <= tol,
                    "{}: binding {slot} disagrees by {worst} at element {at} (cpu {}, cuda {}), tolerance {tol}",
                    case.name,
                    a[at],
                    b[at]
                );
            }
            (Buf::U32(a), Buf::U32(b)) => {
                assert_eq!(a, b, "{}: binding {slot} disagrees", case.name);
            }
            _ => panic!("{}: binding {slot} changed element type between backends", case.name),
        }
    }
}

/// A device to run on, or the reason there is none.
fn device() -> Option<exec::Context> {
    match exec::Context::open(0) {
        Ok(c) => Some(c),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device for kernel execution: {e}"));
            None
        }
    }
}

fn elementwise_case(name: &'static str, wgsl: &'static str, n: usize, inputs: usize) -> Case {
    let mut seed = 0x1234_5678u32;
    let mut bufs: Vec<Buf> = (0..inputs).map(|_| Buf::F32((0..n).map(|_| rnd(&mut seed)).collect())).collect();
    bufs.push(Buf::F32(vec![0.0; n]));
    Case {
        name,
        wgsl,
        params: vec![n as u32],
        bufs,
        grid_x: n.div_ceil(64) as u32,
        grid_y: 1,
        outputs: vec![(inputs, 0.0)],
        alias: vec![],
    }
}

/// The covered catalogue subset, at two shapes each: one that fills whole
/// workgroups and one with a partial tail, because the tail is exactly where a
/// mishandled early-return mask shows up.
#[test]
fn generated_cuda_agrees_with_the_cpu_reference_per_kernel_and_shape() {
    let Some(ctx) = device() else { return };

    for n in [64usize, 130] {
        agree(&ctx, &elementwise_case("add2", kernels::ADD2, n, 2));
        agree(&ctx, &elementwise_case("mul", kernels::MUL, n, 2));
        // `tanh` is a libm call on one side and a device intrinsic on the
        // other, so this one is held to the fp32 noise floor, not bit-equality.
        let mut g = elementwise_case("gelu", kernels::GELU, n, 1);
        g.outputs = vec![(1, 1e-6)];
        agree(&ctx, &g);
    }

    // Packed int8: `dot4I8Packed`, u32 storage, i32 arithmetic. Exact by
    // construction (|S| <= 32*127), so any disagreement at all is a defect.
    for (m, k) in [(2usize, 64usize), (3, 96)] {
        let words = m * k / 4;
        let groups = m * k / 32;
        let xq: Vec<u32> = (0..words)
            .map(|i| {
                let b = |j: usize| ((((i * 4 + j) as i32 * 37 + 11) % 255 - 127) as i8) as u8 as u32;
                b(0) | (b(1) << 8) | (b(2) << 16) | (b(3) << 24)
            })
            .collect();
        agree(
            &ctx,
            &Case {
                name: "quant_group_sum",
                wgsl: kernels::QUANT_GROUP_SUM,
                params: vec![m as u32, k as u32],
                bufs: vec![Buf::U32(xq), Buf::F32(vec![0.0; groups])],
                grid_x: groups.div_ceil(64) as u32,
                grid_y: 1,
                outputs: vec![(1, 0.0)],
                alias: vec![],
            },
        );
    }

    // A cooperative reduction: workgroup memory, one barrier, and a
    // workgroup-uniform early return before that barrier.
    for (numel, n_wg) in [(512usize, 2u32), (700, 3)] {
        let mut seed = 0xfeed_beefu32;
        let grad: Vec<f32> = (0..numel).map(|_| rnd(&mut seed)).collect();
        agree(
            &ctx,
            &Case {
                name: "gradnorm_part",
                wgsl: kernels::GRADNORM_PART,
                params: vec![numel as u32, 0, n_wg],
                bufs: vec![Buf::F32(grad), Buf::F32(vec![0.0; n_wg as usize])],
                grid_x: n_wg,
                grid_y: 1,
                // A tree of 64 partials summed in the same order on both
                // sides, but the loads are fp32 adds either way: the noise
                // floor, not bit-equality.
                outputs: vec![(1, 1e-6)],
                alias: vec![],
            },
        );
    }
}

// ---------------------------------------------------------------------------
// The six codegen hazards, one test each. Every one of these passes if the
// generator is happy-path-correct and the corresponding fix is absent, which
// is why they are written separately from the catalogue comparison above.
// ---------------------------------------------------------------------------

/// Hazard 1 - an early `return` that precedes a barrier.
///
/// WGSL permits it when the predicate is workgroup-uniform, and this repo's
/// barrier-using kernels rely on that. A thread that has actually RETURNED
/// cannot arrive at `__syncthreads()`; on a device without independent thread
/// scheduling the remaining threads then wait on a barrier that will never be
/// complete, and what comes back is whatever shared memory held. The generator
/// must emit the guarded-body form instead of a real `return`.
///
/// The shape here deliberately dispatches MORE workgroups than the kernel's
/// own `n_wg`, so the surplus workgroups take the early return while the live
/// ones must still pass the barrier and produce a correct partial.
#[test]
fn an_early_return_before_a_barrier_does_not_strand_the_barrier() {
    let Some(ctx) = device() else { return };

    let numel = 400usize;
    let n_wg = 2u32;
    let dispatched = 5u32; // 3 workgroups return before the barrier
    let mut seed = 0x0bad_c0deu32;
    let grad: Vec<f32> = (0..numel).map(|_| rnd(&mut seed)).collect();
    let case = Case {
        name: "gradnorm_part",
        wgsl: kernels::GRADNORM_PART,
        params: vec![numel as u32, 0, n_wg],
        bufs: vec![Buf::F32(grad.clone()), Buf::F32(vec![-1.0; dispatched as usize])],
        grid_x: dispatched,
        grid_y: 1,
        outputs: vec![(1, 1e-6)],
        alias: vec![],
    };
    agree(&ctx, &case);

    // And against an independent host oracle, so "both backends are wrong the
    // same way" is not mistaken for agreement.
    let got = run_cuda(&ctx, &case);
    let Buf::F32(parts) = &got[1] else { panic!("parts is f32") };
    for (w, got) in parts.iter().take(n_wg as usize).enumerate() {
        let mut want = 0.0f64;
        for t in 0..64usize {
            let mut lane = 0.0f64;
            let mut i = w * 64 + t;
            while i < numel {
                lane += (grad[i] as f64) * (grad[i] as f64);
                i += n_wg as usize * 64;
            }
            want += lane;
        }
        assert!(
            (*got as f64 - want).abs() < 1e-5,
            "workgroup {w} partial {got} != host oracle {want}"
        );
    }
    // The workgroups that returned early must have written nothing at all.
    for (w, got) in parts.iter().enumerate().skip(n_wg as usize) {
        assert_eq!(*got, -1.0, "a returned workgroup wrote parts[{w}]");
    }
}

/// Hazard 2 - WGSL uniform layout is not C++ struct layout.
///
/// A `vec3<u32>` member is 16-byte ALIGNED and 12 bytes long in WGSL, so a
/// scalar after it starts at 28; a transliterated C++ struct with a `uint3`
/// puts it at 16. Reading the wrong 4 bytes is not a crash, it is a plausible
/// wrong number. The generator must never emit the struct - it must load each
/// member at the offset naga's layouter computed.
#[test]
fn a_uniform_member_is_read_at_the_wgsl_offset_not_the_cplusplus_one() {
    const SRC: &str = r#"
struct Params {
    n: u32,
    pad: vec3<u32>,
    scale: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    y[i] = x[i] * p.scale;
}
"#;
    let gen = wgsl_cuda::generate("uniform_layout", SRC).expect("generate");
    assert!(
        !gen.source.contains("struct Params"),
        "the generated source transliterates the uniform struct:\n{}",
        gen.source
    );
    assert_eq!(gen.uniform_bytes, 32, "WGSL sizes this Params at 32 bytes");

    let Some(ctx) = device() else { return };
    let n = 70usize;
    let mut seed = 0x5eed_0001u32;
    let x: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
    // n at 0, vec3<u32> at 16 (align 16, size 12), f32 `scale` at 28.
    let mut params = vec![0u32; 8];
    params[0] = n as u32;
    params[4] = 0xdead_beef;
    params[5] = 0xdead_beef;
    params[6] = 0xdead_beef;
    params[7] = 2.5f32.to_bits();
    let case = Case {
        name: "uniform_layout",
        wgsl: SRC,
        params,
        bufs: vec![Buf::F32(x.clone()), Buf::F32(vec![0.0; n])],
        grid_x: n.div_ceil(64) as u32,
        grid_y: 1,
        outputs: vec![(1, 0.0)],
        alias: vec![],
    };
    agree(&ctx, &case);
    let got = run_cuda(&ctx, &case);
    let Buf::F32(y) = &got[1] else { panic!("y is f32") };
    for i in 0..n {
        assert_eq!(y[i], x[i] * 2.5, "scale was read from the wrong offset at {i}");
    }
}

/// Hazard 3 - shift amounts at or above the word width.
///
/// The PTX ISA CLAMPS a shift amount greater than the register width, so
/// `x << 32` yields 0; x86 (and therefore the Cranelift reference) MASKS it
/// modulo 32, so the same expression yields `x`. WGSL itself calls that range
/// indeterminate, which is precisely why the generated tier may not inherit
/// whichever behaviour the target happens to have: it must reproduce the
/// reference's, explicitly.
#[test]
fn a_shift_at_or_above_the_word_width_matches_the_reference() {
    const SRC: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<u32>;
@group(0) @binding(2) var<storage, read>       s: array<u32>;
@group(0) @binding(3) var<storage, read_write> o: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    o[i] = (a[i] << s[i]) ^ (a[i] >> s[i]);
}
"#;
    let gen = wgsl_cuda::generate("shifts", SRC).expect("generate");
    assert!(
        gen.source.contains("& 31u"),
        "the generated source does not mask the shift amount:\n{}",
        gen.source
    );

    let Some(ctx) = device() else { return };
    let n = 64usize;
    // Shift amounts straddling the word width, including the exact boundary.
    let shifts: Vec<u32> = (0..n).map(|i| (i as u32 * 7) % 70).collect();
    let a: Vec<u32> = (0..n).map(|i| 0x8000_0001u32.wrapping_mul(i as u32 + 1) | 1).collect();
    assert!(shifts.iter().any(|&s| s >= 32), "the case must cross the word width");
    agree(
        &ctx,
        &Case {
            name: "shifts",
            wgsl: SRC,
            params: vec![n as u32],
            bufs: vec![Buf::U32(a), Buf::U32(shifts), Buf::U32(vec![0; n])],
            grid_x: 1,
            grid_y: 1,
            outputs: vec![(2, 0.0)],
            alias: vec![],
        },
    );
}

/// Hazard 4 - NVRTC contracts `a*b + c` into an FMA by default.
///
/// One rounding instead of two is a BETTER answer and still a disagreement:
/// the cross-backend parity assertions this project already ships are absolute
/// (maxabs < 1e-6) and a contracted accumulation drifts past that over a long
/// reduction. The generated tier compiles with `--fmad=false`.
///
/// The inputs are checked on the host to actually discriminate, so this cannot
/// pass by picking values where contraction makes no difference.
#[test]
fn multiply_add_is_not_contracted() {
    const SRC: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read>       c: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    o[i] = a[i] * b[i] + c[i];
}
"#;
    let Some(ctx) = device() else { return };

    // Search a deterministic stream for triples where the two roundings
    // actually differ, and assert we found some - otherwise the test is vacuous.
    let mut seed = 0x00c0_ffeeu32;
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    let mut discriminating = 0usize;
    while a.len() < 64 {
        let x = rnd(&mut seed) * 3.0;
        let y = rnd(&mut seed) * 3.0;
        let z = rnd(&mut seed) * 3.0;
        if x.mul_add(y, z) != x * y + z {
            discriminating += 1;
        }
        a.push(x);
        b.push(y);
        c.push(z);
    }
    assert!(discriminating > 0, "no input in this case distinguishes a contracted multiply-add");

    agree(
        &ctx,
        &Case {
            name: "fmad",
            wgsl: SRC,
            params: vec![64],
            bufs: vec![Buf::F32(a), Buf::F32(b), Buf::F32(c), Buf::F32(vec![0.0; 64])],
            grid_x: 1,
            grid_y: 1,
            // Bit-exact: with contraction off both sides round twice.
            outputs: vec![(3, 0.0)],
            alias: vec![],
        },
    );
}

/// Hazard 5 - `__restrict__` is a promise brain's callers do not keep.
///
/// `DeviceBuffer` clones alias by design and a sliced step binds overlapping
/// ranges of one allocation, so a generated kernel may not tell the compiler
/// its pointers are distinct. Two checks: the emitted text carries no
/// `__restrict__` for any covered kernel, and a run with two bindings pointing
/// at the SAME allocation still agrees with the reference.
#[test]
fn generated_kernels_never_promise_their_pointers_do_not_alias() {
    for (name, src) in covered_kernels() {
        let gen = wgsl_cuda::generate(name, src).expect("generate");
        assert!(
            !gen.source.contains("__restrict"),
            "{name}: the generated tier may not declare __restrict__"
        );
    }

    let Some(ctx) = device() else { return };
    let n = 96usize;
    let mut seed = 0xa11a_5000u32;
    let a: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
    // `out` and `a` are the same allocation: out[i] = out[i] + out[i].
    agree(
        &ctx,
        &Case {
            name: "add_inplace",
            wgsl: kernels::ADD_INPLACE,
            params: vec![n as u32],
            bufs: vec![Buf::F32(a.clone()), Buf::F32(a)],
            grid_x: n.div_ceil(64) as u32,
            grid_y: 1,
            outputs: vec![(0, 0.0)],
            alias: vec![(1, 0)],
        },
    );
}

/// Hazard 6 - WGSL zero-initialises `var<workgroup>`; `__shared__` does not.
///
/// A reduction whose tail threads never write their slot reads zeros under
/// WGSL and reads whatever the last resident block left behind under CUDA.
/// The generator must emit the zeroing plus the barrier that publishes it.
///
/// The kernel below is written to depend on that: only `t < p.live` writes a
/// partial, but thread 0 folds all 64 slots.
#[test]
fn workgroup_memory_is_zero_initialised_the_way_wgsl_promises() {
    const SRC: &str = r#"
struct Params { live: u32, n_wg: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (w >= p.n_wg) { return; }
    if (t < p.live) {
        partial[t] = x[w * 64u + t];
    }
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            s = s + partial[k];
        }
        o[w] = s;
    }
}
"#;
    let gen = wgsl_cuda::generate("wg_zero", SRC).expect("generate");
    assert!(
        gen.source.contains("__shared__"),
        "the kernel declares var<workgroup> but no __shared__ was emitted"
    );

    let Some(ctx) = device() else { return };
    let n_wg = 3u32;
    let live = 5u32;

    // Dirty the shared memory first. `__shared__` is a window into the
    // multiprocessor's scratch that the NEXT resident block inherits as it was
    // left, so without a dirtying pass a test on an idle device reads zeros by
    // luck and would pass with the zeroing removed. This kernel declares the
    // same 64 floats and fills every one of them, so the slots the kernel
    // under test never writes are demonstrably non-zero when it starts.
    const DIRTY: &str = r#"
struct Params { n_wg: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    let t = li.x;
    partial[t] = 12345.5;
    workgroupBarrier();
    if (t == 0u) {
        o[w] = partial[63];
    }
}
"#;
    let dirty = run_cuda(
        &ctx,
        &Case {
            name: "wg_dirty",
            wgsl: DIRTY,
            params: vec![n_wg],
            bufs: vec![Buf::F32(vec![0.0; n_wg as usize])],
            grid_x: n_wg,
            grid_y: 1,
            outputs: vec![],
            alias: vec![],
        },
    );
    let Buf::F32(seen) = &dirty[0] else { panic!("o is f32") };
    assert!(seen.iter().all(|v| *v == 12345.5), "the dirtying pass did not run: {seen:?}");

    let mut seed = 0x2222_3333u32;
    let x: Vec<f32> = (0..(n_wg as usize) * 64).map(|_| rnd(&mut seed)).collect();
    let case = Case {
        name: "wg_zero",
        wgsl: SRC,
        params: vec![live, n_wg],
        bufs: vec![Buf::F32(x.clone()), Buf::F32(vec![0.0; n_wg as usize])],
        grid_x: n_wg,
        grid_y: 1,
        outputs: vec![(1, 1e-6)],
        alias: vec![],
    };
    agree(&ctx, &case);

    // Independent oracle: only the live lanes contribute, the rest are zeros.
    let got = run_cuda(&ctx, &case);
    let Buf::F32(o) = &got[1] else { panic!("o is f32") };
    for w in 0..n_wg as usize {
        let want: f32 = (0..live as usize).map(|t| x[w * 64 + t]).sum();
        assert!((o[w] - want).abs() < 1e-6, "workgroup {w}: {} != {want}", o[w]);
    }
}

/// The kernels this milestone claims. Named in one place so the hazard checks
/// above and the ledger cannot drift apart.
fn covered_kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("add2", kernels::ADD2),
        ("mul", kernels::MUL),
        ("gelu", kernels::GELU),
        ("add_inplace", kernels::ADD_INPLACE),
        ("quant_group_sum", kernels::QUANT_GROUP_SUM),
        ("gradnorm_part", kernels::GRADNORM_PART),
    ]
}

/// Every covered kernel must at least GENERATE and compile, independently of
/// whether a shape was written for it above - a kernel that only compiles for
/// the one shape a test happens to run is not covered.
#[test]
fn every_covered_kernel_generates_and_compiles() {
    let sources: Vec<(&str, String)> = covered_kernels()
        .into_iter()
        .map(|(name, src)| {
            let g = wgsl_cuda::generate(name, src).unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, g.source)
        })
        .collect();

    let Some(ctx) = device() else { return };
    for (name, src) in &sources {
        let entry = wgsl_cuda::entry_name(name);
        ctx.compile(src, &entry).unwrap_or_else(|e| panic!("{name}: NVRTC rejected:\n{e}\n{src}"));
    }
}

/// A compiled cubin is cached on disk under a key that includes everything
/// that can change the output, and a second compile of the same source comes
/// back from that cache rather than from NVRTC.
#[test]
fn a_compiled_cubin_is_cached_under_a_key_that_covers_its_inputs() {
    let Some(ctx) = device() else { return };
    let gen = wgsl_cuda::generate("add2", kernels::ADD2).expect("generate");
    // A source this run has never compiled before. The cache is a real
    // directory that outlives the process, so asserting "the first compile
    // misses" on a fixed source would pass once and then never again.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let src = format!("// cache probe {} {nonce}\n{}", std::process::id(), gen.source);

    let first = ctx.cubin(&src, &gen.entry).expect("compile");
    assert!(!first.cached, "the first compile of a fresh source cannot be a cache hit");
    let second = ctx.cubin(&src, &gen.entry).expect("compile");
    assert!(second.cached, "the second compile of the same source must hit the cache");
    assert_eq!(first.cubin, second.cubin, "the cache returned different code");

    // The key must separate what can change the output: a different entry name
    // and a different source are different keys.
    let other = ctx.cubin(&src.replace(&gen.entry, "brain_other"), "brain_other").expect("compile");
    assert_ne!(other.key, first.key, "entry name and source must take part in the cache key");
}
