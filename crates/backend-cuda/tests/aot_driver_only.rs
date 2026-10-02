// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A machine with a driver and no compiler runs the kernels it was shipped.
//!
//! Swedish Embedded AB implements deployment pipelines for GPU software for its
//! clients, including shipping compiled kernels to machines that carry a
//! driver and no toolchain. If your team needs expertise in making a CUDA
//! application run on a locked-down box without a compiler, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! NVRTC is a process-wide singleton, so "NVRTC is not there" cannot be made
//! true inside the process that builds the images. Each scenario therefore
//! builds its AOT directory here, then re-runs ONE test of this very binary as
//! a child with `BRAIN_NVRTC` pointing nowhere and `BRAIN_CUDA_AOT_DIR` at the
//! directory, and asserts that child ran to its marker line (a child that
//! skipped for lack of a device would otherwise look like a pass).
//!
//! Asserted:
//! - the AOT cubins run: a native kernel through `Context`, and a catalogue
//!   kernel through the `Backend`, both with no NVRTC loadable;
//! - the PTX fallback runs when no cubin matches, through the driver's own
//!   compiler;
//! - a kernel with no image is a clear error naming the remedy, and
//!   `register_native` declines (`None`) rather than panicking;
//! - a corrupted image is refused by its checksum before the driver sees it;
//! - the AOT image is consulted before NVRTC (a forged key shows which ran).

use backend_api::Backend as _;
use backend_cuda::aot::{self, Job, Plan};
use backend_cuda::exec::Context;
use backend_cuda::nvrtc::Target;
use backend_cuda::CudaBackend;
use std::path::{Path, PathBuf};

const CHILD: &str = "BRAIN_AOT_CHILD";
const MARKER: &str = "AOT-CHILD-OK";

const FILL: &str = r#"extern "C" __global__ void aot_fill(unsigned *out) { out[threadIdx.x] = 0xA070u + threadIdx.x; }"#;
/// Same entry point, different answer: tells which of two sources ran.
const FILL_OTHER: &str = r#"extern "C" __global__ void aot_fill(unsigned *out) { out[threadIdx.x] = 0xB070u + threadIdx.x; }"#;

const CATALOGUE: &[(&str, &str)] = &[("add2", kernels::ADD2)];

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("brain-aot-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The device and a loadable NVRTC, or the reason a scenario cannot be built.
fn builder() -> Option<Context> {
    let ctx = match Context::open(0) {
        Ok(c) => c,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            return None;
        }
    };
    if backend_cuda::nvrtc::version().is_err() {
        brain_testutil::skip_unavailable("no NVRTC to build the images with");
        return None;
    }
    Some(ctx)
}

fn native_job(src: &str) -> Job {
    Job { kernel: "native:aot_fill".into(), src: src.into(), entry: "aot_fill".into(), defines: vec![], min_cc: (5, 0) }
}

fn catalogue_jobs() -> Vec<Job> {
    CATALOGUE
        .iter()
        .map(|(name, wgsl)| {
            let g = wgsl_cuda::generate(name, wgsl).expect("add2 translates");
            Job { kernel: (*name).into(), src: g.source, entry: g.entry, defines: vec![], min_cc: (5, 0) }
        })
        .collect()
}

/// Build every job for this device's own architecture only, or PTX only.
fn build_into(dir: &Path, ctx: &Context, jobs: &[Job], cubin: bool) {
    let cc = ctx.compute_capability();
    let plan = Plan { cubins: if cubin { vec![Target::plain(cc)] } else { vec![] }, ptx: Some(cc) };
    let report = aot::build(dir, jobs, &plan).expect("build");
    assert!(report.built >= jobs.len(), "{report:?}");
}

/// Re-run `child` of this binary with NVRTC unloadable (or as given) and the
/// AOT directory set; assert it reached its marker.
fn run_child(child: &str, dir: &Path, nvrtc: Option<&str>) {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", child, "--nocapture", "--test-threads=1"]).env(CHILD, "1").env(aot::ENV_DIR, dir);
    if let Some(n) = nvrtc {
        cmd.env("BRAIN_NVRTC", n);
    }
    let out = cmd.output().expect("spawn the child test");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains(MARKER), "child `{child}` failed:\n{text}");
}

fn in_child() -> bool {
    std::env::var(CHILD).is_ok()
}

fn read_words(ctx: &Context, mem: &backend_cuda::exec::DeviceMem, n: usize) -> Vec<u32> {
    let mut bytes = vec![0u8; n * 4];
    ctx.download(mem, &mut bytes).unwrap();
    bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Launch `entry` of `src` on 32 threads and return what it wrote.
fn run_fill(ctx: &Context, src: &str) -> Result<Vec<u32>, String> {
    let module = ctx.compile(src, "aot_fill")?;
    let f = module.function("aot_fill")?;
    let out = ctx.alloc(32 * 4)?;
    ctx.zero(&out)?;
    ctx.launch(&f, (1, 1, 1), (32, 1, 1), &[&out])?;
    Ok(read_words(ctx, &out, 32))
}

fn expect_fill(words: &[u32], base: u32) {
    assert_eq!(words, (0..32).map(|i| base + i).collect::<Vec<_>>());
}

// --- scenario: cubins, no NVRTC -------------------------------------------

#[test]
fn aot_cubins_run_with_no_nvrtc() {
    let Some(ctx) = builder() else { return };
    let dir = scratch("cubins");
    let mut jobs = catalogue_jobs();
    jobs.push(native_job(FILL));
    build_into(&dir, &ctx, &jobs, true);
    drop(ctx);
    run_child("child_cubins_run_with_no_nvrtc", &dir, Some("no-such-dir/libnvrtc.so"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn child_cubins_run_with_no_nvrtc() {
    if !in_child() {
        return;
    }
    assert!(backend_cuda::nvrtc::version().is_err(), "the child must have no NVRTC");
    let ctx = Context::open(0).expect("device");
    expect_fill(&run_fill(&ctx, FILL).expect("an AOT cubin runs"), 0xA070);

    // A catalogue kernel through the Backend: generated at run time (cheap),
    // never compiled, taken from the AOT directory.
    let b = CudaBackend::try_new(CATALOGUE).expect("backend");
    let n = 256usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| 1000.0 + i as f32).collect();
    let (bx, by, out) = (b.storage_init("x", &x), b.storage_init("y", &y), b.storage(n as u64));
    b.submit(&[], &[b.step(0, &[&bx, &by, &out], &[n as u32], n as u32)]);
    let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_eq!(b.read(&out, n), want);

    // A native kernel through register_native.
    let spec = backend_api::NativeSpec::Cuda {
        src: FILL,
        entry: "aot_fill",
        block_dim: 32,
        bindings: &[backend_api::BindKind::StorageReadWrite],
        shared_bytes: 0,
        launch: backend_api::CudaLaunch::NONE,
    };
    assert!(b.register_native(&spec).is_some(), "a native kernel with an AOT image registers without NVRTC");
    println!("{MARKER}");
}

// --- scenario: PTX fallback ------------------------------------------------

#[test]
fn aot_ptx_is_compiled_by_the_driver_when_no_cubin_exists() {
    let Some(ctx) = builder() else { return };
    let dir = scratch("ptx");
    build_into(&dir, &ctx, &[native_job(FILL)], false);
    let store = aot::Store::open(&dir).unwrap().expect("manifest");
    assert!(store.entries().iter().all(|e| e.kind == aot::Kind::Ptx), "this scenario ships PTX only");
    drop(ctx);
    run_child("child_ptx_is_compiled_by_the_driver", &dir, Some("no-such-dir/libnvrtc.so"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn child_ptx_is_compiled_by_the_driver() {
    if !in_child() {
        return;
    }
    assert!(backend_cuda::nvrtc::version().is_err());
    let ctx = Context::open(0).expect("device");
    expect_fill(&run_fill(&ctx, FILL).expect("the driver JIT-compiles the PTX"), 0xA070);
    println!("{MARKER}");
}

// --- scenario: no image -----------------------------------------------------

#[test]
fn a_kernel_with_no_image_is_a_clear_error_and_not_a_crash() {
    let Some(ctx) = builder() else { return };
    let dir = scratch("missing");
    build_into(&dir, &ctx, &[native_job(FILL)], true);
    drop(ctx);
    run_child("child_missing_image_is_a_clear_error", &dir, Some("no-such-dir/libnvrtc.so"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn child_missing_image_is_a_clear_error() {
    if !in_child() {
        return;
    }
    let ctx = Context::open(0).expect("device");
    // Same entry point, source the build never saw.
    let e = match run_fill(&ctx, FILL_OTHER) {
        Ok(_) => panic!("a kernel nobody built must not run"),
        Err(e) => e,
    };
    assert!(e.contains("no usable binary") && e.contains("aot_fill"), "{e}");
    assert!(e.contains("make cuda/aot"), "the error must name the remedy: {e}");
    assert!(e.contains("NVRTC"), "and say NVRTC was tried: {e}");

    let b = CudaBackend::try_new(CATALOGUE).expect("backend");
    let spec = backend_api::NativeSpec::Cuda {
        src: FILL_OTHER,
        entry: "aot_fill",
        block_dim: 32,
        bindings: &[backend_api::BindKind::StorageReadWrite],
        shared_bytes: 0,
        launch: backend_api::CudaLaunch::NONE,
    };
    assert!(b.register_native(&spec).is_none(), "the provider is told no, and falls back");
    println!("{MARKER}");
}

// --- scenario: corrupted image ---------------------------------------------

#[test]
fn a_corrupted_image_is_refused_before_the_driver_sees_it() {
    let Some(ctx) = builder() else { return };
    let dir = scratch("corrupt");
    build_into(&dir, &ctx, &[native_job(FILL)], true);
    let store = aot::Store::open(&dir).unwrap().unwrap();
    for e in store.entries() {
        let path = dir.join(&e.file);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();
    }
    drop(ctx);
    run_child("child_corrupt_image_is_refused", &dir, Some("no-such-dir/libnvrtc.so"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn child_corrupt_image_is_refused() {
    if !in_child() {
        return;
    }
    let ctx = Context::open(0).expect("device");
    let e = run_fill(&ctx, FILL).expect_err("a corrupted image must not load");
    assert!(e.contains("checksum"), "{e}");
    println!("{MARKER}");
}

// --- scenario: AOT before NVRTC ---------------------------------------------

#[test]
fn an_aot_image_is_used_in_preference_to_compiling() {
    let Some(ctx) = builder() else { return };
    let dir = scratch("precedence");
    // Build FILL, then file its image under the key FILL_OTHER would look up.
    // If the loader consulted NVRTC first it would compile FILL_OTHER and the
    // child would see 0xB070; seeing 0xA070 proves the image was taken.
    build_into(&dir, &ctx, &[native_job(FILL)], true);
    let arch = Target::plain(ctx.compute_capability()).name();
    let text = std::fs::read_to_string(dir.join(aot::MANIFEST_FILE)).unwrap();
    let mut m = aot::Manifest::parse(&text).unwrap();
    let forged = aot::aot_key(FILL_OTHER, "aot_fill", &arch, &[]);
    for e in m.entries.iter_mut().filter(|e| e.kind == aot::Kind::Cubin) {
        e.key = forged.clone();
    }
    std::fs::write(dir.join(aot::MANIFEST_FILE), m.render()).unwrap();
    drop(ctx);
    // NVRTC left available, so a loader that preferred it could run.
    run_child("child_aot_precedes_nvrtc", &dir, None);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn child_aot_precedes_nvrtc() {
    if !in_child() {
        return;
    }
    assert!(backend_cuda::nvrtc::version().is_ok(), "this scenario keeps NVRTC available");
    let ctx = Context::open(0).expect("device");
    expect_fill(&run_fill(&ctx, FILL_OTHER).expect("runs"), 0xA070);
    println!("{MARKER}");
}

// --- generated source is deterministic (the AOT key depends on it) ----------

/// The key is a hash of the generated CUDA text, so a generator that emits the
/// same kernel differently from one run to the next would make every image
/// miss. Each translation runs twice in one process, where hash-map iteration
/// order already differs between instances.
#[test]
fn the_generator_is_deterministic_for_the_whole_catalogue() {
    for &(name, wgsl) in kernels::ALL {
        let (Ok(a), Ok(b)) = (wgsl_cuda::generate(name, wgsl), wgsl_cuda::generate(name, wgsl)) else { continue };
        assert_eq!((a.source, a.entry), (b.source, b.entry), "{name}: two translations of one kernel differ");
    }
}
