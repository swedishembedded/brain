// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The launch contract a native CUDA kernel can state beyond its source:
//! dynamic shared memory with the opt-in past 48 KiB, thread-block clusters,
//! typed by-value scalar arguments, an occupancy query - and that a captured
//! graph keeps all of them.
//!
//! Swedish Embedded AB implements Hopper-class GPU kernels (thread-block
//! clusters, distributed shared memory, TMA) and the launch plumbing under
//! them for its clients. If your team needs expertise in kernels that use the
//! newest launch features while older cards keep a valid path, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! The probe kernels write a value that depends on the block, the element and
//! every scalar (a `u32`, an `f64` and an `i64`), so an argument passed at the
//! wrong width or slot, a launch that dropped the dynamic shared memory, or a
//! cluster that did not form each produce a wrong number rather than a
//! plausible one. The cluster kernel reads its PEER block's shared memory, so
//! it returns the right answer only if the two blocks really ran as one
//! cluster. Skipped without a device and NVRTC (or an AOT image); the cluster
//! cases skip on a device that cannot launch clusters.

use backend_api::{
    ArchFeatures, Backend as _, BindKind, CudaLaunch, DeviceBuffer, NativeId, NativeSpec, ScalarKind,
};
use backend_cuda::{live_resources, CudaBackend};
use std::sync::{Mutex, MutexGuard};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `out[block * n + i] = (block * n + i) * scale + bias`, staged through
/// dynamic shared memory and read back reversed, so the shared memory is
/// load-bearing. Valid on every architecture.
const SRC: &str = r#"
extern "C" __global__ void brain_dsmem_scalar(float *out, unsigned n, double scale, long long bias) {
    extern __shared__ float smem[];
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
        smem[i] = (float)((double)(blockIdx.x * n + i) * scale + (double)bias);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
        out[blockIdx.x * n + i] = smem[n - 1 - i];
}

// The same values, but each block reads the shared memory of the OTHER block
// of its two-block cluster (distributed shared memory), so it is right only if
// the pair really is one cluster.
extern "C" __global__ void brain_cluster_dsmem(float *out, unsigned n, double scale, long long bias) {
    extern __shared__ float smem[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
        smem[i] = (float)((double)(blockIdx.x * n + i) * scale + (double)bias);
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned rank, peer, local = (unsigned)__cvta_generic_to_shared(smem), remote;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    peer = rank ^ 1u;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(peer));
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float v;
        asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote + i * 4u));
        out[blockIdx.x * n + i] = v;
    }
    // Nobody may exit while a peer can still read its shared memory.
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
#else
    __trap();
#endif
}
"#;

const SCALARS: &[ScalarKind] = &[ScalarKind::U32, ScalarKind::F64, ScalarKind::I64];
const BLOCK: u32 = 128;
/// 32768 floats: 128 KiB, past the 48 KiB every device grants by default.
const N: u32 = 32768;
const BINDINGS: &[BindKind] = &[BindKind::StorageReadWrite];

fn spec(entry: &'static str, launch: CudaLaunch) -> NativeSpec {
    NativeSpec::Cuda { src: SRC, entry, block_dim: BLOCK, bindings: BINDINGS, shared_bytes: 0, launch }
}

fn launch(dynamic_bytes: u32, opt_in: bool, cluster: Option<[u32; 3]>) -> CudaLaunch {
    CudaLaunch { dynamic_shared_bytes: dynamic_bytes, shared_opt_in: opt_in, cluster, scalars: SCALARS, arch: ArchFeatures::Portable }
}

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(&[]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// The step's parameter words: `u32`, then each 64-bit scalar low word first.
fn params(n: u32, scale: f64, bias: i64) -> Vec<u32> {
    let (s, b) = (scale.to_bits(), bias as u64);
    vec![n, s as u32, (s >> 32) as u32, b as u32, (b >> 32) as u32]
}

fn want(blocks: u32, n: u32, scale: f64, bias: i64, reversed: bool) -> Vec<f32> {
    (0..blocks)
        .flat_map(|b| {
            (0..n).map(move |i| {
                // The probe stores element j and the kernel hands back element `n-1-i` (or the
                // peer block's `i`).
                let (blk, j) = if reversed { (b, n - 1 - i) } else { (b ^ 1, i) };
                ((f64::from(blk * n + j)) * scale + bias as f64) as f32
            })
        })
        .collect()
}

fn run(b: &CudaBackend, id: NativeId, out: &DeviceBuffer, blocks: u32, p: &[u32]) {
    let step = b.step_native(id, &[out], p, blocks).expect("the dispatch is accepted");
    b.submit(&[], &[step]);
    b.poll_wait();
}

fn supports_clusters(b: &CudaBackend) -> bool {
    // The driver's cluster-launch attribute is what decides; the probe kernel
    // itself is Hopper code, so anything older skips whatever the attribute says.
    if b.compute_capability().0 < 9 {
        brain_testutil::skip_unavailable("the device cannot launch thread-block clusters");
        return false;
    }
    true
}

#[test]
fn dynamic_shared_memory_past_the_default_limit_needs_the_opt_in() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let optin = b.native_shared_optin_bytes();
    if optin < N * 4 {
        brain_testutil::skip_unavailable("the device grants less than 128 KiB of opt-in shared memory");
        return;
    }
    assert!(
        b.register_native(&spec("brain_dsmem_scalar", launch(N * 4, false, None))).is_none(),
        "past 48 KiB without the opt-in the kernel must be declined, not launched into an error"
    );
    let id = b.register_native(&spec("brain_dsmem_scalar", launch(N * 4, true, None))).expect("opted in, and the device allows it");
    let out = b.storage((4 * N) as u64);
    let p = params(N, 0.5, -3);
    run(&b, id, &out, 4, &p);
    assert_eq!(b.read(&out, (4 * N) as usize), want(4, N, 0.5, -3, true));
    // More than the device grants is declined even when opted in.
    assert!(b.register_native(&spec("brain_dsmem_scalar", launch(optin + 4, true, None))).is_none());
}

#[test]
fn scalar_arguments_arrive_at_their_declared_type() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let id = b.register_native(&spec("brain_dsmem_scalar", launch(1024 * 4, false, None))).expect("registers");
    let out = b.storage(3 * 1024);
    // 1024 elements: values 0.. and an f64 scale that a 32-bit argument cannot hold, a negative i64.
    let p = params(1024, 1.0 / 3.0, -(1i64 << 40));
    run(&b, id, &out, 3, &p);
    assert_eq!(b.read(&out, 3 * 1024), want(3, 1024, 1.0 / 3.0, -(1i64 << 40), true));
    // The wrong number of words for the declared scalars is declined.
    assert!(b.step_native(id, &[&out], &p[..4], 3).is_none(), "a short parameter list");
    assert!(b.step_native(id, &[&out], &[p.clone(), vec![0]].concat(), 3).is_none(), "a long parameter list");
}

#[test]
fn scalars_and_a_uniform_block_cannot_both_be_declared() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let both = NativeSpec::Cuda {
        src: SRC,
        entry: "brain_dsmem_scalar",
        block_dim: BLOCK,
        bindings: &[BindKind::Uniform, BindKind::StorageReadWrite],
        shared_bytes: 0,
        launch: launch(1024 * 4, false, None),
    };
    assert!(b.register_native(&both).is_none());
}

#[test]
fn a_cluster_launch_forms_clusters_and_shares_shared_memory_across_the_pair() {
    let _s = serial();
    let Some(b) = backend() else { return };
    if !supports_clusters(&b) {
        return;
    }
    let id = b.register_native(&spec("brain_cluster_dsmem", launch(N * 4, true, Some([2, 1, 1])))).expect("a cluster kernel registers on Hopper");
    let out = b.storage((8 * N) as u64);
    let p = params(N, 0.25, 7);
    run(&b, id, &out, 8, &p);
    assert_eq!(b.read(&out, (8 * N) as usize), want(8, N, 0.25, 7, false), "each block must have read its peer's shared memory");
    // A grid that does not divide into clusters is not launched.
    assert!(b.step_native(id, &[&out], &p, 7).is_none());
}

#[test]
fn occupancy_is_the_drivers_answer_for_the_kernels_own_shared_memory() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let small = b.register_native(&spec("brain_dsmem_scalar", launch(4 * 1024, false, None))).expect("small");
    let many = b.native_max_active_blocks(small).expect("the driver answers");
    assert!(many >= 2, "a 4 KiB kernel of 128 threads must fit several blocks per multiprocessor, got {many}");
    if b.native_shared_optin_bytes() >= N * 4 {
        let big = b.register_native(&spec("brain_dsmem_scalar", launch(N * 4, true, None))).expect("big");
        let few = b.native_max_active_blocks(big).expect("the driver answers");
        assert!(few >= 1 && few < many, "128 KiB per block must lower occupancy ({few} vs {many})");
    }
    assert_eq!(b.native_max_active_blocks(NativeId(u32::MAX)), None, "an id nobody registered");
}

/// A repeated submission is captured into a graph; the replay keeps the
/// dynamic shared memory and the cluster, and takes this dispatch's scalars
/// and grid rather than the captured ones.
#[test]
fn a_captured_graph_keeps_the_launch_and_takes_each_replays_scalars_and_grid() {
    let _s = serial();
    let Some(b) = backend() else { return };
    if !supports_clusters(&b) {
        return;
    }
    let id = b.register_native(&spec("brain_cluster_dsmem", launch(N * 4, true, Some([2, 1, 1])))).expect("registers");
    let out = b.storage((16 * N) as u64);
    let before = b.launch_stats();
    let rounds: [(u32, f64, i64); 6] = [(4, 0.5, 1), (4, 0.5, 1), (4, 0.5, 1), (4, 2.0, -9), (4, 2.0, -9), (8, 2.0, -9)];
    for (blocks, scale, bias) in rounds {
        run(&b, id, &out, blocks, &params(N, scale, bias));
        let got = b.read(&out, (blocks * N) as usize);
        assert_eq!(got, want(blocks, N, scale, bias, false), "blocks {blocks} scale {scale} bias {bias}");
    }
    let after = b.launch_stats();
    assert!(after.graph_captures > before.graph_captures, "the repeated shape must have been captured: {after:?}");
    assert!(after.graph_replays > before.graph_replays, "and replayed: {after:?}");
    assert!(after.graph_scalar_updates > before.graph_scalar_updates, "a changed scalar is a node update, not a stale replay: {after:?}");
    assert!(after.grid_updates > before.grid_updates, "a grown grid is re-pointed with the cluster intact: {after:?}");
}

#[test]
fn a_native_kernel_with_the_v2_launch_returns_everything_it_took() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let register = |b: &CudaBackend| b.register_native(&spec("brain_dsmem_scalar", launch(4 * 1024, false, None))).expect("registers");
    // Warm: the driver's lazy state is not what is being measured.
    {
        let id = register(&b);
        let out = b.storage(1024);
        run(&b, id, &out, 1, &params(1024, 1.0, 0));
    }
    let base = live_resources();
    for _ in 0..5 {
        let b2 = backend().expect("backend");
        let id = register(&b2);
        let out = b2.storage(1024);
        run(&b2, id, &out, 1, &params(1024, 1.0, 0));
    }
    assert_eq!(live_resources(), base);
}
