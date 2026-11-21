// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device collectives over NCCL (`libnccl.so.2`), reached by `dlopen` exactly as
//! `libcuda` and cuBLAS are: never linked, so a box without the library simply
//! has no device collectives and the caller reduces through the host, saying so.
//!
//! Swedish Embedded AB implements multi-GPU training and inference runtimes for
//! its clients. If your team needs expertise in keeping tensor-parallel and
//! data-parallel traffic on the GPU interconnect instead of staging it through
//! host memory, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Two shapes of communicator:
//!
//! - [`NcclGroup`]: one process drives every rank (`ncclCommInitAll`), the
//!   shape a tensor-parallel server uses across the cards of one process.
//! - [`NcclRank`]: one rank per process, one card per process, as brain runs a
//!   distributed job. Rank 0 creates the rendezvous id ([`unique_id`]), the
//!   job hands it to every rank over its authenticated channel, and each rank
//!   joins with `ncclCommInitRankConfig`. The communicator is nonblocking, so
//!   neither its creation, a split, nor a collective can hang a rank on a peer
//!   that never arrives: each is bounded by a deadline after which the
//!   communicator is aborted (`ncclCommAbort`) and the call fails.
//!
//! Either way each rank owns a [`CudaBackend`] handle of its device with no
//! kernels, so the collective runs on that handle's own stream and is ordered
//! against the kernels that produced and will consume its buffers by the same
//! event fences every other cross-handle use goes through: nothing waits for the
//! whole device.
//!
//! NCCL's own configuration (`NCCL_*` variables a launcher sets: network
//! plugin, interfaces, debug output) is read by NCCL itself; brain reads none
//! of it.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use backend_api::{CollectiveDtype, CollectiveId, DeviceBuffer, DeviceCollective, RankCollective};

use crate::backend::CudaBackend;
use crate::driver::CuStream;

type NcclResult = c_int;
type Comm = *mut c_void;

const SUCCESS: NcclResult = 0;
/// `ncclInProgress`: a nonblocking communicator has not finished the call yet.
const IN_PROGRESS: NcclResult = 7;
/// `ncclFloat32`.
const FLOAT32: c_int = 7;
/// `ncclBfloat16`.
const BFLOAT16: c_int = 9;
/// `ncclSum`.
const SUM: c_int = 0;
/// `NCCL_SPLIT_NOCOLOR`.
const NO_COLOR: c_int = -1;
/// `ncclScalarHostImmediate`.
const SCALAR_HOST: c_int = 1;
/// `NCCL_API_MAGIC`.
const API_MAGIC: u32 = 0xcafe_beef;
/// `NCCL_CONFIG_UNDEF_INT`.
const UNDEF: c_int = c_int::MIN;
/// The `NCCL_VERSION_CODE` whose `ncclConfig_t` [`Config`] mirrors (2.32.3).
const CONFIG_VERSION: u32 = 23203;
/// The oldest library the per-process communicator is used with: nonblocking configs, `ncclCommSplit` and
/// `ncclCommFinalize` all exist from 2.18.
const MIN_RANK_VERSION: c_int = 21800;

/// `ncclUniqueId`, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
struct UniqueId([u8; CollectiveId::BYTES]);

/// `ncclConfig_t` as `nccl.h` 2.32 declares it (`ncclConfig_v23100`); every field brain does not set is
/// `NCCL_CONFIG_UNDEF_*`, as `NCCL_CONFIG_INITIALIZER` leaves it. A library reads the prefix `size` covers.
#[repr(C)]
struct Config {
    size: usize,
    magic: u32,
    version: u32,
    blocking: c_int,
    cga_cluster_size: c_int,
    min_ctas: c_int,
    max_ctas: c_int,
    net_name: *const c_char,
    split_share: c_int,
    traffic_class: c_int,
    comm_name: *const c_char,
    /// `collnetEnable` through `nvlsHostMode`: fifteen integer attributes, all left to the library.
    rest: [c_int; 15],
}

impl Config {
    /// The library's defaults, except that calls return at once and report completion through
    /// `ncclCommGetAsyncError` (`blocking = 0`), which is what lets a deadline abort them.
    fn nonblocking() -> Config {
        Config {
            size: std::mem::size_of::<Config>(),
            magic: API_MAGIC,
            version: CONFIG_VERSION,
            blocking: 0,
            cga_cluster_size: UNDEF,
            min_ctas: UNDEF,
            max_ctas: UNDEF,
            net_name: std::ptr::null(),
            split_share: UNDEF,
            traffic_class: UNDEF,
            comm_name: std::ptr::null(),
            rest: [UNDEF; 15],
        }
    }
}

/// The SONAME tried; the major version is part of it.
const SONAMES: &[&str] = &["libnccl.so.2"];

struct Lib {
    _lib: libloading::Library,
    comm_init_all: unsafe extern "C" fn(*mut Comm, c_int, *const c_int) -> NcclResult,
    comm_destroy: unsafe extern "C" fn(Comm) -> NcclResult,
    group_start: unsafe extern "C" fn() -> NcclResult,
    group_end: unsafe extern "C" fn() -> NcclResult,
    all_reduce: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, CuStream) -> NcclResult,
    all_gather: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, Comm, CuStream) -> NcclResult,
    get_error_string: unsafe extern "C" fn(NcclResult) -> *const c_char,
    /// The per-process communicator's entry points, or why this library cannot provide one.
    rank: Result<RankFns, String>,
}

/// What a one-rank-per-process communicator needs beyond [`Lib`]'s single-process set.
struct RankFns {
    get_unique_id: unsafe extern "C" fn(*mut UniqueId) -> NcclResult,
    comm_init_rank_config: unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int, *mut Config) -> NcclResult,
    comm_split: unsafe extern "C" fn(Comm, c_int, c_int, *mut Comm, *mut Config) -> NcclResult,
    comm_finalize: unsafe extern "C" fn(Comm) -> NcclResult,
    comm_abort: unsafe extern "C" fn(Comm) -> NcclResult,
    comm_get_async_error: unsafe extern "C" fn(Comm, *mut NcclResult) -> NcclResult,
    comm_count: unsafe extern "C" fn(Comm, *mut c_int) -> NcclResult,
    comm_user_rank: unsafe extern "C" fn(Comm, *mut c_int) -> NcclResult,
    reduce_scatter: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, CuStream) -> NcclResult,
    broadcast: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, CuStream) -> NcclResult,
    red_op_create_pre_mul_sum: unsafe extern "C" fn(*mut c_int, *mut c_void, c_int, c_int, Comm) -> NcclResult,
    red_op_destroy: unsafe extern "C" fn(c_int, Comm) -> NcclResult,
}

// A mapped library plus bare function pointers, immutable after load.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: std::sync::OnceLock<Result<Lib, String>> = std::sync::OnceLock::new();

fn lib() -> Result<&'static Lib, &'static str> {
    LIB.get_or_init(load).as_ref().map_err(String::as_str)
}

fn load() -> Result<Lib, String> {
    let mut tried = Vec::new();
    for soname in crate::nvrtc::toolkit_library_candidates(SONAMES) {
        // SAFETY: loading a shared object runs its initialisers; NCCL's are ordinary.
        let lib = match unsafe { libloading::Library::new(&soname) } {
            Ok(l) => l,
            Err(e) => {
                tried.push(format!("{soname}: {e}"));
                continue;
            }
        };
        // SAFETY: each name is resolved at the signature `nccl.h` declares for it; the pointers live as long
        // as `lib`, which the returned struct owns.
        let built = unsafe {
            (|| -> Result<Lib, String> {
                Ok(Lib {
                    comm_init_all: sym(&lib, b"ncclCommInitAll\0")?,
                    comm_destroy: sym(&lib, b"ncclCommDestroy\0")?,
                    group_start: sym(&lib, b"ncclGroupStart\0")?,
                    group_end: sym(&lib, b"ncclGroupEnd\0")?,
                    all_reduce: sym(&lib, b"ncclAllReduce\0")?,
                    all_gather: sym(&lib, b"ncclAllGather\0")?,
                    get_error_string: sym(&lib, b"ncclGetErrorString\0")?,
                    rank: rank_fns(&lib),
                    _lib: lib,
                })
            })()
        };
        match built {
            Ok(l) => return Ok(l),
            Err(e) => tried.push(format!("{soname}: {e}")),
        }
    }
    Err(format!("NCCL is not available ({})", tried.join("; ")))
}

/// The per-process entry points of `lib`, refusing a library older than [`MIN_RANK_VERSION`].
///
/// # Safety
/// `lib` must be a loaded NCCL; each name is resolved at the signature `nccl.h` declares for it.
unsafe fn rank_fns(lib: &libloading::Library) -> Result<RankFns, String> {
    let get_version: unsafe extern "C" fn(*mut c_int) -> NcclResult = sym(lib, b"ncclGetVersion\0")?;
    let mut version: c_int = 0;
    if get_version(&mut version) != SUCCESS || version < MIN_RANK_VERSION {
        return Err(format!("NCCL {version} predates the one-rank-per-process communicator ({MIN_RANK_VERSION} or newer)"));
    }
    Ok(RankFns {
        get_unique_id: sym(lib, b"ncclGetUniqueId\0")?,
        comm_init_rank_config: sym(lib, b"ncclCommInitRankConfig\0")?,
        comm_split: sym(lib, b"ncclCommSplit\0")?,
        comm_finalize: sym(lib, b"ncclCommFinalize\0")?,
        comm_abort: sym(lib, b"ncclCommAbort\0")?,
        comm_get_async_error: sym(lib, b"ncclCommGetAsyncError\0")?,
        comm_count: sym(lib, b"ncclCommCount\0")?,
        comm_user_rank: sym(lib, b"ncclCommUserRank\0")?,
        reduce_scatter: sym(lib, b"ncclReduceScatter\0")?,
        broadcast: sym(lib, b"ncclBroadcast\0")?,
        red_op_create_pre_mul_sum: sym(lib, b"ncclRedOpCreatePreMulSum\0")?,
        red_op_destroy: sym(lib, b"ncclRedOpDestroy\0")?,
    })
}

/// # Safety
/// `T` must be the exact ABI signature the library declares for `name`.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    lib.get::<T>(name).map(|s| *s).map_err(|e| format!("no {}: {e}", String::from_utf8_lossy(&name[..name.len() - 1])))
}

impl Lib {
    fn check(&self, rc: NcclResult, what: &str) -> Result<(), String> {
        if rc == SUCCESS {
            return Ok(());
        }
        // SAFETY: `ncclGetErrorString` returns a pointer to a static string for any result code.
        let text = unsafe {
            let p = (self.get_error_string)(rc);
            if p.is_null() { format!("ncclResult_t {rc}") } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        Err(format!("{what} failed: {text}"))
    }
}

/// Whether the library can be loaded at all, with the reason when it cannot.
pub fn availability() -> Result<(), String> {
    lib().map(|_| ()).map_err(str::to_string)
}

/// The library and its per-process entry points, or why there are none.
fn rank_lib() -> Result<(&'static Lib, &'static RankFns), String> {
    let lib = lib().map_err(str::to_string)?;
    lib.rank.as_ref().map(|r| (lib, r)).map_err(Clone::clone)
}

/// Whether the library provides the one-rank-per-process communicator, with the reason when it does not.
pub fn rank_support() -> Result<(), String> {
    rank_lib().map(|_| ())
}

/// A fresh rendezvous id (`ncclGetUniqueId`) for a communicator of one rank per process. It names the creating
/// host's bootstrap endpoint and a random magic: whoever holds it can join, so it travels only over a channel the
/// job authenticates.
pub fn unique_id() -> Result<CollectiveId, String> {
    let (lib, fns) = rank_lib()?;
    let mut id = UniqueId([0; CollectiveId::BYTES]);
    // SAFETY: `id` is a writable `ncclUniqueId`.
    lib.check(unsafe { (fns.get_unique_id)(&mut id) }, "ncclGetUniqueId")?;
    Ok(CollectiveId(id.0))
}

/// A single-process communicator over several CUDA devices.
pub struct NcclGroup {
    lib: &'static Lib,
    /// One communicator per rank, as addresses so the struct is `Send`.
    comms: Vec<usize>,
    /// One kernel-less handle per rank: its stream carries the collective and its fences order it.
    ranks: Vec<CudaBackend>,
    /// Collectives of one communicator must be enqueued in the same order on every rank; two threads
    /// interleaving their per-rank calls could swap that order, so they take turns.
    turn: Mutex<()>,
}

// The communicators are used under `turn`; the handles are `Send + Sync` themselves.
unsafe impl Send for NcclGroup {}
unsafe impl Sync for NcclGroup {}

impl NcclGroup {
    /// A communicator over the devices at CUDA `ordinals` (rank = position). `Err` when the library is
    /// absent, an ordinal repeats, or NCCL refuses the set.
    pub fn open(ordinals: &[u32]) -> Result<NcclGroup, String> {
        let lib = lib().map_err(str::to_string)?;
        if ordinals.len() < 2 {
            return Err(format!("a collective needs at least two devices, got {}", ordinals.len()));
        }
        for (i, o) in ordinals.iter().enumerate() {
            if ordinals[..i].contains(o) {
                return Err(format!("CUDA device {o} appears twice in the communicator"));
            }
        }
        let ranks = ordinals.iter().map(|&o| CudaBackend::try_new_on_ordinal(&[], o)).collect::<Result<Vec<_>, _>>()?;
        let devlist: Vec<c_int> = ordinals.iter().map(|&o| o as c_int).collect();
        let mut comms: Vec<Comm> = vec![std::ptr::null_mut(); ordinals.len()];
        // SAFETY: `comms` has room for one communicator per entry of `devlist`, which names live devices.
        lib.check(unsafe { (lib.comm_init_all)(comms.as_mut_ptr(), devlist.len() as c_int, devlist.as_ptr()) }, "ncclCommInitAll")?;
        Ok(NcclGroup { lib, comms: comms.into_iter().map(|c| c as usize).collect(), ranks, turn: Mutex::new(()) })
    }

    fn check_ranks(&self, what: &str, lens: &[usize]) -> Result<(), String> {
        for l in lens {
            if *l != self.ranks.len() {
                return Err(format!("{what}: {} buffers for a communicator of {} ranks", l, self.ranks.len()));
            }
        }
        Ok(())
    }

    /// Enqueue `op` once per rank inside one NCCL group, between the fences that order the ranks' streams against
    /// the other handles' work on `touched` (per rank).
    fn collective(&self, touched: &[Vec<&DeviceBuffer>], what: &str, mut op: impl FnMut(usize, &Lib, Comm, CuStream) -> NcclResult) -> Result<(), String> {
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        for (rank, bufs) in touched.iter().enumerate() {
            self.ranks[rank].fence_collective_in(bufs)?;
        }
        // SAFETY: every call below is made on a communicator this group owns and a stream of the same device.
        unsafe {
            self.lib.check((self.lib.group_start)(), "ncclGroupStart")?;
            let mut first_error = Ok(());
            for rank in 0..self.ranks.len() {
                let rc = op(rank, self.lib, self.comms[rank] as Comm, self.ranks[rank].stream());
                if first_error.is_ok() {
                    first_error = self.lib.check(rc, what);
                }
            }
            let ended = self.lib.check((self.lib.group_end)(), "ncclGroupEnd");
            first_error.and(ended)?;
        }
        for (rank, bufs) in touched.iter().enumerate() {
            self.ranks[rank].fence_collective_out(bufs);
        }
        Ok(())
    }
}

impl DeviceCollective for NcclGroup {
    fn provider(&self) -> &'static str {
        "nccl"
    }

    fn all_reduce_sum(&self, bufs: &[&DeviceBuffer], words: u64) -> Result<(), String> {
        self.check_ranks("all_reduce_sum", &[bufs.len()])?;
        let addrs = bufs.iter().map(|b| bounded(b, 0, words, "all_reduce_sum")).collect::<Result<Vec<_>, _>>()?;
        let touched: Vec<Vec<&DeviceBuffer>> = bufs.iter().map(|b| vec![*b]).collect();
        self.collective(&touched, "ncclAllReduce", |rank, lib, comm, stream| {
            let p = addrs[rank] as *mut c_void;
            // SAFETY: `p` addresses at least `words` f32 words of this rank's device (checked by `bounded`);
            // an in-place reduction (send == recv) is NCCL's documented fast path.
            unsafe { (lib.all_reduce)(p, p, words as usize, FLOAT32, SUM, comm, stream) }
        })
    }

    fn all_gather(&self, send: &[&DeviceBuffer], recv: &[&DeviceBuffer], words: u64) -> Result<(), String> {
        self.check_ranks("all_gather", &[send.len(), recv.len()])?;
        let ranks = self.ranks.len() as u64;
        let from = send.iter().map(|b| bounded(b, 0, words, "all_gather send")).collect::<Result<Vec<_>, _>>()?;
        let to = recv.iter().map(|b| bounded(b, 0, words * ranks, "all_gather recv")).collect::<Result<Vec<_>, _>>()?;
        let touched: Vec<Vec<&DeviceBuffer>> = send.iter().zip(recv).map(|(s, r)| vec![*s, *r]).collect();
        self.collective(&touched, "ncclAllGather", |rank, lib, comm, stream| {
            // SAFETY: the send range holds `words` words and the receive range `ranks * words`, both on this
            // rank's device (checked by `bounded`).
            unsafe { (lib.all_gather)(from[rank] as *const c_void, to[rank] as *mut c_void, words as usize, FLOAT32, comm, stream) }
        })
    }
}

// ---- one rank per process ------------------------------------------------------------------------------------

/// How often a deadline-bounded wait asks the library and the stream whether the work is done.
const POLL: Duration = Duration::from_micros(50);

/// Poll a nonblocking communicator until its pending call (`what`) has finished: `Ok` once the library reports
/// success, the library's error when it fails, and an error naming the deadline when `deadline` passes first.
fn settle(lib: &Lib, fns: &RankFns, comm: Comm, deadline: Instant, what: &str) -> Result<(), String> {
    loop {
        let mut state = SUCCESS;
        // SAFETY: `comm` is a live communicator of this library and `state` a writable result.
        lib.check(unsafe { (fns.comm_get_async_error)(comm, &mut state) }, "ncclCommGetAsyncError")?;
        match state {
            SUCCESS => return Ok(()),
            IN_PROGRESS if Instant::now() >= deadline => return Err(format!("{what} did not complete before its deadline: a peer did not take part")),
            IN_PROGRESS => std::thread::sleep(POLL),
            failed => return lib.check(failed, what),
        }
    }
}

/// The NCCL type of `dtype`.
fn nccl_type(dtype: CollectiveDtype) -> c_int {
    match dtype {
        CollectiveDtype::F32 => FLOAT32,
        CollectiveDtype::Bf16 => BFLOAT16,
    }
}

/// `x` rounded to the nearest bfloat16 (ties to even), as its bits.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The device address of `buf` after checking it holds `count` elements of `dtype`.
fn bounded_elems(buf: &DeviceBuffer, count: u64, dtype: CollectiveDtype, what: &str) -> Result<u64, String> {
    let (ptr, len) = CudaBackend::address_of(buf)?;
    let bytes = count.checked_mul(dtype.bytes()).ok_or_else(|| format!("{what}: {count} elements overflow"))?;
    if bytes > len {
        return Err(format!("{what}: {count} {dtype:?} elements do not fit a {len}-byte buffer"));
    }
    Ok(ptr)
}

/// The communicator of a [`NcclRank`], and why it is gone once it is.
struct RankState {
    /// The live communicator as an address (so the rank is `Send`); `0` once aborted or destroyed.
    comm: usize,
    /// Why the communicator was aborted, repeated to every later call.
    failed: Option<String>,
}

impl RankState {
    fn live(&self) -> Result<Comm, String> {
        match (&self.failed, self.comm) {
            (Some(why), _) => Err(format!("the communicator was aborted: {why}")),
            (None, 0) => Err("the communicator is closed".into()),
            (None, comm) => Ok(comm as Comm),
        }
    }
}

/// One rank of a communicator whose ranks are separate processes, each on one card (`ncclCommInitRankConfig`,
/// nonblocking). See the module documentation and [`RankCollective`] for the contract.
pub struct NcclRank {
    lib: &'static Lib,
    fns: &'static RankFns,
    state: Mutex<RankState>,
    /// A kernel-less handle of this rank's device: its stream carries the collectives and its fences order them.
    handle: CudaBackend,
    world: usize,
    rank: usize,
    /// How long a call may wait for the library to accept it (the first use of a peer connects to it).
    settle: Duration,
}

// The communicator is used under `state`; the handle is `Send + Sync` itself.
unsafe impl Send for NcclRank {}
unsafe impl Sync for NcclRank {}

impl NcclRank {
    /// Join the communicator `id` names as `rank` of `world` on CUDA device `ordinal`, or fail once `timeout` passes
    /// without every rank having joined (the half-made communicator is aborted, nothing is left behind).
    pub fn join(ordinal: u32, id: &CollectiveId, world: usize, rank: usize, timeout: Duration) -> Result<NcclRank, String> {
        if world == 0 || rank >= world || world > c_int::MAX as usize {
            return Err(format!("rank {rank} of {world} is not a place in a communicator"));
        }
        let (lib, fns) = rank_lib()?;
        let handle = CudaBackend::try_new_on_ordinal(&[], ordinal)?;
        handle.bind_context()?;
        let deadline = Instant::now() + timeout;
        let mut comm: Comm = std::ptr::null_mut();
        let mut config = Config::nonblocking();
        // SAFETY: `comm` receives the new communicator; `config` is a valid `ncclConfig_t` for the call's duration;
        // the device is the one whose context is current.
        let rc = unsafe { (fns.comm_init_rank_config)(&mut comm, world as c_int, UniqueId(id.0), rank as c_int, &mut config) };
        let what = format!("ncclCommInitRankConfig (rank {rank} of {world})");
        let joined = match rc {
            SUCCESS | IN_PROGRESS if !comm.is_null() => settle(lib, fns, comm, deadline, &what),
            SUCCESS | IN_PROGRESS => Err(format!("{what} returned no communicator")),
            failed => lib.check(failed, &what),
        };
        if let Err(e) = joined {
            if !comm.is_null() {
                // SAFETY: `comm` was created above and is not used again.
                unsafe { (fns.comm_abort)(comm) };
            }
            return Err(e);
        }
        Ok(NcclRank { lib, fns, state: Mutex::new(RankState { comm: comm as usize, failed: None }), handle, world, rank, settle: timeout })
    }

    /// Abort the communicator (in-flight kernels stop, peers see the failure) and remember why.
    fn fail(&self, state: &mut RankState, why: String) -> String {
        if state.comm != 0 {
            // SAFETY: `state.comm` is a live communicator of this library; it is not used after this.
            unsafe { (self.fns.comm_abort)(state.comm as Comm) };
            state.comm = 0;
        }
        state.failed.get_or_insert(why).clone()
    }

    /// Enqueue `op` on this rank's stream, between the fences that order it against the other handles' work on
    /// `touched`, and wait (bounded) for the library to accept it. A failure aborts the communicator.
    fn enqueue(&self, touched: &[&DeviceBuffer], what: &str, op: impl FnOnce(Comm, CuStream) -> NcclResult) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let comm = state.live()?;
        self.handle.bind_context()?;
        self.handle.fence_collective_in(touched)?;
        let rc = op(comm, self.handle.stream());
        let accepted = match rc {
            SUCCESS | IN_PROGRESS => settle(self.lib, self.fns, comm, Instant::now() + self.settle, what),
            failed => self.lib.check(failed, what),
        };
        self.handle.fence_collective_out(touched);
        accepted.map_err(|e| self.fail(&mut state, e))
    }

    /// `scale` as the host-immediate scalar of a pre-multiplied sum, in `dtype`'s own encoding.
    fn premul_sum(&self, comm: Comm, dtype: CollectiveDtype, scale: f32) -> Result<c_int, NcclResult> {
        let mut op: c_int = SUM;
        let (mut f, mut h) = (scale, bf16_bits(scale));
        let scalar: *mut c_void = match dtype {
            CollectiveDtype::F32 => (&mut f as *mut f32).cast(),
            CollectiveDtype::Bf16 => (&mut h as *mut u16).cast(),
        };
        // SAFETY: `scalar` points at one value of `dtype`, read during the call (host-immediate residence).
        match unsafe { (self.fns.red_op_create_pre_mul_sum)(&mut op, scalar, nccl_type(dtype), SCALAR_HOST, comm) } {
            SUCCESS => Ok(op),
            failed => Err(failed),
        }
    }
}

impl RankCollective for NcclRank {
    fn provider(&self) -> &'static str {
        "nccl"
    }

    fn world(&self) -> usize {
        self.world
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_sum(&self, bufs: &[(&DeviceBuffer, u64)], dtype: CollectiveDtype, scale: f32) -> Result<(), String> {
        let addrs = bufs.iter().map(|(b, n)| bounded_elems(b, *n, dtype, "all_reduce_sum")).collect::<Result<Vec<_>, _>>()?;
        if !scale.is_finite() {
            return Err(format!("all_reduce_sum: a scale of {scale} is not a number to multiply by"));
        }
        let touched: Vec<&DeviceBuffer> = bufs.iter().map(|(b, _)| *b).collect();
        self.enqueue(&touched, "ncclAllReduce", |comm, stream| {
            let op = if scale == 1.0 {
                SUM
            } else {
                match self.premul_sum(comm, dtype, scale) {
                    Ok(op) => op,
                    Err(rc) => return rc,
                }
            };
            // SAFETY: each address holds its `count` elements on this rank's device (checked above); in place
            // (send == recv) is NCCL's documented fast path; the custom op is destroyed only after the group that
            // uses it has been enqueued.
            unsafe {
                let mut rc = (self.lib.group_start)();
                for (addr, (_, count)) in addrs.iter().zip(bufs) {
                    if rc != SUCCESS {
                        break;
                    }
                    let p = *addr as *mut c_void;
                    rc = (self.lib.all_reduce)(p, p, *count as usize, nccl_type(dtype), op, comm, stream);
                }
                let ended = (self.lib.group_end)();
                if op != SUM {
                    (self.fns.red_op_destroy)(op, comm);
                }
                if rc == SUCCESS { ended } else { rc }
            }
        })
    }

    fn all_gather(&self, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype) -> Result<(), String> {
        let from = bounded_elems(send, count, dtype, "all_gather send")?;
        let to = bounded_elems(recv, count * self.world as u64, dtype, "all_gather recv")?;
        self.enqueue(&[send, recv], "ncclAllGather", |comm, stream| {
            // SAFETY: both ranges are on this rank's device and hold what the op reads and writes (checked above).
            unsafe { (self.lib.all_gather)(from as *const c_void, to as *mut c_void, count as usize, nccl_type(dtype), comm, stream) }
        })
    }

    fn reduce_scatter(&self, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype) -> Result<(), String> {
        let from = bounded_elems(send, count * self.world as u64, dtype, "reduce_scatter send")?;
        let to = bounded_elems(recv, count, dtype, "reduce_scatter recv")?;
        self.enqueue(&[send, recv], "ncclReduceScatter", |comm, stream| {
            // SAFETY: both ranges are on this rank's device and hold what the op reads and writes (checked above).
            unsafe { (self.fns.reduce_scatter)(from as *const c_void, to as *mut c_void, count as usize, nccl_type(dtype), SUM, comm, stream) }
        })
    }

    fn broadcast(&self, buf: &DeviceBuffer, count: u64, dtype: CollectiveDtype, root: usize) -> Result<(), String> {
        if root >= self.world {
            return Err(format!("broadcast: root {root} is outside a communicator of {} ranks", self.world));
        }
        let at = bounded_elems(buf, count, dtype, "broadcast")?;
        self.enqueue(&[buf], "ncclBroadcast", |comm, stream| {
            // SAFETY: the range is on this rank's device and holds `count` elements (checked above); in place.
            unsafe { (self.fns.broadcast)(at as *const c_void, at as *mut c_void, count as usize, nccl_type(dtype), root as c_int, comm, stream) }
        })
    }

    fn split(&self, color: Option<u32>, key: u32, timeout: Duration) -> Result<Option<Box<dyn RankCollective>>, String> {
        let color = match color {
            Some(c) if c > c_int::MAX as u32 => return Err(format!("split: color {c} does not fit NCCL's int")),
            Some(c) => c as c_int,
            None => NO_COLOR,
        };
        if key > c_int::MAX as u32 {
            return Err(format!("split: key {key} does not fit NCCL's int"));
        }
        let handle = CudaBackend::try_new_on_ordinal(&[], self.handle.ordinal())?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let parent = state.live()?;
        self.handle.bind_context()?;
        let deadline = Instant::now() + timeout;
        let mut child: Comm = std::ptr::null_mut();
        // SAFETY: `parent` is live; `child` receives the new communicator (NULL for no color); a NULL config
        // inherits the parent's, nonblocking included.
        let rc = unsafe { (self.fns.comm_split)(parent, color, key as c_int, &mut child, std::ptr::null_mut()) };
        let settled = match rc {
            SUCCESS | IN_PROGRESS => settle(self.lib, self.fns, if child.is_null() { parent } else { child }, deadline, "ncclCommSplit"),
            failed => self.lib.check(failed, "ncclCommSplit"),
        };
        if let Err(e) = settled {
            if !child.is_null() {
                // SAFETY: `child` was created by the split above and is not used again.
                unsafe { (self.fns.comm_abort)(child) };
            }
            // The peers are inside the split with this rank's parent; it cannot be used consistently any more.
            return Err(self.fail(&mut state, e));
        }
        if child.is_null() {
            return Ok(None);
        }
        let (mut world, mut rank): (c_int, c_int) = (0, 0);
        // SAFETY: `child` is a live communicator; both outputs are writable.
        let described = unsafe { self.lib.check((self.fns.comm_count)(child, &mut world), "ncclCommCount").and_then(|()| self.lib.check((self.fns.comm_user_rank)(child, &mut rank), "ncclCommUserRank")) };
        if let Err(e) = described {
            // SAFETY: as above; not used again.
            unsafe { (self.fns.comm_abort)(child) };
            return Err(e);
        }
        let state = RankState { comm: child as usize, failed: None };
        Ok(Some(Box::new(NcclRank { lib: self.lib, fns: self.fns, state: Mutex::new(state), handle, world: world as usize, rank: rank as usize, settle: self.settle })))
    }

    fn wait(&self, timeout: Duration) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let comm = state.live()?;
        let deadline = Instant::now() + timeout;
        loop {
            let mut async_state = SUCCESS;
            // SAFETY: `comm` is live and `async_state` writable.
            let polled = self.lib.check(unsafe { (self.fns.comm_get_async_error)(comm, &mut async_state) }, "ncclCommGetAsyncError");
            let peer = polled.and_then(|()| if async_state == IN_PROGRESS { Ok(()) } else { self.lib.check(async_state, "a collective") });
            if let Err(e) = peer {
                return Err(self.fail(&mut state, e));
            }
            match self.handle.stream_idle() {
                Ok(true) => return Ok(()),
                Ok(false) if Instant::now() >= deadline => {
                    return Err(self.fail(&mut state, format!("the collectives did not complete within {timeout:?}: a peer stopped taking part")));
                }
                Ok(false) => std::thread::sleep(POLL),
                Err(e) => return Err(self.fail(&mut state, e)),
            }
        }
    }
}

impl Drop for NcclRank {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if state.comm == 0 {
            return;
        }
        let comm = state.comm as Comm;
        state.comm = 0;
        let deadline = Instant::now() + self.settle;
        // In-flight collectives read and write the caller's buffers: let them finish first, then flush and free the
        // communicator. A peer that never completes them is not waited for past the deadline; the communicator is
        // aborted instead, which stops its kernels.
        let quiet = self.handle.bind_context().and_then(|()| loop {
            match self.handle.stream_idle() {
                Ok(true) => break Ok(()),
                Ok(false) if Instant::now() >= deadline => break Err("in-flight collectives did not finish".to_string()),
                Ok(false) => std::thread::sleep(POLL),
                Err(e) => break Err(e),
            }
        });
        // SAFETY: `comm` is live and used by nobody else; it is finalized and destroyed, or aborted, exactly once.
        let closed = quiet.and_then(|()| unsafe {
            match (self.fns.comm_finalize)(comm) {
                SUCCESS | IN_PROGRESS => settle(self.lib, self.fns, comm, deadline, "ncclCommFinalize"),
                failed => self.lib.check(failed, "ncclCommFinalize"),
            }
        });
        match closed {
            // SAFETY: as above.
            Ok(()) => {
                let rc = unsafe { (self.lib.comm_destroy)(comm) };
                if rc != SUCCESS {
                    tracing::warn!(rc, "ncclCommDestroy failed");
                }
            }
            Err(e) => {
                tracing::warn!(reason = %e, "aborting the NCCL communicator instead of closing it");
                // SAFETY: as above.
                unsafe { (self.fns.comm_abort)(comm) };
            }
        }
    }
}

/// The device address of `buf` after checking it holds `words` f32 words from `offset_words`.
fn bounded(buf: &DeviceBuffer, offset_words: u64, words: u64, what: &str) -> Result<u64, String> {
    let (ptr, len) = CudaBackend::address_of(buf)?;
    let end = (offset_words + words).checked_mul(4).ok_or_else(|| format!("{what}: range overflows"))?;
    if end > len {
        return Err(format!("{what}: {words} words from word {offset_words} do not fit a {len}-byte buffer"));
    }
    Ok(ptr + offset_words * 4)
}

impl Drop for NcclGroup {
    fn drop(&mut self) {
        // The collectives still in flight read the buffers and use the communicators: let every rank's stream
        // finish before the communicators go.
        for r in &self.ranks {
            backend_api::Backend::poll_wait(r);
        }
        for &c in &self.comms {
            // SAFETY: each communicator was created by `ncclCommInitAll` and is destroyed once.
            let rc = unsafe { (self.lib.comm_destroy)(c as Comm) };
            if rc != SUCCESS {
                tracing::warn!(rc, "ncclCommDestroy failed");
            }
        }
    }
}
