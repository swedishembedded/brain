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
//!   joins with `ncclCommInitRankConfig`. Joining is nonblocking, so a peer
//!   that never arrives cannot hang a rank: the join is bounded by a deadline
//!   after which the half-made communicator is aborted. The collectives then run
//!   on a blocking communicator split from it: a nonblocking one launches every
//!   collective from a thread of its own and reports it through polling, which
//!   costs each small operation tens of microseconds more than the operation
//!   itself. A blocking call that does not return by its deadline (a peer lost
//!   while a connection is being made) is ended by the rank's watchdog thread,
//!   which aborts the communicator (`ncclCommAbort`; NCCL's guide to error
//!   handling names a second thread's abort as the way to end a hung call of a
//!   blocking communicator); the call then fails like any other. Once bounded
//!   ([`RankCollective::bound_in_flight`]), the same thread also ends an
//!   operation the stream has waited on past the bound (enqueued, every earlier
//!   one done, itself not done), waited for or not: the stream marks each
//!   operation's completion in a page-locked word the watchdog reads without
//!   the driver, so it still sees them while another
//!   thread is blocked inside the driver behind the stuck operation (a read of
//!   its result, an enqueue into a full stream), which nothing else would end.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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
    /// `ncclCommGetAsyncError` (`blocking = 0`): how the join is bounded by its deadline.
    fn nonblocking() -> Config {
        Config { blocking: 0, ..Config::blocking() }
    }

    /// The library's defaults: calls return when their work is enqueued, which a [`Watchdog`] bounds.
    fn blocking() -> Config {
        Config {
            size: std::mem::size_of::<Config>(),
            magic: API_MAGIC,
            version: CONFIG_VERSION,
            blocking: 1,
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
    /// `ncclGetVersion`'s code (major * 10000 + minor * 100 + patch since 2.9).
    version: c_int,
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
        version,
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

/// How long a wait on the library or the stream spins before it sleeps between polls: a collective on one node
/// completes in tens of microseconds, and a sleep of `POLL` lasts about twice that, which used to set the floor of
/// every small operation.
const SPIN: Duration = Duration::from_micros(500);

/// One pause of a polling loop that began at `started`: a yield while the wait is young, a sleep once it has gone on
/// long enough that it is not a small operation finishing.
fn pause(started: Instant) {
    if started.elapsed() < SPIN {
        std::thread::yield_now();
    } else {
        std::thread::sleep(POLL);
    }
}

/// Poll a nonblocking communicator until its pending call (`what`) has finished: `Ok` once the library reports
/// success, the library's error when it fails, and an error naming the deadline when `deadline` passes first.
fn settle(lib: &Lib, fns: &RankFns, comm: Comm, deadline: Instant, what: &str) -> Result<(), String> {
    let started = Instant::now();
    loop {
        let mut state = SUCCESS;
        // SAFETY: `comm` is a live communicator of this library and `state` a writable result.
        lib.check(unsafe { (fns.comm_get_async_error)(comm, &mut state) }, "ncclCommGetAsyncError")?;
        match state {
            SUCCESS => return Ok(()),
            IN_PROGRESS if Instant::now() >= deadline => return Err(format!("{what} did not complete before its deadline: a peer did not take part")),
            IN_PROGRESS => pause(started),
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

/// How many of a rank's collectives have completed, as one word a host thread reads without the driver: the
/// collectives' stream writes `k + 1` into it once its operation `k` (counted from 0) has run. One mark per operation:
/// a mark costs the stream about two microseconds, and a start mark as well doubled that on every small operation.
trait Progress: Send + Sync {
    fn read(&self) -> u32;
}

impl Progress for crate::exec::PinnedMem {
    fn read(&self) -> u32 {
        self.read_word(0)
    }
}

/// The operations of one communicator bounded on the device ([`RankCollective::bound_in_flight`]).
struct Bound {
    comm: usize,
    limit: Duration,
    progress: Arc<dyn Progress>,
    /// Operations enqueued and not yet seen complete, oldest first, by when each was enqueued.
    outstanding: std::collections::VecDeque<Instant>,
    /// How many operations were seen complete (wrapping, as the progress word counts).
    completed: u32,
    /// Since when the oldest outstanding operation is the one the stream waits on: enqueued, and every operation
    /// before it seen complete.
    front_since: Option<Instant>,
}

impl Bound {
    fn new(comm: Comm, limit: Duration, progress: Arc<dyn Progress>) -> Bound {
        Bound { comm: comm as usize, limit, progress, outstanding: std::collections::VecDeque::new(), completed: 0, front_since: None }
    }

    fn enqueued(&mut self, at: Instant) {
        if self.outstanding.is_empty() {
            self.front_since = Some(at);
        }
        self.outstanding.push_back(at);
    }

    /// The operation past its bound at `now`, if one is: its index and how long it has been the oldest outstanding.
    fn overdue(&mut self, now: Instant) -> Option<(u32, Duration)> {
        let done = self.progress.read();
        while self.completed != done && !self.outstanding.is_empty() {
            self.outstanding.pop_front();
            self.completed = self.completed.wrapping_add(1);
            // The next one is the stream's now; at the earliest from when this tick saw its predecessor complete.
            self.front_since = self.outstanding.front().map(|&at| at.max(now));
        }
        let since = self.front_since.filter(|_| !self.outstanding.is_empty())?;
        let waited = now.saturating_duration_since(since);
        (waited >= self.limit).then_some((self.completed, waited))
    }
}

/// What the watchdog thread and the rank's calls share, under one lock: the watchdog aborts a communicator only
/// while holding it, and a call checks under it that its communicator was not aborted before using it.
#[derive(Default)]
struct WatchState {
    /// The blocking call under watch: its deadline and the communicator to abort, as an address.
    armed: Option<(Instant, usize)>,
    /// The communicator whose enqueued operations are bounded on the device.
    bound: Option<Bound>,
    /// A communicator the watchdog aborted, and why.
    aborted: Option<(usize, String)>,
}

impl WatchState {
    fn abort(&mut self, comm: usize, why: String, abort: unsafe extern "C" fn(Comm) -> NcclResult) {
        if self.armed.is_some_and(|(_, c)| c == comm) {
            self.armed = None;
        }
        if self.bound.as_ref().is_some_and(|b| b.comm == comm) {
            self.bound = None;
        }
        // SAFETY: `comm` is live: it is aborted only here, under the lock every user of it checks first, and the
        // owner retires it (`Watchdog::retire`) under the same lock before closing it.
        unsafe { abort(comm as Comm) };
        self.aborted = Some((comm, why));
    }

    /// One look at the watched call and the bounded operations at `now`.
    fn tick(&mut self, now: Instant, abort: unsafe extern "C" fn(Comm) -> NcclResult) {
        if let Some((deadline, comm)) = self.armed {
            if now >= deadline {
                self.armed = None;
                self.abort(comm, "a call did not return before its deadline".into(), abort);
            }
        }
        let Some(bound) = self.bound.as_mut() else { return };
        if let Some((op, waited)) = bound.overdue(now) {
            let (comm, limit) = (bound.comm, bound.limit);
            self.abort(comm, format!("collective {op} was not done {waited:.1?} after the ones before it, past the {limit:?} bound: a peer stopped taking part"), abort);
        }
    }
}

/// A thread that aborts a communicator whose blocking call has not returned by its deadline, or whose enqueued
/// operation has run on the device past its bound ([`Watchdog::bound`]).
struct Watchdog {
    shared: Arc<Watch>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct Watch {
    state: Mutex<WatchState>,
    stop: AtomicBool,
}

/// How often the watchdog looks at the call and the operations it watches.
const WATCH_TICK: Duration = Duration::from_millis(10);

impl Watchdog {
    fn start(abort: unsafe extern "C" fn(Comm) -> NcclResult) -> Watchdog {
        let shared = Arc::new(Watch { state: Mutex::new(WatchState::default()), stop: AtomicBool::new(false) });
        let watch = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("nccl-watchdog".into())
            .spawn(move || {
                while !watch.stop.load(Ordering::Acquire) {
                    watch.state.lock().unwrap_or_else(|e| e.into_inner()).tick(Instant::now(), abort);
                    std::thread::sleep(WATCH_TICK);
                }
            })
            .ok();
        Watchdog { shared, thread }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, WatchState> {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `call` on `comm`, aborting `comm` if it has not returned by `deadline`. Returns what it returned and
    /// whether the watchdog aborted the communicator (it is then gone: the caller neither uses nor aborts it again).
    /// A communicator the watchdog already aborted is not called at all (`R::default()`, aborted).
    fn guard<R: Default>(&self, comm: Comm, deadline: Instant, call: impl FnOnce() -> R) -> (R, bool) {
        if self.thread.is_none() {
            // No watchdog thread could be started: the call runs unwatched rather than not at all.
            return (call(), false);
        }
        {
            let mut state = self.state();
            if state.aborted.as_ref().is_some_and(|(c, _)| *c == comm as usize) {
                return (R::default(), true);
            }
            state.armed = Some((deadline, comm as usize));
        }
        let out = call();
        let aborted = self.state().armed.take().is_none();
        (out, aborted)
    }

    /// Bound the operations enqueued on `comm` by `limit` on the device, as `progress` reports their completion.
    /// `false` when no watchdog thread runs to do it.
    fn bound(&self, comm: Comm, limit: Duration, progress: Arc<dyn Progress>) -> bool {
        if self.thread.is_none() {
            return false;
        }
        self.state().bound = Some(Bound::new(comm, limit, progress));
        true
    }

    /// Record that an operation was enqueued on the bounded communicator `comm` at `at`.
    fn enqueued(&self, comm: Comm, at: Instant) {
        if let Some(bound) = self.state().bound.as_mut().filter(|b| b.comm == comm as usize) {
            bound.enqueued(at);
        }
    }

    /// `f()` under the lock the watchdog aborts under, unless it already aborted `comm` (`None`): how a call that
    /// does not block (a poll) uses the communicator without racing the abort.
    fn while_live<R>(&self, comm: Comm, f: impl FnOnce() -> R) -> Option<R> {
        let state = self.state();
        if state.aborted.as_ref().is_some_and(|(c, _)| *c == comm as usize) {
            return None;
        }
        Some(f())
    }

    /// Why the watchdog aborted `comm`, if it did.
    fn aborted(&self, comm: Comm) -> Option<String> {
        self.state().aborted.as_ref().filter(|(c, _)| *c == comm as usize).map(|(_, why)| why.clone())
    }

    /// Take `comm` out of the watchdog's reach before its owner aborts or closes it; `false` when the watchdog
    /// already aborted it (the owner must not touch it again). With `abort`, the owner's abort runs under the same
    /// lock, so the two can never both abort it.
    fn retire(&self, comm: Comm, abort: Option<unsafe extern "C" fn(Comm) -> NcclResult>) -> bool {
        let mut state = self.state();
        if state.aborted.as_ref().is_some_and(|(c, _)| *c == comm as usize) {
            return false;
        }
        if state.bound.as_ref().is_some_and(|b| b.comm == comm as usize) {
            state.bound = None;
        }
        if let Some(abort) = abort {
            // SAFETY: `comm` is live (the watchdog has not aborted it, and cannot now: the lock is held).
            unsafe { abort(comm) };
        }
        true
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct RankState {
    /// The live communicator as an address (so the rank is `Send`); `0` once aborted or destroyed.
    comm: usize,
    /// Why the communicator was aborted, repeated to every later call.
    failed: Option<String>,
    /// With the operations bounded on the device: the word the stream marks their progress in, and how many were
    /// enqueued.
    progress: Option<(Arc<crate::exec::PinnedMem>, u32)>,
    /// The bound, which a split inherits.
    bound: Option<Duration>,
}

impl RankState {
    fn new(comm: Comm) -> RankState {
        RankState { comm: comm as usize, failed: None, progress: None, bound: None }
    }
}

/// One rank of a communicator whose ranks are separate processes, each on one card (joined nonblocking with
/// `ncclCommInitRankConfig`, its collectives on a blocking split of it). See the module documentation and
/// [`RankCollective`] for the contract.
pub struct NcclRank {
    lib: &'static Lib,
    fns: &'static RankFns,
    state: Mutex<RankState>,
    /// Ends a blocking call of the communicator that outlives its deadline, and an enqueued operation that runs on
    /// the device past its bound.
    watchdog: Watchdog,
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
        // The collectives' communicator: the same ranks, blocking (see the module documentation), split while the
        // watchdog bounds the split by what is left of the deadline.
        let watchdog = Watchdog::start(fns.comm_abort);
        let mut ops: Comm = std::ptr::null_mut();
        let mut blocking = Config::blocking();
        // SAFETY: `comm` is live; `ops` receives the new communicator; `blocking` is valid for the call.
        let (rc, aborted) = watchdog.guard(comm, deadline, || unsafe { (fns.comm_split)(comm, 0, rank as c_int, &mut ops, &mut blocking) });
        let split = match (aborted, rc) {
            (true, _) => Err(format!("splitting the collectives' communicator (rank {rank} of {world}) did not finish before its deadline")),
            // The parent is nonblocking, so the split may finish (and fill in `ops`) after the call returns: the
            // parent reports when it has.
            (false, SUCCESS | IN_PROGRESS) => settle(lib, fns, if ops.is_null() { comm } else { ops }, deadline, "ncclCommSplit").and_then(|()| {
                if ops.is_null() {
                    Err("ncclCommSplit returned no communicator".to_string())
                } else {
                    Ok(())
                }
            }),
            (false, failed) => lib.check(failed, "ncclCommSplit"),
        };
        // The joining communicator has done its part: free it (nonblocking, so bounded), or abort it.
        if !aborted {
            // SAFETY: `comm` is live and not used after this.
            let closed = unsafe {
                match (fns.comm_finalize)(comm) {
                    SUCCESS | IN_PROGRESS => settle(lib, fns, comm, deadline, "ncclCommFinalize"),
                    failed => lib.check(failed, "ncclCommFinalize"),
                }
            };
            // SAFETY: as above, exactly once.
            unsafe {
                if closed.is_ok() {
                    (lib.comm_destroy)(comm);
                } else {
                    (fns.comm_abort)(comm);
                }
            }
        }
        if let Err(e) = split {
            if !ops.is_null() {
                // SAFETY: `ops` was created by the split above and is not used again.
                unsafe { (fns.comm_abort)(ops) };
            }
            return Err(e);
        }
        Ok(NcclRank { lib, fns, state: Mutex::new(RankState::new(ops)), watchdog, handle, world, rank, settle: timeout })
    }

    /// The live communicator, unless it was closed or aborted (by a failed call, or by the watchdog in the
    /// background: an operation past its bound).
    fn live(&self, state: &mut RankState) -> Result<Comm, String> {
        if state.comm != 0 {
            if let Some(why) = self.watchdog.aborted(state.comm as Comm) {
                self.aborted(state, why);
            }
        }
        match (&state.failed, state.comm) {
            (Some(why), _) => Err(format!("the communicator was aborted: {why}")),
            (None, 0) => Err("the communicator is closed".into()),
            (None, comm) => Ok(comm as Comm),
        }
    }

    /// The communicator the watchdog already aborted: forget it (it must not be aborted again) and remember why.
    fn aborted(&self, state: &mut RankState, why: String) -> String {
        state.comm = 0;
        state.failed.get_or_insert(why).clone()
    }

    /// Abort the communicator (in-flight kernels stop, peers see the failure) and remember why. One the watchdog
    /// aborted first keeps the watchdog's reason.
    fn fail(&self, state: &mut RankState, why: String) -> String {
        if state.comm != 0 {
            let comm = state.comm as Comm;
            if !self.watchdog.retire(comm, Some(self.fns.comm_abort)) {
                let first = self.watchdog.aborted(comm).unwrap_or_default();
                return self.aborted(state, first);
            }
            state.comm = 0;
        }
        state.failed.get_or_insert(why).clone()
    }

    /// Mark the operation just enqueued done on the collectives' stream, when the operations are bounded.
    fn mark_done(&self, state: &mut RankState, comm: Comm) -> Result<(), String> {
        let Some((word, enqueued)) = state.progress.as_mut() else { return Ok(()) };
        *enqueued = enqueued.wrapping_add(1);
        self.handle.mark_progress(word, *enqueued)?;
        self.watchdog.enqueued(comm, Instant::now());
        Ok(())
    }

    /// Enqueue `op` on this rank's stream, between the fences that order it against the other handles' work on
    /// `touched`, and wait (bounded) for the library to accept it. A failure aborts the communicator.
    fn enqueue(&self, touched: &[&DeviceBuffer], what: &str, op: impl FnOnce(Comm, CuStream) -> NcclResult) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let comm = self.live(&mut state)?;
        self.handle.bind_context()?;
        self.handle.fence_collective_in(touched)?;
        let deadline = Instant::now() + self.settle;
        let stream = self.handle.stream();
        let (rc, aborted) = self.watchdog.guard(comm, deadline, || op(comm, stream));
        if aborted {
            self.handle.fence_collective_out(touched);
            let why = self.watchdog.aborted(comm).unwrap_or_else(|| format!("{what} did not return within {:?}: a peer did not take part", self.settle));
            return Err(self.aborted(&mut state, why));
        }
        let accepted = match rc {
            SUCCESS => Ok(()),
            IN_PROGRESS => settle(self.lib, self.fns, comm, deadline, what),
            failed => self.lib.check(failed, what),
        }
        // Operation k (counted from 0) marks k + 1 once it has run.
        .and_then(|()| self.mark_done(&mut state, comm));
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

    fn timed(&self, enqueue: &mut dyn FnMut() -> Result<(), String>, timeout: Duration) -> Result<Option<Duration>, String> {
        self.handle.bind_context()?;
        self.handle.device_time(enqueue, &mut || self.wait(timeout)).map(Some)
    }

    fn library_version(&self) -> Option<String> {
        let v = self.fns.version;
        Some(format!("{}.{}.{}", v / 10000, (v / 100) % 100, v % 100))
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

    fn reduce_scatter(&self, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype, scale: f32) -> Result<(), String> {
        let from = bounded_elems(send, count * self.world as u64, dtype, "reduce_scatter send")?;
        let to = bounded_elems(recv, count, dtype, "reduce_scatter recv")?;
        if !scale.is_finite() {
            return Err(format!("reduce_scatter: a scale of {scale} is not a number to multiply by"));
        }
        self.enqueue(&[send, recv], "ncclReduceScatter", |comm, stream| {
            let op = if scale == 1.0 {
                SUM
            } else {
                match self.premul_sum(comm, dtype, scale) {
                    Ok(op) => op,
                    Err(rc) => return rc,
                }
            };
            // SAFETY: both ranges are on this rank's device and hold what the op reads and writes (checked above);
            // the custom op is destroyed only after the operation that uses it has been enqueued.
            unsafe {
                let rc = (self.fns.reduce_scatter)(from as *const c_void, to as *mut c_void, count as usize, nccl_type(dtype), op, comm, stream);
                if op != SUM {
                    (self.fns.red_op_destroy)(op, comm);
                }
                rc
            }
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
        let parent = self.live(&mut state)?;
        self.handle.bind_context()?;
        let deadline = Instant::now() + timeout;
        let mut child: Comm = std::ptr::null_mut();
        // SAFETY: `parent` is live; `child` receives the new communicator (NULL for no color); a NULL config
        // inherits the parent's (blocking: the watchdog bounds the call).
        let (rc, aborted) = self.watchdog.guard(parent, deadline, || unsafe { (self.fns.comm_split)(parent, color, key as c_int, &mut child, std::ptr::null_mut()) });
        if aborted {
            return Err(self.aborted(&mut state, "ncclCommSplit did not return before its deadline".into()));
        }
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
        let child = NcclRank { lib: self.lib, fns: self.fns, state: Mutex::new(RankState::new(child)), watchdog: Watchdog::start(self.fns.comm_abort), handle, world: world as usize, rank: rank as usize, settle: self.settle };
        // A sub-group's operations are bounded as its parent's are.
        if let Some(limit) = state.bound {
            child.bound_in_flight(limit)?;
        }
        Ok(Some(Box::new(child)))
    }

    fn bound_in_flight(&self, limit: Duration) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let comm = self.live(&mut state)?;
        if state.progress.is_some() {
            return Err("the operations of this communicator are already bounded".into());
        }
        let word = Arc::new(self.handle.progress_word()?);
        // Nothing enqueued yet is the even value 0; the first mark also proves the driver can make them.
        if !self.handle.mark_progress(&word, 0)? {
            return Err("this driver has no stream memory operations (cuStreamWriteValue32) to mark the operations' progress with".into());
        }
        if !self.watchdog.bound(comm, limit, Arc::clone(&word) as Arc<dyn Progress>) {
            return Err("no watchdog thread could be started for this communicator".into());
        }
        state.progress = Some((word, 0));
        state.bound = Some(limit);
        Ok(())
    }

    fn wait(&self, timeout: Duration) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let comm = self.live(&mut state)?;
        let started = Instant::now();
        let deadline = started + timeout;
        loop {
            let mut async_state = SUCCESS;
            // Under the watchdog's lock, so it cannot abort (and free) the communicator during the poll.
            // SAFETY: `comm` is live (checked under the same lock) and `async_state` writable.
            let polled = self.watchdog.while_live(comm, || self.lib.check(unsafe { (self.fns.comm_get_async_error)(comm, &mut async_state) }, "ncclCommGetAsyncError"));
            let Some(polled) = polled else {
                // An operation ran past its bound: the watchdog aborted the communicator, which also ended the
                // operation the stream was waiting on.
                let why = self.watchdog.aborted(comm).unwrap_or_default();
                return Err(format!("the communicator was aborted: {}", self.aborted(&mut state, why)));
            };
            let peer = polled.and_then(|()| if async_state == IN_PROGRESS { Ok(()) } else { self.lib.check(async_state, "a collective") });
            if let Err(e) = peer {
                return Err(self.fail(&mut state, e));
            }
            match self.handle.stream_idle() {
                // An idle stream after the watchdog's abort is the aborted operation's end, not its completion.
                Ok(true) => return self.live(&mut state).map(|_| ()),
                Ok(false) if Instant::now() >= deadline => {
                    return Err(self.fail(&mut state, format!("the collectives did not complete within {timeout:?}: a peer stopped taking part")));
                }
                Ok(false) => pause(started),
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
        // Out of the watchdog's reach before it is closed; one the watchdog already aborted is gone.
        if !self.watchdog.retire(comm, None) {
            return;
        }
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
        let mut aborted = false;
        let closed = quiet.and_then(|()| {
            let (rc, gone) = self.watchdog.guard(comm, deadline, || unsafe { (self.fns.comm_finalize)(comm) });
            aborted = gone;
            match (gone, rc) {
                (true, _) => Err("ncclCommFinalize did not return before its deadline".to_string()),
                (false, SUCCESS | IN_PROGRESS) => settle(self.lib, self.fns, comm, deadline, "ncclCommFinalize"),
                (false, failed) => self.lib.check(failed, "ncclCommFinalize"),
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
                if !aborted {
                    // SAFETY: as above.
                    unsafe { (self.fns.comm_abort)(comm) };
                }
            }
        }
        // The stream may still hold marks for the progress word: it is never freed under them.
        if !matches!(self.handle.stream_idle(), Ok(true)) {
            if let Some((word, _)) = state.progress.take() {
                std::mem::forget(word);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, AtomicUsize};

    static ABORTED: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn record_abort(comm: Comm) -> NcclResult {
        ABORTED.store(comm as usize, Ordering::SeqCst);
        SUCCESS
    }

    /// A call that returns in time is left alone; one that outlives its deadline has its communicator aborted, once,
    /// and the guard reports it so the caller does not abort it again.
    #[test]
    fn watchdog_aborts_only_calls_past_their_deadline() {
        let watchdog = Watchdog::start(record_abort);
        let (out, aborted) = watchdog.guard(0x10 as Comm, Instant::now() + Duration::from_secs(5), || 7);
        assert_eq!((out, aborted), (7, false));
        assert_eq!(ABORTED.load(Ordering::SeqCst), 0);

        let (_, aborted) = watchdog.guard(0x20 as Comm, Instant::now() + WATCH_TICK, || {
            let until = Instant::now() + Duration::from_secs(5);
            while ABORTED.load(Ordering::SeqCst) == 0 && Instant::now() < until {
                std::thread::sleep(WATCH_TICK);
            }
        });
        assert!(aborted);
        assert_eq!(ABORTED.load(Ordering::SeqCst), 0x20);
    }

    struct Word(AtomicU32);

    impl Progress for Word {
        fn read(&self) -> u32 {
            self.0.load(Ordering::SeqCst)
        }
    }

    static BOUND_ABORTED: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn record_bound_abort(comm: Comm) -> NcclResult {
        BOUND_ABORTED.store(comm as usize, Ordering::SeqCst);
        SUCCESS
    }

    /// An enqueued operation is bounded by how long the stream has been waiting on it, whoever waits for it: one
    /// that completes within the bound is left alone, and the clock of the next starts when its predecessor was seen
    /// complete; one still outstanding past the bound has its communicator aborted, with the reason every later use
    /// of it reports, and is never watched again.
    #[test]
    fn an_operation_outstanding_past_its_bound_aborts_its_communicator() {
        let word = Arc::new(Word(AtomicU32::new(0)));
        let limit = Duration::from_secs(5);
        let mut state = WatchState { bound: Some(Bound::new(0x30 as Comm, limit, Arc::clone(&word) as Arc<dyn Progress>)), ..WatchState::default() };
        let t0 = Instant::now();
        // Nothing enqueued: nothing is overdue, however long.
        state.tick(t0 + 2 * limit, record_bound_abort);
        assert!(state.aborted.is_none());
        // Operations 0 and 1 enqueued at t0; 0 completes just inside the bound.
        state.bound.as_mut().unwrap().enqueued(t0);
        state.bound.as_mut().unwrap().enqueued(t0);
        word.0.store(1, Ordering::SeqCst);
        state.tick(t0 + limit - Duration::from_millis(1), record_bound_abort);
        state.tick(t0 + limit + limit / 2, record_bound_abort);
        assert!(state.aborted.is_none(), "operation 1's clock starts when operation 0 was seen complete");
        assert_eq!(BOUND_ABORTED.load(Ordering::SeqCst), 0);
        // Operation 1 is still outstanding a full bound after that.
        state.tick(t0 + 2 * limit, record_bound_abort);
        assert_eq!(BOUND_ABORTED.load(Ordering::SeqCst), 0x30);
        let (comm, why) = state.aborted.clone().expect("aborted");
        assert_eq!(comm, 0x30);
        assert!(why.contains("collective 1 was not done"), "{why}");
        assert!(state.bound.is_none(), "an aborted communicator is no longer watched");
    }

    /// A blocking call on a communicator the watchdog already aborted is not made at all.
    #[test]
    fn a_call_on_an_aborted_communicator_is_not_made() {
        let watchdog = Watchdog::start(record_abort);
        watchdog.state().aborted = Some((0x40, "gone".into()));
        let (out, aborted) = watchdog.guard(0x40 as Comm, Instant::now() + Duration::from_secs(5), || 7);
        assert_eq!((out, aborted), (0, true));
        assert!(!watchdog.retire(0x40 as Comm, None), "its owner must not close it again");
        assert_eq!(watchdog.while_live(0x40 as Comm, || 1), None);
        assert_eq!(watchdog.while_live(0x41 as Comm, || 1), Some(1));
    }
}
