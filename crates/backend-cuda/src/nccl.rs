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
//! One process drives every rank (`ncclCommInitAll`). Each rank owns a
//! [`CudaBackend`] handle of its device with no kernels, so the collective runs
//! on that handle's own stream and is ordered against the kernels that produced
//! and will consume its buffers by the same event fences every other cross-handle
//! use goes through: nothing waits for the whole device.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Mutex;

use backend_api::{DeviceBuffer, DeviceCollective};

use crate::backend::CudaBackend;
use crate::driver::CuStream;

type NcclResult = c_int;
type Comm = *mut c_void;

const SUCCESS: NcclResult = 0;
/// `ncclFloat32`.
const FLOAT32: c_int = 7;
/// `ncclSum`.
const SUM: c_int = 0;

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
