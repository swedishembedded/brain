// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The** CPU-parallel primitives - the on-CPU scheduler's public face for
//! host-side data-parallel work.
//!
//! rayon lives in exactly one crate: this one. `backend-cpu` *is* brain's CPU
//! scheduler - it already fans every JIT-compiled kernel out across the cores -
//! so host-side parallel loops (an `m=1` decode matvec, a grad-norm reduction,
//! image-row post-processing) belong to the same pool under the same policy.
//! Before this module, six crates depended on rayon directly and each ad-hoc
//! `par_iter` ran on a pool nobody owned: thread-count policy (`--device cpu0-7`
//! pins affinity and sizes the pool, see [`set_global_threads`]) only holds if
//! every parallel loop goes through the scheduler that honours it.
//!
//! The API is deliberately domain-shaped (rows of `f32`, tensor lists) rather
//! than a re-export of `ParallelIterator`: a caller states *what* is parallel,
//! and the scheduler owns *how*.

use rayon::prelude::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

thread_local! {
    /// The pool the calling thread's parallel loops are confined to by [`with_cores`], if any.
    static NEAR: RefCell<Option<Arc<rayon::ThreadPool>>> = const { RefCell::new(None) };
}

/// Run `f` on the pool [`with_cores`] confined this thread to, or on the process pool.
fn routed<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match NEAR.with(|n| n.borrow().clone()) {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// The cores this process may run on.
pub fn allowed_cores() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: a zeroed cpu_set_t is the empty set; sched_getaffinity fills it for the calling thread.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } == 0 {
            return (0..libc::CPU_SETSIZE as usize).filter(|&c| unsafe { libc::CPU_ISSET(c, &set) }).collect();
        }
    }
    (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect()
}

/// The affinity of the calling thread, or `None` where it cannot be read.
#[cfg(target_os = "linux")]
fn thread_affinity() -> Option<libc::cpu_set_t> {
    // SAFETY: see `allowed_cores`.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    (unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } == 0).then_some(set)
}

/// Pin the calling thread to `cores`; best effort (a refused mask leaves the thread where it was).
#[cfg(target_os = "linux")]
fn pin_thread(cores: &[usize]) {
    // SAFETY: CPU_SET writes only inside the set for ids below CPU_SETSIZE, which the filter guarantees.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cores.iter().filter(|&&c| c < libc::CPU_SETSIZE as usize) {
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
}

/// Run `f` with this thread and the parallel loops it starts confined to `cores`, in a pool of one thread per core that
/// is kept for the next call with the same cores. For host work that moves data to and from an accelerator: on a
/// machine of several CPU/GPU modules the cores next to the card reach the card's memory path and their own memory
/// without crossing to another module, which is several times the bandwidth of the same work spread over every core.
///
/// Memory first touched inside `f` (by this thread or the pool) is placed on the cores' own node by the kernel's local
/// policy. Only the calling thread's loops are routed; other threads keep the process pool. An empty `cores` runs `f`
/// unchanged.
pub fn with_cores<R>(cores: &[usize], f: impl FnOnce() -> R) -> R {
    static POOLS: Mutex<Option<HashMap<Vec<usize>, Arc<rayon::ThreadPool>>>> = Mutex::new(None);
    if cores.is_empty() {
        return f();
    }
    let pool = {
        let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
        pools
            .get_or_insert_with(HashMap::new)
            .entry(cores.to_vec())
            .or_insert_with(|| {
                let pinned = cores.to_vec();
                let builder = rayon::ThreadPoolBuilder::new().num_threads(cores.len()).thread_name(|i| format!("brain-near-{i}"));
                #[cfg(target_os = "linux")]
                let builder = builder.start_handler(move |_| pin_thread(&pinned));
                #[cfg(not(target_os = "linux"))]
                let _ = pinned;
                Arc::new(builder.build().expect("a rayon pool"))
            })
            .clone()
    };
    /// Restores the thread's pool and affinity, also when `f` unwinds.
    struct Restore {
        pool: Option<Arc<rayon::ThreadPool>>,
        #[cfg(target_os = "linux")]
        affinity: Option<libc::cpu_set_t>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            NEAR.with(|n| *n.borrow_mut() = self.pool.take());
            #[cfg(target_os = "linux")]
            if let Some(set) = &self.affinity {
                // SAFETY: `set` came from sched_getaffinity.
                unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set) };
            }
        }
    }
    let _restore = Restore {
        pool: NEAR.with(|n| n.borrow_mut().replace(pool)),
        #[cfg(target_os = "linux")]
        affinity: thread_affinity(),
    };
    #[cfg(target_os = "linux")]
    pin_thread(cores);
    f()
}


/// Apply `f(row_index, row)` to each `row_len`-sized row of `buf`, in parallel.
///
/// The row split is exact: `buf.len()` must be a multiple of `row_len` for the
/// last row to be full; a short tail row is passed as-is (matching
/// `chunks_mut`), never dropped.
pub fn rows_mut(buf: &mut [f32], row_len: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    chunks_mut(buf, row_len, f)
}

/// Threads the parallel loops fan out over right now: the process pool, or the pool of the enclosing
/// [`with_threads`]. A caller sizing its work items to the pool asks here rather than the machine.
pub fn threads() -> usize {
    routed(rayon::current_num_threads)
}

/// Run `f` with the CPU backend's parallel loops confined to a pool of `threads`
/// threads (its own, dropped on return), however many cores the machine has.
/// A model whose kernels are a few hundred elements wide spends its time waking
/// and spinning a pool sized for a large one: a tiny training loop burned
/// hundreds of times its arithmetic in CPU time on a wide machine. For tests and
/// small jobs.
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().expect("a rayon pool").install(f)
}

/// Size the process-wide pool the parallel loops run in, before anything has used
/// it. This replaces setting `RAYON_NUM_THREADS`, which is process-global state
/// and which no running pool re-reads. Errs when the pool already exists.
pub fn set_global_threads(threads: usize) -> Result<(), String> {
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build_global().map_err(|e| format!("the CPU pool is already built: {e}"))
}

/// [`rows_mut`] over any `Send` element type - the shape of a loop whose
/// output rows are not `f32`: packed int8/int4 weight words (`u32`), decoded
/// block spans, index tables.
///
/// Same exactness contract as [`rows_mut`]: chunk `i` covers
/// `[i*chunk_len, (i+1)*chunk_len)` regardless of how many threads run, so a
/// loop whose per-chunk body reads only its own chunk and shared immutable
/// state produces a bit-identical result to the serial form - which is what
/// makes converting an existing serial loop a scheduling change rather than a
/// numerical one. A short tail chunk is passed as-is, never dropped.
pub fn chunks_mut<T: Send>(buf: &mut [T], chunk_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    assert!(chunk_len > 0, "chunks_mut: chunk_len must be non-zero");
    routed(|| buf.par_chunks_mut(chunk_len).enumerate().for_each(|(i, c)| f(i, c)));
}

/// [`chunks_mut`] over TWO outputs at once, each with its OWN chunk length:
/// index `i` gets `a[i*a_len..]` and `b[i*b_len..]`, in parallel.
///
/// The shape of a loop that produces two different things per row and needs
/// both - packed weight words plus that row's group scales being the
/// motivating case, where the two outputs have different widths (`k/4` u32
/// against `k/32` f32). Splitting it into two passes would either read the
/// input twice or recompute half the work, and neither is necessary when the
/// two outputs are disjoint.
///
/// Same exactness contract as [`chunks_mut`]: index `i` covers exactly its own
/// two chunks regardless of how many threads run, so a body reading only those
/// and shared immutable state is bit-identical to the serial form. `a` and `b`
/// must split into the same number of chunks.
pub fn chunks2_mut<T: Send, U: Send>(a: &mut [T], a_len: usize, b: &mut [U], b_len: usize, f: impl Fn(usize, &mut [T], &mut [U]) + Sync) {
    assert!(a_len > 0 && b_len > 0, "chunks2_mut: chunk lengths must be non-zero");
    assert_eq!(a.len().div_ceil(a_len), b.len().div_ceil(b_len), "chunks2_mut: {} chunks of a but {} of b", a.len().div_ceil(a_len), b.len().div_ceil(b_len));
    routed(|| a.par_chunks_mut(a_len).zip(b.par_chunks_mut(b_len)).enumerate().for_each(|(i, (ca, cb))| f(i, ca, cb)));
}

/// Apply `f(index, element)` to every element of `buf`, in parallel.
pub fn each_mut(buf: &mut [f32], f: impl Fn(usize, &mut f32) + Sync) {
    routed(|| buf.par_iter_mut().enumerate().for_each(|(i, v)| f(i, v)));
}

/// `(0..n).map(f)` in parallel, one `f32` per index - the shape of a matvec
/// fanned out over output rows.
pub fn map_f32(n: usize, f: impl Fn(usize) -> f32 + Sync + Send) -> Vec<f32> {
    routed(|| (0..n).into_par_iter().map(f).collect())
}

/// `(0..n).flat_map(f)` in parallel, preserving index order - the shape of a
/// per-head attention fan-out.
pub fn flat_map_f32(n: usize, f: impl Fn(usize) -> Vec<f32> + Sync + Send) -> Vec<f32> {
    routed(|| (0..n).into_par_iter().map(f).flatten().collect())
}

/// `(0..n).map(f)` in parallel, index-ordered, returning any `Send` value - the
/// shape of a fan-out over independent work items (one forecast per name, one
/// window per training row). Generalises [`map_f32`] to non-`f32` results so a
/// caller never reaches for a direct `rayon` dependency (the whole point of this
/// module: one pool, one policy). The pool is the scheduler's, so `--device
/// cpuN` sizing/affinity still governs it, and it fans out over the machine's
/// cores automatically at runtime.
pub fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    routed(|| (0..n).into_par_iter().map(f).collect())
}

/// Sum of squares over a set of tensors, accumulated in `f64` - the global
/// grad-norm reduction. `f64` accumulation is part of the contract: summing
/// millions of squares in `f32` loses the low bits the clip threshold needs.
///
/// The order of the additions is fixed: every tensor is cut into chunks of
/// 16K elements, each chunk is summed in sequence, and the partials
/// are added in order. Rayon's own `sum` splits adaptively, so its last bits
/// depend on how many threads are free, and a grad-norm that differs in its
/// last bit scales every weight update differently.
pub fn sum_sq_f64(tensors: &[Vec<f32>]) -> f64 {
    const SUM_CHUNK: usize = 16 * 1024;
    let partials: Vec<Vec<f64>> = routed(|| {
        tensors
            .par_iter()
            .map(|t| t.par_chunks(SUM_CHUNK).map(|c| c.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).collect())
            .collect()
    });
    partials.iter().flatten().sum()
}

/// Pairwise `f(a[i], b[i])` over two equal-length slices, in parallel - the
/// shape of an optimizer step over (state, grad) tensor pairs.
pub fn zip_each<T: Send, U: Sync>(a: &mut [T], b: &[U], f: impl Fn(&mut T, &U) + Sync) {
    assert_eq!(a.len(), b.len(), "zip_each: length mismatch {} vs {}", a.len(), b.len());
    routed(|| a.par_iter_mut().zip(b.par_iter()).for_each(|(x, y)| f(x, y)));
}

/// [`zip_each`] over three mutable state slices driven by one shared read-only
/// slice, in parallel - the shape of a flattened AdamW update (master/m/v all
/// mutated per-element from the same gradient element).
pub fn zip3_mut(a: &mut [f32], b: &mut [f32], c: &mut [f32], d: &[f32], f: impl Fn(&mut f32, &mut f32, &mut f32, &f32) + Sync) {
    assert_eq!(a.len(), b.len(), "zip3_mut: length mismatch {} vs {}", a.len(), b.len());
    assert_eq!(a.len(), c.len(), "zip3_mut: length mismatch {} vs {}", a.len(), c.len());
    assert_eq!(a.len(), d.len(), "zip3_mut: length mismatch {} vs {}", a.len(), d.len());
    routed(|| {
        a.par_iter_mut()
            .zip(b.par_iter_mut())
            .zip(c.par_iter_mut())
            .zip(d.par_iter())
            .for_each(|(((x, y), z), w)| f(x, y, z, w))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_mut_visits_every_row_once_with_its_index() {
        let mut buf = vec![0.0f32; 12];
        rows_mut(&mut buf, 4, |i, row| {
            for v in row.iter_mut() {
                *v = i as f32;
            }
        });
        assert_eq!(buf, vec![0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]);
    }

    #[test]
    fn rows_mut_passes_a_short_tail_rather_than_dropping_it() {
        let mut buf = vec![1.0f32; 5];
        rows_mut(&mut buf, 2, |_, row| {
            for v in row.iter_mut() {
                *v += 1.0;
            }
        });
        assert_eq!(buf, vec![2.0; 5], "the 1-element tail row must be processed");
    }

    #[test]
    fn map_and_flat_map_preserve_index_order() {
        assert_eq!(map_f32(4, |i| i as f32 * 2.0), vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(
            flat_map_f32(3, |i| vec![i as f32; 2]),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "flat_map must keep per-index blocks in order despite parallelism"
        );
    }

    #[test]
    fn map_generic_preserves_order() {
        assert_eq!(map(4, |i| (i, i * i)), vec![(0, 0), (1, 1), (2, 4), (3, 9)]);
        assert_eq!(map(3, |i| vec![i as u32; i]), vec![vec![], vec![1], vec![2, 2]]);
    }

    #[test]
    fn sum_sq_accumulates_in_f64() {
        // 1e8 elements of 1e-4: each square is 1e-8; the f32 running sum would
        // stall at ~0.25 once the accumulator dwarfs the addend. Use a smaller
        // proxy asserting exactness instead of a heavyweight demo.
        let t = vec![vec![3.0f32; 4], vec![4.0f32; 4]];
        assert_eq!(sum_sq_f64(&t), (9.0 * 4.0) + (16.0 * 4.0));
    }

    /// The reduction order is part of the result: a grad-norm that differs in
    /// its last bit scales every weight update differently, so two runs of one
    /// seed diverge. The sum must not depend on how many threads share it.
    #[test]
    fn sum_sq_is_bitwise_the_same_on_any_number_of_threads() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 10f32.powi((state % 7) as i32 - 3)
        };
        let tensors: Vec<Vec<f32>> = (0..37).map(|i| (0..1000 + 7919 * i).map(|_| next()).collect()).collect();
        let on = |threads: usize| rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| sum_sq_f64(&tensors).to_bits());
        let reference = on(1);
        for threads in [2, 3, 7, 16] {
            assert_eq!(on(threads), reference, "{threads} threads");
        }
    }

    #[test]
    fn with_threads_sizes_the_pool_the_loops_run_in() {
        assert_eq!(with_threads(3, rayon::current_num_threads), 3);
        let mut v = vec![0u32; 100];
        with_threads(2, || chunks_mut(&mut v, 10, |i, c| c.iter_mut().for_each(|x| *x = i as u32)));
        assert_eq!(v[99], 9);
    }

    /// `with_cores` confines the calling thread's loops to the named cores for the duration of the call and nothing
    /// after it: the loop bodies run on those cores, and the thread's affinity and pool are back as they were.
    #[cfg(target_os = "linux")]
    #[test]
    fn with_cores_runs_the_loops_on_those_cores_and_restores_the_thread() {
        let allowed = allowed_cores();
        if allowed.len() < 2 {
            return brain_testutil::skip_unavailable("fewer than two cores are available to this process");
        }
        let chosen = vec![allowed[0], allowed[allowed.len() - 1]];
        let (pool_threads, cores) = with_cores(&chosen, || {
            // Enough items that every pool thread is likely to take some; each reports the core it ran on.
            (threads(), map(512, |_| {
                std::thread::sleep(std::time::Duration::from_micros(200));
                // SAFETY: sched_getcpu has no preconditions.
                unsafe { libc::sched_getcpu() as usize }
            }))
        });
        assert_eq!(pool_threads, 2, "one pool thread per core");
        assert!(cores.iter().all(|c| chosen.contains(c)), "ran on {cores:?}, asked for {chosen:?}");
        assert_eq!(allowed_cores(), allowed, "the thread's affinity is restored");
        assert_eq!(threads(), rayon::current_num_threads(), "the process pool serves loops again");
    }

    #[test]
    fn zip_each_pairs_by_index_and_rejects_mismatch() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        zip_each(&mut a, &b, |x, y| *x += *y);
        assert_eq!(a, vec![11.0, 22.0, 33.0]);
        let r = std::panic::catch_unwind(|| {
            let mut a = vec![0.0f32; 2];
            zip_each(&mut a, &[0.0f32; 3], |_, _| {});
        });
        assert!(r.is_err(), "length mismatch must fail loudly, not truncate");
    }

    #[test]
    fn each_mut_is_elementwise_with_index() {
        let mut v = vec![0.0f32; 5];
        each_mut(&mut v, |i, x| *x = i as f32);
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zip3_mut_updates_all_three_from_the_shared_slice() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let mut b = vec![10.0f32, 20.0, 30.0];
        let mut c = vec![100.0f32, 200.0, 300.0];
        let d = vec![1.0f32, 1.0, 1.0];
        zip3_mut(&mut a, &mut b, &mut c, &d, |x, y, z, w| {
            *x += w;
            *y += 2.0 * w;
            *z += 3.0 * w;
        });
        assert_eq!(a, vec![2.0, 3.0, 4.0]);
        assert_eq!(b, vec![12.0, 22.0, 32.0]);
        assert_eq!(c, vec![103.0, 203.0, 303.0]);
        let r = std::panic::catch_unwind(|| {
            let mut a = vec![0.0f32; 2];
            let mut b = vec![0.0f32; 2];
            let mut c = vec![0.0f32; 2];
            zip3_mut(&mut a, &mut b, &mut c, &[0.0f32; 3], |_, _, _, _| {});
        });
        assert!(r.is_err(), "length mismatch must fail loudly, not truncate");
    }
}
