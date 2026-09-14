// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compile a kernel and run it on a device: context, allocations, modules,
//! launches.
//!
//! Swedish Embedded AB implements accelerator runtimes for its clients - the
//! context, memory and launch plumbing under a compute backend. If your team
//! needs expertise in driving a GPU from Rust without a vendor SDK in the
//! build, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Scope
//!
//! This is the execution substrate the generated (T0) tier is validated
//! against: enough to compile one kernel, move a few tiny buffers and launch a
//! grid. It is deliberately NOT [`backend_api::Backend`] - there is no
//! allocator, no stream, no graph and no step cache here, and a partial
//! `Backend` impl would be a backend that claims to run models and cannot.
//!
//! # The context is per device and shared
//!
//! `cuDevicePrimaryCtxRetain` rather than `cuCtxCreate`: the primary context is
//! what every other CUDA library on the process (and the runtime API, if
//! anything links it) uses, so retaining it keeps one context per device
//! instead of two competing ones. It is made current on the calling thread
//! before every operation, because a CUDA context is thread-local state and
//! this type is used from whichever test thread got it.
//!
//! # Nothing about the hardware is assumed
//!
//! The compute capability a kernel is compiled for is read back from the
//! device that will run it. There is no default and no fallback capability: a
//! cubin compiled for a capability the device does not have is either rejected
//! at module load or, worse, silently mis-scheduled.

use crate::driver::{
    CuContext, CuDevicePtr, CuFunction, CuGraph, CuGraphExec, CuGraphNode, CuKernelNodeParams,
    CuModule, CuStream, Driver, ExecFns, GraphFns, CAPTURE_MODE_THREAD_LOCAL, CAPTURE_STATUS_ACTIVE,
};
use std::ffi::{c_int, c_void, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A compiled cubin, and whether it came back from the on-disk cache.
pub struct Cubin {
    pub cubin: Vec<u8>,
    /// The content-addressed cache key - see `nvrtc::cache_key`.
    pub key: String,
    /// True when this compilation was served from disk rather than run.
    pub cached: bool,
}

/// One device with its primary context retained.
pub struct Context {
    d: &'static Driver,
    fns: &'static ExecFns,
    dev: std::ffi::c_int,
    ctx: CuContext,
    cc: (u32, u32),
    name: String,
    info: crate::driver::CudaDevice,
    /// Every dispatch and every clear this context issues goes here rather
    /// than on the legacy default stream, for one reason: the legacy stream
    /// cannot be captured. It is created blocking, so the synchronous host
    /// transfers that still use the legacy stream stay ordered against it.
    stream: CuStream,
    /// Incremented every time a device allocation is freed.
    ///
    /// A freed address may be handed straight back out by the next
    /// `cuMemAlloc`, so anything that recorded a device address and expects to
    /// reuse it later has to be able to ask whether any free has happened
    /// since. That is a question about the allocator, not about any one
    /// allocation, which is why the counter lives on the context and is
    /// bumped in `DeviceMem`'s `Drop` - the one place a free actually occurs.
    alloc_epoch: Arc<AtomicU64>,
}

// The context handle is used under `cuCtxSetCurrent` before every call, so it
// is sound to move between threads; the driver's own entry points are
// thread-safe.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Open the device at CUDA `ordinal`.
    ///
    /// An ordinal is a handle within THIS process, not an identity:
    /// `CUDA_VISIBLE_DEVICES` both filters and renumbers it. Callers that mean
    /// a particular physical card resolve it through the device registry's
    /// UUID first.
    pub fn open(ordinal: u32) -> Result<Context, String> {
        let d = crate::driver::driver().map_err(str::to_string)?;
        let fns = d.exec().map_err(str::to_string)?;
        let devices = d.devices()?;
        let info = devices
            .get(ordinal as usize)
            .ok_or_else(|| format!("CUDA ordinal {ordinal} does not exist ({} devices)", devices.len()))?;
        let dev = d.device_handle(ordinal)?;
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: `ctx` is a valid out-parameter and `dev` came from
        // `cuDeviceGet`. The retain is released in `Drop`.
        d.check(unsafe { (fns.primary_ctx_retain)(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
        let mut c = Context {
            d,
            fns,
            dev,
            ctx,
            cc: (info.cc_major, info.cc_minor),
            name: info.name.clone(),
            info: info.clone(),
            stream: std::ptr::null_mut(),
            alloc_epoch: Arc::new(AtomicU64::new(0)),
        };
        c.make_current()?;
        // SAFETY: `stream` is a valid out-parameter and the context is
        // current. Flag 0 is `CU_STREAM_DEFAULT` - see `ExecFns::stream_create`
        // on why it must not be the non-blocking one.
        d.check(unsafe { (fns.stream_create)(&mut c.stream, 0) }, "cuStreamCreate")?;
        Ok(c)
    }

    /// How many device allocations this context has freed. See
    /// [`Context::alloc_epoch`]'s field doc: a caller that cached a device
    /// address compares this against what it saw when it cached.
    pub fn alloc_epoch(&self) -> u64 {
        self.alloc_epoch.load(Ordering::Acquire)
    }

    /// The stream every dispatch and clear runs on.
    pub fn stream(&self) -> CuStream {
        self.stream
    }

    /// Everything the driver answered about this device, queried once when the
    /// context was opened. The ONE source a capability report may be built
    /// from - no field of it is written down anywhere in this crate.
    pub fn device_info(&self) -> &crate::driver::CudaDevice {
        &self.info
    }

    /// `cuMemGetInfo` on this device: `(free, total)` bytes, right now.
    ///
    /// The free figure moves while other processes run, so a caller that puts
    /// it in a cached capability must say when it was asked - see
    /// [`crate::backend::CudaBackend::caps`], which reads it once at
    /// construction on purpose rather than re-querying per call.
    pub fn mem_info(&self) -> Result<(u64, u64), String> {
        self.make_current()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are valid out-parameters for the duration of the call.
        self.d.check(unsafe { (self.fns.mem_get_info)(&mut free, &mut total) }, "cuMemGetInfo")?;
        Ok((free as u64, total as u64))
    }

    /// Compute capability of THIS device, as the driver reported it. Every
    /// compilation and every capability decision keys on this value; none may
    /// be written down anywhere.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.cc
    }

    /// The device's product name, for error messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn make_current(&self) -> Result<(), String> {
        // SAFETY: `self.ctx` is retained for the lifetime of `self`.
        self.d.check(unsafe { (self.fns.ctx_set_current)(self.ctx) }, "cuCtxSetCurrent")
    }

    /// Allocate `bytes` of device memory.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceMem, String> {
        self.make_current()?;
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: `ptr` is a valid out-parameter; the `_v2` entry point takes
        // a 64-bit size.
        self.d.check(unsafe { (self.fns.mem_alloc)(&mut ptr, bytes.max(1)) }, "cuMemAlloc")?;
        Ok(DeviceMem {
            d: self.d,
            fns: self.fns,
            ctx: self.ctx,
            ptr,
            len: bytes,
            epoch: self.alloc_epoch.clone(),
        })
    }

    /// Page-locked host staging of `words` u32s.
    ///
    /// The only reason this exists rather than a `Vec<u32>`: a copy whose
    /// source is ordinary pageable memory is rejected during a stream capture,
    /// because the driver cannot defer a staging copy it would otherwise make
    /// synchronously. A captured host->device copy node therefore has to name
    /// host memory that is already pinned.
    pub fn pinned(&self, words: usize) -> Result<PinnedMem, String> {
        self.make_current()?;
        let bytes = (words * 4).max(4);
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid out-parameter; the block is released in Drop.
        self.d.check(unsafe { (self.fns.mem_alloc_host)(&mut p, bytes) }, "cuMemAllocHost")?;
        Ok(PinnedMem { d: self.d, fns: self.fns, ctx: self.ctx, ptr: p as *mut u32, words })
    }

    /// Copy host bytes into a device allocation.
    pub fn upload(&self, mem: &DeviceMem, src: &[u8]) -> Result<(), String> {
        if src.len() > mem.len {
            return Err(format!("upload of {} bytes into a {}-byte allocation", src.len(), mem.len));
        }
        self.make_current()?;
        // SAFETY: `src` is valid for `src.len()` bytes and the destination was
        // allocated with at least that many.
        self.d.check(
            unsafe { (self.fns.memcpy_htod)(mem.ptr, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// Copy host bytes into a device allocation starting `offset` bytes in.
    ///
    /// The offset form exists so a bounded host upload can be split into
    /// chunks (`backend_api::Backend::write_at`) instead of one call sized to
    /// a whole multi-gigabyte tensor.
    pub fn upload_at(&self, mem: &DeviceMem, offset: usize, src: &[u8]) -> Result<(), String> {
        let end = offset.checked_add(src.len()).ok_or("upload offset + length overflows")?;
        if end > mem.len {
            return Err(format!(
                "upload of {} bytes at offset {offset} into a {}-byte allocation",
                src.len(),
                mem.len
            ));
        }
        if src.is_empty() {
            return Ok(());
        }
        self.make_current()?;
        // SAFETY: `src` is valid for `src.len()` bytes, and the bound above
        // proves the destination range lies inside the allocation.
        self.d.check(
            unsafe {
                (self.fns.memcpy_htod)(mem.ptr + offset as CuDevicePtr, src.as_ptr() as *const c_void, src.len())
            },
            "cuMemcpyHtoD",
        )
    }

    /// Zero the whole allocation.
    ///
    /// Storage a caller asked for is zeroed on every other backend in this
    /// engine (wgpu maps buffers zero-filled, the CPU backend allocates a
    /// zeroed `Vec`), and model code relies on it - an accumulator buffer is
    /// allocated and then added into. `cuMemAlloc` returns whatever the last
    /// tenant left, so the zeroing is explicit here or it does not happen.
    pub fn zero(&self, mem: &DeviceMem) -> Result<(), String> {
        if mem.len == 0 {
            return Ok(());
        }
        self.make_current()?;
        // SAFETY: `mem.ptr` names `mem.len` bytes allocated on this context.
        self.d.check(unsafe { (self.fns.memset_d8)(mem.ptr, 0, mem.len) }, "cuMemsetD8")
    }

    /// [`Self::zero`] enqueued on [`Self::stream`] instead of executed on the
    /// legacy stream.
    ///
    /// This is the form a submission's clears use, and it is not an
    /// optimisation: a legacy-stream operation issued from a thread that is
    /// capturing is one of the "potentially unsafe API calls" a capture
    /// rejects, so a clear that must be *inside* a graph has nowhere else to
    /// go. Outside capture it is the same operation one queue later.
    pub fn zero_async(&self, mem: &DeviceMem) -> Result<(), String> {
        if mem.len == 0 {
            return Ok(());
        }
        self.make_current()?;
        // SAFETY: `mem.ptr` names `mem.len` bytes on this context, and `mem`
        // outlives the enqueue; the caller keeps it alive until the stream has
        // drained (every caller here holds an `Arc` to it).
        self.d.check(
            unsafe { (self.fns.memset_d8_async)(mem.ptr, 0, mem.len, self.stream) },
            "cuMemsetD8Async",
        )
    }

    /// Enqueue a copy of `src`'s words into `mem` on [`Self::stream`].
    ///
    /// `src` must stay alive and unmodified until the stream reaches the copy,
    /// which is why it is a [`PinnedMem`] rather than a slice: the pinning is
    /// what a capture requires, and the ownership is what makes the lifetime
    /// statable.
    pub fn upload_async(&self, mem: &DeviceMem, src: &PinnedMem) -> Result<(), String> {
        let bytes = src.words * 4;
        if bytes > mem.len {
            return Err(format!("async upload of {bytes} bytes into a {}-byte allocation", mem.len));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.make_current()?;
        // SAFETY: `src` owns `bytes` page-locked bytes and the bound above
        // proves the destination holds at least that many.
        self.d.check(
            unsafe { (self.fns.memcpy_htod_async)(mem.ptr, src.ptr as *const c_void, bytes, self.stream) },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Copy device bytes back to the host.
    pub fn download(&self, mem: &DeviceMem, dst: &mut [u8]) -> Result<(), String> {
        if dst.len() > mem.len {
            return Err(format!("download of {} bytes from a {}-byte allocation", dst.len(), mem.len));
        }
        self.make_current()?;
        // SAFETY: `dst` is writable for `dst.len()` bytes and the source holds
        // at least that many.
        self.d.check(
            unsafe { (self.fns.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, mem.ptr, dst.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Compile `src` for THIS device's capability, through the on-disk cubin
    /// cache.
    pub fn cubin(&self, src: &str, entry: &str) -> Result<Cubin, String> {
        let version = crate::nvrtc::version()?;
        let key = crate::nvrtc::cache_key(src, entry, self.cc, version);
        if let Some(cubin) = crate::nvrtc::cache_load(&key) {
            return Ok(Cubin { cubin, key, cached: true });
        }
        let cubin = crate::nvrtc::compile(src, self.cc)?;
        crate::nvrtc::cache_store(&key, &cubin);
        Ok(Cubin { cubin, key, cached: false })
    }

    /// Compile and load `src`, returning the module its entry points live in.
    pub fn compile(&self, src: &str, entry: &str) -> Result<Module, String> {
        let c = self.cubin(src, entry)?;
        self.load(&c.cubin)
    }

    /// Load an already-compiled cubin.
    pub fn load(&self, cubin: &[u8]) -> Result<Module, String> {
        self.make_current()?;
        let mut m: CuModule = std::ptr::null_mut();
        // SAFETY: `cubin` is a complete image for this device's capability and
        // outlives the call; the driver copies what it needs.
        self.d.check(
            unsafe { (self.fns.module_load_data)(&mut m, cubin.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Module { d: self.d, fns: self.fns, ctx: self.ctx, m })
    }

    /// Launch `f` over `grid` blocks of `block` threads with one pointer
    /// argument per entry in `args`, in order.
    ///
    /// `cuLaunchKernel` takes an array of pointers TO the argument values, so
    /// a pointer argument is passed as a pointer to the `CUdeviceptr` - not as
    /// the device address itself. Getting that one indirection wrong reads a
    /// host stack address as a device pointer, which faults rather than
    /// miscomputing, but only once a kernel dereferences it.
    pub fn launch(
        &self,
        f: &Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[&DeviceMem],
    ) -> Result<(), String> {
        let ptrs: Vec<CuDevicePtr> = args.iter().map(|a| a.ptr).collect();
        self.launch_at(f, grid, block, &ptrs)
    }

    /// [`Self::launch`] against raw device addresses rather than whole
    /// allocations - what a sub-range binding needs.
    ///
    /// A WGSL `step_sliced` binds `(word_offset, word_len)` of a buffer, which
    /// on this API is just the base address plus `4 * word_offset`: a kernel
    /// argument is a bare pointer, so the slice is expressed by the address
    /// handed in and nothing else. The LENGTH is deliberately not passed -
    /// the generated kernel bounds itself from its own uniform, exactly as the
    /// WGSL does, and a binding size would be a second, redundant source of
    /// truth about the same range.
    pub fn launch_at(
        &self,
        f: &Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[CuDevicePtr],
    ) -> Result<(), String> {
        // SAFETY: `f` borrows the module that defines it, so it is loaded.
        unsafe { self.launch_raw(f.f, grid, block, args) }
    }

    /// [`Self::launch_at`] against an already-resolved entry point.
    ///
    /// The resolution is hoisted out for one reason: a launch that is being
    /// recorded into a graph should be a launch and nothing else, so the
    /// module lookup happens before the capture opens rather than inside it.
    ///
    /// # Safety
    /// `f` must name an entry point of a module that is still loaded in this
    /// context, and `args` must be exactly the arguments it takes.
    pub unsafe fn launch_raw(
        &self,
        f: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[CuDevicePtr],
    ) -> Result<(), String> {
        self.make_current()?;
        let mut values: Vec<CuDevicePtr> = args.to_vec();
        let mut params: Vec<*mut c_void> =
            values.iter_mut().map(|v| v as *mut CuDevicePtr as *mut c_void).collect();
        // SAFETY: `params` names `args.len()` valid pointers to device
        // addresses that outlive the call, the function belongs to a module
        // still loaded in this context (the caller's obligation), and no
        // dynamic shared memory is requested (the generated kernels declare
        // theirs statically).
        self.d.check(
            unsafe {
                (self.fns.launch_kernel)(
                    f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// Whether this driver exposes the CUDA Graphs entry points at all, and if
    /// not, why. An `Err` costs launch batching and nothing else.
    pub fn graphs(&self) -> Result<&'static GraphFns, &'static str> {
        self.d.graph()
    }

    /// Begin capturing [`Self::stream`] into a graph.
    ///
    /// The returned guard ends the capture whatever happens next, including a
    /// panic: a stream left in capture mode swallows every subsequent
    /// operation on it silently, so a dropped capture is not a leak but a
    /// device that stops computing.
    pub fn begin_capture(&self) -> Result<Capture<'_>, String> {
        let g = self.graphs().map_err(str::to_string)?;
        self.make_current()?;
        // SAFETY: the context is current and `self.stream` was created on it.
        // The mode is thread-local, so this constrains only this thread.
        self.d.check(
            unsafe { (g.stream_begin_capture)(self.stream, CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;
        Ok(Capture { ctx: self, g, ended: false })
    }

    /// Instantiate `graph` into an executable one.
    pub fn instantiate(&self, graph: &Graph) -> Result<GraphExec, String> {
        let g = self.graphs().map_err(str::to_string)?;
        self.make_current()?;
        let mut e: CuGraphExec = std::ptr::null_mut();
        // SAFETY: `e` is a valid out-parameter and `graph` is a complete graph
        // built on this context. No instantiation flags are requested.
        self.d.check(unsafe { (g.graph_instantiate)(&mut e, graph.graph, 0) }, "cuGraphInstantiate")?;
        Ok(GraphExec { d: self.d, g, ctx: self.ctx, e })
    }

    /// Enqueue a whole instantiated graph on [`Self::stream`] - one driver
    /// call in place of every launch it contains.
    pub fn launch_graph(&self, e: &GraphExec) -> Result<(), String> {
        let g = self.graphs().map_err(str::to_string)?;
        self.make_current()?;
        // SAFETY: `e` was instantiated on this context and every buffer its
        // nodes name is kept alive by the caller for as long as `e` lives.
        self.d.check(unsafe { (g.graph_launch)(e.e, self.stream) }, "cuGraphLaunch")
    }

    /// Re-point one already-instantiated kernel node at a new grid (and, if it
    /// moved, new arguments), without rebuilding or re-instantiating anything.
    ///
    /// This is what makes a captured graph survive a dispatch whose thread
    /// count grew - a sequence length advancing, most often. The alternative,
    /// one instantiated graph per grid size, costs more in instantiation than
    /// the batching saves.
    ///
    /// # Safety
    /// `node` must be a node of the graph `e` was instantiated from, and
    /// `params.kernel_params` must name exactly the arguments the node's
    /// function takes. The driver copies the argument values during the call.
    pub unsafe fn set_kernel_node_grid(
        &self,
        e: &GraphExec,
        node: CuGraphNode,
        params: &CuKernelNodeParams,
    ) -> Result<(), String> {
        let g = self.graphs().map_err(str::to_string)?;
        self.make_current()?;
        self.d.check(
            (g.graph_exec_kernel_node_set_params)(e.e, node, params),
            "cuGraphExecKernelNodeSetParams",
        )
    }

    /// Block until every launch on this context has completed. A launch is
    /// asynchronous and reports only the errors it can see BEFORE running, so
    /// this is where a faulting kernel is actually reported.
    pub fn sync(&self) -> Result<(), String> {
        self.make_current()?;
        // SAFETY: the context is current on this thread.
        self.d.check(unsafe { (self.fns.ctx_synchronize)() }, "cuCtxSynchronize")
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the stream was created on this context and is destroyed
        // once; the release balances the retain in `open`. The primary context
        // is reference-counted, so this does not tear down a context another
        // holder is still using.
        unsafe {
            if !self.stream.is_null() && (self.fns.ctx_set_current)(self.ctx) == 0 {
                (self.fns.stream_destroy)(self.stream);
            }
            (self.fns.primary_ctx_release)(self.dev);
        }
    }
}

/// An in-progress stream capture. Ending it is [`Capture::finish`]; dropping it
/// without finishing ends the capture anyway and discards the graph.
///
/// The unconditional end in `Drop` is the important part. `cuStreamEndCapture`
/// is the ONLY way out of capture mode, and a stream still in capture mode
/// accepts every subsequent operation and executes none of them - so an error
/// path that forgot to end the capture would not leak a handle, it would
/// silently stop the device from computing anything for the rest of the
/// process.
pub struct Capture<'a> {
    ctx: &'a Context,
    g: &'static GraphFns,
    ended: bool,
}

impl Capture<'_> {
    /// The node the launch just recorded on the capture stream.
    ///
    /// Read as the stream's current dependency set, which after a single
    /// recorded operation is exactly that operation's node.
    /// `cuGraphGetNodes` cannot answer this: it returns a graph's nodes in an
    /// unspecified order and so cannot say which node came from which launch.
    pub fn last_node(&self) -> Result<CuGraphNode, String> {
        let mut status: c_int = 0;
        let mut id: u64 = 0;
        let mut graph: CuGraph = std::ptr::null_mut();
        let mut deps: *const CuGraphNode = std::ptr::null();
        let mut n: usize = 0;
        // SAFETY: every out-parameter is valid for the call, and
        // `dependencies_out` borrows driver-owned storage that stays valid
        // until the next operation is recorded on this stream - it is read
        // below, before anything else is recorded.
        self.ctx.d.check(
            unsafe {
                (self.g.stream_get_capture_info)(
                    self.ctx.stream,
                    &mut status,
                    &mut id,
                    &mut graph,
                    &mut deps,
                    &mut n,
                )
            },
            "cuStreamGetCaptureInfo",
        )?;
        if status != CAPTURE_STATUS_ACTIVE {
            return Err(format!("the capture stream is no longer capturing (status {status})"));
        }
        if n != 1 || deps.is_null() {
            // More than one would mean the launch was not the only thing
            // recorded since the last read, which is a defect in this file's
            // own bookkeeping rather than a driver condition.
            return Err(format!("a single recorded launch left {n} capture dependencies, not 1"));
        }
        // SAFETY: `n == 1` and `deps` is non-null, so `deps[0]` is in bounds.
        Ok(unsafe { *deps })
    }

    /// End the capture and take the graph that was recorded.
    pub fn finish(mut self) -> Result<Graph, String> {
        self.ended = true;
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: `graph` is a valid out-parameter; the stream is the one
        // `begin_capture` started on.
        self.ctx
            .d
            .check(unsafe { (self.g.stream_end_capture)(self.ctx.stream, &mut graph) }, "cuStreamEndCapture")?;
        if graph.is_null() {
            return Err("cuStreamEndCapture produced no graph".into());
        }
        Ok(Graph { d: self.ctx.d, g: self.g, graph })
    }
}

impl Drop for Capture<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: ends the capture begun on this same stream. The graph it
        // hands back is discarded immediately - this path is reached only
        // when the capture is being abandoned.
        unsafe {
            if (self.g.stream_end_capture)(self.ctx.stream, &mut graph) == 0 && !graph.is_null() {
                (self.g.graph_destroy)(graph);
            }
        }
    }
}

/// A recorded, not yet executable, graph.
pub struct Graph {
    d: &'static Driver,
    g: &'static GraphFns,
    graph: CuGraph,
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: destroyed once; an instantiated `GraphExec` made from it
        // stays valid, the driver documents the two lifetimes as independent.
        unsafe {
            let rc = (self.g.graph_destroy)(self.graph);
            if rc != 0 {
                tracing::warn!("cuGraphDestroy failed: {}", self.d.error_text(rc));
            }
        }
    }
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

/// An instantiated graph: the launchable form.
pub struct GraphExec {
    d: &'static Driver,
    g: &'static GraphFns,
    ctx: CuContext,
    e: CuGraphExec,
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: the exec was instantiated on this context and is destroyed
        // once.
        unsafe {
            let rc = (self.g.graph_exec_destroy)(self.e);
            if rc != 0 {
                tracing::warn!("cuGraphExecDestroy failed: {}", self.d.error_text(rc));
            }
            let _ = self.ctx;
        }
    }
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

/// A page-locked host block of `words` u32s.
pub struct PinnedMem {
    d: &'static Driver,
    fns: &'static ExecFns,
    ctx: CuContext,
    ptr: *mut u32,
    words: usize,
}

impl PinnedMem {
    /// Length in u32 words.
    pub fn words(&self) -> usize {
        self.words
    }

    /// Overwrite the block with `src`.
    ///
    /// `&self` rather than `&mut self`: the block is reachable through an
    /// `Arc` shared with a recorded graph node, and the node holds only the
    /// ADDRESS. Writing here is how a replay supplies new parameters, and it
    /// is the caller's obligation - stated on every caller in this crate - to
    /// do it while the stream that reads it has drained.
    ///
    /// # Panics
    /// If `src` is longer than the block.
    pub fn fill(&self, src: &[u32]) {
        assert!(
            src.len() <= self.words,
            "backend-cuda: {} words written into a {}-word pinned staging block",
            src.len(),
            self.words
        );
        // SAFETY: the block owns `self.words` u32s, the bound above proves
        // `src` fits, and the two regions cannot overlap (one is page-locked
        // driver memory, the other is the caller's).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, src.len()) };
    }
}

impl Drop for PinnedMem {
    fn drop(&mut self) {
        // SAFETY: the block came from `cuMemAllocHost` on this context and is
        // freed once.
        unsafe {
            if (self.fns.ctx_set_current)(self.ctx) == 0 {
                let rc = (self.fns.mem_free_host)(self.ptr as *mut c_void);
                if rc != 0 {
                    tracing::warn!("cuMemFreeHost failed: {}", self.d.error_text(rc));
                }
            }
        }
    }
}

// Same reasoning as `DeviceMem`: the context is made current before use, and
// page-locked host memory is not thread-affine.
unsafe impl Send for PinnedMem {}
unsafe impl Sync for PinnedMem {}

/// One device allocation, freed when dropped.
pub struct DeviceMem {
    d: &'static Driver,
    fns: &'static ExecFns,
    ctx: CuContext,
    ptr: CuDevicePtr,
    len: usize,
    /// The context's free counter, bumped here - see
    /// [`Context::alloc_epoch`]'s field doc.
    epoch: Arc<AtomicU64>,
}

impl DeviceMem {
    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The device address of this allocation, for a caller building a
    /// sub-range binding - see [`Context::launch_at`].
    pub fn device_ptr(&self) -> CuDevicePtr {
        self.ptr
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        // The context must be current for the free to reach the right device.
        // SAFETY: `ctx` is kept alive by the `Context` that made this
        // allocation, and `ptr` came from `cuMemAlloc` on it.
        unsafe {
            if (self.fns.ctx_set_current)(self.ctx) == 0 {
                let rc = (self.fns.mem_free)(self.ptr);
                if rc != 0 {
                    tracing::warn!("cuMemFree failed: {}", self.d.error_text(rc));
                }
            }
        }
        // AFTER the free, and unconditionally: from here on the driver may
        // hand this address to anyone, so anything caching device addresses
        // must be able to see that it happened even if the free itself failed.
        self.epoch.fetch_add(1, Ordering::Release);
    }
}

// Same reasoning as `Context`: a device address is not thread-affine, the
// context is made current before it is used.
unsafe impl Send for DeviceMem {}
unsafe impl Sync for DeviceMem {}

/// A loaded module, unloaded when dropped.
pub struct Module {
    d: &'static Driver,
    fns: &'static ExecFns,
    ctx: CuContext,
    m: CuModule,
}

impl Module {
    /// Resolve an entry point by name. The generated kernels are `extern "C"`,
    /// so the name in the source is the name in the symbol table.
    pub fn function(&self, name: &str) -> Result<Function<'_>, String> {
        let cname = CString::new(name).map_err(|_| "entry name contains a NUL byte".to_string())?;
        let mut f: CuFunction = std::ptr::null_mut();
        // SAFETY: `self.m` is loaded in `self.ctx`, and `cname` outlives the
        // call.
        self.d.check(
            unsafe { (self.fns.module_get_function)(&mut f, self.m, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function { f, _module: self })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the module was loaded in this context and is unloaded once.
        unsafe {
            if (self.fns.ctx_set_current)(self.ctx) == 0 {
                (self.fns.module_unload)(self.m);
            }
        }
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// An entry point. Borrows its module, because a `CUfunction` is only valid
/// while the module that defines it is loaded.
pub struct Function<'a> {
    f: CuFunction,
    _module: &'a Module,
}

impl Function<'_> {
    /// The raw handle, for a caller that keeps the owning [`Module`] alive by
    /// some other means than this borrow - a recorded graph node names a
    /// `CUfunction` and must outlive the resolution that produced it.
    pub fn raw(&self) -> CuFunction {
        self.f
    }
}
