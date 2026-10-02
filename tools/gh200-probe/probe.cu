// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring and tuning GPU
// workloads on Grace-Hopper systems for its clients. If your team needs
// expertise in CUDA performance engineering then you can procure our services
// by sending an email to info@swedishembedded.com.

// Hardware characterization of one CUDA device, written to be read as JSON by
// the people (and tests) deciding where brain's kernels stand against the
// machine's ceilings. It reports what the driver exposes (compute capability,
// memory-access attributes that decide whether system memory is GPU-visible),
// and measures: HBM copy bandwidth, host-memory read/write bandwidth through
// the CPU-GPU link (pinned, pageable-over-ATS and managed allocations), kernel
// launch latency, and dense GEMM throughput through cuBLASLt for the operand
// types a Hopper tensor core offers. Every number is a ceiling measurement of
// this machine under this driver; none is a claim about brain.
//
// Build:  make -C tools/gh200-probe   (needs a CUDA toolkit, see `make cuda/install`)
// Run:    tools/gh200-probe/probe > probe.json
//         tools/gh200-probe/probe --placement   (device vs managed vs system memory)

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>

#define CK(x)                                                                      \
    do {                                                                           \
        cudaError_t e_ = (x);                                                      \
        if (e_ != cudaSuccess) {                                                   \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                               \
        }                                                                          \
    } while (0)
#define CKL(x)                                                                     \
    do {                                                                           \
        cublasStatus_t s_ = (x);                                                   \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "%s:%d: %s: cublas status %d\n", __FILE__, __LINE__, #x, (int)s_); \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

static const int WARMUP = 3;
static const int REPS = 10;

// Times `fn` with CUDA events: the median of REPS runs after WARMUP, in ms.
template <class F>
static double time_ms(F fn) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < WARMUP; i++) fn();
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < REPS; i++) {
        CK(cudaEventRecord(a));
        fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return t[t.size() / 2];
}

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
// Streams a read-only region and folds it, so the compiler cannot drop the loads.
__global__ void read_kernel(const float4* __restrict__ src, size_t n, float* out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) *out = acc;
}
__global__ void write_kernel(float4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = make_float4(1.f, 2.f, 3.f, 4.f);
}
__global__ void empty_kernel() {}

static int attr(cudaDeviceAttr a, int dev) {
    int v = -1;
    cudaDeviceGetAttribute(&v, a, dev);
    return v;
}

// GB/s for `bytes` moved in `ms`.
static double gbs(double bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; }

static void bandwidth(int sms, double* hbm_copy, double* hbm_read, double* hbm_write) {
    const size_t bytes = 4ull << 30;
    float4 *a, *b;
    float* sink;
    CK(cudaMalloc(&a, bytes));
    CK(cudaMalloc(&b, bytes));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(a, 1, bytes));
    size_t n = bytes / sizeof(float4);
    int blocks = sms * 8;
    *hbm_copy = gbs(2.0 * bytes, time_ms([&] { copy_kernel<<<blocks, 512>>>(a, b, n); }));
    *hbm_read = gbs((double)bytes, time_ms([&] { read_kernel<<<blocks, 512>>>(a, n, sink); }));
    *hbm_write = gbs((double)bytes, time_ms([&] { write_kernel<<<blocks, 512>>>(b, n); }));
    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(sink));
}

struct HostBw {
    double read_gbs, write_gbs, h2d_gbs, d2h_gbs;
    bool ok;
};

// Bandwidth of a kernel (and of cudaMemcpy) against a host-side allocation.
// `kind`: 0 = cudaMallocHost (pinned), 1 = malloc (pageable, needs ATS or HMM),
// 2 = cudaMallocManaged.
static HostBw host_bw(int kind, int sms) {
    const size_t bytes = 2ull << 30;
    HostBw r{0, 0, 0, 0, false};
    void* h = nullptr;
    if (kind == 0) {
        if (cudaMallocHost(&h, bytes) != cudaSuccess) return r;
    } else if (kind == 1) {
        h = aligned_alloc(1 << 16, bytes);
        if (!h) return r;
    } else {
        if (cudaMallocManaged(&h, bytes) != cudaSuccess) return r;
    }
    memset(h, 1, bytes);  // first touch: place the pages on the CPU side
    float* sink;
    CK(cudaMalloc(&sink, 4));
    size_t n = bytes / sizeof(float4);
    int blocks = sms * 8;
    // A kernel that cannot dereference this pointer reports an error rather than a number.
    read_kernel<<<blocks, 512>>>((const float4*)h, n, sink);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        if (kind == 0) cudaFreeHost(h); else if (kind == 1) free(h); else cudaFree(h);
        cudaFree(sink);
        return r;
    }
    r.read_gbs = gbs((double)bytes, time_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)h, n, sink); }));
    r.write_gbs = gbs((double)bytes, time_ms([&] { write_kernel<<<blocks, 512>>>((float4*)h, n); }));
    void* d;
    CK(cudaMalloc(&d, bytes));
    r.h2d_gbs = gbs((double)bytes, time_ms([&] { cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); }));
    r.d2h_gbs = gbs((double)bytes, time_ms([&] { cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost); }));
    r.ok = true;
    CK(cudaFree(d));
    CK(cudaFree(sink));
    if (kind == 0) cudaFreeHost(h); else if (kind == 1) free(h); else cudaFree(h);
    return r;
}

static void print_host(const char* name, const HostBw& r, bool last) {
    if (!r.ok) {
        printf("    \"%s\": null%s\n", name, last ? "" : ",");
        return;
    }
    printf("    \"%s\": {\"gpu_read_gbs\": %.1f, \"gpu_write_gbs\": %.1f, \"memcpy_h2d_gbs\": %.1f, \"memcpy_d2h_gbs\": %.1f}%s\n", name, r.read_gbs, r.write_gbs, r.h2d_gbs, r.d2h_gbs, last ? "" : ",");
}

// Dense GEMM: C[m,n] = A[m,k] B[k,n] on cuBLASLt, returning TFLOP/s or a negative
// number when this operand combination is not offered on the device.
struct GemmCase {
    const char* name;
    cudaDataType a, b, c;
    cublasComputeType_t compute;
    cudaDataType scale;
    size_t elem_a, elem_c;
};

static double gemm_tflops(cublasLtHandle_t lt, const GemmCase& g, int m, int n, int k, void* ws, size_t ws_bytes) {
    void *A, *B, *C;
    CK(cudaMalloc(&A, (size_t)m * k * g.elem_a));
    CK(cudaMalloc(&B, (size_t)k * n * g.elem_a));
    CK(cudaMalloc(&C, (size_t)m * n * g.elem_c));
    CK(cudaMemset(A, 0x3c, (size_t)m * k * g.elem_a));
    CK(cudaMemset(B, 0x3c, (size_t)k * n * g.elem_a));
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, lc;
    if (cublasLtMatmulDescCreate(&op, g.compute, g.scale) != CUBLAS_STATUS_SUCCESS) return -1;
    // cuBLASLt is column-major; FP8 requires the "TN" form (A transposed).
    cublasOperation_t t = CUBLAS_OP_T, nt = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nt, sizeof(nt)));
    CKL(cublasLtMatrixLayoutCreate(&la, g.a, k, m, k));
    CKL(cublasLtMatrixLayoutCreate(&lb, g.b, k, n, k));
    CKL(cublasLtMatrixLayoutCreate(&lc, g.c, m, n, m));
    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
    cublasLtMatmulHeuristicResult_t h;
    int found = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &h, &found);
    double tf = -1;
    if (st == CUBLAS_STATUS_SUCCESS && found > 0) {
        float one = 1.f, zero = 0.f;
        double one_d = 1.0, zero_d = 0.0;
        const void* alpha = g.scale == CUDA_R_32I ? (const void*)&one : (g.scale == CUDA_R_64F ? (const void*)&one_d : (const void*)&one);
        const void* beta = g.scale == CUDA_R_64F ? (const void*)&zero_d : (const void*)&zero;
        int ia = 1, ib = 0;
        if (g.scale == CUDA_R_32I) { alpha = &ia; beta = &ib; }
        double ms = time_ms([&] { cublasLtMatmul(lt, op, alpha, A, la, B, lb, beta, C, lc, C, lc, &h.algo, ws, ws_bytes, 0); });
        tf = 2.0 * m * n * k / (ms * 1e-3) / 1e12;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    CK(cudaFree(A));
    CK(cudaFree(B));
    CK(cudaFree(C));
    return tf;
}


// ---------------------------------------------------------------------------
// Placement: the same read/write workload against device, managed and system
// (pageable, GPU-addressable through ATS) memory, so the cost of each placement
// is a measurement on this machine and not an assumption. Run with
// `probe --placement`; BUFFER_MIB (default 1024) sizes the buffer, and
// OVERSUBSCRIBE=1 additionally allocates managed memory past the card's free
// memory (it takes all of the free memory: do not run it on a shared card).
// ---------------------------------------------------------------------------

// One timed pass of `fn` with CUDA events, in ms. Unlike time_ms this runs once,
// because a first touch cannot be repeated.
template <class F>
static double once_ms(F fn) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    fn();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return ms;
}

enum Kind { DEVICE = 0, MANAGED = 1, SYSTEM = 2 };

// Fresh pages every call: system memory comes from an anonymous mapping so no
// page is populated until something touches it.
static void* place_alloc(Kind k, size_t bytes) {
    void* p = nullptr;
    if (k == DEVICE) {
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    } else if (k == MANAGED) {
        if (cudaMallocManaged(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    } else {
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return nullptr;
    }
    return p;
}

static void place_free(Kind k, void* p, size_t bytes) {
    if (k == DEVICE || k == MANAGED) cudaFree(p); else munmap(p, bytes);
}

static const char* kind_name(Kind k) { return k == DEVICE ? "device" : k == MANAGED ? "managed" : "system"; }

// A pass that faults or errors reports null instead of a number.
static bool pass_ok() {
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

static void placement_one(Kind k, int sms, size_t bytes, float* sink, bool last) {
    const size_t n = bytes / sizeof(float4);
    const int blocks = sms * 8;
    void* p = place_alloc(k, bytes);
    printf("    \"%s\": ", kind_name(k));
    if (!p) { printf("null%s\n", last ? "" : ","); return; }

    // 1. First touch by the GPU: the first write pass over pages nobody has touched.
    double gpu_first = once_ms([&] { write_kernel<<<blocks, 512>>>((float4*)p, n); });
    bool ok = pass_ok();
    // 2. Steady state: the same pages read again, after warm-up.
    double steady = ok ? gbs((double)bytes, time_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)p, n, sink); })) : 0;

    // 3. First touch by the CPU, then the GPU's first read of those pages: where
    //    the data lives after the CPU wrote it decides what this costs.
    double cpu_first_read = 0, cpu_placed_steady = 0, prefetch_ms = 0, after_prefetch_read = 0, host_prefetch_back_ms = 0;
    if (k != DEVICE && ok) {
        place_free(k, p, bytes);
        p = place_alloc(k, bytes);
        memset(p, 1, bytes);
        cpu_first_read = gbs((double)bytes, once_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)p, n, sink); }));
        // Repeated reads of pages the CPU placed: whether they stay on the host side
        // (the link's rate) or are migrated to the device (HBM's) is the finding.
        cpu_placed_steady = pass_ok() ? gbs((double)bytes, time_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)p, n, sink); })) : 0;
        pass_ok();
        // 4. Prefetch effect: put the pages back on the host side, then move them to
        //    the device ahead of the read and time both the move and the read.
        int dev = 0;
#if CUDART_VERSION >= 13000
        cudaMemLocation host_loc = {cudaMemLocationTypeHost, 0};
        cudaMemLocation dev_loc = {cudaMemLocationTypeDevice, dev};
#define PREFETCH_HOST() cudaMemPrefetchAsync(p, bytes, host_loc, 0, 0)
#define PREFETCH_DEVICE() cudaMemPrefetchAsync(p, bytes, dev_loc, 0, 0)
#else
#define PREFETCH_HOST() cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId, 0)
#define PREFETCH_DEVICE() cudaMemPrefetchAsync(p, bytes, dev, 0)
#endif
        if (PREFETCH_HOST() == cudaSuccess) {
            host_prefetch_back_ms = once_ms([&] {});  // drain
            prefetch_ms = once_ms([&] { PREFETCH_DEVICE(); });
            after_prefetch_read = gbs((double)bytes, once_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)p, n, sink); }));
            pass_ok();
        } else {
            cudaGetLastError();
            prefetch_ms = -1;
        }
    }
    printf("{\"gpu_first_touch_write_gbs\": %.1f, \"steady_read_gbs\": %.1f", ok ? gbs((double)bytes, gpu_first) : 0.0, steady);
    if (k != DEVICE) {
        printf(", \"cpu_first_touch_then_gpu_read_gbs\": %.1f, \"cpu_placed_steady_read_gbs\": %.1f, \"prefetch_to_device_ms\": %.2f, \"read_after_prefetch_gbs\": %.1f",
               cpu_first_read, cpu_placed_steady, prefetch_ms, after_prefetch_read);
    }
    printf("}%s\n", last ? "" : ",");
    (void)host_prefetch_back_ms;
    place_free(k, p, bytes);
}

// Managed memory past the card's free memory: the driver evicts to host memory,
// so a read pass shows the cost of thrashing. Takes all free device memory.
static void oversubscription(int sms, float* sink) {
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    size_t bytes = free_b + (free_b / 4);
    bytes &= ~((size_t)(1 << 21) - 1);
    void* p = place_alloc(MANAGED, bytes);
    printf("  \"oversubscription\": ");
    if (!p) { printf("null\n"); return; }
    const size_t n = bytes / sizeof(float4);
    const int blocks = sms * 8;
    memset(p, 1, bytes);
    double first = gbs((double)bytes, once_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)p, n, sink); }));
    bool ok = pass_ok();
    double second = ok ? gbs((double)bytes, once_ms([&] { read_kernel<<<blocks, 512>>>((const float4*)p, n, sink); })) : 0;
    ok = ok && pass_ok();
    printf("{\"bytes\": %zu, \"free_bytes_before\": %zu, \"ok\": %s, \"first_pass_gbs\": %.1f, \"second_pass_gbs\": %.1f}\n",
           bytes, free_b, ok ? "true" : "false", first, second);
    place_free(MANAGED, p, bytes);
}

static void placement(int sms) {
    const char* e = getenv("BUFFER_MIB");
    size_t bytes = (e ? strtoull(e, nullptr, 10) : 1024ull) << 20;
    float* sink;
    CK(cudaMalloc(&sink, 4));
    printf("{\n  \"buffer_bytes\": %zu,\n  \"placement\": {\n", bytes);
    placement_one(DEVICE, sms, bytes, sink, false);
    placement_one(MANAGED, sms, bytes, sink, false);
    placement_one(SYSTEM, sms, bytes, sink, true);
    const char* over = getenv("OVERSUBSCRIBE");
    printf("  }%s\n", over && over[0] == '1' ? "," : "");
    if (over && over[0] == '1') oversubscription(sms, sink);
    printf("}\n");
    CK(cudaFree(sink));
}

int main(int argc, char** argv) {
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    if (argc > 1 && strcmp(argv[1], "--placement") == 0) {
        placement(p.multiProcessorCount);
        return 0;
    }
    int rt, drv;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    printf("{\n  \"device\": {\"name\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"clock_khz\": %d, \"mem_clock_khz\": %d, \"bus_width_bits\": %d,\n",
           p.name, p.major, p.minor, p.multiProcessorCount, attr(cudaDevAttrClockRate, dev), attr(cudaDevAttrMemoryClockRate, dev), attr(cudaDevAttrGlobalMemoryBusWidth, dev));
    printf("             \"total_mem_bytes\": %zu, \"free_mem_bytes\": %zu, \"l2_bytes\": %d, \"smem_per_block_optin\": %d, \"smem_per_sm\": %d, \"regs_per_sm\": %d,\n",
           total_b, free_b, p.l2CacheSize, (int)p.sharedMemPerBlockOptin, (int)p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor);
    printf("             \"cuda_runtime\": %d, \"cuda_driver\": %d, \"integrated\": %d, \"unified_addressing\": %d,\n", rt, drv, p.integrated, p.unifiedAddressing);
    printf("             \"pageable_memory_access\": %d, \"pageable_via_host_page_tables\": %d, \"concurrent_managed_access\": %d, \"direct_managed_from_host\": %d,\n",
           attr(cudaDevAttrPageableMemoryAccess, dev), attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev), attr(cudaDevAttrConcurrentManagedAccess, dev), attr(cudaDevAttrDirectManagedMemAccessFromHost, dev));
    printf("             \"host_native_atomics\": %d, \"cluster_launch\": %d, \"memory_pools\": %d, \"gpu_direct_rdma\": %d, \"async_engines\": %d}",
           attr(cudaDevAttrHostNativeAtomicSupported, dev), attr(cudaDevAttrClusterLaunch, dev), attr(cudaDevAttrMemoryPoolsSupported, dev), attr(cudaDevAttrGPUDirectRDMASupported, dev), p.asyncEngineCount);

    double copy, rd, wr;
    bandwidth(p.multiProcessorCount, &copy, &rd, &wr);
    printf(",\n  \"hbm_gbs\": {\"copy\": %.1f, \"read\": %.1f, \"write\": %.1f},\n", copy, rd, wr);

    printf("  \"host_memory\": {\n");
    print_host("pinned", host_bw(0, p.multiProcessorCount), false);
    print_host("pageable_system", host_bw(1, p.multiProcessorCount), false);
    print_host("managed", host_bw(2, p.multiProcessorCount), true);
    printf("  },\n");

    double launch_us = time_ms([&] { for (int i = 0; i < 1000; i++) empty_kernel<<<1, 32>>>(); }) * 1000.0 / 1000.0;
    printf("  \"launch_latency_us\": %.2f,\n", launch_us);

    cublasLtHandle_t lt;
    CKL(cublasLtCreate(&lt));
    size_t ws_bytes = 64ull << 20;
    void* ws;
    CK(cudaMalloc(&ws, ws_bytes));
    const GemmCase cases[] = {
        {"fp32", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, 4, 4},
        {"tf32", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, 4, 4},
        {"fp16", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, 2, 2},
        {"bf16", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, 2, 2},
        {"fp8_e4m3", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, 1, 2},
        {"int8", CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, 1, 4},
    };
    const int sizes[][3] = {{8192, 8192, 8192}, {4096, 4096, 4096}, {1024, 1024, 1024}};
    printf("  \"gemm_tflops\": {\n");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        printf("    \"%s\": {", cases[i].name);
        for (size_t s = 0; s < 3; s++) {
            double tf = gemm_tflops(lt, cases[i], sizes[s][0], sizes[s][1], sizes[s][2], ws, ws_bytes);
            if (tf < 0) printf("\"%d\": null", sizes[s][0]); else printf("\"%d\": %.1f", sizes[s][0], tf);
            printf("%s", s + 1 < 3 ? ", " : "");
        }
        printf("}%s\n", i + 1 < sizeof(cases) / sizeof(cases[0]) ? "," : "");
    }
    printf("  }\n}\n");
    CKL(cublasLtDestroy(lt));
    CK(cudaFree(ws));
    return 0;
}
