// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fp16 tensor-core paged flash-prefill kernels at head_dim 128 (Qwen3: 32 query heads, 8 KV heads) agree with the
//! portable fp32 `paged_flash_prefill_hd256` (head_dim up to 256) and with an f64 host oracle over every cache tier, with
//! ragged chunk lengths, non-zero starts and scattered blocks; a round in the decode regime keeps the portable kernel.
//! Two forms serve the width: the per-call kernel every tensor-core part has (`flash_prefill_f16_hd128*`), and on a part
//! with warpgroup tensor cores the staged warpgroup kernels head_dim 256 runs (`flash_prefill_f16_hd128_wgs*`), the same
//! source compiled at this width.
//!
//! Swedish Embedded AB implements long-context inference throughput for its clients. If your team needs expertise in
//! proving that a reduced-precision attention kernel is faster AND numerically faithful to the fp32 path it replaces, you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! The bar is head_dim 256's ([`REL_TOL`]): an absolute error relative to the output RMS (fp16 rounding of Q, K, V and
//! the softmax weights, fp32 accumulation). Skipped without a CUDA device.

const N_HEADS: u32 = 32;
const N_KV: u32 = 8;
const HD: u32 = 128;

include!("flash_prefill_harness.rs");

const TIERS: [Kv; 4] = [Kv::F32, Kv::Bf16, Kv::Int8, Kv::Fp8];

/// The per-call kernel reads every tier in place and agrees with the portable kernel of the tier and the oracle.
#[test]
fn the_head_dim_128_kernel_reads_every_cache_tier() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 64, start: 31, block_size: 16, scatter: true },
        Case { n: 33, start: 5, block_size: 16, scatter: false },
    ];
    for kv in TIERS {
        for (i, c) in cases.iter().enumerate() {
            let desc = format!("{kv:?} case {i} (n={} start={} block={} scatter={})", c.n, c.start, c.block_size, c.scatter);
            let p = Problem::new(Case { ..*c }, 0xBEEF + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let want = d.base.run(&gpu, &d.portable(&gpu, &seen), &seen);
            let step = d.native(&gpu, &seen).unwrap_or_else(|| panic!("{desc}: head_dim 128 takes the tensor-core kernel"));
            gpu.write(&d.base.ctx, &vec![0x7fc0_0000u32; (seen.c.n * N_HEADS * HD) as usize]);
            let got = d.base.run(&gpu, &step, &seen);
            let scale = rms(&want);
            let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(got.iter().all(|x| x.is_finite()), "{desc}: non-finite output");
            eprintln!("cuda_flash_prefill_hd128: {desc}: max|tc - portable| = {maxdiff:e} ({:.2e} of rms)", maxdiff / scale);
            assert!(maxdiff <= REL_TOL * scale, "{desc}: differs from the portable kernel by {maxdiff:e}");
            let mut rng = Rng(7 + i as u64);
            for _ in 0..ORACLE_SAMPLES {
                let (row, head) = ((rng.next() % seen.c.n as u64) as usize, (rng.next() % N_HEADS as u64) as usize);
                let truth = seen.oracle(row, head);
                let base = (row * N_HEADS as usize + head) * HD as usize;
                let err = (0..HD as usize).map(|e| (got[base + e] as f64 - truth[e]).abs()).fold(0f64, f64::max);
                assert!(err <= (REL_TOL * scale) as f64, "{desc}: row {row} head {head} is {err:e} from the f64 oracle");
            }
        }
    }
}

#[test]
fn a_round_in_the_decode_regime_keeps_the_portable_kernel_at_head_dim_128() {
    let Some(gpu) = device() else { return };
    for kv in TIERS {
        let p = Problem::new(Case { n: 32, start: 100, block_size: 4096, scatter: false }, 1);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        assert!(d.native(&gpu, &seen).is_none(), "{kv:?}: a 32-row round must not take the fp16 kernel");
        assert!(d.native_wg(&gpu, &seen).is_none(), "{kv:?}: a 32-row round must not take the warpgroup kernel");
    }
}

/// The reason the kernel exists: the tensor-core kernel is materially faster than the portable one at a prefill shape.
#[test]
fn the_head_dim_128_kernel_is_materially_faster_than_the_portable_kernel() {
    let Some(gpu) = device() else { return };
    let (n, start) = (256u32, 1792u32);
    let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 21);
    let d = Device::upload(&gpu, &p);
    let (portable, native) = (d.portable(&gpu, &p), d.native(&gpu, &p).expect("native step"));
    d.run(&gpu, &portable, &p); // compile outside the timed region
    d.run(&gpu, &native, &p);
    let (t_ref, t_tc) = d.best_ms_pair(&gpu, &portable, &native, SPEEDUP_FLOOR);
    let speedup = t_ref / t_tc;
    eprintln!("cuda_flash_prefill_hd128: n={n} start={start}: portable {t_ref:.3} ms, tensor-core {t_tc:.3} ms, {speedup:.1}x");
    assert!(speedup >= SPEEDUP_FLOOR, "only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
}

/// The staged warpgroup kernels at head_dim 128 read every tier (the widen pass turns each into the same fp16 records) and
/// agree with the portable kernel and the oracle over [`WARPGROUP_CASES`]; this width has only the staged form.
#[test]
fn the_warpgroup_prefill_at_head_dim_128_reads_every_cache_tier() {
    let Some(gpu) = device() else { return };
    gpu_core::provider::cuda::set_flash_prefill_wg(true);
    warpgroup_agrees(&gpu, &TIERS.map(|kv| (true, kv)), &WARPGROUP_CASES);
}

/// Every round of this width is staged, whatever its rows and context: a widen pass, the attention and (below 2048 rows)
/// the merge of the key splits.
#[test]
fn every_head_dim_128_round_is_staged() {
    let Some(gpu) = device() else { return };
    gpu_core::provider::cuda::set_flash_prefill_staged_min_rows(gpu_core::provider::cuda::FLASH_PREFILL_STAGED_MIN_ROWS);
    for kv in TIERS {
        let p = Problem::new(Case { n: 128, start: 300, block_size: 4096, scatter: false }, 9);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        let steps = d.native_wg(&gpu, &seen).expect("a 128-row round takes the warpgroup kernel");
        assert_eq!(steps.len(), 3, "{kv:?} 128 rows behind 300 keys: a widen pass, the attention and the merge");
    }
}

/// Large scores (see the head_dim 256 test of the same name): the staged form stays within 3e-3 relative L2 of the fp32
/// portable kernel at score standard deviations up to 16.
#[test]
fn the_staged_form_stays_close_to_fp32_at_large_score_magnitudes_at_head_dim_128() {
    let Some(gpu) = device() else { return };
    for kv in TIERS {
        for mag in [1.0f32, 2.0, 4.0] {
            let mut p = Problem::new(Case { n: 128, start: 1920, block_size: 4096, scatter: false }, 5);
            p.q.iter_mut().for_each(|x| *x *= mag);
            p.k_pool.iter_mut().for_each(|x| *x *= mag);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let want = d.base.run(&gpu, &d.portable(&gpu, &seen), &seen);
            let rel = staged_rel_l2(&gpu, &d, &seen, &want);
            eprintln!("cuda_flash_prefill_hd128: {kv:?} score scale {}x staged: rel-L2 to the portable kernel {rel:.2e}", mag * mag);
            assert!(rel <= 3e-3, "{kv:?} at score scale {}x: rel-L2 {rel:.2e} from the fp32 kernel (bar 3e-3)", mag * mag);
        }
    }
}

/// Back-to-back staged rounds (knowledge 410), every tier:
///
/// ```text
/// BRAIN_DEVICE=gpu3 BRAIN_BACKEND=cuda make test/one P=brain-gpu-core T=it ARGS="--ignored --nocapture hd128::staged_back_to_back"
/// ```
#[test]
#[ignore = "a stress run"]
fn staged_back_to_back_stress_at_head_dim_128() {
    let Some(gpu) = device() else { return };
    staged_rounds_repeat(&gpu, &TIERS);
}

/// The Qwen3 prefill shapes: a whole 8K prompt in one round and 512-row chunks into long contexts, for `ncu -k regex:wgs -c 1`
/// on this binary. An FP8 cache stages the same records as an int8 one (only its widen pass differs), so it is timed at the
/// short shapes alone, where encoding the test cache on the host stays quick:
///
/// ```text
/// BRAIN_DEVICE=gpu3 BRAIN_BACKEND=cuda make test/one P=brain-gpu-core T=it ARGS="--ignored --nocapture hd128::the_qwen3_prefill_rates"
/// ```
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_qwen3_prefill_rates() {
    for kv in [Kv::F32, Kv::Bf16, Kv::Int8] {
        prefill_rates(kv, &[(8192u32, 0u32), (4096, 0), (4096, 12288), (512, 3584), (512, 15872), (512, 32256)]);
    }
    prefill_rates(Kv::Fp8, &[(512u32, 3584u32)]);
}
