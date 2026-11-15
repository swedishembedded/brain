// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fp16 tensor-core paged flash-prefill kernel must agree with the portable
//! fp32 `paged_flash_prefill_hd256` it replaces on real Qwen3.8-27B attention
//! shapes (24 query heads, 4 KV heads, head_dim 256, flat and paged caches,
//! ragged chunk lengths and non-zero starts), and must be much faster.
//!
//! Swedish Embedded AB implements long-context inference throughput for its
//! clients. If your team needs expertise in proving that a reduced-precision
//! attention kernel is faster AND numerically faithful to the fp32 path it
//! replaces, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What "agrees" means here
//!
//! The portable kernel is fp32 end to end; the tensor-core kernel rounds Q, K,
//! V and the softmax weights to fp16 (fp32 accumulation). The two therefore
//! differ by fp16 rounding, ~5e-4 relative per operand, averaged down by the
//! sums that make up an attention output. The bar is stated relative to the
//! output's own RMS ([`REL_TOL`]), so it keeps meaning at another activation
//! scale, and an f64 host oracle over a sample of rows checks both kernels
//! against the truth so a bug the two would share cannot hide.
//!
//! The same kernel serves a bf16 and an int8 KV cache (planes widened while they are
//! staged); those tiers are checked against their own portable kernel and against the
//! f64 oracle over the values the cache actually holds.
//!
//! Skipped without a device, or below the kernel's capability floor.

const N_HEADS: u32 = 24;
const N_KV: u32 = 4;
const HD: u32 = 256;

include!("flash_prefill_harness.rs");

#[test]
fn the_tensor_core_flash_prefill_agrees_with_the_portable_kernel_and_the_oracle() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 256, start: 256, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 33, start: 513, block_size: 4096, scatter: false },
        Case { n: 130, start: 0, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 64, start: 31, block_size: 16, scatter: true },
    ];
    let mut worst = 0f32;
    for (i, c) in cases.into_iter().enumerate() {
        let desc = format!("case {i} (n={} start={} block={} scatter={})", c.n, c.start, c.block_size, c.scatter);
        let p = Problem::new(c, 0xF1A5 + i as u64);
        let d = Device::upload(&gpu, &p);
        let want = d.run(&gpu, &d.portable(&gpu, &p), &p);
        let step = d.native(&gpu, &p).expect("the production helper must take a head_dim-256 request on this device");
        // The output must not depend on what the buffer held before.
        gpu.write(&d.ctx, &vec![0x7fc0_0000u32; (p.c.n * N_HEADS * HD) as usize]);
        let got = d.run(&gpu, &step, &p);

        let scale = rms(&want);
        let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(got.iter().all(|x| x.is_finite()), "{desc}: non-finite output");
        eprintln!("cuda_flash_prefill: {desc}: max|tc - fp32| = {maxdiff:e} ({:.2e} of rms {scale:e})", maxdiff / scale);
        assert!(maxdiff <= REL_TOL * scale, "{desc}: differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
        worst = worst.max(maxdiff / scale);

        let mut rng = Rng(9 + i as u64);
        for _ in 0..ORACLE_SAMPLES {
            let (row, head) = ((rng.next() % p.c.n as u64) as usize, (rng.next() % N_HEADS as u64) as usize);
            let truth = p.oracle(row, head);
            let base = (row * N_HEADS as usize + head) * HD as usize;
            let err = (0..HD as usize).map(|d| (got[base + d] as f64 - truth[d]).abs()).fold(0f64, f64::max);
            assert!(err <= (REL_TOL * scale) as f64, "{desc}: row {row} head {head} is {err:e} from the f64 oracle");
        }
    }
    eprintln!("cuda_flash_prefill: worst relative difference {worst:.2e} (bar {REL_TOL:.0e})");
}

/// A verify or ragged-tail round is not a prefill: it stays on the fp32 kernel so
/// speculative decoding keeps computing what the token-by-token tape computes.
#[test]
fn a_round_in_the_decode_regime_keeps_the_fp32_kernel() {
    let Some(gpu) = device() else { return };
    for n in [1u32, 8, 32] {
        let p = Problem::new(Case { n, start: 100, block_size: 4096, scatter: false }, 5);
        let d = Device::upload(&gpu, &p);
        assert!(d.native(&gpu, &p).is_none(), "a {n}-row round must not take the fp16 kernel");
    }
    let p = Problem::new(Case { n: 33, start: 100, block_size: 4096, scatter: false }, 5);
    let d = Device::upload(&gpu, &p);
    assert!(d.native(&gpu, &p).is_some(), "33 rows is past the decode regime and takes the kernel");
}

#[test]
fn a_head_width_the_kernel_was_not_written_for_is_declined() {
    let Some(gpu) = device() else { return };
    let p = Problem::new(Case { n: 64, start: 0, block_size: 64, scatter: false }, 3);
    let d = Device::upload(&gpu, &p);
    assert!(gpu_core::provider::cuda::paged_flash_prefill_step(&gpu, 96, &d.bufs(), &Device::params(&p), 24).is_none());
}

#[test]
fn the_tensor_core_flash_prefill_is_materially_faster_than_the_portable_kernel() {
    let Some(gpu) = device() else { return };
    // The deep shape is the gate. The shallow one (a first round, ~40 us of work
    // for the kernel) is printed but not asserted: at that size the event timer
    // and launch overhead are most of what is being measured.
    for (n, start, gate) in [(256u32, 1792u32, true), (256, 0, false)] {
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 21);
        let d = Device::upload(&gpu, &p);
        let portable = d.portable(&gpu, &p);
        let native = d.native(&gpu, &p).expect("native step");
        d.run(&gpu, &portable, &p); // compile outside the timed region
        d.run(&gpu, &native, &p);
        let (t_ref, t_tc) = d.best_ms_pair(&gpu, &portable, &native, if gate { SPEEDUP_FLOOR } else { 0.0 });
        // causal attention flops: 4 * sum_i len_i * hd * heads
        let flops: f64 = (0..n).map(|i| (start + i + 1) as f64).sum::<f64>() * 4.0 * HD as f64 * N_HEADS as f64;
        let speedup = t_ref / t_tc;
        eprintln!(
            "cuda_flash_prefill: n={n} start={start}: portable {t_ref:.3} ms ({:.1} TFLOPS), tensor-core {t_tc:.3} ms ({:.1} TFLOPS), {speedup:.1}x",
            flops / (t_ref * 1e-3) / 1e12,
            flops / (t_tc * 1e-3) / 1e12
        );
        assert!(!gate || speedup >= SPEEDUP_FLOOR, "n={n} start={start}: only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
    }
}

/// A compact cache is read in place by the tensor-core kernel: it agrees with the
/// portable kernel of its tier and with an f64 oracle over the values the cache holds.
#[test]
fn the_tensor_core_flash_prefill_reads_a_bf16_an_int8_and_an_fp8_cache() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 64, start: 31, block_size: 16, scatter: true },
    ];
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        for (i, c) in cases.iter().enumerate() {
            let desc = format!("{kv:?} case {i} (n={} start={} block={} scatter={})", c.n, c.start, c.block_size, c.scatter);
            let p = Problem::new(Case { ..*c }, 0xC0DE + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let want = d.base.run(&gpu, &d.portable(&gpu, &seen), &seen);
            let step = d.native(&gpu, &seen).expect("the production helper must take a head_dim-256 request on this device");
            gpu.write(&d.base.ctx, &vec![0x7fc0_0000u32; (seen.c.n * N_HEADS * HD) as usize]);
            let got = d.base.run(&gpu, &step, &seen);

            let scale = rms(&want);
            let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(got.iter().all(|x| x.is_finite()), "{desc}: non-finite output");
            eprintln!("cuda_flash_prefill: {desc}: max|tc - portable| = {maxdiff:e} ({:.2e} of rms {scale:e})", maxdiff / scale);
            assert!(maxdiff <= REL_TOL * scale, "{desc}: differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
            let mut rng = Rng(77 + i as u64);
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

/// Version 2 of the compact-cache kernel (one warp per 16 rows over the whole head, two blocks per SM) does the
/// arithmetic of the original with a different key tile (so a different running-max schedule and fp16 rounding of P),
/// so the two agree to rounding noise; this holds it to 2e-3 of the output scale (the bar against the portable fp32
/// kernel is 1e-2) on ragged and scattered shapes, and the NaN-filled output proves every row is written.
/// It flips the process-wide version switch; the other tests in this file are valid under either kernel.
#[test]
fn version_2_matches_the_original_kernel_on_a_bf16_an_int8_and_an_fp8_cache() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 33, start: 513, block_size: 16, scatter: true },
    ];
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        for (i, c) in cases.iter().enumerate() {
            let p = Problem::new(Case { ..*c }, 0xFA57 + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let mut out = Vec::new();
            for v2 in [false, true] {
                gpu_core::provider::cuda::set_flash_prefill_v2(v2);
                let step = d.native(&gpu, &seen).expect("the production helper must take a head_dim-256 request on this device");
                gpu.write(&d.base.ctx, &vec![0x7fc0_0000u32; (seen.c.n * N_HEADS * HD) as usize]);
                out.push(d.base.run(&gpu, &step, &seen));
            }
            gpu_core::provider::cuda::set_flash_prefill_v2(true);
            assert!(out[1].iter().all(|x| x.is_finite()), "{kv:?} case {i}: version 2 left a row unwritten");
            let maxdiff = out[0].iter().zip(&out[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let bar = 2e-3 * rms(&out[0]);
            assert!(maxdiff <= bar, "{kv:?} case {i} (n={} start={}): version 2 differs from version 1 by {maxdiff:e} (bar {bar:e})", c.n, c.start);
        }
    }
}

/// The warpgroup (`wgmma`) kernel reads an int8 and an FP8 cache in place (and an fp32 and a bf16 one through its staged
/// records), agrees with the portable kernel and the f64 oracle, and writes every row ([`WARPGROUP_CASES`]).
#[test]
fn the_warpgroup_prefill_reads_an_fp32_a_bf16_an_int8_and_an_fp8_cache() {
    let Some(gpu) = device() else { return };
    gpu_core::provider::cuda::set_flash_prefill_wg(true);
    // A bf16 or fp32 cache has only the staged form (`short_rounds_are_staged_by_their_tier_and_context`).
    let tiers = [(false, Kv::Int8), (false, Kv::Fp8), (true, Kv::Int8), (true, Kv::Fp8), (true, Kv::Bf16), (true, Kv::F32)];
    warpgroup_agrees(&gpu, &tiers, &WARPGROUP_CASES);
}

/// The staged rounds of every tier submitted back to back many times without a wait between them (knowledge 410: a TMA
/// fault showed only under back-to-back launches) give the output of the first round, bit for bit, every time. Run it clean
/// before a staged kernel lands, then `dmesg | grep -i xid` stays empty (`#[ignore]`: a stress run).
///
/// ```text
/// BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda make test/one P=brain-gpu-core T=cuda_flash_prefill ARGS="--ignored --nocapture staged_back_to_back"
/// ```
#[test]
#[ignore = "a stress run"]
fn staged_back_to_back_stress() {
    let Some(gpu) = device() else { return };
    staged_rounds_repeat(&gpu, &[Kv::F32, Kv::Bf16, Kv::Int8, Kv::Fp8]);
}

/// Which short rounds are staged at the default settings: an fp32 or bf16 round always (it has no form that widens in every block, and
/// version 2 is slower than a widen pass at any row count), an 8-bit round from `FLASH_PREFILL_STAGED_MIN_KEYS` keys (a
/// staged round is a widen pass, the attention and the merge of its key splits; an unstaged one the attention and the merge).
#[test]
fn short_rounds_are_staged_by_their_tier_and_context() {
    let Some(gpu) = device() else { return };
    gpu_core::provider::cuda::set_flash_prefill_staged_min_rows(gpu_core::provider::cuda::FLASH_PREFILL_STAGED_MIN_ROWS);
    for (kv, start, staged) in [(Kv::F32, 300, true), (Kv::Bf16, 300, true), (Kv::Bf16, 4000, true), (Kv::Int8, 300, false), (Kv::Int8, 4000, true), (Kv::Fp8, 300, false), (Kv::Fp8, 4000, true)] {
        let p = Problem::new(Case { n: 128, start, block_size: 4096, scatter: false }, 9);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        let steps = d.native_wg(&gpu, &seen).expect("a 128-row round takes the warpgroup kernel");
        assert_eq!(steps.len(), if staged { 3 } else { 2 }, "{kv:?} 128 rows behind {start} keys: staged must be {staged}");
    }
}

/// The compact-tier kernels keep the decode-regime rule of the fp32 one.
#[test]
fn a_compact_cache_in_the_decode_regime_keeps_its_portable_kernel() {
    let Some(gpu) = device() else { return };
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        let p = Problem::new(Case { n: 8, start: 100, block_size: 4096, scatter: false }, 5);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        assert!(d.native(&gpu, &seen).is_none(), "{kv:?}: an 8-row round must not take the fp16 kernel");
    }
}

/// The reason the kernel exists: an int8 cache prefill on the portable kernel is the
/// slowest part of a served prefill.
#[test]
fn an_8_bit_cache_prefill_is_materially_faster_than_the_portable_kernel() {
    let Some(gpu) = device() else { return };
    for kv in [Kv::Int8, Kv::Fp8] {
        let (n, start) = (256u32, 1792u32);
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 21);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        let portable = d.portable(&gpu, &seen);
        let native = d.native(&gpu, &seen).expect("native step");
        d.base.run(&gpu, &portable, &seen);
        d.base.run(&gpu, &native, &seen);
        let (t_ref, t_tc) = d.base.best_ms_pair(&gpu, &portable, &native, SPEEDUP_FLOOR);
        let speedup = t_ref / t_tc;
        eprintln!("cuda_flash_prefill: {kv:?} cache n={n} start={start}: portable {t_ref:.3} ms, tensor-core {t_tc:.3} ms, {speedup:.1}x");
        assert!(speedup >= SPEEDUP_FLOOR, "only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
    }
}

/// The serving shape of the 27B's agentic prefill (a 512-row chunk behind 3584 cached tokens, int8 cache): 192
/// blocks, i.e. more than one wave on 132 SMs, which the smaller cases above do not exercise. Reports the time of
/// the original kernel and of version 2 and how far their outputs differ; it asserts nothing about speed. It flips the
/// process-wide version switch, so run it alone (`--test-threads 1`, as `runs/b27/flash-bench.sh` does).
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_int8_cache_prefill_rates_at_the_serving_chunk() {
    let Some(gpu) = device() else { return };
    let (n, start) = (512u32, 3584u32);
    let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 23);
    let (seen, stored) = p.as_stored(Kv::Int8);
    let d = TierDevice::upload(&gpu, &seen, Kv::Int8, &stored);
    let mut out = Vec::new();
    for (v2, name) in [(false, "flash_prefill_f16_hd256_kv8"), (true, "flash_prefill_f16_hd256_v2_kv8")] {
        gpu_core::provider::cuda::set_flash_prefill_v2(v2);
        let native = d.native(&gpu, &seen).expect("native step");
        let ctx = d.base.run(&gpu, &native, &seen);
        assert!(gpu.set_kernel_timing(true));
        let ms = (0..TRIALS).map(|_| d.base.trial_ms(&gpu, &native, name)).fold(f64::MAX, f64::min);
        gpu.set_kernel_timing(false);
        eprintln!("cuda_flash_prefill: serving chunk int8 n={n} start={start}: {name} {ms:.3} ms");
        out.push(ctx);
    }
    gpu_core::provider::cuda::set_flash_prefill_v2(true);
    let diff = out[0].iter().zip(&out[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("cuda_flash_prefill: version 1 vs version 2 max |diff| {diff:.2e} on outputs of rms {:.2e}", rms(&out[0]));
}

/// Scores are exponentiated, so an error in `q . k` becomes a relative error in the attention weight: what matters is
/// the absolute error of the score, which grows with the operands' magnitude, and the unit-scale cases above hide that.
/// A real model's scaled scores reach several tens, so this scales queries and keys until the scores have a standard
/// deviation of `mag * mag` (up to 16) and holds both kernel versions and the staged warpgroup form to the portable fp32 kernel by relative L2 over
/// the whole output. Measured: 3e-4 at unit scale growing to 1.3e-3 at mag = 4; the bar is 3e-3. (Through the 27B's logits the
/// same kernels differ from fp32 by 0.1 - that is the W8A8 model's own sensitivity, `qwen35_gguf_logit_check` reports it
/// next to its floor - so the kernel is gated here, where the cause of an error can be told.)
#[test]
fn both_flash_versions_stay_close_to_fp32_at_large_score_magnitudes() {
    let Some(gpu) = device() else { return };
    for kv in [Kv::F32, Kv::Bf16, Kv::Int8, Kv::Fp8] {
        for mag in [1.0f32, 2.0, 4.0] {
            let mut p = Problem::new(Case { n: 128, start: 1920, block_size: 4096, scatter: false }, 5);
            p.q.iter_mut().for_each(|x| *x *= mag);
            p.k_pool.iter_mut().for_each(|x| *x *= mag);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let want = d.base.run(&gpu, &d.portable(&gpu, &seen), &seen);
            let norm = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
            for v2 in [false, true] {
                gpu_core::provider::cuda::set_flash_prefill_v2(v2);
                let step = d.native(&gpu, &seen).expect("the production helper must take a head_dim-256 request on this device");
                let got = d.base.run(&gpu, &step, &seen);
                gpu_core::provider::cuda::set_flash_prefill_v2(true);
                let rel = (want.iter().zip(&got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / norm).sqrt();
                eprintln!("cuda_flash_prefill: {kv:?} score scale {}x v2={v2}: rel-L2 to the portable kernel {rel:.2e}", mag * mag);
                assert!(rel <= 3e-3, "{kv:?} at score scale {}x, v2={v2}: rel-L2 {rel:.2e} from the fp32 kernel (bar 3e-3)", mag * mag);
            }
            // The staged warpgroup form, every tier.
            let rel = staged_rel_l2(&gpu, &d, &seen, &want);
            eprintln!("cuda_flash_prefill: {kv:?} score scale {}x staged: rel-L2 to the portable kernel {rel:.2e}", mag * mag);
            assert!(rel <= 3e-3, "{kv:?} at score scale {}x, staged: rel-L2 {rel:.2e} from the fp32 kernel (bar 3e-3)", mag * mag);
        }
    }
}

/// The int8-cache prefill kernels at the serving chunk at growing context, for `ncu -k regex:flash_prefill -c 1`:
/// `make test/one P=brain-gpu-core T=cuda_flash_prefill ARGS="--ignored --nocapture contexts"`.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_int8_cache_prefill_rates_at_growing_contexts() {
    int8_prefill_rates(&[(512u32, 512u32), (512, 3584), (512, 7680), (512, 15872), (512, 31232)]);
}

/// The same at the 4096-row round the server runs a long prompt in.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_int8_cache_prefill_rates_at_the_long_prompt_round() {
    int8_prefill_rates(&[(4096u32, 0u32), (4096, 4096), (4096, 12288), (4096, 20480), (4096, 28672)]);
}

/// The prefill of every compact tier at the short rounds a busy server runs (a chunk tail or a mixed round's prompt rows):
/// version 2, the form that widens in every block (8-bit tiers) and the staged form, which a bf16 round takes at every row count.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_compact_cache_prefill_rates_at_short_rounds() {
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        prefill_rates(kv, &[(64u32, 448u32), (128, 384), (64, 1984), (128, 1920), (64, 8000), (128, 8000), (128, 32000), (255, 8000)]);
    }
}

/// A bf16 cache at the 2048-row chunks a 64-sequence 8K-prompt run prefills in, at the four chunk starts of an 8K prompt.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_bf16_cache_prefill_rates_at_the_8k_prompt_chunks() {
    prefill_rates(Kv::Bf16, &[(4096u32, 0u32), (3968, 3924), (2048, 0), (2048, 6144)]);
}

/// A bf16 cache at the 4096-row round a long prompt is served in, over contexts of 4K to 64K: the shapes that decide a
/// prefill-dominated run. The sustained figure is the one to read (the card holds a lower clock under load than a best trial).
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_bf16_cache_prefill_rates_at_the_long_prompt_round() {
    prefill_rates(Kv::Bf16, &[(4096u32, 0u32), (4096, 4096), (4096, 12288), (4096, 28672), (4096, 61440)]);
}

/// One 4096-row round behind 28672 cached tokens on a bf16 cache, for `ncu -k regex:wgs -c 1`.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_bf16_cache_prefill_rate_at_a_32k_context() {
    prefill_rates(Kv::Bf16, &[(4096u32, 28672u32)]);
}

fn int8_prefill_rates(cases: &[(u32, u32)]) {
    prefill_rates(Kv::Int8, cases);
}

