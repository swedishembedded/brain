// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dropping a Qwen3.8 model returns every CUDA object it took.
//!
//! Swedish Embedded AB implements long-running model-serving systems for its
//! clients, where a model that is loaded, replaced and loaded again must not
//! ratchet the card's memory upward. If your team needs expertise in keeping
//! GPU residency honest across model swaps, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Two gates, one claim:
//!
//! - a synthetic decoder, built, decoded with and dropped in a loop on a
//!   device handle that outlives it, in every drop order a model uses; and
//! - the real `Qwen3.8-27B` Q8_0 GGUF through `Qwen35GgufResident` - the whole
//!   serving path - which self-skips unless `BRAIN_QWEN35_GGUF` names it.
//!
//! Both read `backend_cuda::live_resources` (exact, by kind) and `cuMemGetInfo`
//! (the driver's own opinion, within a tolerance because the card is shared).
//! They run on the CUDA backend only and skip elsewhere.

use backend_cuda::exec::Context;
use backend_cuda::{live_resources, LiveResources};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use std::sync::{Mutex, MutexGuard};

/// The counters are process-global, so the two gates must not overlap.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// How far `cuMemGetInfo` free memory may fall across a workload that
/// returned everything.
const FREE_TOLERANCE: u64 = 96 << 20;

struct Baseline {
    live: LiveResources,
    /// Device-wide free memory, the fallback when the driver will not say what
    /// this process holds.
    free: u64,
    /// What the driver attributes to this process, in MiB, when it will say.
    own_mib: Option<u64>,
}

fn baseline(probe: &Context) -> Baseline {
    Baseline { live: live_resources(), free: probe.mem_info().expect("cuMemGetInfo").0, own_mib: brain_testutil::own_gpu_memory_mib() }
}

/// The exact counters, then the driver's opinion - about THIS process when it
/// can say (`brain_testutil::own_gpu_memory_mib`). The card is shared, and
/// another process loading a model moves device-wide free memory by gigabytes,
/// which a gate on it reports as a leak that is not ours; only where the
/// per-process figure is unavailable does this fall back to free memory.
fn assert_returned(probe: &Context, base: &Baseline, what: &str) {
    assert_eq!(live_resources(), base.live, "{what}: live CUDA objects did not return to baseline");
    let tolerance_mib = FREE_TOLERANCE >> 20;
    if let Some(own_before) = base.own_mib {
        let mut own = own_before;
        let mut asked = true;
        for _ in 0..50 {
            match brain_testutil::own_gpu_memory_mib() {
                Some(now) => own = now,
                None => {
                    asked = false;
                    break;
                }
            }
            if own <= own_before + tolerance_mib {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        if asked {
            panic!("{what}: the driver attributes {own} MiB to this process against {own_before} MiB before - the model's memory was not returned");
        }
    }
    let mut free = 0;
    for _ in 0..50 {
        free = probe.mem_info().expect("cuMemGetInfo").0;
        if free + FREE_TOLERANCE >= base.free {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!(
        "{what}: cuMemGetInfo free memory fell from {} MiB to {} MiB - {} MiB were not returned",
        base.free >> 20,
        free >> 20,
        (base.free - free) >> 20
    );
}

/// One model's life on `gpu`: build an INT8 decoder on a shared handle, run a
/// prefill and enough decode steps for graphs to form, drop it.
fn model_lifetime(gpu: &Gpu) {
    let cfg = Qwen35Config::tiny_i8();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu.share(), cfg.clone(), 1, t, &init, &TierPolicy::uniform(Dtype::I8));
    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    assert!(m.logits_all(&tokens).iter().all(|x| x.is_finite()));
    m.reset_decode_cache();
    for &tok in &tokens {
        assert!(m.step(tok).iter().all(|x| x.is_finite()));
    }
    // Chunked prefill rounds in the scratch-arena regime: the backend holds freed
    // blocks for reuse while the arena is open and must give every one back when
    // the round releases it.
    m.set_chunk_arena_min_rows(1);
    m.reset_decode_cache();
    assert!(m.prefill_chunked(&tokens, 4).iter().all(|x| x.is_finite()));
    drop(m);
}

#[test]
fn a_synthetic_decoder_returns_everything_while_its_device_handle_lives_on() {
    let _s = serial();
    let all_before = live_resources();
    let Ok(probe) = Context::open(0) else {
        brain_testutil::skip_unavailable("no usable CUDA device");
        return;
    };
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    model_lifetime(&gpu); // warm-up: driver-lazy state and reusable staging
    gpu.poll_wait();
    let base = baseline(&probe);
    for _ in 0..4 {
        model_lifetime(&gpu);
        gpu.poll_wait();
    }
    assert_returned(&probe, &base, "synthetic decoder on a surviving handle");

    // And when the handle goes too, nothing at all is left: modules unloaded,
    // streams destroyed, the context released.
    drop(gpu);
    drop(probe);
    assert_eq!(live_resources(), all_before, "dropping the device handle left CUDA objects behind");
}

/// A paged engine in every KV tier, built, prefilled, decoded and dropped. The
/// compact tiers own more device objects per plane than the `f32` one (packed
/// words, row scales, the int8 append ceiling), so each must be returned by
/// the plane that took it - exactly, by kind: allocations, bytes, streams,
/// modules, retains. The driver's own free-memory figure is deliberately not
/// consulted here: at this size it is dwarfed by what another process on a
/// shared card allocates between two readings, and the exact counters are the
/// stronger claim.
#[test]
fn a_paged_engine_in_every_kv_tier_returns_every_device_object_it_took() {
    use model::kv_tier::KvTier;
    use model::paged::BlockTable;
    use qwen35::serve::Engine;

    let _s = serial();
    let Ok(_probe) = Context::open(0) else {
        brain_testutil::skip_unavailable("no usable CUDA device");
        return;
    };
    // The probe holds a stream and a primary-context retain of its own.
    let baseline = live_resources();
    let cfg = Qwen35Config::tiny_i8();
    let init = qwen35::init::init_weights(&cfg, 7);
    let lifetime = |tier: KvTier| {
        let mut engine = Engine::from_map_kv(cfg.clone(), &init, 64, 2, tier);
        let mut table = BlockTable::new();
        let hidden = model::serve::PagedDecoder::prefill(&mut engine, &mut table, &[3, 8, 1, 4, 1, 5, 9, 2, 6]);
        assert!(hidden.iter().all(|x| x.is_finite()), "{tier}: prefill hidden state is not finite");
        let next = engine.forward_batched_greedy(&mut [&mut table], &[7]);
        assert_eq!(next.len(), 1);
        engine.release_table(&mut table);
    };
    for tier in KvTier::ALL {
        for round in 0..3 {
            lifetime(tier);
            assert_eq!(live_resources(), baseline, "{tier}, round {round}: dropping the engine and its device handle left CUDA objects behind");
        }
    }
}

/// The whole serving path on the real checkpoint: plan, load, prefill, decode,
/// drop - twice, so a second load on a card that did not get its memory back
/// would also run out.
#[test]
fn dropping_a_real_gguf_instance_returns_all_device_memory() {
    use qwen35::int8_gguf_resident::{Qwen35GgufResident, GGUF_ENV};
    use residency::multi::MultiDeviceResidentModel;
    use residency::{Device, ResidentModel};

    let _s = serial();
    let Ok(path) = std::env::var(GGUF_ENV) else {
        brain_testutil::skip(&format!("{GGUF_ENV} unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)"));
        return;
    };
    let Ok(probe) = Context::open(0) else {
        brain_testutil::skip_unavailable("no usable CUDA device");
        return;
    };
    const RESERVE: u64 = 2 << 30;
    let devices: Vec<(Device, u64)> = gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM");
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, 512, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();

    let base = baseline(&probe);
    for round in 0..2 {
        let inst = r.activate_owned(&placed).expect("activate the real checkpoint");
        // What the model holds, by this process's own exact counter: the card's
        // free memory moves with every other tenant's allocations and cannot say
        // whether THIS model occupied anything.
        let held = live_resources().device_bytes.saturating_sub(base.live.device_bytes);
        assert!(held > 1 << 30, "round {round}: the model did not occupy any device memory, so this proves nothing");
        println!("round {round}: model holds {} MiB of device memory", held >> 20);
        inst.prefill_timed(&[1, 2, 3, 4, 5, 6, 7, 8]).expect("prefill");
        for pos in 8..16u32 {
            inst.decode_batch_at(&[pos + 1], &[pos]).expect("decode");
        }
        drop(inst);
        assert_returned(&probe, &base, &format!("round {round}: real GGUF instance"));
    }
}

/// The same on the real checkpoint with a compact KV cache and a batch: the
/// planes' packed words, row scales and append ceilings, and the fused decode's
/// per-layer partial-state scratch, all returned - exactly, by the process's own
/// counters.
#[test]
fn dropping_a_real_gguf_instance_with_a_compact_kv_cache_returns_every_device_object() {
    use model::kv_tier::KvTier;
    use qwen35::int8_gguf_resident::{Qwen35GgufResident, GGUF_ENV};
    use residency::multi::MultiDeviceResidentModel;
    use residency::{Device, ResidentModel};

    let _s = serial();
    let Ok(path) = std::env::var(GGUF_ENV) else {
        brain_testutil::skip(&format!("{GGUF_ENV} unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)"));
        return;
    };
    let Ok(_probe) = Context::open(0) else {
        brain_testutil::skip_unavailable("no usable CUDA device");
        return;
    };
    const RESERVE: u64 = 2 << 30;
    let devices: Vec<(Device, u64)> = gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM");
        return;
    }
    let baseline = live_resources();
    for tier in [KvTier::Bf16, KvTier::Int8] {
        let r = Qwen35GgufResident::new(path.clone(), devices.clone(), 4096, TierPolicy::uniform(Dtype::I8)).with_kv_tier(tier).with_max_batch(2);
        let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
        if placed.is_empty() {
            brain_testutil::skip_unavailable("the checkpoint does not fit the GPUs this run may use");
            return;
        }
        let inst = r.activate_owned(&placed).expect("activate the real checkpoint");
        assert_eq!(inst.kv_tier(), tier);
        inst.prefill_timed(&[1, 2, 3, 4, 5, 6, 7, 8]).expect("prefill");
        for pos in 8..12u32 {
            inst.decode_batch_at(&[pos + 1, pos + 2], &[pos, pos + 20]).expect("batched decode");
        }
        assert!(live_resources().device_bytes > baseline.device_bytes + (1 << 30), "{tier}: the model occupied no device memory, so this proves nothing");
        drop(inst);
        assert_eq!(live_resources(), baseline, "{tier}: dropping the instance left CUDA objects behind");
    }
}
