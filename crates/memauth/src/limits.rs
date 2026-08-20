// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The process-wide memory CEILING: `--limit-vram-total` / `--limit-ram-total`.
//!
//! Every reserve/limit brain had before this was ADVISORY and LOCAL: a
//! per-GPU `--reserve-gb` inside `brain serve`, a `MAX_FP32_KV_POOL_BYTES`
//! inside one resident, a `DEFAULT_RESERVE_BYTES` inside one model crate.
//! None of them bound a model that allocated straight through
//! `gpu_core::Gpu`, which is what every model actually does. This module is
//! the one process-wide ceiling instead: published ONCE from `main` (or the
//! environment), read by the single allocation chokepoint
//! (`gpu_core::Gpu::{storage,storage_init,buffer,uniform_dynamic}`) and by the
//! advisory budgets, so both sides agree and neither can be bypassed by
//! adding a model.
//!
//! Two ceilings, each governing one class of device:
//!
//! * `--limit-vram-total` - every [`Device::Gpu`], as ONE total across the
//!   process, not a per-card cap (see [`VRAM_POOL`]).
//! * `--limit-ram-total` - [`Device::Cpu`] and every [`Device::Npu`] (an NPU's
//!   bytes ARE host RAM), intersected with the LIVE [`HostProbe`] view, so
//!   whichever of the flag and the machine is tighter wins.
//!
//! Unset means unset: [`limits`] reports `None`, [`authority`] hands back
//! `None`, and the allocation path forwards to the backend with one relaxed
//! atomic load of overhead. Most runs never set either flag.
//!
//! Swedish Embedded AB implements hard, process-wide memory ceilings for
//! edge-AI inference stacks for its clients. If your team needs expertise in
//! bounding the GPU/host memory of a model runtime then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::{Arc, OnceLock};

use crate::{Denied, Device, MemoryAuthority, PoolId, PoolProbe, Topology, HOST_POOL};

#[cfg(test)]
use crate::FixedProbe;

/// The pool every [`Device::Gpu`] draws from under `--limit-vram-total`.
///
/// One pool for every card ON PURPOSE: the flag is a ceiling on what the
/// PROCESS may hold, so a two-GPU box under `--limit-vram-total 8G` holds 8
/// GiB in total, not 8 GiB per card. Disjoint from `HOST_POOL` (0) and from
/// both of `default_pool`'s index ranges.
pub const VRAM_POOL: PoolId = PoolId(3_000_000);

/// Environment fallbacks, read exactly once (in [`Limits::from_env`]) - the
/// same flag-beats-env shape as `--device`/`BRAIN_DEVICE`.
const ENV_VRAM: &str = "BRAIN_LIMIT_VRAM_TOTAL";
const ENV_RAM: &str = "BRAIN_LIMIT_RAM_TOTAL";

/// The published ceilings. `None` in a field means "no ceiling for that class
/// of device" - never zero, which would mean "no memory at all".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    /// `--limit-vram-total`: bytes, summed over every GPU.
    pub vram_total: Option<u64>,
    /// `--limit-ram-total`: bytes of host RAM (CPU + every NPU).
    pub ram_total: Option<u64>,
}

impl Limits {
    /// Whether ANY ceiling is in force. The one check every hot allocation
    /// path makes before doing anything else.
    pub fn enforcing(&self) -> bool {
        self.vram_total.is_some() || self.ram_total.is_some()
    }

    /// The ceiling governing `device`, if any.
    pub fn ceiling(&self, device: Device) -> Option<u64> {
        match device {
            Device::Gpu(_) => self.vram_total,
            Device::Cpu | Device::Npu(_) => self.ram_total,
        }
    }

    /// `capacity` bounded by `device`'s ceiling - how an advisory budget
    /// (`residency::budget::Budgets`) is kept from planning a placement the
    /// hard ceiling would refuse. Never inflates a capacity that is already
    /// smaller than the ceiling.
    pub fn clamp(&self, device: Device, capacity: u64) -> u64 {
        match self.ceiling(device) {
            Some(limit) => capacity.min(limit),
            None => capacity,
        }
    }

    /// Ceilings from `BRAIN_LIMIT_VRAM_TOTAL` / `BRAIN_LIMIT_RAM_TOTAL`. A
    /// value that does not parse is reported on stderr and ignored rather than
    /// killing a process that never asked for a ceiling on the command line -
    /// the CLI flag path, where the user just typed it, hard-exits instead.
    pub fn from_env() -> Limits {
        let read = |key: &str| -> Option<u64> {
            let raw = std::env::var(key).ok()?;
            match parse_size(&raw) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("brain: {key}={raw:?} ignored: {e}");
                    None
                }
            }
        };
        Limits { vram_total: read(ENV_VRAM), ram_total: read(ENV_RAM) }
    }

    /// Which pool each class of device draws from under these ceilings.
    pub fn topology(&self) -> Topology {
        let mut t = Topology::new();
        t.declare_all_gpus(VRAM_POOL);
        t.declare_all_npus(HOST_POOL);
        t.declare(Device::Cpu, HOST_POOL);
        t
    }

    /// The authority enforcing these ceilings, or `None` when none is set.
    /// `host` is the live host-RAM view (production passes [`HostProbe`];
    /// tests pass a [`FixedProbe`], so no test reads a real machine).
    pub fn authority_with_host(&self, host: Arc<dyn PoolProbe>) -> Option<MemoryAuthority> {
        if !self.enforcing() {
            return None;
        }
        let probe: Arc<dyn PoolProbe> = Arc::new(LimitProbe { limits: *self, host });
        Some(MemoryAuthority::new(self.topology(), probe, Default::default()))
    }

    /// [`Self::authority_with_host`] over the real machine.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn authority(&self) -> Option<MemoryAuthority> {
        self.authority_with_host(Arc::new(crate::HostProbe::new(HOST_POOL)))
    }

    /// wasm has no `/proc/meminfo`; the host pool is simply unbounded there
    /// (the VRAM ceiling still applies).
    #[cfg(target_arch = "wasm32")]
    pub fn authority(&self) -> Option<MemoryAuthority> {
        self.authority_with_host(Arc::new(Unbounded))
    }
}

/// The pool view the ceilings impose: a synthetic [`VRAM_POOL`] whose capacity
/// IS the flag (no driver query - brain owns the cards it allocates on, so its
/// own charge ledger is the honest accounting), and a host pool that is the
/// tighter of the flag and what [`HostProbe`] says is really free right now.
/// A pool with no ceiling reads as unbounded, so setting one flag never starts
/// silently enforcing the other.
struct LimitProbe {
    limits: Limits,
    host: Arc<dyn PoolProbe>,
}

impl LimitProbe {
    fn bound(&self, pool: PoolId, live: impl Fn() -> u64) -> u64 {
        if pool == VRAM_POOL {
            return self.limits.vram_total.unwrap_or(u64::MAX);
        }
        if pool == HOST_POOL {
            return match self.limits.ram_total {
                Some(l) => live().min(l),
                None => u64::MAX,
            };
        }
        u64::MAX
    }
}

impl PoolProbe for LimitProbe {
    fn total(&self, pool: PoolId) -> u64 {
        self.bound(pool, || self.host.total(pool))
    }
    fn available(&self, pool: PoolId) -> u64 {
        self.bound(pool, || self.host.available(pool))
    }
}

/// An always-unbounded probe (wasm, where there is no host view to read).
#[cfg(target_arch = "wasm32")]
struct Unbounded;
#[cfg(target_arch = "wasm32")]
impl PoolProbe for Unbounded {
    fn total(&self, _pool: PoolId) -> u64 {
        u64::MAX
    }
    fn available(&self, _pool: PoolId) -> u64 {
        u64::MAX
    }
}

// ---- the process-wide publication ------------------------------------------
//
// Exactly one `OnceLock`, same shape (and same rationale) as
// `gpu_core::devices::publish_compute_set`/`ambient_compute_set`: the CLI
// publishes what it parsed before any model is built, and any other entry
// point (a test binary, a library caller with no CLI in the loop) lazily
// resolves the environment instead. Whoever gets there first wins, so there is
// exactly one answer for the process's lifetime.

static LIMITS: OnceLock<Limits> = OnceLock::new();
static AUTHORITY: OnceLock<Option<MemoryAuthority>> = OnceLock::new();

/// Publish the ceilings for this process. Each `None` argument falls back to
/// that ceiling's environment variable, so a flag beats the environment and an
/// omitted flag does not erase it.
///
/// Called once from `main`, before any `Gpu` exists. A second call (or a call
/// after something already read [`limits`]) is ignored - the process cannot
/// change its own ceiling halfway through, which is what makes it hard to
/// bypass.
pub fn publish_limits(vram_total: Option<u64>, ram_total: Option<u64>) {
    let env = Limits::from_env();
    let _ = LIMITS.set(Limits { vram_total: vram_total.or(env.vram_total), ram_total: ram_total.or(env.ram_total) });
}

/// This process's ceilings: whatever [`publish_limits`] recorded, else the
/// environment, else none at all.
pub fn limits() -> Limits {
    *LIMITS.get_or_init(Limits::from_env)
}

/// Whether any ceiling is in force. The single branch an allocation takes when
/// nobody asked for a limit.
pub fn enforcing() -> bool {
    limits().enforcing()
}

/// The process-wide authority every allocation is charged against, or `None`
/// when no ceiling was set (the overwhelmingly common case - then nothing is
/// built, nothing is probed, and nothing is charged).
pub fn authority() -> Option<&'static MemoryAuthority> {
    AUTHORITY.get_or_init(|| limits().authority()).as_ref()
}

/// A one-line, actionable message for a refused allocation: which device, how
/// much was asked for, what the ceiling had left, and WHICH FLAG to raise.
pub fn denial_message(device: Device, tag: &str, bytes: u64, denied: Denied) -> String {
    let flag = match device {
        Device::Gpu(_) => "--limit-vram-total",
        Device::Cpu | Device::Npu(_) => "--limit-ram-total",
    };
    let name = match device {
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
        Device::Cpu => "cpu".to_string(),
    };
    let why = match denied {
        Denied::WouldExceedPool { headroom, .. } => format!("only {} is left under the ceiling", human_bytes(headroom)),
        Denied::NeverFits { usable, .. } => format!("the whole ceiling is {}", human_bytes(usable)),
    };
    format!("{flag} exceeded: {name} {tag} of {} refused - {why}. Raise {flag}, or run a smaller model/resolution/batch.", human_bytes(bytes))
}

/// Bytes in the largest binary unit that keeps the number readable. GiB is
/// what every memory message in this workspace speaks, but a ceiling can be
/// crossed by a 256 KiB buffer too, and reporting that as "0.00 GiB refused,
/// 0.00 GiB left" tells an operator nothing at all.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [("TiB", 1u64 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)];
    for (unit, scale) in UNITS {
        if bytes >= scale {
            return format!("{:.2} {unit}", bytes as f64 / scale as f64);
        }
    }
    format!("{bytes} B")
}

/// Parse a human memory size: a decimal number with an optional binary
/// suffix - `K`/`M`/`G`/`T`, case-insensitive, with an optional `i` and/or
/// `B` (`8G`, `8g`, `8GB`, `8GiB` are all 8 GiB). No suffix means bytes.
///
/// Binary (1024-based) throughout, deliberately: every other memory number in
/// this workspace - `--reserve-gb`, VRAM totals, `/proc/meminfo` - is binary,
/// and a `G` that meant 10^9 here would silently under-budget by 7%.
pub fn parse_size(text: &str) -> Result<u64, String> {
    let s = text.trim();
    if s.is_empty() {
        return Err("empty size (expected e.g. 8G, 8GiB, or a byte count)".to_string());
    }
    let digits_end = s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len());
    let (num, suffix) = s.split_at(digits_end);
    if num.is_empty() {
        return Err(format!("{text:?} has no number (expected e.g. 8G, 8GiB, or a byte count)"));
    }
    let scale: u64 = match suffix.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        "t" | "tb" | "tib" => 1u64 << 40,
        other => return Err(format!("unknown size suffix {other:?} in {text:?} (expected K, M, G or T)")),
    };
    let value: f64 = num.parse().map_err(|_| format!("{num:?} in {text:?} is not a number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{text:?} is not a valid size"));
    }
    let bytes = value * scale as f64;
    if bytes > u64::MAX as f64 {
        return Err(format!("{text:?} does not fit in 64 bits"));
    }
    Ok(bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const GB: u64 = 1 << 30;

    /// A stand-in for [`HostProbe`] on a 256 GiB box with `avail` bytes free
    /// right now, so the RAM ceiling is testable without reading a real
    /// `/proc/meminfo`. Physical total and live availability are deliberately
    /// two different numbers - that gap is what tells "can never fit" apart
    /// from "does not fit right now".
    fn host(avail: u64) -> Arc<dyn PoolProbe> {
        let p = FixedProbe::new();
        p.set(HOST_POOL, 256 * GB, avail);
        Arc::new(p)
    }

    #[test]
    fn parse_size_reads_human_suffixes_and_plain_bytes() {
        assert_eq!(parse_size("8G").unwrap(), 8 * GB);
        assert_eq!(parse_size("8GB").unwrap(), 8 * GB);
        assert_eq!(parse_size("8GiB").unwrap(), 8 * GB);
        assert_eq!(parse_size("8g").unwrap(), 8 * GB);
        assert_eq!(parse_size("1536M").unwrap(), 1536 << 20);
        assert_eq!(parse_size("1.5G").unwrap(), GB + GB / 2);
        assert_eq!(parse_size(" 4096 ").unwrap(), 4096);
        assert_eq!(parse_size("2T").unwrap(), 2 << 40);
        assert_eq!(parse_size("512K").unwrap(), 512 << 10);
        for bad in ["", "G", "8X", "-1", "8 G", "eight"] {
            assert!(parse_size(bad).is_err(), "{bad:?} must not parse as a size");
        }
    }

    /// The whole point of the feature: with no flag set, nothing changes and
    /// nothing is built. Every allocation path checks exactly this.
    #[test]
    fn no_limit_is_a_true_no_op() {
        let l = Limits::default();
        assert!(!l.enforcing());
        assert_eq!(l.clamp(Device::Gpu(0), 24 * GB), 24 * GB);
        assert_eq!(l.clamp(Device::Cpu, 128 * GB), 128 * GB);
        assert!(l.authority_with_host(host(200 * GB)).is_none(), "with no ceiling there is nothing to authorize against");
    }

    /// `--limit-vram-total` is a TOTAL over the process, not a per-card cap:
    /// two cards draw from one ceiling. A per-card reading would let a 2-GPU
    /// box quietly use twice what the operator asked for.
    #[test]
    fn the_vram_ceiling_is_one_total_across_every_card() {
        let l = Limits { vram_total: Some(8 * GB), ram_total: None };
        let a = l.authority_with_host(host(200 * GB)).expect("a ceiling was set");
        let g0 = a.request(Device::Gpu(0), 6 * GB, "weights").expect("6 GiB fits an 8 GiB ceiling");
        assert!(
            matches!(a.request(Device::Gpu(1), 6 * GB, "weights"), Err(Denied::WouldExceedPool { .. })),
            "the second card must draw from the SAME total"
        );
        assert!(matches!(a.request(Device::Gpu(0), 9 * GB, "huge"), Err(Denied::NeverFits { .. })), "bigger than the whole ceiling can never fit");
        drop(g0);
        assert!(a.request(Device::Gpu(1), 6 * GB, "weights").is_ok(), "releasing a grant must return its bytes to the shared total");
    }

    /// The RAM ceiling binds against the LIVE host view, not instead of it:
    /// whichever of the flag and the real `MemAvailable`/cgroup headroom is
    /// tighter must win, in both directions.
    #[test]
    fn the_ram_ceiling_and_the_live_host_probe_both_bind() {
        let l = Limits { vram_total: None, ram_total: Some(4 * GB) };
        let a = l.authority_with_host(host(200 * GB)).expect("a ceiling was set");
        assert!(matches!(a.request(Device::Cpu, 5 * GB, "kv"), Err(Denied::NeverFits { .. })), "the flag must win when the box has plenty");
        assert!(a.request(Device::Cpu, 3 * GB, "kv").is_ok());

        let tight = l.authority_with_host(host(GB)).expect("a ceiling was set");
        assert!(
            matches!(tight.request(Device::Cpu, 3 * GB, "kv"), Err(Denied::WouldExceedPool { .. })),
            "the live probe must win when the box has LESS free than the flag allows"
        );
    }

    /// An NPU's bytes are host RAM (`docs/models/*`'s Meteor-Lake note), so
    /// they belong to the RAM ceiling; a GPU's belong to the VRAM one. Setting
    /// only one flag must leave the other pool entirely unbounded.
    #[test]
    fn each_ceiling_governs_only_its_own_class_of_device() {
        let l = Limits { vram_total: Some(2 * GB), ram_total: None };
        let a = l.authority_with_host(host(GB / 2)).expect("a ceiling was set");
        assert!(a.request(Device::Cpu, 64 * GB, "host").is_ok(), "--limit-vram-total alone must not start bounding host RAM");
        assert!(a.request(Device::Npu(0), 64 * GB, "npu").is_ok(), "an NPU is host memory, unbounded while --limit-ram-total is unset");
        assert!(matches!(a.request(Device::Gpu(0), 3 * GB, "dit"), Err(Denied::NeverFits { .. })));

        let l = Limits { vram_total: None, ram_total: Some(2 * GB) };
        let a = l.authority_with_host(host(200 * GB)).expect("a ceiling was set");
        assert!(matches!(a.request(Device::Npu(0), 3 * GB, "npu"), Err(Denied::NeverFits { .. })), "an NPU draws from the RAM ceiling");
        assert!(a.request(Device::Gpu(0), 64 * GB, "dit").is_ok(), "--limit-ram-total alone must not bound VRAM");
    }

    /// The advisory side (`residency::Budgets`) and the hard side (the
    /// authority above) must agree, so a placement is never planned against
    /// capacity the ceiling would refuse.
    #[test]
    fn clamp_reports_the_tighter_of_the_real_capacity_and_the_ceiling() {
        let l = Limits { vram_total: Some(8 * GB), ram_total: Some(16 * GB) };
        assert_eq!(l.clamp(Device::Gpu(0), 24 * GB), 8 * GB, "a 24 GiB card under an 8 GiB ceiling is an 8 GiB budget");
        assert_eq!(l.clamp(Device::Gpu(0), 4 * GB), 4 * GB, "the ceiling never INFLATES a smaller real capacity");
        assert_eq!(l.clamp(Device::Cpu, 128 * GB), 16 * GB);
        assert_eq!(l.clamp(Device::Npu(0), 128 * GB), 16 * GB);
    }

    /// `declare_all_gpus`/`declare_all_npus`: a class-wide pool declaration,
    /// which is what a process-wide ceiling IS (every card, however many, one
    /// pool). An explicit per-device `declare` still wins over it.
    #[test]
    fn a_class_wide_pool_declaration_covers_every_index() {
        let mut t = Topology::new();
        t.declare_all_gpus(PoolId(7));
        assert_eq!(t.pool_of(Device::Gpu(0)), PoolId(7));
        assert_eq!(t.pool_of(Device::Gpu(31)), PoolId(7), "an index nobody enumerated must still land in the class pool");
        assert_eq!(t.pool_of(Device::Cpu), HOST_POOL, "the CPU is not a GPU");
        t.declare(Device::Gpu(1), PoolId(9));
        assert_eq!(t.pool_of(Device::Gpu(1)), PoolId(9), "an explicit declaration must beat the class-wide one");
        assert_eq!(t.pool_of(Device::Gpu(0)), PoolId(7));
    }

    /// The message a denied allocation carries must name the flag to raise and
    /// the numbers involved - a bare `Denied` debug print does not tell an
    /// operator what to do next.
    #[test]
    fn a_denial_names_the_flag_and_the_numbers() {
        let l = Limits { vram_total: Some(GB), ram_total: None };
        let a = l.authority_with_host(host(200 * GB)).expect("a ceiling was set");
        let d = a.request(Device::Gpu(0), 4 * GB, "storage").unwrap_err();
        let msg = denial_message(Device::Gpu(0), "storage", 4 * GB, d);
        assert!(msg.contains("--limit-vram-total"), "must name the flag to raise: {msg}");
        assert!(msg.contains("gpu0"), "must name the device: {msg}");
        assert!(msg.contains("4.00 GiB"), "must name the requested size: {msg}");

        // A ceiling can be crossed by a small buffer too; the message must
        // stay informative there instead of rounding everything to 0.00 GiB.
        let tiny = Limits { vram_total: Some(128 << 10), ram_total: None }.authority_with_host(host(200 * GB)).expect("a ceiling was set");
        let small = tiny.request(Device::Gpu(0), 256 << 10, "uniform").unwrap_err();
        let msg = denial_message(Device::Gpu(0), "uniform", 256 << 10, small);
        assert!(msg.contains("256.00 KiB"), "a sub-MiB request must be readable: {msg}");

        let l = Limits { vram_total: None, ram_total: Some(GB) };
        let a = l.authority_with_host(host(200 * GB)).expect("a ceiling was set");
        let d = a.request(Device::Cpu, 4 * GB, "storage").unwrap_err();
        assert!(denial_message(Device::Cpu, "storage", 4 * GB, d).contains("--limit-ram-total"));
    }
}
