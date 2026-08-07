// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device telemetry: what the GPU was actually doing, read live from sysfs.
//!
//! Every field is independently `Option` — a missing or unreadable path is
//! `None`, never a fabricated `0`, matching the rule the rest of this crate
//! already enforces for unmeasured metrics (`schema.rs`).
//!
//! Two privilege tiers apply on an Intel iGPU box (see `docs/performance/
//! arc.md` §0 once it exists):
//!
//! * **Frequency / RC6 / throttle reasons / package temperature** — plain
//!   `/sys/class/drm/*/gt/gt0/*` and `/sys/class/hwmon/*` reads. No privilege
//!   required, and this is the half this module implements.
//! * **RAPL package energy** (`/sys/class/powercap/intel-rapl:*`) and the
//!   i915 PMU (`perf_event_open`, needs `CAP_PERFMON`) are deliberately NOT
//!   attempted here — RAPL is `energy.rs`'s concern (a joules-over-a-window
//!   sampler, not a point-in-time GPU-state sample), and the PMU needs a
//!   privileged `perf_event_open` this crate has no reason to link.
//!
//! Nothing here is Arc-specific in principle — it reads the standard i915
//! sysfs ABI — but it has only been exercised against an Intel Arc iGPU
//! (Meteor Lake, `card1` on that box). A discrete NVIDIA/AMD card, or a box
//! with no i915 driver at all, correctly reports every field `None`.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Which power/thermal limit(s) the driver reports as currently active.
/// Every field independently `Option` — `None` means "this reason file did
/// not exist or was not readable", not "not throttling".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThrottleFlags {
    pub status: Option<bool>,
    pub pl1: Option<bool>,
    pub pl2: Option<bool>,
    pub pl4: Option<bool>,
    pub prochot: Option<bool>,
    pub thermal: Option<bool>,
    pub vr_thermalert: Option<bool>,
}

impl ThrottleFlags {
    /// True only when at least one reason is confirmed `Some(true)` — a
    /// device with no readable throttle files (all `None`) is reported as
    /// "unknown", by `DeviceSample::throttle` staying `None` at the call
    /// site, not folded into this as a false "not throttling".
    pub fn any_active(&self) -> bool {
        [self.status, self.pl1, self.pl2, self.pl4, self.prochot, self.thermal, self.vr_thermalert]
            .into_iter()
            .any(|f| f == Some(true))
    }

    fn all_none(&self) -> bool {
        *self == ThrottleFlags::default()
    }
}

/// One point-in-time sample of GPU device state. Every field independently
/// optional — see the module doc.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceSample {
    /// `rps_act_freq_mhz` — the frequency the GPU was actually running at.
    pub gt_act_mhz: Option<u32>,
    /// `punit_req_freq_mhz` — what the power unit is currently requesting.
    pub gt_req_mhz: Option<u32>,
    /// `rps_max_freq_mhz` — the currently configured ceiling (a knob, not
    /// necessarily the hardware maximum — see [`static_info`] for that).
    pub gt_max_mhz: Option<u32>,
    /// `rc6_residency_ms` — cumulative time spent in the RC6 idle state
    /// since boot. A monotonic counter; callers wanting a rate take a delta
    /// across two samples themselves (see [`Self` docs on `rc6_pct_between`]).
    pub rc6_residency_ms: Option<u64>,
    /// `None` when no throttle_reason_* file was readable at all; `Some`
    /// once at least one was, even if every flag inside reads `false`.
    pub throttle: Option<ThrottleFlags>,
    /// Package temperature, degrees C, from `hwmon`'s `coretemp` driver.
    pub pkg_temp_c: Option<f32>,
}

impl DeviceSample {
    /// RC6 residency as a percent of wall-clock time elapsed between two
    /// samples — `self` earlier, `later` after `elapsed`. `None` if either
    /// sample is missing the counter, `elapsed` is non-positive, or the
    /// counter went backwards (a reset or a wraparound, not a valid delta).
    pub fn rc6_pct_between(&self, later: &DeviceSample, elapsed: std::time::Duration) -> Option<f32> {
        let (a, b) = (self.rc6_residency_ms?, later.rc6_residency_ms?);
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        if elapsed_ms <= 0.0 || b < a {
            return None;
        }
        Some((100.0 * (b - a) as f64 / elapsed_ms) as f32)
    }
}

/// Static (rarely-changing) device configuration — the ceiling the driver
/// will allow, as opposed to [`DeviceSample`]'s live state. Read once; safe
/// to cache for a whole process.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticInfo {
    /// `rps_RP0_freq_mhz` — the hardware's real maximum, as opposed to
    /// `gt_max_mhz`'s currently-configured (and overridable) ceiling.
    pub rp0_freq_mhz: Option<u32>,
    /// `rps_min_freq_mhz` — non-default here means someone pinned the floor.
    pub rps_min_freq_mhz: Option<u32>,
}

/// A thin wrapper around a directory of sysfs-shaped files, so the parsing
/// logic is testable against a fixture tree instead of only the real
/// `/sys/class/drm/...` path.
struct SysfsRoot(PathBuf);

impl SysfsRoot {
    fn read_raw(&self, rel: &str) -> Option<String> {
        std::fs::read_to_string(self.0.join(rel)).ok().map(|s| s.trim().to_string())
    }

    fn read_u32(&self, rel: &str) -> Option<u32> {
        self.read_raw(rel)?.parse().ok()
    }

    fn read_u64(&self, rel: &str) -> Option<u64> {
        self.read_raw(rel)?.parse().ok()
    }

    /// i915's `throttle_reason_*` files hold `0` or `1`.
    fn read_bool01(&self, rel: &str) -> Option<bool> {
        self.read_u32(rel).map(|v| v != 0)
    }
}

/// Find the first Intel (`vendor == 0x8086`) DRM card directory —
/// *discovered*, never hardcoded, so this works whichever index the kernel
/// assigned (on the box this was written against, the Arc iGPU happens to
/// be `card1`, not `card0`). `env.rs`'s own rule applies here too: nothing
/// is configured, so this can never drift out of sync with the machine
/// that runs it.
fn intel_card_dir(drm_root: &Path) -> Option<PathBuf> {
    let mut cards: Vec<PathBuf> = std::fs::read_dir(drm_root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            // Only bare `cardN` directories carry a `device/` subtree; the
            // `cardN-<connector>` siblings (`card1-DP-1`, ...) do not.
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
        })
        .collect();
    cards.sort();
    cards.into_iter().find(|c| {
        std::fs::read_to_string(c.join("device/vendor")).map(|v| v.trim() == "0x8086").unwrap_or(false)
    })
}

fn gt0_root(drm_root: &Path) -> Option<SysfsRoot> {
    Some(SysfsRoot(intel_card_dir(drm_root)?.join("gt/gt0")))
}

/// Package temperature via `hwmon`'s `coretemp` driver: find the `hwmon*`
/// directory whose `name` file reads `coretemp`, then prefer the sensor
/// labelled "Package" over a bare `temp1_input` guess.
fn pkg_temp_c_from(hwmon_root: &Path) -> Option<f32> {
    let entries = std::fs::read_dir(hwmon_root).ok()?;
    for e in entries.filter_map(|e| e.ok()) {
        let dir = e.path();
        let name = std::fs::read_to_string(dir.join("name")).ok()?;
        if name.trim() != "coretemp" {
            continue;
        }
        // Prefer a sensor explicitly labelled "Package ...".
        for n in 1..=8 {
            let label = std::fs::read_to_string(dir.join(format!("temp{n}_label"))).ok();
            if label.as_deref().map(|l| l.contains("Package")).unwrap_or(false) {
                if let Some(m) = read_milli_c(&dir.join(format!("temp{n}_input"))) {
                    return Some(m);
                }
            }
        }
        // No labelled package sensor found; temp1_input is the conventional
        // fallback on single-package coretemp boxes.
        if let Some(m) = read_milli_c(&dir.join("temp1_input")) {
            return Some(m);
        }
    }
    None
}

fn read_milli_c(path: &Path) -> Option<f32> {
    let raw: f64 = std::fs::read_to_string(path).ok()?.trim().parse().ok()?;
    Some((raw / 1000.0) as f32)
}

fn sample_from(root: &SysfsRoot, hwmon_root: &Path) -> DeviceSample {
    let throttle = ThrottleFlags {
        status: root.read_bool01("throttle_reason_status"),
        pl1: root.read_bool01("throttle_reason_pl1"),
        pl2: root.read_bool01("throttle_reason_pl2"),
        pl4: root.read_bool01("throttle_reason_pl4"),
        prochot: root.read_bool01("throttle_reason_prochot"),
        thermal: root.read_bool01("throttle_reason_thermal"),
        vr_thermalert: root.read_bool01("throttle_reason_vr_thermalert"),
    };
    DeviceSample {
        gt_act_mhz: root.read_u32("rps_act_freq_mhz"),
        gt_req_mhz: root.read_u32("punit_req_freq_mhz"),
        gt_max_mhz: root.read_u32("rps_max_freq_mhz"),
        rc6_residency_ms: root.read_u64("rc6_residency_ms"),
        throttle: (!throttle.all_none()).then_some(throttle),
        pkg_temp_c: pkg_temp_c_from(hwmon_root),
    }
}

fn static_from(root: &SysfsRoot) -> StaticInfo {
    StaticInfo {
        rp0_freq_mhz: root.read_u32("rps_RP0_freq_mhz"),
        rps_min_freq_mhz: root.read_u32("rps_min_freq_mhz"),
    }
}

/// Sample the first Intel GPU's live state. Every field `None` when this
/// box has no i915 sysfs tree (a discrete-only NVIDIA/AMD box, a container
/// without `/sys/class/drm` mounted, ...) — never a fabricated zero.
pub fn sample() -> DeviceSample {
    match gt0_root(Path::new("/sys/class/drm")) {
        Some(root) => sample_from(&root, Path::new("/sys/class/hwmon")),
        None => DeviceSample::default(),
    }
}

/// Read the static configuration once (the frequency ceiling the driver
/// will allow). Cheap; safe to call per-process rather than cache, but
/// `env.rs` calls it once at `Env::capture` time regardless.
pub fn static_info() -> StaticInfo {
    match gt0_root(Path::new("/sys/class/drm")) {
        Some(root) => static_from(&root),
        None => StaticInfo::default(),
    }
}

/// Merge a before/after telemetry pair into `crate::schema::empty_resources()`'s
/// JSON shape, filling only the keys this module can honestly answer.
///
/// Deliberately does **not** report a "mean frequency" — that needs
/// continuous sampling across the run (a background sampler, mirroring
/// `energy::PowerSampler`'s shape), which is a separate follow-up. Two point
/// samples can honestly report where the run started, where it ended, and
/// the RC6 fraction of the window — nothing more, and every key stays
/// `null` when its source sample lacks it, never a fabricated `0`.
pub fn resources_json(before: &DeviceSample, after: &DeviceSample, elapsed: Duration) -> Value {
    let mut v = crate::schema::empty_resources();
    let obj = v.as_object_mut().expect("empty_resources() is always a JSON object");
    obj.insert("gpu_freq_mhz_start".into(), before.gt_act_mhz.map(Value::from).unwrap_or(Value::Null));
    obj.insert("gpu_freq_mhz_end".into(), after.gt_act_mhz.map(Value::from).unwrap_or(Value::Null));
    obj.insert(
        "gpu_rc6_pct".into(),
        before.rc6_pct_between(after, elapsed).map(Value::from).unwrap_or(Value::Null),
    );
    let throttle_flag = |f: fn(&ThrottleFlags) -> Option<bool>| {
        after.throttle.as_ref().and_then(f).map(Value::from).unwrap_or(Value::Null)
    };
    obj.insert("gpu_throttled_pl1_at_end".into(), throttle_flag(|t| t.pl1));
    obj.insert("gpu_throttled_thermal_at_end".into(), throttle_flag(|t| t.thermal));
    obj.insert("pkg_temp_c_end".into(), after.pkg_temp_c.map(Value::from).unwrap_or(Value::Null));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, rel: &str, contents: &str) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, contents).unwrap();
    }

    #[test]
    fn missing_tree_reports_every_field_none_not_zero() {
        let tmp = std::env::temp_dir().join(format!("devicetel-missing-{}", std::process::id()));
        let s = sample_from(&SysfsRoot(tmp.join("gt0")), &tmp.join("hwmon"));
        assert_eq!(s, DeviceSample::default());
        assert!(s.gt_act_mhz.is_none());
        assert!(s.throttle.is_none());
        assert!(s.pkg_temp_c.is_none());
    }

    #[test]
    fn full_fixture_tree_parses_every_field() {
        let tmp = std::env::temp_dir().join(format!("devicetel-full-{}", std::process::id()));
        let gt0 = tmp.join("gt0");
        write(&gt0, "rps_act_freq_mhz", "1180\n");
        write(&gt0, "punit_req_freq_mhz", "1200\n");
        write(&gt0, "rps_max_freq_mhz", "2250\n");
        write(&gt0, "rps_RP0_freq_mhz", "2250\n");
        write(&gt0, "rps_min_freq_mhz", "100\n");
        write(&gt0, "rc6_residency_ms", "48130683\n");
        write(&gt0, "throttle_reason_status", "1\n");
        write(&gt0, "throttle_reason_pl1", "1\n");
        write(&gt0, "throttle_reason_pl2", "0\n");
        write(&gt0, "throttle_reason_pl4", "0\n");
        write(&gt0, "throttle_reason_prochot", "0\n");
        write(&gt0, "throttle_reason_thermal", "0\n");
        write(&gt0, "throttle_reason_vr_thermalert", "0\n");

        let hwmon = tmp.join("hwmon");
        write(&hwmon.join("hwmon3"), "name", "coretemp\n");
        write(&hwmon.join("hwmon3"), "temp1_label", "Package id 0\n");
        write(&hwmon.join("hwmon3"), "temp1_input", "68000\n");

        let root = SysfsRoot(gt0.clone());
        let s = sample_from(&root, &hwmon);
        assert_eq!(s.gt_act_mhz, Some(1180));
        assert_eq!(s.gt_req_mhz, Some(1200));
        assert_eq!(s.gt_max_mhz, Some(2250));
        assert_eq!(s.rc6_residency_ms, Some(48130683));
        assert_eq!(s.pkg_temp_c, Some(68.0));
        let t = s.throttle.expect("at least one throttle reason was readable");
        assert_eq!(t.status, Some(true));
        assert_eq!(t.pl1, Some(true));
        assert_eq!(t.pl2, Some(false));
        assert!(t.any_active());

        let si = static_from(&root);
        assert_eq!(si.rp0_freq_mhz, Some(2250));
        assert_eq!(si.rps_min_freq_mhz, Some(100));

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn throttle_is_none_when_no_reason_file_exists_but_freq_does() {
        let tmp = std::env::temp_dir().join(format!("devicetel-nothrottle-{}", std::process::id()));
        let gt0 = tmp.join("gt0");
        write(&gt0, "rps_act_freq_mhz", "800\n");
        let root = SysfsRoot(gt0.clone());
        let s = sample_from(&root, &tmp.join("hwmon"));
        assert_eq!(s.gt_act_mhz, Some(800));
        // No throttle_reason_* files were written at all -> unknown, not
        // "not throttling".
        assert!(s.throttle.is_none());
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn rc6_pct_between_rejects_a_backwards_or_zero_delta() {
        let a = DeviceSample { rc6_residency_ms: Some(1000), ..Default::default() };
        let b = DeviceSample { rc6_residency_ms: Some(1500), ..Default::default() };
        let dt = std::time::Duration::from_secs(1);
        assert_eq!(a.rc6_pct_between(&b, dt), Some(50.0));
        // Backwards (a counter reset) must not report a negative rate.
        assert_eq!(b.rc6_pct_between(&a, dt), None);
        // Zero elapsed must not divide by zero.
        assert_eq!(a.rc6_pct_between(&b, std::time::Duration::ZERO), None);
        // Missing counter on either side -> None, not a guess.
        let missing = DeviceSample::default();
        assert_eq!(a.rc6_pct_between(&missing, dt), None);
    }

    #[test]
    fn intel_card_dir_skips_connector_and_non_intel_siblings() {
        let tmp = std::env::temp_dir().join(format!("devicetel-drm-{}", std::process::id()));
        // A non-Intel card at a lower sort order than the real one, plus
        // connector subdirectories that must not be mistaken for cards.
        write(&tmp, "card0/device/vendor", "0x10de\n"); // NVIDIA
        write(&tmp, "card1/device/vendor", "0x8086\n"); // Intel
        fs::create_dir_all(tmp.join("card1-DP-1")).unwrap();
        fs::create_dir_all(tmp.join("renderD128")).unwrap();

        let found = intel_card_dir(&tmp).expect("an Intel card exists in the fixture");
        assert_eq!(found, tmp.join("card1"));
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn intel_card_dir_is_none_on_an_all_non_intel_box() {
        let tmp = std::env::temp_dir().join(format!("devicetel-nonintel-{}", std::process::id()));
        write(&tmp, "card0/device/vendor", "0x10de\n");
        assert!(intel_card_dir(&tmp).is_none());
        fs::remove_dir_all(&tmp).ok();
    }
}
