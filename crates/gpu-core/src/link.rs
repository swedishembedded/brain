// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How two cards reach each other, measured and never assumed.
//!
//! A model split across cards hands activations from one to the next. Whether
//! that goes straight over the device-to-device link (NVLink, PCIe peer-to-peer)
//! or through host memory is a fact about the pair, so brain asks the driver
//! ([`Gpu::peer_link`]), uses the direct path whenever there is one
//! ([`Gpu::copy_to`]), and states - once per pair - why it did not.
//!
//! What a link *sustains* is not in any driver table, so [`bandwidth`] times it:
//! a short copy loop between two small buffers, in both paths, kept in memory
//! and on disk beside the other device measurements (`roof`, `tune`), keyed by
//! the two physical cards and a fingerprint of the probe so a changed probe
//! re-measures instead of reusing an old number.
//!
//! Swedish Embedded AB implements multi-GPU inference runtimes for its clients.
//! If your team needs expertise in splitting a model across accelerators
//! without paying for a host round trip at every boundary, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::Gpu;
use std::sync::Mutex;
use std::time::Instant;

/// How a [`Gpu::copy_to`] moved its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// Straight from one card's memory into the other's.
    Peer,
    /// Down to host memory and back up: the fallback every backend has.
    HostStaged,
}

/// The measured bandwidth of one directed pair of cards, GB/s (1e9 bytes/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bandwidth {
    /// The direct path; `None` when the pair has none.
    pub peer_gbs: Option<f32>,
    /// The host-staged path, which every pair has.
    pub host_gbs: f32,
}

impl Bandwidth {
    /// The rate [`Gpu::copy_to`] gets on this pair.
    pub fn used_gbs(&self) -> f32 {
        self.peer_gbs.unwrap_or(self.host_gbs)
    }
}

/// Bytes moved per probe copy. A few MiB: large enough to be bandwidth-bound on
/// the fastest link there is, small enough to be harmless on a card that is
/// serving something else.
const PROBE_BYTES: u64 = 8 << 20;
/// Copies per timed round of the direct path, and rounds timed (the minimum
/// over rounds is kept: another process's traffic only ever adds time).
const PROBE_REPS: usize = 32;
const PROBE_ROUNDS: usize = 3;
/// The host-staged path is slower and synchronous; fewer copies are enough.
const PROBE_HOST_REPS: usize = 4;

/// What is in the probe, as text: part of the cache key, so editing a constant
/// above re-measures instead of trusting an old file.
fn probe_description() -> String {
    format!("link-probe-v1 bytes={PROBE_BYTES} reps={PROBE_REPS} rounds={PROBE_ROUNDS} host_reps={PROBE_HOST_REPS}")
}

/// The identities of a directed pair, when both cards have one.
fn pair_of(from: &Gpu, to: &Gpu) -> Option<(backend_api::GpuIdentity, backend_api::GpuIdentity)> {
    Some((from.identity()?, to.identity()?))
}

static MEMO: Mutex<Vec<(&'static str, backend_api::GpuIdentity, backend_api::GpuIdentity, Bandwidth)>> = Mutex::new(Vec::new());
/// Serialises probing: two threads measuring the same link would each time the
/// other's traffic.
static PROBING: Mutex<()> = Mutex::new(());
static STATED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The measured bandwidth of copies from `from` to `to`, probing the pair the
/// first time and reading the on-disk record after that. `None` when the pair
/// cannot be measured: a backend without device identity, or the same card.
pub fn bandwidth(from: &Gpu, to: &Gpu) -> Option<Bandwidth> {
    let (a, b) = pair_of(from, to)?;
    let kind = from.kind();
    let known = |m: &Vec<(&'static str, backend_api::GpuIdentity, backend_api::GpuIdentity, Bandwidth)>| {
        m.iter().find(|(k, x, y, _)| *k == kind && x.same_device(&a) && y.same_device(&b)).map(|e| e.3)
    };
    if let Some(bw) = known(&MEMO.lock().unwrap_or_else(|e| e.into_inner())) {
        return Some(bw);
    }
    let _probing = PROBING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bw) = known(&MEMO.lock().unwrap_or_else(|e| e.into_inner())) {
        return Some(bw);
    }
    let store = store::path(kind, &a, &b);
    let bw = match store.as_ref().and_then(store::load) {
        Some(bw) => bw,
        None => {
            let bw = measure(from, to)?;
            if let Some(p) = &store {
                store::save(p, bw);
            }
            bw
        }
    };
    MEMO.lock().unwrap_or_else(|e| e.into_inner()).push((kind, a, b, bw));
    Some(bw)
}

/// Time both paths now, ignoring and not updating the cache. `None` if a copy
/// cannot be timed (zero elapsed time).
pub fn measure(from: &Gpu, to: &Gpu) -> Option<Bandwidth> {
    let words = PROBE_BYTES / 4;
    let (src, dst) = (from.storage(words), to.storage(words));
    let direct = from.peer_link(to).direct;
    let settle = || {
        from.poll_wait();
        to.poll_wait();
    };
    let gbs = |reps: usize, secs: f64| (secs > 0.0).then(|| (PROBE_BYTES as f64 * reps as f64 / secs / 1e9) as f32);

    let peer_gbs = if direct {
        // Warm the pair (peer access is enabled by the first copy), then keep the fastest round.
        let mut peer = from.copy_to(&src, 0, to, &dst, 0, PROBE_BYTES) == Path::Peer;
        settle();
        let mut best = f64::INFINITY;
        for _ in 0..PROBE_ROUNDS {
            let t = Instant::now();
            for _ in 0..PROBE_REPS {
                peer &= from.copy_to(&src, 0, to, &dst, 0, PROBE_BYTES) == Path::Peer;
            }
            settle();
            best = best.min(t.elapsed().as_secs_f64());
        }
        // A pair the driver calls direct but whose copy fell back has no direct bandwidth to report.
        gbs(PROBE_REPS, best).filter(|_| peer)
    } else {
        None
    };

    from.copy_via_host(&src, 0, to, &dst, 0, PROBE_BYTES);
    settle();
    let t = Instant::now();
    for _ in 0..PROBE_HOST_REPS {
        from.copy_via_host(&src, 0, to, &dst, 0, PROBE_BYTES);
    }
    settle();
    let host_gbs = gbs(PROBE_HOST_REPS, t.elapsed().as_secs_f64())?;
    Some(Bandwidth { peer_gbs, host_gbs })
}

/// Print, once per directed pair of cards in this process, why a transfer goes
/// through the host. A fallback that is silent is a performance bug nobody can
/// find.
pub(crate) fn state_host_staging(from: &Gpu, to: &Gpu, reason: &str) {
    let (key, what) = match pair_of(from, to) {
        Some((a, b)) => (
            format!("{}>{}", crate::roof::identity_slug(&a), crate::roof::identity_slug(&b)),
            format!("transfers from {} to {}", a.name, b.name),
        ),
        // A backend without device identity shares one line for every pair.
        None => (from.kind().to_string(), format!("device-to-device transfers on the {} backend", from.kind())),
    };
    let mut once = STATED.lock().unwrap_or_else(|e| e.into_inner());
    if !once.contains(&key) {
        eprintln!("brain: {what} go through host memory ({reason})");
        once.push(key);
    }
}

mod store {
    use super::{probe_description, Bandwidth};
    use backend_api::GpuIdentity;
    use std::path::PathBuf;

    pub fn path(kind: &str, from: &GpuIdentity, to: &GpuIdentity) -> Option<PathBuf> {
        let dir = crate::roof::cache_dir()?;
        let hash = crate::tune::source_fingerprint(&[&probe_description()]);
        let slug = |id: &GpuIdentity| crate::roof::identity_slug(id);
        Some(dir.join(format!("link-{kind}-{}-{}-{hash:016x}.txt", slug(from), slug(to))))
    }

    /// A missing, unparseable or non-finite record is ignored, never trusted.
    pub fn load(path: &PathBuf) -> Option<Bandwidth> {
        let text = std::fs::read_to_string(path).ok()?;
        let (mut peer, mut host) = (None, None);
        for line in text.lines() {
            let (k, v) = line.split_once('=')?;
            let v: f32 = v.trim().parse().ok()?;
            match k.trim() {
                "peer_gbs" => peer = Some(v),
                "host_gbs" => host = Some(v),
                _ => {}
            }
        }
        let bw = Bandwidth { peer_gbs: peer, host_gbs: host? };
        let sane = |v: f32| v.is_finite() && v > 0.0;
        (sane(bw.host_gbs) && bw.peer_gbs.is_none_or(sane)).then_some(bw)
    }

    pub fn save(path: &PathBuf, bw: Bandwidth) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("tmp");
        let body = format!("{}host_gbs={}\n", bw.peer_gbs.map(|v| format!("peer_gbs={v}\n")).unwrap_or_default(), bw.host_gbs);
        if std::fs::write(&tmp, body).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_record_round_trips_and_a_damaged_one_is_ignored() {
        let dir = std::env::temp_dir().join(format!("brain-link-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("link.txt");
        let bw = Bandwidth { peer_gbs: Some(123.5), host_gbs: 9.25 };
        store::save(&path, bw);
        assert_eq!(store::load(&path), Some(bw));
        store::save(&path, Bandwidth { peer_gbs: None, host_gbs: 9.25 });
        assert_eq!(store::load(&path).map(|b| b.peer_gbs), Some(None));
        for bad in ["host_gbs=0\n", "peer_gbs=nan\nhost_gbs=1\n", "peer_gbs=5\n", "garbage"] {
            std::fs::write(&path, bad).unwrap();
            assert_eq!(store::load(&path), None, "{bad:?} must not be trusted");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
