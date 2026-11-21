// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for putting one inference endpoint
// in front of every card of a cluster for its clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `brain serve --upstream URL ...`: the serve verb as a router over other `brain serve` endpoints
//! (`apiserve::route`). It loads no model and claims no device; it parses the upstream flags, refuses the ones that
//! only mean something to a server that runs models, and serves the OpenAI dialect over the fleet.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use apiserve::route::{self, Fleet, FleetConfig, UpstreamTls, UpstreamUrl};

/// The upstream flags of `brain serve`, as parsed.
#[derive(Default)]
pub(crate) struct UpstreamFlags {
    pub urls: Vec<String>,
    pub files: Vec<PathBuf>,
    pub key_file: Option<PathBuf>,
    pub ca: Option<PathBuf>,
    pub timeout_ms: Option<u64>,
    pub health_ms: Option<u64>,
    /// `--route-queue N`: the requests that may wait in the router for a slot.
    pub queue: Option<usize>,
    /// `--drain-deadline-ms N`: how long a drained upstream may finish what it runs.
    pub drain_ms: Option<u64>,
}

impl UpstreamFlags {
    /// Whether the server routes to upstreams instead of serving models.
    pub fn routing(&self) -> bool {
        !self.urls.is_empty() || !self.files.is_empty()
    }

    /// Whether a flag that only a router reads was given.
    pub fn any_given(&self) -> bool {
        self.routing() || self.key_file.is_some() || self.ca.is_some() || self.timeout_ms.is_some() || self.health_ms.is_some() || self.queue.is_some() || self.drain_ms.is_some()
    }

    /// The fleet these flags name; a request waits in the router's queue for at most `queue_deadline`.
    ///
    /// # Errors
    /// A malformed URL or upstreams file, an unreadable key or CA file, a plaintext upstream off this machine without
    /// `insecure_plaintext`, or a fleet [`Fleet::new`] refuses.
    pub fn fleet(&self, insecure_plaintext: bool, queue_deadline: Duration) -> Result<Arc<Fleet>, String> {
        let mut upstreams = Vec::new();
        for url in &self.urls {
            upstreams.extend(url.split(',').filter(|u| !u.trim().is_empty()).map(UpstreamUrl::parse).collect::<Result<Vec<_>, _>>()?);
        }
        for file in &self.files {
            let text = std::fs::read_to_string(file).map_err(|e| format!("--upstreams-file {}: {e}", file.display()))?;
            upstreams.extend(route::parse_upstreams_file(&text).map_err(|e| format!("--upstreams-file {}: {e}", file.display()))?);
        }
        route::check_upstream_exposure(&upstreams, insecure_plaintext)?;
        let key = self.key_file.as_deref().map(|f| apiserve::surface::read_key_file(f).map_err(|e| format!("--upstream-key-file {e}"))).transpose()?;
        let tls = self.ca.as_deref().map(UpstreamTls::from_pem_file).transpose()?;
        let mut config = FleetConfig::new(upstreams).with_upstream_key(key).with_upstream_tls(tls);
        let bound = self.queue.unwrap_or(config.queue_bound);
        config = config.with_queue(bound, queue_deadline);
        if let Some(ms) = self.timeout_ms {
            config = config.with_response_timeout(Duration::from_millis(ms));
        }
        if let Some(ms) = self.drain_ms {
            config = config.with_drain_deadline(Some(Duration::from_millis(ms)));
        }
        if let Some(ms) = self.health_ms {
            config = config.with_health_interval(Duration::from_millis(ms));
        }
        Fleet::new(config)
    }
}

/// What the router serves with, beyond the fleet.
pub(crate) struct RouteServe {
    pub addrs: Vec<SocketAddr>,
    pub fixed_key: Option<String>,
    pub fixed_key_file: Option<PathBuf>,
    pub callers: Option<Arc<apiserve::Callers>>,
    pub api_keys_out: Option<String>,
    pub tls: Option<apiserve::Tls>,
    pub ready: brain_shutdown::ready::Gate,
    pub record: PathBuf,
}

/// Serve `fleet` until a signal stops it. Keys are announced as `brain serve` announces them for its own surfaces.
pub(crate) fn run_route(fleet: Arc<Fleet>, serve: RouteServe) {
    let RouteServe { addrs, fixed_key, fixed_key_file, callers, api_keys_out, tls, ready, record } = serve;
    let key = if callers.is_some() { String::new() } else { apiserve::surface::resolved_key(fixed_key.as_deref()) };
    let surfaces: Vec<apiserve::Surface> = addrs.iter().map(|&a| apiserve::Surface::new(apiserve::Provider::OpenAI, a, key.clone())).collect();
    let keys_file = api_keys_out.as_deref().map(Path::new).filter(|path| match apiserve::write_keys(&surfaces, path) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("brain serve: --api-keys-out {}: {e}", path.display());
            false
        }
    });
    if let Some(first) = surfaces.first() {
        match (&fixed_key_file, &callers) {
            (_, Some(_)) => println!("APIKEY {} per caller (--caller)", first.provider),
            (Some(file), None) => println!("{}", first.announcement_of_pinned(file)),
            (None, None) => first.announce(keys_file),
        }
    }
    let authenticator: Arc<dyn apiserve::Authenticator> = match callers {
        Some(callers) => callers,
        None => Arc::new(apiserve::StaticKey::new(key.clone())),
    };
    let (trigger, shutdown) = brain_shutdown::channel();
    brain_shutdown::install_signals(trigger);
    let opts = route::RouteOpts { shutdown: Some(shutdown), ready, limits: apiserve::ConnectionLimits::default(), tls, record: Some((record, key)) };
    if let Err(e) = route::serve(fleet, addrs, authenticator, opts) {
        eprintln!("brain serve: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstreams_come_from_flags_and_files_and_plaintext_off_loopback_is_refused() {
        let dir = brain_testutil::Scratch::dir("brain-route-flags");
        let file = dir.path().join("upstreams");
        std::fs::write(&file, "# fleet\nhttp://127.0.0.1:9002\n\nhttp://127.0.0.1:9003\n").unwrap();
        let flags = UpstreamFlags { urls: vec!["http://127.0.0.1:9000,http://127.0.0.1:9001".into()], files: vec![file.clone()], ..UpstreamFlags::default() };
        assert!(flags.routing());
        let fleet = flags.fleet(false, Duration::from_secs(1)).expect("four loopback upstreams");
        assert_eq!(fleet.status()["upstreams"].as_array().unwrap().len(), 4);
        let far = UpstreamFlags { urls: vec!["http://10.0.0.1:9000".into()], ..UpstreamFlags::default() };
        assert!(far.fleet(false, Duration::from_secs(1)).unwrap_err().contains("--insecure-plaintext"));
        assert!(far.fleet(true, Duration::from_secs(1)).is_ok());
        std::fs::write(&file, "http://127.0.0.1:9002/v1\n").unwrap();
        let bad = UpstreamFlags { files: vec![file], ..UpstreamFlags::default() };
        assert!(bad.fleet(false, Duration::from_secs(1)).unwrap_err().contains("line 1"));
    }
}
