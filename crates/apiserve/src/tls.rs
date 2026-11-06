// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving models over networks
// that cannot be trusted for its clients. If your team needs expertise in
// transport security for inference services then you can procure our services
// by sending an email to info@swedishembedded.com.

//! TLS for the HTTP surfaces, and the rule that decides when it is required.
//!
//! The key a client sends and every prompt and output cross the network in a
//! request. On loopback that is the host's own business; on any other address
//! a private network is still a network, and a host on it reads plaintext.
//! So a surface bound anywhere but loopback is refused without TLS unless the
//! operator says, by flag, that something else already encrypts the path.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// A loaded certificate chain and key, shared by every surface of a process.
#[derive(Clone)]
pub struct Tls {
    pub(crate) acceptor: tokio_rustls::TlsAcceptor,
}

impl std::fmt::Debug for Tls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tls { .. }")
    }
}

impl Tls {
    /// Load a PEM certificate chain (leaf first) and its PEM private key.
    /// Only HTTP/1.1 is offered by ALPN, matching what the listener serves.
    ///
    /// # Errors
    ///
    /// A file that cannot be read, holds no certificate or no key, or a key
    /// that does not match the leaf certificate. The message names the file;
    /// it never carries key material.
    pub fn from_pem_files(cert: &Path, key: &Path) -> Result<Tls, String> {
        let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)
            .map_err(|e| format!("TLS certificate {}: {e}", cert.display()))?
            .collect::<Result<_, _>>()
            .map_err(|e| format!("TLS certificate {}: {e}", cert.display()))?;
        if chain.is_empty() {
            return Err(format!("TLS certificate {} holds no PEM certificate", cert.display()));
        }
        let key = PrivateKeyDer::from_pem_file(key).map_err(|e| format!("TLS key {}: {e}", key.display()))?;
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("TLS configuration: {e}"))?
            .with_no_client_auth()
            .with_single_cert(chain, key)
            .map_err(|e| format!("TLS certificate {} and its key: {e}", cert.display()))?;
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(Tls { acceptor: tokio_rustls::TlsAcceptor::from(Arc::new(config)) })
    }
}

/// Refuse to serve `addrs` in plaintext anywhere but loopback.
///
/// `tls` is whether the surfaces are served with TLS; `insecure_plaintext` is
/// the operator's explicit statement that plaintext off loopback is intended
/// (a TLS-terminating proxy, a VPN).
///
/// # Errors
///
/// The first address that would carry plaintext off loopback, with the flags
/// that fix it.
pub fn check_exposure(addrs: &[SocketAddr], tls: bool, insecure_plaintext: bool) -> Result<(), String> {
    if tls || insecure_plaintext {
        return Ok(());
    }
    match addrs.iter().find(|addr| !addr.ip().is_loopback()) {
        None => Ok(()),
        Some(addr) => Err(format!(
            "{addr} is not a loopback address, and without TLS the API key and every request would cross the network readable; \
             pass --tls-cert <pem> --tls-key <pem>, or --insecure-plaintext when a proxy or VPN already encrypts the path"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addrs(list: &[&str]) -> Vec<SocketAddr> {
        list.iter().map(|a| a.parse().unwrap()).collect()
    }

    #[test]
    fn plaintext_is_refused_off_loopback_unless_tls_or_the_explicit_flag() {
        let local = addrs(&["127.0.0.1:8788", "[::1]:8789"]);
        assert_eq!(check_exposure(&local, false, false), Ok(()));
        for exposed in [addrs(&["0.0.0.0:8788"]), addrs(&["[::]:8788"]), addrs(&["127.0.0.1:1", "192.168.1.5:8788"])] {
            let refusal = check_exposure(&exposed, false, false).expect_err("plaintext off loopback");
            assert!(refusal.contains("--tls-cert") && refusal.contains("--insecure-plaintext"), "{refusal}");
            assert_eq!(check_exposure(&exposed, true, false), Ok(()));
            assert_eq!(check_exposure(&exposed, false, true), Ok(()));
        }
    }
}
