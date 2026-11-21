// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for putting one inference endpoint
// in front of every card of a cluster for its clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! One upstream `brain serve` endpoint: its address, and the client that reaches it.
//!
//! A request opens its own connection ([`Connector::open`]). That costs a TCP (and TLS) handshake per request, small
//! beside a generation, and buys the one property the router's retry rule rests on: a failure to open the connection
//! is a request that was never sent, so it may go to another upstream, while any failure after it opened may have run
//! and is never sent again. A pooled connection that turns out dead at write time is ambiguous, which is why there is
//! no pool.

use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, HeaderValue, Method, Request, Response, StatusCode};
use hyper::body::Incoming;
use hyper::client::conn::http1::SendRequest;
use hyper_util::rt::TokioIo;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;

/// The longest upstream URL accepted.
const MAX_URL_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    Http,
    Https,
}

/// The base URL of one upstream: `http[s]://HOST[:PORT]`, nothing else.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamUrl {
    scheme: Scheme,
    host: String,
    port: u16,
}

impl UpstreamUrl {
    /// Parse `text`. Only a scheme, a host and a port are accepted: a path, query, fragment or credentials in the URL
    /// would be forwarded nowhere or leak somewhere, so they are refused rather than ignored.
    ///
    /// # Errors
    /// A malformed URL, another scheme, or any part beyond scheme, host and port.
    pub fn parse(text: &str) -> Result<UpstreamUrl, String> {
        let text = text.trim();
        if text.len() > MAX_URL_BYTES {
            return Err(format!("an upstream URL is at most {MAX_URL_BYTES} bytes"));
        }
        let url = url::Url::parse(text).map_err(|e| format!("upstream {text:?}: {e}"))?;
        let scheme = match url.scheme() {
            "http" => Scheme::Http,
            "https" => Scheme::Https,
            other => return Err(format!("upstream {text:?}: scheme {other:?} is not http or https")),
        };
        if !url.username().is_empty() || url.password().is_some() {
            return Err(format!("upstream {text:?}: credentials do not belong in the URL; pass --upstream-key-file"));
        }
        if url.query().is_some() || url.fragment().is_some() || !matches!(url.path(), "" | "/") {
            return Err(format!("upstream {text:?}: give the server's base URL (scheme, host and port) with no path, query or fragment"));
        }
        let host = match url.host() {
            Some(url::Host::Domain(d)) => d.to_string(),
            Some(url::Host::Ipv4(ip)) => ip.to_string(),
            Some(url::Host::Ipv6(ip)) => ip.to_string(),
            None => return Err(format!("upstream {text:?} names no host")),
        };
        let port = url.port_or_known_default().ok_or_else(|| format!("upstream {text:?} names no port"))?;
        Ok(UpstreamUrl { scheme, host, port })
    }

    /// Whether the connection stays on this machine.
    pub fn is_loopback(&self) -> bool {
        match self.host.parse::<IpAddr>() {
            Ok(ip) => ip.is_loopback(),
            Err(_) => self.host.eq_ignore_ascii_case("localhost"),
        }
    }

    /// Whether the connection is TLS.
    pub fn is_tls(&self) -> bool {
        self.scheme == Scheme::Https
    }

    /// The `Host` header value.
    fn authority(&self) -> String {
        match self.host.parse::<IpAddr>() {
            Ok(IpAddr::V6(ip)) => format!("[{ip}]:{}", self.port),
            _ => format!("{}:{}", self.host, self.port),
        }
    }
}

impl std::fmt::Display for UpstreamUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let scheme = if self.is_tls() { "https" } else { "http" };
        write!(f, "{scheme}://{}", self.authority())
    }
}

/// Refuse a plaintext upstream off this machine unless the operator said the path is already protected: the router
/// sends the upstream key in every request.
///
/// # Errors
/// The first `http://` upstream that is not loopback, without `insecure_plaintext`.
pub fn check_upstream_exposure(upstreams: &[UpstreamUrl], insecure_plaintext: bool) -> Result<(), String> {
    if insecure_plaintext {
        return Ok(());
    }
    match upstreams.iter().find(|u| !u.is_tls() && !u.is_loopback()) {
        None => Ok(()),
        Some(u) => Err(format!(
            "upstream {u} is plaintext off this machine, so the upstream key and every request would cross the network readable; \
             serve the upstream over TLS (https:// with --upstream-ca <pem>), or pass --insecure-plaintext when the network path is already protected"
        )),
    }
}

/// The certificate authorities an `https://` upstream's certificate must chain to.
#[derive(Clone)]
pub struct UpstreamTls {
    connector: tokio_rustls::TlsConnector,
}

impl std::fmt::Debug for UpstreamTls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UpstreamTls { .. }")
    }
}

impl UpstreamTls {
    /// Trust the PEM certificates in `ca`.
    ///
    /// # Errors
    /// A file that cannot be read or holds no usable certificate.
    pub fn from_pem_file(ca: &Path) -> Result<UpstreamTls, String> {
        let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(ca)
            .map_err(|e| format!("upstream CA {}: {e}", ca.display()))?
            .collect::<Result<_, _>>()
            .map_err(|e| format!("upstream CA {}: {e}", ca.display()))?;
        let mut roots = rustls::RootCertStore::empty();
        let (added, _) = roots.add_parsable_certificates(certs);
        if added == 0 {
            return Err(format!("upstream CA {} holds no usable PEM certificate", ca.display()));
        }
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("TLS configuration: {e}"))?
            .with_root_certificates(roots)
            .with_no_client_auth();
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(UpstreamTls { connector: tokio_rustls::TlsConnector::from(Arc::new(config)) })
    }
}

trait Io: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Io for T {}

/// Aborts the task driving a connection when dropped: a client that hangs up closes the upstream connection, which
/// cancels the upstream's generation.
pub(crate) struct ConnectionTask(tokio::task::AbortHandle);

impl Drop for ConnectionTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// An open connection to one upstream, ready for one request.
pub(crate) struct Connection {
    sender: SendRequest<Body>,
    task: ConnectionTask,
    authority: String,
}

/// Opening a connection failed: nothing was sent.
#[derive(Debug)]
pub(crate) struct NotSent(pub String);

/// Opens connections to upstreams.
#[derive(Clone)]
pub(crate) struct Connector {
    tls: Option<UpstreamTls>,
    connect_timeout: Duration,
}

impl Connector {
    pub(crate) fn new(tls: Option<UpstreamTls>, connect_timeout: Duration) -> Connector {
        Connector { tls, connect_timeout }
    }

    /// Connect to `url` and complete the TLS and HTTP handshakes within the connect timeout. An error here means no
    /// byte of a request left this process.
    pub(crate) async fn open(&self, url: &UpstreamUrl) -> Result<Connection, NotSent> {
        let opened = tokio::time::timeout(self.connect_timeout, self.open_inner(url)).await;
        opened.unwrap_or_else(|_| Err(NotSent(format!("no connection within {:?}", self.connect_timeout))))
    }

    async fn open_inner(&self, url: &UpstreamUrl) -> Result<Connection, NotSent> {
        let tcp = TcpStream::connect((url.host.as_str(), url.port)).await.map_err(|e| NotSent(format!("connect: {e}")))?;
        tcp.set_nodelay(true).ok();
        let io: Box<dyn Io> = match (url.scheme, &self.tls) {
            (Scheme::Http, _) => Box::new(tcp),
            (Scheme::Https, Some(tls)) => {
                let name = ServerName::try_from(url.host.clone()).map_err(|e| NotSent(format!("TLS name: {e}")))?;
                Box::new(tls.connector.connect(name, tcp).await.map_err(|e| NotSent(format!("TLS handshake: {e}")))?)
            }
            (Scheme::Https, None) => return Err(NotSent("an https upstream needs --upstream-ca".into())),
        };
        let (sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(io)).await.map_err(|e| NotSent(format!("HTTP handshake: {e}")))?;
        let task = tokio::spawn(async move {
            let _ = conn.await;
        });
        Ok(Connection { sender, task: ConnectionTask(task.abort_handle()), authority: url.authority() })
    }
}

impl Connection {
    /// Send one request. `body` is sent whole with its length; only the headers set here reach the upstream.
    pub(crate) async fn send(mut self, method: Method, path: &str, key: Option<&str>, accept: Option<&HeaderValue>, body: bytes::Bytes) -> Result<(Response<Incoming>, ConnectionTask), String> {
        let mut req = Request::builder().method(method).uri(path).header(header::HOST, self.authority.as_str());
        if !body.is_empty() {
            req = req.header(header::CONTENT_TYPE, "application/json").header(header::CONTENT_LENGTH, body.len());
        }
        if let Some(key) = key {
            req = req.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        if let Some(accept) = accept {
            req = req.header(header::ACCEPT, accept.clone());
        }
        let req = req.body(Body::from(body)).map_err(|e| format!("request: {e}"))?;
        let response = self.sender.send_request(req).await.map_err(|e| format!("upstream: {e}"))?;
        Ok((response, self.task))
    }
}

/// `GET path` on `url` and its JSON body, read up to `limit` bytes, within `timeout`.
pub(crate) async fn get_json(connector: &Connector, url: &UpstreamUrl, path: &str, key: Option<&str>, limit: usize, timeout: Duration) -> Result<serde_json::Value, String> {
    let fetch = async {
        let conn = connector.open(url).await.map_err(|e| e.0)?;
        let (response, _task) = conn.send(Method::GET, path, key, None, bytes::Bytes::new()).await?;
        if response.status() != StatusCode::OK {
            return Err(format!("{path}: HTTP {}", response.status()));
        }
        let body = axum::body::to_bytes(Body::new(response.into_body()), limit).await.map_err(|e| format!("{path}: {e}"))?;
        serde_json::from_slice(&body).map_err(|e| format!("{path}: {e}"))
    };
    tokio::time::timeout(timeout, fetch).await.unwrap_or_else(|_| Err(format!("{path}: no answer within {timeout:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_upstream_url_is_a_scheme_a_host_and_a_port_only() {
        let u = UpstreamUrl::parse("http://10.0.0.7:8790").expect("plain");
        assert_eq!(u.to_string(), "http://10.0.0.7:8790");
        assert!(!u.is_loopback() && !u.is_tls());
        assert_eq!(UpstreamUrl::parse("https://node3/").expect("default port").to_string(), "https://node3:443");
        assert!(UpstreamUrl::parse("http://[::1]:9000").expect("v6").is_loopback());
        assert!(UpstreamUrl::parse("http://localhost:9000").expect("name").is_loopback());
        for bad in ["ftp://h:1", "http://user:pw@h:1", "http://h:1/v1", "http://h:1/?a=b", "http://h:1/#x", "h:1", "", "http://", "unix:/run/x"] {
            assert!(UpstreamUrl::parse(bad).is_err(), "{bad:?} must be refused");
        }
        assert!(UpstreamUrl::parse(&format!("http://{}:1", "a".repeat(600))).is_err());
    }

    #[test]
    fn a_plaintext_upstream_off_this_machine_needs_the_explicit_flag() {
        let local = [UpstreamUrl::parse("http://127.0.0.1:1").unwrap(), UpstreamUrl::parse("https://far:1").unwrap()];
        assert!(check_upstream_exposure(&local, false).is_ok());
        let far = [UpstreamUrl::parse("http://10.1.2.3:1").unwrap()];
        let refusal = check_upstream_exposure(&far, false).expect_err("plaintext off loopback");
        assert!(refusal.contains("--insecure-plaintext") && refusal.contains("--upstream-ca"), "{refusal}");
        assert!(check_upstream_exposure(&far, true).is_ok());
    }
}
