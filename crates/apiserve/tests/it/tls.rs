// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving models over networks
// that cannot be trusted for its clients. If your team needs expertise in
// transport security for inference services then you can procure our services
// by sending an email to info@swedishembedded.com.

//! A surface served with TLS answers a verifying client and nothing in
//! plaintext, and a certificate that cannot be used is refused at load, naming
//! the file.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use apiserve::{serve_all, Provider, ServeOpts, Surface, Tls};
use residency::budget::Budgets;
use residency::{Device, Executor, Policy};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};

const KEY: &str = "sk-brain-test-key";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-tls-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A self-signed certificate for `127.0.0.1` and its key, written into `dir`
/// by the `openssl` CLI; `None` (and a skip line) when it is not installed.
fn self_signed(dir: &Path, name: &str) -> Option<(PathBuf, PathBuf)> {
    let (cert, key) = (dir.join(format!("{name}.crt")), dir.join(format!("{name}.key")));
    let made = Command::new("openssl")
        .args(["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-days", "1", "-subj", "/CN=brain-test"])
        .args(["-addext", "subjectAltName=IP:127.0.0.1", "-addext", "basicConstraints=critical,CA:FALSE", "-addext", "extendedKeyUsage=serverAuth"])
        .arg("-keyout")
        .arg(&key)
        .arg("-out")
        .arg(&cert)
        .output();
    match made {
        Ok(out) if out.status.success() => Some((cert, key)),
        Ok(out) => panic!("openssl could not make a test certificate: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => {
            brain_testutil::skip(&format!("the openssl CLI makes the test certificate and is not runnable here: {e}"));
            None
        }
    }
}

/// A server with no models on a fresh loopback port, serving `tls`.
fn serve(tls: Tls) -> SocketAddr {
    let addr = TcpListener::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr");
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    let exec = Executor::start(Vec::new(), budgets, Policy::default());
    let (trigger, shutdown) = brain_shutdown::channel();
    std::thread::spawn(move || {
        let _keep = trigger;
        let _ = serve_all(exec, vec![Surface::new(Provider::OpenAI, addr, KEY)], ServeOpts::new().with_shutdown(shutdown).with_tls(Some(tls)));
    });
    let until = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(addr).is_err() {
        assert!(Instant::now() < until, "the server never started listening");
        std::thread::sleep(Duration::from_millis(20));
    }
    addr
}

/// `request` over TLS to `addr`, trusting only `cert`; the response text.
fn over_tls(addr: SocketAddr, cert: &Path, request: &str) -> std::io::Result<String> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(CertificateDer::from_pem_file(cert).unwrap()).unwrap();
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connection = rustls::ClientConnection::new(Arc::new(config), ServerName::try_from("127.0.0.1").unwrap().to_owned()).unwrap();
    let socket = TcpStream::connect(addr)?;
    socket.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut stream = rustls::StreamOwned::new(connection, socket);
    stream.write_all(request.as_bytes())?;
    let mut response = Vec::new();
    match stream.read_to_end(&mut response) {
        Ok(_) => {}
        // A peer that closes without close_notify after `Connection: close`.
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && !response.is_empty() => {}
        Err(e) => return Err(e),
    }
    Ok(String::from_utf8_lossy(&response).into_owned())
}

fn models_request() -> String {
    format!("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {KEY}\r\nConnection: close\r\n\r\n")
}

#[test]
fn a_tls_surface_answers_a_verifying_client_and_nothing_in_plaintext() {
    let dir = scratch("serve");
    let Some((cert, key)) = self_signed(&dir, "server") else { return };
    let addr = serve(Tls::from_pem_files(&cert, &key).expect("a valid certificate loads"));

    let response = over_tls(addr, &cert, &models_request()).expect("a TLS request");
    assert!(response.starts_with("HTTP/1.1 200"), "{response:?}");

    // The same request in plaintext never reaches HTTP: no response line.
    let mut plain = TcpStream::connect(addr).unwrap();
    plain.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    plain.write_all(models_request().as_bytes()).unwrap();
    let mut answer = Vec::new();
    let _ = plain.read_to_end(&mut answer);
    assert!(!String::from_utf8_lossy(&answer).contains("HTTP/1.1"), "a plaintext request was answered: {:?}", String::from_utf8_lossy(&answer));

    // A client that trusts a different certificate refuses the server.
    let Some((other, _)) = self_signed(&dir, "other") else { return };
    assert!(over_tls(addr, &other, &models_request()).is_err(), "an unverified server must not be trusted");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_certificate_that_cannot_be_served_is_refused_naming_the_file() {
    let dir = scratch("load");
    let Some((cert, key)) = self_signed(&dir, "a") else { return };
    let Some((_, other_key)) = self_signed(&dir, "b") else { return };

    let missing = dir.join("missing.crt");
    let refusal = Tls::from_pem_files(&missing, &key).unwrap_err();
    assert!(refusal.contains("missing.crt"), "{refusal}");

    let refusal = Tls::from_pem_files(&key, &key).unwrap_err();
    assert!(refusal.contains("a.key"), "a file without a certificate: {refusal}");

    let refusal = Tls::from_pem_files(&cert, &other_key).unwrap_err();
    assert!(refusal.contains("a.crt"), "a key that is not the certificate's: {refusal}");
    assert!(!refusal.contains("PRIVATE KEY"), "key material never reaches a message: {refusal}");
    std::fs::remove_dir_all(&dir).ok();
}
