// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training across machines on
// networks that cannot be trusted for its clients. If your team needs
// expertise in securing distributed training traffic then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The authenticated, encrypted channel every rank of a job talks over.
//!
//! The collective moves gradients and weights. A host that could join it
//! could poison the model with its own gradients; a host that could read it
//! could take the weights. So no byte of the collective crosses the wire
//! unauthenticated or in the clear:
//!
//! 1. **Mutual proof of the job secret.** The coordinator sends a fresh random
//!    nonce; the worker answers with its own nonce and an HMAC-SHA256 over
//!    both under the secret; the coordinator answers with its own HMAC over
//!    both, under a different label. Each side checks the other's tag in
//!    constant time before anything else is read, so a peer without the
//!    secret learns nothing and is dropped after a fixed-size exchange.
//! 2. **Per-connection keys.** Two ChaCha20-Poly1305 keys, one per direction,
//!    are derived by HKDF-SHA256 from the secret salted with both nonces. A
//!    recorded session never decrypts under another one's keys.
//! 3. **Sealed, ordered frames.** Every frame is `u32` ciphertext length, then
//!    the ciphertext and tag. The nonce is the frame's sequence number in its
//!    direction, and the length is authenticated, so a frame that is
//!    altered, replayed, reordered or dropped fails to open and ends the
//!    connection.
//!
//! The secret is shared out of band by whoever launches the job (one file
//! every rank reads), so it never appears in the plan, on the wire or in a
//! checkpoint.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::Path;

use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey, CHACHA20_POLY1305};
use ring::rand::{SecureRandom, SystemRandom};
use ring::{hkdf, hmac};

/// Bytes in a job secret.
pub const SECRET_BYTES: usize = 32;

/// What the coordinator sends first, so a stray client of another protocol
/// fails at once instead of being read as a nonce.
const MAGIC: &[u8; 8] = b"brainjc1";
const NONCE_BYTES: usize = 32;
const TAG_BYTES: usize = 32;
const WORKER_LABEL: &[u8] = b"brain collective v1: worker proves the job secret";
const COORDINATOR_LABEL: &[u8] = b"brain collective v1: coordinator proves the job secret";
const COORDINATOR_TO_WORKER: &[u8] = b"brain collective v1: coordinator to worker";
const WORKER_TO_COORDINATOR: &[u8] = b"brain collective v1: worker to coordinator";

/// The secret every rank of one job holds. It never leaves the process: its
/// `Debug` is redacted and nothing serializes it.
#[derive(Clone)]
pub struct JobSecret([u8; SECRET_BYTES]);

impl std::fmt::Debug for JobSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JobSecret(..)")
    }
}

impl JobSecret {
    /// A fresh secret from the operating system's random source.
    pub fn random() -> JobSecret {
        let mut bytes = [0u8; SECRET_BYTES];
        SystemRandom::new().fill(&mut bytes).expect("the system random source is available");
        JobSecret(bytes)
    }

    /// The secret spelled as [`JobSecret::to_hex`] spells it.
    ///
    /// # Errors
    ///
    /// Anything but exactly 64 hexadecimal digits (surrounding whitespace aside).
    pub fn from_hex(text: &str) -> Result<JobSecret, String> {
        let text = text.trim();
        if text.len() != SECRET_BYTES * 2 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("a job secret is {} hexadecimal digits", SECRET_BYTES * 2));
        }
        let mut bytes = [0u8; SECRET_BYTES];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("checked hexadecimal");
        }
        Ok(JobSecret(bytes))
    }

    /// The secret as 64 lowercase hexadecimal digits, for the file a launcher
    /// hands every rank.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Read the secret file a launcher wrote (`openssl rand -hex 32`).
    ///
    /// # Errors
    ///
    /// An unreadable file, one that is not a secret, or on Unix one that users
    /// other than its owner may read: a secret anyone on the machine can read
    /// lets anyone on the machine join the job.
    pub fn read(path: &Path) -> Result<JobSecret, String> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path).map_err(|e| format!("job secret {}: {e}", path.display()))?.permissions().mode();
            if mode & 0o077 != 0 {
                return Err(format!("job secret {} is readable by other users (mode {:o}); chmod 600 it", path.display(), mode & 0o777));
            }
        }
        let text = std::fs::read_to_string(path).map_err(|e| format!("job secret {}: {e}", path.display()))?;
        JobSecret::from_hex(&text).map_err(|e| format!("job secret {}: {e}", path.display()))
    }

    /// Write the secret to a new owner-only (`0600`) file at `path`, the form
    /// [`JobSecret::read`] accepts. An existing file is not overwritten.
    ///
    /// # Errors
    ///
    /// The file exists or cannot be created.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(path).map_err(|e| format!("job secret {}: {e}", path.display()))?;
        file.write_all(format!("{}\n", self.to_hex()).as_bytes()).map_err(|e| format!("job secret {}: {e}", path.display()))
    }

    fn proof(&self, label: &[u8], coordinator_nonce: &[u8], worker_nonce: &[u8]) -> hmac::Tag {
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.0);
        let mut context = hmac::Context::with_key(&key);
        context.update(label);
        context.update(coordinator_nonce);
        context.update(worker_nonce);
        context.sign()
    }

    fn verify(&self, label: &[u8], coordinator_nonce: &[u8], worker_nonce: &[u8], tag: &[u8]) -> io::Result<()> {
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.0);
        let mut message = Vec::with_capacity(label.len() + 2 * NONCE_BYTES);
        message.extend_from_slice(label);
        message.extend_from_slice(coordinator_nonce);
        message.extend_from_slice(worker_nonce);
        hmac::verify(&key, &message, tag).map_err(|_| unauthenticated("the peer does not hold the job secret"))
    }

    fn direction_key(&self, coordinator_nonce: &[u8], worker_nonce: &[u8], direction: &[u8]) -> LessSafeKey {
        let mut salt_bytes = Vec::with_capacity(2 * NONCE_BYTES);
        salt_bytes.extend_from_slice(coordinator_nonce);
        salt_bytes.extend_from_slice(worker_nonce);
        let salt = hkdf::Salt::new(hkdf::HKDF_SHA256, &salt_bytes);
        let info = [direction];
        let prk = salt.extract(&self.0);
        let okm = prk.expand(&info, &CHACHA20_POLY1305).expect("a ChaCha20-Poly1305 key is a valid HKDF length");
        LessSafeKey::new(UnboundKey::from(okm))
    }
}

fn unauthenticated(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, what.to_string())
}

fn fresh_nonce() -> [u8; NONCE_BYTES] {
    let mut nonce = [0u8; NONCE_BYTES];
    SystemRandom::new().fill(&mut nonce).expect("the system random source is available");
    nonce
}

/// One direction's frame counter, spelled as an AEAD nonce.
fn frame_nonce(sequence: u64) -> Nonce {
    let mut nonce = [0u8; aead::NONCE_LEN];
    nonce[4..].copy_from_slice(&sequence.to_le_bytes());
    Nonce::assume_unique_for_key(nonce)
}

/// A TCP stream whose every frame is sealed with the job's per-connection keys.
pub struct SealedStream {
    stream: TcpStream,
    seal: LessSafeKey,
    open: LessSafeKey,
    sent: u64,
    received: u64,
}

impl std::fmt::Debug for SealedStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SealedStream").field("peer", &self.stream.peer_addr().ok()).field("sent", &self.sent).field("received", &self.received).finish()
    }
}

impl SealedStream {
    /// The coordinator's side of the handshake on an accepted `stream`. The
    /// caller sets the stream's timeouts first: the exchange is fixed-size, so
    /// they bound everything a peer without the secret can hold.
    ///
    /// # Errors
    ///
    /// The peer does not hold `secret` (`PermissionDenied`), or the stream fails.
    pub fn accept(mut stream: TcpStream, secret: &JobSecret) -> io::Result<SealedStream> {
        let coordinator_nonce = fresh_nonce();
        let mut greeting = Vec::with_capacity(MAGIC.len() + NONCE_BYTES);
        greeting.extend_from_slice(MAGIC);
        greeting.extend_from_slice(&coordinator_nonce);
        stream.write_all(&greeting)?;
        stream.flush()?;
        let mut answer = [0u8; NONCE_BYTES + TAG_BYTES];
        stream.read_exact(&mut answer)?;
        let (worker_nonce, worker_tag) = answer.split_at(NONCE_BYTES);
        secret.verify(WORKER_LABEL, &coordinator_nonce, worker_nonce, worker_tag)?;
        stream.write_all(secret.proof(COORDINATOR_LABEL, &coordinator_nonce, worker_nonce).as_ref())?;
        stream.flush()?;
        Ok(SealedStream {
            seal: secret.direction_key(&coordinator_nonce, worker_nonce, COORDINATOR_TO_WORKER),
            open: secret.direction_key(&coordinator_nonce, worker_nonce, WORKER_TO_COORDINATOR),
            stream,
            sent: 0,
            received: 0,
        })
    }

    /// The worker's side of the handshake on a `stream` connected to the
    /// coordinator. The caller sets the stream's timeouts first.
    ///
    /// # Errors
    ///
    /// The other end is not a brain coordinator, does not hold `secret`
    /// (`PermissionDenied`), or the stream fails.
    pub fn connect(mut stream: TcpStream, secret: &JobSecret) -> io::Result<SealedStream> {
        let mut greeting = [0u8; MAGIC.len() + NONCE_BYTES];
        stream.read_exact(&mut greeting)?;
        let (magic, coordinator_nonce) = greeting.split_at(MAGIC.len());
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "the coordinator address does not answer as a brain job coordinator"));
        }
        let worker_nonce = fresh_nonce();
        let mut answer = Vec::with_capacity(NONCE_BYTES + TAG_BYTES);
        answer.extend_from_slice(&worker_nonce);
        answer.extend_from_slice(secret.proof(WORKER_LABEL, coordinator_nonce, &worker_nonce).as_ref());
        stream.write_all(&answer)?;
        stream.flush()?;
        let mut coordinator_tag = [0u8; TAG_BYTES];
        stream.read_exact(&mut coordinator_tag)?;
        secret.verify(COORDINATOR_LABEL, coordinator_nonce, &worker_nonce, &coordinator_tag)?;
        Ok(SealedStream {
            seal: secret.direction_key(coordinator_nonce, &worker_nonce, WORKER_TO_COORDINATOR),
            open: secret.direction_key(coordinator_nonce, &worker_nonce, COORDINATOR_TO_WORKER),
            stream,
            sent: 0,
            received: 0,
        })
    }

    /// The underlying stream, for timeouts and socket options.
    pub fn tcp(&self) -> &TcpStream {
        &self.stream
    }

    /// Seal `plaintext` as the next frame and send it.
    ///
    /// # Errors
    ///
    /// A frame too long for its `u32` length, or a stream failure.
    pub fn send(&mut self, plaintext: &[u8]) -> io::Result<()> {
        let sealed_len = plaintext.len() + CHACHA20_POLY1305.tag_len();
        let len = u32::try_from(sealed_len).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, format!("a {}-byte frame does not fit the wire's u32 length", plaintext.len())))?;
        let header = len.to_le_bytes();
        let mut frame = Vec::with_capacity(header.len() + sealed_len);
        frame.extend_from_slice(&header);
        frame.extend_from_slice(plaintext);
        let tag = self
            .seal
            .seal_in_place_separate_tag(frame_nonce(self.sent), Aad::from(header), &mut frame[header.len()..])
            .map_err(|_| io::Error::other("sealing a frame failed"))?;
        frame.extend_from_slice(tag.as_ref());
        self.sent += 1;
        self.stream.write_all(&frame)?;
        self.stream.flush()
    }

    /// Receive and open the next frame, refusing one whose plaintext would be
    /// longer than `max_plaintext` before allocating for it.
    ///
    /// # Errors
    ///
    /// A frame over the bound or failing authentication (`InvalidData`: it was
    /// altered, replayed or reordered, or the peer is not who the handshake
    /// proved), or a stream failure.
    pub fn recv(&mut self, max_plaintext: usize) -> io::Result<Vec<u8>> {
        let mut header = [0u8; 4];
        self.stream.read_exact(&mut header)?;
        let sealed_len = u32::from_le_bytes(header) as usize;
        let tag_len = CHACHA20_POLY1305.tag_len();
        if sealed_len < tag_len || sealed_len - tag_len > max_plaintext {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("a sealed frame claims {sealed_len} bytes, outside 16..={} for this message", max_plaintext + tag_len)));
        }
        let mut sealed = vec![0u8; sealed_len];
        self.stream.read_exact(&mut sealed)?;
        let plaintext_len = self
            .open
            .open_in_place(frame_nonce(self.received), Aad::from(header), &mut sealed)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "a frame failed authentication (altered, replayed, reordered, or not from the authenticated peer)"))?
            .len();
        self.received += 1;
        sealed.truncate(plaintext_len);
        Ok(sealed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    /// A coordinator holding `coordinator` and a worker holding `worker`,
    /// handshaking over loopback: the two sides' results.
    fn handshake(coordinator: JobSecret, worker: JobSecret) -> (io::Result<SealedStream>, io::Result<SealedStream>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker_side = thread::spawn(move || SealedStream::connect(TcpStream::connect(addr).unwrap(), &worker));
        let (stream, _) = listener.accept().unwrap();
        let coordinator_side = SealedStream::accept(stream, &coordinator);
        (coordinator_side, worker_side.join().unwrap())
    }

    #[test]
    fn a_secret_round_trips_its_hex_and_refuses_anything_else() {
        let secret = JobSecret::random();
        assert_eq!(JobSecret::from_hex(&format!("  {}\n", secret.to_hex())).unwrap().0, secret.0);
        for bad in ["", "00", &"g".repeat(64), &"0".repeat(63), &"0".repeat(65)] {
            assert!(JobSecret::from_hex(bad).is_err(), "{bad:?}");
        }
        assert_eq!(format!("{secret:?}"), "JobSecret(..)", "the secret never reaches a log");
    }

    #[test]
    fn a_secret_file_is_written_owner_only_and_refused_when_others_may_read_it() {
        let dir = std::env::temp_dir().join(format!("brain-secret-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("job.secret");
        let _ = std::fs::remove_file(&path);
        let secret = JobSecret::random();
        secret.write(&path).unwrap();
        assert_eq!(JobSecret::read(&path).unwrap().0, secret.0);
        assert!(secret.write(&path).is_err(), "an existing secret is never overwritten");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
            let refusal = JobSecret::read(&path).unwrap_err();
            assert!(refusal.contains("readable by other users"), "{refusal}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn peers_with_the_same_secret_exchange_frames_both_ways() {
        let secret = JobSecret::random();
        let (coordinator, worker) = handshake(secret.clone(), secret);
        let (mut coordinator, mut worker) = (coordinator.unwrap(), worker.unwrap());
        for round in 0..3u8 {
            worker.send(&[round; 100]).unwrap();
            assert_eq!(coordinator.recv(100).unwrap(), vec![round; 100]);
            coordinator.send(&[round ^ 0xff; 7]).unwrap();
            assert_eq!(worker.recv(7).unwrap(), vec![round ^ 0xff; 7]);
        }
    }

    #[test]
    fn a_peer_without_the_secret_is_refused_on_both_sides() {
        let (coordinator, worker) = handshake(JobSecret::random(), JobSecret::random());
        assert_eq!(coordinator.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        // The coordinator hung up without proving itself.
        assert!(worker.is_err());
    }

    #[test]
    fn a_coordinator_without_the_secret_is_refused_by_the_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let secret = JobSecret::random();
        let worker = thread::spawn(move || SealedStream::connect(TcpStream::connect(addr).unwrap(), &secret));
        // An impostor that greets correctly, then answers with a made-up tag.
        let (mut impostor, _) = listener.accept().unwrap();
        impostor.write_all(MAGIC).unwrap();
        impostor.write_all(&[7u8; NONCE_BYTES]).unwrap();
        let mut answer = [0u8; NONCE_BYTES + TAG_BYTES];
        impostor.read_exact(&mut answer).unwrap();
        impostor.write_all(&[0u8; TAG_BYTES]).unwrap();
        assert_eq!(worker.join().unwrap().unwrap_err().kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn an_altered_or_replayed_frame_fails_to_open() {
        let secret = JobSecret::random();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker_secret = secret.clone();
        let worker = thread::spawn(move || {
            let mut worker = SealedStream::connect(TcpStream::connect(addr).unwrap(), &worker_secret).unwrap();
            worker.send(b"gradient").unwrap();
            worker.send(b"gradient").unwrap();
        });
        let (stream, _) = listener.accept().unwrap();
        let mut raw = stream.try_clone().unwrap();
        let mut coordinator = SealedStream::accept(stream, &secret).unwrap();
        worker.join().unwrap();
        // Take the first frame off the wire, flip one ciphertext bit, and
        // hand it to a second reader keyed like the first: it must not open.
        let mut first = vec![0u8; 4 + 8 + 16];
        raw.read_exact(&mut first).unwrap();
        let mut tampered = first.clone();
        tampered[4] ^= 1;
        let opened = open_frame(&coordinator, &tampered, 0);
        assert!(opened.is_err(), "a flipped bit must not open");
        assert_eq!(open_frame(&coordinator, &first, 0).unwrap(), b"gradient");
        // The same bytes again, as the second frame, are a replay.
        assert!(open_frame(&coordinator, &first, 1).is_err(), "a replayed frame must not open");
        // The real second frame is still in order on the stream.
        assert_eq!(coordinator.received, 0);
        coordinator.received = 1;
        assert_eq!(coordinator.recv(8).unwrap(), b"gradient");
    }

    /// Open `frame` (header included) as frame number `sequence` with
    /// `stream`'s receive key, without touching the stream.
    fn open_frame(stream: &SealedStream, frame: &[u8], sequence: u64) -> Result<Vec<u8>, ring::error::Unspecified> {
        let (header, sealed) = frame.split_at(4);
        let mut sealed = sealed.to_vec();
        let header: [u8; 4] = header.try_into().unwrap();
        let len = stream.open.open_in_place(frame_nonce(sequence), Aad::from(header), &mut sealed)?.len();
        sealed.truncate(len);
        Ok(sealed)
    }

    #[test]
    fn a_frame_over_the_bound_is_refused_before_it_is_read() {
        let secret = JobSecret::random();
        let (coordinator, worker) = handshake(secret.clone(), secret);
        let (mut coordinator, mut worker) = (coordinator.unwrap(), worker.unwrap());
        worker.send(&[0u8; 64]).unwrap();
        assert_eq!(coordinator.recv(63).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }
}
