// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A minimal RTSP server for one live H.264 stream (RFC 2326 / RFC 6184).
//!
//! Two parties touch it. The video thread calls [`RtspServer::publish`] with
//! each encoded access unit: the unit is packetized once and written straight
//! to every playing client's socket, so "sent" means "written to the kernel"
//! with no queue in between. A control thread accepts connections and answers
//! `OPTIONS`, `DESCRIBE`, `SETUP`, `PLAY`, `GET_PARAMETER` and `TEARDOWN`.
//!
//! Scope: RTP is carried over the control TCP connection (interleaved), which
//! is the one transport that works through any firewall and needs no extra
//! sockets on a board with 11 MB of free RAM. A client asking for UDP gets
//! `461 Unsupported Transport` and, like ffmpeg, retries with TCP. A client
//! that cannot take data within the write timeout is dropped rather than
//! allowed to stall the video path.
//!
//! Swedish Embedded AB implements low-latency video streaming for embedded
//! Linux boards for its clients. If your team needs expertise in RTSP
//! servers then you can procure our services by sending an email to
//! info@swedishembedded.com.

pub mod proto;
pub mod rtp;

use crate::error::{Context, Result};
use proto::{Parsed, ParameterSets, Request, Transport};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

const MAX_CLIENTS: usize = 4;
/// A client that cannot absorb a frame within this long is disconnected.
const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
/// How often the control thread wakes to notice shutdown and dead clients.
const POLL_MS: i32 = 100;
const RTP_PAYLOAD_TYPE: u8 = 96;
const SUPPORTED_METHODS: &str = "OPTIONS, DESCRIBE, SETUP, PLAY, GET_PARAMETER, TEARDOWN";

#[derive(Clone, Debug)]
pub struct RtspConfig {
    pub port: u16,
    /// Mount point, e.g. `/live/0`.
    pub path: String,
    pub fps: u32,
}

/// One connected viewer.
struct Client {
    wire: Mutex<TcpStream>,
    session: String,
    playing: AtomicBool,
    dead: AtomicBool,
    channel: AtomicU8,
}

impl Client {
    fn wire(&self) -> MutexGuard<'_, TcpStream> {
        self.wire.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn kill(&self) {
        self.dead.store(true, Ordering::Release);
        let _ = self.wire().shutdown(Shutdown::Both);
    }
}

struct Shared {
    cfg: RtspConfig,
    stop: AtomicBool,
    idr_wanted: AtomicBool,
    clients: Mutex<Vec<Arc<Client>>>,
    sets: Mutex<Option<ParameterSets>>,
}

/// Packetizer state, used only by the thread that publishes frames.
struct Publisher {
    packetizer: rtp::Packetizer,
    packets: rtp::Packets,
    channel: u8,
}

pub struct RtspServer {
    shared: Arc<Shared>,
    publisher: Mutex<Publisher>,
    control: Option<JoinHandle<()>>,
}

/// What one [`RtspServer::publish`] call did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Published {
    pub clients: usize,
    pub bytes: usize,
}

impl RtspServer {
    pub fn start(cfg: RtspConfig) -> Result<RtspServer> {
        let listener = TcpListener::bind(("0.0.0.0", cfg.port)).context(format!("listening on port {}", cfg.port))?;
        let shared = Arc::new(Shared {
            cfg,
            stop: AtomicBool::new(false),
            idr_wanted: AtomicBool::new(false),
            clients: Mutex::new(Vec::new()),
            sets: Mutex::new(None),
        });
        let worker = Arc::clone(&shared);
        let control = std::thread::Builder::new()
            .name("rtsp-control".into())
            .spawn(move || control_loop(listener, worker))
            .context("starting the RTSP control thread")?;
        let ssrc = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x5eed_0001, |d| d.subsec_nanos() ^ 0x5eed_0000);
        Ok(RtspServer {
            shared,
            publisher: Mutex::new(Publisher {
                packetizer: rtp::Packetizer::new(RTP_PAYLOAD_TYPE, ssrc),
                packets: rtp::Packets::default(),
                channel: 0,
            }),
            control: Some(control),
        })
    }

    /// True once after a viewer started playing: the encoder should then emit
    /// an IDR frame so that viewer can start decoding immediately.
    pub fn take_idr_request(&self) -> bool {
        self.shared.idr_wanted.swap(false, Ordering::AcqRel)
    }

    /// Packetizes `au` (Annex B) and writes it to every playing client.
    /// `pts_us` is the capture time; it becomes the RTP timestamp.
    pub fn publish(&self, au: &[u8], pts_us: u64) -> Published {
        self.remember_parameter_sets(au);
        let playing: Vec<Arc<Client>> = {
            let clients = self.shared.clients.lock().unwrap_or_else(|e| e.into_inner());
            clients.iter().filter(|c| c.playing.load(Ordering::Acquire) && !c.dead.load(Ordering::Acquire)).cloned().collect()
        };
        if playing.is_empty() {
            return Published::default();
        }
        let mut p = self.publisher.lock().unwrap_or_else(|e| e.into_inner());
        let p = &mut *p;
        p.packets.clear();
        // 90 kHz clock; the u32 wraps by design (RFC 3550 section 5.1).
        let timestamp = (pts_us.wrapping_mul(9) / 100) as u32;
        p.packetizer.packetize(au, timestamp, &mut p.packets);

        let mut sent = Published::default();
        for client in playing {
            let channel = client.channel.load(Ordering::Acquire);
            if channel != p.channel {
                p.packets.set_channel(channel);
                p.channel = channel;
            }
            let result = client.wire().write_all(p.packets.bytes());
            match result {
                Ok(()) => {
                    sent.clients += 1;
                    sent.bytes += p.packets.bytes().len();
                }
                Err(e) => {
                    eprintln!("rtsp: dropping client {}: {e}", client.session);
                    client.kill();
                }
            }
        }
        sent
    }

    /// Records SPS and PPS from an access unit that carries them, so DESCRIBE
    /// can advertise them to clients that connect later.
    fn remember_parameter_sets(&self, au: &[u8]) {
        let starts_with_sps = au.len() > 4 && au[..3] == [0, 0, 0] && au[3] == 1 && au[4] & 0x1f == 7
            || au.len() > 3 && au[..3] == [0, 0, 1] && au[3] & 0x1f == 7;
        if !starts_with_sps {
            return;
        }
        let nals = rtp::split_nals(au);
        let sps = nals.iter().find(|n| n[0] & 0x1f == 7);
        let pps = nals.iter().find(|n| n[0] & 0x1f == 8);
        if let (Some(sps), Some(pps)) = (sps, pps) {
            let sets = ParameterSets { sps: sps.to_vec(), pps: pps.to_vec() };
            let mut slot = self.shared.sets.lock().unwrap_or_else(|e| e.into_inner());
            if slot.as_ref() != Some(&sets) {
                *slot = Some(sets);
            }
        }
    }

    pub fn url_hint(&self) -> String {
        format!("rtsp://<board-ip>:{}{}", self.shared.cfg.port, self.shared.cfg.path)
    }
}

impl Drop for RtspServer {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for c in self.shared.clients.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            c.kill();
        }
        if let Some(t) = self.control.take() {
            let _ = t.join();
        }
    }
}

/// A connection as the control thread sees it: the read side plus buffered
/// bytes not yet forming a complete request.
struct Conn {
    reader: TcpStream,
    client: Arc<Client>,
    inbuf: Vec<u8>,
}

fn control_loop(listener: TcpListener, shared: Arc<Shared>) {
    let mut conns: Vec<Conn> = Vec::new();
    let mut next_session = 0u64;
    while !shared.stop.load(Ordering::Acquire) {
        conns.retain(|c| !c.client.dead.load(Ordering::Acquire));
        shared.clients.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| !c.dead.load(Ordering::Acquire));

        let mut fds = vec![libc::pollfd { fd: listener.as_raw_fd(), events: libc::POLLIN, revents: 0 }];
        fds.extend(conns.iter().map(|c| libc::pollfd { fd: c.reader.as_raw_fd(), events: libc::POLLIN, revents: 0 }));
        // SAFETY: `fds` is a valid array of `fds.len()` pollfd structs.
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, POLL_MS) };
        if ready <= 0 {
            continue;
        }
        if fds[0].revents & libc::POLLIN != 0 {
            match listener.accept() {
                Ok((stream, peer)) => {
                    if conns.len() >= MAX_CLIENTS {
                        let _ = (&stream).write_all(&proto::response("503 Service Unavailable", "0", &[], None));
                    } else {
                        next_session += 1;
                        match new_conn(stream, next_session) {
                            Ok(conn) => {
                                eprintln!("rtsp: client {peer} connected");
                                shared.clients.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::clone(&conn.client));
                                conns.push(conn);
                            }
                            Err(e) => eprintln!("rtsp: rejecting {peer}: {e}"),
                        }
                    }
                }
                Err(e) => eprintln!("rtsp: accept failed: {e}"),
            }
        }
        // A client accepted just above has no poll entry yet; `zip` leaves it
        // for the next round.
        for (conn, polled) in conns.iter_mut().zip(&fds[1..]) {
            if polled.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0 {
                if let Err(e) = service(conn, &shared) {
                    eprintln!("rtsp: closing client {}: {e}", conn.client.session);
                    conn.client.kill();
                }
            }
        }
    }
}

fn new_conn(stream: TcpStream, serial: u64) -> Result<Conn> {
    stream.set_nodelay(true).context("setting TCP_NODELAY")?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT)).context("setting the write timeout")?;
    let reader = stream.try_clone().context("cloning the client socket")?;
    let client = Arc::new(Client {
        wire: Mutex::new(stream),
        session: format!("{:08x}{serial:04x}", std::process::id()),
        playing: AtomicBool::new(false),
        dead: AtomicBool::new(false),
        channel: AtomicU8::new(0),
    });
    Ok(Conn { reader, client, inbuf: Vec::new() })
}

/// Reads what the client sent and answers every complete request in it.
fn service(conn: &mut Conn, shared: &Shared) -> Result<()> {
    let mut chunk = [0u8; 2048];
    // `poll` said the socket is readable, so this does not block.
    let n = conn.reader.read(&mut chunk).context("reading from the client")?;
    if n == 0 {
        conn.client.kill();
        return Ok(());
    }
    conn.inbuf.extend_from_slice(&chunk[..n]);
    loop {
        // Interleaved RTCP from the client: `$`, channel, 16-bit length, data.
        if conn.inbuf.first() == Some(&b'$') {
            if conn.inbuf.len() < 4 {
                return Ok(());
            }
            let len = u16::from_be_bytes([conn.inbuf[2], conn.inbuf[3]]) as usize;
            if conn.inbuf.len() < 4 + len {
                return Ok(());
            }
            conn.inbuf.drain(..4 + len);
            continue;
        }
        match proto::parse_request(&conn.inbuf) {
            Parsed::Incomplete => return Ok(()),
            Parsed::Malformed => return Err(crate::error::Error::new("malformed RTSP request")),
            Parsed::Request(req, used) => {
                conn.inbuf.drain(..used);
                let reply = answer(&req, &conn.client, shared);
                conn.client.wire().write_all(&reply).context("writing the response")?;
                if req.method == "TEARDOWN" {
                    conn.client.kill();
                    return Ok(());
                }
            }
        }
    }
}

/// True if `uri` names the stream or one of its tracks.
fn names_stream(uri: &str, mount: &str) -> bool {
    let path = uri.split_once("://").map_or(uri, |(_, rest)| rest.find('/').map_or("/", |i| &rest[i..]));
    path.strip_prefix(mount).is_some_and(|tail| tail.is_empty() || tail.starts_with('/'))
}

fn answer(req: &Request, client: &Client, shared: &Shared) -> Vec<u8> {
    let session = || ("Session", format!("{};timeout=60", client.session));
    match req.method.as_str() {
        "OPTIONS" => proto::response("200 OK", &req.cseq, &[("Public", SUPPORTED_METHODS.into())], None),
        _ if !names_stream(&req.uri, &shared.cfg.path) => proto::response("404 Not Found", &req.cseq, &[], None),
        "DESCRIBE" => {
            let sets = shared.sets.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let body = proto::sdp("brain", shared.cfg.fps, sets.as_ref());
            let base = format!("{}/", req.uri.trim_end_matches('/'));
            proto::response("200 OK", &req.cseq, &[("Content-Base", base)], Some(("application/sdp", &body)))
        }
        "SETUP" => match req.header("Transport").map(proto::parse_transport) {
            Some(Transport::TcpInterleaved { rtp, rtcp }) => {
                client.channel.store(rtp, Ordering::Release);
                let transport = format!("RTP/AVP/TCP;unicast;interleaved={rtp}-{rtcp}");
                proto::response("200 OK", &req.cseq, &[("Transport", transport), session()], None)
            }
            _ => proto::response("461 Unsupported Transport", &req.cseq, &[], None),
        },
        "PLAY" => {
            client.playing.store(true, Ordering::Release);
            shared.idr_wanted.store(true, Ordering::Release);
            proto::response("200 OK", &req.cseq, &[session(), ("Range", "npt=0.000-".into())], None)
        }
        "GET_PARAMETER" => proto::response("200 OK", &req.cseq, &[session()], None),
        "TEARDOWN" => proto::response("200 OK", &req.cseq, &[session()], None),
        _ => proto::response("405 Method Not Allowed", &req.cseq, &[("Allow", SUPPORTED_METHODS.into())], None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_uris_match_the_mount_point_and_its_tracks_only() {
        assert!(names_stream("rtsp://10.0.0.1:8554/live/0", "/live/0"));
        assert!(names_stream("rtsp://10.0.0.1:8554/live/0/", "/live/0"));
        assert!(names_stream("rtsp://10.0.0.1:8554/live/0/streamid=0", "/live/0"));
        assert!(!names_stream("rtsp://10.0.0.1:8554/live/01", "/live/0"));
        assert!(!names_stream("rtsp://10.0.0.1:8554/other", "/live/0"));
    }
}
