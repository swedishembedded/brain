// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Collective` over TCP - the multi-machine transport. Same trait, same call
//! sites as [`HostCollective`](crate::collective::HostCollective); swapping this
//! in is what turns single-box multi-GPU training into a cluster, with no change
//! to the drivers, the grid, or any model.
//!
//! Topology is a **coordinator star**: rank 0 binds a socket, ranks `1..world`
//! connect. Every op, the workers send their tensor to the coordinator, which
//! reduces/gathers in a fixed rank order (so results are bit-reproducible, same
//! as the host transport) and sends each rank its result. A star is the simplest
//! correct transport and is ideal for the two workloads that matter here:
//! federated rounds (infrequent, one average per round) and modest cluster sizes.
//! A bandwidth-optimal ring/tree all-reduce is a drop-in replacement behind this
//! same trait when the world grows large - the point of the abstraction.
//!
//! Framing is one dtype tag byte, then length-prefixed little-endian f32 (`u32`
//! count, then the payload), read/written with `read_exact`/`write_all` so
//! partial TCP reads are handled.
//!
//! A wire failure now surfaces as `Err(CollectiveError::Transport(..))` (M7.2
//! gave this trait a real error channel - the "a wire failure can only panic
//! the rank" gap this file used to carry is closed) carrying which peer/op it
//! happened on, same context the old panic message did.
//!
//! What M7.2 does NOT add here: cross-rank validation (dtype/length/world-size
//! agreement) on the WORKER side, or a graceful abort for a coordinator-side
//! validation failure. The coordinator (which alone sees every rank's payload,
//! via [`NetworkCollective::gather_slots`]) DOES run the same
//! [`crate::collective::Payload`] validation [`crate::collective::HostCollective`]
//! does and returns a real `Err` for it - but the wire protocol has no "abort,
//! here's why" frame, so a coordinator-side validation `Err` returns to the
//! coordinator's own caller WITHOUT scattering a result, leaving workers
//! blocked in [`NetworkCollective::round_trip`]'s `recv`. Giving every worker a
//! deterministic `Err` here too needs a wire-protocol extension (an "abort"
//! sentinel workers can distinguish from a real payload), which is real
//! follow-up work, not a trait-shape change - HostCollective, whose ranks all
//! see the same shared memory, has no such gap.
//! (Rank/root range checks are the exception: they're validated on EACH side's
//! own call before any I/O for that op, so an invalid `rank`/`root` argument
//! never even reaches the wire and never strands a peer.)

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::collective::{check_dtypes, check_equal_lengths, check_rank, Collective, CollectiveError, CollectiveFuture, CollectiveResult, Payload};
use gpu_core::select::Dtype;

/// Wire tag for [`Payload::dtype`] - one byte ahead of the length-prefixed f32
/// payload. Declaration order must match [`dtype_from_tag`]; covered by
/// `tests::every_dtype_round_trips_its_wire_tag`.
fn dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::F32 => 0,
        Dtype::F16 => 1,
        Dtype::BF16 => 2,
        Dtype::I8 => 3,
        Dtype::Q4 => 4,
        Dtype::Q4K => 5,
        Dtype::Q8K => 6,
        Dtype::NF4 => 7,
        Dtype::F4E2M1 => 8,
        Dtype::F8E4M3 => 9,
        Dtype::F8E5M2 => 10,
    }
}

fn dtype_from_tag(t: u8) -> Option<Dtype> {
    Some(match t {
        0 => Dtype::F32,
        1 => Dtype::F16,
        2 => Dtype::BF16,
        3 => Dtype::I8,
        4 => Dtype::Q4,
        5 => Dtype::Q4K,
        6 => Dtype::Q8K,
        7 => Dtype::NF4,
        8 => Dtype::F4E2M1,
        9 => Dtype::F8E4M3,
        10 => Dtype::F8E5M2,
        _ => return None,
    })
}

fn send_vec(s: &mut TcpStream, v: &[f32]) -> std::io::Result<()> {
    s.write_all(&(v.len() as u32).to_le_bytes())?;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    s.write_all(&bytes)?;
    s.flush()
}

/// Largest accepted message, in f32 elements (1 GiB of payload). The length
/// prefix arrives off the WIRE: without a bound a corrupt/hostile 4-byte
/// header could demand a 16 GB allocation before any data is read (and
/// `n * 4` itself overflows usize on a 32-bit target).
const MAX_MSG_ELEMS: usize = 1 << 28;

fn recv_vec(s: &mut TcpStream) -> std::io::Result<Vec<f32>> {
    let mut len = [0u8; 4];
    s.read_exact(&mut len)?;
    let n = u32::from_le_bytes(len) as usize;
    if n > MAX_MSG_ELEMS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("collective message claims {n} f32 elements (> {MAX_MSG_ELEMS} cap) - corrupt length prefix or a non-brain peer"),
        ));
    }
    let mut bytes = vec![0u8; n.checked_mul(4).expect("bounded by MAX_MSG_ELEMS")];
    s.read_exact(&mut bytes)?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// The frame tag no dtype uses: an abort carrying a UTF-8 reason. A rank that
/// cannot take part sends one instead of a result, so its peers fail with the
/// reason rather than waiting for a payload that will never come.
const ABORT_TAG: u8 = 255;

/// Largest accepted join announcement, in bytes. The plan is a few hundred
/// bytes; the bound stops a corrupt length prefix from allocating arbitrarily.
pub const MAX_PLAN_BYTES: usize = 1 << 20;

fn send_payload(s: &mut TcpStream, p: &Payload) -> std::io::Result<()> {
    s.write_all(&[dtype_tag(p.dtype)])?;
    send_vec(s, &p.data)
}

fn recv_payload(s: &mut TcpStream) -> std::io::Result<Payload> {
    let mut tag = [0u8; 1];
    s.read_exact(&mut tag)?;
    if tag[0] == ABORT_TAG {
        let reason = read_abort_body(s)?;
        return Err(std::io::Error::other(format!("peer aborted the collective: {reason}")));
    }
    let dtype = dtype_from_tag(tag[0])
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("collective message has unknown dtype tag {} - corrupt frame or a non-brain peer", tag[0])))?;
    Ok(Payload::new(dtype, recv_vec(s)?))
}

/// What a rank announces when it joins: the plan every rank must share, the
/// world it was told and the rank it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupHello {
    /// The serialized job plan, compared byte for byte.
    pub plan: String,
    /// The world size the plan declares.
    pub world: usize,
    /// The rank this process was started as.
    pub rank: usize,
}

/// Tell the coordinator which rank this is and what plan it holds.
pub fn write_hello(s: &mut TcpStream, hello: &GroupHello) -> std::io::Result<()> {
    let plan = hello.plan.as_bytes();
    if plan.len() > MAX_PLAN_BYTES {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("job plan is {} bytes, over the {MAX_PLAN_BYTES}-byte limit", plan.len())));
    }
    let len = u32::try_from(plan.len()).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "job plan length does not fit u32"))?;
    let world = u32::try_from(hello.world).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "world size does not fit u32"))?;
    let rank = u32::try_from(hello.rank).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "rank does not fit u32"))?;
    s.write_all(&rank.to_le_bytes())?;
    s.write_all(&world.to_le_bytes())?;
    s.write_all(&len.to_le_bytes())?;
    s.write_all(plan)?;
    s.flush()
}

/// Read a rank's join announcement.
pub fn read_hello(s: &mut TcpStream) -> std::io::Result<GroupHello> {
    let mut word = [0u8; 4];
    s.read_exact(&mut word)?;
    let rank = u32::from_le_bytes(word) as usize;
    s.read_exact(&mut word)?;
    let world = u32::from_le_bytes(word) as usize;
    s.read_exact(&mut word)?;
    let len = u32::from_le_bytes(word) as usize;
    if len > MAX_PLAN_BYTES {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("join announcement claims a {len}-byte plan (over {MAX_PLAN_BYTES})")));
    }
    let mut plan = vec![0u8; len];
    s.read_exact(&mut plan)?;
    let plan = String::from_utf8(plan).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "join announcement plan is not UTF-8"))?;
    Ok(GroupHello { plan, world, rank })
}

/// The coordinator's answer to a join: accepted, or refused with a reason.
const JOIN_OK: u8 = 0;

/// Refuse a peer that announced a plan or rank this group does not have.
pub fn write_abort(s: &mut TcpStream, reason: &str) -> std::io::Result<()> {
    let body = reason.as_bytes();
    let len = u32::try_from(body.len()).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "abort reason too long"))?;
    s.write_all(&[ABORT_TAG])?;
    s.write_all(&len.to_le_bytes())?;
    s.write_all(body)?;
    s.flush()
}

fn read_abort_body(s: &mut TcpStream) -> std::io::Result<String> {
    let mut word = [0u8; 4];
    s.read_exact(&mut word)?;
    let len = u32::from_le_bytes(word) as usize;
    if len > MAX_PLAN_BYTES {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("abort reason claims {len} bytes")));
    }
    let mut body = vec![0u8; len];
    s.read_exact(&mut body)?;
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Tell every peer of a group that is being torn down why, ignoring peers that
/// already closed.
fn abort_all(conns: &[Option<Mutex<TcpStream>>], reason: &str) {
    for conn in conns.iter().skip(1).flatten() {
        let mut stream = conn.lock().unwrap();
        let _ = write_abort(&mut stream, reason);
    }
}

fn write_join_ok(s: &mut TcpStream) -> std::io::Result<()> {
    s.write_all(&[JOIN_OK])?;
    s.flush()
}

/// Read the coordinator's answer: `Ok(())` when accepted, the reason when not.
fn read_join_answer(s: &mut TcpStream) -> std::io::Result<()> {
    let mut answer = [0u8; 1];
    s.read_exact(&mut answer)?;
    if answer[0] == JOIN_OK {
        Ok(())
    } else if answer[0] == ABORT_TAG {
        Err(std::io::Error::other(format!("refused by the coordinator: {}", read_abort_body(s)?)))
    } else {
        Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unknown join answer {}", answer[0])))
    }
}

/// A TCP [`Collective`]. Construct rank 0 with [`NetworkCollective::coordinator`]
/// (from a bound listener) and each other rank with [`NetworkCollective::worker`].
#[derive(Debug)]
pub struct NetworkCollective {
    rank: usize,
    world: usize,
    /// Coordinator (rank 0): one stream per worker, indexed by peer rank (`[0]`
    /// unused). Worker: a single stream to the coordinator in slot 0.
    conns: Vec<Option<Mutex<TcpStream>>>,
}

impl NetworkCollective {
    /// Rank 0. Accept `world-1` worker connections on `listener` (each worker
    /// sends its rank on connect), returning once all peers are attached.
    pub fn coordinator(listener: &TcpListener, world: usize) -> std::io::Result<NetworkCollective> {
        let mut conns: Vec<Option<Mutex<TcpStream>>> = (0..world).map(|_| None).collect();
        for _ in 1..world {
            let (mut s, _) = listener.accept()?;
            s.set_nodelay(true).ok();
            let mut rb = [0u8; 4];
            s.read_exact(&mut rb)?;
            let peer = u32::from_le_bytes(rb) as usize;
            // The announced rank arrives off the wire - validate it before
            // using it as an index (an out-of-range or duplicate rank used
            // to panic the coordinator).
            if peer == 0 || peer >= world {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("worker announced invalid rank {peer} for world size {world}")));
            }
            if conns[peer].is_some() {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("two workers announced rank {peer}")));
            }
            conns[peer] = Some(Mutex::new(s));
        }
        Ok(NetworkCollective { rank: 0, world, conns })
    }

    /// A worker rank (`1..world`). Connect to the coordinator at `addr` and
    /// announce `rank`.
    pub fn worker(rank: usize, world: usize, addr: &str) -> std::io::Result<NetworkCollective> {
        let mut s = TcpStream::connect(addr)?;
        s.set_nodelay(true).ok();
        s.write_all(&(rank as u32).to_le_bytes())?;
        s.flush()?;
        let mut conns: Vec<Option<Mutex<TcpStream>>> = (0..world).map(|_| None).collect();
        conns[0] = Some(Mutex::new(s));
        Ok(NetworkCollective { rank, world, conns })
    }

    /// Rank 0 of a job: accept the other ranks, require each one's plan to match
    /// `hello`, and give up after `accept_timeout`.
    ///
    /// A peer that announces a different plan, world size or rank is told why
    /// and the group is torn down: every rank already accepted is sent the same
    /// abort, so none of them starts a collective the others will not join.
    pub fn coordinator_checked(listener: &TcpListener, hello: &GroupHello, accept_timeout: Duration, io_timeout: Duration) -> Result<NetworkCollective, CollectiveError> {
        let world = hello.world;
        let deadline = Instant::now() + accept_timeout;
        let mut conns: Vec<Option<Mutex<TcpStream>>> = (0..world).map(|_| None).collect();
        listener
            .set_nonblocking(true)
            .map_err(|e| CollectiveError::Transport(format!("rank 0 cannot set the coordinator listener non-blocking: {e}")))?;
        while conns.iter().skip(1).any(Option::is_none) {
            let (stream, _) = match listener.accept() {
                Ok(pair) => pair,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(CollectiveError::Transport(format!("rank 0 waited {accept_timeout:?} for {} of {} ranks to join", conns.iter().skip(1).filter(|c| c.is_none()).count(), world - 1)));
                    }
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(e) => return Err(CollectiveError::Transport(format!("rank 0 accepting a joining rank: {e}"))),
            };
            let mut stream = stream;
            stream.set_nonblocking(false).ok();
            stream.set_nodelay(true).ok();
            stream.set_read_timeout(Some(accept_timeout)).ok();
            stream.set_write_timeout(Some(accept_timeout)).ok();
            let peer = match read_hello(&mut stream) {
                Ok(peer) => peer,
                Err(e) => {
                    let _ = write_abort(&mut stream, "the join announcement could not be read");
                    let reason = format!("rank 0 could not read a joining rank's announcement: {e}");
                    abort_all(&conns, &reason);
                    return Err(CollectiveError::Transport(reason));
                }
            };
            if peer.plan != hello.plan || peer.world != world {
                let reason = format!("a rank joined with a different plan (world {}, expected world {world})", peer.world);
                let _ = write_abort(&mut stream, &reason);
                abort_all(&conns, &reason);
                return Err(CollectiveError::Transport(reason));
            }
            if peer.rank == 0 || peer.rank >= world {
                let reason = format!("a worker announced rank {}, outside 1..{world}", peer.rank);
                let _ = write_abort(&mut stream, &reason);
                abort_all(&conns, &reason);
                return Err(CollectiveError::Transport(reason));
            }
            if conns[peer.rank].is_some() {
                let reason = format!("two workers announced rank {}", peer.rank);
                let _ = write_abort(&mut stream, &reason);
                abort_all(&conns, &reason);
                return Err(CollectiveError::Transport(reason));
            }
            write_join_ok(&mut stream).map_err(|e| CollectiveError::Transport(format!("rank 0 accepting rank {}: {e}", peer.rank)))?;
            stream.set_read_timeout(Some(io_timeout)).ok();
            stream.set_write_timeout(Some(io_timeout)).ok();
            conns[peer.rank] = Some(Mutex::new(stream));
        }
        Ok(NetworkCollective { rank: 0, world, conns })
    }

    /// A worker rank of a job: connect to the coordinator until `connect_timeout`
    /// runs out, announce the plan, and start only once rank 0 accepts it.
    pub fn worker_checked(hello: &GroupHello, addr: &str, connect_timeout: Duration, io_timeout: Duration) -> Result<NetworkCollective, CollectiveError> {
        if hello.rank == 0 {
            return Err(CollectiveError::Transport("rank 0 must bind the coordinator, not dial it".into()));
        }
        let world = hello.world;
        let target = addr
            .to_socket_addrs()
            .map_err(|e| CollectiveError::Transport(format!("job coordinator {addr:?} does not resolve: {e}")))?
            .next()
            .ok_or_else(|| CollectiveError::Transport(format!("job coordinator {addr:?} resolves to no address")))?;
        let deadline = Instant::now() + connect_timeout;
        let mut stream = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(CollectiveError::Transport(format!("rank {}/{} could not reach the coordinator at {addr} within {connect_timeout:?}", hello.rank, world)));
            }
            match TcpStream::connect_timeout(&target, remaining) {
                Ok(stream) => break stream,
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::ConnectionRefused) => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => return Err(CollectiveError::Transport(format!("rank {}/{} connecting to {addr}: {e}", hello.rank, world))),
            }
        };
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(connect_timeout)).ok();
        stream.set_write_timeout(Some(connect_timeout)).ok();
        write_hello(&mut stream, hello).map_err(|e| CollectiveError::Transport(format!("rank {}/{} announcing its plan: {e}", hello.rank, world)))?;
        read_join_answer(&mut stream).map_err(|e| CollectiveError::Transport(format!("rank {}/{} joining the group: {e}", hello.rank, world)))?;
        stream.set_read_timeout(Some(io_timeout)).ok();
        stream.set_write_timeout(Some(io_timeout)).ok();
        let mut conns: Vec<Option<Mutex<TcpStream>>> = (0..world).map(|_| None).collect();
        conns[0] = Some(Mutex::new(stream));
        Ok(NetworkCollective { rank: hello.rank, world, conns })
    }

    /// Tell every worker why the group stopped, so none waits for a result that
    /// will not come. A worker that is already gone is ignored.
    fn abort_peers(&self, reason: &str) {
        for conn in self.conns.iter().skip(1).flatten() {
            let mut stream = conn.lock().unwrap();
            let _ = write_abort(&mut stream, reason);
        }
    }

    /// Wrap a wire I/O failure with which peer/op it happened on, as the
    /// trait's own [`CollectiveError::Transport`] (M7.2 gave this a real error
    /// channel - previously this could only panic the rank).
    fn wire_err(&self, what: &str, peer: usize, e: std::io::Error) -> CollectiveError {
        CollectiveError::Transport(format!("NetworkCollective rank {}/{}: {what} (peer rank {peer}): {e}", self.rank, self.world))
    }

    /// Coordinator: collect every rank's contribution in rank order (`local` is
    /// rank 0's own), returning `slots[0..world]`.
    fn gather_slots(&self, local: Payload) -> Result<Vec<Payload>, CollectiveError> {
        let mut slots: Vec<Payload> = (0..self.world).map(|_| Payload::f32(Vec::new())).collect();
        slots[0] = local;
        for (r, slot) in slots.iter_mut().enumerate().skip(1) {
            let mut s = self.conns[r].as_ref().unwrap().lock().unwrap();
            *slot = recv_payload(&mut s).map_err(|e| self.wire_err("recv contribution", r, e))?;
        }
        Ok(slots)
    }

    /// Coordinator: send `per_rank(r)` to each worker `r`.
    fn scatter(&self, per_rank: impl Fn(usize) -> Payload) -> Result<(), CollectiveError> {
        for r in 1..self.world {
            let mut s = self.conns[r].as_ref().unwrap().lock().unwrap();
            send_payload(&mut s, &per_rank(r)).map_err(|e| self.wire_err("send result", r, e))?;
        }
        Ok(())
    }

    /// Worker: send `local` to the coordinator, receive the op's result.
    fn round_trip(&self, local: &Payload) -> Result<Payload, CollectiveError> {
        let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
        send_payload(&mut s, local).map_err(|e| self.wire_err("send to coordinator", 0, e))?;
        recv_payload(&mut s).map_err(|e| self.wire_err("recv from coordinator", 0, e))
    }
}

impl Collective for NetworkCollective {
    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            check_rank(rank, self.world)?;
            if self.world == 1 {
                return Ok(local);
            }
            if self.rank == 0 {
                let out = (|| -> CollectiveResult<Payload> {
                    let slots = self.gather_slots(local)?;
                    let dtype = check_dtypes(&slots)?;
                    let n = check_equal_lengths(&slots)?;
                    let mut sum = vec![0f32; n];
                    for s in &slots {
                        for (a, b) in sum.iter_mut().zip(&s.data) {
                            *a += b;
                        }
                    }
                    let out = Payload::new(dtype, sum);
                    self.scatter(|_| out.clone())?;
                    Ok(out)
                })();
                if let Err(e) = &out {
                    self.abort_peers(&e.to_string());
                }
                out
            } else {
                self.round_trip(&local)
            }
        })
    }

    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move {
            check_rank(rank, self.world)?;
            if self.world == 1 {
                return Ok(local);
            }
            if self.rank == 0 {
                let out = (|| -> CollectiveResult<Payload> {
                    let slots = self.gather_slots(local)?;
                    let dtype = check_dtypes(&slots)?;
                    let cat = Payload::new(dtype, slots.iter().flat_map(|s| s.data.iter().copied()).collect());
                    self.scatter(|_| cat.clone())?;
                    Ok(cat)
                })();
                if let Err(e) = &out {
                    self.abort_peers(&e.to_string());
                }
                out
            } else {
                self.round_trip(&local)
            }
        })
    }

    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        let world = self.world;
        Box::pin(async move {
            check_rank(rank, world)?;
            if world == 1 {
                return Ok(local);
            }
            if self.rank == 0 {
                let out = (|| -> CollectiveResult<Payload> {
                    let slots = self.gather_slots(local)?;
                    let dtype = check_dtypes(&slots)?;
                    let n = check_equal_lengths(&slots)?;
                    if n % world != 0 {
                        return Err(CollectiveError::WorldSizeMismatch { world_size: world, len: n });
                    }
                    let chunk = n / world;
                    let mut sum = vec![0f32; n];
                    for s in &slots {
                        for (a, b) in sum.iter_mut().zip(&s.data) {
                            *a += b;
                        }
                    }
                    self.scatter(|r| Payload::new(dtype, sum[r * chunk..(r + 1) * chunk].to_vec()))?;
                    Ok(Payload::new(dtype, sum[0..chunk].to_vec()))
                })();
                if let Err(e) = &out {
                    self.abort_peers(&e.to_string());
                }
                out
            } else {
                self.round_trip(&local)
            }
        })
    }

    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        let world = self.world;
        Box::pin(async move {
            check_rank(rank, world)?;
            check_rank(root, world)?;
            if world == 1 {
                return Ok(local);
            }
            // Route through the coordinator: root's data reaches rank 0 (if root != 0),
            // then rank 0 fans it out to everyone.
            if self.rank == 0 {
                let out = (|| -> CollectiveResult<Payload> {
                    let data = if root == 0 {
                        local
                    } else {
                        let mut s = self.conns[root].as_ref().unwrap().lock().unwrap();
                        recv_payload(&mut s).map_err(|e| self.wire_err("recv broadcast root", root, e))?
                    };
                    self.scatter(|_| data.clone())?;
                    Ok(data)
                })();
                if let Err(e) = &out {
                    self.abort_peers(&e.to_string());
                }
                out
            } else if self.rank == root {
                // send my data up, then receive the fan-out copy back.
                {
                    let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
                    send_payload(&mut s, &local).map_err(|e| self.wire_err("send broadcast up", 0, e))?;
                }
                let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
                recv_payload(&mut s).map_err(|e| self.wire_err("recv broadcast back", 0, e))
            } else {
                let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
                recv_payload(&mut s).map_err(|e| self.wire_err("recv broadcast", 0, e))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Drive an op across `world` ranks over loopback TCP; rank 0 is the
    /// coordinator. Returns each rank's `Ok` result (panics on an `Err` - none
    /// of these ok-path tests expect one).
    fn run_net<F>(world: usize, f: F) -> Vec<Payload>
    where
        F: Fn(&NetworkCollective, usize) -> CollectiveFuture<'_, Payload> + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let f = Arc::new(f);
        let results: Arc<Vec<Mutex<Option<Payload>>>> = Arc::new((0..world).map(|_| Mutex::new(None)).collect());

        let mut handles = Vec::new();
        // coordinator (rank 0)
        {
            let (f, results) = (f.clone(), results.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::coordinator(&listener, world).unwrap();
                *results[0].lock().unwrap() = Some(pollster::block_on(f(&coll, 0)).unwrap());
            }));
        }
        // workers
        for r in 1..world {
            let (f, results, addr) = (f.clone(), results.clone(), addr.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::worker(r, world, &addr).unwrap();
                *results[r].lock().unwrap() = Some(pollster::block_on(f(&coll, r)).unwrap());
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        Arc::try_unwrap(results).unwrap().into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
    }

    #[test]
    fn net_all_reduce_matches_host() {
        let out = run_net(4, |c, r| c.all_reduce(r, Payload::f32(vec![r as f32, 10.0 + r as f32, 20.0 + r as f32])));
        for row in &out {
            assert_eq!(row.data, vec![6.0, 46.0, 86.0]);
        }
    }

    #[test]
    fn net_all_gather_in_rank_order() {
        let out = run_net(3, |c, r| c.all_gather(r, Payload::f32(vec![r as f32, r as f32 + 0.5])));
        for row in &out {
            assert_eq!(row.data, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        }
    }

    #[test]
    fn net_reduce_scatter_sums_then_slices() {
        let out = run_net(2, |c, r| c.reduce_scatter(r, Payload::f32(vec![1.0, 2.0, 3.0, 4.0])));
        assert_eq!(out[0].data, vec![2.0, 4.0]);
        assert_eq!(out[1].data, vec![6.0, 8.0]);
    }

    #[test]
    fn net_broadcast_from_nonzero_root() {
        let out = run_net(3, |c, r| {
            let local = if r == 2 { vec![7.0, 8.0, 9.0] } else { Vec::new() };
            c.broadcast(r, Payload::f32(local), 2)
        });
        for row in &out {
            assert_eq!(row.data, vec![7.0, 8.0, 9.0]);
        }
    }

    #[test]
    fn net_reusable_across_ops() {
        // `run_net`'s closure returns one CollectiveFuture, so drive the
        // sequence inline via its own async block instead.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let results: Arc<Vec<Mutex<Vec<f32>>>> = Arc::new((0..2).map(|_| Mutex::new(Vec::new())).collect());
        let mut handles = Vec::new();
        {
            let results = results.clone();
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::coordinator(&listener, 2).unwrap();
                let out = pollster::block_on(async {
                    let a = coll.all_reduce(0, Payload::f32(vec![1.0])).await.unwrap(); // [3]
                    let b = coll.all_gather(0, a).await.unwrap(); // [3,3]
                    coll.all_reduce(0, b).await.unwrap() // [6,6]
                });
                *results[0].lock().unwrap() = out.data;
            }));
        }
        {
            let results = results.clone();
            let addr = addr.clone();
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::worker(1, 2, &addr).unwrap();
                let out = pollster::block_on(async {
                    let a = coll.all_reduce(1, Payload::f32(vec![2.0])).await.unwrap(); // [3]
                    let b = coll.all_gather(1, a).await.unwrap(); // [3,3]
                    coll.all_reduce(1, b).await.unwrap() // [6,6]
                });
                *results[1].lock().unwrap() = out.data;
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        for m in results.iter() {
            assert_eq!(*m.lock().unwrap(), vec![6.0, 6.0]);
        }
    }

    #[test]
    fn every_dtype_round_trips_its_wire_tag() {
        for d in [Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::I8, Dtype::Q4, Dtype::Q4K, Dtype::Q8K, Dtype::NF4, Dtype::F4E2M1, Dtype::F8E4M3, Dtype::F8E5M2] {
            assert_eq!(dtype_from_tag(dtype_tag(d)), Some(d), "{d:?} did not round-trip its wire tag");
        }
    }

    #[test]
    fn rank_out_of_range_is_reported_not_a_panic() {
        // world=1, so this never touches the network at all.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let coll = NetworkCollective::coordinator(&listener, 1).unwrap();
        let err = pollster::block_on(coll.all_reduce(3, Payload::f32(vec![1.0])));
        assert_eq!(err, Err(CollectiveError::RankOutOfRange { rank: 3, world_size: 1 }));
    }
}
