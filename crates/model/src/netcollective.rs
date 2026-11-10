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
//! Every connection is a [`SealedStream`]: before anything else, each side
//! proves the job's [`JobSecret`] to the other, and every frame after that is
//! sealed with keys of that connection alone (`crate::sealedwire`). A host
//! that reaches the coordinator without the secret is dropped after a
//! fixed-size exchange, so it can neither join, nor read or alter the
//! traffic, nor tear the forming group down.
//!
//! Inside a sealed frame a payload is one dtype tag byte, then little-endian
//! f32 elements; an abort is the tag no dtype uses, then a UTF-8 reason. A
//! rank that cannot take part sends an abort instead of a result, so its peers
//! fail with the reason rather than waiting. A wire failure surfaces as
//! `Err(CollectiveError::Transport(..))` naming the peer and op. Rank and root
//! range checks run on each side's own call before any I/O, so an invalid
//! argument never reaches the wire and never strands a peer.

use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::collective::{check_dtypes, check_equal_lengths, check_rank, Collective, CollectiveError, CollectiveFuture, CollectiveResult, Payload};
use crate::sealedwire::{JobSecret, SealedStream};
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
        Dtype::Cq2 => 11,
        Dtype::Cq3 => 12,
        Dtype::Cq4 => 13,
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
        11 => Dtype::Cq2,
        12 => Dtype::Cq3,
        13 => Dtype::Cq4,
        _ => return None,
    })
}

/// Largest accepted message, in f32 elements (1 GiB of payload). The length
/// arrives off the wire: without a bound a corrupt or hostile frame header
/// could demand a 16 GB allocation before any data is read.
const MAX_MSG_ELEMS: usize = 1 << 28;

/// The frame tag no dtype uses: an abort carrying a UTF-8 reason. A rank that
/// cannot take part sends one instead of a result, so its peers fail with the
/// reason rather than waiting for a payload that will never come.
const ABORT_TAG: u8 = 255;

/// Largest accepted join announcement, in bytes. The plan is a few hundred
/// bytes; the bound stops a corrupt length from allocating arbitrarily.
pub const MAX_PLAN_BYTES: usize = 1 << 20;

/// How long a connecting peer has to prove it holds the job secret. The
/// exchange is a few dozen bytes, so this only has to cover a slow network;
/// it is what bounds a peer that connects and then says nothing.
const AUTH_TIMEOUT: Duration = Duration::from_secs(10);

/// How many connections may be proving the secret at once. A connection over
/// the bound is closed unread, so a flood of peers cannot grow threads.
const MAX_PENDING_JOINS: usize = 64;

fn send_payload(s: &mut SealedStream, p: &Payload) -> std::io::Result<()> {
    if p.data.len() > MAX_MSG_ELEMS {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("a {}-element payload is over the {MAX_MSG_ELEMS}-element wire limit", p.data.len())));
    }
    let mut frame = Vec::with_capacity(1 + p.data.len() * 4);
    frame.push(dtype_tag(p.dtype));
    for x in &p.data {
        frame.extend_from_slice(&x.to_le_bytes());
    }
    s.send(&frame)
}

fn recv_payload(s: &mut SealedStream) -> std::io::Result<Payload> {
    let frame = s.recv(1 + MAX_MSG_ELEMS * 4)?;
    let Some((&tag, body)) = frame.split_first() else {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "an empty collective frame"));
    };
    if tag == ABORT_TAG {
        return Err(std::io::Error::other(format!("peer aborted the collective: {}", String::from_utf8_lossy(body))));
    }
    let dtype = dtype_from_tag(tag).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("collective message has unknown dtype tag {tag}")))?;
    if body.len() % 4 != 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("a collective payload of {} bytes is not whole f32 elements", body.len())));
    }
    Ok(Payload::new(dtype, body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()))
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
fn write_hello(s: &mut SealedStream, hello: &GroupHello) -> std::io::Result<()> {
    let plan = hello.plan.as_bytes();
    if plan.len() > MAX_PLAN_BYTES {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("job plan is {} bytes, over the {MAX_PLAN_BYTES}-byte limit", plan.len())));
    }
    let world = u32::try_from(hello.world).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "world size does not fit u32"))?;
    let rank = u32::try_from(hello.rank).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "rank does not fit u32"))?;
    let mut frame = Vec::with_capacity(8 + plan.len());
    frame.extend_from_slice(&rank.to_le_bytes());
    frame.extend_from_slice(&world.to_le_bytes());
    frame.extend_from_slice(plan);
    s.send(&frame)
}

/// Read a rank's join announcement.
fn read_hello(s: &mut SealedStream) -> std::io::Result<GroupHello> {
    let frame = s.recv(8 + MAX_PLAN_BYTES)?;
    if frame.len() < 8 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "a join announcement shorter than its header"));
    }
    let rank = u32::from_le_bytes(frame[0..4].try_into().expect("four bytes")) as usize;
    let world = u32::from_le_bytes(frame[4..8].try_into().expect("four bytes")) as usize;
    let plan = String::from_utf8(frame[8..].to_vec()).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "join announcement plan is not UTF-8"))?;
    Ok(GroupHello { plan, world, rank })
}

/// The coordinator's answer to a join: accepted, or refused with a reason.
const JOIN_OK: u8 = 0;

/// Refuse a peer that announced a plan or rank this group does not have, or
/// tell a joined peer why the group stopped.
fn write_abort(s: &mut SealedStream, reason: &str) -> std::io::Result<()> {
    let reason = &reason.as_bytes()[..reason.len().min(MAX_PLAN_BYTES)];
    let mut frame = Vec::with_capacity(1 + reason.len());
    frame.push(ABORT_TAG);
    frame.extend_from_slice(reason);
    s.send(&frame)
}

/// Tell every peer of a group that is being torn down why, ignoring peers that
/// already closed.
fn abort_all(conns: &[Option<Mutex<SealedStream>>], reason: &str) {
    for conn in conns.iter().skip(1).flatten() {
        let mut stream = conn.lock().unwrap();
        let _ = write_abort(&mut stream, reason);
    }
}

/// Read the coordinator's answer: `Ok(())` when accepted, the reason when not.
fn read_join_answer(s: &mut SealedStream) -> std::io::Result<()> {
    let frame = s.recv(1 + MAX_PLAN_BYTES)?;
    match frame.split_first() {
        Some((&JOIN_OK, [])) => Ok(()),
        Some((&ABORT_TAG, reason)) => Err(std::io::Error::other(format!("refused by the coordinator: {}", String::from_utf8_lossy(reason)))),
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "an unknown join answer")),
    }
}

/// One accepted connection's way through authentication and its announcement,
/// off the accept loop's thread: a peer that is slow to prove the secret then
/// delays nobody else's join. Sends nothing for a peer that fails to prove it.
fn admit(stream: TcpStream, secret: JobSecret, joined: std::sync::mpsc::Sender<(SealedStream, std::io::Result<GroupHello>)>, pending: std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let peer = stream.peer_addr().ok();
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(AUTH_TIMEOUT)).ok();
    stream.set_write_timeout(Some(AUTH_TIMEOUT)).ok();
    match SealedStream::accept(stream, &secret) {
        Ok(mut sealed) => {
            let hello = read_hello(&mut sealed);
            let _ = joined.send((sealed, hello));
        }
        // Not a member of this job: dropped, and the group keeps waiting for
        // its real ranks. A stranger cannot tear down a job by connecting.
        Err(e) => eprintln!("job coordinator: refused a connection from {peer:?}: {e}"),
    }
    pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
}

/// A TCP [`Collective`] over [`SealedStream`]s. Rank 0 is built with
/// [`NetworkCollective::coordinator`] and every other rank with
/// [`NetworkCollective::worker`]; both refuse any peer that does not hold the
/// job's [`JobSecret`].
#[derive(Debug)]
pub struct NetworkCollective {
    rank: usize,
    world: usize,
    /// Coordinator (rank 0): one stream per worker, indexed by peer rank (`[0]`
    /// unused). Worker: a single stream to the coordinator in slot 0.
    conns: Vec<Option<Mutex<SealedStream>>>,
}

impl NetworkCollective {
    /// Rank 0 of a job: accept the other ranks on `listener`, require each to
    /// prove `secret` and to announce the plan in `hello`, and give up after
    /// `accept_timeout`.
    ///
    /// A connection that does not prove the secret is dropped and the wait
    /// goes on. A proven peer that announces a different plan, world size or
    /// rank is told why and the group is torn down: every rank already
    /// accepted is sent the same abort, so none of them starts a collective
    /// the others will not join.
    pub fn coordinator(listener: &TcpListener, hello: &GroupHello, secret: &JobSecret, accept_timeout: Duration, io_timeout: Duration) -> Result<NetworkCollective, CollectiveError> {
        let world = hello.world;
        let deadline = Instant::now() + accept_timeout;
        let mut conns: Vec<Option<Mutex<SealedStream>>> = (0..world).map(|_| None).collect();
        listener
            .set_nonblocking(true)
            .map_err(|e| CollectiveError::Transport(format!("rank 0 cannot set the coordinator listener non-blocking: {e}")))?;
        let (joined_tx, joined) = std::sync::mpsc::channel();
        let pending = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        while conns.iter().skip(1).any(Option::is_none) {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).ok();
                    if pending.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= MAX_PENDING_JOINS {
                        pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        continue; // dropped unread
                    }
                    let (secret, joined_tx, pending) = (secret.clone(), joined_tx.clone(), pending.clone());
                    std::thread::spawn(move || admit(stream, secret, joined_tx, pending));
                    continue;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(CollectiveError::Transport(format!("rank 0 accepting a joining rank: {e}"))),
            }
            let (mut stream, peer) = match joined.recv_timeout(Duration::from_millis(2)) {
                Ok(joined) => joined,
                Err(_) => {
                    if Instant::now() >= deadline {
                        let reason = format!("rank 0 waited {accept_timeout:?} for {} of {} ranks to join", conns.iter().skip(1).filter(|c| c.is_none()).count(), world - 1);
                        abort_all(&conns, &reason);
                        return Err(CollectiveError::Transport(reason));
                    }
                    continue;
                }
            };
            let peer = match peer {
                Ok(peer) => peer,
                Err(e) => {
                    let _ = write_abort(&mut stream, "the join announcement could not be read");
                    let reason = format!("rank 0 could not read a joining rank's announcement: {e}");
                    abort_all(&conns, &reason);
                    return Err(CollectiveError::Transport(reason));
                }
            };
            let refusal = if peer.plan != hello.plan || peer.world != world {
                Some(format!("a rank joined with a different plan (world {}, expected world {world})", peer.world))
            } else if peer.rank == 0 || peer.rank >= world {
                Some(format!("a worker announced rank {}, outside 1..{world}", peer.rank))
            } else if conns[peer.rank].is_some() {
                Some(format!("two workers announced rank {}", peer.rank))
            } else {
                None
            };
            if let Some(reason) = refusal {
                let _ = write_abort(&mut stream, &reason);
                abort_all(&conns, &reason);
                return Err(CollectiveError::Transport(reason));
            }
            stream.send(&[JOIN_OK]).map_err(|e| CollectiveError::Transport(format!("rank 0 accepting rank {}: {e}", peer.rank)))?;
            stream.tcp().set_read_timeout(Some(io_timeout)).ok();
            stream.tcp().set_write_timeout(Some(io_timeout)).ok();
            conns[peer.rank] = Some(Mutex::new(stream));
        }
        Ok(NetworkCollective { rank: 0, world, conns })
    }

    /// A worker rank of a job: connect to the coordinator until `connect_timeout`
    /// runs out, prove `secret` and require the coordinator to prove it too,
    /// announce the plan, and start only once rank 0 accepts it.
    pub fn worker(hello: &GroupHello, addr: &str, secret: &JobSecret, connect_timeout: Duration, io_timeout: Duration) -> Result<NetworkCollective, CollectiveError> {
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
        let stream = loop {
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
        let mut stream = SealedStream::connect(stream, secret).map_err(|e| CollectiveError::Transport(format!("rank {}/{} authenticating with the coordinator at {addr}: {e}", hello.rank, world)))?;
        write_hello(&mut stream, hello).map_err(|e| CollectiveError::Transport(format!("rank {}/{} announcing its plan: {e}", hello.rank, world)))?;
        read_join_answer(&mut stream).map_err(|e| CollectiveError::Transport(format!("rank {}/{} joining the group: {e}", hello.rank, world)))?;
        stream.tcp().set_read_timeout(Some(io_timeout)).ok();
        stream.tcp().set_write_timeout(Some(io_timeout)).ok();
        let mut conns: Vec<Option<Mutex<SealedStream>>> = (0..world).map(|_| None).collect();
        conns[0] = Some(Mutex::new(stream));
        Ok(NetworkCollective { rank: hello.rank, world, conns })
    }

    /// Tell every worker why the group stopped, so none waits for a result that
    /// will not come. A worker that is already gone is ignored.
    fn abort_peers(&self, reason: &str) {
        abort_all(&self.conns, reason);
    }

    /// Wrap a wire I/O failure with which peer/op it happened on, as the
    /// trait's own [`CollectiveError::Transport`].
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

    const TIMEOUT: Duration = Duration::from_secs(30);

    fn hello(world: usize, rank: usize) -> GroupHello {
        GroupHello { plan: "test".into(), world, rank }
    }

    /// Drive an op across `world` ranks over loopback TCP; rank 0 is the
    /// coordinator. Returns each rank's `Ok` result (panics on an `Err` - none
    /// of these ok-path tests expect one).
    fn run_net<F>(world: usize, f: F) -> Vec<Payload>
    where
        F: Fn(&NetworkCollective, usize) -> CollectiveFuture<'_, Payload> + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let secret = JobSecret::random();
        let f = Arc::new(f);
        let results: Arc<Vec<Mutex<Option<Payload>>>> = Arc::new((0..world).map(|_| Mutex::new(None)).collect());

        let mut handles = Vec::new();
        // coordinator (rank 0)
        {
            let (f, results, secret) = (f.clone(), results.clone(), secret.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::coordinator(&listener, &hello(world, 0), &secret, TIMEOUT, TIMEOUT).unwrap();
                *results[0].lock().unwrap() = Some(pollster::block_on(f(&coll, 0)).unwrap());
            }));
        }
        // workers
        for r in 1..world {
            let (f, results, addr, secret) = (f.clone(), results.clone(), addr.clone(), secret.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::worker(&hello(world, r), &addr, &secret, TIMEOUT, TIMEOUT).unwrap();
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
        let secret = JobSecret::random();
        let results: Arc<Vec<Mutex<Vec<f32>>>> = Arc::new((0..2).map(|_| Mutex::new(Vec::new())).collect());
        let mut handles = Vec::new();
        {
            let (results, secret) = (results.clone(), secret.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::coordinator(&listener, &hello(2, 0), &secret, TIMEOUT, TIMEOUT).unwrap();
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
                let coll = NetworkCollective::worker(&hello(2, 1), &addr, &secret, TIMEOUT, TIMEOUT).unwrap();
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
        let coll = NetworkCollective::coordinator(&listener, &hello(1, 0), &JobSecret::random(), TIMEOUT, TIMEOUT).unwrap();
        let err = pollster::block_on(coll.all_reduce(3, Payload::f32(vec![1.0])));
        assert_eq!(err, Err(CollectiveError::RankOutOfRange { rank: 3, world_size: 1 }));
    }
}
