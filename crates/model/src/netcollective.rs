// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Collective` over TCP — the multi-machine transport. Same trait, same call
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
//! same trait when the world grows large — the point of the abstraction.
//!
//! Framing is length-prefixed little-endian f32 (`u32` count, then the payload),
//! read/written with `read_exact`/`write_all` so partial TCP reads are handled.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use crate::collective::Collective;

fn send_vec(s: &mut TcpStream, v: &[f32]) -> std::io::Result<()> {
    s.write_all(&(v.len() as u32).to_le_bytes())?;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    s.write_all(&bytes)?;
    s.flush()
}

fn recv_vec(s: &mut TcpStream) -> std::io::Result<Vec<f32>> {
    let mut len = [0u8; 4];
    s.read_exact(&mut len)?;
    let n = u32::from_le_bytes(len) as usize;
    let mut bytes = vec![0u8; n * 4];
    s.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// A TCP [`Collective`]. Construct rank 0 with [`NetworkCollective::coordinator`]
/// (from a bound listener) and each other rank with [`NetworkCollective::worker`].
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

    /// Coordinator: collect every rank's contribution in rank order (`local` is
    /// rank 0's own), returning `slots[0..world]`.
    fn gather_slots(&self, local: Vec<f32>) -> Vec<Vec<f32>> {
        let mut slots: Vec<Vec<f32>> = vec![Vec::new(); self.world];
        slots[0] = local;
        for (r, slot) in slots.iter_mut().enumerate().skip(1) {
            let mut s = self.conns[r].as_ref().unwrap().lock().unwrap();
            *slot = recv_vec(&mut s).expect("coordinator recv");
        }
        slots
    }

    /// Coordinator: send `per_rank(r)` to each worker `r`.
    fn scatter(&self, per_rank: impl Fn(usize) -> Vec<f32>) {
        for r in 1..self.world {
            let mut s = self.conns[r].as_ref().unwrap().lock().unwrap();
            send_vec(&mut s, &per_rank(r)).expect("coordinator send");
        }
    }

    /// Worker: send `local` to the coordinator, receive the op's result.
    fn round_trip(&self, local: &[f32]) -> Vec<f32> {
        let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
        send_vec(&mut s, local).expect("worker send");
        recv_vec(&mut s).expect("worker recv")
    }
}

impl Collective for NetworkCollective {
    fn world_size(&self) -> usize {
        self.world
    }

    fn all_reduce(&self, _rank: usize, local: Vec<f32>) -> Vec<f32> {
        if self.world == 1 {
            return local;
        }
        if self.rank == 0 {
            let slots = self.gather_slots(local);
            let n = slots[0].len();
            let mut sum = vec![0f32; n];
            for s in &slots {
                for (a, b) in sum.iter_mut().zip(s) {
                    *a += b;
                }
            }
            self.scatter(|_| sum.clone());
            sum
        } else {
            self.round_trip(&local)
        }
    }

    fn all_gather(&self, _rank: usize, local: Vec<f32>) -> Vec<f32> {
        if self.world == 1 {
            return local;
        }
        if self.rank == 0 {
            let slots = self.gather_slots(local);
            let cat: Vec<f32> = slots.iter().flatten().copied().collect();
            self.scatter(|_| cat.clone());
            cat
        } else {
            self.round_trip(&local)
        }
    }

    fn reduce_scatter(&self, rank: usize, local: Vec<f32>) -> Vec<f32> {
        if self.world == 1 {
            return local;
        }
        let world = self.world;
        if self.rank == 0 {
            let slots = self.gather_slots(local);
            let n = slots[0].len();
            assert_eq!(n % world, 0, "reduce_scatter: length not divisible by world_size");
            let chunk = n / world;
            let mut sum = vec![0f32; n];
            for s in &slots {
                for (a, b) in sum.iter_mut().zip(s) {
                    *a += b;
                }
            }
            self.scatter(|r| sum[r * chunk..(r + 1) * chunk].to_vec());
            sum[0..chunk].to_vec()
        } else {
            let _ = rank;
            self.round_trip(&local)
        }
    }

    fn broadcast(&self, _rank: usize, local: Vec<f32>, root: usize) -> Vec<f32> {
        if self.world == 1 {
            return local;
        }
        // Route through the coordinator: root's data reaches rank 0 (if root != 0),
        // then rank 0 fans it out to everyone.
        if self.rank == 0 {
            let data = if root == 0 {
                local
            } else {
                let mut s = self.conns[root].as_ref().unwrap().lock().unwrap();
                recv_vec(&mut s).expect("coordinator recv root")
            };
            self.scatter(|_| data.clone());
            data
        } else if self.rank == root {
            // send my data up, then receive the fan-out copy back.
            {
                let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
                send_vec(&mut s, &local).expect("root send up");
            }
            let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
            recv_vec(&mut s).expect("root recv back")
        } else {
            let mut s = self.conns[0].as_ref().unwrap().lock().unwrap();
            recv_vec(&mut s).expect("worker recv bcast")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Drive an op across `world` ranks over loopback TCP; rank 0 is the
    /// coordinator. Returns each rank's result.
    fn run_net<F>(world: usize, f: F) -> Vec<Vec<f32>>
    where
        F: Fn(&NetworkCollective, usize) -> Vec<f32> + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let f = Arc::new(f);
        let results: Arc<Vec<Mutex<Vec<f32>>>> = Arc::new((0..world).map(|_| Mutex::new(Vec::new())).collect());

        let mut handles = Vec::new();
        // coordinator (rank 0)
        {
            let (f, results) = (f.clone(), results.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::coordinator(&listener, world).unwrap();
                *results[0].lock().unwrap() = f(&coll, 0);
            }));
        }
        // workers
        for r in 1..world {
            let (f, results, addr) = (f.clone(), results.clone(), addr.clone());
            handles.push(thread::spawn(move || {
                let coll = NetworkCollective::worker(r, world, &addr).unwrap();
                *results[r].lock().unwrap() = f(&coll, r);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        Arc::try_unwrap(results).unwrap().into_iter().map(|m| m.into_inner().unwrap()).collect()
    }

    #[test]
    fn net_all_reduce_matches_host() {
        let out = run_net(4, |c, r| c.all_reduce(r, vec![r as f32, 10.0 + r as f32, 20.0 + r as f32]));
        for row in &out {
            assert_eq!(row, &vec![6.0, 46.0, 86.0]);
        }
    }

    #[test]
    fn net_all_gather_in_rank_order() {
        let out = run_net(3, |c, r| c.all_gather(r, vec![r as f32, r as f32 + 0.5]));
        for row in &out {
            assert_eq!(row, &vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        }
    }

    #[test]
    fn net_reduce_scatter_sums_then_slices() {
        let out = run_net(2, |c, r| c.reduce_scatter(r, vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(out[0], vec![2.0, 4.0]);
        assert_eq!(out[1], vec![6.0, 8.0]);
    }

    #[test]
    fn net_broadcast_from_nonzero_root() {
        let out = run_net(3, |c, r| {
            let local = if r == 2 { vec![7.0, 8.0, 9.0] } else { Vec::new() };
            c.broadcast(r, local, 2)
        });
        for row in &out {
            assert_eq!(row, &vec![7.0, 8.0, 9.0]);
        }
    }

    #[test]
    fn net_reusable_across_ops() {
        let out = run_net(2, |c, r| {
            let a = c.all_reduce(r, vec![r as f32 + 1.0]); // [3]
            let b = c.all_gather(r, a); // [3,3]
            c.all_reduce(r, b) // [6,6]
        });
        for row in &out {
            assert_eq!(row, &vec![6.0, 6.0]);
        }
    }
}
