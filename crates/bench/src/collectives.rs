// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for qualifying GPU clusters for
// distributed training for its clients: what the fabric delivers, measured
// with the same collectives the trainer runs. If your team needs expertise in
// multi-node collective performance then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `brain bench collectives`: the device collectives a distributed job uses, timed and checked as one rank of a job
//! plan.
//!
//! Every rank of the job runs it with the same plan and secret (`--job-plan`, `--job-rank`, `--job-secret`) on its own
//! card. The ranks join the job's sealed host transport, form a device group over it ([`RankGroup`], NCCL on CUDA)
//! the way a trainer does, and then, for every operation, dtype and size of the sweep, time each iteration on the
//! device: CUDA events recorded on the communicator's stream around a batch of the operation run back to back (one
//! for a large size), divided by the batch, so a small message's time is the collective's, not the host's polling or
//! the ranks' arrival skew (`config.timing` says which clock a run used). Sizes follow the usual collective-benchmark convention: the whole buffer of an
//! all-reduce or broadcast, the gathered (all-gather) or scattered (reduce-scatter) buffer of the others. Algorithm
//! bandwidth is that size over the slowest rank's median time; bus bandwidth scales it by the bytes each link carries
//! (`2 (n - 1) / n` for an all-reduce, `(n - 1) / n` for a gather or scatter, `1` for a broadcast), so it compares
//! with the link's own rate.
//!
//! Every operation is also checked against a reference the host computes from the same exactly representable inputs,
//! on every rank. Rank 0 writes ONE result document: hardware (hosts, cards), software (brain, the collective
//! library and its version, the communication libraries the process loaded), configuration, per-case p50/p99 and
//! bandwidths, the transport NCCL reports in its own log when one is given, the NIC octets of the run where the node
//! exposes CXI telemetry, and the correctness verdict. A field the run could not observe is null, never guessed.
//!
//! The transport is what NCCL itself logs (`--nccl-log FILE`: the file the launcher pointed `NCCL_DEBUG_FILE` at with
//! `NCCL_DEBUG=INFO`): the network it chose (`Using network ...`) and the path of every channel (`via ...`). With
//! `--expect-network NAME` a rank that chose another network, or a job spanning nodes whose channels never cross the
//! named network, fails the run: a silent socket fallback is the failure this exists to catch.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ::model::collective::{Collective, Payload};
use ::model::devcollective::rendezvous;
use ::model::JobContext;
use gpu_core::devgroup::RankGroup;
use gpu_core::{CollectiveDtype, DeviceBuffer, Gpu};
use serde_json::{json, Value};

/// The result document's format version.
pub const FORMAT: u32 = 1;

/// The longest text one rank contributes to the result (a host name, a transport list).
const MAX_TEXT: usize = 2048;

/// One collective operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
}

impl Op {
    pub const ALL: [Op; 4] = [Op::AllReduce, Op::AllGather, Op::ReduceScatter, Op::Broadcast];

    pub fn name(self) -> &'static str {
        match self {
            Op::AllReduce => "all_reduce",
            Op::AllGather => "all_gather",
            Op::ReduceScatter => "reduce_scatter",
            Op::Broadcast => "broadcast",
        }
    }

    pub fn parse(name: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|op| op.name() == name)
    }

    /// The bytes each link carries per byte of the operation's size, over `world` ranks.
    pub fn bus_factor(self, world: usize) -> f64 {
        let n = world.max(1) as f64;
        match self {
            Op::AllReduce => 2.0 * (n - 1.0) / n,
            Op::AllGather | Op::ReduceScatter => (n - 1.0) / n,
            Op::Broadcast => 1.0,
        }
    }
}

fn dtype_name(dtype: CollectiveDtype) -> &'static str {
    match dtype {
        CollectiveDtype::F32 => "f32",
        CollectiveDtype::Bf16 => "bf16",
    }
}

/// Parse `f32` or `bf16`.
pub fn parse_dtype(name: &str) -> Option<CollectiveDtype> {
    match name {
        "f32" => Some(CollectiveDtype::F32),
        "bf16" => Some(CollectiveDtype::Bf16),
        _ => None,
    }
}

/// Parse a byte size: a number with an optional `K`, `M` or `G` (powers of 1024).
pub fn parse_size(text: &str) -> Option<u64> {
    let text = text.trim();
    let (digits, scale) = match text.chars().last()? {
        'K' | 'k' => (&text[..text.len() - 1], 1u64 << 10),
        'M' | 'm' => (&text[..text.len() - 1], 1 << 20),
        'G' | 'g' => (&text[..text.len() - 1], 1 << 30),
        _ => (text, 1),
    };
    digits.parse::<u64>().ok()?.checked_mul(scale).filter(|&n| n > 0)
}

/// What one run measures.
#[derive(Clone, Debug)]
pub struct Config {
    pub ops: Vec<Op>,
    pub dtypes: Vec<CollectiveDtype>,
    /// Operation sizes in bytes.
    pub sizes: Vec<u64>,
    pub warmup: usize,
    pub iters: usize,
    /// The file NCCL logs to for this rank, to read the transport from.
    pub nccl_log: Option<PathBuf>,
    /// The network NCCL must have chosen (`Libfabric`, `IB`, `Socket`...).
    pub expect_network: Option<String>,
    /// How long one operation may take to complete before the group is aborted.
    pub op_timeout: Duration,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            ops: Op::ALL.to_vec(),
            dtypes: vec![CollectiveDtype::F32, CollectiveDtype::Bf16],
            sizes: vec![8, 1 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20, 1 << 30],
            warmup: 5,
            iters: 20,
            nccl_log: None,
            expect_network: None,
            op_timeout: Duration::from_secs(120),
        }
    }
}

/// An exactly representable input value: element `i` of `rank`'s buffer, small integers, so every sum over the ranks
/// is exact in f32 and bf16 alike.
fn input(rank: usize, i: usize) -> f32 {
    (((i * 7 + rank * 13) % 17) as i32 - 8) as f32
}

fn inputs(rank: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| input(rank, i)).collect()
}

/// `values` as the words of a bf16 buffer (two elements a word, low half first).
fn bf16_words(values: &[f32]) -> Vec<u32> {
    values.chunks(2).map(|p| (p[0].to_bits() >> 16) | (p.get(1).map_or(0, |x| x.to_bits() >> 16) << 16)).collect()
}

fn from_bf16_words(words: &[f32], n: usize) -> Vec<f32> {
    words.iter().flat_map(|w| [w.to_bits() << 16, w.to_bits() & 0xffff_0000]).take(n).map(f32::from_bits).collect()
}

/// Words a buffer of `n` elements of `dtype` occupies.
fn words(dtype: CollectiveDtype, n: u64) -> u64 {
    match dtype {
        CollectiveDtype::F32 => n,
        CollectiveDtype::Bf16 => n.div_ceil(2),
    }
}

fn upload(gpu: &Gpu, dtype: CollectiveDtype, values: &[f32]) -> DeviceBuffer {
    match dtype {
        CollectiveDtype::F32 => {
            let buf = gpu.storage(values.len() as u64);
            gpu.write_f32(&buf, values);
            buf
        }
        CollectiveDtype::Bf16 => {
            let buf = gpu.storage(words(dtype, values.len() as u64));
            gpu.write(&buf, &bf16_words(values));
            buf
        }
    }
}

fn download(gpu: &Gpu, dtype: CollectiveDtype, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
    match dtype {
        CollectiveDtype::F32 => gpu.read(buf, n),
        CollectiveDtype::Bf16 => from_bf16_words(&gpu.read(buf, n.div_ceil(2)), n),
    }
}

/// The buffers of one case: what each rank sends and receives, by element count.
struct Shape {
    /// Elements of the buffer the operation reads on each rank.
    send: u64,
    /// Elements of the buffer it writes (the same buffer for an all-reduce and a broadcast).
    recv: u64,
}

/// The per-rank buffers of an operation of `bytes` over `world` ranks, or `None` when the size is too small to give
/// every rank one element.
fn shape(op: Op, dtype: CollectiveDtype, bytes: u64, world: u64) -> Option<Shape> {
    let total = bytes / dtype.bytes();
    match op {
        Op::AllReduce | Op::Broadcast => (total > 0).then_some(Shape { send: total, recv: total }),
        Op::AllGather => {
            let part = total / world;
            (part > 0).then_some(Shape { send: part, recv: part * world })
        }
        Op::ReduceScatter => {
            let part = total / world;
            (part > 0).then_some(Shape { send: part * world, recv: part })
        }
    }
}

/// Run `op` `repeat` times back to back on `group` over the case's buffers and wait for them: the time they took on
/// the device when the provider measures that (CUDA events around them on the communicator's stream), else `None`.
fn run_op(group: &RankGroup, op: Op, dtype: CollectiveDtype, s: &Shape, send: &DeviceBuffer, recv: &DeviceBuffer, repeat: usize, timeout: Duration) -> Result<Option<Duration>, String> {
    let mut enqueue = || {
        for _ in 0..repeat {
            match op {
                Op::AllReduce => group.all_reduce_sum(&[(send, s.send)], dtype, 1.0),
                Op::AllGather => group.all_gather(send, recv, s.send, dtype),
                Op::ReduceScatter => group.reduce_scatter(send, recv, s.recv, dtype, 1.0),
                Op::Broadcast => group.broadcast(send, s.send, dtype, 0),
            }?;
        }
        Ok(())
    };
    group.timed(&mut enqueue, timeout).map_err(|e| format!("{} {}: {e}", op.name(), dtype_name(dtype)))
}

/// Operations timed together as one sample, so the sample is the collective's own time: the first operation of a
/// batch also waits for the slowest rank to reach it (the host's work between samples skews the ranks by tens of
/// microseconds), the rest run back to back. Small operations take a batch of up to 32; large ones, whose time
/// dwarfs the skew, one.
fn batch_for(bytes: u64) -> usize {
    (((64u64 << 20) / bytes.max(1)) as usize).clamp(1, 32)
}

/// What `op` must leave in the received buffer of `rank`, from every rank's [`inputs`].
fn reference(op: Op, s: &Shape, rank: usize, world: usize) -> Vec<f32> {
    let sum = |i: usize| (0..world).map(|r| input(r, i)).sum::<f32>();
    match op {
        Op::AllReduce => (0..s.send as usize).map(sum).collect(),
        Op::Broadcast => inputs(0, s.send as usize),
        Op::AllGather => (0..world).flat_map(|r| inputs(r, s.send as usize)).collect(),
        Op::ReduceScatter => (0..s.recv as usize).map(|i| sum(rank * s.recv as usize + i)).collect(),
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// `text` from every rank, in rank order, over the job's host transport (one byte to an f32, so it arrives exactly).
fn gather_text(host: &dyn Collective, rank: usize, text: &str) -> Result<Vec<String>, String> {
    let mut bytes: Vec<f32> = text.bytes().take(MAX_TEXT).map(f32::from).collect();
    bytes.resize(MAX_TEXT, 0.0);
    let all = pollster::block_on(host.all_gather(rank, Payload::f32(bytes))).map_err(|e| e.to_string())?.data;
    Ok(all.chunks(MAX_TEXT).map(|c| String::from_utf8_lossy(&c.iter().take_while(|&&b| b != 0.0).map(|&b| b as u8).collect::<Vec<u8>>()).into_owned()).collect())
}

fn gather_numbers(host: &dyn Collective, rank: usize, values: Vec<f32>) -> Result<Vec<Vec<f32>>, String> {
    let len = values.len();
    let all = pollster::block_on(host.all_gather(rank, Payload::f32(values))).map_err(|e| e.to_string())?.data;
    Ok(all.chunks(len.max(1)).map(<[f32]>::to_vec).collect())
}

/// What NCCL's own log says about the transport: the network it chose and the path of every channel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Transport {
    pub network: Option<String>,
    pub channels: Vec<String>,
}

impl Transport {
    /// Read `log` (NCCL's `NCCL_DEBUG=INFO` output). Lines naming another process's log are not mixed in: the launcher
    /// gives each rank its own file.
    pub fn parse(log: &str) -> Transport {
        let mut transport = Transport::default();
        for line in log.lines().filter(|l| l.contains("NCCL INFO")) {
            if let Some(rest) = line.split("Using network ").nth(1) {
                transport.network = rest.split_whitespace().next().map(str::to_string);
            }
            if let Some(rest) = line.split(" via ").nth(1) {
                if let Some(path) = rest.split_whitespace().next() {
                    if !transport.channels.iter().any(|c| c == path) {
                        transport.channels.push(path.to_string());
                    }
                }
            }
        }
        transport.channels.sort();
        transport
    }

    fn encode(&self) -> String {
        format!("{}\u{1f}{}", self.network.as_deref().unwrap_or(""), self.channels.join(","))
    }

    fn decode(text: &str) -> Option<Transport> {
        let (network, channels) = text.split_once('\u{1f}')?;
        Some(Transport { network: (!network.is_empty()).then(|| network.to_string()), channels: channels.split(',').filter(|c| !c.is_empty()).map(str::to_string).collect() })
    }
}

/// Whether the ranks' transports are what `expected` asks for: every rank chose that network, and when the job spans
/// several hosts some channel crosses it (`NET/<expected>/...`). `Err` names what is wrong.
pub fn check_transport(transports: &[Option<Transport>], hosts: usize, expected: &str) -> Result<(), String> {
    for (rank, t) in transports.iter().enumerate() {
        let Some(t) = t else { return Err(format!("rank {rank} has no NCCL log to read the transport from")) };
        match t.network.as_deref() {
            Some(n) if n == expected => {}
            Some(n) => return Err(format!("rank {rank} uses network {n}, not {expected}")),
            None => return Err(format!("rank {rank}'s NCCL log names no network")),
        }
        if let Some(other) = t.channels.iter().find(|c| c.starts_with("NET/") && !c.starts_with(&format!("NET/{expected}"))) {
            return Err(format!("rank {rank} has a channel via {other}, not {expected}"));
        }
    }
    let crosses = transports.iter().flatten().any(|t| t.channels.iter().any(|c| c.starts_with(&format!("NET/{expected}"))));
    if hosts > 1 && !crosses {
        return Err(format!("the job spans {hosts} hosts but no channel goes via NET/{expected}"));
    }
    Ok(())
}

/// The octets every Slingshot (CXI) NIC of this host has sent and received, from its telemetry; empty where the
/// node exposes none.
fn cxi_octets() -> BTreeMap<String, (u64, u64)> {
    let mut nics = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/cxi") else { return nics };
    for entry in entries.flatten() {
        let telemetry = entry.path().join("device").join("telemetry");
        let read = |name: &str| std::fs::read_to_string(telemetry.join(name)).ok().and_then(|t| t.split('@').next().and_then(|v| v.trim().parse::<u64>().ok()));
        if let (Some(tx), Some(rx)) = (read("hni_sts_tx_ok_octets"), read("hni_sts_rx_ok_octets")) {
            nics.insert(entry.file_name().to_string_lossy().into_owned(), (tx, rx));
        }
    }
    nics
}

/// The communication libraries this process has loaded (NCCL, its network plugin, libfabric, CXI, the CUDA driver).
fn loaded_libraries() -> Option<Vec<String>> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut libs: Vec<String> = maps
        .lines()
        .filter_map(|l| l.split_whitespace().nth(5))
        .filter_map(|p| Path::new(p).file_name().map(|f| f.to_string_lossy().into_owned()))
        .filter(|f| ["nccl", "fabric", "cxi", "libcuda.so", "gdrapi"].iter().any(|k| f.contains(k)))
        .collect();
    libs.sort();
    libs.dedup();
    Some(libs)
}

fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|h| h.trim().to_string()).unwrap_or_default()
}

/// Run the benchmark as this rank of `ctx`'s job on `gpu`. Every rank measures and checks; rank 0 returns the result
/// document (the others `None`). `Err` when the group cannot form or an operation fails.
pub fn run(ctx: &JobContext, gpu: &Gpu, cfg: &Config) -> Result<Option<Value>, String> {
    let rank = ctx.rank();
    let world = ctx.plan().world_size;
    let job = ctx.job_collective().map_err(|e| format!("joining the job: {e}"))?;
    let host: &dyn Collective = &job;
    let join_timeout = Duration::from_millis(ctx.plan().handshake_timeout_ms);
    let group = rendezvous(host, rank, RankGroup::supported(gpu), || RankGroup::id(gpu), |id| RankGroup::join(gpu, id, world, rank, join_timeout))
        .map_err(|e| format!("the job's transport: {e}"))?
        .map_err(|e| format!("no device group: {e}"))?;
    // Bounded on the card as a training job's group is, so the times are what the trainer's collectives cost.
    group.bound_in_flight(cfg.op_timeout).map_err(|e| format!("bounding the device group's operations: {e}"))?;
    let device = gpu_core::devices::selected_device();
    let hosts = gather_text(host, rank, &host_name())?;
    let devices = gather_text(host, rank, &device.map_or(String::new(), |d| format!("{}\u{1f}{}", d.identity.name, d.identity.pci_bus.clone().unwrap_or_default())))?;
    let nic_before = cxi_octets();
    let started_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());

    // The sweep: every rank times every case; the slowest rank's median sets the bandwidth.
    let mut cases = Vec::new();
    let mut timings = Vec::new();
    let mut device_timed = true;
    for &dtype in &cfg.dtypes {
        for &op in &cfg.ops {
            for &bytes in &cfg.sizes {
                let Some(s) = shape(op, dtype, bytes, world as u64) else { continue };
                let send = upload(gpu, dtype, &inputs(rank, s.send as usize));
                let recv = if matches!(op, Op::AllGather | Op::ReduceScatter) { gpu.storage(words(dtype, s.recv)) } else { gpu.storage(1) };
                // The uploads above are on the device before the first timed operation starts.
                gpu.flush();
                let mut times = Vec::with_capacity(cfg.iters);
                for i in 0..cfg.warmup + cfg.iters {
                    let t = Instant::now();
                    let batch = batch_for(bytes);
                    let device = run_op(&group, op, dtype, &s, &send, &recv, batch, cfg.op_timeout)?;
                    device_timed &= device.is_some();
                    if i >= cfg.warmup {
                        times.push(device.unwrap_or_else(|| t.elapsed()).as_secs_f64() * 1e6 / batch as f64);
                    }
                }
                times.sort_by(f64::total_cmp);
                timings.push(percentile(&times, 0.5) as f32);
                timings.push(percentile(&times, 0.99) as f32);
                cases.push((op, dtype, bytes, s));
            }
        }
    }
    let nic_after = cxi_octets();

    // The check: every operation once on fresh inputs of an awkward length, against the host reference.
    let mut checks = Vec::new();
    let mut check_flags = Vec::new();
    let check_elems = 1001 * world as u64;
    for &dtype in &cfg.dtypes {
        for &op in &cfg.ops {
            let s = shape(op, dtype, check_elems * dtype.bytes(), world as u64).expect("a non-empty check");
            let send = upload(gpu, dtype, &inputs(rank, s.send as usize));
            let recv = if matches!(op, Op::AllGather | Op::ReduceScatter) { gpu.storage(words(dtype, s.recv)) } else { gpu.storage(1) };
            run_op(&group, op, dtype, &s, &send, &recv, 1, cfg.op_timeout)?;
            let out_buf = if matches!(op, Op::AllGather | Op::ReduceScatter) { &recv } else { &send };
            let n = if matches!(op, Op::AllGather | Op::ReduceScatter) { s.recv } else { s.send } as usize;
            let got = download(gpu, dtype, out_buf, n);
            let want = reference(op, &s, rank, world);
            let max_error = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
            check_flags.push(if got.len() == want.len() && max_error == 0.0 { 1.0 } else { 0.0 });
            check_flags.push(max_error);
            checks.push((op, dtype, n));
        }
    }

    // What every rank saw, to rank 0.
    let transport = cfg.nccl_log.as_deref().and_then(|p| std::fs::read_to_string(p).ok()).map(|log| Transport::parse(&log));
    let transports: Vec<Option<Transport>> = gather_text(host, rank, &transport.as_ref().map_or(String::new(), Transport::encode))?.iter().map(|t| Transport::decode(t)).collect();
    let all_timings = gather_numbers(host, rank, timings)?;
    let all_checks = gather_numbers(host, rank, check_flags)?;
    let nic_rows = |m: &BTreeMap<String, (u64, u64)>| m.iter().map(|(k, (tx, rx))| format!("{k}={tx}:{rx}")).collect::<Vec<_>>().join(",");
    let nics = gather_text(host, rank, &format!("{}|{}", nic_rows(&nic_before), nic_rows(&nic_after)))?;
    if rank != 0 {
        return Ok(None);
    }

    let mut distinct_hosts: Vec<&String> = hosts.iter().collect();
    distinct_hosts.sort();
    distinct_hosts.dedup();
    let measurements: Vec<Value> = cases
        .iter()
        .enumerate()
        .map(|(c, (op, dtype, bytes, s))| {
            let p50s: Vec<f64> = all_timings.iter().map(|t| f64::from(t[2 * c])).collect();
            let p99s: Vec<f64> = all_timings.iter().map(|t| f64::from(t[2 * c + 1])).collect();
            let p50 = p50s.iter().copied().fold(0.0, f64::max);
            let p99 = p99s.iter().copied().fold(0.0, f64::max);
            let algbw = *bytes as f64 / (p50 * 1e-6) / 1e9;
            json!({
                "op": op.name(), "dtype": dtype_name(*dtype), "bytes": bytes, "elements_sent": s.send, "elements_received": s.recv,
                "p50_us": p50, "p99_us": p99, "algbw_gb_s": algbw, "busbw_gb_s": algbw * op.bus_factor(world),
                "rank_p50_us": p50s,
            })
        })
        .collect();
    let correctness: Vec<Value> = checks
        .iter()
        .enumerate()
        .map(|(c, (op, dtype, n))| {
            let ok = all_checks.iter().all(|f| f[2 * c] == 1.0);
            let max_error = all_checks.iter().map(|f| f64::from(f[2 * c + 1])).fold(0.0, f64::max);
            json!({"op": op.name(), "dtype": dtype_name(*dtype), "elements": n, "ok": ok, "max_abs_error": max_error})
        })
        .collect();
    let correct = correctness.iter().all(|c| c["ok"] == json!(true));
    let transport_verdict = cfg.expect_network.as_deref().map(|net| check_transport(&transports, distinct_hosts.len(), net));
    // NIC octets per host, from the first rank on it.
    let mut nic_octets = serde_json::Map::new();
    for (r, text) in nics.iter().enumerate() {
        if hosts[..r].contains(&hosts[r]) {
            continue;
        }
        let parse = |s: &str| -> BTreeMap<String, (u64, u64)> {
            s.split(',').filter_map(|row| {
                let (nic, v) = row.split_once('=')?;
                let (tx, rx) = v.split_once(':')?;
                Some((nic.to_string(), (tx.parse().ok()?, rx.parse().ok()?)))
            }).collect()
        };
        let (before, after) = text.split_once('|').unwrap_or(("", ""));
        let (before, after) = (parse(before), parse(after));
        if after.is_empty() {
            continue;
        }
        let per_nic: serde_json::Map<String, Value> = after
            .iter()
            .filter_map(|(nic, (tx, rx))| before.get(nic).map(|(tx0, rx0)| (nic.clone(), json!({"tx_octets": tx.saturating_sub(*tx0), "rx_octets": rx.saturating_sub(*rx0)}))))
            .collect();
        nic_octets.insert(hosts[r].clone(), Value::Object(per_nic));
    }
    let device_rows: Vec<Value> = devices
        .iter()
        .map(|d| {
            let (name, pci) = d.split_once('\u{1f}').unwrap_or((d.as_str(), ""));
            json!({"name": (!name.is_empty()).then_some(name), "pci_bus": (!pci.is_empty()).then_some(pci)})
        })
        .collect();
    let verdict = correct && transport_verdict.as_ref().is_none_or(|v| v.is_ok());
    Ok(Some(json!({
        "kind": "collectives",
        "format": FORMAT,
        "started_unix_s": started_at,
        "job": ctx.plan().job,
        "hardware": {
            "world": world,
            "hosts": hosts,
            "nodes": distinct_hosts.len(),
            "devices": device_rows,
        },
        "software": {
            "brain": env!("CARGO_PKG_VERSION"),
            "backend": gpu_core::backend_name(),
            "collective_provider": group.provider(),
            "collective_library_version": group.library_version(),
            "loaded_libraries": loaded_libraries(),
        },
        "config": {
            "ops": cfg.ops.iter().map(|o| o.name()).collect::<Vec<_>>(),
            "dtypes": cfg.dtypes.iter().map(|d| dtype_name(*d)).collect::<Vec<_>>(),
            "sizes": cfg.sizes,
            "warmup": cfg.warmup,
            "iters": cfg.iters,
            "op_timeout_ms": cfg.op_timeout.as_millis() as u64,
            // How each iteration was timed: device events around the operation on the communicator's stream, or the
            // host's wall clock from enqueue until it saw the operation complete (which includes its polling).
            "timing": if device_timed { "device_events" } else { "host_wall" },
            "plan": ctx.plan(),
        },
        "transport": {
            "ranks": transports.iter().map(|t| t.as_ref().map(|t| json!({"network": t.network, "channels": t.channels}))).collect::<Vec<_>>(),
            "expected_network": cfg.expect_network,
            "ok": transport_verdict.as_ref().map(|v| v.is_ok()),
            "reason": transport_verdict.and_then(|v| v.err()),
        },
        "measurements": measurements,
        "nic_octets": (!nic_octets.is_empty()).then_some(Value::Object(nic_octets)),
        "correctness": correctness,
        "verdict": if verdict { "pass" } else { "fail" },
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_dtypes_and_ops_parse_and_bus_factors_follow_the_convention() {
        assert_eq!((parse_size("8"), parse_size("64K"), parse_size("1G"), parse_size("0"), parse_size("x")), (Some(8), Some(65536), Some(1 << 30), None, None));
        assert_eq!(parse_dtype("bf16"), Some(CollectiveDtype::Bf16));
        assert_eq!(Op::parse("reduce_scatter"), Some(Op::ReduceScatter));
        assert_eq!(Op::AllReduce.bus_factor(4), 1.5);
        assert_eq!(Op::AllGather.bus_factor(4), 0.75);
        assert_eq!(Op::Broadcast.bus_factor(8), 1.0);
        let s = shape(Op::AllGather, CollectiveDtype::F32, 1 << 20, 4).unwrap();
        assert_eq!((s.send, s.recv), (1 << 16, 1 << 18), "an all-gather's size is the gathered buffer");
        assert!(shape(Op::ReduceScatter, CollectiveDtype::Bf16, 4, 4).is_none(), "too small to give every rank an element");
    }

    #[test]
    fn the_reference_is_exact_for_every_op() {
        let s = shape(Op::ReduceScatter, CollectiveDtype::F32, 4 * 6, 2).unwrap();
        let rs = reference(Op::ReduceScatter, &s, 1, 2);
        assert_eq!(rs, (3..6).map(|i| input(0, i) + input(1, i)).collect::<Vec<_>>());
        let g = shape(Op::AllGather, CollectiveDtype::F32, 4 * 6, 2).unwrap();
        assert_eq!(reference(Op::AllGather, &g, 0, 2), [inputs(0, 3), inputs(1, 3)].concat());
        assert_eq!(from_bf16_words(&bf16_words(&[1.0, -3.0, 5.0]).iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(), 3), vec![1.0, -3.0, 5.0]);
    }

    #[test]
    fn the_transport_is_read_from_nccls_log_and_a_socket_fallback_fails_the_check() {
        let fabric = Transport::parse(
            "n442:1:1 [0] NCCL INFO Using network Libfabric\n\
             n442:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM\n\
             n442:1:1 [0] NCCL INFO Channel 00/0 : 3[3] -> 4[0] [send] via NET/Libfabric/0/GDRDMA\n\
             unrelated line via nothing\n",
        );
        assert_eq!(fabric.network.as_deref(), Some("Libfabric"));
        assert_eq!(fabric.channels, ["NET/Libfabric/0/GDRDMA", "P2P/CUMEM"]);
        let socket = Transport::parse("x NCCL INFO Using network Socket\nx NCCL INFO Channel 00/0 : 3[3] -> 4[0] [send] via NET/Socket/0\n");
        assert!(check_transport(&[Some(fabric.clone()), Some(fabric.clone())], 2, "Libfabric").is_ok());
        assert!(check_transport(&[Some(fabric.clone()), Some(socket)], 2, "Libfabric").unwrap_err().contains("Socket"));
        let local = Transport::parse("x NCCL INFO Using network Libfabric\nx NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM\n");
        assert!(check_transport(&[Some(local.clone())], 1, "Libfabric").is_ok(), "one host: no channel needs the network");
        assert!(check_transport(&[Some(local)], 2, "Libfabric").unwrap_err().contains("no channel"));
        assert!(check_transport(&[None], 1, "Libfabric").is_err(), "no log is no evidence");
        assert_eq!(Transport::decode(&fabric.encode()), Some(fabric));
    }
}
