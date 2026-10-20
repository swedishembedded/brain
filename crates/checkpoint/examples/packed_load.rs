// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Plan and run a bounded load of named tensors from a real checkpoint and
//! print the `LoadPlan` and the measured staging numbers. A short smoke tool:
//!
//! ```text
//! cargo run --release -p brain-checkpoint --example packed_load -- \
//!     --path <file-or-dir> --policy preserve|nf4|int8|bf16|repack-int8 --budget-mib 64 --tile-kib 4096 \
//!     [--tensor NAME]... [--limit N]
//! ```
//!
//! Without `--tensor` it takes the `--limit` largest tensors. The destination is
//! a discarding sink, so the process holds only the staging budget.

use std::path::PathBuf;

use checkpoint::packed::{BoundedWorkspace, CodecRegistry, LoadConfig, LoadPolicy, Loader, MemoryBudget, PackedReader, PackedTensorReader, Pool, QuantEncoding, TensorDescriptor, TensorSink, Tile};

struct Discard(u64);

impl TensorSink for Discard {
    fn begin(&mut self, _: &TensorDescriptor) -> Result<(), String> {
        Ok(())
    }
    fn write_tile(&mut self, _: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        self.0 += tile.payload.len() as u64;
        Ok(())
    }
    fn finish(&mut self, _: &TensorDescriptor) -> Result<(), String> {
        Ok(())
    }
    fn abort(&mut self, _: &TensorDescriptor) {}
}

fn main() -> Result<(), String> {
    let (mut path, mut policy, mut budget_mib, mut tile_kib, mut limit) = (None, "preserve".to_string(), 64u64, 4096u64, 3usize);
    let mut tensors: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |what: &str| args.next().ok_or_else(|| format!("{what} needs a value"));
        match a.as_str() {
            "--path" => path = Some(PathBuf::from(val("--path")?)),
            "--policy" => policy = val("--policy")?,
            "--budget-mib" => budget_mib = val("--budget-mib")?.parse().map_err(|e| format!("--budget-mib: {e}"))?,
            "--tile-kib" => tile_kib = val("--tile-kib")?.parse().map_err(|e| format!("--tile-kib: {e}"))?,
            "--limit" => limit = val("--limit")?.parse().map_err(|e| format!("--limit: {e}"))?,
            "--tensor" => tensors.push(val("--tensor")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let path = path.ok_or("--path is required")?;
    let policy = match policy.as_str() {
        "preserve" => LoadPolicy::Preserve,
        "nf4" => LoadPolicy::Convert(QuantEncoding::nf4_blocked(32)),
        "int8" => LoadPolicy::Convert(QuantEncoding::int8_blocked(32)),
        "bf16" => LoadPolicy::Convert(QuantEncoding::bf16()),
        "repack-int8" => LoadPolicy::Repack(QuantEncoding::int8_blocked(32)),
        other => return Err(format!("--policy {other}: expected preserve, nf4, int8, bf16 or repack-int8")),
    };
    let reader = PackedReader::open(&path)?;
    if tensors.is_empty() {
        let mut sized: Vec<(u64, String)> = reader.names().into_iter().filter_map(|n| reader.descriptor(&n).ok().map(|d| (d.payload_bytes(), n))).collect();
        sized.sort_by(|a, b| b.cmp(a));
        tensors = sized.into_iter().take(limit).map(|(_, n)| n).collect();
    }
    let budget = MemoryBudget::new(budget_mib << 20, 0);
    let codecs = CodecRegistry::reference();
    let loader = Loader { reader: &reader, codecs: &codecs, workspace: BoundedWorkspace::new(budget.clone()), config: LoadConfig { tile_bytes: tile_kib << 10 } };
    for name in &tensors {
        let step = loader.plan(name, &policy)?;
        print!("{}", checkpoint::packed::LoadPlan { steps: vec![step.clone()] });
        let mut sink = Discard(0);
        let rep = loader.execute(&step, &mut sink)?;
        println!(
            "  {} tiles, {} B read, {} B written, staging peak {} B (budget {} B), file-page growth {:?} B",
            rep.tiles_written,
            rep.bytes_read,
            sink.0,
            rep.peak_staging_bytes,
            budget.stats(Pool::HostStaging).capacity,
            rep.file_resident_growth_bytes
        );
    }
    println!("budget high-water {} B", budget.stats(Pool::HostStaging).high_water);
    // Process-level cross-check of the budget's own numbers (Linux).
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines().filter(|l| ["VmHWM:", "RssAnon:", "RssFile:"].iter().any(|k| l.starts_with(k))) {
            println!("{}", line.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }
    Ok(())
}
