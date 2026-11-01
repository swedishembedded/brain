// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Process-wide runtime options a backend honours when it opens a device: submission capture, the stream-ordered
//! memory pool and the retention cap of freed device blocks. The application sets them from its flags before it builds
//! any device (`brain --no-graphs`, `--no-mempool`, `--block-cache-mb`); a backend reads them when it opens one. Nothing
//! here is read from the environment, so a changed environment cannot change how a device runs.
//!
//! Swedish Embedded AB implements solutions for predictable GPU runtime behaviour for its clients. If your team needs
//! expertise in GPU runtime configuration then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NO_GRAPHS: AtomicBool = AtomicBool::new(false);
static NO_MEMPOOL: AtomicBool = AtomicBool::new(false);
/// `u64::MAX`: size the cap from the device's memory.
static BLOCK_CACHE_BYTES: AtomicU64 = AtomicU64::new(u64::MAX);

/// Whether a backend may capture repeated submissions into graphs and replay them (`true`, the default); `false`
/// launches every dispatch on its own, for bisecting a suspected capture bug.
pub fn set_graph_capture(on: bool) {
    NO_GRAPHS.store(!on, Ordering::Relaxed);
}

/// [`set_graph_capture`]'s setting.
pub fn graph_capture() -> bool {
    !NO_GRAPHS.load(Ordering::Relaxed)
}

/// Whether a backend may issue scratch blocks from a stream-ordered memory pool (`true`, the default); `false` keeps the
/// exact-size cache of freed blocks.
pub fn set_stream_pool(on: bool) {
    NO_MEMPOOL.store(!on, Ordering::Relaxed);
}

/// [`set_stream_pool`]'s setting.
pub fn stream_pool() -> bool {
    !NO_MEMPOOL.load(Ordering::Relaxed)
}

/// The most idle bytes a backend keeps of freed device blocks: `Some(bytes)` (`Some(0)` keeps none), or `None` to size it
/// from the device's memory (the default).
pub fn set_block_cache_bytes(bytes: Option<u64>) {
    BLOCK_CACHE_BYTES.store(bytes.unwrap_or(u64::MAX), Ordering::Relaxed);
}

/// [`set_block_cache_bytes`]'s setting.
pub fn block_cache_bytes() -> Option<u64> {
    Some(BLOCK_CACHE_BYTES.load(Ordering::Relaxed)).filter(|&b| b != u64::MAX)
}
