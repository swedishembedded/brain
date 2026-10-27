// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The board has 33 MB of RAM visible to Linux and 24 MB of CMA shared by the
//! camera, the encoder and the NPU, so the sample reports what it uses.

/// One line: this process's resident and peak resident set, and the CMA allocated
/// system-wide (of the pool) and free system memory it leaves for others.
pub fn report() -> String {
    let field = |text: &str, key: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(key))
            .map_or_else(|| "?".to_string(), |v| v.trim_start_matches(':').trim().to_string())
    };
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    format!(
        "VmRSS {} | VmHWM {} | CmaAllocated {} of {} | MemAvailable {}",
        field(&status, "VmRSS"),
        field(&status, "VmHWM"),
        field(&meminfo, "CmaAllocated"),
        field(&meminfo, "CmaTotal"),
        field(&meminfo, "MemAvailable"),
    )
}
