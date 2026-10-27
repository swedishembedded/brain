// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A job: a chain of tasks and the one contiguous arena every address in them
//! points into.
//!
//! The arena holds the data region first (tensors, weights, biases; its layout
//! is the caller's) and the task programs after it, so the caller can assign
//! data addresses before it knows how many tasks the layers produce. Every
//! program's tail links to the next, so the whole chain runs as one
//! submission and the CPU is not involved between tasks.

use crate::driver::{Buffer, Device, TaskEntry};
use crate::regs::{Program, PROFILE, PROGRAM_BYTES, PROGRAM_WORDS};
use crate::Result;

/// One task: a register program and the engines it starts.
#[derive(Clone)]
pub struct Task {
    pub program: Program,
    pub enable: u32,
    pub int_mask: u32,
}

/// Interrupt sources the driver clears before each task (all of them).
const INT_CLEAR_ALL: u32 = 0x1ffff;

/// A built job, resident in NPU memory.
pub struct Job {
    arena: Buffer,
    table: Buffer,
    data_bytes: usize,
    count: u32,
}

impl Job {
    /// Allocate the arena (`data_bytes` of data plus the programs), encode and
    /// link the programs and fill the task table. The data region is zeroed;
    /// the caller writes its constants through [`Job::data_mut`].
    pub fn new(device: &Device, data_bytes: usize, tasks: &[Task]) -> Result<Job> {
        let data_bytes = data_bytes.next_multiple_of(PROGRAM_BYTES.max(64));
        let mut arena = device.alloc(data_bytes + tasks.len() * PROGRAM_BYTES, false)?;
        let mut table = device.alloc(tasks.len().max(1) * std::mem::size_of::<TaskEntry>(), true)?;
        arena.as_mut_slice().fill(0);
        let base = arena.dma_addr();
        let mut words = [0u64; PROGRAM_WORDS];
        for (i, task) in tasks.iter().enumerate() {
            let offset = data_bytes + i * PROGRAM_BYTES;
            let next = (i + 1 < tasks.len()).then(|| ((offset + PROGRAM_BYTES) as u32, PROFILE.len()));
            task.program.encode(&mut words, next, task.enable);
            for (j, w) in words.iter().enumerate() {
                arena.as_mut_slice()[offset + j * 8..offset + j * 8 + 8].copy_from_slice(&w.to_le_bytes());
            }
            let entry = TaskEntry {
                op_idx: i as u32 + 1,
                enable_mask: task.enable,
                int_mask: task.int_mask,
                int_clear: INT_CLEAR_ALL,
                regcfg_amount: PROFILE.len() as u32,
                regcmd_addr: u64::from(base) + offset as u64,
                ..Default::default()
            };
            let at = i * std::mem::size_of::<TaskEntry>();
            // SAFETY: TaskEntry is plain-old-data with no padding (packed).
            let bytes = unsafe { std::slice::from_raw_parts((&entry as *const TaskEntry).cast::<u8>(), std::mem::size_of::<TaskEntry>()) };
            table.as_mut_slice()[at..at + bytes.len()].copy_from_slice(bytes);
        }
        let job = Job { arena, table, data_bytes, count: tasks.len() as u32 };
        job.table.sync_to_device(0..job.table.len())?;
        Ok(job)
    }

    /// The data region (tensors and constants), as the CPU sees it.
    pub fn data(&self) -> &[u8] {
        &self.arena.as_slice()[..self.data_bytes]
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.arena.as_mut_slice()[..self.data_bytes]
    }

    /// The arena as a dma-buf, for zero-copy producers writing an input.
    pub fn arena_dmabuf_fd(&self) -> std::os::fd::RawFd {
        self.arena.dmabuf_fd()
    }

    pub fn task_count(&self) -> u32 {
        self.count
    }

    /// Bytes of NPU memory this job holds.
    pub fn resident_bytes(&self) -> usize {
        self.arena.len() + self.table.len()
    }

    /// Make CPU writes to `range` of the data region visible to the NPU.
    pub fn flush(&self, range: std::ops::Range<usize>) -> Result<()> {
        self.arena.sync_to_device(range)
    }

    /// Make NPU writes to `range` of the data region visible to the CPU.
    pub fn invalidate(&self, range: std::ops::Range<usize>) -> Result<()> {
        self.arena.sync_from_device(range)
    }

    /// Flush the whole arena (constants and programs). Call once after
    /// writing the constants.
    pub fn flush_all(&self) -> Result<()> {
        self.arena.sync_to_device(0..self.arena.len())
    }

    /// Run tasks `range` (all of them by default) as one submission and block
    /// until the last one finishes.
    pub fn run(&self, device: &Device, range: Option<std::ops::Range<u32>>) -> Result<()> {
        let range = range.unwrap_or(0..self.count);
        if range.is_empty() {
            return Ok(());
        }
        device.submit(&self.table, range.start, range.end - range.start, self.arena.dma_addr(), PROFILE.len() as u32)
    }
}
