// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A job: a chain of task programs in NPU memory, linked so the whole chain
//! (or any contiguous run of it) executes as one submission without the CPU.
//!
//! Every address a program holds, its successor's included, is relative to
//! the job's base address. The data the tasks address lives in buffers the
//! caller owns; the base is the lowest of their DMA addresses (and the
//! program buffer's), so memory can be split across several modest
//! allocations instead of one large contiguous block the board may not have.

use crate::driver::{Buffer, Device, TaskEntry};
use crate::regs::{Program, PROFILE, PROGRAM_BYTES, PROGRAM_WORDS};
use crate::{Error, Result};

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
    programs: Buffer,
    table: Buffer,
    base: u32,
    count: u32,
}

impl Job {
    /// Bytes of program memory `tasks` tasks need.
    pub fn program_bytes(tasks: usize) -> usize {
        tasks.max(1) * PROGRAM_BYTES
    }

    /// Encode and link `tasks` into `programs` (allocated with at least
    /// [`Job::program_bytes`]) and fill the task table. `base` must not exceed
    /// the DMA address of `programs` or of anything the tasks address.
    pub fn new(device: &Device, mut programs: Buffer, tasks: &[Task], base: u32) -> Result<Job> {
        if programs.len() < Job::program_bytes(tasks.len()) || programs.dma_addr() < base {
            return Err(Error::Unsupported(format!("program buffer of {} bytes at {:#x} cannot hold {} tasks above base {base:#x}", programs.len(), programs.dma_addr(), tasks.len())));
        }
        let mut table = device.alloc(tasks.len().max(1) * std::mem::size_of::<TaskEntry>(), true)?;
        let relative = programs.dma_addr() - base;
        let mut words = [0u64; PROGRAM_WORDS];
        for (i, task) in tasks.iter().enumerate() {
            let offset = i * PROGRAM_BYTES;
            let next = (i + 1 < tasks.len()).then(|| (relative + (offset + PROGRAM_BYTES) as u32, PROFILE.len()));
            task.program.encode(&mut words, next, task.enable);
            for (j, w) in words.iter().enumerate() {
                programs.as_mut_slice()[offset + j * 8..offset + j * 8 + 8].copy_from_slice(&w.to_le_bytes());
            }
            let entry = TaskEntry {
                op_idx: i as u32 + 1,
                enable_mask: task.enable,
                int_mask: task.int_mask,
                int_clear: INT_CLEAR_ALL,
                regcfg_amount: PROFILE.len() as u32,
                regcmd_addr: u64::from(programs.dma_addr()) + offset as u64,
                ..Default::default()
            };
            let at = i * std::mem::size_of::<TaskEntry>();
            // SAFETY: TaskEntry is plain-old-data with no padding (packed).
            let bytes = unsafe { std::slice::from_raw_parts((&entry as *const TaskEntry).cast::<u8>(), std::mem::size_of::<TaskEntry>()) };
            table.as_mut_slice()[at..at + bytes.len()].copy_from_slice(bytes);
        }
        programs.sync_to_device(0..programs.len())?;
        table.sync_to_device(0..table.len())?;
        Ok(Job { programs, table, base, count: tasks.len() as u32 })
    }

    pub fn task_count(&self) -> u32 {
        self.count
    }

    /// Bytes of NPU memory the programs and the task table hold.
    pub fn resident_bytes(&self) -> usize {
        self.programs.len() + self.table.len()
    }

    /// Run tasks `range` (all of them when `None`) as one submission and
    /// block until the last one finishes.
    pub fn run(&self, device: &Device, range: Option<std::ops::Range<u32>>) -> Result<()> {
        let range = range.unwrap_or(0..self.count);
        if range.is_empty() {
            return Ok(());
        }
        device.submit(&self.table, range.start, range.end - range.start, self.base, PROFILE.len() as u32)
    }
}

/// The base address for a job over `buffers`: the lowest DMA address.
pub fn base_of<'a>(buffers: impl IntoIterator<Item = &'a Buffer>) -> u32 {
    buffers.into_iter().map(Buffer::dma_addr).min().unwrap_or(0)
}
