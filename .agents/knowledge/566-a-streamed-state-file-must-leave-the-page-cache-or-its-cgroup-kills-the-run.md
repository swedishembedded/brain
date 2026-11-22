<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 566. A streamed state file must leave the page cache, or its cgroup kills the run

**Lesson:** an optimizer state kept in a file (`ResidentFtConfig::state_dir`, `optim::storage::FileBacking`) is read
and rewritten tile by tile once per step and not touched again until the next one. Through plain `pread`/`pwrite`
every one of those pages stays in the page cache, and on Linux the page cache is charged to the process's memory
cgroup. When the state is several times the process's own memory (the Inkling 2 26B full fine-tune: about 300 GB of
f32 master and AdamW moments against about 90 GB resident), the dirty pages cannot be written back as fast as the
update produces them and the cgroup's limit is reached with the resident set at a fifth of it: Slurm reported an
`oom_kill` of the step. `FileBacking` now drops every read range from the cache (`posix_fadvise(DONTNEED)`) and
starts the write-back of every written range, waiting for it and dropping it when the next range is written
(`sync_file_range`; at most two ranges dirty, the disk writing one while the update computes the next). Linux, best
effort.

## Evidence (2026-11-22)

- `inkling2` `real_full_ft` on a GH200 node whose job cgroup allows 440 GB: before the change the step was OOM-killed
  during the first update (VmHWM 93 GB, the state file being written); after it the run's cgroup read 52.3 GB after
  the first step with the state file at its full size on the node's disk.
- The price is the update's speed: the first step took 268 s before the kill and 471 s with a synchronous write-back of
  each range (one card, one 20-token row); the pipelined form above replaced it.
