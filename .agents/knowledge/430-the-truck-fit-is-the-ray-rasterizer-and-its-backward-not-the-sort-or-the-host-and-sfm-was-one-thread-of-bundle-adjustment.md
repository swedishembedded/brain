<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 430. The truck fit is the ray rasterizer and its backward, not the sort or the host; structure from motion was one thread of bundle adjustment

Question (the brief of 2026-11-02): where do the 1139 s of structure from motion and the 4.2 s a step of `brain splat train` on the 251-photograph Tanks and Temples truck go (GH200, `--max-width 1024`, 979x546, 9.1M gaussians)?

## Measured

- **Structure from motion**: 1247 s (CLI, host matcher, whole node), of which features 14 s, matching 77 s, focal-length search 434 s, global initializer 172 s, lens models 178 s, two re-triangulations and the world 316 s. The CPU samples were 57 % pool spinning: every one of the 19 focal candidates, the initializer, the lens fit and the re-triangulations was a chain of bundle adjustments whose linearization, Schur complement and dense Cholesky of the 1500-column camera system ran on ONE thread. Now each camera owns its rows of the system, the calibration rows are summed per chunk in a fixed order, the Cholesky is by panels on the pool and pairs verify concurrently: 319 s on 72 cores (focal search 178 s, matching 69 s, solve 63 s), 251/251 placed, 0.68 px rms, the same focal length; in `train`, 234-248 s. A 84-photograph subset: solve 62 s -> 14 s.
- **Multi-view stereo**: 57-68 s, no change attempted; it is 1.5 % of a run.
- **Fit step** (2 views, 6.56M gaussians, 979x546, device time per call): `splat_ray_rasterize` 49 ms (57 % of the profile; ncu: 71 % SM throughput, IPC 2.9, 90 registers, 28 % occupancy, DRAM 0.3 %: it is instruction-bound on the pair evaluation, not memory-bound), `splat_ray_bwd_tile` 74 ms and `splat_ray_bwd_walk` 51-60 ms per view, `splat_ray_emit` 5 ms, `splat_grad_reduce` 9 ms. At a lower resolution level the rasterizer costs the SAME (pixels x list length is 256 x instances whatever the tile count), so the coarse levels are not cheap.
- **Sort and scan**: 12.3 s of 44.8 s device time: one thread walked 256 keys with a 256-bin local table. 256 -> 12.3 s, 512 -> 8.9, 1024 -> 7.3, 2048 -> 7.0, 4096 -> 8.8; the chunk now follows the key count (256..2048): 7.1 s, whole profile pass 52.6 s -> 47.0 s. Still 2.3 ms a scatter pass for 10-25M keys against ~0.1 ms at the bandwidth bound: a CUDA candidate for the sort is the next gain.
- **Density rounds** (every `iters/20` steps until 60 %): one round is ~95-125 s at 6.5-7.3M gaussians, a pass over every view (render 83 ms, diagnose + host 77 ms, full backward 176 ms, accumulate 25 ms -> 3 ms). The backward exists only to read the colour channel back; a colour-only backward is the largest saving left in the rounds. 11 rounds are about as much time as the 3000 steps.
- The stage clocks of a step ("render forward", "absgrad readback") absorb the wait for the previous step's queued device work, so a large "readback" is device time, not transfer: attribute with the kernel table.

## Rules

- A serial O(n^3) or O(obs * tracks^2) step inside an iterated solver is the first thing to look for in a stage that burns hundreds of CPU-minutes at 19 cores average; a sampled profile that is mostly pool spin says the parallelism is in the wrong place.
- Chunk size of a thread-per-chunk device sort is a per-key-count decision, not a constant.
