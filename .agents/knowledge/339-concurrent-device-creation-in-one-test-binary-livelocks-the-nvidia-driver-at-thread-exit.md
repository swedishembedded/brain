<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 339. Concurrent device creation in one test binary livelocks the NVIDIA driver at thread exit

Symptom: `crates/model/tests/ops_kquant.rs` ran at 100% CPU for 1 h 29 min and had to be killed by hand (a killed cargo
leaves its test child running). Reproduced on 2026-10-20: 3 tests, `--test-threads=4`, wgpu/Vulkan on a GH200, one run in
fourteen did not finish within 120 s (13 runs took 6-7 s each).

Cause, from a `gdb -p` backtrace of the stuck process: not a brain wait. Every test of the file built its own `Gpu::new`,
so up to three Vulkan devices were created and destroyed at the same time from three libtest threads. One driver-created
worker thread was spinning inside `libnvidia-eglcore` in its thread-exit TSD destructor (`__nptl_deallocate_tsd`), a second
worker was blocked on a priority-inheritance futex in `libnvidia-glsi`, a test thread sat in `vkDestroyDevice` joining the
spinning one, and another test thread sat in `vkBeginCommandBuffer` behind the same lock. It is the hazard `gpu_core::testgpu`
already documents ("many concurrent devices deadlock"), here as a livelock at 100% CPU, which `BRAIN_GPU_WAIT_S`-style
fence deadlines cannot bound because no brain wait is involved. The file simply lacked the per-file `DEVICE_SERIAL` lock that
a dozen sibling test files copy.

Fix: `gpu_core::testgpu::exclusive()`, one process-wide lock a test holds for its whole body when it builds its own `Gpu`
(model/tests/ops_kquant.rs and matmul_kq.rs use it; the per-file `DEVICE_SERIAL` copies elsewhere can move to it as they are touched).
The wait for it is bounded (10 minutes): a waiter whose predecessor is wedged in the driver prints which test waited and
exits the process (`hard_exit(101)`), so the failure is a loud CI error, not hours of idle CPU. Unit test:
`testgpu::tests::the_wait_for_the_device_lock_is_bounded_and_survives_a_panicking_holder`. After the fix 40 consecutive runs of
`ops_kquant` completed (runs/b27/kq-loop-fixed.log). Rule: a test that builds its own `Gpu` takes `exclusive()` first; a test
that can share the device uses `testgpu::dev`.
