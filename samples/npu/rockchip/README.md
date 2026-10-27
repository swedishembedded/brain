<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# npu/rockchip - real-time detection on a Rockchip RV1103/RV1106 board

Runs brain's `DetectionPipeline` on the board's NPU and streams the camera, with
the detected boxes drawn on it, over RTSP. Work in progress: the status and the
measured numbers are in the `rknpu` roadmap ledger.

## Build and deploy

The board runs 32-bit uClibc, so the sample is cross-built with the board
vendor's uClibc gcc and copied to `brain/` in the board user's home:

```sh
make samples/npu/rockchip/board/build UCLIBC_TOOLCHAIN=<path/to/uclibc-gcc/bin>
make samples/npu/rockchip/board/deploy BOARD=<ssh host>
```
