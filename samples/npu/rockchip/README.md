<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# npu/rockchip - real-time detection on a Rockchip RV1103/RV1106 board

Streams the board's camera over RTSP with objects detected by brain's
`DetectionPipeline` drawn on it. YOLOv5 runs on the board's NPU, compiled by
brain's own NPU compiler (`crates/rknpu`): no vendor NPU SDK and no offline
conversion. The board loads a brain checkpoint and compiles it itself. The
same program also detects in a single image and prints the boxes.

Tested on a Luckfox Pico Plus (RV1103, one Cortex-A7, 64 MB DDR of which Linux
sees 33 MB, SC3336 camera).

## 1. Prepare the model (on a host)

The weights are Rockchip's ReLU export of Ultralytics YOLOv5s,
`yolov5s_relu.onnx` from Rockchip's model zoo. Like the published YOLOv5
weights they are AGPL-3.0-derived. Convert it to a brain
checkpoint, then measure its NPU calibration over a few dozen images like the
ones the camera will see:

```sh
python tools/yolov5_export/export_yolov5.py --onnx path/to/yolov5s_relu.onnx
make crates/yolov5/examples/npu-calibrate/build
make crates/yolov5/examples/npu-calibrate/run \
  ARGS="<models-dir>/local/yolov5s-relu/model.safetensors path/to/images 320"
```

The calibration is written beside the checkpoint as
`model.npu-calibration.json`. It records every tensor's range at the 320x320
input the board runs. It takes about 12 s for 32 images.

## 2. Build and deploy

The board runs 32-bit uClibc. The sample is cross-built with the board
vendor's uClibc gcc and copied, with the model, to `brain/` in the board
user's home (persistent storage; the board's `/tmp` is RAM it cannot spare):

```sh
make samples/npu/rockchip/board/build UCLIBC_TOOLCHAIN=<path/to/uclibc-gcc/bin>
make samples/npu/rockchip/board/deploy BOARD=<ssh host> \
  MODEL_DIR=<models-dir>/local/yolov5s-relu
```

The result is `brain/sample-npu-rockchip` and
`brain/models/local/yolov5s-relu/`. Model ids resolve in the `models/`
directory beside the program (`--models-dir` overrides that), so the `brain/`
directory runs from wherever it is copied.

## 3. Run

One image, boxes as JSON in the image's pixels:

```sh
ssh <board> 'cd brain && ./sample-npu-rockchip --model local/yolov5s-relu --image photo.jpg'
```

Live video with detection:

```sh
ssh <board> 'cd brain && ./sample-npu-rockchip --model local/yolov5s-relu --stream'
ffplay -rtsp_transport tcp rtsp://<board-ip>:8554/live/0
```

`--stream` takes the camera, so stop the board's own camera daemon first. It
loads the model before the camera, because the model's NPU memory is the
largest allocation. Ctrl-C or SIGTERM releases the camera, the ISP and the
encoder, so the next run starts cleanly. `--help` lists every option: stream
size, bit rate, GOP, port, mount point, duration and the detection thresholds.

The camera, ISP 3A, 2D scaler and encoder are the board's own media libraries
(`librockit`, `librkaiq`, `librga` in `/oem/usr/lib`). They are opened at run
time with `dlopen`; nothing is linked or redistributed. The program restarts
itself once with `LD_LIBRARY_PATH` set, because the board's loader does not
search `/oem`.

## How it is arranged

Two threads, joined only by the seam in `detect`:

- **The video thread**: camera frame -> detector tap -> box overlay -> encoder
  -> RTSP socket. Each thread hand-off would cost a wake-up, and the stages in
  between are hardware. It never waits for detection.
- **The detector thread** (`detector`): it takes the freshest tap frame (a
  320x320 letterboxed RGB copy, made by the 2D accelerator) and runs
  `brain::DetectionPipeline` on the NPU. It maps the boxes to stream pixels
  through the frame's letterbox and publishes them. The video thread draws
  the newest result on every frame and drops one older than 600 ms. A frame
  that arrives during an inference is simply not taken, so nothing queues.

| module | job |
|---|---|
| `ffi/` | declarations per vendor library (`rockit`, `rkaiq`, `rga`), loaded by `ffi::dl`; struct layouts are asserted against sizes measured with the vendor headers |
| `camera` | ISP 3A loop plus one NV12 video-input channel |
| `encoder` | H.264 baseline, no B-frames, CBR, intra refresh, one output buffer |
| `rtsp` | RTSP server (`proto`: requests, SDP; `rtp`: RFC 6184 packetizer with FU-A) |
| `overlay` | boxes and labels drawn into the NV12 frame |
| `tap` | scaled RGB copy of the frame for the detector |
| `detect` | the seam: detections, the board they are published to, the letterbox |
| `detector` | the detection thread |
| `latency`, `memory` | capture-to-send statistics, RSS and CMA reporting |
| `stream` | the pipeline loop and its start/stop |

**RTSP**: `OPTIONS`, `DESCRIBE`, `SETUP`, `PLAY`, `GET_PARAMETER`, `TEARDOWN`,
with RTP carried over the TCP control connection (interleaved). A UDP request
gets `461`, and clients such as ffmpeg then retry over TCP. A client that
cannot take a frame within 250 ms is dropped, so it cannot stall the camera.

**Overlay**: the CPU draws straight into the camera frame instead of using the
encoder's hardware overlay. A box outline is a few thousand pixels, while the
hardware path needs bitmap memory in CMA that this board cannot spare. The
frame buffer is CPU-cached, so the touched rows are invalidated and cleaned
around the draw (`overlay::dirty_ranges`).

## Measured (Luckfox Pico Plus, SC3336, 800x448 at 30 fps)

| | |
|---|---|
| video | 30.0 fps |
| frame captured -> last RTP packet sent | p50 3.9 ms, p95 4.6 ms, max 6.3 ms (tap 0.1, overlay up to 1, encode 2.9, send under 0.5) |
| sensor readout + ISP | 33.5 ms, before the frame is complete |
| start of readout -> last RTP packet sent | p50 37 ms, p95 38 ms |
| detection | 15 inferences/s, 46 ms per inference (frame copy + NPU + decode) |
| boxes on screen | about 84 ms after the frame's readout began, so about 50 ms after it is complete |
| still image (640x426) | 71 ms including the letterbox; model loads in 4.8 s |
| memory | VmRSS 10 MB, peak 11.4 MB; CMA 20.2 of 24.5 MB (model 9.6 MB, camera and encoder 7.9 MB) |

The sensor timestamp on a frame is the start of its readout. The SC3336 has one
mode (2304x1296 at 30 fps), which reads a frame out over 33 ms of its 33.3 ms
frame period; the capture driver reports `readout: 33 ms` in
`/proc/rkcif-mipi-lvds`. The ISP runs online on the lines as they arrive and
finishes with the readout (`/proc/rkisp-vir0`: `delay: 32ms`). So a frame is
complete about 33.5 ms after its timestamp, whatever runs downstream, and the
sample reports that span separately from its own latency. That latency, from
the complete frame to its last RTP packet written to the socket, is about 4 ms.
The program logs all three every 100 frames. The camera path alone at other
stream sizes:

| stream | CMA used by the camera path | frame captured -> sent, p50 |
|---|---|---|
| 800x448 (default) | 7.9 MB | 4 ms |
| 960x544 | 9.0 MB | 7.6 ms |
| 1280x720 | 11.7 MB | 10.6 ms |

1280x720 does not fit beside the model in the 24 MB CMA pool.

The detector's int8 results match the float network: on the YOLOv5 golden
photograph, the same people are found with confidence within 0.012
(`crates/yolov5/tests/npu_int8.rs`). The board's detections equal the host
simulation of the same integers exactly.
