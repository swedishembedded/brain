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

## Live video: `--stream`

```sh
ssh <board> 'cd brain && ./sample-npu-rockchip --stream'
ffplay -rtsp_transport tcp rtsp://<board-ip>:8554/live/0
```

`--stream` takes the camera (stop the board's own camera daemon first), encodes
H.264 in hardware and serves it over RTSP. Ctrl-C or SIGTERM releases the
camera, the ISP and the encoder so the next run starts cleanly. `--help` lists
every option (size, bit rate, GOP, port, mount point, duration, ...).

It uses the board's own media libraries (`librockit`, `librkaiq`, `librga` in
`/oem/usr/lib`), opened at run time with `dlopen`; nothing is linked or
redistributed. The program restarts itself once with `LD_LIBRARY_PATH` set,
because the board's loader does not search `/oem`.

### How the video path is arranged

One thread does camera frame -> detector tap -> box overlay -> encoder -> RTSP
socket, because each thread hand-off costs a wake-up and the stages in between
are hardware. Detection never sits on that thread.

| module | job |
|---|---|
| `ffi/` | declarations per vendor library (`rockit`, `rkaiq`, `rga`), loaded by `ffi::dl`; struct layouts are asserted against sizes measured with the vendor headers |
| `camera` | ISP 3A loop plus one NV12 video-input channel |
| `encoder` | H.264 baseline, no B-frames, CBR, intra refresh, one output buffer |
| `rtsp` | RTSP server (`proto`: requests, SDP; `rtp`: RFC 6184 packetizer with FU-A) |
| `overlay` | boxes and labels drawn into the NV12 frame |
| `tap` | scaled RGB copy of the frame for the detector |
| `detect` | the seam types |
| `latency`, `memory` | capture-to-send statistics, RSS and CMA reporting |
| `stream` | the pipeline loop and its start/stop |

RTSP: `OPTIONS`, `DESCRIBE`, `SETUP`, `PLAY`, `GET_PARAMETER`, `TEARDOWN`, with
RTP carried over the TCP control connection (interleaved). A UDP request gets
`461`, and clients such as ffmpeg then retry over TCP. A client that cannot take
a frame within 250 ms is dropped so it cannot stall the camera.

Overlay: drawn by the CPU straight into the camera frame, not with the
encoder's hardware overlay. A box outline is a few thousand pixels, while the
hardware path needs bitmap memory in CMA, which this board does not have to
spare. The frame buffer is CPU-cached, so the touched rows are invalidated and
cleaned around the draw (`overlay::dirty_ranges`).

### Plugging in the detector

`stream::Stream::start(config)` returns a handle with both ends of the seam:

```rust
let stream = Stream::start(StreamConfig::default())?;

// detector thread
while let Some(frame) = stream.frames.recv(Duration::from_millis(200)) {
    let rgb: &[u8] = frame.rgb();          // TAP_SIZE * TAP_SIZE * 3, RGB
    let lb = frame.meta().letterbox;       // model space -> stream space
    let [l, t, r, b] = lb.box_to_stream([x1, y1, x2, y2], w, h);
    stream.board.publish(Detections { boxes: vec![Detection {
        left: l, top: t, right: r, bottom: b, class_id, score, label }] });
}
```

* **Tap frame**: 320 x 320 RGB888, letterboxed (aspect ratio kept, grey 114
  padding), made by the 2D accelerator. `TapMeta::letterbox` maps tap
  coordinates back to stream pixels (`Letterbox::to_stream`, `box_to_stream`).
* **Delivery**: a frame is converted only while the detector is blocked in
  `recv`, so it is always the freshest one, and a busy detector costs the
  video path nothing. One buffer: while a `TapFrame` is held no new one is
  made, so the pixels can be fed to the NPU without a copy.
* **Results**: `DetectionBoard::publish` replaces the result; the video thread
  draws the newest one on every frame. A result older than 600 ms is dropped
  from the picture, so boxes vanish if the detector stalls.
* `--stream` currently runs a sweeping demo box and a tap consumer in place of
  the detector (`--no-demo-box` turns the box off, `--dump-tap FILE` saves a
  tap image as a PPM).

### Memory and latency (Luckfox Pico Plus, SC3336, 30 fps)

CMA is the scarce resource (24 MB shared with the NPU). The ISP's own buffers
are about 5 MB at any stream size; the rest scales with the stream:

| stream | CMA used by the sample | in-process p50 |
|---|---|---|
| 800x448 (default) | 7.9 MB | 5.6 ms |
| 960x544 | 9.0 MB | 7.6 ms |
| 1280x720 | 11.7 MB | 10.6 ms |

At 800x448, process memory is VmRSS 8.7 MB. Latency is the sensor timestamp on
a frame to its last RTP packet written; the program logs p50/p95 every 100
frames. Typical: p50 41 ms, p95 43 ms, of which about 35 ms is sensor readout
and ISP before the frame reaches the program; the program's own share is
5.6 ms (tap 1.4, overlay 1.2, encode 2.9, send under 0.5).
