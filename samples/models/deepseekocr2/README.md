<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/deepseekocr2

DeepSeek-OCR-2 as a standalone program: read a document image into text or
markdown, and export the resampler and the decoder as ONNX graphs for
OpenVINO and the Intel NPU.

`generate` is the model's served action, with a command line built from its
manifest, so everything `brain serve` exposes is here with the same parameters
and defaults. `npu-export` is this program's own.

## Build

```bash
make samples/models/deepseekocr2/build
D=target/release/sample-models-deepseekocr2
```

## Weights

The checkpoint directory is resolved from the model store, as
`brain deepseekocr2 generate` does; `--dir DIR` names one outright (the two
GGUFs, their expanded fp32 forms, or an HF checkpoint).

## Run

```bash
$D generate --in image=page.png --out text=page.md
$D npu-export --seq 512 --out-dir out/deepseekocr2-onnx
```

`npu-export` is pure Rust and needs neither OpenVINO nor a device. It writes
`resampler_local.onnx`, `resampler_global.onnx` and `decoder.onnx` (`--seq` is
the decoder's fixed length); the SAM backbone is not exported. Check what the
NPU compiler makes of a graph with `brain npu check --onnx FILE`.

`$D <action> --help` lists every parameter with its default.
