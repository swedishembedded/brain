<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/qwen3omnimoe

Qwen3-Omni as a standalone program: text, speech, image and video in, text and
speech out, the conversion of the published checkpoint into the int8 one the
GPU-sharded Thinker serves, and the export of its audio tower, vision tower and
Code2Wav vocoder as ONNX graphs for OpenVINO and the Intel NPU.

`generate`, `speak`, `converse` and `import` are the model's served actions,
with a command line built from the manifest, so everything `brain serve`
exposes is here with the same parameters and defaults. `npu-export` is this
program's own.

## Build

```bash
make samples/models/qwen3omnimoe/build
O=target/release/sample-models-qwen3omnimoe
```

## Weights

The published checkpoint directory is resolved from the model store, as
`brain qwen3omnimoe <action>` does; `--weights DIR` names one outright.

## Run

```bash
$O generate --prompt "Describe the sound of rain." --out text=out.txt
$O npu-export --out-dir out/omni --n-audio 100 --grid-h 16 --grid-w 16 --code-len 32
```

`npu-export` writes `audio_tower.onnx`, `vision_tower.onnx` and
`codec_decoder.onnx` (not the 30B-parameter MoE Thinker, which is not an NPU
target) and prints, for each, the structural check `brain npu check` runs and,
where OpenVINO and a device are present, whether it compiles
(`--ov-device`, `--hint`, `--cache-dir`, `--turbo`, `--allow-fallback`,
`--profile`). A piece that fails to export is reported and the others still
run; the command then fails.

`$O <action> --help` lists every parameter with its default.
