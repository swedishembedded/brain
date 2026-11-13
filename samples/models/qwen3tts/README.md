<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/qwen3tts

Qwen3-TTS as a standalone program: speak a sentence, clone a voice from a few
seconds of reference audio, design a voice from a description, fine-tune the
Talker on one speaker - and run the same synthesis on the Intel NPU, or keep
the compiled NPU graphs resident behind a Unix socket.

The first seven actions are the model's served ones, with a command line built
from its manifest, so everything `brain serve` exposes for the model is here
with the same parameters and defaults. The NPU actions and `npu-serve` are the
part that is an application rather than the served model.

## Build

```bash
make samples/models/qwen3tts/build
Q=target/release/sample-models-qwen3tts
```

## Weights

The checkpoint is read from the model store, as downloaded (the Talker, MTP,
codec and speaker encoder are read from its safetensors; nothing is written
beside it):

```bash
brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

`--weights-dir` and `--ckpt` name a checkpoint outright (a directory of
brain-format files from `import`, and the Hugging Face directory for its
config and tokenizer). Voice cloning needs a Base checkpoint, which ships a
speaker encoder; `design` needs a CustomVoice or VoiceDesign one.

To write the four brain-format component files once:

```bash
$Q import --source ~/.local/share/brain/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base --dest out/tts
$Q synth --weights-dir out/tts --ckpt ~/.local/share/brain/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Read from the imported files."  --out audio=out.wav
```

## Run

```bash
$Q synth --text "Hello from brain." --seed $RANDOM --out audio=hello.wav
$Q clone --text "Spoken in the reference voice." --ref voice.wav --ref-text "what voice.wav says" \
  --seed $RANDOM --out audio=cloned.wav
$Q design --text "Hello." --instruct "a calm, low voice" --seed $RANDOM --out audio=designed.wav
$Q sim --a hello.wav --b cloned.wav
$Q finetune --data data/tts --save talker_lora.safetensors --steps 200 --seed $RANDOM
```

`$Q <action> --help` lists every parameter with its default: the sampling
knobs for the first codebook (`--temp`, `--top-k`, `--top-p`,
`--repetition-penalty`) and for the residual ones (`--residual-temp`, ...),
which resolve from the checkpoint's own `generation_config.json` when left out;
`--profile` prints the resolved plan and the stage timings. The reference
recording of `clone` is an input (`--ref`), as are optional externally encoded
codes (`--ref-codes`); a clip is an output (`--out audio=PATH`). `batch` decodes
several texts interleaved in one call from a JSON array
(`--requests '[{"text":"one"},{"text":"two"}]'`, outputs `audio_0`, `audio_1`).
`--device` places the run.

## On the NPU

```bash
$Q npu-synth --text "Hello from the NPU." --out audio=npu.wav
$Q npu-clone --text "..." --ref voice.wav --ref-text "..." --out audio=npu-cloned.wav
$Q npu-design --text "..." --instruct "a calm, low voice" --out audio=npu-designed.wav
```

These take every flag of `synth`/`clone`/`design` plus `--ov-device`, `--turbo`,
`--talker`, `--mtp` and `--parity`. They need OpenVINO at run time (an Intel NPU,
or `--ov-device cpu`); the first run per checkpoint compiles the graphs, which
takes a while, and later runs reuse the cache.

`npu-serve` keeps the compiled graphs resident and streams synthesized audio
back over a Unix socket, one JSON object per line:

```bash
$Q npu-serve --socket /tmp/brain-tts.sock --clone-ref voice.wav --clone-ref-text voice.txt &
echo '{"engine":"synth","text":"Hello from the server.","seed":7}' | nc -U /tmp/brain-tts.sock
```

Each reply line is an `audio_chunk` (base64 f32 PCM at 24 kHz) until the last,
which carries `"done":true` and the seed the request ran under. `$Q npu-serve
--help` lists the engines (`synth`, `clone`, `design`, `customvoice`), the
Talker precision (`--talker-quant int8|int4|fp32`), the MTP and codec modes
and the streaming chunk and window.
