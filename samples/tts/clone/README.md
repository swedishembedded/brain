# sample: tts/clone

Clone a voice from a reference clip and speak new text in it, through the
public `brain` SDK's `TtsPipeline::clone_voice` - the smallest complete
voice-cloning application you can build on brain.

```bash
make samples/tts/clone/build
make samples/tts/clone/run ARGS="--voice me.wav --text 'hello, this is my own voice'"
```

With no `--out`, the cloned clip plays straight on the default audio output
device. Pass `--out clone.wav` to write a file instead.

## What it demonstrates

* `brain::TtsPipeline::from_pretrained` resolves a model out of the local
  model store using the same resolver the CLI uses - no parallel path, no
  environment variable required. It tries Qwen3-TTS first, then CosyVoice
  (`--model` selects which checkpoint it resolves against).
* `clone_voice` + `Audio::save`/playing `Audio::samples()` directly: the
  whole voice-cloning surface is two calls either way.
* No CLI process and no capability-dispatch server in the loop. This binary
  links `brain` the way a product would.

## What it needs

Real weights, plus a reference voice clip (any WAV `brain::TtsPipeline` can
read). The default model is `Qwen/Qwen3-TTS-12Hz-0.6B-Base`; if it is not in
the local store the sample prints the `brain pull`/`brain tts import`
command to run and exits non-zero rather than downloading several gigabytes
behind your back.

```bash
brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

## Options

| flag | default |
|---|---|
| `--model ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| `--voice PATH` | required |
| `--text TEXT` | required |
| `--ref-text TEXT` | unset (optional on Qwen3-TTS; REQUIRED if `--model` resolves to CosyVoice) |
| `--out PATH` | unset - plays on the default output device instead of writing a file |
