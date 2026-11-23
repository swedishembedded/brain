<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/cosyvoice

CosyVoice 2 and 3 as a standalone program: speak a sentence in the voice of a
few seconds of reference audio, given that clip's transcript, with no training
on the speaker (zero-shot cloning).

The program is CosyVoice's served `synth` action with a command line built
from its manifest, so everything `brain serve` exposes for the model is here
with the same parameters and defaults, and nothing is reimplemented.

## Build

```bash
make samples/models/cosyvoice/build
C=target/release/sample-models-cosyvoice
```

## Weights

The six roles are found in the model store by their content, and `--<role>
PATH` names one outright for a run:

| Role | File |
|---|---|
| `llm` | the speech-token language model, `llm.pt` |
| `flow` | the flow-matching mel decoder, `flow.pt` |
| `hift` | the HiFT vocoder, `hift.pt` |
| `tokenizer` | `CosyVoice-BlankEN`'s `tokenizer.json` |
| `s3tokenizer` | the speech tokenizer, `speech_tokenizer_v2.onnx` |
| `campplus` | the CAM++ speaker encoder, `campplus.onnx` |

`llm`, `flow` and `hift` must be the generation
`--variant` names (`cosyvoice2`, the default, or `cosyvoice3`): the two ship
different files under the same names, and a mismatch fails in the importer.

## Run

```bash
$C synth --text "Spoken in the reference voice." \
  --ref_text "what reference.wav says" --in ref_audio=reference.wav \
  --seed $RANDOM --out audio=cloned.wav
$C synth --variant cosyvoice3 --n_timesteps 10 --profile \
  --text "..." --ref_text "..." --in ref_audio=reference.wav --out audio=cv3.wav
```

`$C synth --help` lists every parameter with its default. `--in
ref_audio=FILE` reads the clip through the shared command-line audio loader,
which resamples it to 16 kHz; a D-Bus or HTTP caller that sends the WAV itself
keeps its full rate. Without `--seed` a random seed is drawn and echoed in the
outcome, so a take can be reproduced.
