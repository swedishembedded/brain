# Speech and multimodal: text-to-speech, transcription, and beyond

brain runs models for turning text into speech, speech into text, and - via
one omni-modal assistant - mixed audio/image/video input into a spoken or
text reply.

## Capabilities

### Text-to-speech and voice cloning - `brain/qwen3tts`

Turns text into a 24 kHz speech waveform. Give it a few seconds of reference
audio and it clones that voice for the sentence you ask it to speak;
without one it falls back to a default voice. Reach for it for narration, a
cloned voice for a demo, or a spoken-output leg for an assistant pipeline,
all running locally. See [the TTS page](../models/tts.md).

### Speech-to-text - `brain/nemotronasr` and `brain/qwen3asr`

Two transcription models behind the same audio-in/text-out contract.
Nemotron transcribes audio as it arrives, for live/interactive use (a mic
feed, a call); Qwen3-ASR transcribes a complete clip in one pass, for
offline batch transcription where accuracy on the whole file matters more
than latency. See [the ASR page](../models/asr.md).

### An assistant that can listen, see, and speak - `brain/qwen3omnimoe`

A single served model that takes text, audio, image, or video in, answers
in text, and - when asked to speak - replies with an actual synthesized
waveform instead of just text. Reach for it when you want one model handling
mixed-modality input and a spoken reply, rather than wiring separate
ASR/vision/TTS models together yourself. See
[the Qwen3-Omni page](../models/omni/readme.md).

### Understanding images and video as text - `brain/fastvlm` and `brain/qwen3vl`

If what you need is a text description or answer about an image rather than
spoken output, brain also has two dedicated image-understanding models -
one for fast captioning, one for general prompted Q&A about an image. See
[the vision-language model page](../models/vlm.md).
