# ECAPA-TDNN (speaker encoder, component)

The speaker encoder (1024-d) [Qwen3-TTS](qwen3tts.md) uses for voice
cloning - turns a reference clip into the speaker embedding synthesis
conditions on. Not independently servable: it has no capability manifest or
CLI verb of its own, reached only as part of Qwen3-TTS's `clone` action.

Package: `brain-ecapatdnn`.
