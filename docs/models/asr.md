# Speech-to-Text

Two speech-to-text architectures behind the same audio-in/text-out contract.
Each has its own reference page with getting-started commands and options.

| Architecture | Solves | Batching |
|---|---|---|
| [Nemotron 3.5 ASR Streaming](nemotronasr.md) | live/interactive transcription (mic feed, a call - text as the speaker talks) | real streaming + batching |
| [Qwen3-ASR](qwen3asr.md) | offline batch transcription (whole file up front, more accurate offline pass) | single-request |

Reach for Nemotron when you want text as the speaker talks; reach for
Qwen3-ASR when you have the complete file and want the more accurate
offline pass.
