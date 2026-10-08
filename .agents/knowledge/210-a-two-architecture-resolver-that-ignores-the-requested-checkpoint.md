# 210. A two-architecture resolver that ignores the requested checkpoint

`resolve_two_with_policy` (and `try_two` under it) resolves each architecture
over the whole model store and only uses the requested model id to decide what
to fetch. With two checkpoints of one task in the store, the architecture
tried first wins whichever one was asked for.

How it showed: `TranscribePipeline::from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")`
with both `Qwen/Qwen3-ASR-1.7B` and the Nemotron checkpoint present returned a
pipeline backed by Qwen3-ASR. Transcribing a clip the Nemotron CLI reads
correctly gave an empty text and token ids `[11528, 2240, 151704]`, 151704
being a Qwen3-ASR vocabulary id. Nothing errored; the wrong model answered.

Fix: `resolve_either_for_reference` resolves each side over the requested
model's own directory (`loader::resolver::resolve_reference`), as the
single-architecture `resolve_with_policy` already did. The test that pins it
puts both checkpoints in one store and asks for the Nemotron one.

`TtsPipeline` still uses the whole-store tie-break between Qwen3-TTS and
CosyVoice, which is right only while a store holds at most one checkpoint of
each; moving it over is a separate change.

Also measured while building the round-trip test: unseeded Qwen3-TTS renders
one sentence ("The quick brown fox jumps over the lazy dog.") in anywhere from
3.68 s to 8.08 s depending on the seed, and the 3.68 s rendering is
recognised as "The quick brown" by Qwen3-ASR. A round-trip test that must be
deterministic seeds the synthesis; a data pipeline must filter by round trip.
