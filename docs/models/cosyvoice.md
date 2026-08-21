# CosyVoice (not yet servable)

FunAudioLLM's LLM-based streaming zero-shot TTS: a Qwen2.5-0.5B speech-token
LM (hosted on [Qwen3](qwen3.md)'s decoder), a causal flow-matching mel decoder
(a UNet estimator for CosyVoice 2, a DiT for CosyVoice 3), and an ISTFT/NSF
source-filter HiFT vocoder - conditioned on a reference clip via
[S3Tokenizer](s3tokenizer.md) (FSQ speech tokens) and [CAM++](campplus.md)
(a 192-d x-vector). One crate, one architecture id, covering both released
generations (`FunAudioLLM/CosyVoice2-0.5B`,
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) as a config, not two ids - see
`crates/arch`'s own naming rule.

Name reserved only today - import, forward, training and the serving contract
are not yet implemented.

Weights env vars (not yet read by any code - reserved alongside the
architecture id, per `crates/arch`'s naming rule):

| Variable | Role |
|---|---|
| `BRAIN_COSYVOICE_LLM` | speech-token LM (`llm.pt`) |
| `BRAIN_COSYVOICE_FLOW` | flow decoder (`flow.pt`) |
| `BRAIN_COSYVOICE_HIFT` | HiFT vocoder (`hift.pt`) |

Package: `brain-cosyvoice`.
