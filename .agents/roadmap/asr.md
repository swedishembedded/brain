# asr — roadmap

Two speech-to-text models (Nemotron 3.5 ASR streaming, Qwen3-ASR) on the shared
engine, imported from HF and served over the capability/residency/D-Bus stack.

## Not yet done

- [ ] Qwen3-ASR variable-length serving (currently a fixed input window;
      needs a probe-per-length or padded-KV scheme to generalize)
- [ ] Streaming-session survival across residency eviction (session state is
      dropped when the instance is evicted; a restarted session id starts fresh)
