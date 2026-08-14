# qwen3tts - roadmap

Qwen3-TTS voice synthesis stack (Talker + MTP + Mimi-style codec + ECAPA
speaker encoder): speaker-free synthesis, voice cloning (x-vector and
in-context), instruct-style voice design, NPU/CPU/GPU backends, and LoRA
fine-tuning. Parity against the reference is verified.

## Not yet done

- [ ] Cancellation support for in-flight synth/clone requests
- [ ] Batched inference (`run_batch`) - only sequential single-request
      inference exists; autoregressive decode makes a genuine batched
      forward nontrivial
- [ ] Consolidate the private socket-based serving side-channel into the
      standard D-Bus serving surface
- [ ] An example D-Bus client for TTS
- [ ] A windowed attention mask in the codec for long-form decode beyond the
      current fixed window
- [ ] A fused single-inference MTP graph - MTP and the codec are now the
      dominant per-clip cost after the Talker path was optimized
- [ ] From-scratch training for the codec and speaker encoder (only Talker
      LoRA fine-tuning exists today)
- [ ] Wire a real TTS model into the runtime event/state-machine flow
      (currently only a stub model is wired there)

CPU codec decoding is computationally sub-real-time for this architecture;
the NPU backend is the only realtime synthesis path today.
