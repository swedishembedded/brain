# omni — roadmap

Qwen3-Omni-30B-A3B port: a multimodal (text/image/audio/video in, text/speech
out) sparse-MoE model composed of a Thinker (MoE text decoder), a Talker (MoE
speech decoder), an audio tower, a vision tower, a code predictor, and a
Code2Wav vocoder, on brain's fp32/WGSL engine. Forward-pass parity against the
reference implementation is verified component-by-component (audio tower,
vision tower image path, Thinker/Talker decoder layers, code predictor,
Code2Wav vocoder), and the shared sparse-MoE core has a gradient-checked
backward pass.

## Not yet done

- [ ] `converse` action (real audio/image input combined with speech output
      in the same turn)
- [ ] `transcribe` action
- [ ] Multimodal user-turn input (image/audio/video) combined with `speak`
      speech output — `speak` currently only supports text-only, single-turn
      prompts
- [ ] Multi-turn Talker conversation context (only single-turn is supported)
- [ ] `speak` (speech output) is not yet wired into the `examples/omni.py`
      client, only into the D-Bus/HTTP serving layer
- [ ] Streaming Code2Wav decode (chunked, with left-context) — decoding is
      currently whole-utterance only
- [ ] Real wall-clock audio-timestamp scaling for M-RoPE on audio spans
      (currently approximated by frame-ordinal advance instead)
- [ ] DeepStack multi-scale vision features are not applied on the served
      image path — only the primary (non-DeepStack) merger output is used
- [ ] True temporal patching for video input — frames are currently encoded
      as independent single images rather than grouped per the model's
      temporal-patch convention
- [ ] A numeric/parity golden for the composed Thinker→Talker→Code2Wav speech
      pipeline — end-to-end validation currently only checks that the output
      is a real, non-silent waveform, not that it matches a reference
- [ ] Multimodal input and speech output through the int8 dual-GPU resident
      path - it serves Thinker TEXT generation only (over the ordinary chat
      contract); multimodal input and `speak` still require the slower,
      non-sharded streaming resident that reads straight from the HF
      checkpoint directory
- [ ] A workgroup-tiled int8 MoE expert kernel (see the note below) - with
      weights resident and decode KV-cached, the remaining ~2.3 s/token is
      almost entirely those naive per-thread expert dispatches
- [ ] `BRAIN_PROFILE`'s Vulkan kernel table silently skips any batch over
      `MAX_TIMED_DISPATCHES` (8192), which a 48-layer 128-expert MoE forward
      always exceeds - so on the int8 path the table covers only the head
      matmul, and per-kernel attribution inside a layer is unavailable
- [ ] OpenAI/Anthropic HTTP surfaces drop `image_url`/`input_audio` content
      parts instead of routing them to the model
- [ ] `/v1/audio/speech` and `/v1/audio/transcriptions` HTTP endpoints
- [ ] Real NPU hardware execution — only CPU-side ONNX/OpenVINO structural
      and numeric parity checks have been run
- [ ] DeepStack vision-tower NPU export
- [ ] Vulkan device churn (rapid create/drop) can lose driver visibility of a
      GPU — a separate, still-open driver-level issue distinct from the
      concurrency bugs already fixed
- [ ] `braintop`/stats display for a multi-device resident model (the schema
      is single-device only)

A workgroup-tiled int8 MoE expert kernel is not yet implemented: WGSL gives
no legal way for only some threads in a workgroup to skip a shared-memory
barrier, so a barrier-based tiled kernel needs routed rows gathered into a
compact, non-ragged batch first; the production int8 Thinker path currently
uses a slower, naive per-thread kernel instead.
