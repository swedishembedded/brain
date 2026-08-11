# kronos — roadmap

Kronos financial candlestick foundation model: a BSQ tokenizer (OHLCV bar to
hierarchical discrete tokens) plus a causal autoregressive decoder with a
dual head, ported to brain's Rust+WGSL engine. Tokenizer encode/decode and
the decoder's primary head are verified against the reference implementation,
as is the NPU export via OpenVINO; a KV-cached rollout gives an exact,
significant speedup over the naive autoregressive loop.

## Not yet done

- [ ] Full parity validation for the remaining stages of the ladder: a
      single isolated decoder block, the exposure-bias sampling path, and
      the complete autoregressive `generate` loop (only tokenizer
      encode/decode and the first decoder head are confirmed against the
      reference so far)
- [ ] `ForecastModel` adapter — mapping panel data to OHLCV bars and wiring
      the multi-variate inputs it requires
- [ ] On-NPU end-to-end autoregressive driver that grows context over the
      two exported graphs plus the host-side BSQ decode step
- [ ] Threaded host matvec path (rayon) for the KV-cached rollout
- [ ] GPU-accelerated prefill for the KV-cached rollout
- [ ] Sample-batching in the stochastic forecast adapter — the kernels
      already support batch size > 1, but the sampling loop doesn't use it

The BSQ tokenizer and token sampling stay host-side rather than being
exported to the NPU graph: they're cheap per bar and depend on stateful
bit-packing/RNG, so only the decoder transformer blocks are worth exporting.
