# forecast — roadmap

The forecasting foundation models (chronos2, fincast, kronos) are served over
the shared D-Bus surface and the residency scheduler, including NPU
auto-placement, KV-cached NPU/CPU rollout for kronos, and batched fine-tuning.

## Not yet done

- [ ] `run_batch` is the sequential default for all three models — chronos2
      and fincast share a batchable transformer core (equal-shape contexts
      could share one forward) but a genuine batched forward is not wired
- [ ] Streaming (chunked) training for very long contexts — long training
      contexts can exhaust GPU memory in one shot today
