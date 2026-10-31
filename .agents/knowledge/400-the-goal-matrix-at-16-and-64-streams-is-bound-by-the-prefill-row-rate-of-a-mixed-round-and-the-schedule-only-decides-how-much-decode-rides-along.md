<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 400. The goal matrix at 16 and 64 streams is bound by the prefill row rate of a mixed round, and the schedule only decides how much decode rides along

Question: FP8 Qwen3.8-27B, 8K..64K prompts at 1/4/16/64 streams, 256 output tokens, one GH200: why does brain trail the reference servers at 16 and 64 streams, and what
does the schedule control?

Measured (nsys on `brain serve`, `vllm bench serve` clients, gpu0; 8Kx16 = 32 prompts of 7.9K tokens):

- The run is a closed loop of waves: 16 (64) prompts arrive together, are read in, decode 256 tokens in lockstep, finish together and the next wave arrives. A wave costs the card
  its prefill rows at the round's row rate plus 256 decode-only steps (16.9 ms at 16 rows, 30 ms at 64) in which no prompt waits. At 8Kx64 the prefill is 80 of 100 s.
- The default 45 ms iteration target made mixed rounds of ~775 rows (326 rounds, median 72.6 ms, card 96 % busy inside them): 252 output tok/s. Rounds of 4096 rows (330 ms, 12.4K
  rows/s, decode rows riding along) read the same prompts 15 % faster: 280 tok/s; `--itl-target-ms 1000` at 2048 rows was already 266.
- Round shape is therefore a throughput lever until the round fills the GEMM waves (4096 rows; 8192 gains nothing, knowledge 398). Reading the prompt with the fewest rows left first (the
  oldest keeps a minimum slice) staggers the finish times of a wave, so decode starts earlier and the median first-token time falls: 8Kx64 32.4 -> 2.5 s.
- What the schedule cannot change: a 4096-row round is 330 ms of kernels, GEMM 62 % (1.05 PFLOPS), attention 11 %, GDN 7 %, norm/quant/glue 20 %; the reference servers run the same
  prompts at 15-16K rows/s. The remaining gap at 16 and 64 streams is that per-row cost plus the decode-only steps of the wave tail.
