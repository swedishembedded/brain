# dtype - roadmap

Docs-honesty program: `docs/**/*.md` accumulates real, measured performance
numbers (wall-clock times, throughput, speedups) as optimization passes land.
Those numbers are true of the specific machine/driver/checkpoint they were
measured on and go stale - or get copy-pasted as if they were general
promises - the moment anything changes. The fix is mechanical, not just
editorial discipline: a gate that denies any bare number next to a
performance unit or claim unless a human has reviewed and deliberately
excepted it with an inline comment.

Phased: A1 builds the gate and proves it fires (RED) on the current tree. A2
(separate task) drives the violation count to zero - rephrasing each claim to
point at `brain perf run`/`brain flops` instead of a fixed figure, or marking
a genuinely-illustrative number as a reviewed exception.

## A1 - perf-number gate

**Gate**: `scripts/gates/check-no-perf-numbers.sh`, modelled on
`scripts/gates/check-env-docs.sh`'s house style (scan, list every violation
as `file:line: matched text`, exit non-zero on any hit, clear fix
instructions in the failure message).

Scans `docs/**/*.md` and denies a bare number adjacent to a performance unit
or claim: `ms`, bare `s`/`min` (durations), `fps`, `tok/s`/`tokens/s`,
`GB/s`/`MB/s`, `TFLOP`/`GFLOP`, `p50`/`p95`/`p99` followed by an actual
value - all unambiguous, checked without further context. `%` and
`Nx`/`N×` speedup patterns are also denied, but only when nearby lines
(a 4-line-back/2-line-forward window, since this repo's prose wraps a
paragraph's "measured"/"speedup" cue word across several physical lines)
carry measurement vocabulary - this is what keeps "a 16× convolutional token
compressor" (an architecture constant) and "30% token masking" (a training
hyperparameter) from being confused with "measured 2.5–3.6× the throughput"
(a real result). Escape hatch: `<!-- perf-number: <reason> -->` on the same
line or the line immediately before it.

**Makefile wiring**: added to `check/scripts` in the root `Makefile`,
alongside `check-scripts.sh`/`check-env-docs.sh`/`check-no-doc-citations.sh`
(same `bash scripts/gates/<name>.sh` invocation pattern); the `make help`
line for `check/scripts` updated to mention it.

**RED baseline on the current tree** (this is what A2 must drive to zero -
counts are real hits from a live run, not estimates):

| file | violations |
|---|---|
| `docs/performance/overview.md` | 209 |
| `docs/models/deepseek-ocr.md` | 36 |
| `docs/performance/hardware-notes.md` | 3 |
| `docs/scaling/data-parallel.md` | 2 |
| `docs/scaling/pipeline.md` | 1 |
| `docs/models/asr.md` | 1 |
| **total** | **252** |

`docs/models/asr.md`'s one hit ("default 30s" fixed transcription window) is
a capability limit rather than a measured result - left flagged rather than
gate-refined around, since a human still has to decide "rephrase" vs.
"escape-hatch" for it in A2; over-fitting the regex to one observed line
would just move the false-positive risk somewhere else.

`make check/scripts` currently fails **before** reaching this gate, on
`check-env-docs.sh` (`BRAIN_WGPU_SERIAL`/`BRAIN_WGPU_NO_SERIAL` undocumented)
- a pre-existing, unrelated failure this task did not introduce and does not
fix. Run the new gate directly to see it fire:
`bash scripts/gates/check-no-perf-numbers.sh`.

**Known imprecision**: the `%`/`Nx` context-gate is a heuristic (nearby
measurement vocabulary), not a parser. It is verified to correctly exclude
the false positives found during development (`16×` downsample factor, `1x`
render-quality option, `4x` checkpoint scale, `30%` token-masking
hyperparameter) and to correctly include every genuine measured claim in the
five files above. Any further false positive found in A2 belongs on the
escape-hatch list or as a targeted gate refinement - never a reason to weaken
the pattern wholesale.

**What's left (A2, separate task)**: rephrase or escape-hatch all 252
violations above so `check-no-perf-numbers.sh` - and therefore
`make check/scripts` - passes clean.
