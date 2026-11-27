<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 615. A fine-tune whose card refused one allocation lost the whole run

**Lesson:** a device out-of-memory in a training step is usually a moment, not a verdict. Observed 2026-11-27 on a
GH200 serving and training at once (brain a2e6c80a9): a LoRA fine-tune died with `cuMemAlloc` out of memory when
serving KV grew on its card, and 12 minutes of training were lost. #611 keeps serving out of the run's reservation;
the run must still survive pressure it cannot see coming (a neighbouring process, a transient peak).

## Why

The SDK already turned the unwind into `brain::Error::DeviceOutOfMemory` and released the run's memory, but the
job answered with it at once, and a run without a `resume_key` saved no state, so even a resubmission began at step 0.

## Fix

`trainjob::job::run` keeps a single-card run's failure typed (`RunFailure`) and retries a device out-of-memory
(`retry_device_oom`: back-off 2 s doubling, at most 5 retries, about a minute in all; cancellation ends the wait),
telling the client in a `stage` frame. Every such run saves its state every `DEFAULT_CHECKPOINT_EVERY` (25) steps in
the job's own directory when it has none of its own, with a seed drawn once, so a retry continues from the last
saved step. Spec: `crates/trainjob/tests/it/device_oom.rs` (an allocation refused at step 27 of 30: the run resumes
at 25 and completes; red before, the job failed).

**How to apply:** make a failure retryable only where it is typed, and make the retry cheap by saving progress before
it is needed. Left out: data-parallel ranks and streaming runs still fail on a device out-of-memory (their ranks
would have to retry together).
