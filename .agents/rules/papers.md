<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Research paper rules

Read this before you write or change anything under `papers/`, or make a
change that produces or invalidates a finding a paper states.

`papers/` holds research papers, not plans. A paper is a proper paper in the
style of an arXiv submission (abstract, method, evaluation, results,
limitations, references) that builds to PDF. A final paper contains only
findings that are measured and verified. While a paper is a draft
(`status.txt` says `draft`) it may also hold registered measurements whose
results are not in yet, stated as such. Roadmaps, intentions and next steps
belong in `.agents/roadmap/`; a paper that reads like a ledger is wrong.

- **Route by scope.** Every paper states its scope in its introduction, and
  `papers/README.md` lists the scopes. A new finding goes into the paper whose
  scope it falls in.
- **Keep papers current.** A change that produces a new verified finding, or
  invalidates a stated one (a corrected defect, a re-measured result, a
  changed method), updates the affected paper in the same change.
- **Numbers come from records.** Every number in a paper's tables is generated
  from a committed result record under the paper's `data/` directory by the
  paper's own scripts, never typed by hand. A record names the commit, the
  command, the seeds and the device that produced it.
- **Verified means reproducible.** Every number traces to a measurement that
  can be rerun with the commands the paper documents, comes with its sample
  size and uncertainty, and is labelled a pilot when underpowered. A result
  that was not measured is absent, never zero; a result that did not hold is
  reported as such.
- **No speculation as findings.** Claims the evidence does not support stay
  out or are stated as limitations. Every citation is checked to exist
  against its primary source.
- **The paper must build.** `make papers/<name>/pdf` produces the PDF with no
  undefined references or citations before the change is committed. Output
  goes to `build/papers/` (`papers/paper.mk`); nothing is written into the
  paper's directory.
- **Scope rule applies.** A paper names only projects brain depends on
  (`make check/scope`).
