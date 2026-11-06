# AI use

This file records how our group used AI tools in this project, organized by the part of the work each use contributed to.

## Report

We wrote most of the report ourselves, including the opening of the abstract, the first paragraph of the introduction, the Related Work paragraphs on diffusion and flow matching, efficient generation on devices, feature caching and prediction, and speculative techniques, and the Data section.

### Abstract, introduction, and related work

Claude Code drafted a few additions to these sections, listed below, in `report/report.tex` and `report/reference.bib`. We decided what to add and how to frame it, and revised the drafts over several rounds of feedback. The hand-written text around these additions was not changed.

- **Abstract:** Claude Code wrote the sentences after our opening, on the preliminary experiments, the evaluation, one finding, and next steps.
- **Introduction:** Claude Code added the second paragraph describing the existing methods we experimented with (feature reuse, higher-order samplers, and guidance policies) and our plan to potentially develop our own method.
- **Related Work:** Claude Code added a paragraph on classifier-free guidance, limited-interval guidance, and adaptive guidance, with three new BibTeX entries.

## Repo

### Samplers and guidance

We chose which sampling methods to add to the codebase: DPM-Solver++ 2M and UniPC as higher-order samplers, and limited-interval guidance and adaptive guidance as guidance policies. We also ran the sampler and guidance experiments on our benchmark prompts. Claude Code wrote the implementation of these methods in the following files:

- `feature_reuse/samplers.py`: DPM-Solver++ 2M and UniPC samplers for flow matching
- `feature_reuse/guidance.py`: limited-interval and adaptive guidance policies
- `scripts/diagnose-guidance.py` and `scripts/summarize-guidance.py`: guidance diagnostics and threshold selection
- `tests/test_samplers.py` and `tests/test_guidance.py`
- The "Samplers and guidance" section of `README.md`

The unit tests check the samplers' accuracy against Euler and their behavior at the first and last steps. The samplers were also compared with the diffusers implementations outside the test suite, and `scripts/check-klein-equivalence.py` confirms on the real model that the default settings reproduce plain CFG and MFLUX's Euler sampler.
