# Test fixture: upstream Duck harness modules (unmodified)

`src/ARC3-Inference/inference/agent/`, `src/ARC3-Inference/inference/utils/` and
`src/ARC3-Inference/inference/framework/solver.py` (text only: the tests patch it but do not import it) are verbatim copies of the files in the
Kaggle dataset `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (the animation-aware TAAF / Duck source bundle,
published CC0), which packages Tufa Labs' ARC3-Inference code (MIT License, per its pyproject classifiers; credit to
the Tufa Labs team). They are here only so `tests/test_taaf_ours_patch.py` can check that every patch in
`scripts/taaf_ours_patch.py` still applies exactly once and that the patched harness imports and behaves, without
network access or the dataset. Do not edit them: they must match the bundle the notebooks patch at run time.
